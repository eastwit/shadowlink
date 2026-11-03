"""Asynchronous perception with latest-only semantics.

Detection and (for HaMeR) inference can run in separate threads so the two
stages overlap.  Each stage has a single-slot mailbox: when a newer frame
arrives, the unprocessed older frame is dropped instead of queued.  The main
loop never waits for perception; it polls the most recent result and ignores
stale ones.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Generic, TypeVar

from shadowlink.perception.base import WorkerHealth
from shadowlink.perception.selector import DetectionSelector, filter_by_side
from shadowlink.types import (
    FramePacket,
    HandDetection,
    HandObservation,
    PerceptionResult,
)

log = logging.getLogger(__name__)

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """A one-item mailbox that always keeps the newest value."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._item: T | None = None

    def put(self, item: T) -> bool:
        """Store ``item``; return True if an unconsumed item was replaced."""
        with self._lock:
            replaced = self._item is not None
            self._item = item
            self._event.set()
            return replaced

    def take(self, timeout: float | None = None) -> T | None:
        if not self._event.wait(timeout):
            return None
        with self._lock:
            item = self._item
            self._item = None
            self._event.clear()
            return item

    def clear(self) -> None:
        with self._lock:
            self._item = None
            self._event.clear()


class LatestValue(Generic[T]):
    """A lock-protected box whose value can be read repeatedly."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: T | None = None

    def put(self, value: T) -> None:
        with self._lock:
            self._value = value

    def get(self) -> T | None:
        with self._lock:
            return self._value

    def clear(self) -> None:
        with self._lock:
            self._value = None


class PerceptionService:
    def __init__(
        self,
        detector,
        backend,
        selector: DetectionSelector,
        *,
        side: str = "any",
        pipeline: str = "two_stage",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if pipeline not in {"single", "two_stage"}:
            raise ValueError(f"unknown perception pipeline: {pipeline!r}")
        if side not in {"any", "left", "right"}:
            raise ValueError(f"unknown side filter: {side!r}")
        self.detector = detector
        self.backend = backend
        self.selector = selector
        self.side = side
        self.pipeline = pipeline
        self._clock = clock
        self._input: LatestSlot[FramePacket] = LatestSlot()
        self._detections: LatestSlot[tuple] = LatestSlot()
        self._result: LatestValue[PerceptionResult] = LatestValue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._health = WorkerHealth()
        self._dropped = 0
        self._threads: list[threading.Thread] = []
        self._started = False

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._started:
            return
        self._stop.clear()
        with self._lock:
            self._dropped = 0
            self._health = WorkerHealth(state="running")
        self._threads = [self._spawn(self._detect_loop, "shadowlink-detect")]
        if self.pipeline == "two_stage":
            self._threads.append(self._spawn(self._estimate_loop, "shadowlink-infer"))
        self._started = True

    def _spawn(self, target, name: str) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        self._stop.set()
        self._input.clear()
        self._detections.clear()
        for thread in self._threads:
            thread.join(timeout=2.0)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            log.warning("perception threads did not stop: %s", ", ".join(alive))
        self._threads.clear()
        self._started = False
        for component in (self.detector, self.backend):
            closer = getattr(component, "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:  # pragma: no cover - best-effort cleanup
                    log.exception("error closing perception component")
        with self._lock:
            if self._health.state != "error":
                self._health = replace(self._health, state="stopped")

    @property
    def health(self) -> WorkerHealth:
        with self._lock:
            return replace(self._health)

    # ---------------------------------------------------------------- inputs
    def submit(self, frame: FramePacket) -> bool:
        if not self._started or self._stop.is_set():
            return False
        if self._input.put(frame):
            self._count_dropped()
        return True

    def latest_result(self) -> PerceptionResult | None:
        return self._result.get()

    # ------------------------------------------------------------- internals
    def _count_dropped(self) -> None:
        with self._lock:
            self._dropped += 1
            self._health = replace(self._health, dropped=self._dropped)

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._input.take(timeout=0.1)
            if frame is None:
                continue
            try:
                detect_start = self._clock()
                detections = self.detector.detect(frame)
                detect_end = self._clock()
                eligible = filter_by_side(detections, self.side)
                selected = self.selector.select(eligible, frame.width, frame.height)
                chosen: list[HandDetection] = [] if selected is None else [selected]
                detect_ms = (detect_end - detect_start) * 1000.0

                if not chosen:
                    self._publish(frame, [], detect_ms, 0.0)
                    continue
                if self.pipeline == "single":
                    estimate_start = self._clock()
                    observations = self.backend.estimate(frame, chosen)
                    estimate_end = self._clock()
                    self._publish(
                        frame,
                        observations,
                        detect_ms,
                        (estimate_end - estimate_start) * 1000.0,
                    )
                else:
                    if self._detections.put((frame, chosen, detect_ms)):
                        self._count_dropped()
            except Exception as exc:  # noqa: BLE001 - surfaced through health
                self._fail("detector", exc)

    def _estimate_loop(self) -> None:
        while not self._stop.is_set():
            item = self._detections.take(timeout=0.1)
            if item is None:
                continue
            frame, chosen, detect_ms = item
            try:
                start = self._clock()
                observations = self.backend.estimate(frame, chosen)
                infer_ms = (self._clock() - start) * 1000.0
                self._publish(frame, observations, detect_ms, infer_ms)
            except Exception as exc:  # noqa: BLE001 - surfaced through health
                self._fail("backend", exc)

    def _publish(
        self,
        frame: FramePacket,
        observations: list[HandObservation],
        detect_ms: float,
        infer_ms: float,
    ) -> None:
        with self._lock:
            dropped = self._dropped
        result = PerceptionResult(
            frame_seq=frame.seq,
            capture_ts=frame.capture_ts,
            finished_ts=self._clock(),
            observations=tuple(observations),
            detect_ms=detect_ms,
            infer_ms=infer_ms,
            dropped=dropped,
        )
        self._result.put(result)
        with self._lock:
            self._health = replace(self._health, frames=self._health.frames + 1)

    def _fail(self, stage: str, exc: Exception) -> None:
        log.exception("perception %s failed", stage)
        with self._lock:
            self._health = replace(
                self._health,
                state="error",
                message=f"{stage}: {exc}",
                errors=self._health.errors + 1,
            )
        self._stop.set()
