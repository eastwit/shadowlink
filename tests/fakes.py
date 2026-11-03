"""Test doubles shared by the application-level tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

import numpy as np

from shadowlink.perception.base import WorkerHealth
from shadowlink.types import (
    FramePacket,
    HandDetection,
    HandObservation,
    PerceptionResult,
    Side,
)


class StepClock:
    """A deterministic, thread-safe ``time.monotonic`` replacement."""

    def __init__(self, step_s: float = 0.005, start: float = 1000.0) -> None:
        self._step = float(step_s)
        self._now = float(start)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            self._now += self._step
            return self._now


class FakeSource:
    """A finite sequence of blank frames."""

    def __init__(
        self,
        frames: int = 60,
        *,
        width: int = 64,
        height: int = 48,
        fps: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total = int(frames)
        self.width_px = int(width)
        self.height_px = int(height)
        self._fps = float(fps)
        self._clock = clock
        self._seq = 0
        self.closed = False

    def read(self) -> FramePacket | None:
        if self.closed or self._seq >= self.total:
            return None
        image = np.zeros((self.height_px, self.width_px, 3), dtype=np.uint8)
        packet = FramePacket(
            seq=self._seq,
            capture_ts=self._clock(),
            image_bgr=image,
            source_fps=self._fps,
        )
        self._seq += 1
        return packet

    @property
    def produced(self) -> int:
        return self._seq

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def size(self) -> tuple[int, int] | None:
        return (self.width_px, self.height_px)

    def close(self) -> None:
        self.closed = True


class FakePerception:
    """Synchronous perception with the same interface as ``PerceptionService``."""

    def __init__(
        self,
        observations: Sequence[HandObservation] | None = None,
        *,
        keypoints: np.ndarray | None = None,
        side: Side = Side.RIGHT,
        backend: str = "fake",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._observations = tuple(observations or ())
        self._keypoints = None if keypoints is None else np.asarray(keypoints, dtype=np.float64)
        self._side = side
        self._backend = backend
        self._clock = clock
        self._result: PerceptionResult | None = None
        self.health = WorkerHealth()
        self.started = False
        self.closed = False
        self.submitted: list[int] = []

    def start(self) -> None:
        self.started = True
        self.health = WorkerHealth(state="running")

    def submit(self, frame: FramePacket) -> bool:
        if self.closed:
            return False
        self.submitted.append(frame.seq)
        now = self._clock()
        if self._keypoints is None:
            observations = self._observations
        else:
            observations = (
                HandObservation(
                    keypoints_3d=self._keypoints,
                    apparent_side=self._side,
                    anatomical_side=self._side,
                    score=1.0,
                    frame_seq=frame.seq,
                    capture_ts=frame.capture_ts,
                    inference_ts=now,
                    backend=self._backend,
                ),
            )
        self._result = PerceptionResult(
            frame_seq=frame.seq,
            capture_ts=frame.capture_ts,
            finished_ts=now,
            observations=observations,
        )
        return True

    def latest_result(self) -> PerceptionResult | None:
        return self._result

    def close(self) -> None:
        self.closed = True
        self.health = WorkerHealth(state="stopped")


class FakeDetector:
    """Always reports one right hand."""

    name = "fake"

    def __init__(self, *, bbox=(10.0, 10.0, 50.0, 40.0), side: Side = Side.RIGHT) -> None:
        self._bbox = np.asarray(bbox, dtype=np.float64)
        self._side = side
        self.closed = False
        self.calls = 0

    def detect(self, frame: FramePacket) -> list[HandDetection]:
        self.calls += 1
        return [
            HandDetection(
                bbox_xyxy=self._bbox,
                apparent_side=self._side,
                anatomical_side=self._side,
                frame_seq=frame.seq,
            )
        ]

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    """Turns every detection into an observation with fixed keypoints."""

    name = "fake"
    requires_detections = True
    supports_mesh = False

    def __init__(self, keypoints: np.ndarray, *, delay_s: float = 0.0) -> None:
        self.keypoints = np.asarray(keypoints, dtype=np.float64)
        self.delay_s = float(delay_s)
        self.closed = False
        self.calls = 0

    def estimate(self, frame: FramePacket, detections) -> list[HandObservation]:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        now = time.monotonic()
        return [
            HandObservation(
                keypoints_3d=self.keypoints,
                apparent_side=detection.apparent_side,
                anatomical_side=detection.anatomical_side,
                score=detection.score,
                frame_seq=frame.seq,
                capture_ts=frame.capture_ts,
                inference_ts=now,
                backend=self.name,
                bbox_xyxy=detection.bbox_xyxy,
            )
            for detection in detections
        ]

    def close(self) -> None:
        self.closed = True
