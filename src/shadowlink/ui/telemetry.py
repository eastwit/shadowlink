"""Frame rate and latency statistics for the HUD and benchmark logs."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from shadowlink.types import PerceptionResult


class RateMeter:
    """Exponential moving average of frame rate."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = float(alpha)
        self._last: float | None = None
        self.fps: float | None = None

    def tick(self, now: float) -> float | None:
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                instant = 1.0 / dt
                self.fps = (
                    instant
                    if self.fps is None
                    else (self.alpha * instant + (1.0 - self.alpha) * self.fps)
                )
        self._last = now
        return self.fps


class MetricWindow:
    """Fixed-length window with mean and percentiles."""

    def __init__(self, size: int = 120) -> None:
        if size < 1:
            raise ValueError("window size must be >= 1")
        self._values: deque[float] = deque(maxlen=int(size))

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self._values.append(float(value))

    @property
    def latest(self) -> float | None:
        return self._values[-1] if self._values else None

    @property
    def mean(self) -> float | None:
        return float(np.mean(self._values)) if self._values else None

    def percentile(self, q: float) -> float | None:
        return float(np.percentile(self._values, q)) if self._values else None

    def __len__(self) -> int:
        return len(self._values)


class Telemetry:
    def __init__(self, window: int = 120) -> None:
        self.fps = RateMeter()
        self.detect_ms = MetricWindow(window)
        self.infer_ms = MetricWindow(window)
        self.age_ms = MetricWindow(window)
        self.process_ms = MetricWindow(window)
        self._last_result_seq = -1

    def tick(self, now: float) -> None:
        self.fps.tick(now)

    def update(self, result: PerceptionResult | None, now: float) -> None:
        if result is None or result.frame_seq == self._last_result_seq:
            return
        self._last_result_seq = result.frame_seq
        self.detect_ms.add(result.detect_ms)
        self.infer_ms.add(result.infer_ms)
        self.age_ms.add(max(0.0, (now - result.capture_ts) * 1000.0))

    def add_process_time(self, seconds: float) -> None:
        self.process_ms.add(seconds * 1000.0)

    def summary(self) -> dict[str, float | None]:
        return {
            "fps": self.fps.fps,
            "detect_ms_p50": self.detect_ms.percentile(50),
            "detect_ms_p95": self.detect_ms.percentile(95),
            "infer_ms_p50": self.infer_ms.percentile(50),
            "infer_ms_p95": self.infer_ms.percentile(95),
            "age_ms_p50": self.age_ms.percentile(50),
            "process_ms_p50": self.process_ms.percentile(50),
        }

    def hud_lines(self) -> list[str]:
        fps = self.fps.fps
        lines = [f"fps {fps:5.1f}" if fps else "fps   --"]
        detect = self.detect_ms
        infer = self.infer_ms
        if len(detect):
            lines.append(
                f"detect p50/p95 {detect.percentile(50):5.1f}/{detect.percentile(95):5.1f} ms"
            )
        if len(infer):
            lines.append(
                f"infer  p50/p95 {infer.percentile(50):5.1f}/{infer.percentile(95):5.1f} ms"
            )
        if len(self.age_ms):
            lines.append(f"age    p50     {self.age_ms.percentile(50):5.1f} ms")
        return lines
