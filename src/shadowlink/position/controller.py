"""Turn a camera-space palm measurement into a smooth world translation.

The mapping is relative: the first usable sample (or a ``zero`` request) becomes
the camera-space reference, and later motion is applied 1:1 (configurable) to
the kinematic hand base.  Losing the measurement freezes the last translation
instead of returning to zero.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from shadowlink.config import PositionConfig
from shadowlink.position.depth_sampling import PalmMeasurement
from shadowlink.retarget.smoothing import OneEuroFilter, SlewLimiter

MAX_DT_S = 0.1

# Camera axes are x right, y down, z away from the sensor.
PRESET_MATRICES: dict[str, np.ndarray] = {
    # Screen-aligned: image right -> world +y, image up -> world +z, away -> -x.
    "front": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    # Palm toward the camera, fingers up: anatomical frame alignment.  Motion
    # toward the camera maps to world +z, the simulated palm normal at rest.
    "palm": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    # Back of the hand toward the camera, fingers up.
    "back": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


@dataclass(frozen=True, slots=True)
class PositionSettings:
    rotation: np.ndarray
    sign: np.ndarray
    gain: float = 1.0
    deadzone_m: float = 0.01
    max_offset_m: float = 0.25
    temporal_window: int = 3
    stale_timeout_s: float = 0.5
    auto_zero: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.1
    d_cutoff: float = 1.0
    max_rate_m_s: float = 0.8

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        sign = np.asarray(self.sign, dtype=np.float64).reshape(3)
        rotation = rotation.copy()
        sign = sign.copy()
        rotation.setflags(write=False)
        sign.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "sign", sign)
        if not math.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be finite and > 0")
        if not math.isfinite(self.max_offset_m) or self.max_offset_m <= 0:
            raise ValueError("max_offset_m must be finite and > 0")
        if not math.isfinite(self.deadzone_m) or not 0 <= self.deadzone_m < self.max_offset_m:
            raise ValueError("deadzone_m must be in [0, max_offset_m)")
        if self.temporal_window < 1:
            raise ValueError("temporal_window must be >= 1")
        if not math.isfinite(self.stale_timeout_s) or self.stale_timeout_s <= 0:
            raise ValueError("stale_timeout_s must be finite and > 0")
        if self.min_cutoff <= 0 or self.d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be > 0")
        if self.beta < 0:
            raise ValueError("beta must be >= 0")
        if not math.isfinite(self.max_rate_m_s) or self.max_rate_m_s <= 0:
            raise ValueError("max_rate_m_s must be finite and > 0")

    @classmethod
    def from_config(cls, cfg: PositionConfig) -> PositionSettings:
        try:
            rotation = PRESET_MATRICES[cfg.preset]
        except KeyError as exc:
            raise ValueError(f"unknown position preset: {cfg.preset!r}") from exc
        sign = np.array(
            [
                -1.0 if cfg.invert_x else 1.0,
                -1.0 if cfg.invert_y else 1.0,
                -1.0 if cfg.invert_z else 1.0,
            ],
            dtype=np.float64,
        )
        return cls(
            rotation=rotation,
            sign=sign,
            gain=cfg.gain,
            deadzone_m=cfg.deadzone_m,
            max_offset_m=cfg.max_offset_m,
            temporal_window=cfg.temporal_window,
            stale_timeout_s=cfg.stale_timeout_s,
            auto_zero=cfg.auto_zero,
            min_cutoff=cfg.min_cutoff,
            beta=cfg.beta,
            d_cutoff=cfg.d_cutoff,
            max_rate_m_s=cfg.max_rate_m_s,
        )


class PositionController:
    """Map relative camera motion to a bounded world translation."""

    def __init__(self, settings: PositionSettings) -> None:
        self.settings = settings
        self._filter = OneEuroFilter(
            settings.min_cutoff,
            settings.beta,
            settings.d_cutoff,
        )
        self._rate = SlewLimiter(settings.max_rate_m_s)
        self._window: deque[tuple[int, np.ndarray]] = deque(maxlen=settings.temporal_window)
        self._reference: np.ndarray | None = None
        self._anchor = np.zeros(3, dtype=np.float64)
        self._output = np.zeros(3, dtype=np.float64)
        self._last_output = np.zeros(3, dtype=np.float64)
        self._last_now: float | None = None
        self._last_seen: float | None = None
        self._state = "waiting"
        self._zero_pending = False
        self._latest: PalmMeasurement | None = None

    # ------------------------------------------------------------- properties
    @property
    def output(self) -> np.ndarray:
        return self._output.copy()

    @property
    def state(self) -> str:
        return self._state

    @property
    def camera_position(self) -> np.ndarray | None:
        return None if self._latest is None else self._latest.xyz_m.copy()

    # ---------------------------------------------------------------- updates
    def update(
        self,
        measurement: PalmMeasurement | None,
        now: float,
        rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance the mapping with one palm measurement.

        ``rotation`` overrides the static camera preset for this sample.  The
        app passes a hand-relative camera-to-world matrix built from the palm
        basis so anatomical directions follow the observed hand pose.
        """
        if measurement is None:
            if self._last_seen is not None:
                self._state = "frozen"
            return self.output
        if now - measurement.capture_ts > self.settings.stale_timeout_s:
            self._state = "frozen"
            return self.output

        matrix = (
            self.settings.rotation if rotation is None else np.asarray(rotation, dtype=np.float64)
        )
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("rotation must be a finite 3x3 matrix")

        self._latest = measurement
        self._last_seen = now
        self._append(measurement)

        if self._reference is None and not self.settings.auto_zero and not self._zero_pending:
            self._state = "waiting"
            return self.output

        if self._reference is None or self._zero_pending:
            self._rebase(self._median(), now)
            self._zero_pending = False
            self._state = "tracking"
            return self.output

        dt = 0.0 if self._last_now is None else min(max(now - self._last_now, 0.0), MAX_DT_S)
        self._last_now = now

        delta = self._median() - self._reference
        distance = float(np.linalg.norm(delta))
        if distance < self.settings.deadzone_m:
            delta = np.zeros(3, dtype=np.float64)
        elif distance > 0.0:
            delta *= (distance - self.settings.deadzone_m) / distance

        target = self._anchor + self.settings.gain * self.settings.sign * (matrix @ delta)
        target = np.clip(target, -self.settings.max_offset_m, self.settings.max_offset_m)
        smoothed = self._filter.filter(target, dt)
        limited = self._rate.limit(self._last_output, smoothed, dt)
        self._output = np.clip(
            limited,
            -self.settings.max_offset_m,
            self.settings.max_offset_m,
        )
        self._last_output = self._output.copy()
        self._state = "tracking"
        return self.output

    def zero(self, now: float) -> bool:
        """Make the current palm position the camera-space zero.

        Returns ``False`` when there is no fresh measurement; in that case the
        next valid update performs the zero instead.
        """
        if self._latest is None or now - self._latest.capture_ts > self.settings.stale_timeout_s:
            self._zero_pending = True
            return False
        self._append(self._latest)
        self._rebase(self._median(), now)
        self._zero_pending = False
        self._state = "tracking"
        return True

    def reset(self) -> None:
        self._window.clear()
        self._reference = None
        self._anchor = np.zeros(3, dtype=np.float64)
        self._output = np.zeros(3, dtype=np.float64)
        self._last_output = np.zeros(3, dtype=np.float64)
        self._filter.reset(None)
        self._last_now = None
        self._last_seen = None
        self._state = "waiting"
        self._zero_pending = False
        self._latest = None

    # --------------------------------------------------------------- internals
    def _append(self, measurement: PalmMeasurement) -> None:
        if self._window and self._window[-1][0] == measurement.frame_seq:
            return
        if self._window and measurement.frame_seq < self._window[-1][0]:
            self._window.clear()
        self._window.append((measurement.frame_seq, measurement.xyz_m.copy()))

    def _median(self) -> np.ndarray:
        values = np.stack([xyz for _, xyz in self._window], axis=0)
        return np.median(values, axis=0)

    def _rebase(self, reference: np.ndarray, now: float) -> None:
        self._reference = np.asarray(reference, dtype=np.float64).reshape(3).copy()
        self._anchor = self._output.copy()
        self._filter.reset(self._anchor)
        self._last_output = self._anchor.copy()
        self._last_now = now
