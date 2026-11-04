"""Attitude integration for phyphox experiments without a quaternion buffer.

The primary phone path is phyphox's ``TYPE_ROTATION_VECTOR`` quaternion.  When
an experiment exposes only raw gyroscope and accelerometer channels, the same
Mahony filter that the Joy-Con prototype validated is used here.  It is a
fallback: the fused phone quaternion is more accurate and does not depend on
host-side timing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from shadowlink.math3d import q_from_two_vectors, q_mul, q_norm, q_rotate


class Attitude:
    """Mahony complementary filter (proportional gravity correction only)."""

    INIT_SAMPLES = 20

    def __init__(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.kp = 2.0
        self.initialized = False
        self._init_acc: list[np.ndarray] = []

    def reset_from_accel(self, accel) -> None:
        vector = np.asarray(accel, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return
        self.q = q_from_two_vectors(vector / norm, np.array([0.0, 0.0, 1.0]))
        self.initialized = True

    def update(self, accel, gyro, dt: float) -> None:
        acc = np.asarray(accel, dtype=np.float64).reshape(3)
        rate = np.asarray(gyro, dtype=np.float64).reshape(3)
        if not self.initialized:
            self._init_acc.append(acc.copy())
            if len(self._init_acc) >= self.INIT_SAMPLES:
                self.reset_from_accel(np.mean(np.stack(self._init_acc), axis=0))
            return
        if not np.isfinite(acc).all() or not np.isfinite(rate).all():
            return
        if not math.isfinite(dt) or dt <= 0:
            return

        norm = float(np.linalg.norm(acc))
        if norm > 1e-6:
            measured = acc / norm
            estimated = q_rotate(
                np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]]),
                np.array([0.0, 0.0, 1.0]),
            )
            error = np.cross(measured, estimated)
            rate = rate + self.kp * error

        half = 0.5 * float(dt)
        dq = np.array([1.0, rate[0] * half, rate[1] * half, rate[2] * half])
        self.q = q_norm(q_mul(self.q, dq))


class ImuFusion:
    """Fuse phyphox gyroscope/accelerometer (and optional magnetometer) buffers."""

    KP_MAG = 0.05

    def __init__(
        self,
        gyro: Mapping[str, str],
        accel: Mapping[str, str],
        mag: Mapping[str, str] | None = None,
    ) -> None:
        self.gyro = dict(gyro)
        self.accel = dict(accel)
        self.mag = dict(mag) if mag else None
        self.attitude = Attitude()
        self._last: float | None = None

    @property
    def names(self) -> list[str]:
        names = [self.gyro[axis] for axis in "xyz"] + [self.accel[axis] for axis in "xyz"]
        if self.mag:
            names += [self.mag[axis] for axis in "xyz"]
        return names

    def update(self, values: Mapping[str, float], now: float) -> np.ndarray | None:
        if self._last is None:
            self._last = float(now)
            return None
        dt = min(0.2, max(0.001, float(now) - self._last))
        self._last = float(now)
        try:
            gyro = [float(values[self.gyro[axis]]) for axis in "xyz"]
            accel = [float(values[self.accel[axis]]) for axis in "xyz"]
        except (KeyError, TypeError, ValueError):
            return None
        self.attitude.update(accel, gyro, dt)
        if self.mag:
            try:
                magnetic = [float(values[self.mag[axis]]) for axis in "xyz"]
            except (KeyError, TypeError, ValueError):
                magnetic = None
            if magnetic is not None and all(math.isfinite(value) for value in magnetic):
                self._level(magnetic)
        return self.attitude.q.copy() if self.attitude.initialized else None

    def _level(self, magnetic_body) -> None:
        """Pull the horizontal heading toward magnetic north (+Y)."""
        magnetic_world = q_rotate(self.attitude.q, magnetic_body)
        angle = math.atan2(magnetic_world[0], magnetic_world[1])
        delta = 0.5 * angle * self.KP_MAG
        dq = (math.cos(delta), 0.0, 0.0, math.sin(delta))
        self.attitude.q = q_norm(q_mul(dq, self.attitude.q))
