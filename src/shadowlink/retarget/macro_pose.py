"""Relative macro orientation mapped from a controller onto the hand base.

The controller reports an absolute, gravity-aligned orientation whose yaw is
unobservable.  Only the rotation *since the follow toggle was pressed* is
meaningful, so the mapping is always ``q_zero^-1 ⊗ q_now``.  The resulting
rotation is expressed in the controller's body frame, conjugated into the
hand base frame by the fixed mount rotation, and finally applied on top of the
hand orientation captured at toggle time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from shadowlink.math3d import (
    q_angle_between,
    q_conj,
    q_from_axis_angle,
    q_mul,
    q_norm,
    q_slerp,
    q_to_axis_angle,
)

MAX_DT_S = 0.1

# Fixed phone-body -> hand-base rotations for the two common physical mounts.
# They differ by 180 degrees about the phone's long axis, so rotations about
# the width and screen-normal axes reverse between them.
BACK_MOUNT = np.array([0.0, 0.7071068, 0.0, -0.7071068])
IN_HAND_MOUNT = np.array([0.7071068, 0.0, 0.7071068, 0.0])


def mount_for_config(config) -> np.ndarray:
    """Return the normalized mount for a ``PhoneConfig``."""
    preset = getattr(config, "mount_preset", "custom")
    if preset == "back":
        return BACK_MOUNT.copy()
    if preset == "in_hand":
        return IN_HAND_MOUNT.copy()
    return q_norm([config.mount_w, config.mount_x, config.mount_y, config.mount_z])


class OrientationSample(Protocol):
    """The small part of ``ControllerSample`` this layer depends on."""

    @property
    def quat_wxyz(self) -> np.ndarray: ...

    @property
    def timestamp(self) -> float: ...

    @property
    def ready(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class MacroPoseSettings:
    mount: np.ndarray
    gain: float
    deadzone_rad: float
    max_angle_rad: float
    smoothing_tau_s: float
    max_rate_rad_s: float
    stale_timeout_s: float

    @classmethod
    def from_config(cls, config) -> MacroPoseSettings:
        mount = mount_for_config(config)
        return cls(
            mount=mount,
            gain=float(config.gain),
            deadzone_rad=math.radians(float(config.deadzone_deg)),
            max_angle_rad=math.radians(float(config.max_angle_deg)),
            smoothing_tau_s=float(config.smoothing_tau_s),
            max_rate_rad_s=float(config.max_rate_rad_s),
            stale_timeout_s=float(config.stale_timeout_s),
        )


class MacroPoseController:
    """Turn controller samples into smooth absolute hand-base orientations."""

    def __init__(self, settings: MacroPoseSettings, initial_quat) -> None:
        self.settings = settings
        self._initial = q_norm(initial_quat)
        self._out = self._initial.copy()
        self._zero: np.ndarray | None = None
        self._reference_hand: np.ndarray | None = None
        self._last_t: float | None = None

    @property
    def initialized(self) -> bool:
        return self._zero is not None

    @property
    def orientation(self) -> np.ndarray:
        return self._out.copy()

    @property
    def zero_hand(self) -> np.ndarray:
        """Hand orientation that defines the current zero.

        Before following starts this is the hand's initial orientation.  After
        ``set_zero`` it is the visible hand pose captured at that moment.
        """
        if self._reference_hand is not None:
            return self._reference_hand.copy()
        return self._initial.copy()

    def set_zero(self, sample: OrientationSample, current_hand_quat) -> None:
        """Record the controller zero and the hand orientation to build on."""
        self._zero = q_norm(sample.quat_wxyz)
        self._reference_hand = q_norm(current_hand_quat)
        self._out = self._reference_hand.copy()
        self._last_t = None

    def reset(self) -> None:
        self._zero = None
        self._reference_hand = None
        self._out = self._initial.copy()
        self._last_t = None

    def update(
        self,
        sample: OrientationSample | None,
        now: float,
        enabled: bool,
    ) -> np.ndarray | None:
        """Return a new hand quaternion, or ``None`` when the pose is frozen."""
        if not enabled or sample is None or not sample.ready:
            return None
        if self._zero is None or self._reference_hand is None:
            return None
        if now - float(sample.timestamp) > self.settings.stale_timeout_s:
            return None

        relative = q_mul(q_conj(self._zero), sample.quat_wxyz)
        axis, angle = q_to_axis_angle(relative)
        if angle < self.settings.deadzone_rad:
            angle = 0.0
        else:
            angle = min(angle * self.settings.gain, self.settings.max_angle_rad)
        control = q_from_axis_angle(axis, angle)
        mount = self.settings.mount
        hand_delta = q_mul(q_mul(mount, control), q_conj(mount))
        target = q_mul(self._reference_hand, hand_delta)

        dt = 0.0 if self._last_t is None else max(0.0, min(MAX_DT_S, now - self._last_t))
        self._last_t = now
        if dt <= 0.0:
            return self._out.copy()

        tau = self.settings.smoothing_tau_s
        alpha = 1.0 if tau <= 0.0 else 1.0 - math.exp(-dt / tau)
        error = q_angle_between(self._out, target)
        if error > 1e-9 and self.settings.max_rate_rad_s > 0.0:
            limit = self.settings.max_rate_rad_s * dt / error
            alpha = min(alpha, limit)
        self._out = q_slerp(self._out, target, alpha)
        return self._out.copy()
