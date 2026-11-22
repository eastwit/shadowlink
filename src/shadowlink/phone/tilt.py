"""Gravity-referenced tilt mapping from a phyphox phone to the hand base.

The tag supplies the stable yaw axis; this module supplies only the two axes
that are noisy in the tag pose.  Instead of using the phone's full attitude we
align its measured gravity direction with the direction captured at zero.  The
result is a shortest-arc rotation with no yaw component, so the tag remains the
only source of heading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shadowlink.math3d import (
    q_angle_between,
    q_conj,
    q_from_axis_angle,
    q_from_two_vectors,
    q_mul,
    q_norm,
    q_rotate,
    q_swing_twist,
    q_to_axis_angle,
)
from shadowlink.phone.service import PhoneSample

# Phone-user frame: +x out of the charging port, +y across the phone, +z out of
# the screen.  The service rotates Android's native frame into this one.
BACK_MOUNT = np.array([0.0, 0.7071068, 0.0, -0.7071068], dtype=np.float64)
IN_HAND_MOUNT = np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float64)
# The phone's long axis (+X) aligns with the hand's middle-finger axis (roll),
# its width (+Y) aligns with the hand's left-right axis (pitch), and its screen
# normal (+Z) aligns with the palm normal (yaw). The tag already supplies yaw,
# so the phone screen-normal twist is discarded. This mount is swap-d: both
# phone tilt axes point in the hand-positive direction.
YAW_PITCH_MOUNT = np.array([0.0, -0.7071068, 0.0, 0.7071068], dtype=np.float64)

_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
# The service rotates Android into a user-facing frame whose +Z is the screen
# normal. The phone roll channel is rotation about that +Z axis.
_ROLL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# All proper rotations that map the phone screen normal onto +/- hand yaw while
# keeping phone X/Y on hand Z/Y. Cycling these live covers swapped axes and
# every independent sign combination without a restart.
MOUNT_OPTIONS: tuple[tuple[str, np.ndarray], ...] = (
    ("swap-a", np.array([0.7071068, 0.0, -0.7071068, 0.0])),  # X+Z  Y+Y  Z-X
    ("swap-b", np.array([0.0, 0.7071068, 0.0, 0.7071068])),  # X+Z  Y-Y  Z+X
    ("swap-c", np.array([0.7071068, 0.0, 0.7071068, 0.0])),  # X-Z  Y+Y  Z+X
    ("swap-d", np.array([0.0, -0.7071068, 0.0, 0.7071068])),  # X-Z  Y-Y  Z-X
    ("cyclic-a", np.array([0.5, 0.5, 0.5, 0.5])),  # X+Y  Y+Z  Z+X
    ("cyclic-b", np.array([0.5, -0.5, -0.5, 0.5])),  # X+Y  Y-Z  Z-X
    ("cyclic-c", np.array([0.5, 0.5, -0.5, -0.5])),  # X-Y  Y+Z  Z-X
    ("cyclic-d", np.array([0.5, -0.5, 0.5, -0.5])),  # X-Y  Y-Z  Z+X
)


def mount_for_config(config) -> np.ndarray:
    """Return the normalized phone-body to hand-base mount quaternion."""
    preset = getattr(config, "mount_preset", "custom")
    if preset == "yaw_pitch":
        return q_norm(YAW_PITCH_MOUNT)
    if preset == "back":
        return q_norm(BACK_MOUNT)
    if preset == "in_hand":
        return q_norm(IN_HAND_MOUNT)
    return q_norm(
        [
            getattr(config, "mount_w", 0.0),
            getattr(config, "mount_x", 0.0),
            getattr(config, "mount_y", 0.0),
            getattr(config, "mount_z", 1.0),
        ]
    )


def up_in_body(quat) -> np.ndarray:
    """World-up direction expressed in the phone body frame."""
    return q_rotate(q_conj(quat), _WORLD_UP)


@dataclass(frozen=True, slots=True)
class PhoneTiltSettings:
    mount: np.ndarray
    gain: float
    deadzone_rad: float
    max_angle_rad: float
    stale_timeout_s: float
    axes: str = "yaw_pitch"

    def __post_init__(self) -> None:
        mount = q_norm(self.mount)
        mount.setflags(write=False)
        object.__setattr__(self, "mount", mount)
        if self.axes not in ("yaw_pitch", "pitch_roll"):
            raise ValueError("phone axes must be yaw_pitch or pitch_roll")
        if not math.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("phone gain must be finite and > 0")
        if not math.isfinite(self.deadzone_rad) or self.deadzone_rad < 0:
            raise ValueError("phone deadzone must be finite and >= 0")
        if not math.isfinite(self.max_angle_rad) or self.max_angle_rad <= 0:
            raise ValueError("phone max angle must be finite and > 0")
        if self.deadzone_rad >= self.max_angle_rad:
            raise ValueError("phone deadzone must be smaller than max angle")
        if not math.isfinite(self.stale_timeout_s) or self.stale_timeout_s <= 0:
            raise ValueError("phone stale timeout must be finite and > 0")

    @classmethod
    def from_config(cls, config) -> PhoneTiltSettings:
        return cls(
            mount=mount_for_config(config),
            gain=float(config.gain),
            deadzone_rad=math.radians(float(config.deadzone_deg)),
            max_angle_rad=math.radians(float(config.max_angle_deg)),
            stale_timeout_s=float(config.stale_timeout_s),
            axes=str(getattr(config, "axes", "yaw_pitch")),
        )


class PhoneTiltController:
    """Turn phone attitude samples into a smoothable hand-frame tilt rotation."""

    def __init__(self, settings: PhoneTiltSettings) -> None:
        self.settings = settings
        self._mount = settings.mount.copy()
        self._mount_index = self._index_of(settings.mount)
        self._zero_quat: np.ndarray | None = None
        self._zero_up: np.ndarray | None = None
        self._up: np.ndarray | None = None
        self._state = "waiting"

    @staticmethod
    def _index_of(mount) -> int:
        for index, (_, candidate) in enumerate(MOUNT_OPTIONS):
            if q_angle_between(mount, candidate) < 1e-6:
                return index
        return -1

    @property
    def mount(self) -> np.ndarray:
        return self._mount.copy()

    @property
    def mount_name(self) -> str:
        if 0 <= self._mount_index < len(MOUNT_OPTIONS):
            return MOUNT_OPTIONS[self._mount_index][0]
        return "custom"

    def cycle_mount(self) -> str:
        self._mount_index = (self._mount_index + 1) % len(MOUNT_OPTIONS)
        self._mount = MOUNT_OPTIONS[self._mount_index][1].copy()
        return self.mount_name

    @property
    def initialized(self) -> bool:
        return self._zero_up is not None

    @property
    def state(self) -> str:
        return self._state

    @property
    def up(self) -> np.ndarray | None:
        return None if self._up is None else self._up.copy()

    def zero(self, sample: PhoneSample | None) -> bool:
        if sample is None or not sample.ready:
            self._state = "waiting"
            return False
        up = up_in_body(sample.quat_wxyz)
        if not np.isfinite(up).all():
            self._state = "waiting"
            return False
        self._zero_quat = q_norm(sample.quat_wxyz)
        self._zero_up = up
        self._up = up.copy()
        self._state = "tracking"
        return True

    def reset(self) -> None:
        self._zero_quat = None
        self._zero_up = None
        self._up = None
        self._state = "waiting"

    def update(
        self,
        sample: PhoneSample | None,
        now: float,
    ) -> np.ndarray | None:
        """Return the hand-frame tilt delta since zero, or ``None`` to freeze."""
        if sample is None or not sample.ready:
            self._state = "frozen"
            return None
        if now - sample.timestamp > self.settings.stale_timeout_s:
            self._state = "frozen"
            return None
        if self._zero_up is None or self._zero_quat is None:
            self._state = "waiting"
            return None

        up_now = up_in_body(sample.quat_wxyz)
        if not np.isfinite(up_now).all():
            self._state = "frozen"
            return None
        self._up = up_now.copy()

        if self.settings.axes == "yaw_pitch":
            # Keep yaw and pitch, discard roll about the phone's long axis.
            relative = q_mul(q_conj(self._zero_quat), sample.quat_wxyz)
            relative, _ = q_swing_twist(relative, _ROLL_AXIS)
        else:
            # q_from_two_vectors(now, zero) is the inverse gravity alignment,
            # which equals conj(q_zero) * q_now up to the discarded yaw.
            relative = q_from_two_vectors(up_now, self._zero_up)
        if q_angle_between(relative, [1.0, 0.0, 0.0, 0.0]) < 1e-6:
            self._state = "tracking"
            return np.array([1.0, 0.0, 0.0, 0.0])

        axis, angle = q_to_axis_angle(relative)
        if angle < self.settings.deadzone_rad:
            angle = 0.0
        else:
            angle = min(
                angle * self.settings.gain,
                self.settings.max_angle_rad,
            )
        control = q_from_axis_angle(axis, angle)
        mount = self._mount
        delta = q_mul(q_mul(mount, control), q_conj(mount))
        self._state = "tracking"
        return q_norm(delta)
