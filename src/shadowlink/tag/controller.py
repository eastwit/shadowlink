"""Relative 6-DoF mapping from a hand-held AprilTag to the Shadow Hand base.

The tag is an independent controller: at the zero instant its pose is recorded
together with the hand's current base orientation.  Later tag motion is applied
relative to that zero, so a comfortable starting pose never snaps the hand to
the camera frame.

Translation uses a camera preset, radial deadzone, gain, bound, One-Euro filter
and slew limiter. Rotation uses the tag's relative camera-frame rotation,
converts it to world axes with the same preset, and slerps toward the target
with a configurable deadzone, gain, angle clamp and rate limit.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from shadowlink.config import PositionConfig, TagConfig
from shadowlink.math3d import (
    q_angle_between,
    q_conj,
    q_from_axis_angle,
    q_from_matrix,
    q_mul,
    q_norm,
    q_slerp,
    q_to_axis_angle,
)
from shadowlink.retarget.smoothing import OneEuroFilter, SlewLimiter
from shadowlink.tag.types import TagSample

MAX_DT_S = 0.1

# Tag-axis convention supplied for the hand-mounted tag:
#   tag +X (red)   -> hand +Y (pitch axis)
#   tag +Y (green) -> hand +Z (roll axis / forward)
#   tag +Z (blue)  -> hand +X (yaw axis / palm normal)
# This is a cyclic permutation, so it preserves handedness and leaves yaw on
# blue while moving green onto roll and red onto pitch.
TAG_TO_HAND = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
TAG_TO_HAND_QUAT = q_from_matrix(TAG_TO_HAND)

# Camera axes are x right, y down, z away from the sensor.
PRESET_MATRICES: dict[str, np.ndarray] = {
    # Screen-aligned: image right -> world +y, image up -> world +z, away -> -x.
    "front": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    # Palm toward the camera, fingers up: anatomical frame alignment.
    "palm": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    # Back of the hand toward the camera, fingers up.
    "back": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    # Camera above the operator looking down: tag right -> +world x (ball),
    # tag closer -> +world z (up).
    "overhead": np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
}


@dataclass(frozen=True, slots=True)
class TagTranslationSettings:
    rotation: np.ndarray
    sign: np.ndarray
    gain: float
    deadzone_m: float
    temporal_window: int
    stale_timeout_s: float
    auto_zero: bool
    min_cutoff: float
    beta: float
    d_cutoff: float
    max_rate_m_s: float
    fixed_reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("rotation must be a finite 3x3 matrix")
        rotation = rotation.copy()
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        sign = np.asarray(self.sign, dtype=np.float64).reshape(3)
        if not np.isfinite(sign).all():
            raise ValueError("sign must be finite")
        sign = sign.copy()
        sign.setflags(write=False)
        object.__setattr__(self, "sign", sign)
        if not math.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("gain must be finite and > 0")
        if not math.isfinite(self.deadzone_m) or self.deadzone_m < 0:
            raise ValueError("deadzone_m must be finite and >= 0")
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
        if self.fixed_reference is not None:
            reference = np.asarray(self.fixed_reference, dtype=np.float64).reshape(3)
            if not np.isfinite(reference).all():
                raise ValueError("fixed_reference must be finite")
            reference = reference.copy()
            reference.setflags(write=False)
            object.__setattr__(self, "fixed_reference", reference)

    @classmethod
    def from_config(
        cls,
        cfg: PositionConfig,
        fixed_reference: np.ndarray | None = None,
    ) -> TagTranslationSettings:
        try:
            rotation = PRESET_MATRICES[cfg.preset]
        except KeyError as exc:
            raise ValueError(f"unknown position preset: {cfg.preset!r}") from exc
        return cls(
            rotation=rotation,
            sign=np.array(
                [
                    -1.0 if cfg.invert_x else 1.0,
                    -1.0 if cfg.invert_y else 1.0,
                    -1.0 if cfg.invert_z else 1.0,
                ],
                dtype=np.float64,
            ),
            gain=cfg.gain,
            deadzone_m=cfg.deadzone_m,
            temporal_window=cfg.temporal_window,
            stale_timeout_s=cfg.stale_timeout_s,
            auto_zero=cfg.auto_zero,
            min_cutoff=cfg.min_cutoff,
            beta=cfg.beta,
            d_cutoff=cfg.d_cutoff,
            max_rate_m_s=cfg.max_rate_m_s,
            fixed_reference=fixed_reference,
        )


@dataclass(frozen=True, slots=True)
class TagControlSettings:
    position: TagTranslationSettings
    mount: np.ndarray
    rotation_gain: float
    rotation_deadzone_rad: float
    rotation_max_angle_rad: float
    rotation_smoothing_tau_s: float
    rotation_max_rate_rad_s: float
    rotation_sign: np.ndarray

    def __post_init__(self) -> None:
        mount = q_norm(self.mount)
        mount.setflags(write=False)
        object.__setattr__(self, "mount", mount)
        sign = np.asarray(self.rotation_sign, dtype=np.float64).reshape(3)
        if not np.isfinite(sign).all() or not np.all(np.abs(sign) == 1.0):
            raise ValueError("rotation_sign must contain +1/-1 values")
        sign = sign.copy()
        sign.setflags(write=False)
        object.__setattr__(self, "rotation_sign", sign)
        for name in (
            "rotation_gain",
            "rotation_max_angle_rad",
            "rotation_max_rate_rad_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not math.isfinite(self.rotation_deadzone_rad) or self.rotation_deadzone_rad < 0:
            raise ValueError("rotation_deadzone_rad must be finite and >= 0")
        if self.rotation_deadzone_rad >= self.rotation_max_angle_rad:
            raise ValueError("rotation deadzone must be smaller than max angle")
        if (
            not math.isfinite(self.rotation_smoothing_tau_s)
            or self.rotation_smoothing_tau_s < 0
        ):
            raise ValueError("rotation_smoothing_tau_s must be finite and >= 0")

    @classmethod
    def from_config(
        cls,
        tag: TagConfig,
        position: PositionConfig,
    ) -> TagControlSettings:
        fixed_reference = (
            np.array([0.0, 0.0, float(tag.zero_distance_m)], dtype=np.float64)
            if tag.zero_mode == "fixed"
            else None
        )
        pos = TagTranslationSettings.from_config(position, fixed_reference)
        sign = np.array(
            [
                -1.0 if tag.rotation_invert_x else 1.0,
                -1.0 if tag.rotation_invert_y else 1.0,
                -1.0 if tag.rotation_invert_z else 1.0,
            ],
            dtype=np.float64,
        )
        return cls(
            position=pos,
            mount=TAG_TO_HAND_QUAT.copy(),
            rotation_gain=float(tag.rotation_gain),
            rotation_deadzone_rad=math.radians(float(tag.rotation_deadzone_deg)),
            rotation_max_angle_rad=math.radians(float(tag.rotation_max_angle_deg)),
            rotation_smoothing_tau_s=float(tag.rotation_smoothing_tau_s),
            rotation_max_rate_rad_s=float(tag.rotation_max_rate_rad_s),
            rotation_sign=sign,
        )


class _TagTranslation:
    """Median-windowed, relative tag translation with bounded smoothing."""

    def __init__(self, settings: TagTranslationSettings) -> None:
        self.settings = settings
        self._filter = OneEuroFilter(
            settings.min_cutoff,
            settings.beta,
            settings.d_cutoff,
        )
        self._rate = SlewLimiter(settings.max_rate_m_s)
        self._window: deque[tuple[int, np.ndarray]] = deque(maxlen=settings.temporal_window)
        self._fixed_reference = settings.fixed_reference
        self._reference: np.ndarray | None = (
            None
            if settings.fixed_reference is None
            else settings.fixed_reference.copy()
        )
        self._anchor = np.zeros(3, dtype=np.float64)
        self._output = np.zeros(3, dtype=np.float64)
        self._last_output = np.zeros(3, dtype=np.float64)
        self._last_now: float | None = None
        self._latest: TagSample | None = None
        self._state = "waiting"
        self._zero_pending = False

    @property
    def output(self) -> np.ndarray:
        return self._output.copy()

    @property
    def state(self) -> str:
        return self._state

    @property
    def latest_translation(self) -> np.ndarray | None:
        return None if self._latest is None else self._latest.translation_m.copy()

    @property
    def fixed_translation_zero(self) -> bool:
        return self._fixed_reference is not None

    def update(self, sample: TagSample | None, now: float, enabled: bool) -> np.ndarray:
        if sample is None:
            if self._state != "waiting":
                self._state = "frozen"
            return self.output
        if now - sample.timestamp > self.settings.stale_timeout_s:
            self._state = "frozen"
            return self.output

        self._latest = sample
        self._append(sample)

        if self._reference is None and not self._zero_pending:
            if not self.settings.auto_zero:
                self._state = "waiting"
                return self.output
            self._rebase(self._median(), now)
            self._state = "tracking"
            return self.output

        if self._reference is None or self._zero_pending:
            self._rebase(self._median(), now)
            self._zero_pending = False
            self._state = "tracking"
            return self.output

        if not enabled:
            self._state = "paused"
            return self.output

        dt = 0.0 if self._last_now is None else min(max(now - self._last_now, 0.0), MAX_DT_S)
        self._last_now = now

        delta = self._median() - self._reference
        distance = float(np.linalg.norm(delta))
        if distance < self.settings.deadzone_m:
            delta = np.zeros(3, dtype=np.float64)
        elif distance > 0.0:
            delta *= (distance - self.settings.deadzone_m) / distance

        mapped = self.settings.rotation @ delta
        target = self._anchor + self.settings.gain * self.settings.sign * mapped
        smoothed = self._filter.filter(target, dt)
        self._output = self._rate.limit(self._last_output, smoothed, dt)
        self._last_output = self._output.copy()
        self._state = "tracking"
        return self.output

    def zero(self, now: float) -> bool:
        if self._fixed_reference is not None:
            self._reference = self._fixed_reference.copy()
            self._anchor = np.zeros(3, dtype=np.float64)
            self._output = np.zeros(3, dtype=np.float64)
            self._last_output = np.zeros(3, dtype=np.float64)
            self._filter.reset(self._output)
            self._last_now = now
            self._zero_pending = False
            self._state = "tracking"
            return True
        if (
            self._latest is None
            or now - self._latest.timestamp > self.settings.stale_timeout_s
        ):
            self._zero_pending = True
            return False
        self._append(self._latest)
        self._rebase(self._median(), now)
        self._zero_pending = False
        self._state = "tracking"
        return True

    def reset(self) -> None:
        self._window.clear()
        self._reference = (
            None
            if self._fixed_reference is None
            else self._fixed_reference.copy()
        )
        self._anchor = np.zeros(3, dtype=np.float64)
        self._output = np.zeros(3, dtype=np.float64)
        self._last_output = np.zeros(3, dtype=np.float64)
        self._filter.reset(None)
        self._last_now = None
        self._latest = None
        self._state = "waiting"
        self._zero_pending = False

    def _append(self, sample: TagSample) -> None:
        if self._window and self._window[-1][0] == sample.seq:
            return
        if self._window and sample.seq < self._window[-1][0]:
            self._window.clear()
        self._window.append((sample.seq, sample.translation_m.copy()))

    def _median(self) -> np.ndarray:
        return np.median(
            np.stack([xyz for _, xyz in self._window], axis=0),
            axis=0,
        )

    def _rebase(self, reference: np.ndarray, now: float) -> None:
        self._reference = np.asarray(reference, dtype=np.float64).reshape(3).copy()
        self._anchor = self._output.copy()
        self._filter.reset(self._anchor)
        self._last_output = self._anchor.copy()
        self._last_now = now


class TagPoseController:
    """Relative tag translation and orientation for the whole-hand base."""

    def __init__(self, settings: TagControlSettings, initial_hand_quat) -> None:
        self.settings = settings
        self._translation = _TagTranslation(settings.position)
        self._initial_hand = q_norm(initial_hand_quat)
        self._out = self._initial_hand.copy()
        self._zero_tag: np.ndarray | None = None
        self._reference_hand: np.ndarray | None = None
        self._last_t: float | None = None
        self._orientation_state = "waiting"
        self._mount = settings.mount

    # ------------------------------------------------------------- properties
    @property
    def translation(self) -> np.ndarray:
        return self._translation.output

    @property
    def position_state(self) -> str:
        return self._translation.state

    @property
    def latest_translation(self) -> np.ndarray | None:
        return self._translation.latest_translation

    @property
    def orientation(self) -> np.ndarray:
        return self._out.copy()

    @property
    def orientation_state(self) -> str:
        return self._orientation_state

    @property
    def zero_hand(self) -> np.ndarray:
        if self._reference_hand is not None:
            return self._reference_hand.copy()
        return self._initial_hand.copy()

    @property
    def initialized(self) -> bool:
        return self._zero_tag is not None

    @property
    def fixed_translation_zero(self) -> bool:
        return self._translation.fixed_translation_zero

    # ---------------------------------------------------------------- updates
    def update_translation(
        self,
        sample: TagSample | None,
        now: float,
        enabled: bool,
    ) -> np.ndarray:
        return self._translation.update(sample, now, enabled)

    def update_orientation(
        self,
        sample: TagSample | None,
        now: float,
        enabled: bool,
    ) -> np.ndarray | None:
        if not enabled:
            self._orientation_state = "paused"
            return None
        if sample is None or not sample.ready:
            self._orientation_state = "frozen"
            return None
        if now - sample.timestamp > self.settings.position.stale_timeout_s:
            self._orientation_state = "frozen"
            return None
        if self._zero_tag is None or self._reference_hand is None:
            self._orientation_state = "waiting"
            return None

        relative_tag = q_mul(q_conj(self._zero_tag), sample.quat_wxyz)
        axis, angle = q_to_axis_angle(relative_tag)
        if angle < self.settings.rotation_deadzone_rad:
            angle = 0.0
        else:
            angle = min(
                angle * self.settings.rotation_gain,
                self.settings.rotation_max_angle_rad,
            )
        axis = axis * self.settings.rotation_sign
        control_tag = q_from_axis_angle(axis, angle)
        control_hand = q_mul(
            self._mount,
            q_mul(control_tag, q_conj(self._mount)),
        )
        target = q_mul(self._reference_hand, control_hand)

        dt = 0.0 if self._last_t is None else min(max(now - self._last_t, 0.0), MAX_DT_S)
        self._last_t = now
        if dt <= 0.0:
            self._orientation_state = "tracking"
            return self._out.copy()

        tau = self.settings.rotation_smoothing_tau_s
        alpha = 1.0 if tau <= 0.0 else 1.0 - math.exp(-dt / tau)
        error = q_angle_between(self._out, target)
        if error > 1e-9 and self.settings.rotation_max_rate_rad_s > 0.0:
            alpha = min(alpha, self.settings.rotation_max_rate_rad_s * dt / error)
        self._out = q_slerp(self._out, target, alpha)
        self._orientation_state = "tracking"
        return self._out.copy()

    def zero_orientation(self, sample: TagSample | None, hand_quat) -> bool:
        if sample is None or not sample.ready:
            self._orientation_state = "waiting"
            return False
        self._zero_tag = q_norm(sample.quat_wxyz)
        self._reference_hand = q_norm(hand_quat)
        self._out = self._reference_hand.copy()
        self._last_t = None
        self._orientation_state = "tracking"
        return True

    def zero_translation(self, now: float) -> bool:
        return self._translation.zero(now)

    def zero(self, sample: TagSample | None, now: float, hand_quat) -> bool:
        """Zero translation and orientation at the current tag/hand pose."""
        if sample is not None:
            self.update_translation(sample, now, False)
        ok = self.zero_translation(now)
        ok = self.zero_orientation(sample, hand_quat) and ok
        return ok

    def reset(self) -> None:
        self._translation.reset()
        self._out = self._initial_hand.copy()
        self._zero_tag = None
        self._reference_hand = None
        self._last_t = None
        self._orientation_state = "waiting"
