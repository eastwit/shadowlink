"""Target smoothing, rate limiting, and the lost-hand state machine.

Every filter here is vectorised over the 20 actuator targets and keeps its own
history.  Filtering is deliberately separated from rate limiting: the filters
remove jitter, while :class:`SlewLimiter` bounds how far a target may move in
one control period so that a re-detected hand never snaps the simulation.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

import numpy as np

from shadowlink.config import SmoothingConfig

# A long stall (debugger, window drag, model reload) should not be integrated as
# one huge step.  Clamping the period keeps rate limits meaningful.
MAX_DT_S = 0.1


class SmoothingError(ValueError):
    """Raised for non-finite or malformed targets."""


class TrackingState(str, Enum):
    WAITING = "waiting"
    TRACKING = "tracking"
    LOST = "lost"


def _as_target(target, size: int | None = None) -> np.ndarray:
    arr = np.asarray(target, dtype=np.float64).reshape(-1)
    if size is not None and arr.shape != (size,):
        raise SmoothingError(f"expected {size} targets, got {arr.shape}")
    if arr.size == 0:
        raise SmoothingError("target vector is empty")
    if not np.isfinite(arr).all():
        raise SmoothingError("target vector contains NaN or infinity")
    return arr


def _alpha(cutoff, dt: float):
    """First-order low-pass coefficient for a cutoff frequency in Hz."""
    return 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))


class ExponentialFilter:
    """First-order exponential low-pass, matching the previous implementation."""

    def __init__(self, tau: float) -> None:
        if not math.isfinite(tau) or tau < 0:
            raise SmoothingError("tau must be finite and >= 0")
        self.tau = float(tau)
        self._value: np.ndarray | None = None

    def reset(self, value=None) -> None:
        self._value = None if value is None else _as_target(value)

    def filter(self, target, dt: float) -> np.ndarray:
        target = _as_target(target, None if self._value is None else self._value.size)
        if self._value is None:
            self._value = target.copy()
            return target.copy()
        if dt <= 0.0 or self.tau <= 0.0:
            self._value = target.copy()
            return target.copy()
        alpha = 1.0 - math.exp(-dt / self.tau)
        self._value = self._value + (target - self._value) * alpha
        return self._value.copy()


class OneEuroFilter:
    """Adaptive low-pass: more smoothing at rest, less lag during fast motion."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        for name, value in (
            ("min_cutoff", min_cutoff),
            ("beta", beta),
            ("d_cutoff", d_cutoff),
        ):
            if not math.isfinite(value) or value < 0:
                raise SmoothingError(f"{name} must be finite and >= 0")
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise SmoothingError("cutoff frequencies must be > 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._value: np.ndarray | None = None
        self._derivative: np.ndarray | None = None

    def reset(self, value=None) -> None:
        self._value = None if value is None else _as_target(value)
        self._derivative = None if value is None else np.zeros_like(self._value)

    def filter(self, target, dt: float) -> np.ndarray:
        target = _as_target(target, None if self._value is None else self._value.size)
        if self._value is None:
            self._value = target.copy()
            self._derivative = np.zeros_like(target)
            return target.copy()
        if dt <= 0.0:
            return self._value.copy()

        derivative = (target - self._value) / dt
        alpha_d = _alpha(self.d_cutoff, dt)
        self._derivative = alpha_d * derivative + (1.0 - alpha_d) * self._derivative
        cutoff = self.min_cutoff + self.beta * np.abs(self._derivative)
        alpha = _alpha(cutoff, dt)
        self._value = alpha * target + (1.0 - alpha) * self._value
        return self._value.copy()


class SlewLimiter:
    """Limit the per-update change to ``rate * dt`` radians."""

    def __init__(self, rate_rad_s) -> None:
        rate = np.asarray(rate_rad_s, dtype=np.float64).reshape(-1)
        if not np.isfinite(rate).all() or np.any(rate <= 0):
            raise SmoothingError("slew rate must be finite and > 0")
        self.rate = float(rate[0]) if rate.size == 1 else rate

    def limit(self, current, target, dt: float) -> np.ndarray:
        current = _as_target(current)
        target = _as_target(target, current.size)
        if dt <= 0.0:
            return current.copy()
        max_step = self.rate * dt
        return current + np.clip(target - current, -max_step, max_step)


def make_filter(cfg: SmoothingConfig):
    kind: Literal["none", "exponential", "one_euro"] = cfg.kind
    if kind == "none":
        return None
    if kind == "exponential":
        return ExponentialFilter(tau=cfg.tau)
    if kind == "one_euro":
        return OneEuroFilter(min_cutoff=cfg.min_cutoff, beta=cfg.beta, d_cutoff=cfg.d_cutoff)
    raise SmoothingError(f"unknown smoothing kind: {kind!r}")


class TargetController:
    """Turn (possibly missing) retargeted targets into smooth actuator targets.

    States::

        waiting --first observation--> tracking --no observation--> lost

    ``lost`` keeps the last target for ``hold_s`` so that a single dropped frame
    is invisible, then moves back to neutral at ``return_rate_rad_s``.  Tracking
    changes are bounded by ``max_rate_rad_s``.
    """

    def __init__(self, cfg: SmoothingConfig, neutral) -> None:
        self.cfg = cfg
        self._neutral = _as_target(neutral)
        self._filter = make_filter(cfg)
        self._rate = SlewLimiter(cfg.max_rate_rad_s)
        self._state = TrackingState.WAITING
        self._output = self._neutral.copy()
        self._last_update: float | None = None
        self._last_seen: float | None = None
        self._last_dt = 0.0

    # ------------------------------------------------------------- properties
    @property
    def state(self) -> TrackingState:
        return self._state

    @property
    def neutral(self) -> np.ndarray:
        return self._neutral.copy()

    @property
    def output(self) -> np.ndarray:
        return self._output.copy()

    @property
    def last_dt(self) -> float:
        return self._last_dt

    def reset(self) -> None:
        if self._filter is not None:
            self._filter.reset()
        self._state = TrackingState.WAITING
        self._output = self._neutral.copy()
        self._last_update = None
        self._last_seen = None
        self._last_dt = 0.0

    def update(self, target, now: float) -> np.ndarray:
        """Advance the controller; ``target=None`` means "no fresh hand"."""
        if not math.isfinite(now):
            raise SmoothingError("timestamp must be finite")
        dt = 0.0 if self._last_update is None else min(max(now - self._last_update, 0.0), MAX_DT_S)
        self._last_update = now
        self._last_dt = dt

        if target is None:
            return self._handle_missing(now, dt)

        target = _as_target(target, self._neutral.size)
        self._last_seen = now
        if self._state is not TrackingState.TRACKING:
            self._state = TrackingState.TRACKING
        smoothed = self._filter.filter(target, dt) if self._filter is not None else target
        self._output = self._rate.limit(self._output, smoothed, dt)
        return self._output.copy()

    def _handle_missing(self, now: float, dt: float) -> np.ndarray:
        if self._last_seen is None:
            # Nothing has ever been tracked: the hand has not appeared yet.
            self._state = TrackingState.WAITING
            self._output = self._neutral.copy()
            return self._output.copy()

        self._state = TrackingState.LOST
        elapsed = now - self._last_seen
        if elapsed <= self.cfg.hold_s:
            return self._output.copy()

        max_step = self.cfg.return_rate_rad_s * dt
        delta = self._neutral - self._output
        self._output = self._output + np.clip(delta, -max_step, max_step)
        return self._output.copy()
