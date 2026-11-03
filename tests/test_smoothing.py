"""Tests for filters, rate limiting, and the lost-hand state machine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadowlink.config import SmoothingConfig
from shadowlink.retarget.smoothing import (
    ExponentialFilter,
    OneEuroFilter,
    SlewLimiter,
    SmoothingError,
    TargetController,
    TrackingState,
    make_filter,
)


def test_exponential_filter_matches_the_closed_form():
    filt = ExponentialFilter(tau=0.1)
    assert filt.filter([0.0], dt=0.1)[0] == 0.0
    out = filt.filter([1.0], dt=0.1)[0]
    assert out == pytest.approx(1.0 - math.exp(-1.0), abs=1e-12)


def test_exponential_filter_initializes_to_first_target():
    filt = ExponentialFilter(tau=0.05)
    np.testing.assert_allclose(filt.filter([3.0, -2.0], dt=0.016), [3.0, -2.0])


def test_exponential_filter_with_zero_tau_is_passthrough():
    filt = ExponentialFilter(tau=0.0)
    filt.filter([0.0], dt=0.1)
    assert filt.filter([1.0], dt=0.1)[0] == 1.0


def test_one_euro_does_not_overshoot_and_is_monotone():
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.05, d_cutoff=1.0)
    filt.filter(np.zeros(3), dt=0.016)
    previous = np.zeros(3)
    for _ in range(120):
        current = filt.filter(np.ones(3), dt=0.016)
        assert np.all(current >= 0.0) and np.all(current <= 1.0)
        assert np.all(current >= previous - 1e-12)
        previous = current
    assert np.all(current > 0.5)


def test_one_euro_stays_within_the_target_range_on_reversal():
    filt = OneEuroFilter(min_cutoff=0.5, beta=0.5, d_cutoff=1.0)
    filt.filter(np.zeros(1), dt=0.016)
    for _ in range(30):
        out = filt.filter(np.ones(1), dt=0.016)
    for _ in range(30):
        out = filt.filter(np.zeros(1), dt=0.016)
        assert np.all(out >= -1e-12) and np.all(out <= 1.0 + 1e-12)


def test_slew_limiter_bounds_one_step():
    limiter = SlewLimiter(10.0)
    out = limiter.limit([0.0, 0.0], [1.0, -1.0], dt=0.05)
    np.testing.assert_allclose(out, [0.5, -0.5])
    assert limiter.limit(out, [1.0, -1.0], dt=0.0).tolist() == out.tolist()


def test_slew_limiter_rejects_invalid_rates():
    for rate in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(SmoothingError):
            SlewLimiter(rate)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("none", type(None)), ("exponential", ExponentialFilter), ("one_euro", OneEuroFilter)],
)
def test_filter_factory(kind, expected):
    assert isinstance(make_filter(SmoothingConfig(kind=kind)), expected)


def controller(**kwargs) -> TargetController:
    cfg = SmoothingConfig(
        kind="none",
        max_rate_rad_s=10.0,
        return_rate_rad_s=5.0,
        hold_s=0.15,
        lost_timeout_s=0.5,
        **kwargs,
    )
    return TargetController(cfg, np.zeros(2))


def test_controller_waits_before_the_first_observation():
    ctrl = controller()
    out = ctrl.update(None, now=0.0)
    assert ctrl.state is TrackingState.WAITING
    np.testing.assert_allclose(out, [0.0, 0.0])


def test_controller_rate_limits_tracking():
    ctrl = controller()
    np.testing.assert_allclose(ctrl.update([1.0, 1.0], now=0.0), [0.0, 0.0])
    assert ctrl.state is TrackingState.TRACKING
    np.testing.assert_allclose(ctrl.update([1.0, 1.0], now=0.1), [1.0, 1.0], atol=1e-12)


def test_controller_holds_then_returns_to_neutral_monotonically():
    ctrl = controller()
    ctrl.update([1.0, -1.0], now=0.0)
    ctrl.update([1.0, -1.0], now=0.1)

    held = ctrl.update(None, now=0.2)
    assert ctrl.state is TrackingState.LOST
    np.testing.assert_allclose(held, [1.0, -1.0])

    distances = []
    now = 0.36
    for _ in range(20):
        out = ctrl.update(None, now=now)
        distances.append(float(np.linalg.norm(out)))
        now += 0.1
    assert distances == sorted(distances, reverse=True)
    assert distances[-1] == pytest.approx(0.0, abs=1e-12)


def test_controller_rejects_non_finite_targets():
    ctrl = controller()
    with pytest.raises(SmoothingError):
        ctrl.update([np.nan, 0.0], now=0.0)
    with pytest.raises(SmoothingError):
        ctrl.update([np.inf], now=0.0)
    with pytest.raises(SmoothingError):
        ctrl.update([1.0, 2.0, 3.0], now=0.0)


def test_controller_resets_to_waiting():
    ctrl = controller()
    ctrl.update([1.0, 1.0], now=0.0)
    ctrl.update([1.0, 1.0], now=0.1)
    ctrl.reset()
    assert ctrl.state is TrackingState.WAITING
    np.testing.assert_allclose(ctrl.output, [0.0, 0.0])


def test_controller_resumes_tracking_after_a_loss():
    ctrl = controller()
    ctrl.update([1.0, 1.0], now=0.0)
    ctrl.update([1.0, 1.0], now=0.1)
    ctrl.update(None, now=0.3)
    ctrl.update([0.5, 0.5], now=0.4)
    assert ctrl.state is TrackingState.TRACKING
    assert float(np.max(ctrl.output)) <= 1.0
