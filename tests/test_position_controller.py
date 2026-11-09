"""Tests for the relative camera-to-world translation controller."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.position.controller import (
    PRESET_MATRICES,
    PositionController,
    PositionSettings,
)
from shadowlink.position.depth_sampling import PalmMeasurement


def make_settings(**overrides) -> PositionSettings:
    values = dict(
        rotation=np.eye(3),
        sign=np.ones(3),
        gain=1.0,
        deadzone_m=0.0,
        max_offset_m=1.0,
        temporal_window=1,
        stale_timeout_s=0.5,
        auto_zero=True,
        min_cutoff=1e9,
        beta=0.0,
        d_cutoff=1e9,
        max_rate_m_s=1e9,
    )
    values.update(overrides)
    return PositionSettings(**values)


def measurement(xyz, seq: int, ts: float) -> PalmMeasurement:
    return PalmMeasurement(
        xyz_m=np.asarray(xyz, dtype=np.float64),
        frame_seq=seq,
        capture_ts=ts,
        depth_ts=ts,
        valid_landmarks=5,
        spread_m=0.0,
    )


def test_the_first_sample_zeros_the_mapping():
    controller = PositionController(make_settings())
    result = controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    assert controller.state == "tracking"
    np.testing.assert_allclose(result, 0.0, atol=1e-12)
    np.testing.assert_allclose(controller.output, 0.0)


def test_motion_is_applied_one_to_one_from_the_reference():
    controller = PositionController(make_settings())
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    result = controller.update(measurement([0.1, -0.05, 1.0], 2, 1.1), 1.1)
    np.testing.assert_allclose(result, [0.1, -0.05, 0.0], atol=1e-6)


def test_the_soft_deadzone_removes_jitter_without_a_jump():
    controller = PositionController(make_settings(deadzone_m=0.05))
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    inside = controller.update(measurement([0.03, 0.0, 1.0], 2, 1.1), 1.1)
    np.testing.assert_allclose(inside, 0.0, atol=1e-12)
    outside = controller.update(measurement([0.1, 0.0, 1.0], 3, 1.2), 1.2)
    np.testing.assert_allclose(outside, [0.05, 0.0, 0.0], atol=1e-6)


def test_the_preset_rotates_camera_axes_into_the_world():
    controller = PositionController(make_settings(rotation=PRESET_MATRICES["front"]))
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    result = controller.update(measurement([0.1, 0.0, 1.0], 2, 1.1), 1.1)
    np.testing.assert_allclose(result, [0.0, 0.1, 0.0], atol=1e-6)


def test_the_palm_preset_maps_motion_toward_the_camera_to_the_palm_normal():
    controller = PositionController(make_settings(rotation=PRESET_MATRICES["palm"]))
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    # Closer to the camera (smaller z) pushes the hand along its palm normal.
    result = controller.update(measurement([0.0, 0.0, 0.9], 2, 1.1), 1.1)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.1], atol=1e-6)

    farther = controller.update(measurement([0.0, 0.0, 1.1], 3, 1.2), 1.2)
    np.testing.assert_allclose(farther, [0.0, 0.0, -0.1], atol=1e-6)


def test_a_per_sample_rotation_overrides_the_static_preset():
    controller = PositionController(make_settings())
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    # 90 degrees about y: camera x -> world -z, camera z -> world +x.
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    result = controller.update(
        measurement([0.1, 0.0, 1.0], 2, 1.1),
        1.1,
        rotation=rotation,
    )
    np.testing.assert_allclose(result, [0.0, 0.0, -0.1], atol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 2)), np.full((3, 3), np.nan), np.zeros(3)],
)
def test_an_invalid_per_sample_rotation_is_rejected(bad):
    controller = PositionController(make_settings())
    with pytest.raises(ValueError):
        controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0, rotation=bad)


def test_invert_flags_and_gain_are_applied():
    controller = PositionController(make_settings(sign=np.array([1.0, -1.0, 1.0]), gain=2.0))
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    result = controller.update(measurement([0.1, 0.05, 1.0], 2, 1.1), 1.1)
    np.testing.assert_allclose(result, [0.2, -0.1, 0.0], atol=1e-6)


def test_the_output_is_clamped_to_the_maximum_offset():
    controller = PositionController(make_settings(max_offset_m=0.05))
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    result = controller.update(measurement([1.0, 0.0, 1.0], 2, 1.1), 1.1)
    np.testing.assert_allclose(result, [0.05, 0.0, 0.0], atol=1e-6)


def test_the_slew_limiter_bounds_the_per_update_step():
    controller = PositionController(make_settings(max_offset_m=10.0, max_rate_m_s=0.5))
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    result = controller.update(measurement([1.0, 0.0, 1.0], 2, 1.1), 1.1)
    assert 0.0 < result[0] <= 0.05 + 1e-12


def test_losing_the_measurement_freezes_the_output():
    controller = PositionController(make_settings())
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    tracked = controller.update(measurement([0.1, 0.0, 1.0], 2, 1.1), 1.1)
    frozen = controller.update(None, 1.5)
    np.testing.assert_allclose(frozen, tracked)
    assert controller.state == "frozen"
    stale = controller.update(measurement([0.2, 0.0, 1.0], 3, 1.0), 2.0)
    np.testing.assert_allclose(stale, tracked)
    assert controller.state == "frozen"


def test_zero_rebases_without_moving_the_output():
    controller = PositionController(make_settings())
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    controller.update(measurement([0.1, 0.0, 1.0], 2, 1.1), 1.1)
    before = controller.output.copy()

    assert controller.zero(1.2) is True
    np.testing.assert_allclose(controller.output, before)
    shifted = controller.update(measurement([0.15, 0.0, 1.0], 3, 1.3), 1.3)
    np.testing.assert_allclose(shifted, before + [0.05, 0.0, 0.0], atol=1e-6)


def test_zero_without_a_sample_is_deferred_to_the_next_update():
    controller = PositionController(make_settings())
    assert controller.zero(1.0) is False
    result = controller.update(measurement([0.1, 0.0, 1.0], 1, 1.1), 1.1)
    assert controller.state == "tracking"
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_auto_zero_can_be_disabled_until_the_user_zeroes():
    controller = PositionController(make_settings(auto_zero=False))
    result = controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    assert controller.state == "waiting"
    np.testing.assert_allclose(result, 0.0)

    assert controller.zero(1.0) is True
    assert controller.state == "tracking"
    result = controller.update(measurement([0.1, 0.0, 1.0], 2, 1.1), 1.1)
    np.testing.assert_allclose(result, [0.1, 0.0, 0.0], atol=1e-6)


def test_reset_clears_the_reference_and_output():
    controller = PositionController(make_settings())
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    controller.update(measurement([0.1, 0.0, 1.0], 2, 1.1), 1.1)

    controller.reset()

    assert controller.state == "waiting"
    assert controller.camera_position is None
    np.testing.assert_allclose(controller.output, 0.0)


def test_camera_position_exposes_the_latest_accepted_sample():
    controller = PositionController(make_settings())
    controller.update(measurement([0.0, 0.0, 1.0], 1, 1.0), 1.0)
    np.testing.assert_allclose(controller.camera_position, [0.0, 0.0, 1.0])
    controller.update(None, 1.1)
    np.testing.assert_allclose(controller.camera_position, [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "overrides",
    [
        {"gain": 0.0},
        {"gain": float("nan")},
        {"max_offset_m": 0.0},
        {"deadzone_m": -0.1},
        {"deadzone_m": 1.0, "max_offset_m": 1.0},
        {"temporal_window": 0},
        {"stale_timeout_s": 0.0},
        {"max_rate_m_s": 0.0},
        {"rotation": np.zeros((2, 2))},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        make_settings(**overrides)
