"""Tests for the relative AprilTag 6-DoF hand-base controller."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadowlink.config import PositionConfig, TagConfig
from shadowlink.math3d import (
    q_angle_between,
    q_from_axis_angle,
    q_rotate,
)
from shadowlink.tag import TagControlSettings, TagPoseController, TagSample

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def position_config(**overrides) -> PositionConfig:
    values = {
        "enabled": True,
        "preset": "front",
        "deadzone_m": 0.0,
        "temporal_window": 1,
        "stale_timeout_s": 0.5,
        "auto_zero": True,
        "min_cutoff": 1e9,
        "beta": 0.0,
        "d_cutoff": 1e9,
        "max_rate_m_s": 1e9,
    }
    values.update(overrides)
    return PositionConfig(**values)


def tag_config(**overrides) -> TagConfig:
    values = {
        "zero_mode": "sample",
        "rotation_gain": 1.0,
        "rotation_deadzone_deg": 0.0,
        "rotation_max_angle_deg": 180.0,
        "rotation_smoothing_tau_s": 0.0,
        "rotation_max_rate_rad_s": 1e6,
        "rotation_jump_reject_deg": 60.0,
    }
    values.update(overrides)
    return TagConfig(**values)


def controller(position=None, tag=None) -> TagPoseController:
    settings = TagControlSettings.from_config(
        tag or tag_config(),
        position or position_config(),
    )
    return TagPoseController(settings, IDENTITY)


def sample(x=0.0, y=0.0, z=0.8, quat=IDENTITY, t=0.0, seq=0) -> TagSample:
    return TagSample(
        translation_m=[x, y, z],
        quat_wxyz=quat,
        timestamp=t,
        seq=seq,
        tag_id=5,
    )


def drive_orientation(ctrl, value: TagSample):
    """The first update after zero primes dt; the second applies the target."""
    result = None
    for now in (0.1, 0.2):
        result = ctrl.update_orientation(value, now, True)
    return result


def test_translation_is_relative_to_the_first_sample_and_uses_the_front_preset():
    ctrl = controller()
    ctrl.update_translation(sample(0.1, 0.2, 0.8, t=0.0, seq=0), 0.0, True)
    out = ctrl.update_translation(sample(0.11, 0.2, 0.8, t=0.1, seq=1), 0.1, True)
    # Camera +x (image right) -> world +y for the front preset.
    np.testing.assert_allclose(out, [0.0, 0.01, 0.0], atol=1e-6)
    assert ctrl.position_state == "tracking"


def test_overhead_preset_maps_tag_right_to_block_and_closer_to_up():
    ctrl = controller(position=position_config(preset="overhead"))
    ctrl.update_translation(sample(0.0, 0.0, 1.0, t=0.0, seq=0), 0.0, True)
    out = ctrl.update_translation(sample(0.1, 0.0, 0.9, t=0.1, seq=1), 0.1, True)
    np.testing.assert_allclose(out, [0.1, 0.0, 0.1], atol=1e-6)


def test_translation_uses_gain_and_deadzone_without_a_spatial_bound():
    ctrl = controller(position_config(deadzone_m=0.005, gain=2.0))
    ctrl.update_translation(sample(0.0, 0.0, 0.8, t=0.0, seq=0), 0.0, True)
    out = ctrl.update_translation(sample(0.02, 0.0, 0.8, t=0.1, seq=1), 0.1, True)
    # (0.02 - 0.005) * 2 = 0.03 on world +y; no positional clip.
    np.testing.assert_allclose(out, [0.0, 0.03, 0.0], atol=1e-6)

    far = ctrl.update_translation(sample(2.0, 0.0, 0.8, t=0.2, seq=2), 0.2, True)
    assert np.linalg.norm(far) > 1.0


def test_zero_freezes_the_current_translation_and_rebases():
    ctrl = controller()
    ctrl.update_translation(sample(0.0, 0.0, 0.8, t=0.0, seq=0), 0.0, True)
    moved = ctrl.update_translation(sample(0.03, 0.0, 0.8, t=0.1, seq=1), 0.1, True)
    assert np.linalg.norm(moved) > 0.02
    assert ctrl.zero_translation(0.1) is True
    np.testing.assert_allclose(ctrl.translation, moved, atol=1e-9)
    still = ctrl.update_translation(sample(0.03, 0.0, 0.8, t=0.2, seq=2), 0.2, True)
    np.testing.assert_allclose(still, moved, atol=1e-6)


def test_paused_translation_holds_and_resumes_after_zeroing():
    ctrl = controller()
    ctrl.update_translation(sample(0.0, 0.0, 0.8, t=0.0, seq=0), 0.0, True)
    ctrl.update_translation(sample(0.02, 0.0, 0.8, t=0.1, seq=1), 0.1, True)
    held = ctrl.update_translation(sample(0.05, 0.0, 0.8, t=0.2, seq=2), 0.2, False)
    assert ctrl.position_state == "paused"
    ctrl.update_translation(sample(0.05, 0.0, 0.8, t=0.3, seq=3), 0.3, False)
    ctrl.zero_translation(0.3)
    out = ctrl.update_translation(sample(0.05, 0.0, 0.8, t=0.4, seq=4), 0.4, True)
    np.testing.assert_allclose(out, held, atol=1e-6)


def test_fixed_zero_is_image_center_at_one_metre_depth():
    ctrl = controller(
        position=position_config(),
        tag=tag_config(zero_mode="fixed", zero_distance_m=1.0),
    )
    assert ctrl.fixed_translation_zero is True
    ctrl.update_translation(sample(0.0, 0.0, 1.0, t=0.0, seq=0), 0.0, True)
    np.testing.assert_allclose(ctrl.translation, 0.0, atol=1e-9)

    # Camera +y is image down, which maps to world -z for the front preset.
    out = ctrl.update_translation(sample(0.0, 0.1, 1.0, t=0.1, seq=1), 0.1, True)
    np.testing.assert_allclose(out, [0.0, 0.0, -0.1], atol=1e-6)

    ctrl.zero_translation(0.1)
    np.testing.assert_allclose(ctrl.translation, 0.0, atol=1e-12)
    out = ctrl.update_translation(sample(0.0, 0.0, 0.9, t=0.2, seq=2), 0.2, True)
    np.testing.assert_allclose(out, [0.1, 0.0, 0.0], atol=1e-6)


def test_lost_tag_holds_translation_and_orientation_in_place():
    ctrl = controller()
    ctrl.update_translation(sample(0.0, 0.0, 0.8, t=0.0, seq=0), 0.0, True)
    moved = ctrl.update_translation(sample(0.03, 0.0, 0.8, t=0.1, seq=1), 0.1, True)
    held = ctrl.update_translation(None, 0.2, True)
    np.testing.assert_allclose(held, moved, atol=1e-12)
    assert ctrl.position_state == "frozen"

    ctrl.zero_orientation(sample(t=0.0, seq=0), IDENTITY)
    ctrl.update_orientation(sample(t=0.0, seq=0), 0.1, True)
    assert ctrl.update_orientation(None, 0.2, True) is None
    assert ctrl.orientation_state == "frozen"


def test_tag_axes_map_to_hand_anatomy():
    ctrl = controller()
    np.testing.assert_allclose(q_rotate(ctrl.settings.mount, [1, 0, 0]), [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(q_rotate(ctrl.settings.mount, [0, 1, 0]), [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(q_rotate(ctrl.settings.mount, [0, 0, 1]), [1, 0, 0], atol=1e-6)


def test_orientation_rotates_about_tag_axes_mapped_to_the_hand():
    ctrl = controller()
    ctrl.zero_orientation(sample(t=0.0, seq=0), IDENTITY)
    # Rotating about tag +Z (palm direction) rotates the hand about local +X.
    rotated = q_from_axis_angle([0.0, 0.0, 1.0], math.radians(30.0))
    out = drive_orientation(ctrl, sample(quat=rotated, t=0.0, seq=1))
    expected = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(30.0))
    assert q_angle_between(out, expected) < 1e-9
    assert ctrl.orientation_state == "tracking"


def test_orientation_deadzone_gain_and_clamp():
    ctrl = controller(
        tag=tag_config(
            rotation_deadzone_deg=2.0,
            rotation_gain=2.0,
            rotation_max_angle_deg=45.0,
        )
    )
    ctrl.zero_orientation(sample(t=0.0, seq=0), IDENTITY)
    tiny = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(1.0))
    assert q_angle_between(drive_orientation(ctrl, sample(quat=tiny, t=0.0)), IDENTITY) < 1e-9

    ctrl = controller(
        tag=tag_config(
            rotation_deadzone_deg=2.0,
            rotation_gain=2.0,
            rotation_max_angle_deg=45.0,
        )
    )
    ctrl.zero_orientation(sample(t=0.0, seq=0), IDENTITY)
    large = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(40.0))
    out = drive_orientation(ctrl, sample(quat=large, t=0.0))
    # 40 * 2 = 80 degrees, clamped to 45.
    assert q_angle_between(out, IDENTITY) == pytest.approx(math.radians(45.0), abs=1e-6)


def test_orientation_sign_flips_the_axis():
    ctrl = controller(tag=tag_config(rotation_invert_z=True))
    ctrl.zero_orientation(sample(t=0.0, seq=0), IDENTITY)
    rotated = q_from_axis_angle([0.0, 0.0, 1.0], math.radians(25.0))
    out = drive_orientation(ctrl, sample(quat=rotated, t=0.0))
    expected = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(-25.0))
    assert q_angle_between(out, expected) < 1e-6


def test_isolated_orientation_jump_is_rejected_but_sustained_motion_is_accepted():
    ctrl = controller(tag=tag_config(rotation_jump_reject_deg=35.0))
    ctrl.zero_orientation(sample(t=0.0, seq=0), IDENTITY)
    ctrl.update_orientation(sample(t=0.1, seq=1), 0.1, True)

    jumped = q_from_axis_angle([0.0, 0.0, 1.0], math.radians(90.0))
    assert ctrl.update_orientation(sample(quat=jumped, t=0.2, seq=2), 0.2, True) is None
    assert ctrl.orientation_state == "rejecting"

    out = None
    for index, now in enumerate((0.3, 0.4, 0.5, 0.6), start=3):
        out = ctrl.update_orientation(sample(quat=jumped, t=now, seq=index), now, True)
    expected = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(90.0))
    assert out is not None
    assert q_angle_between(out, expected) < 1e-6


def test_stale_or_missing_tag_freezes_orientation():
    ctrl = controller()
    ctrl.zero_orientation(sample(t=0.0, seq=0), IDENTITY)
    assert ctrl.update_orientation(None, 0.1, True) is None
    assert ctrl.orientation_state == "frozen"
    assert ctrl.update_orientation(sample(quat=IDENTITY, t=0.0, seq=1), 1.0, True) is None
    assert ctrl.update_orientation(sample(t=0.0, seq=2), 0.1, False) is None
    assert ctrl.orientation_state == "paused"


def test_reset_clears_the_zero_and_translation():
    ctrl = controller()
    ctrl.zero(sample(0.0, 0.0, 0.8, t=0.0, seq=0), 0.0, IDENTITY)
    ctrl.update_translation(sample(0.02, 0.0, 0.8, t=0.1, seq=1), 0.1, True)
    assert np.linalg.norm(ctrl.translation) > 0
    ctrl.reset()
    np.testing.assert_allclose(ctrl.translation, 0.0, atol=1e-12)
    assert ctrl.initialized is False
    assert ctrl.position_state == "waiting"
