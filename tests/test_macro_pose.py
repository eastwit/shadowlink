"""Tests for relative macro-pose mapping and smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from shadowlink.config import PhoneConfig
from shadowlink.math3d import q_angle_between, q_from_axis_angle, q_mul
from shadowlink.retarget.macro_pose import (
    BACK_MOUNT,
    IN_HAND_MOUNT,
    MacroPoseController,
    MacroPoseSettings,
    mount_for_config,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class Sample:
    quat_wxyz: np.ndarray
    timestamp: float
    ready: bool = True


def settings(**overrides) -> MacroPoseSettings:
    values = {
        "mount": IDENTITY.copy(),
        "gain": 1.0,
        "deadzone_rad": 0.0,
        "max_angle_rad": math.radians(75.0),
        "smoothing_tau_s": 0.0,
        "max_rate_rad_s": 1000.0,
        "stale_timeout_s": 0.5,
    }
    values.update(overrides)
    return MacroPoseSettings(**values)


def rotated(axis, degrees) -> np.ndarray:
    return q_from_axis_angle(axis, math.radians(degrees))


def drive(controller, sample, times=(0.0, 0.1)):
    result = None
    for now in times:
        result = controller.update(sample, now, True)
    return result


def test_no_jump_and_relative_rotation():
    reference = rotated([0.0, 0.0, 1.0], 20.0)
    controller = MacroPoseController(settings(), reference)
    zero = rotated([0.0, 0.0, 1.0], 5.0)
    controller.set_zero(Sample(zero, 0.0), reference)
    assert q_angle_between(controller.orientation, reference) < 1e-9

    sample = Sample(rotated([0.0, 0.0, 1.0], 35.0), 0.0)
    output = drive(controller, sample)
    expected = q_mul(reference, rotated([0.0, 0.0, 1.0], 30.0))
    assert q_angle_between(output, expected) < 1e-6


def test_rezero_mid_follow_preserves_the_current_hand_pose():
    initial = rotated([0.0, 0.0, 1.0], 10.0)
    controller = MacroPoseController(settings(), initial)
    controller.set_zero(Sample(IDENTITY, 0.0), initial)

    first = Sample(rotated([1.0, 0.0, 0.0], 20.0), 0.0)
    out = drive(controller, first)
    assert q_angle_between(out, initial) > 0.1

    # Re-zero at the current phone attitude while keeping the visible hand pose.
    controller.set_zero(Sample(first.quat_wxyz, 0.0), out)
    assert q_angle_between(controller.orientation, out) < 1e-9
    assert q_angle_between(drive(controller, Sample(first.quat_wxyz, 0.0)), out) < 1e-9

    # Subsequent phone motion is relative to the new zero, not the old one.
    next_sample = Sample(q_mul(first.quat_wxyz, rotated([1.0, 0.0, 0.0], 15.0)), 0.0)
    expected = q_mul(out, rotated([1.0, 0.0, 0.0], 15.0))
    assert q_angle_between(drive(controller, next_sample), expected) < 1e-6


def test_deadzone_holds_the_reference():
    controller = MacroPoseController(settings(deadzone_rad=math.radians(1.5)), IDENTITY)
    controller.set_zero(Sample(IDENTITY, 0.0), IDENTITY)
    output = drive(controller, Sample(rotated([1.0, 0.0, 0.0], 1.0), 0.0))
    assert q_angle_between(output, IDENTITY) < 1e-9


def test_gain_scales_the_angle():
    controller = MacroPoseController(settings(gain=2.0), IDENTITY)
    controller.set_zero(Sample(IDENTITY, 0.0), IDENTITY)
    output = drive(controller, Sample(rotated([1.0, 0.0, 0.0], 10.0), 0.0))
    assert q_angle_between(output, IDENTITY) == pytest.approx(math.radians(20.0), abs=1e-6)


def test_max_angle_clamps_the_output():
    controller = MacroPoseController(settings(max_angle_rad=math.radians(75.0)), IDENTITY)
    controller.set_zero(Sample(IDENTITY, 0.0), IDENTITY)
    output = drive(controller, Sample(rotated([1.0, 0.0, 0.0], 100.0), 0.0))
    assert q_angle_between(output, IDENTITY) == pytest.approx(math.radians(75.0), abs=1e-6)


def test_mount_presets_select_the_expected_frame():
    assert q_angle_between(mount_for_config(PhoneConfig()), BACK_MOUNT) < 1e-9
    assert (
        q_angle_between(mount_for_config(PhoneConfig(mount_preset="in_hand")), IN_HAND_MOUNT) < 1e-9
    )
    custom = PhoneConfig(
        mount_preset="custom",
        mount_w=1.0,
        mount_x=2.0,
        mount_y=0.0,
        mount_z=0.0,
    )
    np.testing.assert_allclose(mount_for_config(custom), [1.0, 2.0, 0.0, 0.0] / np.sqrt(5.0))
    # Presets ignore the custom quaternion fields.
    ignored = PhoneConfig(
        mount_preset="back",
        mount_w=1.0,
        mount_x=2.0,
        mount_y=0.0,
        mount_z=0.0,
    )
    assert q_angle_between(mount_for_config(ignored), BACK_MOUNT) < 1e-9


def test_mount_conjugation_maps_controller_axes_to_hand_axes():
    mount = rotated([0.0, 0.0, 1.0], 90.0)
    controller = MacroPoseController(settings(mount=mount), IDENTITY)
    controller.set_zero(Sample(IDENTITY, 0.0), IDENTITY)
    output = drive(controller, Sample(rotated([1.0, 0.0, 0.0], 30.0), 0.0))
    expected = rotated([0.0, 1.0, 0.0], 30.0)
    assert q_angle_between(output, expected) < 1e-6


def test_stale_or_unready_samples_freeze():
    controller = MacroPoseController(settings(stale_timeout_s=0.2), IDENTITY)
    controller.set_zero(Sample(IDENTITY, 0.0), IDENTITY)
    assert controller.update(Sample(rotated([1.0, 0.0, 0.0], 30.0), 0.0), 1.0, True) is None
    assert (
        controller.update(Sample(rotated([1.0, 0.0, 0.0], 30.0), 0.0, ready=False), 0.1, True)
        is None
    )
    assert controller.update(None, 0.1, True) is None
    assert controller.update(Sample(rotated([1.0, 0.0, 0.0], 30.0), 0.1), 0.1, False) is None


def test_smoothing_and_rate_limit():
    controller = MacroPoseController(settings(smoothing_tau_s=0.1, max_rate_rad_s=1000.0), IDENTITY)
    controller.set_zero(Sample(IDENTITY, 0.0), IDENTITY)
    controller.update(Sample(ROT90, 0.0), 0.0, True)
    output = controller.update(Sample(ROT90, 0.1), 0.1, True)
    angle = q_angle_between(IDENTITY, output)
    assert math.radians(40.0) < angle < math.radians(70.0)

    limited = MacroPoseController(settings(smoothing_tau_s=0.0, max_rate_rad_s=1.0), IDENTITY)
    limited.set_zero(Sample(IDENTITY, 0.0), IDENTITY)
    limited.update(Sample(ROT90, 0.0), 0.0, True)
    output = limited.update(Sample(ROT90, 0.1), 0.1, True)
    assert q_angle_between(IDENTITY, output) <= math.radians(6.0)
    assert q_angle_between(IDENTITY, output) > 0.0


ROT90 = rotated([1.0, 0.0, 0.0], 90.0)


def test_zero_hand_reports_the_visible_reference_orientation():
    initial = rotated([0.0, 1.0, 0.0], 15.0)
    controller = MacroPoseController(settings(), initial)
    assert q_angle_between(controller.zero_hand, initial) < 1e-9

    visible = rotated([1.0, 0.0, 0.0], 40.0)
    controller.set_zero(Sample(IDENTITY, 0.0), visible)
    assert q_angle_between(controller.zero_hand, visible) < 1e-9

    controller.reset()
    assert q_angle_between(controller.zero_hand, initial) < 1e-9


def test_reset_returns_the_initial_orientation():
    initial = rotated([0.0, 1.0, 0.0], 15.0)
    controller = MacroPoseController(settings(), initial)
    controller.set_zero(Sample(IDENTITY, 0.0), initial)
    drive(controller, Sample(ROT90, 0.0))
    assert q_angle_between(controller.orientation, initial) > 0.1
    controller.reset()
    assert q_angle_between(controller.orientation, initial) < 1e-9
    assert not controller.initialized
