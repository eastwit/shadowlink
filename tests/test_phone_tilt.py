"""Tests for phone tilt selection and tag-blue-yaw composition."""

from __future__ import annotations

import math

import numpy as np

from shadowlink.config import PhoneConfig, PositionConfig, TagConfig
from shadowlink.math3d import (
    q_angle_between,
    q_from_axis_angle,
    q_mul,
    q_swing_twist,
)
from shadowlink.phone.service import PhoneSample
from shadowlink.phone.tilt import (
    MOUNT_OPTIONS,
    PhoneTiltController,
    PhoneTiltSettings,
)
from shadowlink.tag.controller import TagControlSettings, TagPoseController
from shadowlink.tag.types import TagSample


def phone_sample(quat, ts=1.0, seq=1):
    return PhoneSample(
        quat_wxyz=np.asarray(quat, dtype=np.float64),
        timestamp=ts,
        seq=seq,
        ready=True,
    )


def tag_sample(quat, ts=1.0, seq=1):
    return TagSample(
        translation_m=[0.0, 0.0, 1.0],
        quat_wxyz=np.asarray(quat, dtype=np.float64),
        timestamp=ts,
        seq=seq,
        tag_id=5,
    )


def identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def test_swing_twist_reconstructs_the_rotation_and_isolates_the_axis():
    tilt = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(25.0))
    yaw = q_from_axis_angle([0.0, 0.0, 1.0], math.radians(40.0))
    combined = q_mul(tilt, yaw)
    swing, twist = q_swing_twist(combined, [0.0, 0.0, 1.0])
    assert q_angle_between(q_mul(swing, twist), combined) < 1e-9
    assert q_angle_between(twist, yaw) < 1e-9
    np.testing.assert_allclose(twist[1:3], [0.0, 0.0], atol=1e-12)


def test_mount_cycle_covers_every_axis_mapping():
    settings = PhoneTiltSettings(
        mount=MOUNT_OPTIONS[0][1],
        gain=1.0,
        deadzone_rad=0.0,
        max_angle_rad=math.pi,
        stale_timeout_s=0.5,
    )
    controller = PhoneTiltController(settings)
    assert controller.mount_name == MOUNT_OPTIONS[0][0]
    names = {controller.cycle_mount() for _ in range(len(MOUNT_OPTIONS))}
    assert names == {name for name, _ in MOUNT_OPTIONS}


def test_gravity_mode_uses_tilt_without_yaw():
    settings = PhoneTiltSettings(
        mount=identity(),
        gain=1.0,
        deadzone_rad=0.0,
        max_angle_rad=math.pi,
        stale_timeout_s=0.5,
        axes="pitch_roll",
    )
    controller = PhoneTiltController(settings)
    assert controller.zero(phone_sample(identity(), ts=1.0))

    yaw_only = phone_sample(q_from_axis_angle([0.0, 0.0, 1.0], math.radians(50.0)), ts=1.1)
    delta = controller.update(yaw_only, now=1.1)
    assert delta is not None
    assert q_angle_between(delta, identity()) < 1e-9

    tilt = phone_sample(q_from_axis_angle([1.0, 0.0, 0.0], math.radians(30.0)), ts=1.2)
    delta = controller.update(tilt, now=1.2)
    assert delta is not None
    expected = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(30.0))
    assert q_angle_between(delta, expected) < 1e-6

    assert controller.update(tilt, now=2.0) is None  # stale


def test_yaw_pitch_mode_discards_roll():
    settings = PhoneTiltSettings(
        mount=identity(),
        gain=1.0,
        deadzone_rad=0.0,
        max_angle_rad=math.pi,
        stale_timeout_s=0.5,
        axes="yaw_pitch",
    )
    controller = PhoneTiltController(settings)
    assert controller.zero(phone_sample(identity(), ts=1.0))

    roll = phone_sample(q_from_axis_angle([0.0, 0.0, 1.0], math.radians(40.0)), ts=1.1)
    delta = controller.update(roll, now=1.1)
    assert delta is not None
    assert q_angle_between(delta, identity()) < 1e-9

    for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
        rotated = phone_sample(q_from_axis_angle(axis, math.radians(30.0)), ts=1.2)
        delta = controller.update(rotated, now=1.2)
        assert delta is not None
        expected = q_from_axis_angle(axis, math.radians(30.0))
        assert q_angle_between(delta, expected) < 1e-6


def test_hybrid_controller_keeps_tag_yaw_and_phone_yaw_pitch():
    phone_cfg = PhoneConfig(
        source="phyphox",
        host="192.0.2.1",
        axes="yaw_pitch",
        mount_preset="yaw_pitch",
        deadzone_deg=0.0,
    )
    settings = TagControlSettings.from_config(
        TagConfig(
            zero_mode="fixed",
            rotation_deadzone_deg=0.0,
            rotation_smoothing_tau_s=0.0,
            rotation_max_rate_rad_s=1e6,
        ),
        PositionConfig(),
        phone_cfg,
    )
    controller = TagPoseController(settings, identity())
    t0 = tag_sample(identity(), ts=1.0)
    p0 = phone_sample(identity(), ts=1.0)
    assert controller.zero_orientation(t0, identity(), phone_sample=p0)
    assert controller.phone_initialized
    assert controller.update_orientation(t0, 1.01, True, phone_sample=p0) is not None

    # Tag blue-axis yaw drives the hand yaw.
    yaw = q_from_axis_angle([0.0, 0.0, 1.0], math.radians(25.0))
    t1 = tag_sample(yaw, ts=1.02)
    controller.update_orientation(t1, 1.02, True, phone_sample=p0)
    out = controller.update_orientation(t1, 1.03, True, phone_sample=p0)
    assert out is not None
    assert abs(abs(out[1]) - abs(math.sin(math.radians(12.5)))) < 1e-3

    # Tag pitch alone is ignored; only its blue-axis twist is used.
    tag_pitch = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(35.0))
    controller.update_orientation(
        tag_sample(tag_pitch, ts=1.04), 1.04, True, phone_sample=p0
    )
    out = controller.update_orientation(
        tag_sample(tag_pitch, ts=1.05), 1.05, True, phone_sample=p0
    )
    assert out is not None
    assert q_angle_between(out, q_from_axis_angle([0.0, 0.0, 1.0], 0.0)) < 1e-6

    # Phone yaw (+X) maps to hand roll.
    phone_yaw = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(20.0))
    p1 = phone_sample(phone_yaw, ts=1.06, seq=2)
    controller.update_orientation(t0, 1.06, True, phone_sample=p1)
    out = controller.update_orientation(t0, 1.07, True, phone_sample=p1)
    assert out is not None
    assert abs(out[3] + math.sin(math.radians(10.0))) < 2e-3

    # Phone pitch (+Y) maps to hand pitch.
    phone_pitch = q_from_axis_angle([0.0, 1.0, 0.0], math.radians(20.0))
    p2 = phone_sample(phone_pitch, ts=1.08, seq=3)
    controller.update_orientation(t0, 1.08, True, phone_sample=p2)
    out = controller.update_orientation(t0, 1.09, True, phone_sample=p2)
    assert out is not None
    assert abs(out[2] + math.sin(math.radians(10.0))) < 2e-3

    # Phone roll about the screen normal (+Z) is discarded.
    phone_roll = q_from_axis_angle([0.0, 0.0, 1.0], math.radians(30.0))
    p3 = phone_sample(phone_roll, ts=1.10, seq=4)
    controller.update_orientation(t0, 1.10, True, phone_sample=p3)
    out = controller.update_orientation(t0, 1.11, True, phone_sample=p3)
    assert out is not None
    assert q_angle_between(out, identity()) < 1e-6

    assert controller.update_orientation(t0, 1.12, True, phone_sample=None) is None
