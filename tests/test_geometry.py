"""Tests for the keypoint geometry used by retargeting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadowlink.retarget.geometry import (
    HandGeometryError,
    canonicalize,
    hand_metrics,
    palm_frame,
    wrap_pi,
)
from shadowlink.types import Side


def test_straight_hand_has_zero_flexion(make_hand):
    metrics = hand_metrics(make_hand())
    for finger, angles in metrics.flex.items():
        for angle in angles:
            assert angle == pytest.approx(0.0, abs=1e-6), finger


def test_spread_and_opposition_match_construction(make_hand):
    metrics = hand_metrics(make_hand(opposition=0.25))
    assert metrics.spread["index"] == pytest.approx(-0.12, abs=1e-9)
    assert metrics.spread["middle"] == pytest.approx(0.0, abs=1e-9)
    assert metrics.spread["ring"] == pytest.approx(0.10, abs=1e-9)
    assert metrics.spread["pinky"] == pytest.approx(0.20, abs=1e-9)
    assert metrics.opposition == pytest.approx(0.25, abs=1e-9)


def test_single_joint_angle_is_exact(make_hand):
    flex = {name: (0.0, 0.0, 0.0) for name in ("index", "middle", "ring", "pinky", "thumb")}
    flex["index"] = (0.0, math.pi / 2, 0.0)
    flex["ring"] = (0.0, 0.0, math.pi / 3)
    metrics = hand_metrics(make_hand(flex=flex))
    assert metrics.flex["index"][1] == pytest.approx(math.pi / 2, abs=1e-9)
    assert metrics.flex["index"][0] == pytest.approx(0.0, abs=1e-6)
    assert metrics.flex["ring"][2] == pytest.approx(math.pi / 3, abs=1e-9)


def test_palm_frame_is_orthonormal(make_hand):
    f, s, n = palm_frame(make_hand())
    for vector in (f, s, n):
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-12)
    assert np.dot(f, s) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(f, n) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(s, n) == pytest.approx(0.0, abs=1e-12)


def test_left_hand_is_mirrored_back_to_the_same_metrics(make_hand):
    right = make_hand()
    left = right.copy()
    left[:, 0] *= -1.0
    mirrored = canonicalize(left, Side.LEFT)
    np.testing.assert_allclose(mirrored, right, atol=1e-12)
    assert hand_metrics(mirrored) == hand_metrics(right)


def test_canonicalize_does_not_mutate_input(make_hand):
    left = make_hand()
    left[:, 0] *= -1.0
    before = left.copy()
    canonicalize(left, Side.LEFT)
    np.testing.assert_array_equal(left, before)


def test_wrap_pi_handles_full_turns():
    assert wrap_pi(0.0) == pytest.approx(0.0)
    assert wrap_pi(math.radians(358.0)) == pytest.approx(math.radians(-2.0))
    assert wrap_pi(math.radians(-358.0)) == pytest.approx(math.radians(2.0))
    assert wrap_pi(math.radians(180.0)) == pytest.approx(-math.pi)
    assert -math.pi <= wrap_pi(123.456) <= math.pi


def test_spread_wrap_is_applied_inside_hand_metrics(make_hand):
    # The index proximal points at +179 degrees while the middle proximal
    # points at -179.  They are two degrees apart, but a naive subtraction
    # reports 358 degrees.
    hand = make_hand()
    theta_index = math.radians(179.0)
    theta_middle = math.radians(-179.0)
    for base, theta in ((5, theta_index), (9, theta_middle)):
        direction = np.array([math.cos(theta), math.sin(theta), 0.0])
        for step in (1, 2, 3):
            hand[base + step] = hand[base] + direction * 0.03 * step
    metrics = hand_metrics(hand)
    assert abs(metrics.spread["index"]) == pytest.approx(math.radians(2.0), abs=1e-9)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kp: kp.__setitem__((0, 0), np.nan),
        lambda kp: kp.__setitem__((5, 1), np.inf),
    ],
)
def test_non_finite_keypoints_raise(make_hand, mutate):
    kp = make_hand()
    mutate(kp)
    with pytest.raises(HandGeometryError):
        hand_metrics(kp)


def test_wrong_shape_raises():
    with pytest.raises(HandGeometryError):
        hand_metrics(np.zeros((20, 3)))


def test_degenerate_palm_raises(make_hand):
    kp = make_hand()
    kp[5] = kp[0]
    with pytest.raises(HandGeometryError):
        hand_metrics(kp)


def test_zero_length_bone_raises(make_hand):
    kp = make_hand()
    kp[7] = kp[6]
    with pytest.raises(HandGeometryError):
        hand_metrics(kp)
