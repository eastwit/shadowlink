"""Tests for the quaternion helpers shared by controller and retargeting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadowlink.math3d import (
    q_angle_between,
    q_conj,
    q_from_axis_angle,
    q_from_two_vectors,
    q_mul,
    q_norm,
    q_rotate,
    q_slerp,
    q_to_axis_angle,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_identity_and_conjugate():
    np.testing.assert_allclose(q_mul(IDENTITY, IDENTITY), IDENTITY)
    q = q_from_axis_angle([0.0, 0.0, 1.0], math.pi / 2)
    np.testing.assert_allclose(q_mul(q, q_conj(q)), IDENTITY, atol=1e-12)


def test_rotation_direction_matches_right_hand_rule():
    q = q_from_axis_angle([0.0, 0.0, 1.0], math.pi / 2)
    np.testing.assert_allclose(q_rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_axis_angle_round_trip():
    axis = np.array([1.0, -2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    q = q_from_axis_angle(axis, math.radians(37.0))
    recovered_axis, recovered_angle = q_to_axis_angle(q)
    np.testing.assert_allclose(recovered_axis, axis, atol=1e-12)
    assert recovered_angle == pytest.approx(math.radians(37.0), abs=1e-12)

    flipped = -q
    recovered_axis, recovered_angle = q_to_axis_angle(flipped)
    np.testing.assert_allclose(recovered_axis, axis, atol=1e-12)
    assert recovered_angle == pytest.approx(math.radians(37.0), abs=1e-12)


def test_identity_axis_angle_is_safe():
    axis, angle = q_to_axis_angle(IDENTITY)
    assert angle == 0.0
    np.testing.assert_allclose(axis, [1.0, 0.0, 0.0])


def test_slerp_endpoints_and_shortest_arc():
    left = IDENTITY
    right = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(90.0))
    np.testing.assert_allclose(q_slerp(left, right, 0.0), left, atol=1e-12)
    np.testing.assert_allclose(q_slerp(left, right, 1.0), right, atol=1e-12)
    midpoint = q_slerp(left, right, 0.5)
    assert q_angle_between(left, midpoint) == pytest.approx(math.radians(45.0), abs=1e-7)
    # -right represents the same orientation and must not take the long way.
    opposite = q_slerp(left, -right, 0.5)
    np.testing.assert_allclose(opposite, midpoint, atol=1e-12)


def test_two_vector_alignment():
    q = q_from_two_vectors([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(q_rotate(q, [0.0, 0.0, 1.0]), [0.0, 1.0, 0.0], atol=1e-12)
    opposite = q_from_two_vectors([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(q_rotate(opposite, [1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-12)


def test_invalid_quaternions_are_rejected():
    with pytest.raises(ValueError):
        q_norm([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        q_norm([1.0, 0.0, np.nan, 0.0])
    with pytest.raises(ValueError):
        q_norm([1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        q_from_axis_angle([0.0, 0.0, 0.0], 1.0)
