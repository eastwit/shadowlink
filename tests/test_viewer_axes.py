"""Tests for the hand-base zero marker geometry (no window needed)."""

from __future__ import annotations

import math

import numpy as np

from shadowlink.math3d import q_from_axis_angle
from shadowlink.sim.viewer import (
    ZERO_AXIS_COLORS,
    ZERO_AXIS_LENGTH,
    ZERO_AXIS_OFFSET,
    AxisSegment,
    zero_axis_segments,
)


def test_identity_zero_puts_the_rgb_triad_on_the_hand_axes():
    segments = zero_axis_segments([1.0, 0.0, 0.0, 0.0], [0.2, -0.1, 0.3])
    base = np.array([0.2, -0.1, 0.3]) + ZERO_AXIS_OFFSET
    assert len(segments) == 3
    for axis, segment in enumerate(segments):
        assert isinstance(segment, AxisSegment)
        assert segment.color == ZERO_AXIS_COLORS[axis]
        np.testing.assert_allclose(segment.start, base)
        expected = base.copy()
        expected[axis] += ZERO_AXIS_LENGTH
        np.testing.assert_allclose(segment.end, expected)


def test_zero_triad_follows_the_hand_zero_orientation():
    quat = q_from_axis_angle([0.0, 0.0, 1.0], math.radians(90.0))
    segments = zero_axis_segments(quat, np.zeros(3))
    base = np.array([0.0, ZERO_AXIS_OFFSET[0], 0.0])
    np.testing.assert_allclose(segments[0].start, base, atol=1e-12)
    np.testing.assert_allclose(segments[0].end, base + [0.0, ZERO_AXIS_LENGTH, 0.0], atol=1e-12)
    np.testing.assert_allclose(segments[1].end, base + [-ZERO_AXIS_LENGTH, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(segments[2].end, base + [0.0, 0.0, ZERO_AXIS_LENGTH], atol=1e-12)


def test_zero_triad_normalizes_the_quaternion_and_does_not_mutate_the_origin():
    origin = np.array([0.1, 0.2, 0.3])
    segments = zero_axis_segments([2.0, 0.0, 0.0, 0.0], origin)
    base = origin + ZERO_AXIS_OFFSET
    np.testing.assert_allclose(segments[0].start, base)
    np.testing.assert_allclose(origin, [0.1, 0.2, 0.3])
