"""Tests for hand-relative camera-to-world direction mapping."""

from __future__ import annotations

import math

import numpy as np

from shadowlink.position.anatomy import camera_to_local_axes, hand_base_rotation
from shadowlink.types import Side


def test_right_hand_maps_palmar_thumb_and_distal(make_hand):
    axes = camera_to_local_axes(make_hand(), Side.RIGHT, Side.RIGHT)
    assert axes is not None

    # The synthetic hand has its palm normal along +z, thumb toward -y, and
    # fingers toward +x.
    np.testing.assert_allclose(axes @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(axes @ [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(axes @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-9)
    assert axes.flags.writeable is False


def test_left_hand_uses_the_same_anatomical_components(make_hand):
    right = make_hand()
    left = right.copy()
    left[:, 0] *= -1.0

    axes = camera_to_local_axes(left, Side.LEFT, Side.LEFT)
    assert axes is not None

    # The raw left hand points along -x and its thumb is on the -y side.
    np.testing.assert_allclose(axes @ [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(axes @ [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(axes @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-9)


def test_a_mirrored_processed_image_is_canonicalised_the_same_way(make_hand):
    # A right hand seen in a mirrored frame is labelled left and its x is
    # mirrored; canonicalize() must recover the raw camera-frame mapping.
    right = make_hand()
    mirrored = right.copy()
    mirrored[:, 0] *= -1.0
    axes = camera_to_local_axes(mirrored, Side.LEFT, Side.RIGHT)
    assert axes is not None
    # Displacements are measured on the raw depth image, where the right hand's
    # thumb side is still -y and its palm normal is still +z.
    np.testing.assert_allclose(axes @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(axes @ [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], atol=1e-9)


def test_degenerate_or_malformed_keypoints_return_none(make_hand):
    assert camera_to_local_axes(np.zeros((21, 3)), Side.RIGHT, Side.RIGHT) is None
    assert camera_to_local_axes(np.full((21, 3), np.nan), Side.RIGHT, Side.RIGHT) is None
    assert camera_to_local_axes(np.zeros((20, 3)), Side.RIGHT, Side.RIGHT) is None
    flat = make_hand()
    flat[:, 2] = 0.0
    flat[9] = flat[0]
    assert camera_to_local_axes(flat, Side.RIGHT, Side.RIGHT) is None


def test_hand_base_rotation_uses_the_shadow_hand_rest_frame():
    # The MJCF rest orientation maps local +x to world +z, +y to -y, +z to +x.
    quat = np.array([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    rotation = hand_base_rotation(quat)
    np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(rotation[:, 1], [0.0, -1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(rotation[:, 2], [1.0, 0.0, 0.0], atol=1e-9)
