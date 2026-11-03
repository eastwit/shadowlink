"""Tests for the NumPy camera math used by the HaMeR backend."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.perception.hamer_math import (
    cam_crop_to_full,
    project_points,
    scaled_focal_length,
)


def test_cam_crop_to_full_matches_the_weak_perspective_formula():
    full = cam_crop_to_full(
        pred_cam=np.array([[0.5, 0.1, -0.2]]),
        box_center=np.array([[100.0, 80.0]]),
        box_size=np.array([200.0]),
        img_size=np.array([[640.0, 480.0]]),
        focal_length=1000.0,
    )
    # bs = 100, tz = 20, tx = 2*(100-320)/100 + 0.1, ty = 2*(80-240)/100 - 0.2
    np.testing.assert_allclose(full, [[-4.3, -3.4, 20.0]])


def test_cam_crop_to_full_is_batched():
    full = cam_crop_to_full(
        pred_cam=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        box_center=np.array([[320.0, 240.0], [320.0, 240.0]]),
        box_size=np.array([100.0, 100.0]),
        img_size=np.array([[640.0, 480.0], [640.0, 480.0]]),
        focal_length=np.array([1000.0, 2000.0]),
    )
    assert full.shape == (2, 3)
    np.testing.assert_allclose(full[:, 0], 0.0)
    np.testing.assert_allclose(full[:, 1], 0.0)
    np.testing.assert_allclose(full[:, 2], [20.0, 40.0])


def test_cam_crop_to_full_validates_shapes():
    with pytest.raises(ValueError):
        cam_crop_to_full(
            np.zeros((2, 3)),
            np.zeros((1, 2)),
            np.zeros(2),
            np.zeros((2, 2)),
        )
    with pytest.raises(ValueError):
        cam_crop_to_full(
            np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)), [1.0, 2.0, 3.0]
        )


def test_project_points_center_and_offset():
    center = project_points(np.array([[0.0, 0.0, 0.0]]), [0, 0, 1], 500.0, 640, 480)
    np.testing.assert_allclose(center, [[320.0, 240.0]], atol=1e-9)

    offset = project_points(np.array([[0.1, 0.0, 0.0]]), [0, 0, 1], 500.0, 640, 480)
    np.testing.assert_allclose(offset, [[370.0, 240.0]], atol=1e-9)


def test_project_points_handles_depth_near_zero():
    points = project_points(np.zeros((21, 3)), [0, 0, 0], 500.0, 640, 480)
    assert np.isfinite(points).all()


def test_project_points_validates_inputs():
    with pytest.raises(ValueError):
        project_points(np.zeros((20, 2)), [0, 0, 1], 500.0, 640, 480)
    with pytest.raises(ValueError):
        project_points(np.zeros((21, 3)), [0, 0, 1], 0.0, 640, 480)


def test_scaled_focal_length_uses_the_longest_side():
    assert scaled_focal_length(5000.0, 256, np.array([640, 480])) == pytest.approx(12500.0)
    assert scaled_focal_length(5000.0, 256, np.array([480, 640])) == pytest.approx(12500.0)
    with pytest.raises(ValueError):
        scaled_focal_length(5000.0, 256, np.array([640]))
