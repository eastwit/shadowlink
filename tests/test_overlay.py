"""Tests for the OpenCV overlay helpers."""

from __future__ import annotations

import cv2
import numpy as np

from shadowlink.types import HandObservation, Side
from shadowlink.ui.overlay import (
    AMBER,
    draw_bbox,
    draw_hud,
    draw_mesh,
    draw_observation,
    draw_skeleton,
)


def blank() -> np.ndarray:
    return np.zeros((150, 220, 3), dtype=np.uint8)


def points_2d() -> np.ndarray:
    return np.array(
        [[60 + 8 * (index % 5), 40 + 7 * (index // 5)] for index in range(21)],
        dtype=np.float64,
    )


def observation(**overrides) -> HandObservation:
    values = {
        "keypoints_3d": np.zeros((21, 3)),
        "apparent_side": Side.RIGHT,
        "anatomical_side": Side.RIGHT,
        "score": 0.9,
        "frame_seq": 1,
        "capture_ts": 0.0,
        "inference_ts": 0.0,
        "backend": "test",
        "bbox_xyxy": np.array([40.0, 30.0, 140.0, 120.0]),
        "keypoints_2d": points_2d(),
    }
    values.update(overrides)
    return HandObservation(**values)


def assert_drew(image: np.ndarray) -> None:
    assert np.count_nonzero(image) > 0


def test_draw_skeleton_marks_pixels():
    image = blank()
    draw_skeleton(image, points_2d())
    assert_drew(image)


def test_draw_skeleton_ignores_bad_shapes():
    image = blank()
    draw_skeleton(image, np.zeros((20, 2)))
    draw_skeleton(image, None)
    assert np.count_nonzero(image) == 0


def test_draw_bbox_and_none():
    image = blank()
    draw_bbox(image, None, "label")
    assert np.count_nonzero(image) == 0
    draw_bbox(image, np.array([10, 10, 100, 100]), "hand")
    assert_drew(image)


def test_draw_observation_modes():
    image = blank()
    draw_observation(image, None)
    assert np.count_nonzero(image) == 0

    draw_observation(blank(), observation(), "none")
    skeleton = blank()
    draw_observation(skeleton, observation(), "skeleton")
    assert_drew(skeleton)

    # Mesh mode falls back to the 2D skeleton when no vertices are available.
    mesh = blank()
    draw_observation(mesh, observation(), "mesh")
    assert_drew(mesh)


def mesh_vertices() -> np.ndarray:
    # A small triangle around the camera axis; with focal 220 and z=1 it lands
    # near the centre of the 220x150 test image.
    return np.array(
        [[-0.3, -0.2, 0.1], [0.3, -0.2, 0.1], [0.0, 0.3, 0.1]],
        dtype=np.float32,
    )


def assert_amber_pixels(image: np.ndarray) -> None:
    assert np.any(np.all(image == np.array(AMBER, dtype=np.uint8), axis=-1))


def test_draw_mesh_draws_the_wireframe():
    image = blank()
    draw_mesh(
        image,
        mesh_vertices(),
        np.array([0.0, 0.0, 1.0]),
        220.0,
        np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        stride=1,
    )
    assert_amber_pixels(image)


def test_draw_mesh_rejects_missing_or_inconsistent_inputs():
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32)
    translation = np.array([0.0, 0.0, 1.0])

    image = blank()
    draw_mesh(image, None, translation, 220.0, edges)
    draw_mesh(image, mesh_vertices(), None, 220.0, edges)
    draw_mesh(image, mesh_vertices(), translation, 0.0, edges)
    draw_mesh(image, mesh_vertices(), translation, 220.0, None)
    draw_mesh(image, mesh_vertices(), translation, float("nan"), edges)
    draw_mesh(image, mesh_vertices(), np.array([0.0, np.nan, 1.0]), 220.0, edges)
    draw_mesh(image, mesh_vertices(), np.array([0.0, 0.0]), 220.0, edges)
    draw_mesh(image, mesh_vertices(), translation, 220.0, np.array([[0, 9]]))
    draw_mesh(image, mesh_vertices(), translation, 220.0, np.zeros((0, 2), dtype=np.int32))
    draw_mesh(image, mesh_vertices()[:, :2], translation, 220.0, edges)
    assert np.count_nonzero(image) == 0


def test_draw_observation_mesh_mode_uses_vertices():
    mesh = observation(
        vertices=mesh_vertices(),
        mesh_edges=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        camera_translation=np.array([0.0, 0.0, 1.0]),
        focal_length=220.0,
    )
    image = blank()
    draw_observation(image, mesh, "mesh")
    assert_amber_pixels(image)


def test_draw_observation_mesh_mode_skips_invalid_vertices():
    mesh = observation(
        vertices=mesh_vertices(),
        mesh_edges=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        camera_translation=np.array([0.0, 0.0, 1.0]),
        focal_length=None,
    )
    image = blank()
    draw_observation(image, mesh, "mesh")
    assert_drew(image)  # falls back to the skeleton
    assert not np.any(np.all(image == np.array(AMBER, dtype=np.uint8), axis=-1))


def test_draw_observation_without_2d_points_still_draws_the_box():
    image = blank()
    draw_observation(image, observation(keypoints_2d=None), "skeleton")
    assert_drew(image)


def test_draw_hud_renders_a_panel():
    image = blank()
    draw_hud(image, ["state tracking", "fps 30.0"])
    assert_drew(image)


def test_draw_hud_without_lines_is_a_no_op():
    image = blank()
    draw_hud(image, [])
    assert np.count_nonzero(image) == 0


def test_draw_hud_survives_small_images_and_offsets():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_hud(image, ["a very long line"], origin=(0, 0))
    assert image.shape == (10, 10, 3)
    assert image.dtype == np.uint8


def test_drawing_preserves_shape_and_dtype():
    image = blank()
    draw_observation(image, observation(), "skeleton")
    draw_hud(image, ["hello"])
    assert image.shape == (150, 220, 3)
    assert image.dtype == np.uint8
    assert cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) > 0
