"""Tests for palm sampling from a depth image and colour landmarks."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.config import PositionConfig
from shadowlink.position.depth_sampling import (
    PALM_LANDMARKS,
    PalmMeasurement,
    sample_palm_position,
)
from shadowlink.types import DepthIntrinsics

WIDTH, HEIGHT = 640, 480
K = DepthIntrinsics(fx=500.0, fy=500.0, cx=319.5, cy=239.5, width=WIDTH, height=HEIGHT)
CFG = PositionConfig()


def make_landmarks(pixels, width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    points = np.full((21, 2), (width / 2.0, height / 2.0), dtype=np.float64)
    for index, (u, v) in zip(PALM_LANDMARKS, pixels, strict=True):
        points[index] = (u, v)
    return points


def expected_xyz(
    u_px: float,
    v_px: float,
    *,
    depth_mm: float,
    mirrored: bool = False,
    color_size: tuple[int, int] = (WIDTH, HEIGHT),
    align=(0.0, 0.0),
    intrinsics: DepthIntrinsics = K,
) -> np.ndarray:
    color_width, color_height = color_size
    u_norm = u_px / color_width
    v_norm = v_px / color_height
    if mirrored:
        u_norm = 1.0 - u_norm
    u_norm += align[0]
    v_norm += align[1]
    u = round(u_norm * (intrinsics.width - 1))
    v = round(v_norm * (intrinsics.height - 1))
    z = depth_mm / 1000.0
    return np.array(
        [(u - intrinsics.cx) * z / intrinsics.fx, (v - intrinsics.cy) * z / intrinsics.fy, z]
    )


def sample(depth, landmarks, **overrides):
    kwargs = dict(
        mirrored=False,
        color_size=(WIDTH, HEIGHT),
        cfg=CFG,
        frame_seq=7,
        capture_ts=10.0,
        depth_ts=9.9,
    )
    kwargs.update(overrides)
    return sample_palm_position(depth, K, landmarks, **kwargs)


def test_flat_plane_back_projects_to_the_measured_depth():
    depth = np.full((HEIGHT, WIDTH), 800, dtype=np.uint16)
    pixels = [(300, 220), (340, 220), (300, 260), (340, 260), (320, 240)]
    result = sample(depth, make_landmarks(pixels))

    assert result is not None
    assert result.valid_landmarks == 5
    assert result.frame_seq == 7
    assert result.capture_ts == 10.0
    assert result.depth_ts == 9.9
    expected = np.median([expected_xyz(u, v, depth_mm=800) for u, v in pixels], axis=0)
    np.testing.assert_allclose(result.xyz_m, expected, atol=1e-12)
    assert result.xyz_m[2] == pytest.approx(0.8)


def test_mirroring_flips_the_horizontal_axis():
    depth = np.full((HEIGHT, WIDTH), 800, dtype=np.uint16)
    landmarks = make_landmarks([(420, 240)] * 5)
    plain = sample(depth, landmarks)
    mirrored = sample(depth, landmarks, mirrored=True)

    assert plain is not None and mirrored is not None
    assert plain.xyz_m[0] > 0.0
    assert mirrored.xyz_m[0] < 0.0
    np.testing.assert_allclose(plain.xyz_m, expected_xyz(420, 240, depth_mm=800), atol=1e-12)
    np.testing.assert_allclose(
        mirrored.xyz_m, expected_xyz(420, 240, depth_mm=800, mirrored=True), atol=1e-12
    )


def test_colour_and_depth_resolutions_may_differ():
    depth = np.full((HEIGHT, WIDTH), 800, dtype=np.uint16)
    small = sample(depth, make_landmarks([(320, 240)] * 5))
    large = sample(
        depth,
        make_landmarks([(640, 360)] * 5, width=1280, height=720),
        color_size=(1280, 720),
    )
    assert small is not None and large is not None
    np.testing.assert_allclose(large.xyz_m, small.xyz_m, atol=1e-12)


def test_depth_holes_reject_the_sample():
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    assert sample(depth, make_landmarks([(320, 240)] * 5)) is None


@pytest.mark.parametrize("value", [100, 2000])
def test_depth_outside_the_accepted_range_is_rejected(value):
    depth = np.full((HEIGHT, WIDTH), value, dtype=np.uint16)
    assert sample(depth, make_landmarks([(320, 240)] * 5)) is None


def test_a_spread_outlier_rejects_the_whole_frame():
    depth = np.full((HEIGHT, WIDTH), 800, dtype=np.uint16)
    depth[:, WIDTH // 2 :] = 1400
    pixels = [(200, 240), (220, 240), (240, 240), (500, 240), (520, 240)]
    assert sample(depth, make_landmarks(pixels)) is None


def test_alignment_offsets_move_the_sample_in_the_depth_image():
    depth = np.full((HEIGHT, WIDTH), 500, dtype=np.uint16)
    depth[:, WIDTH // 2 + 5 :] = 1200
    landmarks = make_landmarks([(320, 240)] * 5)

    plain = sample(depth, landmarks)
    shifted = sample(
        depth,
        landmarks,
        cfg=PositionConfig(align_offset_x=0.4),
    )

    assert plain is not None and plain.xyz_m[2] == pytest.approx(0.5)
    assert shifted is not None and shifted.xyz_m[2] == pytest.approx(1.2)


def test_too_few_valid_landmarks_returns_none():
    depth = np.full((HEIGHT, WIDTH), 800, dtype=np.uint16)
    pixels = [(-100, 240), (-100, 240), (-100, 240), (320, 240), (320, 240)]
    assert sample(depth, make_landmarks(pixels)) is None


def test_malformed_inputs_return_none():
    depth = np.full((HEIGHT, WIDTH), 800, dtype=np.uint16)
    assert sample(depth, np.zeros((20, 2))) is None
    assert sample(depth[0], make_landmarks([(320, 240)] * 5)) is None
    assert sample(depth, make_landmarks([(320, 240)] * 5), color_size=(0, 0)) is None


def test_palm_measurement_validates_and_freezes_its_position():
    xyz = np.array([0.0, 0.0, 1.0])
    measurement = PalmMeasurement(
        xyz_m=xyz,
        frame_seq=1,
        capture_ts=1.0,
        depth_ts=0.9,
        valid_landmarks=5,
        spread_m=0.0,
    )
    assert measurement.xyz_m.flags.writeable is False
    with pytest.raises(ValueError):
        PalmMeasurement(
            xyz_m=[np.nan, 0.0, 0.0],
            frame_seq=1,
            capture_ts=1.0,
            depth_ts=0.9,
            valid_landmarks=5,
            spread_m=0.0,
        )
