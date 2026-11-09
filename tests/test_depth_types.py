"""Validation tests for the depth and detector-landmark data types."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.types import DepthIntrinsics, FramePacket, HandObservation, Side


def intrinsics(**overrides) -> DepthIntrinsics:
    values = dict(fx=500.0, fy=500.0, cx=31.5, cy=23.5, width=64, height=48)
    values.update(overrides)
    return DepthIntrinsics(**values)


def packet(**depth) -> FramePacket:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    return FramePacket(seq=0, capture_ts=1.0, image_bgr=image, **depth)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fx": 0.0},
        {"fy": -1.0},
        {"fx": float("nan")},
        {"cx": -1.0},
        {"cy": 48.0},
        {"width": 0},
        {"height": -3},
    ],
)
def test_depth_intrinsics_reject_invalid_values(overrides):
    with pytest.raises(ValueError):
        intrinsics(**overrides)


def test_depth_intrinsics_accepts_the_principal_point_inside_the_image():
    k = intrinsics(cx=63.0, cy=47.0)
    assert (k.cx, k.cy) == (63.0, 47.0)


def test_depth_fields_must_be_provided_together():
    depth = np.full((48, 64), 800, dtype=np.uint16)
    with pytest.raises(ValueError):
        packet(depth_mm=depth)
    with pytest.raises(ValueError):
        packet(depth_intrinsics=intrinsics())
    with pytest.raises(ValueError):
        packet(depth_ts=1.0)
    with pytest.raises(ValueError):
        packet(depth_mm=depth, depth_intrinsics=intrinsics())


def test_depth_image_must_match_the_intrinsics():
    depth = np.full((48, 64), 800, dtype=np.uint16)
    with pytest.raises(ValueError):
        packet(depth_mm=depth, depth_intrinsics=intrinsics(width=32), depth_ts=1.0)


@pytest.mark.parametrize("depth", [np.full((48, 64), 800, dtype=np.int32), np.ones((48, 64, 1))])
def test_depth_image_must_be_a_2d_uint16_array(depth):
    kwargs = dict(depth_mm=depth, depth_intrinsics=intrinsics(), depth_ts=1.0)
    with pytest.raises((TypeError, ValueError)):
        packet(**kwargs)


def test_depth_arrays_are_marked_read_only():
    depth = np.full((48, 64), 800, dtype=np.uint16)
    frame = packet(depth_mm=depth, depth_intrinsics=intrinsics(), depth_ts=2.0)
    assert frame.depth_mm.flags.writeable is False
    with pytest.raises(ValueError):
        frame.depth_mm[0, 0] = 1


def test_depth_timestamp_must_be_finite():
    depth = np.full((48, 64), 800, dtype=np.uint16)
    with pytest.raises(ValueError):
        packet(depth_mm=depth, depth_intrinsics=intrinsics(), depth_ts=float("inf"))


def observation(**overrides) -> HandObservation:
    values = dict(
        keypoints_3d=np.zeros((21, 3)),
        apparent_side=Side.RIGHT,
        anatomical_side=Side.RIGHT,
        score=1.0,
        frame_seq=0,
        capture_ts=1.0,
        inference_ts=1.1,
        backend="fake",
    )
    values.update(overrides)
    return HandObservation(**values)


def test_detector_landmarks_are_validated_and_read_only():
    points = np.zeros((21, 2), dtype=np.float64)
    obs = observation(detector_landmarks_2d=points)
    assert obs.detector_landmarks_2d.flags.writeable is False
    with pytest.raises(ValueError):
        observation(detector_landmarks_2d=np.zeros((20, 2)))
    with pytest.raises(ValueError):
        observation(detector_landmarks_2d=np.zeros((21, 3)))


def test_observations_without_detector_landmarks_are_still_valid():
    obs = observation()
    assert obs.detector_landmarks_2d is None
