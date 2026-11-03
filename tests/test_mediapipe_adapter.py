"""Tests for the MediaPipe adapter helpers and CPU keypoint backend."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from shadowlink.perception.base import ModelUnavailableError
from shadowlink.perception.mediapipe_backend import MediaPipeKeypointBackend
from shadowlink.perception.mediapipe_detector import (
    MediaPipeDetector,
    _clipped_bbox,
    anatomical_side,
    next_timestamp_ms,
)
from shadowlink.types import FramePacket, HandDetection, Side


def make_frame(seq: int = 0) -> FramePacket:
    return FramePacket(
        seq=seq,
        capture_ts=time.monotonic(),
        image_bgr=np.zeros((100, 120, 3), dtype=np.uint8),
    )


def world_landmarks() -> np.ndarray:
    points = np.linspace(-0.1, 0.1, 63, dtype=np.float64).reshape(21, 3)
    return points


def make_detection(**overrides) -> HandDetection:
    values = {
        "bbox_xyxy": np.array([10.0, 20.0, 60.0, 80.0]),
        "apparent_side": Side.RIGHT,
        "anatomical_side": Side.RIGHT,
        "world_landmarks": world_landmarks(),
        "landmarks_2d": np.zeros((21, 2)),
        "frame_seq": 0,
    }
    values.update(overrides)
    return HandDetection(**values)


def test_anatomical_side_flips_only_for_mirrored_frames():
    assert anatomical_side(Side.RIGHT, mirrored=False) is Side.RIGHT
    assert anatomical_side(Side.LEFT, mirrored=False) is Side.LEFT
    assert anatomical_side(Side.RIGHT, mirrored=True) is Side.LEFT
    assert anatomical_side(Side.LEFT, mirrored=True) is Side.RIGHT


def test_mediapipe_label_maps_to_apparent_side():
    assert Side.from_mediapipe_label("Left") is Side.LEFT
    assert Side.from_mediapipe_label("right") is Side.RIGHT
    with pytest.raises(ValueError):
        Side.from_mediapipe_label("unknown")


def test_bbox_is_clipped_to_the_image():
    points = np.array([[-20.0, -5.0], [140.0, 30.0], [50.0, 200.0]] + [[60.0, 60.0]] * 18)
    assert _clipped_bbox(points, width=120, height=100) == pytest.approx([0, 0, 119, 99])


def test_degenerate_bbox_is_rejected():
    points = np.full((21, 2), 5.0)
    assert _clipped_bbox(points, width=120, height=100) is None


def test_backend_promotes_world_landmarks_to_an_observation():
    detection = make_detection()
    observation = MediaPipeKeypointBackend().estimate(make_frame(3), [detection])[0]
    assert observation.backend == "mediapipe"
    assert observation.frame_seq == 3
    np.testing.assert_array_equal(observation.keypoints_3d, detection.world_landmarks)
    np.testing.assert_array_equal(observation.keypoints_2d, detection.landmarks_2d)
    np.testing.assert_array_equal(observation.bbox_xyxy, detection.bbox_xyxy)
    assert observation.anatomical_side is Side.RIGHT
    assert not observation.keypoints_3d.flags.writeable


def test_backend_skips_detections_without_world_landmarks():
    detection = make_detection(world_landmarks=None)
    assert MediaPipeKeypointBackend().estimate(make_frame(), [detection]) == []
    assert MediaPipeKeypointBackend().estimate(make_frame(), []) == []


def test_missing_task_file_raises_model_unavailable(tmp_path: Path):
    with pytest.raises(ModelUnavailableError):
        MediaPipeDetector(tmp_path / "missing.task")


def test_video_timestamps_are_strictly_increasing():
    assert next_timestamp_ms(1.234, -1) == 1234
    assert next_timestamp_ms(1.234, 1234) == 1235
    assert next_timestamp_ms(1.234, 2000) == 2001
    assert next_timestamp_ms(0.0, -1) == 0


def test_invalid_running_mode_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        MediaPipeDetector(tmp_path / "missing.task", running_mode="stream")


@pytest.mark.mediapipe
def test_real_detector_finds_a_hand_in_the_example_image():
    """Run only when a task model and test image are provided by the developer."""
    image_path = os.environ.get("SHADOWLINK_TEST_IMAGE")
    model_path = os.environ.get("SHADOWLINK_TEST_HAND_MODEL")
    if not image_path or not model_path:
        pytest.skip("set SHADOWLINK_TEST_IMAGE and SHADOWLINK_TEST_HAND_MODEL to run")
    if not Path(image_path).exists() or not Path(model_path).exists():
        pytest.skip("configured test image or hand model does not exist")

    import cv2

    image = cv2.imread(image_path)
    frame = FramePacket(seq=0, capture_ts=time.monotonic(), image_bgr=image)
    detector = MediaPipeDetector(model_path)
    try:
        detections = detector.detect(frame)
    finally:
        detector.close()

    assert detections
    height, width = image.shape[:2]
    for detection in detections:
        x0, y0, x1, y1 = detection.bbox_xyxy
        assert 0.0 <= x0 < x1 <= width - 1
        assert 0.0 <= y0 < y1 <= height - 1
        if detection.world_landmarks is not None:
            assert detection.world_landmarks.shape == (21, 3)
            assert np.isfinite(detection.world_landmarks).all()
