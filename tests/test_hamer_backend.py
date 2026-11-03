"""Tests for the HaMeR backend (the model-dependent test is opt-in)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from shadowlink.perception.base import ModelUnavailableError, PerceptionError
from shadowlink.perception.hamer_backend import HaMeRBackend, _unique_edges


def test_unique_edges_are_deduplicated_and_sorted():
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    edges = _unique_edges(faces)
    assert edges.dtype == np.int32
    assert {(int(a), int(b)) for a, b in edges} == {
        (0, 1),
        (1, 2),
        (0, 2),
        (1, 3),
        (2, 3),
    }
    assert (edges[:, 0] < edges[:, 1]).all()


def test_unique_edges_rejects_non_triangles():
    with pytest.raises(PerceptionError):
        _unique_edges(np.zeros((4, 4), dtype=np.int64))


def test_missing_files_fail_before_importing_torch(tmp_path):
    missing = tmp_path / "missing.ckpt"
    with pytest.raises(ModelUnavailableError):
        HaMeRBackend(missing, missing.with_name("config.yaml"), missing.with_name("mano.pkl"))


@pytest.mark.hamer
def test_real_checkpoint_reconstructs_a_hand():
    """Manual integration test; set SHADOWLINK_TEST_DATA and SHADOWLINK_TEST_IMAGE."""
    from shadowlink.paths import ModelPaths

    data_dir = os.environ.get("SHADOWLINK_TEST_DATA")
    image_path = os.environ.get("SHADOWLINK_TEST_IMAGE")
    if not data_dir or not image_path:
        pytest.skip("set SHADOWLINK_TEST_DATA and SHADOWLINK_TEST_IMAGE to run")

    models = ModelPaths.resolve(data_dir)
    missing = models.missing(need_hamer=True)
    if missing:
        pytest.skip(f"model files missing: {missing}")
    if not Path(image_path).exists():
        pytest.skip(f"test image not found: {image_path}")

    import cv2

    from shadowlink.perception.hamer_backend import HaMeRBackend
    from shadowlink.perception.mediapipe_detector import MediaPipeDetector
    from shadowlink.types import FramePacket

    image = cv2.imread(image_path)
    frame = FramePacket(seq=0, capture_ts=time.monotonic(), image_bgr=image)
    detector = MediaPipeDetector(models.hand_landmarker)
    backend = HaMeRBackend(
        models.checkpoint,
        models.model_config,
        models.mano,
        data_dir=models.data_dir,
        fp16=True,
    )
    try:
        detections = detector.detect(frame)
        assert detections, "the test image should contain a hand"
        observations = backend.estimate(frame, detections[:1])
    finally:
        backend.close()
        detector.close()

    assert len(observations) == 1
    observation = observations[0]
    assert observation.backend == "hamer"
    assert observation.keypoints_3d.shape == (21, 3)
    assert np.isfinite(observation.keypoints_3d).all()
    assert observation.keypoints_2d is not None
    assert observation.keypoints_2d.shape == (21, 2)
    assert np.isfinite(observation.keypoints_2d).all()
    assert observation.vertices is not None
    assert observation.vertices.shape == (778, 3)
    assert observation.mesh_edges is not None
    assert observation.mesh_edges.shape[1] == 2
    assert observation.mesh_edges.min() >= 0
    assert observation.mesh_edges.max() < observation.vertices.shape[0]
    assert observation.camera_translation is not None
    assert observation.camera_translation.shape == (3,)
    assert observation.focal_length and observation.focal_length > 0
    # The projected joints should land near the detector box rather than at the
    # image origin.
    x0, y0, x1, y1 = observation.bbox_xyxy
    inside = (
        (observation.keypoints_2d[:, 0] >= x0 - 0.5 * (x1 - x0))
        & (observation.keypoints_2d[:, 0] <= x1 + 0.5 * (x1 - x0))
        & (observation.keypoints_2d[:, 1] >= y0 - 0.5 * (y1 - y0))
        & (observation.keypoints_2d[:, 1] <= y1 + 0.5 * (y1 - y0))
    )
    assert inside.mean() > 0.8
