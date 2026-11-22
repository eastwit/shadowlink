"""Tests for the background AprilTag camera service without hardware."""

from __future__ import annotations

import time

import numpy as np

from shadowlink.config import TagConfig
from shadowlink.tag import TagService


class _Result:
    tag_id = 5
    hamming = 0
    decision_margin = 60.0
    pose_err = 0.01
    pose_t = np.array([[0.05], [-0.02], [0.7]], dtype=np.float64)
    pose_R = np.eye(3, dtype=np.float64)
    corners = np.array(
        [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
        dtype=np.float64,
    )


class _MirrorDetector:
    """Return a pose only after the image has been horizontally corrected."""

    def detect(self, gray, **kwargs):
        return [_Result()] if float(gray[0, 0]) > 200.0 else []


class _FakeCamera:
    def __init__(self):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, 32:] = 255
        self.frame = frame
        self.released = False

    def read(self):
        return True, self.frame.copy()

    def release(self):
        self.released = True


def test_service_publishes_pose_and_auto_flips_a_mirrored_camera():
    camera = _FakeCamera()
    cfg = TagConfig(camera="9", width=64, height=48, mirror="auto", tag_id=5)
    service = TagService(
        cfg,
        detector_factory=lambda: _MirrorDetector(),
        camera_factory=lambda: (camera, 64, 48),
    )
    service.start()
    try:
        sample = None
        for _ in range(100):
            sample = service.latest()
            if sample is not None:
                break
            time.sleep(0.01)
        assert sample is not None
        assert sample.tag_id == 5
        np.testing.assert_allclose(sample.translation_m, [0.05, -0.02, 0.7], atol=1e-12)
        health = service.health
        assert health.state == "running"
        assert health.mirror == "flipped"
        preview = service.latest_preview()
        assert preview is not None
        assert preview.shape == (48, 64, 3)
    finally:
        service.close()
    assert service.health.state == "stopped"
    assert camera.released


def test_prefers_the_highest_decision_margin_for_duplicate_ids():
    class Result:
        tag_id = 5
        hamming = 0
        pose_err = 0.01
        pose_R = np.eye(3, dtype=np.float64)
        corners = np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
            dtype=np.float64,
        )

        def __init__(self, z, margin):
            self.pose_t = np.array([[0.0], [0.0], [z]], dtype=np.float64)
            self.decision_margin = margin

    class Detector:
        def detect(self, gray, **kwargs):
            return [Result(0.5, 20.0), Result(1.1, 80.0)]

    service = TagService(
        TagConfig(camera="9", width=64, height=48, tag_id=5),
        detector_factory=lambda: Detector(),
        camera_factory=lambda: (_FakeCamera(), 64, 48),
    )
    service.start()
    try:
        sample = None
        for _ in range(100):
            sample = service.latest()
            if sample is not None:
                break
            time.sleep(0.01)
        assert sample is not None
        assert sample.translation_m[2] == 1.1
    finally:
        service.close()


def test_service_without_a_tag_keeps_searching():
    class _NoTagDetector:
        def detect(self, gray, **kwargs):
            return []

    cfg = TagConfig(camera="9", width=64, height=48, tag_id=7, mirror="off")
    service = TagService(
        cfg,
        detector_factory=lambda: _NoTagDetector(),
        camera_factory=lambda: (_FakeCamera(), 64, 48),
    )
    service.start()
    try:
        time.sleep(0.1)
        assert service.latest() is None
        assert service.health.state == "searching"
        assert service.health.mirror == "normal"
    finally:
        service.close()
