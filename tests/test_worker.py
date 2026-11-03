"""Tests for asynchronous perception and latest-only mailboxes."""

from __future__ import annotations

import time

import numpy as np

from shadowlink.perception.selector import DetectionSelector
from shadowlink.perception.worker import LatestSlot, LatestValue, PerceptionService
from shadowlink.types import FramePacket, HandDetection, HandObservation, Side


def frame(seq: int) -> FramePacket:
    return FramePacket(
        seq=seq,
        capture_ts=time.monotonic(),
        image_bgr=np.zeros((16, 16, 3), dtype=np.uint8),
    )


def detection(seq: int, side: Side = Side.RIGHT) -> HandDetection:
    return HandDetection(
        bbox_xyxy=np.array([2.0, 2.0, 12.0, 12.0]),
        apparent_side=side,
        anatomical_side=side,
        frame_seq=seq,
    )


class FakeDetector:
    name = "fake"

    def __init__(self, result=None, *, delay: float = 0.0, error: Exception | None = None):
        self.result = result if result is not None else (lambda seq: [])
        self.delay = delay
        self.error = error
        self.calls = 0
        self.closed = False

    def detect(self, packet: FramePacket):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.result(packet.seq)

    def close(self) -> None:
        self.closed = True


class FakeBackend:
    name = "fake"
    requires_detections = True
    supports_mesh = False

    def __init__(self, *, delay: float = 0.0, error: Exception | None = None):
        self.delay = delay
        self.error = error
        self.calls = 0
        self.closed = False

    def estimate(self, packet: FramePacket, detections):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        now = time.monotonic()
        return [
            HandObservation(
                keypoints_3d=np.zeros((21, 3)),
                apparent_side=item.apparent_side,
                anatomical_side=item.anatomical_side,
                score=item.score,
                frame_seq=packet.seq,
                capture_ts=packet.capture_ts,
                inference_ts=now,
                backend=self.name,
                bbox_xyxy=item.bbox_xyxy,
            )
            for item in detections
        ]

    def close(self) -> None:
        self.closed = True


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def service(detector=None, backend=None, **kwargs) -> PerceptionService:
    return PerceptionService(
        detector or FakeDetector(),
        backend or FakeBackend(),
        DetectionSelector("largest"),
        **kwargs,
    )


def test_latest_slot_keeps_only_the_newest_item():
    slot: LatestSlot[int] = LatestSlot()
    assert slot.put(1) is False
    assert slot.put(2) is True
    assert slot.take(timeout=0.0) == 2
    assert slot.take(timeout=0.0) is None


def test_latest_value_can_be_read_repeatedly():
    box: LatestValue[int] = LatestValue()
    assert box.get() is None
    box.put(7)
    assert box.get() == 7
    assert box.get() == 7


def test_two_stage_result_matches_the_frame_sequence():
    backend = FakeBackend()
    perception = service(
        detector=FakeDetector(lambda seq: [detection(seq)]),
        backend=backend,
    )
    perception.start()
    try:
        assert perception.submit(frame(0))
        assert wait_for(lambda: perception.latest_result() is not None)
        result = perception.latest_result()
        assert result is not None
        assert result.observations[0].frame_seq == result.frame_seq
        assert backend.calls >= 1
    finally:
        perception.close()
    assert backend.closed


def test_slow_detector_processes_only_the_latest_frame():
    detector = FakeDetector(lambda seq: [detection(seq)], delay=0.08)
    perception = service(
        detector=detector,
        backend=FakeBackend(),
        pipeline="two_stage",
    )
    perception.start()
    try:
        for seq in range(5):
            perception.submit(frame(seq))
        assert wait_for(
            lambda: (result := perception.latest_result()) is not None and result.frame_seq == 4
        )
        assert perception.health.dropped >= 1
    finally:
        perception.close()


def test_no_detections_skips_the_backend():
    class ExplodingBackend(FakeBackend):
        def estimate(self, packet, detections):
            raise AssertionError("backend must not run without detections")

    perception = service(detector=FakeDetector(lambda seq: []), backend=ExplodingBackend())
    perception.start()
    try:
        perception.submit(frame(0))
        assert wait_for(
            lambda: (result := perception.latest_result()) is not None and result.frame_seq == 0
        )
        result = perception.latest_result()
        assert result.observations == ()
    finally:
        perception.close()


def test_side_filter_is_anatomical():
    detector = FakeDetector(lambda seq: [detection(seq, Side.LEFT)])
    perception = service(detector=detector, side="right")
    perception.start()
    try:
        perception.submit(frame(0))
        assert wait_for(lambda: perception.latest_result() is not None)
        assert perception.latest_result().observations == ()
    finally:
        perception.close()


def test_detector_failure_is_reported_and_stops_the_worker():
    perception = service(
        detector=FakeDetector(error=RuntimeError("boom")),
        backend=FakeBackend(),
    )
    perception.start()
    try:
        perception.submit(frame(0))
        assert wait_for(lambda: perception.health.failed)
        assert "boom" in perception.health.message
        assert "detector" in perception.health.message
    finally:
        perception.close()


def test_backend_failure_is_reported():
    perception = service(
        detector=FakeDetector(lambda seq: [detection(seq)]),
        backend=FakeBackend(error=RuntimeError("kaboom")),
    )
    perception.start()
    try:
        perception.submit(frame(0))
        assert wait_for(lambda: perception.health.failed)
        assert "kaboom" in perception.health.message
        assert "backend" in perception.health.message
    finally:
        perception.close()


def test_close_is_idempotent_and_marks_stopped():
    detector = FakeDetector()
    perception = service(detector=detector)
    perception.start()
    perception.close()
    perception.close()
    assert perception.health.state == "stopped"
    assert detector.closed
    assert perception.submit(frame(0)) is False


def test_single_pipeline_runs_both_stages_in_one_thread():
    backend = FakeBackend()
    perception = service(
        detector=FakeDetector(lambda seq: [detection(seq)]),
        backend=backend,
        pipeline="single",
    )
    perception.start()
    try:
        perception.submit(frame(0))
        assert wait_for(lambda: perception.latest_result() is not None)
        assert backend.calls == 1
    finally:
        perception.close()
