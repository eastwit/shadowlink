"""Tests for using the tag camera as the hand/frame source."""

from __future__ import annotations

import numpy as np

from shadowlink.tag import TagHealth
from shadowlink.tag.source import TagFrameSource


class FakeService:
    def __init__(
        self,
        frame: np.ndarray,
        *,
        startup_misses: int = 0,
        state: str = "running",
    ) -> None:
        self.frame = frame
        self.startup_misses = startup_misses
        self.state = state
        self.calls: list[int] = []
        self.closed = False

    def wait_frame(self, after_seq: int, timeout: float = 0.75):
        self.calls.append(after_seq)
        if len(self.calls) <= self.startup_misses:
            return None
        return after_seq + 1, 12.5, self.frame

    @property
    def health(self) -> TagHealth:
        return TagHealth(state=self.state, rate_hz=29.0)

    def latest_preview(self):
        return self.frame

    def close(self) -> None:
        self.closed = True


def test_tag_frame_source_wraps_raw_tag_frames():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    service = FakeService(frame)
    source = TagFrameSource(service)
    packet = source.read()
    assert packet is not None
    assert packet.seq == 0
    assert packet.capture_ts == 12.5
    assert packet.image_bgr.shape == (48, 64, 3)
    assert packet.mirrored is False
    assert service.calls == [-1]
    assert source.fps == 29.0

    packet = source.read()
    assert packet is not None and packet.seq == 1
    assert service.calls == [-1, 0]

    source.close()
    assert service.closed is False


def test_source_waits_through_camera_startup():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    service = FakeService(frame, startup_misses=2, state="starting")
    source = TagFrameSource(service)
    packet = source.read()
    assert packet is not None
    assert len(service.calls) == 3
    assert packet.seq == 0


def test_source_stops_when_the_camera_service_fails():
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    service = FakeService(frame, startup_misses=100, state="error")
    source = TagFrameSource(service)
    assert source.read() is None
    assert len(service.calls) == 1
