"""Tests for using the tag camera as the hand/frame source."""

from __future__ import annotations

import numpy as np

from shadowlink.tag import TagHealth
from shadowlink.tag.source import TagFrameSource


class FakeService:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.calls: list[int] = []
        self.closed = False

    def wait_frame(self, after_seq: int, timeout: float = 0.75):
        self.calls.append(after_seq)
        return after_seq + 1, 12.5, self.frame

    @property
    def health(self) -> TagHealth:
        return TagHealth(state="running", rate_hz=29.0)

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
