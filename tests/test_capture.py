"""Tests for frame sources and mirroring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shadowlink.capture import CaptureError, OpenCVFrameSource, SyntheticFrameSource
from shadowlink.config import CaptureConfig


def asymmetric_frame() -> np.ndarray:
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    frame[5, 7] = (0, 255, 0)
    return frame


def test_synthetic_source_mirrors_once_and_counts_frames():
    frame = asymmetric_frame()
    source = SyntheticFrameSource(frames=[frame], mirror=True)
    packet = source.read()
    assert packet is not None
    assert packet.seq == 0
    assert packet.mirrored is True
    np.testing.assert_array_equal(packet.image_bgr, cv2.flip(frame, 1))
    assert not packet.image_bgr.flags.writeable

    source.close()
    assert source.read() is None


def test_synthetic_source_stops_at_max_frames():
    source = SyntheticFrameSource(max_frames=3, width=8, height=6)
    assert [source.read().seq for _ in range(3)] == [0, 1, 2]
    assert source.read() is None


def test_missing_video_raises_capture_error(tmp_path):
    cfg = CaptureConfig(source="video", video=str(tmp_path / "missing.mp4"))
    with pytest.raises(CaptureError):
        OpenCVFrameSource(cfg)


def test_reads_a_real_video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (8, 6))
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for index in range(4):
        frame = np.full((6, 8, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    source = OpenCVFrameSource(CaptureConfig(source="video", video=str(path), mirror=False))
    try:
        packets = [source.read() for _ in range(4)]
    finally:
        source.close()
    assert all(packet is not None for packet in packets)
    assert [packet.seq for packet in packets] == [0, 1, 2, 3]
    assert packets[0].mirrored is False
    assert source.read() is None


def test_open_source_api_is_validated(tmp_path):
    cfg = CaptureConfig(source="video", video=str(tmp_path / "missing.mp4"), api="dshow")
    # On non-Windows builds the constant may be missing; either way it must not
    # silently open the wrong source.
    with pytest.raises(CaptureError):
        OpenCVFrameSource(cfg)
