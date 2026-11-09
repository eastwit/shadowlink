"""Opt-in hardware test for the OpenNI2 depth stream.

Run with::

    SHADOWLINK_TEST_RGBD=1 \\
    SHADOWLINK_TEST_DEPTH_SDK=<dir containing OpenNI2.dll> \\
    pytest -m rgbd
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from shadowlink.config import CaptureConfig
from shadowlink.rgbd.source import RGBDFrameSource
from shadowlink.types import FramePacket

pytestmark = pytest.mark.rgbd


class _SilentColor:
    """Colour stand-in: the hardware test exercises depth only."""

    def __init__(self) -> None:
        self.seq = 0
        self.closed = False

    def read(self) -> FramePacket | None:
        packet = FramePacket(
            seq=self.seq,
            capture_ts=time.monotonic(),
            image_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        self.seq += 1
        return packet

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def size(self) -> tuple[int, int]:
        return (640, 480)

    def close(self) -> None:
        self.closed = True


def test_real_depth_stream_produces_labelled_frames():
    if os.environ.get("SHADOWLINK_TEST_RGBD") != "1":
        pytest.skip("set SHADOWLINK_TEST_RGBD=1 to use the depth camera")
    sdk = os.environ.get("SHADOWLINK_TEST_DEPTH_SDK")
    cfg = CaptureConfig(source="rgbd", depth_sdk=sdk)
    source = RGBDFrameSource(cfg, color_source=_SilentColor())
    frame = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            packet = source.read()
            if packet is not None and packet.depth_mm is not None:
                frame = packet
                break
            time.sleep(0.05)
        assert frame is not None, f"no depth frame: {source.depth_health}"
        assert frame.depth_mm.dtype == np.uint16
        assert frame.depth_mm.ndim == 2
        k = frame.depth_intrinsics
        assert k is not None
        assert (k.width, k.height) == (frame.depth_mm.shape[1], frame.depth_mm.shape[0])
        assert k.fx > 0.0 and k.fy > 0.0
        assert np.count_nonzero(frame.depth_mm) > 0
    finally:
        source.close()

    assert not any(
        thread.name == "shadowlink-depth" and thread.is_alive() for thread in threading.enumerate()
    )
