"""Tests for the RGBD frame source using fake colour and depth services."""

from __future__ import annotations

import numpy as np
import pytest

import shadowlink.capture as capture
from shadowlink import rgbd as rgbd_package
from shadowlink.capture import CaptureError
from shadowlink.config import CaptureConfig
from shadowlink.rgbd.depth import DepthError, DepthFrame, DepthHealth
from shadowlink.rgbd.source import RGBDFrameSource
from shadowlink.types import DepthIntrinsics, FramePacket


def intrinsics(width: int = 64, height: int = 48) -> DepthIntrinsics:
    return DepthIntrinsics(
        fx=50.0,
        fy=50.0,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
    )


class FakeColorSource:
    def __init__(self, frames: int = 2, events: list[str] | None = None) -> None:
        self.frames = int(frames)
        self.events = events if events is not None else []
        self.seq = 0
        self.closed = False

    def read(self) -> FramePacket | None:
        if self.closed or self.seq >= self.frames:
            return None
        packet = FramePacket(
            seq=self.seq,
            capture_ts=100.0 + self.seq,
            image_bgr=np.zeros((48, 64, 3), dtype=np.uint8),
        )
        self.seq += 1
        return packet

    @property
    def fps(self) -> float:
        return 30.0

    @property
    def size(self) -> tuple[int, int]:
        return (64, 48)

    def close(self) -> None:
        if not self.closed:
            self.events.append("color")
        self.closed = True


class FakeDepthService:
    def __init__(
        self,
        frames: list[DepthFrame] | None = None,
        *,
        error: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.frames = list(frames or ())
        self.error = error
        self.events = events if events is not None else []
        self.started = False
        self.closed = False
        self.index = 0
        self.health = DepthHealth(state="running", fps=30.0)
        self.device_name = "Fake Depth"
        self.intrinsics = intrinsics()

    def start(self) -> None:
        if self.error is not None:
            raise self.error
        self.started = True

    def latest(self) -> DepthFrame | None:
        if not self.frames:
            return None
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame

    def close(self) -> None:
        if not self.closed:
            self.events.append("depth")
        self.closed = True


def depth_frame(value: int = 800, ts: float = 99.5) -> DepthFrame:
    return DepthFrame(
        depth_mm=np.full((48, 64), value, dtype=np.uint16),
        intrinsics=intrinsics(),
        seq=0,
        capture_ts=ts,
    )


def make_source(depth: FakeDepthService, color: FakeColorSource) -> RGBDFrameSource:
    return RGBDFrameSource(
        CaptureConfig(source="rgbd"),
        color_source=color,
        depth_service=depth,
    )


def test_read_attaches_the_latest_depth_snapshot():
    color = FakeColorSource(1)
    depth = FakeDepthService([depth_frame()])
    source = make_source(depth, color)

    packet = source.read()

    assert depth.started
    assert packet is not None
    assert packet.depth_mm is not None
    np.testing.assert_array_equal(packet.depth_mm, 800)
    assert packet.depth_intrinsics == intrinsics()
    assert packet.depth_ts == pytest.approx(99.5)
    assert packet.seq == 0


def test_read_without_depth_returns_a_colour_only_packet():
    source = make_source(FakeDepthService(), FakeColorSource(1))
    packet = source.read()
    assert packet is not None
    assert packet.depth_mm is None


def test_accessors_delegate_to_the_depth_service():
    health = DepthHealth(state="error", message="unplugged")
    depth = FakeDepthService()
    depth.health = health
    source = make_source(depth, FakeColorSource())
    assert source.depth_health == health
    assert source.depth_device_name == "Fake Depth"
    assert source.depth_intrinsics == intrinsics()


def test_close_stops_depth_before_colour_and_is_idempotent():
    events: list[str] = []
    source = make_source(
        FakeDepthService(events=events),
        FakeColorSource(events=events),
    )
    source.close()
    source.close()
    assert events == ["depth", "color"]


def test_depth_startup_failure_closes_colour_and_raises_capture_error():
    events: list[str] = []
    with pytest.raises(CaptureError, match="depth unavailable"):
        make_source(
            FakeDepthService(error=DepthError("no device"), events=events),
            FakeColorSource(events=events),
        )
    assert events == ["color"]


def test_open_source_dispatches_rgbd_to_the_rgbd_source(monkeypatch):
    calls = {}

    class SpySource:
        def __init__(self, cfg, *, clock=None):
            calls["cfg"] = cfg
            calls["clock"] = clock

    monkeypatch.setattr(rgbd_package.source, "RGBDFrameSource", SpySource)
    cfg = CaptureConfig(source="rgbd")
    result = capture.open_source(cfg, clock=lambda: 1.0)
    assert isinstance(result, SpySource)
    assert calls["cfg"] is cfg
    assert calls["clock"]() == 1.0
