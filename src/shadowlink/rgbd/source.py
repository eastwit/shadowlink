"""A frame source that attaches an OpenNI2 depth snapshot to each colour frame."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from shadowlink.capture import CaptureError, OpenCVFrameSource
from shadowlink.config import CaptureConfig
from shadowlink.rgbd.depth import DepthError, DepthHealth, DepthService
from shadowlink.types import FramePacket


class RGBDFrameSource:
    """Colour through OpenCV (UVC) and depth through OpenNI2.

    The two sensors are independent: depth is the newest frame available when a
    colour frame is read.  The app checks the timestamp skew before using it.
    """

    def __init__(
        self,
        cfg: CaptureConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        color_source=None,
        depth_service=None,
    ) -> None:
        color_cfg = replace(cfg, source="camera") if cfg.source == "rgbd" else cfg
        self.cfg = cfg
        self._color = (
            color_source if color_source is not None else OpenCVFrameSource(color_cfg, clock=clock)
        )
        self._depth = depth_service if depth_service is not None else DepthService(cfg, clock=clock)
        self._closed = False
        try:
            self._depth.start()
        except DepthError as exc:
            self._color.close()
            raise CaptureError(f"RGBD depth unavailable: {exc}") from exc
        except Exception:
            self._color.close()
            raise

    def read(self) -> FramePacket | None:
        packet = self._color.read()
        if packet is None:
            return None
        depth = self._depth.latest()
        if depth is None:
            return packet
        return replace(
            packet,
            depth_mm=depth.depth_mm,
            depth_intrinsics=depth.intrinsics,
            depth_ts=depth.capture_ts,
        )

    @property
    def fps(self) -> float | None:
        return self._color.fps

    @property
    def size(self) -> tuple[int, int] | None:
        return self._color.size

    @property
    def depth_health(self) -> DepthHealth:
        return self._depth.health

    @property
    def depth_device_name(self) -> str | None:
        return getattr(self._depth, "device_name", None)

    @property
    def depth_intrinsics(self):
        return getattr(self._depth, "intrinsics", None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._depth.close()
        finally:
            self._color.close()
