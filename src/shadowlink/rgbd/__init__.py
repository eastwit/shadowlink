"""RGBD capture: OpenCV colour plus an OpenNI2 depth stream."""

from shadowlink.rgbd.depth import (
    DepthError,
    DepthFrame,
    DepthHealth,
    DepthService,
    resolve_depth_sdk,
)
from shadowlink.rgbd.source import RGBDFrameSource

__all__ = [
    "DepthError",
    "DepthFrame",
    "DepthHealth",
    "DepthService",
    "RGBDFrameSource",
    "resolve_depth_sdk",
]
