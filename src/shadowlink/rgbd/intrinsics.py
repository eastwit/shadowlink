"""Depth intrinsics from the device field of view, with explicit overrides."""

from __future__ import annotations

import math

from shadowlink.config import CaptureConfig
from shadowlink.types import DepthIntrinsics

# Nominal Astra Mini S Pro values, used only when the device reports no FOV.
NOMINAL_FX = 570.3
NOMINAL_FY = 570.3


def fov_to_focal(fov, size: int) -> float | None:
    """Convert a horizontal/vertical field of view to a focal length in pixels."""
    if fov is None:
        return None
    value = float(fov)
    if not math.isfinite(value) or value <= 0.0:
        return None
    if value > 3.5:  # degrees; radians are always below pi
        value = math.radians(value)
    tangent = math.tan(value / 2.0)
    if tangent <= 1e-6:
        return None
    return (size / 2.0) / tangent


def intrinsics_from_device(
    stream,
    width: int,
    height: int,
    cfg: CaptureConfig,
) -> DepthIntrinsics:
    """Build intrinsics from the stream's FOV, falling back to nominal values."""
    horizontal = vertical = None
    try:
        horizontal = stream.get_horizontal_fov()
        vertical = stream.get_vertical_fov()
    except Exception:  # pragma: no cover - depends on the OpenNI build
        pass
    fx = fov_to_focal(horizontal, width)
    fy = fov_to_focal(vertical, height)
    if fx is None:
        fx = NOMINAL_FX
    if fy is None:
        fy = NOMINAL_FY
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    if cfg.depth_fx is not None:
        fx = float(cfg.depth_fx)
    if cfg.depth_fy is not None:
        fy = float(cfg.depth_fy)
    if cfg.depth_cx is not None:
        cx = float(cfg.depth_cx)
    if cfg.depth_cy is not None:
        cy = float(cfg.depth_cy)
    return DepthIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=int(width), height=int(height))
