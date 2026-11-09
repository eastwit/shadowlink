"""Locate the palm in camera coordinates from one depth frame.

The colour image and the depth image come from two unsynchronised sensors, so
the mapping here is intentionally approximate: colour landmarks are unmirrored,
normalised, and resampled at the depth resolution.  Each palm landmark uses a
small depth window; the per-landmark median and the cross-landmark median reject
holes and background pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shadowlink.config import PositionConfig
from shadowlink.types import DepthIntrinsics

# Wrist plus the five-finger MCP row: a stable, mostly planar palm patch.
PALM_LANDMARKS = (0, 5, 9, 13, 17)


@dataclass(frozen=True, slots=True)
class PalmMeasurement:
    """The palm centre in the depth sensor frame, in metres."""

    xyz_m: np.ndarray
    frame_seq: int
    capture_ts: float
    depth_ts: float
    valid_landmarks: int
    spread_m: float

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=np.float64).reshape(3)
        if not np.isfinite(xyz).all():
            raise ValueError("palm position must be finite")
        xyz.setflags(write=False)
        object.__setattr__(self, "xyz_m", xyz)
        if self.valid_landmarks < 1:
            raise ValueError("valid_landmarks must be positive")
        if not math.isfinite(self.spread_m) or self.spread_m < 0:
            raise ValueError("spread_m must be finite and >= 0")


def _sample_landmark(
    depth: np.ndarray,
    intrinsics: DepthIntrinsics,
    u: int,
    v: int,
    radius: int,
    min_depth_mm: float,
    max_depth_mm: float,
    min_valid_pixels: int,
) -> np.ndarray | None:
    height, width = depth.shape
    u0, u1 = max(0, u - radius), min(width, u + radius + 1)
    v0, v1 = max(0, v - radius), min(height, v + radius + 1)
    if u0 >= u1 or v0 >= v1:
        return None
    window = depth[v0:v1, u0:u1]
    valid = window[(window >= min_depth_mm) & (window <= max_depth_mm)]
    if valid.size < min_valid_pixels:
        return None
    z = float(np.median(valid)) / 1000.0
    x = (float(u) - intrinsics.cx) * z / intrinsics.fx
    y = (float(v) - intrinsics.cy) * z / intrinsics.fy
    point = np.array([x, y, z], dtype=np.float64)
    return point if np.isfinite(point).all() else None


def sample_palm_position(
    depth_mm: np.ndarray,
    intrinsics: DepthIntrinsics,
    keypoints_2d: np.ndarray,
    *,
    mirrored: bool,
    color_size: tuple[int, int],
    cfg: PositionConfig,
    frame_seq: int,
    capture_ts: float,
    depth_ts: float,
) -> PalmMeasurement | None:
    """Return the palm centre in camera coordinates, or ``None`` when unusable.

    ``keypoints_2d`` are in the processed (possibly mirrored) colour image, in
    pixels.  ``color_size`` is ``(width, height)``.  The depth image is raw.
    """
    depth = np.asarray(depth_mm)
    points = np.asarray(keypoints_2d, dtype=np.float64)
    if depth.ndim != 2 or points.shape != (21, 2):
        return None
    if depth.shape != (intrinsics.height, intrinsics.width):
        return None
    color_width, color_height = float(color_size[0]), float(color_size[1])
    if color_width <= 0 or color_height <= 0:
        return None

    radius = max(1, int(cfg.sample_radius_px))
    samples: list[np.ndarray] = []
    for index in PALM_LANDMARKS:
        u_px, v_px = points[index]
        if not (math.isfinite(u_px) and math.isfinite(v_px)):
            continue
        u_norm = u_px / color_width
        v_norm = v_px / color_height
        if mirrored:
            u_norm = 1.0 - u_norm
        u_norm += cfg.align_offset_x
        v_norm += cfg.align_offset_y
        if not (0.0 <= u_norm <= 1.0 and 0.0 <= v_norm <= 1.0):
            continue
        u = int(round(u_norm * (depth.shape[1] - 1)))
        v = int(round(v_norm * (depth.shape[0] - 1)))
        point = _sample_landmark(
            depth,
            intrinsics,
            u,
            v,
            radius,
            cfg.min_depth_mm,
            cfg.max_depth_mm,
            cfg.min_valid_pixels,
        )
        if point is not None:
            samples.append(point)

    if len(samples) < cfg.min_landmarks:
        return None
    stack = np.stack(samples, axis=0)
    xyz = np.median(stack, axis=0)
    if not np.isfinite(xyz).all():
        return None
    spread = float(np.max(np.linalg.norm(stack - xyz, axis=1)))
    if spread > cfg.max_spread_m:
        return None
    return PalmMeasurement(
        xyz_m=xyz,
        frame_seq=frame_seq,
        capture_ts=capture_ts,
        depth_ts=depth_ts,
        valid_landmarks=len(samples),
        spread_m=spread,
    )
