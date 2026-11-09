"""Diagnose the RGBD colour/depth pair without starting perception.

Use this before a teleoperation run: it confirms the OpenNI2 SDK path, the
colour camera index, the depth intrinsics, and the timestamp skew between the
two unsynchronised streams.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from shadowlink.capture import CaptureError, open_source
from shadowlink.config import CaptureConfig


def _depth_colormap(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm > 0
    scaled = np.zeros(depth_mm.shape, dtype=np.float32)
    if valid.any():
        values = depth_mm[valid].astype(np.float32)
        low = float(np.percentile(values, 1))
        high = float(np.percentile(values, 99))
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth_mm.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    image[~valid] = (0, 0, 0)
    return image


def run_rgbd_check(
    cfg: CaptureConfig,
    *,
    seconds: float = 2.0,
    out: str | None = None,
) -> int:
    """Open the configured RGBD source, print a report, and optionally save a PNG."""
    source = open_source(cfg)
    frames = 0
    with_depth = 0
    skips = 0
    valid_ratios: list[float] = []
    skews: list[float] = []
    last = None
    started = time.monotonic()
    try:
        while time.monotonic() - started < seconds:
            frame = source.read()
            if frame is None:
                skips += 1
                continue
            frames += 1
            last = frame
            if frame.depth_mm is None:
                continue
            with_depth += 1
            depth = frame.depth_mm
            valid_ratios.append(float(np.count_nonzero(depth)) / depth.size)
            if frame.depth_ts is not None:
                skews.append(frame.capture_ts - frame.depth_ts)

        health = source.depth_health
        print(f"depth device: {source.depth_device_name or 'unknown'}")
        print(
            f"depth state:  {health.state} "
            f"({health.fps:.1f} Hz, {health.frames} frames, {health.errors} errors)"
        )
        if health.message:
            print(f"  {health.message}")
        if last is not None and last.depth_intrinsics is not None:
            k = last.depth_intrinsics
            print(
                f"depth image:  {k.width}x{k.height} "
                f"fx={k.fx:.2f} fy={k.fy:.2f} cx={k.cx:.2f} cy={k.cy:.2f}"
            )
        if last is not None:
            print(f"colour image: {last.width}x{last.height} (sequence {last.seq})")
        print(f"frames read:  {frames} ({with_depth} with depth, {skips} skipped)")
        if valid_ratios:
            ratios = np.asarray(valid_ratios, dtype=np.float64)
            print(
                f"valid depth:  mean {100.0 * ratios.mean():.1f}% min {100.0 * ratios.min():.1f}%"
            )
        if skews:
            values = np.asarray(skews, dtype=np.float64)
            print(
                f"colour-depth skew: median {1000.0 * np.median(values):+.1f} ms "
                f"max {1000.0 * np.abs(values).max():.1f} ms"
            )

        if out and last is not None and last.depth_mm is not None:
            depth_vis = cv2.resize(
                _depth_colormap(last.depth_mm),
                (last.image_bgr.shape[1], last.image_bgr.shape[0]),
            )
            combined = np.hstack([last.image_bgr, depth_vis])
            if not cv2.imwrite(out, combined):
                raise CaptureError(f"cannot write {out}")
            print(f"wrote {out}")
    finally:
        source.close()

    if not with_depth:
        print("no depth frames arrived; check --depth-sdk and the USB connection")
        return 1
    mean_ratio = float(np.mean(valid_ratios)) if valid_ratios else 0.0
    if mean_ratio < 0.05:
        print("almost no valid depth pixels; the sensor may be covered or too close")
        return 1
    return 0
