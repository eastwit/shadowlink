"""Geometric hand metrics computed from 21 3D keypoints.

The 21 joints follow the MANO / MediaPipe order: 0 wrist, 1-4 thumb,
5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.  Keypoints are expected in an
image-aligned frame; :func:`canonicalize` mirrors left hands so that the rest of
the pipeline always reasons about right-hand geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shadowlink.types import Side

FINGERS = {
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
ALL_FINGERS = ("index", "middle", "ring", "pinky", "thumb")

SPREAD_SIGN = {"index": 1.0, "middle": 1.0, "ring": -1.0, "pinky": -1.0}
SPREAD_REST = {"index": -0.12, "middle": 0.0, "ring": 0.10, "pinky": 0.20}

# Semantic actuator names.  The physical order comes from the MJCF.
FINGER_SLOTS = {
    "index": {"knuckle": "FFJ4", "proximal": "FFJ3", "tendon": "FFJ0"},
    "middle": {"knuckle": "MFJ4", "proximal": "MFJ3", "tendon": "MFJ0"},
    "ring": {"knuckle": "RFJ4", "proximal": "RFJ3", "tendon": "RFJ0"},
    "pinky": {"knuckle": "LFJ4", "proximal": "LFJ3", "tendon": "LFJ0"},
}


class HandGeometryError(ValueError):
    """Raised when keypoints are degenerate or non-finite."""


@dataclass(frozen=True, slots=True)
class HandMetrics:
    flex: dict[str, tuple[float, float, float]]
    spread: dict[str, float]
    opposition: float


def wrap_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _as_keypoints(kp: np.ndarray) -> np.ndarray:
    arr = np.asarray(kp, dtype=np.float64)
    if arr.shape != (21, 3):
        raise HandGeometryError(f"expected keypoints of shape (21, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise HandGeometryError("keypoints contain NaN or infinity")
    return arr


def canonicalize(kp: np.ndarray, side: Side) -> np.ndarray:
    """Mirror a left hand along x so that it has right-hand geometry."""
    arr = _as_keypoints(kp).copy()
    if not side.is_right:
        arr[:, 0] *= -1.0
    return arr


def palm_frame(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(f, s, n)``: finger direction, index-to-pinky, palm normal."""
    kp = _as_keypoints(kp)
    wrist = kp[0]
    f = kp[9] - wrist
    n = np.cross(kp[5] - wrist, kp[17] - wrist)
    f_norm = np.linalg.norm(f)
    n_norm = np.linalg.norm(n)
    if f_norm < 1e-9 or n_norm < 1e-9:
        raise HandGeometryError("degenerate palm: wrist, middle MCP, or palm width is zero")
    f = f / f_norm
    n = n / n_norm
    s = np.cross(n, f)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-9:
        raise HandGeometryError("degenerate palm frame: f and n are parallel")
    s = s / s_norm
    return f, s, n


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = b - a
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-12:
        raise HandGeometryError("degenerate finger: zero-length bone")
    cos = float(np.dot(v1, v2) / denom)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def hand_metrics(kp: np.ndarray) -> HandMetrics:
    """Extract flexion, spread, and thumb opposition from 21 keypoints.

    Input must already be canonical right-hand geometry; use
    :func:`canonicalize` first.  Flexion is the interior angle at each joint:
    0 means a straight finger.  Spread is the in-palm-plane angle relative to
    the middle finger, wrapped to [-pi, pi].
    """
    kp = _as_keypoints(kp)
    f, s, _ = palm_frame(kp)

    flex: dict[str, tuple[float, float, float]] = {}
    for name, chain in FINGERS.items():
        flex[name] = (
            _angle(kp[chain[0]], kp[chain[1]], kp[chain[2]]),
            _angle(kp[chain[1]], kp[chain[2]], kp[chain[3]]),
            _angle(kp[chain[2]], kp[chain[3]], kp[chain[4]]),
        )
    flex["thumb"] = (
        _angle(kp[0], kp[1], kp[2]),
        _angle(kp[1], kp[2], kp[3]),
        _angle(kp[2], kp[3], kp[4]),
    )

    def planar(vector: np.ndarray) -> float:
        return float(np.arctan2(float(np.dot(vector, s)), float(np.dot(vector, f))))

    middle_angle = planar(kp[10] - kp[9])
    spread = {
        name: wrap_pi(planar(kp[chain[2]] - kp[chain[1]]) - middle_angle)
        for name, chain in FINGERS.items()
    }
    opposition = wrap_pi(planar(kp[6] - kp[5]) - planar(kp[2] - kp[1]))
    return HandMetrics(flex=flex, spread=spread, opposition=opposition)
