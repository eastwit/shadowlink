"""Immutable AprilTag pose samples and service health."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from shadowlink.math3d import q_norm


@dataclass(frozen=True, slots=True)
class TagSample:
    """One tag pose in the camera optical frame.

    ``translation_m`` is the tag origin in camera coordinates (x right, y down,
    z forward, metres).  ``quat_wxyz`` maps tag-body coordinates to the camera
    frame: ``p_camera = R(tag->camera) @ p_tag``.
    """

    translation_m: np.ndarray
    quat_wxyz: np.ndarray
    timestamp: float
    seq: int
    tag_id: int
    ready: bool = True

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_m, dtype=np.float64).reshape(3)
        if not np.isfinite(translation).all():
            raise ValueError("tag translation must be finite")
        quat = q_norm(self.quat_wxyz)
        translation.setflags(write=False)
        quat.setflags(write=False)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "quat_wxyz", quat)
        if not math.isfinite(self.timestamp):
            raise ValueError("tag timestamp must be finite")
        if self.seq < 0:
            raise ValueError("tag seq must be >= 0")
        if self.tag_id < 0:
            raise ValueError("tag id must be >= 0")


@dataclass(frozen=True, slots=True)
class TagHealth:
    """Diagnostics for the background tag camera thread."""

    state: str = "idle"  # idle | starting | searching | running | stale | error | stopped
    message: str = ""
    camera: str = ""
    api: str = ""
    rate_hz: float = 0.0
    tag_id: int | None = None
    mirror: str = "auto"
    errors: int = 0

    @property
    def failed(self) -> bool:
        return self.state == "error"
