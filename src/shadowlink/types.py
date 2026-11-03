"""Shared data types and protocols.

This module deliberately depends on NumPy only.  Perception backends,
retargeting, and the simulation layer communicate through these immutable
records so that no mutable frame or keypoint array is shared across threads.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


class Side(str, Enum):
    """Hand side, independent of the left/right label of a mirrored image."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def is_right(self) -> bool:
        return self is Side.RIGHT

    def flipped(self) -> Side:
        return Side.LEFT if self is Side.RIGHT else Side.RIGHT

    @classmethod
    def from_mediapipe_label(cls, label: str) -> Side:
        # MediaPipe Hands returns "Left" or "Right" for the anatomical hand
        # when the input image is not mirrored.
        normalized = label.strip().lower()
        if normalized == "left":
            return cls.LEFT
        if normalized == "right":
            return cls.RIGHT
        raise ValueError(f"Unknown MediaPipe handedness label: {label!r}")


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One captured frame, already mirrored exactly once if requested."""

    seq: int
    capture_ts: float
    image_bgr: np.ndarray
    mirrored: bool = False
    source_fps: float | None = None

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError("frame sequence must be non-negative")
        if self.image_bgr.ndim != 3 or self.image_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got {self.image_bgr.shape}")
        if self.image_bgr.dtype != np.uint8:
            raise TypeError(f"expected uint8 image, got {self.image_bgr.dtype}")
        self.image_bgr.setflags(write=False)

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])

    def rgb_contiguous(self) -> np.ndarray:
        """Return a contiguous RGB copy suitable for MediaPipe."""
        return np.ascontiguousarray(self.image_bgr[:, :, ::-1])

    def age(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.capture_ts


@dataclass(frozen=True, slots=True)
class HandDetection:
    """A hand bounding box plus detector metadata for one frame."""

    bbox_xyxy: np.ndarray
    apparent_side: Side
    anatomical_side: Side
    score: float = 1.0
    world_landmarks: np.ndarray | None = None
    landmarks_2d: np.ndarray | None = None
    frame_seq: int = -1

    def __post_init__(self) -> None:
        box = np.asarray(self.bbox_xyxy, dtype=np.float64).reshape(4)
        box.setflags(write=False)
        object.__setattr__(self, "bbox_xyxy", box)
        if self.world_landmarks is not None:
            world = np.asarray(self.world_landmarks, dtype=np.float64)
            if world.shape != (21, 3):
                raise ValueError(f"world_landmarks must be (21, 3), got {world.shape}")
            world.setflags(write=False)
            object.__setattr__(self, "world_landmarks", world)
        if self.landmarks_2d is not None:
            points = np.asarray(self.landmarks_2d, dtype=np.float64)
            if points.shape != (21, 2):
                raise ValueError(f"landmarks_2d must be (21, 2), got {points.shape}")
            points.setflags(write=False)
            object.__setattr__(self, "landmarks_2d", points)

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.bbox_xyxy
        return max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0))

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox_xyxy
        return (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)


@dataclass(frozen=True, slots=True)
class HandObservation:
    """A 3D hand observation for one frame.

    ``keypoints_3d`` follows the MANO / MediaPipe 21-joint order and is aligned
    with the processed (possibly mirrored) image so that it can be drawn as an
    overlay.  Left hands are *not* mirrored here; the retargeter calls
    ``canonicalize(..., apparent_side)`` before measuring geometry.
    ``apparent_side`` is the side in the processed image; ``anatomical_side``
    is the side of the person's actual hand.
    """

    keypoints_3d: np.ndarray
    apparent_side: Side
    anatomical_side: Side
    score: float
    frame_seq: int
    capture_ts: float
    inference_ts: float
    backend: str
    bbox_xyxy: np.ndarray | None = None
    keypoints_2d: np.ndarray | None = None
    vertices: np.ndarray | None = None
    mesh_edges: np.ndarray | None = None
    camera_translation: np.ndarray | None = None
    focal_length: float | None = None
    detect_ms: float = 0.0
    infer_ms: float = 0.0

    def __post_init__(self) -> None:
        kp = np.asarray(self.keypoints_3d, dtype=np.float64)
        if kp.shape != (21, 3):
            raise ValueError(f"keypoints_3d must be (21, 3), got {kp.shape}")
        kp.setflags(write=False)
        object.__setattr__(self, "keypoints_3d", kp)
        if self.bbox_xyxy is not None:
            box = np.asarray(self.bbox_xyxy, dtype=np.float64).reshape(4)
            box.setflags(write=False)
            object.__setattr__(self, "bbox_xyxy", box)
        if self.keypoints_2d is not None:
            points = np.asarray(self.keypoints_2d, dtype=np.float64)
            if points.shape != (21, 2):
                raise ValueError(f"keypoints_2d must be (21, 2), got {points.shape}")
            points.setflags(write=False)
            object.__setattr__(self, "keypoints_2d", points)
        if self.vertices is not None:
            verts = np.asarray(self.vertices, dtype=np.float32)
            verts.setflags(write=False)
            object.__setattr__(self, "vertices", verts)
        if self.mesh_edges is not None:
            edges = np.asarray(self.mesh_edges, dtype=np.int32)
            if edges.ndim != 2 or edges.shape[1] != 2:
                raise ValueError(f"mesh_edges must be (N, 2), got {edges.shape}")
            edges.setflags(write=False)
            object.__setattr__(self, "mesh_edges", edges)

    def age(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.capture_ts

    def is_fresh(self, now: float, max_age_s: float) -> bool:
        return self.age(now) <= max_age_s


@dataclass(frozen=True, slots=True)
class PerceptionResult:
    """The output of one perception pass for one captured frame."""

    frame_seq: int
    capture_ts: float
    finished_ts: float
    observations: tuple[HandObservation, ...] = field(default_factory=tuple)
    detect_ms: float = 0.0
    infer_ms: float = 0.0
    dropped: int = 0

    def is_fresh(self, now: float, max_age_s: float) -> bool:
        return now - self.capture_ts <= max_age_s

    @property
    def age(self) -> float:
        return max(0.0, self.finished_ts - self.capture_ts)


@runtime_checkable
class FrameSource(Protocol):
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...
    @property
    def fps(self) -> float | None: ...
    @property
    def size(self) -> tuple[int, int] | None: ...


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, frame: FramePacket) -> list[HandDetection]: ...
    def close(self) -> None: ...


@runtime_checkable
class KeypointBackend(Protocol):
    name: str
    requires_detections: bool
    supports_mesh: bool

    def estimate(
        self, frame: FramePacket, detections: Sequence[HandDetection]
    ) -> list[HandObservation]: ...
    def close(self) -> None: ...
