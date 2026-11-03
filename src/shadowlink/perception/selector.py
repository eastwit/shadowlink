"""Choose one detection per frame.

The old implementation selected the hand closest to the *top-left corner*
because it minimized ``cx**2 + cy**2``.  ``center`` here measures distance to
the actual image center.  ``track`` keeps the previously selected hand by IoU
so two hands cannot swap identities between frames.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from shadowlink.types import HandDetection

PickMode = str  # largest | center | track


class DetectionSelector:
    def __init__(self, pick: PickMode = "track", *, iou_threshold: float = 0.2) -> None:
        if pick not in {"largest", "center", "track"}:
            raise ValueError(f"unknown pick mode: {pick!r}")
        self.pick = pick
        self.iou_threshold = float(iou_threshold)
        self._last_bbox: np.ndarray | None = None

    def reset(self) -> None:
        self._last_bbox = None

    def select(
        self,
        detections: Sequence[HandDetection],
        width: int,
        height: int,
    ) -> HandDetection | None:
        if not detections:
            self._last_bbox = None
            return None
        if self.pick == "largest":
            chosen = max(detections, key=lambda det: (det.area, -det.center[1]))
        elif self.pick == "center":
            cx, cy = width * 0.5, height * 0.5
            chosen = min(
                detections,
                key=lambda det: _distance_to(det, cx, cy),
            )
        else:
            chosen = self._track(detections)
        self._last_bbox = chosen.bbox_xyxy.copy()
        return chosen

    def _track(self, detections: Sequence[HandDetection]) -> HandDetection:
        if self._last_bbox is not None:
            best = max(detections, key=lambda det: iou(self._last_bbox, det.bbox_xyxy))
            if iou(self._last_bbox, best.bbox_xyxy) >= self.iou_threshold:
                return best
        return max(detections, key=lambda det: det.area)


def _distance_to(det: HandDetection, cx: float, cy: float) -> float:
    x, y = det.center
    return (x - cx) ** 2 + (y - cy) ** 2


def filter_by_side(detections: Iterable[HandDetection], side: str) -> list[HandDetection]:
    """Filter on the *anatomical* side, never the mirrored image label."""
    if side == "any":
        return list(detections)
    if side not in {"left", "right"}:
        raise ValueError(f"unknown side filter: {side!r}")
    return [det for det in detections if det.anatomical_side.value == side]


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = np.asarray(a, dtype=np.float64).reshape(4)
    bx0, by0, bx1, by1 = np.asarray(b, dtype=np.float64).reshape(4)
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
