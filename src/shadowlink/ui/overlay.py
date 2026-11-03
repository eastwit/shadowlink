"""ASCII-only preview drawing.

OpenCV's Hershey fonts cannot render non-ASCII text, so all HUD strings are
plain ASCII.  Drawing functions mutate the image they are given; the app hands
them a writable copy of the captured frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from shadowlink.perception.hamer_math import project_points
from shadowlink.types import HandObservation

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

GREEN = (80, 220, 80)
AMBER = (60, 190, 250)
WHITE = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_skeleton(
    image: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int] = GREEN,
    thickness: int = 2,
    radius: int = 3,
) -> None:
    if points is None:
        return
    points = np.asarray(points)
    if points.shape != (21, 2):
        return
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            image,
            _pixel(points[start]),
            _pixel(points[end]),
            color,
            thickness,
            cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(image, _pixel(point), radius, color, -1, cv2.LINE_AA)


def draw_bbox(
    image: np.ndarray,
    bbox_xyxy: np.ndarray | None,
    label: str,
    *,
    color: tuple[int, int, int] = GREEN,
    thickness: int = 1,
) -> None:
    if bbox_xyxy is None:
        return
    x0, y0, x1, y1 = (int(round(float(v))) for v in np.asarray(bbox_xyxy).reshape(4))
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (x0, max(14, y0 - 6)),
            FONT,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_mesh(
    image: np.ndarray,
    vertices: np.ndarray | None,
    camera_translation: np.ndarray | None,
    focal_length: float | None,
    edges: np.ndarray | None,
    *,
    color: tuple[int, int, int] = AMBER,
    thickness: int = 1,
    stride: int = 2,
) -> None:
    """Project a MANO mesh and draw a sampled wireframe."""
    if vertices is None or camera_translation is None or edges is None:
        return
    if focal_length is None:
        return
    focal = float(focal_length)
    if not np.isfinite(focal) or focal <= 0:
        return
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return
    translation = np.asarray(camera_translation, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        return
    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.size == 0:
        return
    edges = edges[:: max(1, int(stride))]
    if edges.min() < 0 or edges.max() >= len(vertices):
        return

    points = project_points(
        vertices,
        translation,
        focal,
        image.shape[1],
        image.shape[0],
    )
    if not np.isfinite(points).all():
        return
    points = np.clip(points, -100000.0, 100000.0)
    for start, end in edges:
        cv2.line(
            image,
            _pixel(points[start]),
            _pixel(points[end]),
            color,
            thickness,
            cv2.LINE_8,
        )


def draw_observation(
    image: np.ndarray,
    observation: HandObservation | None,
    mode: str = "skeleton",
) -> None:
    if observation is None or mode == "none":
        return
    label = f"{observation.backend} {observation.anatomical_side.value} {observation.score:.2f}"
    draw_bbox(image, observation.bbox_xyxy, label)
    if mode == "mesh":
        draw_mesh(
            image,
            observation.vertices,
            observation.camera_translation,
            observation.focal_length,
            observation.mesh_edges,
        )
    if mode in {"skeleton", "mesh"} and observation.keypoints_2d is not None:
        draw_skeleton(image, observation.keypoints_2d)


def draw_hud(
    image: np.ndarray,
    lines: Sequence[str],
    *,
    origin: tuple[int, int] = (10, 10),
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.5,
    line_height: int = 18,
) -> None:
    if not lines:
        return
    x, y = origin
    widths = []
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, FONT, scale, 1)
        widths.append(width)
    panel_width = max(widths) + 12
    panel_height = line_height * len(lines) + 10
    x0, y0 = max(0, x - 6), max(0, y - 6)
    x1 = min(image.shape[1], x0 + panel_width)
    y1 = min(image.shape[0], y0 + panel_height)
    roi = image[y0:y1, x0:x1]
    if roi.size:
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, 0.45, roi, 0.55, 0.0, roi)
    for index, line in enumerate(lines):
        baseline = y + index * line_height
        if baseline >= image.shape[0] - 2:
            break
        cv2.putText(image, line, (x, baseline), FONT, scale, color, 1, cv2.LINE_AA)


def _pixel(point) -> tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))
