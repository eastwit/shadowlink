"""Camera math for the HaMeR inference path.

HaMeR predicts a weakly perspective camera in crop coordinates.  These helpers
lift it back to the full image and project 3D joints for the overlay.  They are
pure NumPy so they can be tested without torch or the model weights.
"""

from __future__ import annotations

import numpy as np


def cam_crop_to_full(
    pred_cam: np.ndarray,
    box_center: np.ndarray,
    box_size: np.ndarray,
    img_size: np.ndarray,
    focal_length: float | np.ndarray = 5000.0,
) -> np.ndarray:
    """Convert HaMeR's crop camera parameters to full-image translation.

    Args:
        pred_cam: ``(B, 3)`` weak-perspective camera ``(scale, tx, ty)``.
        box_center: ``(B, 2)`` crop centre in pixels.
        box_size: ``(B,)`` crop side length in pixels.
        img_size: ``(B, 2)`` full image size ``(width, height)``.
        focal_length: scalar or ``(B,)`` focal length in pixels.

    Returns:
        ``(B, 3)`` translation in the full-image camera frame.
    """
    cam = np.asarray(pred_cam, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(box_center, dtype=np.float64).reshape(-1, 2)
    size = np.asarray(box_size, dtype=np.float64).reshape(-1)
    image = np.asarray(img_size, dtype=np.float64).reshape(-1, 2)
    if not (cam.shape[0] == center.shape[0] == size.shape[0] == image.shape[0]):
        raise ValueError("camera, box, and image arrays must share a batch dimension")

    focal = np.asarray(focal_length, dtype=np.float64).reshape(-1)
    if focal.size == 1:
        focal = np.full(cam.shape[0], focal[0], dtype=np.float64)
    if focal.size != cam.shape[0]:
        raise ValueError("focal_length must be a scalar or match the batch size")

    width = image[:, 0]
    height = image[:, 1]
    bs = size * cam[:, 0] + 1e-9
    tz = 2.0 * focal / bs
    tx = 2.0 * (center[:, 0] - width / 2.0) / bs + cam[:, 1]
    ty = 2.0 * (center[:, 1] - height / 2.0) / bs + cam[:, 2]
    return np.stack([tx, ty, tz], axis=-1)


def project_points(
    points_3d: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Project hand joints in camera space onto full-image pixels.

    ``points_3d`` are root-relative MANO joints; adding ``camera_translation``
    gives their camera-space position, matching HaMeR's own renderer.
    """
    points = np.asarray(points_3d, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_3d must be (N, 3), got {points.shape}")
    translation = np.asarray(camera_translation, dtype=np.float64).reshape(3)
    focal = float(focal_length)
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("focal_length must be finite and positive")

    z = points[:, 2] + translation[2]
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = width / 2.0 + focal * (points[:, 0] + translation[0]) / z
    v = height / 2.0 + focal * (points[:, 1] + translation[1]) / z
    return np.stack([u, v], axis=-1)


def scaled_focal_length(focal_length: float, image_size: int, img_size: np.ndarray) -> float:
    """HaMeR scales its canonical focal length by the longest image side."""
    image = np.asarray(img_size, dtype=np.float64).reshape(-1)
    if image.size != 2:
        raise ValueError("img_size must contain (width, height)")
    return float(focal_length) / float(image_size) * float(image.max())
