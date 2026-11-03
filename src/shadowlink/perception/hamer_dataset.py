"""Minimal HaMeR crop preprocessing.

This replaces ``hamer.datasets.vitdet_dataset.ViTDetDataset`` for inference.  It
keeps the same crop geometry and normalisation but avoids importing HaMeR's
training stack (webdataset, pytorch-lightning datasets) and the per-crop
``print`` in the upstream dataset.  The batch is assembled directly in the
inference thread, so no DataLoader workers are needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CropBatch:
    """A batch of normalised hand crops ready to be copied into tensors."""

    images: np.ndarray  # (B, 3, S, S) float32, normalized
    box_center: np.ndarray  # (B, 2) float32, pixels
    box_size: np.ndarray  # (B,) float32, pixels
    img_size: np.ndarray  # (B, 2) float32, (width, height)
    right: np.ndarray  # (B,) float32, 1 = right hand
    flipped: np.ndarray  # (B,) bool, crop is horizontally flipped

    def __len__(self) -> int:
        return int(self.images.shape[0])


def prepare_batch(
    cfg,
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    rights: np.ndarray,
    rescale_factor: float = 2.0,
) -> CropBatch:
    """Crop every detection out of ``image_bgr`` and normalise it for HaMeR.

    ``rights`` uses the *apparent* hand side: left hands are flipped inside the
    crop exactly as HaMeR's training data was prepared, and the backend flips
    the predictions back afterwards.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR image, got {image.shape}")
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    rights = np.asarray(rights).astype(bool).reshape(-1)
    if boxes.shape[0] != rights.shape[0]:
        raise ValueError("boxes and rights must have the same length")
    if boxes.shape[0] == 0:
        raise ValueError("cannot prepare an empty batch")
    if rescale_factor <= 0:
        raise ValueError("rescale_factor must be positive")

    model_cfg = cfg.MODEL
    image_size = int(model_cfg.IMAGE_SIZE)
    mean = 255.0 * np.asarray(model_cfg.IMAGE_MEAN, dtype=np.float64)
    std = 255.0 * np.asarray(model_cfg.IMAGE_STD, dtype=np.float64)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("MODEL.IMAGE_MEAN and MODEL.IMAGE_STD must have three values")
    bbox_shape = getattr(model_cfg, "BBOX_SHAPE", None)

    center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
    scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
    height, width = image.shape[:2]

    patches = []
    sizes = []
    flipped = []
    for index in range(boxes.shape[0]):
        size = float(expand_to_aspect_ratio(scale[index] * 200.0, bbox_shape).max())
        size = max(size, 1.0)
        cx, cy = float(center[index, 0]), float(center[index, 1])

        source = image
        downsampling = (size / image_size) / 2.0
        if downsampling > 1.1:
            sigma = (downsampling - 1.0) / 2.0
            source = cv2.GaussianBlur(image.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)

        flip = not bool(rights[index])
        patch = _generate_patch(source, cx, cy, size, size, image_size, image_size, flip)
        patch = patch[:, :, ::-1].astype(np.float32)  # BGR -> RGB
        patch = np.transpose(patch, (2, 0, 1))
        patch = (patch - mean[:, None, None]) / std[:, None, None]
        patches.append(patch.astype(np.float32))
        sizes.append(size)
        flipped.append(flip)

    count = len(patches)
    return CropBatch(
        images=np.stack(patches),
        box_center=center.astype(np.float32),
        box_size=np.asarray(sizes, dtype=np.float32),
        img_size=np.tile(np.array([width, height], dtype=np.float32), (count, 1)),
        right=rights.astype(np.float32),
        flipped=np.asarray(flipped, dtype=bool),
    )


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None) -> np.ndarray:
    """Grow a (width, height) box until it matches the target aspect ratio."""
    if target_aspect_ratio is None:
        return np.asarray(input_shape, dtype=np.float64)
    try:
        width, height = np.asarray(input_shape, dtype=np.float64).reshape(2)
        target_w, target_h = np.asarray(target_aspect_ratio, dtype=np.float64).reshape(2)
    except (TypeError, ValueError):
        return np.asarray(input_shape, dtype=np.float64)
    if width <= 0 or height <= 0:
        return np.asarray([width, height], dtype=np.float64)

    if height / width < target_h / target_w:
        new_height = max(width * target_h / target_w, height)
        new_width = width
    else:
        new_height = height
        new_width = max(height * target_w / target_h, width)
    return np.array([new_width, new_height], dtype=np.float64)


def _generate_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    out_width: int,
    out_height: int,
    flip: bool,
) -> np.ndarray:
    if flip:
        image = image[:, ::-1]
        center_x = image.shape[1] - center_x - 1
    transform = _patch_transform(center_x, center_y, box_width, box_height, out_width, out_height)
    return cv2.warpAffine(
        image,
        transform,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _patch_transform(
    center_x: float,
    center_y: float,
    box_width: float,
    box_height: float,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    src = np.array(
        [
            [center_x, center_y],
            [center_x, center_y + box_height * 0.5],
            [center_x + box_width * 0.5, center_y],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [out_width * 0.5, out_height * 0.5],
            [out_width * 0.5, out_height * 1.0],
            [out_width * 1.0, out_height * 0.5],
        ],
        dtype=np.float32,
    )
    return cv2.getAffineTransform(src, dst)
