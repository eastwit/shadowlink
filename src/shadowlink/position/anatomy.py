"""Map camera-space palm motion onto the hand's anatomical axes.

The depth sensor reports motion in camera coordinates (``x`` right, ``y`` down,
``z`` away from the sensor).  The retargeter's 3D keypoints describe the same
physical hand in an image-aligned frame, so the palm basis they define can turn
a camera-space displacement into anatomical components:

* palmar (out of the palm),
* thumb side (from pinky toward index),
* distal (toward the middle finger).

Applying the simulated hand's current base orientation then turns those
components into world directions.  This keeps the translation hand-relative:
moving toward the thumb always moves the simulated hand toward its thumb,
regardless of how the physical hand is rotated.
"""

from __future__ import annotations

import numpy as np

from shadowlink.math3d import q_rotate
from shadowlink.retarget.geometry import HandGeometryError, canonicalize, palm_frame
from shadowlink.types import Side


def camera_to_local_axes(
    keypoints_3d: np.ndarray,
    apparent_side: Side,
    anatomical_side: Side,
) -> np.ndarray | None:
    """Return a camera-to-hand matrix in local Shadow Hand axes.

    The result ``A`` maps a camera-space displacement ``d`` to local coordinates
    ordered ``(+x palmar, +y thumb side, +z distal)``.  It returns ``None`` when
    the keypoints are non-finite or the palm basis is degenerate; callers should
    then fall back to the configured camera preset.

    Left hands are canonicalised exactly like the retargeter (mirror ``x``), so
    anatomical directions keep the same meaning for either hand.
    """
    kp = np.asarray(keypoints_3d, dtype=np.float64)
    if kp.shape != (21, 3) or not np.isfinite(kp).all():
        return None
    try:
        forward, side, normal = palm_frame(canonicalize(kp, apparent_side))
    except (HandGeometryError, ValueError):
        return None
    # Columns are the physical axes in the canonical camera frame.  ``side``
    # runs index -> pinky, so the thumb side is -side.
    basis = np.column_stack((normal, -side, forward))
    axes = np.ascontiguousarray(basis.T)
    if anatomical_side is Side.LEFT:
        # canonicalize() mirrored the left hand's x axis; mirror the camera
        # displacement into the same canonical frame before decomposing it.
        axes[:, 0] *= -1.0
    if not np.isfinite(axes).all():
        return None
    axes.setflags(write=False)
    return axes


def hand_base_rotation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Return the world directions of the hand-local ``+x/+y/+z`` axes."""
    return np.column_stack([q_rotate(quat_wxyz, axis) for axis in np.eye(3)])
