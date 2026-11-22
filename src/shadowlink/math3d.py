"""Small, dependency-free 3D rotation helpers.

ShadowLink uses ``wxyz`` quaternions everywhere because that is MuJoCo's
convention.  Keeping the helpers in one module makes the controller and the
retargeting layer agree on sign and ordering conventions without pulling in
SciPy.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-9


def as_quat(value, *, name: str = "quaternion") -> np.ndarray:
    """Return ``value`` as a finite float64 ``(4,)`` array."""
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError(f"{name} must have 4 components, got {quat.shape}")
    if not np.isfinite(quat).all():
        raise ValueError(f"{name} must be finite")
    return quat


def as_vector(value, size: int = 3, *, name: str = "vector") -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have {size} components, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def q_norm(quat) -> np.ndarray:
    """Normalise a quaternion, rejecting a zero or non-finite input."""
    q = as_quat(quat)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < _EPS:
        raise ValueError("quaternion norm is too small to normalise")
    return q / norm


def q_conj(quat) -> np.ndarray:
    q = q_norm(quat)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def q_mul(left, right) -> np.ndarray:
    """Hamilton product ``left ⊗ right``."""
    a = q_norm(left)
    b = q_norm(right)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def q_rotate(quat, vector) -> np.ndarray:
    """Rotate ``vector`` by ``quat``."""
    q = q_norm(quat)
    v = as_vector(vector)
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return np.array(
        [
            v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float64,
    )


def q_from_axis_angle(axis, angle_rad: float) -> np.ndarray:
    """Quaternion for a rotation of ``angle_rad`` about ``axis``."""
    direction = as_vector(axis)
    norm = float(np.linalg.norm(direction))
    if norm < _EPS:
        raise ValueError("rotation axis must be non-zero")
    direction = direction / norm
    half = 0.5 * float(angle_rad)
    s = math.sin(half)
    return np.array(
        [math.cos(half), direction[0] * s, direction[1] * s, direction[2] * s],
        dtype=np.float64,
    )


def q_to_axis_angle(quat) -> tuple[np.ndarray, float]:
    """Return ``(axis, angle)`` for the shortest rotation.

    The angle is in ``[0, pi]``.  Identity returns the +X axis and a zero
    angle, which keeps callers free of special cases.
    """
    q = q_norm(quat)
    if q[0] < 0.0:
        q = -q
    vector = q[1:]
    sin_half = float(np.linalg.norm(vector))
    if sin_half < _EPS:
        return np.array([1.0, 0.0, 0.0]), 0.0
    angle = 2.0 * math.atan2(sin_half, float(q[0]))
    return vector / sin_half, angle


def q_angle_between(left, right) -> float:
    """Smallest angle between two orientations, in radians."""
    a = q_norm(left)
    b = q_norm(right)
    dot = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def q_slerp(left, right, amount: float) -> np.ndarray:
    """Shortest-arc spherical interpolation."""
    a = q_norm(left)
    b = q_norm(right)
    t = float(amount)
    if not math.isfinite(t):
        raise ValueError("slerp amount must be finite")
    t = min(1.0, max(0.0, t))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        return q_norm(a + t * (b - a))
    theta = math.acos(min(1.0, max(-1.0, dot)))
    sin_theta = math.sin(theta)
    return q_norm(
        (math.sin((1.0 - t) * theta) / sin_theta) * a + (math.sin(t * theta) / sin_theta) * b
    )


def q_from_matrix(matrix) -> np.ndarray:
    """Convert a finite 3x3 rotation matrix to a normalised ``wxyz`` quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got {m.shape}")
    if not np.isfinite(m).all():
        raise ValueError("rotation matrix must be finite")
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = q_norm([w, x, y, z])
    return -q if q[0] < 0.0 else q


def q_to_matrix(quat) -> np.ndarray:
    """Convert a ``wxyz`` quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q_norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def q_from_two_vectors(source, target) -> np.ndarray:
    """Shortest rotation taking ``source`` onto ``target``."""
    a = as_vector(source)
    b = as_vector(target)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        raise ValueError("cannot align a zero-length vector")
    a = a / na
    b = b / nb
    dot = float(np.dot(a, b))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-12:
        # Opposite vectors: any perpendicular axis is a valid half turn.
        candidate = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(candidate)) < 1e-6:
            candidate = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return q_from_axis_angle(candidate, math.pi)
    cross = np.cross(a, b)
    return q_norm(np.array([1.0 + dot, cross[0], cross[1], cross[2]]))


def q_swing_twist(quat, axis) -> tuple[np.ndarray, np.ndarray]:
    """Split ``quat`` into ``swing ⊗ twist`` about a body axis.

    ``twist`` is the pure rotation about ``axis``; ``swing`` is what remains.
    This is used to keep only the tag's blue-axis yaw while the phone supplies
    the two tilt axes.
    """
    q = q_norm(quat)
    direction = as_vector(axis)
    norm = float(np.linalg.norm(direction))
    if norm < _EPS:
        raise ValueError("twist axis must be non-zero")
    direction = direction / norm
    projection = float(np.dot(q[1:], direction))
    twist = np.array(
        [q[0], projection * direction[0], projection * direction[1], projection * direction[2]],
        dtype=np.float64,
    )
    twist_norm = float(np.linalg.norm(twist))
    if twist_norm < _EPS:
        return q.copy(), np.array([1.0, 0.0, 0.0, 0.0])
    twist = twist / twist_norm
    swing = q_mul(q, q_conj(twist))
    return swing, twist


def q_euler_deg(quat) -> tuple[float, float, float]:
    """Return ``(pitch, roll, yaw)`` in degrees for diagnostics.

    Only used for HUD diagnostics; teleoperation always works with quaternions
    so it never depends on Euler singularities.
    """
    w, x, y, z = q_norm(quat)
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return pitch, roll, yaw
