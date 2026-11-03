"""Shared fixtures: a synthetic right hand and a real Shadow Hand layout."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadowlink.config import RetargetConfig
from shadowlink.retarget.calibration import Neutral
from shadowlink.retarget.geometry import SPREAD_REST
from shadowlink.types import HandObservation, Side

# MANO / MediaPipe keypoint block starts.
_FINGER_BASE = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
_FINGER_RADIUS = {"thumb": 0.05, "index": 0.09, "middle": 0.09, "ring": 0.09, "pinky": 0.09}
_FINGER_LENGTHS = {
    "thumb": (0.035, 0.030, 0.025),
    "index": (0.045, 0.028, 0.022),
    "middle": (0.050, 0.032, 0.024),
    "ring": (0.046, 0.030, 0.022),
    "pinky": (0.038, 0.025, 0.020),
}


def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``vector`` around a unit ``axis``."""
    cos = math.cos(angle)
    sin = math.sin(angle)
    return (
        vector * cos
        + np.cross(axis, vector) * sin
        + axis * float(np.dot(axis, vector)) * (1.0 - cos)
    )


def synthetic_hand(
    flex: dict[str, tuple[float, float, float]] | None = None,
    spread: dict[str, float] | None = None,
    opposition: float = 0.0,
) -> np.ndarray:
    """Build 21 right-hand keypoints with exactly known metrics.

    The palm lies in the xy plane, fingers point along +x, and the palm normal
    is +z.  ``flex[finger]`` sets the MCP, PIP, and DIP angles; ``spread`` is
    the absolute in-palm angle of each finger (middle finger is zero), and
    ``opposition`` is the thumb's in-palm offset from the index finger.
    """
    flex = {name: (0.0, 0.0, 0.0) for name in _FINGER_BASE} if flex is None else dict(flex)
    spread = dict(SPREAD_REST) if spread is None else dict(spread)

    kp = np.zeros((21, 3), dtype=np.float64)
    for finger in ("index", "middle", "ring", "pinky"):
        theta = float(spread[finger])
        direction = np.array([math.cos(theta), math.sin(theta), 0.0])
        axis = np.array([-math.sin(theta), math.cos(theta), 0.0])
        base = _FINGER_BASE[finger]
        point = direction * _FINGER_RADIUS[finger]
        kp[base] = point
        lengths = _FINGER_LENGTHS[finger]
        for joint, length in enumerate(lengths):
            direction = _rotate(direction, axis, float(flex[finger][joint]))
            point = point + direction * length
            kp[base + 1 + joint] = point

    theta_thumb = float(spread["index"]) - float(opposition)
    direction = np.array([math.cos(theta_thumb), math.sin(theta_thumb), 0.0])
    axis = np.array([-math.sin(theta_thumb), math.cos(theta_thumb), 0.0])
    point = direction * _FINGER_RADIUS["thumb"]
    kp[1] = point
    for joint, length in enumerate(_FINGER_LENGTHS["thumb"]):
        direction = _rotate(direction, axis, float(flex["thumb"][joint]))
        point = point + direction * length
        kp[2 + joint] = point
    return kp


def make_observation(
    keypoints: np.ndarray,
    *,
    side: Side = Side.RIGHT,
    apparent: Side | None = None,
    backend: str = "test",
    frame_seq: int = 1,
) -> HandObservation:
    return HandObservation(
        keypoints_3d=keypoints,
        apparent_side=apparent if apparent is not None else side,
        anatomical_side=side,
        score=1.0,
        frame_seq=frame_seq,
        capture_ts=0.0,
        inference_ts=0.0,
        backend=backend,
    )


@pytest.fixture
def make_hand():
    return synthetic_hand


@pytest.fixture
def make_obs():
    return make_observation


@pytest.fixture(scope="session")
def layout():
    from shadowlink.paths import SCENE_XML
    from shadowlink.sim.model import load_model

    _, actuator_layout = load_model(SCENE_XML)
    return actuator_layout


@pytest.fixture
def retargeter(layout):
    from shadowlink.retarget.mapper import Retargeter

    return Retargeter(RetargetConfig(), Neutral.default(), layout)
