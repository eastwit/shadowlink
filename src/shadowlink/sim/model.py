"""MuJoCo model loading and actuator introspection.

The Shadow Hand MJCF is the single source of truth for actuator order, joint
versus tendon wiring, and control ranges.  Nothing in this module hard-codes a
range that the model already provides; the expected name list is only used to
fail loudly if the model is swapped or edited.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from shadowlink.paths import KEYFRAMES_XML, SCENE_XML

EXPECTED_ACTUATOR_NAMES: tuple[str, ...] = (
    "rh_A_WRJ2",
    "rh_A_WRJ1",
    "rh_A_THJ5",
    "rh_A_THJ4",
    "rh_A_THJ3",
    "rh_A_THJ2",
    "rh_A_THJ1",
    "rh_A_FFJ4",
    "rh_A_FFJ3",
    "rh_A_FFJ0",
    "rh_A_MFJ4",
    "rh_A_MFJ3",
    "rh_A_MFJ0",
    "rh_A_RFJ4",
    "rh_A_RFJ3",
    "rh_A_RFJ0",
    "rh_A_LFJ5",
    "rh_A_LFJ4",
    "rh_A_LFJ3",
    "rh_A_LFJ0",
)

_EXPECTED_KINDS: dict[str, Literal["joint", "tendon"]] = {
    "FFJ0": "tendon",
    "MFJ0": "tendon",
    "RFJ0": "tendon",
    "LFJ0": "tendon",
}


class ModelError(RuntimeError):
    """Raised when the MJCF does not match the expectations of the retargeter."""


@dataclass(frozen=True, slots=True)
class ActuatorSpec:
    index: int
    name: str
    semantic: str
    kind: Literal["joint", "tendon"]
    target_name: str
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ActuatorLayout:
    specs: tuple[ActuatorSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    @property
    def semantics(self) -> tuple[str, ...]:
        return tuple(spec.semantic for spec in self.specs)

    @property
    def low(self) -> np.ndarray:
        return np.array([spec.low for spec in self.specs], dtype=np.float64)

    @property
    def high(self) -> np.ndarray:
        return np.array([spec.high for spec in self.specs], dtype=np.float64)

    @property
    def index(self) -> dict[str, int]:
        return {spec.name: spec.index for spec in self.specs}

    @property
    def semantic_index(self) -> dict[str, int]:
        return {spec.semantic: spec.index for spec in self.specs}

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> ActuatorLayout:
        if model.nu != len(EXPECTED_ACTUATOR_NAMES):
            raise ModelError(f"expected {len(EXPECTED_ACTUATOR_NAMES)} actuators, got {model.nu}")
        specs = []
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name is None:
                raise ModelError(f"actuator {i} has no name")
            trn_type = int(model.actuator_trntype[i])
            trn_id = int(model.actuator_trnid[i, 0])
            if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT):
                kind: Literal["joint", "tendon"] = "joint"
                target_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_id)
            elif trn_type == int(mujoco.mjtTrn.mjTRN_TENDON):
                kind = "tendon"
                target_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, trn_id)
            else:
                raise ModelError(f"actuator {name} uses unsupported transmission {trn_type}")
            if target_name is None:
                raise ModelError(f"actuator {name} references an unnamed {kind}")
            if not bool(model.actuator_ctrllimited[i]):
                raise ModelError(f"actuator {name} has no ctrlrange")
            low, high = map(float, model.actuator_ctrlrange[i])
            semantic = name.rsplit("_", 1)[-1]
            specs.append(
                ActuatorSpec(
                    index=i,
                    name=name,
                    semantic=semantic,
                    kind=kind,
                    target_name=target_name,
                    low=low,
                    high=high,
                )
            )
        layout = cls(specs=tuple(specs))
        layout.validate()
        return layout

    def validate(self) -> None:
        names = self.names
        if names != EXPECTED_ACTUATOR_NAMES:
            raise ModelError(
                "actuator order does not match the expected Shadow Hand layout:\n"
                f"  expected: {EXPECTED_ACTUATOR_NAMES}\n"
                f"  actual:   {names}"
            )
        low = self.low
        high = self.high
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ModelError("actuator limits must be finite")
        if not np.all(low < high):
            raise ModelError("every actuator must have low < high")
        for spec in self.specs:
            expected = _EXPECTED_KINDS.get(spec.semantic, "joint")
            if spec.kind != expected:
                raise ModelError(f"actuator {spec.name} is a {spec.kind}, expected a {expected}")
            if not spec.target_name.startswith("rh_"):
                raise ModelError(f"actuator {spec.name} targets {spec.target_name!r}")

    def vector(
        self, values: Mapping[str, float], *, fill: float = 0.0, clip: bool = True
    ) -> np.ndarray:
        out = np.full(len(self.specs), fill, dtype=np.float64)
        by_name = self.index
        by_semantic = self.semantic_index
        for key, value in values.items():
            index = by_name.get(key)
            if index is None:
                index = by_semantic.get(key)
            if index is None:
                raise KeyError(f"unknown actuator {key!r}")
            out[index] = float(value)
        if not np.isfinite(out).all():
            raise ValueError("actuator targets must be finite")
        if clip:
            out = np.clip(out, self.low, self.high)
        return out


def load_model(scene_path: str | Path = SCENE_XML) -> tuple[mujoco.MjModel, ActuatorLayout]:
    path = Path(scene_path)
    if not path.exists():
        raise FileNotFoundError(f"MJCF not found: {path}")
    model = mujoco.MjModel.from_xml_path(str(path))
    layout = ActuatorLayout.from_model(model)
    return model, layout


def ctrl_from_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    data: mujoco.MjData | None = None,
) -> np.ndarray:
    """Convert a 24-dimensional keyframe qpos vector to 20 actuator targets.

    Joint actuators evaluate to the joint angle; fixed-tendon actuators evaluate
    to the tendon length, which is the sum of the coupled distal joint angles.
    Using ``data.actuator_length`` keeps this correct without hard-coding sums.
    """
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if qpos.shape != (model.nq,):
        raise ValueError(f"expected qpos of length {model.nq}, got {qpos.shape}")
    scratch = data is not None
    work = data if scratch else mujoco.MjData(model)
    saved = work.qpos.copy() if scratch else None
    try:
        work.qpos[:] = qpos
        mujoco.mj_forward(model, work)
        ctrl = np.array(work.actuator_length, dtype=np.float64)
    finally:
        if scratch and saved is not None:
            work.qpos[:] = saved
            mujoco.mj_forward(model, work)
    return np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def load_keyframes(
    model: mujoco.MjModel,
    path: str | Path = KEYFRAMES_XML,
) -> dict[str, np.ndarray]:
    tree = ET.parse(Path(path))
    poses: dict[str, np.ndarray] = {}
    for key in tree.getroot().findall("keyframe/key"):
        name = key.get("name")
        raw = key.get("qpos")
        if not name or not raw:
            raise ModelError(f"malformed keyframe: {ET.tostring(key, encoding='unicode')}")
        qpos = np.array([float(token) for token in raw.split()], dtype=np.float64)
        poses[name] = ctrl_from_qpos(model, qpos)
    if not poses:
        raise ModelError(f"no keyframes found in {path}")
    return poses


def load_start_pose(
    model: mujoco.MjModel,
    name: str = "open hand",
    path: str | Path = KEYFRAMES_XML,
) -> np.ndarray:
    poses = load_keyframes(model, path)
    try:
        return poses[name]
    except KeyError as exc:
        raise ModelError(f"unknown start pose {name!r}; available: {sorted(poses)}") from exc
