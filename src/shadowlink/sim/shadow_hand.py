"""Shadow Hand position control and tactile sensor access."""

from __future__ import annotations

import re
from collections.abc import Mapping

import mujoco
import numpy as np

from shadowlink.math3d import q_angle_between, q_conj, q_norm, q_rotate
from shadowlink.sim.model import ActuatorLayout

_FINGER_ORDER = ("ff", "mf", "rf", "lf", "th")
_TAXEL_RE = re.compile(r"touch_(ff|mf|rf|lf|th)_taxel_r(\d+)_c(\d+)")
_PIVOT_BODIES = (
    "rh_palm",
    "rh_ffmiddle",
    "rh_mfmiddle",
    "rh_rfmiddle",
    "rh_lfmiddle",
    "rh_thmiddle",
)
_MACRO_PIVOTS = ("palm", "wrist", "forearm")


class ShadowHand:
    """Thin controller around a compiled Shadow Hand model.

    ``data.ctrl`` is written directly by :meth:`set_targets`; the MJCF position
    actuators do the tracking.  Targets are clipped to the model's ctrl range.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        layout: ActuatorLayout | None = None,
        *,
        macro_pivot: str = "wrist",
    ) -> None:
        if model.nu != 20:
            raise ValueError(f"expected 20 actuators, got {model.nu}")
        if macro_pivot not in _MACRO_PIVOTS:
            raise ValueError(f"macro_pivot must be one of {_MACRO_PIVOTS}, got {macro_pivot!r}")
        self.model = model
        self.data = data
        self._macro_pivot = macro_pivot
        self.layout = layout or ActuatorLayout.from_model(model)
        self._touch_sensors: dict[str, int] = {}
        self._taxels: dict[str, list[tuple[int, int, int, int]]] = {}
        self._parse_touch_sensors()
        self._setup_macro_base()

    def _parse_touch_sensors(self) -> None:
        for sensor_id in range(self.model.nsensor):
            if self.model.sensor_type[sensor_id] != mujoco.mjtSensor.mjSENS_TOUCH:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            if not name:
                raise ValueError(f"touch sensor {sensor_id} has no name")
            match = _TAXEL_RE.fullmatch(name)
            if not match:
                raise ValueError(f"unexpected touch sensor name: {name}")
            finger, row, col = match.groups()
            self._touch_sensors[name] = sensor_id
            self._taxels.setdefault(finger, []).append(
                (
                    int(row),
                    int(col),
                    int(self.model.sensor_adr[sensor_id]),
                    int(self.model.sensor_objid[sensor_id]),
                )
            )
        if not self._touch_sensors:
            raise ValueError("model has no touch sensors")
        for finger in _FINGER_ORDER:
            if finger not in self._taxels:
                raise ValueError(f"missing tactile array for finger {finger}")
        ordered = [t for finger in _FINGER_ORDER for t in self._taxels[finger]]
        self._color_addresses = np.array([t[2] for t in ordered], dtype=np.int64)
        self._color_sites = np.array([t[3] for t in ordered], dtype=np.int64)

    # ----------------------------------------------------------- macro base
    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"model has no body named {name!r}")
        return int(body_id)

    def _setup_macro_base(self) -> None:
        forearm_id = self._body_id("rh_forearm")
        mocap_id = int(self.model.body_mocapid[forearm_id])
        if mocap_id < 0:
            raise ValueError('rh_forearm must have mocap="true" to support macro orientation')
        self._forearm_id = forearm_id
        self._mocap_id = mocap_id
        self._initial_pos = np.array(self.model.body_pos[forearm_id], dtype=np.float64)
        self._initial_quat = q_norm(self.model.body_quat[forearm_id])

        # The pivot is measured once at the rest pose; finger motion afterwards
        # must not move the rotation centre.
        mujoco.mj_forward(self.model, self.data)
        self._pivot_world = self._macro_anchor_world()
        self._pivot_local = q_rotate(
            q_conj(self._initial_quat),
            self._pivot_world - self._initial_pos,
        )
        self.data.mocap_pos[mocap_id] = self._initial_pos
        self.data.mocap_quat[mocap_id] = self._initial_quat

    def _macro_anchor_world(self) -> np.ndarray:
        """Return the world point held fixed by :meth:`set_macro_pose`."""
        if self._macro_pivot == "palm":
            pivot_ids = [self._body_id(name) for name in _PIVOT_BODIES]
            return np.array(
                [self.data.xpos[body_id] for body_id in pivot_ids], dtype=np.float64
            ).mean(axis=0)
        if self._macro_pivot == "wrist":
            return np.array(self.data.xpos[self._body_id("rh_wrist")], dtype=np.float64)
        # The forearm anchor is the proximal end of the forearm mesh (the mocap
        # body origin), which behaves like an elbow-side rotation centre.
        return self._initial_pos.copy()

    def set_macro_pose(self, quat_wxyz: np.ndarray | None) -> bool:
        """Rotate the whole hand about the configured pivot.

        Returns ``True`` when the mocap pose actually changed.  The caller is
        responsible for stepping or calling ``mj_forward`` afterwards.
        """
        if quat_wxyz is None:
            return False
        quat = q_norm(quat_wxyz)
        current = self.data.mocap_quat[self._mocap_id]
        position = self._pivot_world - q_rotate(quat, self._pivot_local)
        if (
            q_angle_between(current, quat) < 1e-9
            and float(np.max(np.abs(position - self.data.mocap_pos[self._mocap_id]))) < 1e-12
        ):
            return False
        self.data.mocap_quat[self._mocap_id] = quat
        self.data.mocap_pos[self._mocap_id] = position
        return True

    def reset_macro_pose(self) -> None:
        self.data.mocap_pos[self._mocap_id] = self._initial_pos
        self.data.mocap_quat[self._mocap_id] = self._initial_quat

    @property
    def macro_pose(self) -> np.ndarray:
        return np.array(self.data.mocap_quat[self._mocap_id], dtype=np.float64)

    @property
    def macro_pivot(self) -> str:
        return self._macro_pivot

    @property
    def macro_pivot_world(self) -> np.ndarray:
        return self._pivot_world.copy()

    # --------------------------------------------------------------- control
    def set_targets(self, targets: np.ndarray | Mapping[str, float]) -> None:
        if isinstance(targets, Mapping):
            vec = self.layout.vector(targets)
        else:
            vec = np.asarray(targets, dtype=np.float64).reshape(-1)
            if vec.shape != (self.model.nu,):
                raise ValueError(f"expected {self.model.nu} targets, got {vec.shape}")
            if not np.isfinite(vec).all():
                raise ValueError("actuator targets must be finite")
            vec = np.clip(vec, self.layout.low, self.layout.high)
        self.data.ctrl[:] = vec

    def reset(self, ctrl: np.ndarray | None = None) -> None:
        if ctrl is None:
            self.data.ctrl[:] = 0.0
        else:
            self.set_targets(ctrl)

    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos.copy()

    @property
    def qvel(self) -> np.ndarray:
        return self.data.qvel.copy()

    @property
    def ctrl(self) -> np.ndarray:
        return self.data.ctrl.copy()

    # ----------------------------------------------------------------- touch
    @property
    def touch(self) -> dict[str, float]:
        return {
            name: float(self.data.sensordata[self.model.sensor_adr[sensor_id]])
            for name, sensor_id in self._touch_sensors.items()
        }

    @property
    def touch_map(self) -> dict[str, np.ndarray]:
        maps: dict[str, np.ndarray] = {}
        for finger in _FINGER_ORDER:
            taxels = self._taxels[finger]
            rows = max(t[0] for t in taxels) + 1
            cols = max(t[1] for t in taxels) + 1
            values = np.full((rows, cols), np.nan, dtype=np.float64)
            for row, col, address, _ in taxels:
                values[row, col] = self.data.sensordata[address]
            maps[finger] = values
        return maps

    @property
    def peak_touch(self) -> dict[str, float]:
        return {finger: float(np.nanmax(values)) for finger, values in self.touch_map.items()}

    def update_touch_colors(self, force_scale: float = 1.0) -> None:
        """Color touch sites from blue (0 N) to red (``force_scale`` N)."""
        if force_scale <= 0:
            raise ValueError("force_scale must be positive")
        levels = np.clip(self.data.sensordata[self._color_addresses] / force_scale, 0.0, 1.0)
        rgba = self.model.site_rgba
        rgba[self._color_sites, 0] = 0.15 + 0.85 * levels
        rgba[self._color_sites, 1] = 0.65 * (1.0 - levels)
        rgba[self._color_sites, 2] = 1.0 - 0.9 * levels
        rgba[self._color_sites, 3] = 0.65
