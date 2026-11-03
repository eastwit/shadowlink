"""Shadow Hand position control and tactile sensor access."""

from __future__ import annotations

import re
from collections.abc import Mapping

import mujoco
import numpy as np

from shadowlink.sim.model import ActuatorLayout

_FINGER_ORDER = ("ff", "mf", "rf", "lf", "th")
_TAXEL_RE = re.compile(r"touch_(ff|mf|rf|lf|th)_taxel_r(\d+)_c(\d+)")


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
    ) -> None:
        if model.nu != 20:
            raise ValueError(f"expected 20 actuators, got {model.nu}")
        self.model = model
        self.data = data
        self.layout = layout or ActuatorLayout.from_model(model)
        self._touch_sensors: dict[str, int] = {}
        self._taxels: dict[str, list[tuple[int, int, int, int]]] = {}
        self._parse_touch_sensors()

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
