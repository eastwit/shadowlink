"""MuJoCo passive viewer lifecycle.

The viewer must be created, synced, and closed from the main thread.  All
methods here are expected to be called by :class:`shadowlink.app.TeleopApp`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from shadowlink.math3d import q_norm, q_rotate

log = logging.getLogger(__name__)

DEFAULT_CAMERA = {
    "lookat": (0.36, -0.015, 0.015),
    "distance": 0.32,
    "azimuth": 210.0,
    "elevation": -30.0,
}

ZERO_AXIS_LENGTH = 0.05
# Lift the marker off the palm along the hand-base +x axis (the palmar side at
# rest) so the RGB triad is not hidden inside the hand mesh.
ZERO_AXIS_OFFSET = np.array([0.06, 0.0, 0.0], dtype=np.float64)
ZERO_AXIS_WIDTH = 0.003
ZERO_AXIS_COLORS = (
    (1.0, 0.25, 0.25, 1.0),  # hand +x, red
    (0.25, 0.9, 0.25, 1.0),  # hand +y, green
    (0.35, 0.55, 1.0, 1.0),  # hand +z, blue
)


@dataclass(frozen=True)
class AxisSegment:
    start: np.ndarray
    end: np.ndarray
    color: tuple[float, float, float, float]


def zero_axis_segments(quat_wxyz, origin) -> list[AxisSegment]:
    """Return the RGB segments that visualize a hand-base zero orientation."""
    quat = q_norm(quat_wxyz)
    base = np.asarray(origin, dtype=np.float64).reshape(3) + q_rotate(quat, ZERO_AXIS_OFFSET)
    segments = []
    for index, color in enumerate(ZERO_AXIS_COLORS):
        axis = np.zeros(3, dtype=np.float64)
        axis[index] = ZERO_AXIS_LENGTH
        segments.append(
            AxisSegment(
                start=base.copy(),
                end=base + q_rotate(quat, axis),
                color=color,
            )
        )
    return segments


class Viewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, enabled: bool = True) -> None:
        self.model = model
        self.data = data
        self.enabled = enabled
        self.error: str | None = None
        self._viewer = None
        self._zero_quat: np.ndarray | None = None
        self._zero_origin: np.ndarray | None = None
        self._zero_drawn = False

    def launch(self) -> None:
        if not self.enabled:
            return
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self._viewer.lock():
                self._viewer.opt.sitegroup[4] = 1
                camera = self._viewer.cam
                camera.lookat[:] = DEFAULT_CAMERA["lookat"]
                camera.distance = DEFAULT_CAMERA["distance"]
                camera.azimuth = DEFAULT_CAMERA["azimuth"]
                camera.elevation = DEFAULT_CAMERA["elevation"]
        except Exception as exc:  # pragma: no cover - depends on GL/window system
            self._viewer = None
            self.error = str(exc)
            log.warning("MuJoCo viewer unavailable (%s); continuing headless", exc)

    @property
    def available(self) -> bool:
        return self._viewer is not None

    @property
    def is_running(self) -> bool:
        if self._viewer is None:
            return False
        return bool(self._viewer.is_running())

    def set_macro_zero(self, quat_wxyz, origin) -> None:
        """Cache the hand orientation and world point used for the zero marker."""
        self._zero_quat = q_norm(quat_wxyz)
        self._zero_origin = np.asarray(origin, dtype=np.float64).reshape(3).copy()

    def clear_macro_zero(self) -> None:
        self._zero_quat = None
        self._zero_origin = None

    def sync(self) -> None:
        if self._viewer is None:
            return
        try:
            with self._viewer.lock():
                self._draw_macro_zero()
        except Exception as exc:  # pragma: no cover - viewer/OpenGL specific
            log.debug("macro zero overlay failed: %s", exc)
        self._viewer.sync()

    def _draw_macro_zero(self) -> None:
        if self._viewer is None:
            return
        scene = self._viewer.user_scn
        if self._zero_quat is None or self._zero_origin is None:
            # Drop a previously drawn triad instead of leaving it on screen.
            if self._zero_drawn:
                scene.ngeom = 0
                self._zero_drawn = False
            return
        segments = zero_axis_segments(self._zero_quat, self._zero_origin)
        if scene.maxgeom < len(segments):
            return
        scene.ngeom = 0
        for index, segment in enumerate(segments):
            geom = scene.geoms[index]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                np.eye(3, dtype=np.float64).reshape(-1),
                np.asarray(segment.color, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                ZERO_AXIS_WIDTH,
                segment.start,
                segment.end,
            )
            geom.rgba[:] = segment.color
        scene.ngeom = len(segments)
        self._zero_drawn = True

    def close(self) -> None:
        viewer, self._viewer = self._viewer, None
        if viewer is not None:
            try:
                viewer.close()
            except Exception as exc:  # pragma: no cover - shutdown race
                log.debug("viewer close failed: %s", exc)

    def __enter__(self) -> Viewer:
        self.launch()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
