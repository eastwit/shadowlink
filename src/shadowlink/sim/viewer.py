"""MuJoCo passive viewer lifecycle.

The viewer must be created, synced, and closed from the main thread.  All
methods here are expected to be called by :class:`shadowlink.app.TeleopApp`.
"""

from __future__ import annotations

import logging

import mujoco

log = logging.getLogger(__name__)

DEFAULT_CAMERA = {
    "lookat": (0.36, -0.015, 0.015),
    "distance": 0.32,
    "azimuth": 210.0,
    "elevation": -30.0,
}


class Viewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, enabled: bool = True) -> None:
        self.model = model
        self.data = data
        self.enabled = enabled
        self.error: str | None = None
        self._viewer = None

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

    def sync(self) -> None:
        if self._viewer is not None:
            self._viewer.sync()

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
