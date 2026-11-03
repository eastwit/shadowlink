"""Keyframe cycling demo used by ``shadowlink demo`` and smoke tests."""

from __future__ import annotations

import time
from collections.abc import Mapping

import mujoco
import numpy as np

from shadowlink.sim.model import ActuatorLayout
from shadowlink.sim.shadow_hand import ShadowHand
from shadowlink.sim.stepping import FixedStepClock
from shadowlink.sim.viewer import Viewer


def run_pose_demo(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    layout: ActuatorLayout,
    poses: Mapping[str, np.ndarray],
    *,
    viewer: bool = True,
    seconds_per_pose: float = 1.5,
    max_seconds: float = 0.0,
    touch_force_scale: float = 1.0,
    clock=time.monotonic,
    sleep=time.sleep,
) -> int:
    """Cycle through keyframe poses and advance physics.

    Returns the number of poses shown.  This function is intentionally
    dependency-injected (clock/sleep) so tests can run it headless and fast.
    """
    if not poses:
        raise ValueError("no poses supplied")
    if seconds_per_pose <= 0:
        raise ValueError("seconds_per_pose must be positive")

    hand = ShadowHand(model, data, layout)
    names = list(poses)
    viewer_obj = Viewer(model, data, enabled=viewer)
    viewer_obj.launch()
    stepper = FixedStepClock(
        model.opt.timestep,
        max_catchup_s=0.05,
        reset_stall_s=0.5,
    )
    start = clock()
    stepper.reset(start)
    next_switch = start
    index = 0
    shown = 0
    try:
        while True:
            now = clock()
            if max_seconds > 0 and now - start >= max_seconds:
                break
            if viewer_obj.available and not viewer_obj.is_running:
                break
            if now >= next_switch:
                hand.set_targets(poses[names[index]])
                index = (index + 1) % len(names)
                shown += 1
                next_switch = now + seconds_per_pose
            for _ in range(stepper.advance(now)):
                mujoco.mj_step(model, data)
            hand.update_touch_colors(touch_force_scale)
            viewer_obj.sync()
            if max_seconds <= 0 and not viewer_obj.available:
                # Headless and no duration cap would run forever.
                break
            sleep(0.004)
    finally:
        viewer_obj.close()
    return shown
