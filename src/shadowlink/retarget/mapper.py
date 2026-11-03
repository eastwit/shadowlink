"""Map hand geometry to Shadow Hand actuator targets.

The hand's rotation is never regressed directly.  Instead we measure physical
quantities that are invariant to how the hand is oriented in front of the
camera -- flexion at each joint, spread in the palm plane, and thumb opposition
-- and map those to the joints of the Shadow Hand.

Targets are written by *semantic* actuator name (``FFJ3``, ``FFJ0``, ...).  The
MJCF is the source of truth for physical ordering, so reordering actuators
cannot silently swap fingers.  ``ActuatorLayout.vector`` does the final lookup
and clips to the control range read from the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shadowlink.config import RetargetConfig
from shadowlink.retarget.calibration import Calibration, Neutral
from shadowlink.retarget.geometry import (
    FINGER_SLOTS,
    SPREAD_SIGN,
    HandMetrics,
    canonicalize,
    hand_metrics,
)
from shadowlink.types import HandObservation

if TYPE_CHECKING:  # avoid a retarget -> sim import at runtime
    from shadowlink.sim.model import ActuatorLayout


class Retargeter:
    """Stateless retargeting with a calibrated neutral pose."""

    def __init__(
        self,
        cfg: RetargetConfig,
        neutral: Neutral,
        layout: ActuatorLayout,
    ) -> None:
        self.cfg = cfg
        self.neutral = neutral
        self.layout = layout

    # ------------------------------------------------------------- geometry
    def metrics(self, obs: HandObservation) -> HandMetrics:
        """Compute hand metrics from an observation's image-aligned keypoints.

        Left hands are mirrored into right-hand geometry first, so every
        downstream formula has one sign convention.
        """
        return hand_metrics(canonicalize(obs.keypoints_3d, obs.apparent_side))

    def calibrate(self, obs: HandObservation) -> Calibration:
        return Calibration.create(
            backend=obs.backend,
            side=obs.anatomical_side.value,
            neutral=Neutral.from_metrics(self.metrics(obs)),
        )

    # -------------------------------------------------------------- mapping
    def map_metrics(self, metrics: HandMetrics) -> np.ndarray:
        """Convert hand metrics to a clipped 20-dimensional actuator target."""
        cfg = self.cfg
        neutral = self.neutral

        def bend(finger: str, joint: int) -> float:
            return max(float(metrics.flex[finger][joint]) - neutral.flex[finger][joint], 0.0)

        values: dict[str, float] = {}
        for finger, slots in FINGER_SLOTS.items():
            spread = float(metrics.spread[finger]) - neutral.spread[finger]
            spread = float(np.clip(spread, -cfg.spread_clip, cfg.spread_clip))
            if abs(spread) < cfg.spread_deadzone:
                spread = 0.0
            values[slots["proximal"]] = cfg.flex_gain * bend(finger, 0)
            values[slots["tendon"]] = cfg.distal_gain * (bend(finger, 1) + bend(finger, 2))
            values[slots["knuckle"]] = SPREAD_SIGN[finger] * cfg.spread_gain * spread

        # Wrist (WRJ1/WRJ2) and THJ3 stay at zero: keypoints live in the camera
        # frame, so wrist motion would mostly reflect arm pose, not hand pose.
        values["THJ5"] = cfg.thumb_opp_gain * (float(metrics.opposition) - neutral.opposition)
        values["THJ4"] = cfg.thumb_base_gain * bend("thumb", 0)
        values["THJ2"] = cfg.thumb_flex_gain * bend("thumb", 1)
        values["THJ1"] = cfg.thumb_flex_gain * cfg.thumb_distal_gain * bend("thumb", 2)

        # LFJ5 (little-finger metacarpal) is intentionally not driven.
        targets = self.layout.vector(values)
        if not np.isfinite(targets).all():
            raise ValueError("retargeter produced a non-finite target")
        return targets

    def map_observation(self, obs: HandObservation) -> np.ndarray:
        return self.map_metrics(self.metrics(obs))

    def neutral_targets(self) -> np.ndarray:
        """The actuator target corresponding to the calibrated neutral pose."""
        return self.layout.vector({})
