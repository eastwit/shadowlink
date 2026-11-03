"""Tests for keypoint metrics -> Shadow Hand actuator targets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadowlink.config import RetargetConfig
from shadowlink.retarget.calibration import Neutral
from shadowlink.retarget.geometry import ALL_FINGERS, SPREAD_REST, HandMetrics
from shadowlink.retarget.mapper import Retargeter
from shadowlink.types import Side

ZERO_FLEX = {finger: (0.0, 0.0, 0.0) for finger in ALL_FINGERS}


def make_metrics(flex=None, spread=None, opposition=0.0) -> HandMetrics:
    return HandMetrics(
        flex=dict(ZERO_FLEX if flex is None else flex),
        spread=dict(SPREAD_REST if spread is None else spread),
        opposition=float(opposition),
    )


def by_name(layout, targets):
    return {name: float(targets[index]) for name, index in layout.semantic_index.items()}


def test_proximal_flexion_drives_j3(retargeter, layout):
    flex = {name: (0.0, 0.0, 0.0) for name in ALL_FINGERS}
    flex["index"] = (0.40, 0.0, 0.0)  # neutral flexion is 0.10
    targets = by_name(layout, retargeter.map_metrics(make_metrics(flex=flex)))
    assert targets["FFJ3"] == pytest.approx(0.30, abs=1e-12)
    assert targets["FFJ0"] == pytest.approx(0.0, abs=1e-12)


def test_pip_and_dip_sum_into_the_tendon(retargeter, layout):
    flex = {name: (0.0, 0.0, 0.0) for name in ALL_FINGERS}
    flex["middle"] = (0.40, 0.30, 0.20)
    targets = by_name(layout, retargeter.map_metrics(make_metrics(flex=flex)))
    assert targets["MFJ3"] == pytest.approx(0.30, abs=1e-12)
    assert targets["MFJ0"] == pytest.approx(0.30, abs=1e-12)


def test_spread_signs_are_mirrored_between_finger_groups(retargeter, layout):
    spread = dict(SPREAD_REST)
    spread["index"] += 0.20
    spread["pinky"] += 0.20
    targets = by_name(layout, retargeter.map_metrics(make_metrics(spread=spread)))
    assert targets["FFJ4"] == pytest.approx(+0.14, abs=1e-12)
    assert targets["LFJ4"] == pytest.approx(-0.14, abs=1e-12)


def test_spread_deadzone_suppresses_noise(retargeter, layout):
    spread = dict(SPREAD_REST)
    spread["index"] += 0.02
    targets = by_name(layout, retargeter.map_metrics(make_metrics(spread=spread)))
    assert targets["FFJ4"] == pytest.approx(0.0, abs=1e-12)


def test_spread_is_clipped_before_mapping(retargeter, layout):
    spread = dict(SPREAD_REST)
    spread["index"] += 1.0
    targets = by_name(layout, retargeter.map_metrics(make_metrics(spread=spread)))
    # spread_clip (0.45) * spread_gain (0.7)
    assert targets["FFJ4"] == pytest.approx(0.315, abs=1e-12)


def test_thumb_gains_and_opposition(retargeter, layout):
    flex = {name: (0.0, 0.0, 0.0) for name in ALL_FINGERS}
    flex["thumb"] = (0.40, 0.20, 0.20)
    metrics = make_metrics(flex=flex, opposition=0.5)
    targets = by_name(layout, retargeter.map_metrics(metrics))
    assert targets["THJ4"] == pytest.approx(1.7 * 0.30, abs=1e-12)
    assert targets["THJ2"] == pytest.approx(1.1 * 0.10, abs=1e-12)
    assert targets["THJ1"] == pytest.approx(1.1 * 1.15 * 0.10, abs=1e-12)
    assert targets["THJ5"] == pytest.approx(0.0, abs=1e-12)  # gain is zero


def test_thumb_opposition_gain(layout):
    retargeter = Retargeter(RetargetConfig(thumb_opp_gain=1.0), Neutral.default(), layout)
    targets = by_name(layout, retargeter.map_metrics(make_metrics(opposition=0.4)))
    assert targets["THJ5"] == pytest.approx(0.4, abs=1e-12)


def test_undriven_channels_stay_zero(retargeter, layout):
    flex = {finger: (2.0, 2.0, 2.0) for finger in ALL_FINGERS}
    spread = {finger: 2.0 for finger in SPREAD_REST}
    targets = by_name(
        layout, retargeter.map_metrics(make_metrics(flex=flex, spread=spread, opposition=2.0))
    )
    for name in ("WRJ1", "WRJ2", "THJ3", "LFJ5"):
        assert targets[name] == 0.0


def test_output_is_finite_and_within_limits(retargeter, layout):
    flex = {finger: (10.0, 10.0, 10.0) for finger in ALL_FINGERS}
    spread = {finger: 10.0 for finger in SPREAD_REST}
    targets = retargeter.map_metrics(make_metrics(flex=flex, spread=spread, opposition=10.0))
    assert targets.shape == (20,)
    assert targets.dtype == np.float64
    assert np.isfinite(targets).all()
    assert np.all(targets >= layout.low)
    assert np.all(targets <= layout.high)


def test_neutral_targets_are_zero(retargeter):
    np.testing.assert_allclose(retargeter.neutral_targets(), np.zeros(20), atol=1e-12)


def test_calibration_makes_the_same_pose_neutral(retargeter, layout, make_hand, make_obs):
    hand = make_hand(flex={finger: (0.0, 0.0, 0.0) for finger in ALL_FINGERS})
    obs = make_obs(hand)
    calibration = retargeter.calibrate(obs)
    calibrated = Retargeter(RetargetConfig(), calibration.neutral, layout)
    targets = calibrated.map_observation(obs)
    np.testing.assert_allclose(targets, np.zeros(20), atol=1e-6)


def test_left_hand_matches_right_hand(retargeter, make_hand, make_obs):
    right = make_obs(make_hand(opposition=0.2))
    left_keypoints = make_hand(opposition=0.2)
    left_keypoints[:, 0] *= -1.0
    left = make_obs(left_keypoints, side=Side.LEFT, apparent=Side.LEFT)

    right_metrics = retargeter.metrics(right)
    left_metrics = retargeter.metrics(left)
    for finger in ALL_FINGERS:
        assert left_metrics.flex[finger] == pytest.approx(right_metrics.flex[finger])
    for finger in SPREAD_REST:
        assert left_metrics.spread[finger] == pytest.approx(right_metrics.spread[finger])
    assert left_metrics.opposition == pytest.approx(right_metrics.opposition)
    np.testing.assert_allclose(
        retargeter.map_observation(left), retargeter.map_observation(right), atol=1e-9
    )


def test_metrics_preserve_flexion_range(retargeter, make_hand, make_obs):
    flex = {finger: (0.0, math.pi / 2, math.pi / 2) for finger in ALL_FINGERS}
    metrics = retargeter.metrics(make_obs(make_hand(flex=flex)))
    for finger in ALL_FINGERS:
        assert metrics.flex[finger][1] == pytest.approx(math.pi / 2, abs=1e-9)
        assert metrics.flex[finger][2] == pytest.approx(math.pi / 2, abs=1e-9)
