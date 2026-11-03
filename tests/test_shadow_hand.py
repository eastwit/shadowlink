"""Tests for the MuJoCo Shadow Hand wrapper and tactile access."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.paths import SCENE_XML
from shadowlink.sim.model import load_model
from shadowlink.sim.shadow_hand import ShadowHand


@pytest.fixture(scope="module")
def model():
    return load_model(SCENE_XML)[0]


@pytest.fixture
def hand(model):
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return ShadowHand(model, data)


def test_touch_sensors_cover_five_fingers(hand):
    touch = hand.touch
    assert len(touch) == 120
    assert all(np.isfinite(value) for value in touch.values())
    maps = hand.touch_map
    assert sorted(maps) == ["ff", "lf", "mf", "rf", "th"]
    for values in maps.values():
        assert values.shape == (6, 4)
        assert np.isfinite(values).all()
    assert sorted(hand.peak_touch) == ["ff", "lf", "mf", "rf", "th"]


def test_set_targets_from_an_array_clips_to_the_model(hand):
    hand.set_targets(np.full(hand.model.nu, 100.0))
    np.testing.assert_allclose(hand.ctrl, hand.layout.high)
    hand.set_targets(np.full(hand.model.nu, -100.0))
    np.testing.assert_allclose(hand.ctrl, hand.layout.low)


def test_set_targets_from_semantic_names(hand):
    hand.set_targets({"FFJ3": 0.4, "THJ1": 0.2})
    assert hand.ctrl[hand.layout.semantic_index["FFJ3"]] == pytest.approx(0.4)
    assert hand.ctrl[hand.layout.semantic_index["THJ1"]] == pytest.approx(0.2)
    assert np.count_nonzero(hand.ctrl) == 2


def test_invalid_targets_are_rejected(hand):
    with pytest.raises(ValueError):
        hand.set_targets(np.zeros(hand.model.nu - 1))
    with pytest.raises(ValueError):
        hand.set_targets(np.full(hand.model.nu, np.nan))


def test_reset(hand, model):
    hand.set_targets({"FFJ3": 0.5})
    hand.reset()
    np.testing.assert_allclose(hand.ctrl, 0.0)
    hand.reset(np.full(model.nu, 0.1))
    np.testing.assert_allclose(hand.ctrl, 0.1)


def test_touch_colors_track_force(hand):
    sites = hand._color_sites
    before = hand.model.site_rgba[sites].copy()
    hand.data.sensordata[hand._color_addresses] = 0.5
    hand.update_touch_colors(force_scale=1.0)
    after = hand.model.site_rgba[sites]
    assert np.all(after[:, 0] > before[:, 0])
    assert np.all(after[:, 2] < before[:, 2])
    assert np.allclose(after[:, 0], 0.15 + 0.85 * 0.5)


def test_update_touch_colors_validates_the_scale(hand):
    with pytest.raises(ValueError):
        hand.update_touch_colors(0.0)
