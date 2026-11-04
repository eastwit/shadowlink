"""Tests that the Shadow Hand model matches the assumptions of the pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.paths import KEYFRAMES_XML, SCENE_XML
from shadowlink.sim.model import (
    EXPECTED_ACTUATOR_NAMES,
    ActuatorLayout,
    ModelError,
    ctrl_from_qpos,
    load_keyframes,
    load_model,
    load_start_pose,
)


@pytest.fixture(scope="module")
def model():
    return load_model(SCENE_XML)[0]


@pytest.fixture(scope="module")
def layout(model):
    return ActuatorLayout.from_model(model)


def test_model_dimensions(model):
    assert model.nq == 24
    assert model.nv == 24
    assert model.nu == 20
    assert model.nmocap == 1
    assert model.nsensor == 120
    assert model.opt.timestep == pytest.approx(0.002)


def test_actuator_names_and_order(layout):
    assert layout.names == EXPECTED_ACTUATOR_NAMES
    assert layout.semantics == (
        "WRJ2",
        "WRJ1",
        "THJ5",
        "THJ4",
        "THJ3",
        "THJ2",
        "THJ1",
        "FFJ4",
        "FFJ3",
        "FFJ0",
        "MFJ4",
        "MFJ3",
        "MFJ0",
        "RFJ4",
        "RFJ3",
        "RFJ0",
        "LFJ5",
        "LFJ4",
        "LFJ3",
        "LFJ0",
    )


def test_tendon_actuators_are_coupled_distal_joints(layout):
    by_semantic = {spec.semantic: spec for spec in layout.specs}
    for semantic in ("FFJ0", "MFJ0", "RFJ0", "LFJ0"):
        spec = by_semantic[semantic]
        assert spec.kind == "tendon"
        assert spec.target_name.startswith("rh_")
    for semantic in ("FFJ3", "MFJ4", "LFJ5", "WRJ2"):
        assert by_semantic[semantic].kind == "joint"


def test_limits_come_from_the_model(layout):
    by_semantic = {spec.semantic: spec for spec in layout.specs}
    # Regression: the old code hard-coded +-0.349066 for LFJ5 while the MJCF
    # declares [0, 0.785398].
    assert by_semantic["LFJ5"].low == pytest.approx(0.0)
    assert by_semantic["LFJ5"].high == pytest.approx(0.785398, abs=1e-6)
    assert np.all(layout.low < layout.high)


def test_vector_accepts_semantic_and_full_names(layout):
    vector = layout.vector({"FFJ3": 1.0, "rh_A_MFJ0": 2.0})
    assert vector.shape == (20,)
    assert vector[layout.semantic_index["FFJ3"]] == pytest.approx(1.0)
    assert vector[layout.semantic_index["MFJ0"]] == pytest.approx(2.0)
    assert np.count_nonzero(vector) == 2


def test_vector_clips_and_validates(layout):
    clipped = layout.vector({"FFJ3": 100.0})
    assert clipped[layout.semantic_index["FFJ3"]] == pytest.approx(
        layout.high[layout.semantic_index["FFJ3"]]
    )
    raw = layout.vector({"FFJ3": 100.0}, clip=False)
    assert raw[layout.semantic_index["FFJ3"]] == pytest.approx(100.0)
    with pytest.raises(KeyError):
        layout.vector({"NOPE": 1.0})
    with pytest.raises(ValueError):
        layout.vector({"FFJ3": float("nan")})


def test_layout_rejects_a_foreign_model():
    import mujoco

    simple = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ModelError):
        ActuatorLayout.from_model(simple)


def test_keyframes_are_complete_and_in_range(model, layout):
    poses = load_keyframes(model, KEYFRAMES_XML)
    assert len(poses) == 13
    assert "open hand" in poses
    for name, ctrl in poses.items():
        assert ctrl.shape == (model.nu,), name
        assert np.isfinite(ctrl).all(), name
        assert np.all(ctrl >= layout.low - 1e-6), name
        assert np.all(ctrl <= layout.high + 1e-6), name


def test_open_hand_starts_at_zero(model):
    ctrl = load_start_pose(model, "open hand", KEYFRAMES_XML)
    np.testing.assert_allclose(ctrl, 0.0, atol=1e-9)


def test_unknown_start_pose_is_rejected(model):
    with pytest.raises(ModelError):
        load_start_pose(model, "not a pose", KEYFRAMES_XML)


def test_ctrl_from_qpos_validates_shape(model):
    with pytest.raises(ValueError):
        ctrl_from_qpos(model, np.zeros(5))
    ctrl = ctrl_from_qpos(model, np.zeros(model.nq))
    assert ctrl.shape == (model.nu,)
