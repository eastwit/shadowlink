"""Tests for the offline pose rendering tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from shadowlink.config import AppConfig
from shadowlink.tools import render_pose
from shadowlink.types import Side
from tests.conftest import synthetic_hand
from tests.fakes import FakeBackend, FakeDetector


def _args(**overrides):
    values = dict(
        image="missing.jpg",
        out="out.png",
        backend=None,
        side=None,
        pick=None,
        data_dir=None,
        checkpoint=None,
        hand_model=None,
        calibration=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_missing_image_is_reported_without_touching_models(tmp_path: Path):
    args = _args(image=str(tmp_path / "nope.jpg"), out=str(tmp_path / "out.png"))
    assert render_pose.render_pose_cli(args) == 2
    assert not (tmp_path / "out.png").exists()


def test_describe_covers_every_finger_and_target(retargeter, layout):
    flex = {finger: (0.0, 0.0, 0.0) for finger in ("index", "middle", "ring", "pinky", "thumb")}
    flex["index"] = (0.5, 0.4, 0.3)
    observation = SimpleNamespace(
        keypoints_3d=synthetic_hand(flex=flex),
        apparent_side=Side.RIGHT,
        anatomical_side=Side.RIGHT,
    )
    metrics = retargeter.metrics(observation)
    targets = retargeter.map_metrics(metrics)
    lines = render_pose.describe(metrics, targets, layout)

    text = "\n".join(lines)
    for finger in ("index", "middle", "ring", "pinky", "thumb"):
        assert finger in text
    assert "opposition" in text
    for spec in layout.specs:
        assert f"{spec.semantic}=" in text


def test_observe_image_uses_and_closes_both_stages(monkeypatch):
    keypoints = synthetic_hand()
    detector = FakeDetector()
    backend = FakeBackend(keypoints)
    monkeypatch.setattr(render_pose, "create_detector", lambda cfg, models: detector)
    monkeypatch.setattr(render_pose, "create_backend", lambda cfg, models: backend)

    image = np.zeros((60, 80, 3), dtype=np.uint8)
    observation = render_pose.observe_image(image, AppConfig(), models=None)

    assert observation is not None
    np.testing.assert_allclose(observation.keypoints_3d, keypoints)
    assert detector.calls == 1
    assert backend.calls == 1
    assert detector.closed and backend.closed


def test_observe_image_returns_none_when_no_hand(monkeypatch):
    class NoHandDetector(FakeDetector):
        def detect(self, frame):
            self.calls += 1
            return []

    detector = NoHandDetector()
    backend = FakeBackend(synthetic_hand())
    monkeypatch.setattr(render_pose, "create_detector", lambda cfg, models: detector)
    monkeypatch.setattr(render_pose, "create_backend", lambda cfg, models: backend)

    image = np.zeros((60, 80, 3), dtype=np.uint8)
    assert render_pose.observe_image(image, AppConfig(), models=None) is None
    assert backend.calls == 0
    assert detector.closed and backend.closed


def test_render_target_settles_and_writes_a_png(tmp_path: Path, layout):
    from shadowlink.paths import SCENE_XML
    from shadowlink.sim.model import load_model

    model, _ = load_model(SCENE_XML)
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = tmp_path / "pose.png"
    try:
        path = render_pose.render_target(
            model,
            data,
            layout.vector({}),
            out,
            width=160,
            height=120,
            settle_steps=5,
        )
    except Exception as exc:  # pragma: no cover - headless GL availability
        pytest.skip(f"offscreen rendering unavailable: {exc}")

    assert path == out.resolve()
    image = cv2.imread(str(path))
    assert image is not None
    assert image.shape == (120, 160, 3)
