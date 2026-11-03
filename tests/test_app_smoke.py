"""Headless end-to-end smoke tests for :class:`TeleopApp`.

These use fake frame sources and fake perception so they run without a camera,
model weights, or a display.  The real MuJoCo model and retargeter are used.
"""

from __future__ import annotations

import threading

import numpy as np

from shadowlink.app import TeleopApp
from shadowlink.config import (
    AppConfig,
    DataConfig,
    PerceptionConfig,
    RuntimeConfig,
    SimConfig,
    UIConfig,
)
from shadowlink.perception.selector import DetectionSelector
from shadowlink.perception.worker import PerceptionService
from shadowlink.retarget.smoothing import TrackingState
from tests.conftest import synthetic_hand
from tests.fakes import FakeBackend, FakeDetector, FakePerception, FakeSource, StepClock


def make_config(tmp_path, *, max_frames: int = 0, **perception) -> AppConfig:
    return AppConfig(
        perception=PerceptionConfig(**perception),
        sim=SimConfig(viewer=False),
        ui=UIConfig(show=False),
        data=DataConfig(calibration=str(tmp_path / "calibration.json")),
        runtime=RuntimeConfig(max_frames=max_frames),
    )


def app_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("shadowlink-") and thread.is_alive()
    ]


def closed_fist() -> np.ndarray:
    flex = {name: (1.0, 1.0, 1.0) for name in ("thumb", "index", "middle", "ring", "pinky")}
    return synthetic_hand(flex=flex)


def test_no_hand_stays_neutral_and_cleans_up(tmp_path):
    clock = StepClock()
    source = FakeSource(30, clock=clock)
    perception = FakePerception(clock=clock)
    app = TeleopApp(make_config(tmp_path), source=source, perception=perception, clock=clock)
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert source.closed
    assert perception.closed
    assert app.controller.state is TrackingState.WAITING
    np.testing.assert_allclose(app.hand.ctrl, 0.0, atol=1e-9)
    assert app.hand.ctrl.shape == (app.model.nu,)
    assert app_threads() == []


def test_a_tracked_hand_drives_bounded_actuator_targets(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(60, clock=clock)
    perception = FakePerception(keypoints=closed_fist(), clock=clock)
    app = TeleopApp(make_config(tmp_path), source=source, perception=perception, clock=clock)
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert app.controller.state is TrackingState.TRACKING
    assert app.hand.ctrl.shape == (app.model.nu,)
    assert np.isfinite(app.hand.ctrl).all()
    assert np.all(app.hand.ctrl >= app.layout.low - 1e-9)
    assert np.all(app.hand.ctrl <= app.layout.high + 1e-9)
    assert app.hand.ctrl.max() > 0.1
    assert app.data.time > 0.0
    assert app_threads() == []


def test_real_perception_service_runs_inside_the_app(tmp_path):
    clock = StepClock(step_s=0.01)
    source = FakeSource(80, clock=clock)
    detector = FakeDetector()
    backend = FakeBackend(closed_fist())
    service = PerceptionService(
        detector,
        backend,
        DetectionSelector("largest"),
        clock=clock,
    )
    app = TeleopApp(
        make_config(tmp_path, max_age_s=1.0),
        source=source,
        perception=service,
        clock=clock,
    )
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert app.controller.state is TrackingState.TRACKING
    assert app.data.time > 0.0
    assert detector.calls > 0
    assert backend.calls > 0
    assert detector.closed
    assert backend.closed
    assert service.health.state == "stopped"
    assert app_threads() == []


def test_frame_limit_stops_the_run(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(100, clock=clock)
    app = TeleopApp(
        make_config(tmp_path, max_frames=10),
        source=source,
        perception=FakePerception(clock=clock),
        clock=clock,
    )
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert source.produced == 10
    assert source.closed
