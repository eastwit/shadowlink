"""Headless end-to-end smoke tests for :class:`TeleopApp`.

These use fake frame/perception/tag sources so they run without a camera,
model weights, or a display. The real MuJoCo model and retargeter are used.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace

import numpy as np

from shadowlink.app import TeleopApp
from shadowlink.config import (
    AppConfig,
    CaptureConfig,
    DataConfig,
    PerceptionConfig,
    PositionConfig,
    RuntimeConfig,
    SimConfig,
    TagConfig,
    UIConfig,
)
from shadowlink.math3d import q_angle_between, q_from_axis_angle, q_rotate
from shadowlink.perception.selector import DetectionSelector
from shadowlink.perception.worker import PerceptionService
from shadowlink.retarget.smoothing import TrackingState
from shadowlink.tag import TagHealth, TagSample
from tests.conftest import synthetic_hand
from tests.fakes import FakeBackend, FakeDetector, FakePerception, FakeSource, StepClock


def make_config(tmp_path, *, max_frames: int = 0, **perception) -> AppConfig:
    return AppConfig(
        perception=PerceptionConfig(**perception),
        sim=SimConfig(viewer=False),
        ui=UIConfig(show=False),
        data=DataConfig(calibration=str(tmp_path / "calibration.json")),
        position=PositionConfig(enabled=False),
        tag=TagConfig(camera="0", auto_follow=True),
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


class FakeTagSource:
    """Latest-value tag source with a moving pose, or no tag when disabled."""

    def __init__(self, clock, *, ready: bool = True, moving: bool = True) -> None:
        self._clock = clock
        self._ready = ready
        self._moving = moving
        self.started = False
        self.closed = False
        self.calls = 0

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def latest(self) -> TagSample | None:
        if not self._ready:
            return None
        self.calls += 1
        n = max(0, self.calls - 3) if self._moving else 0
        return TagSample(
            translation_m=[0.10 + 0.002 * n, 0.20, 0.80],
            quat_wxyz=q_from_axis_angle([0.0, 0.0, 1.0], math.radians(2.0 * n)),
            timestamp=self._clock(),
            seq=self.calls,
            tag_id=5,
        )

    @property
    def health(self) -> TagHealth:
        if not self._ready:
            return TagHealth(state="searching", camera="0", api="dshow")
        return TagHealth(state="running", camera="0", api="dshow", rate_hz=20.0, tag_id=5)


def make_tag_config(tmp_path, *, max_frames: int = 0) -> AppConfig:
    return replace(
        make_config(tmp_path, max_frames=max_frames),
        capture=CaptureConfig(camera="1"),
        position=PositionConfig(
            enabled=True,
            preset="front",
            deadzone_m=0.0,
            temporal_window=1,
            min_cutoff=1e9,
            d_cutoff=1e9,
            beta=0.0,
            max_rate_m_s=1e9,
            auto_zero=True,
        ),
        tag=TagConfig(
            camera="0",
            zero_mode="sample",
            auto_follow=True,
            rotation_smoothing_tau_s=0.0,
            rotation_deadzone_deg=0.0,
            rotation_max_rate_rad_s=1e6,
        ),
    )


def test_initial_pose_is_palm_down_with_middle_finger_toward_the_door(tmp_path):
    clock = StepClock()
    app = TeleopApp(
        make_config(tmp_path),
        source=FakeSource(1, clock=clock),
        perception=FakePerception(clock=clock),
        tag_source=FakeTagSource(clock, ready=False),
        clock=clock,
    )
    try:
        np.testing.assert_allclose(
            q_rotate(app.hand.macro_pose, [1.0, 0.0, 0.0]),
            [0.0, 0.0, -1.0],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            q_rotate(app.hand.macro_pose, [0.0, 0.0, 1.0]),
            [1.0, 0.0, 0.0],
            atol=1e-6,
        )
        app._reset()
        np.testing.assert_allclose(
            q_rotate(app.hand.macro_pose, [1.0, 0.0, 0.0]),
            [0.0, 0.0, -1.0],
            atol=1e-6,
        )
    finally:
        app.close()


def test_no_hand_stays_neutral_and_cleans_up(tmp_path):
    clock = StepClock()
    source = FakeSource(30, clock=clock)
    perception = FakePerception(clock=clock)
    tag = FakeTagSource(clock, ready=False)
    app = TeleopApp(
        make_config(tmp_path),
        source=source,
        perception=perception,
        tag_source=tag,
        clock=clock,
    )
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert source.closed
    assert perception.closed
    assert tag.started and tag.closed
    assert app.controller.state is TrackingState.WAITING
    np.testing.assert_allclose(app.hand.ctrl, 0.0, atol=1e-9)
    assert app.hand.ctrl.shape == (app.model.nu,)
    assert app_threads() == []


def test_a_tracked_hand_drives_bounded_actuator_targets(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(60, clock=clock)
    perception = FakePerception(keypoints=closed_fist(), clock=clock)
    app = TeleopApp(
        make_config(tmp_path),
        source=source,
        perception=perception,
        tag_source=FakeTagSource(clock, ready=False),
        clock=clock,
    )
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
        tag_source=FakeTagSource(clock, ready=False),
        clock=clock,
    )
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert app.controller.state is TrackingState.TRACKING
    assert app.data.time > 0.0


def test_frame_limit_stops_the_run(tmp_path):
    clock = StepClock(step_s=0.01)
    source = FakeSource(100, clock=clock)
    app = TeleopApp(
        make_config(tmp_path, max_frames=5),
        source=source,
        perception=FakePerception(clock=clock),
        tag_source=FakeTagSource(clock, ready=False),
        clock=clock,
    )
    try:
        assert app.run() == 0
    finally:
        app.close()
    assert source.produced == 5


def test_tag_source_drives_whole_hand_position_and_orientation(tmp_path):
    clock = StepClock(step_s=0.01)
    tag = FakeTagSource(clock)
    app = TeleopApp(
        make_tag_config(tmp_path),
        source=FakeSource(60, clock=clock),
        perception=FakePerception(clock=clock),
        tag_source=tag,
        clock=clock,
    )
    initial_hand = app.hand.macro_pose.copy()
    try:
        assert app.run() == 0
        assert tag.started and tag.closed
        assert app.tag_following is True
        assert app.tag_positioning is True
        assert np.linalg.norm(app.hand.macro_translation) > 1e-3
        assert q_angle_between(initial_hand, app.hand.macro_pose) > math.radians(10.0)
        assert app.tag_pose.position_state == "tracking"
        assert app.tag_pose.orientation_state == "tracking"
    finally:
        app.close()
    assert app_threads() == []


def test_tag_translation_and_orientation_can_be_toggled_independently(tmp_path):
    clock = StepClock(step_s=0.01)
    app = TeleopApp(
        make_tag_config(tmp_path),
        source=FakeSource(2, clock=clock),
        perception=FakePerception(clock=clock),
        tag_source=FakeTagSource(clock),
        clock=clock,
    )
    try:
        assert app._toggle_tag_follow() is True
        assert app.tag_following is True
        zero = app.tag_pose.zero_hand.copy()
        assert app._toggle_tag_follow() is True
        assert app.tag_following is False
        assert app._toggle_tag_follow() is True
        np.testing.assert_allclose(app.tag_pose.zero_hand, zero, atol=1e-12)

        assert app._toggle_position() is True
        assert app.tag_positioning is False
        assert "frozen" in app._last_status
        assert app._toggle_position() is True
        assert app.tag_positioning is True
        assert "position follow on" in app._last_status
    finally:
        app.close()


def test_reset_clears_the_tag_pose(tmp_path):
    clock = StepClock(step_s=0.01)
    app = TeleopApp(
        make_tag_config(tmp_path),
        source=FakeSource(20, clock=clock),
        perception=FakePerception(clock=clock),
        tag_source=FakeTagSource(clock),
        clock=clock,
    )
    try:
        assert app.run() == 0
        assert np.any(np.abs(app.hand.macro_translation) > 1e-3)
        app._reset()
        np.testing.assert_allclose(app.hand.macro_translation, 0.0, atol=1e-9)
        assert app.tag_pose.position_state == "waiting"
        assert app.tag_positioning is True
        assert app.tag_following is False
    finally:
        app.close()
