"""Headless end-to-end smoke tests for :class:`TeleopApp`.

These use fake frame sources and fake perception so they run without a camera,
model weights, or a display.  The real MuJoCo model and retargeter are used.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace

import numpy as np
import pytest

from shadowlink.app import TeleopApp
from shadowlink.config import (
    AppConfig,
    CaptureConfig,
    DataConfig,
    PerceptionConfig,
    PhoneConfig,
    PositionConfig,
    RuntimeConfig,
    SimConfig,
    UIConfig,
)
from shadowlink.math3d import q_angle_between, q_conj, q_from_axis_angle, q_mul
from shadowlink.perception.selector import DetectionSelector
from shadowlink.perception.worker import PerceptionService
from shadowlink.phone import NullPhoneSource, PhoneHealth, PhoneSample
from shadowlink.retarget.macro_pose import BACK_MOUNT
from shadowlink.retarget.smoothing import TrackingState
from shadowlink.types import HandDetection, HandObservation, Side
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


def make_rgbd_config(tmp_path, **position) -> AppConfig:
    values = dict(
        enabled=True,
        deadzone_m=0.0,
        temporal_window=1,
        min_cutoff=1e9,
        d_cutoff=1e9,
        beta=0.0,
        max_rate_m_s=1e9,
    )
    values.update(position)
    return replace(
        make_config(tmp_path),
        capture=CaptureConfig(source="rgbd", mirror=False),
        position=PositionConfig(**values),
    )


def palm_landmarks(u: float = 32.0, v: float = 24.0) -> np.ndarray:
    points = np.zeros((21, 2), dtype=np.float64)
    points[:, 0] = u
    points[:, 1] = v
    return points


def depth_ramp(seq: int) -> np.ndarray:
    return np.full((48, 64), 800 + 10 * seq, dtype=np.uint16)


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


class FakePhoneSource:
    """First sample defines the zero; later samples rotate about X."""

    def __init__(self, clock) -> None:
        self.clock = clock
        self.calls = 0
        self.started = False
        self.closed = False
        self.target = q_from_axis_angle([1.0, 0.0, 0.0], math.radians(40.0))

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def latest(self):
        self.calls += 1
        quat = np.array([1.0, 0.0, 0.0, 0.0]) if self.calls == 1 else self.target
        return PhoneSample(
            quat_wxyz=quat,
            timestamp=self.clock(),
            seq=self.calls,
            ready=True,
        )

    @property
    def health(self):
        return PhoneHealth(
            state="running",
            message="following",
            host="192.0.2.10:8080",
            measuring=True,
            rate_hz=30.0,
            kind="attitude",
        )


def test_macro_follow_tracks_the_phone_and_stays_frozen_on_close(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(30, clock=clock)
    phone_source = FakePhoneSource(clock)
    config = replace(
        make_config(tmp_path),
        phone=PhoneConfig(auto_follow=True),
    )
    app = TeleopApp(
        config,
        source=source,
        perception=FakePerception(clock=clock),
        phone_source=phone_source,
        clock=clock,
    )
    initial = app.hand.macro_pose.copy()
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert phone_source.started and phone_source.closed
    assert q_angle_between(initial, app.hand.macro_pose) > math.radians(10.0)
    frozen = app.hand.macro_pose.copy()
    app.close()
    np.testing.assert_allclose(app.hand.macro_pose, frozen)
    assert app_threads() == []


def test_zero_phone_aligns_the_hand_to_the_phone_attitude(tmp_path):
    clock = StepClock()
    phone_source = FakePhoneSource(clock)
    app = TeleopApp(
        make_config(tmp_path),
        source=FakeSource(1, clock=clock),
        perception=FakePerception(clock=clock),
        phone_source=phone_source,
        clock=clock,
    )
    try:
        sample = app.orientation.latest()
        app._enable_follow(sample)
        # Move the hand somewhere unrelated; zeroing must not preserve it.
        assert app.hand.set_macro_pose(q_from_axis_angle([0.0, 0.0, 1.0], 0.6))

        app.following = False
        assert app._zero_phone()
        assert app.following is True
        assert "world frame" in app._last_status

        sample = app.orientation.latest()
        expected = q_mul(sample.quat_wxyz, q_conj(BACK_MOUNT))
        np.testing.assert_allclose(app.macro.zero_hand, expected, atol=1e-6)
        np.testing.assert_allclose(app.macro.orientation, expected, atol=1e-6)
    finally:
        app.close()


def test_zero_phone_key_is_dispatched(tmp_path, monkeypatch):
    clock = StepClock(step_s=0.02)
    app = TeleopApp(
        make_config(tmp_path),
        source=FakeSource(20, clock=clock),
        perception=FakePerception(clock=clock),
        phone_source=FakePhoneSource(clock),
        clock=clock,
    )
    calls = {"count": 0}

    def render(*args, **kwargs):
        calls["count"] += 1
        return ord("0") if calls["count"] == 5 else None

    monkeypatch.setattr(app, "_render", render)
    try:
        code = app.run(max_frames=10)
    finally:
        app.close()

    assert code == 0
    assert app.following is True
    assert "world frame" in app._last_status


class RecordingViewer:
    """Stands in for :class:`Viewer` so the zero overlay can be inspected."""

    def __init__(self) -> None:
        self.zeros = []
        self.syncs = 0
        self.closed = False

    @property
    def available(self) -> bool:
        return False

    def launch(self) -> None:
        pass

    def set_macro_zero(self, quat, origin) -> None:
        self.zeros.append((np.array(quat), np.array(origin)))

    def sync(self) -> None:
        self.syncs += 1

    def close(self) -> None:
        self.closed = True


def test_app_feeds_the_visible_zero_pose_to_the_viewer(tmp_path):
    clock = StepClock(step_s=0.02)
    config = replace(make_config(tmp_path), phone=PhoneConfig(auto_follow=True))
    app = TeleopApp(
        config,
        source=FakeSource(20, clock=clock),
        perception=FakePerception(clock=clock),
        phone_source=FakePhoneSource(clock),
        clock=clock,
    )
    viewer = RecordingViewer()
    app.viewer = viewer
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert viewer.syncs > 0
    assert viewer.closed
    assert viewer.zeros
    quat, origin = viewer.zeros[-1]
    np.testing.assert_allclose(quat, app.macro.zero_hand)
    np.testing.assert_allclose(origin, app.hand.macro_origin_world)


def test_zero_phone_requires_a_ready_sample(tmp_path):
    app = TeleopApp(
        make_config(tmp_path),
        source=FakeSource(1),
        perception=FakePerception(),
        phone_source=NullPhoneSource(),
    )
    try:
        assert app._zero_phone() is False
        assert app.following is False
        assert "cannot zero" in app._last_status
    finally:
        app.close()


def test_phone_pivot_reaches_the_hand(tmp_path):
    config = replace(make_config(tmp_path), phone=PhoneConfig(pivot="forearm"))
    app = TeleopApp(
        config,
        source=FakeSource(1),
        perception=FakePerception(),
    )
    try:
        assert app.hand.macro_pivot == "forearm"
        np.testing.assert_allclose(app.hand.macro_pivot_world, [0.0, 0.0, 0.0], atol=1e-9)
    finally:
        app.close()


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


def test_rgbd_depth_drives_a_bounded_translation(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(15, clock=clock, depth_factory=depth_ramp)
    perception = FakePerception(landmarks_2d=palm_landmarks(), clock=clock)
    app = TeleopApp(
        make_rgbd_config(tmp_path),
        source=source,
        perception=perception,
        clock=clock,
    )
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    translation = app.hand.macro_translation
    assert np.isfinite(translation).all()
    assert np.any(np.abs(translation) > 1e-3)
    assert np.max(np.abs(translation)) <= app.config.position.max_offset_m + 1e-9
    # The palm preset maps depth away from the camera to world -z (the palm
    # normal at rest is +z).
    assert translation[2] < 0.0
    assert app.position.state == "tracking"
    assert app_threads() == []


def test_3d_landmarks_make_the_translation_hand_relative(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(15, clock=clock, depth_factory=depth_ramp)
    perception = FakePerception(
        keypoints=synthetic_hand(),
        landmarks_2d=palm_landmarks(),
        clock=clock,
    )
    app = TeleopApp(
        make_rgbd_config(tmp_path),
        source=source,
        perception=perception,
        clock=clock,
    )
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    translation = app.hand.macro_translation
    # The synthetic palm normal is +z, so motion away from the camera (the
    # depth ramp) pushes the simulated hand along world +z.  The static palm
    # fallback would have produced -z for the same motion.
    assert translation[2] > 0.0
    assert app.position.state == "tracking"


def test_hand_relative_mapping_can_be_disabled(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(15, clock=clock, depth_factory=depth_ramp)
    perception = FakePerception(
        keypoints=synthetic_hand(),
        landmarks_2d=palm_landmarks(),
        clock=clock,
    )
    app = TeleopApp(
        make_rgbd_config(tmp_path, hand_relative=False),
        source=source,
        perception=perception,
        clock=clock,
    )
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert app.hand.macro_translation[2] < 0.0


def test_the_detector_view_updates_position_before_inference_finishes(tmp_path):
    clock = StepClock(step_s=0.02)
    source = FakeSource(1, clock=clock, depth_mm=800)
    app = TeleopApp(
        make_rgbd_config(tmp_path),
        source=source,
        perception=FakePerception(clock=clock),
        clock=clock,
    )
    try:
        frame = source.read()
        assert frame is not None
        app._depth_log.add(frame)
        detection = HandDetection(
            bbox_xyxy=np.array([10.0, 10.0, 50.0, 40.0]),
            apparent_side=Side.RIGHT,
            anatomical_side=Side.RIGHT,
            landmarks_2d=palm_landmarks(u=16.0),
            frame_seq=frame.seq,
        )
        app.perception.latest_detection = lambda: (frame, detection)
        observation = HandObservation(
            keypoints_3d=np.zeros((21, 3)),
            apparent_side=Side.RIGHT,
            anatomical_side=Side.RIGHT,
            score=1.0,
            frame_seq=frame.seq,
            capture_ts=frame.capture_ts,
            inference_ts=clock(),
            backend="fake",
            keypoints_2d=palm_landmarks(u=48.0),
        )

        measurement = app._palm_measurement(observation, clock())

        assert measurement is not None
        # The detector landmark (u=16) is left of the observation fallback
        # (u=48); the sample must use the detector's pixel.
        expected = (16.0 - 31.5) * 0.8 / (0.9 * 64.0)
        assert measurement.xyz_m[0] == pytest.approx(expected, rel=1e-6)
    finally:
        app.close()


def test_position_rotation_reuses_the_last_anatomical_axes(tmp_path):
    clock = StepClock(step_s=0.02)
    app = TeleopApp(
        make_rgbd_config(tmp_path),
        source=FakeSource(1, clock=clock),
        perception=FakePerception(clock=clock),
        clock=clock,
    )
    try:
        assert app._position_rotation(None) is None
        observation = HandObservation(
            keypoints_3d=synthetic_hand(),
            apparent_side=Side.RIGHT,
            anatomical_side=Side.RIGHT,
            score=1.0,
            frame_seq=0,
            capture_ts=0.0,
            inference_ts=0.0,
            backend="fake",
        )
        first = app._position_rotation(observation)
        assert first is not None
        # Between inferences the last 3D axes still give a hand-relative frame.
        np.testing.assert_allclose(app._position_rotation(None), first)
    finally:
        app.close()


def test_phone_rotation_and_depth_translation_are_independent(tmp_path):
    clock = StepClock(step_s=0.02)
    config = replace(make_rgbd_config(tmp_path), phone=PhoneConfig(auto_follow=True))
    source = FakeSource(20, clock=clock, depth_factory=depth_ramp)
    app = TeleopApp(
        config,
        source=source,
        perception=FakePerception(landmarks_2d=palm_landmarks(), clock=clock),
        phone_source=FakePhoneSource(clock),
        clock=clock,
    )
    initial = app.hand.macro_pose.copy()
    try:
        code = app.run()
    finally:
        app.close()

    assert code == 0
    assert np.any(np.abs(app.hand.macro_translation) > 1e-3)
    assert q_angle_between(initial, app.hand.macro_pose) > math.radians(10.0)


def test_position_toggle_freezes_and_rezeroes(tmp_path):
    clock = StepClock(step_s=0.02)
    app = TeleopApp(
        make_rgbd_config(tmp_path),
        source=FakeSource(1, clock=clock, depth_factory=depth_ramp),
        perception=FakePerception(landmarks_2d=palm_landmarks(), clock=clock),
        clock=clock,
    )
    try:
        assert app.positioning is True
        assert app._toggle_position() is True
        assert app.positioning is False
        assert "frozen" in app._last_status
        assert app._toggle_position() is True
        assert app.positioning is True
        assert "position follow on" in app._last_status
    finally:
        app.close()


def test_position_toggle_explains_that_rgbd_is_required(tmp_path):
    app = TeleopApp(
        make_config(tmp_path),
        source=FakeSource(1),
        perception=FakePerception(),
    )
    try:
        assert app.position is None
        assert app._toggle_position() is False
        assert "rgbd" in app._last_status
    finally:
        app.close()


def test_position_key_is_dispatched(tmp_path, monkeypatch):
    clock = StepClock(step_s=0.02)
    app = TeleopApp(
        make_rgbd_config(tmp_path),
        source=FakeSource(20, clock=clock, depth_factory=depth_ramp),
        perception=FakePerception(landmarks_2d=palm_landmarks(), clock=clock),
        clock=clock,
    )
    calls = {"count": 0}

    def render(*args, **kwargs):
        calls["count"] += 1
        return ord("p") if calls["count"] == 5 else None

    monkeypatch.setattr(app, "_render", render)
    try:
        code = app.run(max_frames=10)
    finally:
        app.close()

    assert code == 0
    assert app.positioning is False
    assert "frozen" in app._last_status


def test_reset_clears_the_position_layer(tmp_path):
    clock = StepClock(step_s=0.02)
    app = TeleopApp(
        make_rgbd_config(tmp_path),
        source=FakeSource(5, clock=clock, depth_factory=depth_ramp),
        perception=FakePerception(landmarks_2d=palm_landmarks(), clock=clock),
        clock=clock,
    )
    try:
        code = app.run()
        assert code == 0
        app._reset()
        np.testing.assert_allclose(app.hand.macro_translation, 0.0)
        assert app.position.state == "waiting"
        assert app.positioning is True
        assert len(app._depth_log) == 0
    finally:
        app.close()
