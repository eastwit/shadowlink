"""Application orchestration: capture -> perception -> retarget -> MuJoCo."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import replace

import mujoco

from shadowlink.capture import open_source
from shadowlink.config import AppConfig, ConfigError
from shadowlink.math3d import q_euler_deg
from shadowlink.paths import (
    KEYFRAMES_XML,
    SCENE_XML,
    ModelPaths,
    resolve_calibration_path,
    resolve_output_path,
)
from shadowlink.perception.base import create_backend, create_detector
from shadowlink.perception.selector import DetectionSelector
from shadowlink.perception.worker import PerceptionService
from shadowlink.phone import NullPhoneSource, PhyphoxService
from shadowlink.phone.client import parse_host_port
from shadowlink.retarget.calibration import CalibrationError, CalibrationStore, Neutral
from shadowlink.retarget.geometry import HandGeometryError
from shadowlink.retarget.macro_pose import MacroPoseController, MacroPoseSettings
from shadowlink.retarget.mapper import Retargeter
from shadowlink.retarget.smoothing import TargetController
from shadowlink.sim.model import load_model, load_start_pose
from shadowlink.sim.shadow_hand import ShadowHand
from shadowlink.sim.stepping import FixedStepClock
from shadowlink.sim.viewer import Viewer
from shadowlink.types import HandObservation, PerceptionResult
from shadowlink.ui.overlay import draw_hud, draw_observation
from shadowlink.ui.telemetry import Telemetry
from shadowlink.ui.window import PreviewWindow, RecordingError, VideoRecorder

log = logging.getLogger(__name__)

CALIBRATION_MAX_AGE_S = 0.25


class TeleopApp:
    def __init__(
        self,
        config: AppConfig,
        *,
        source=None,
        perception=None,
        phone_source=None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._clock = clock
        self._sleeper = sleeper

        self.model, self.layout = load_model(SCENE_XML)
        self.data = mujoco.MjData(self.model)
        self.hand = ShadowHand(
            self.model,
            self.data,
            self.layout,
            macro_pivot=config.phone.pivot,
        )
        self.start_ctrl = load_start_pose(self.model, config.sim.start_pose, KEYFRAMES_XML)
        self.hand.reset(self.start_ctrl)
        mujoco.mj_forward(self.model, self.data)

        self.macro = MacroPoseController(
            MacroPoseSettings.from_config(config.phone),
            self.hand.macro_pose,
        )
        self.following = False
        self._auto_follow_armed = bool(config.phone.auto_follow)
        self._phone_enabled = config.phone.source != "none"
        if phone_source is not None:
            self.orientation = phone_source
            self._phone_enabled = True
        elif config.phone.source == "phyphox":
            self.orientation = PhyphoxService(config.phone, clock=clock)
        else:
            self.orientation = NullPhoneSource()

        self.calibration_store = CalibrationStore(resolve_calibration_path(config.data.calibration))
        calibration = self._load_calibration()
        neutral = calibration.neutral if calibration is not None else Neutral.default()
        self.retargeter = Retargeter(config.retarget, neutral, self.layout)
        self.controller = TargetController(config.smoothing, self.retargeter.neutral_targets())

        self.stepper = FixedStepClock(
            self.model.opt.timestep,
            max_catchup_s=config.sim.max_catchup_s,
            reset_stall_s=config.sim.reset_stall_s,
        )

        self.source = source if source is not None else open_source(config.capture, clock=clock)
        if perception is None:
            models = ModelPaths.resolve(
                config.data.data_dir,
                checkpoint=config.data.checkpoint,
                hand_model=config.data.hand_model,
            )
            detector = create_detector(config.perception, models)
            backend = create_backend(config.perception, models)
            perception = PerceptionService(
                detector,
                backend,
                DetectionSelector(config.perception.pick),
                side=config.perception.side,
                pipeline=config.perception.pipeline,
                clock=clock,
            )
        self.perception = perception

        self.viewer = Viewer(self.model, self.data, enabled=config.sim.viewer)
        self.preview = PreviewWindow(config.ui.window_name, enabled=config.ui.show)
        self.telemetry = Telemetry()
        self.recorder: VideoRecorder | None = None
        if config.ui.record:
            fps = self.source.fps or 30.0
            self.recorder = VideoRecorder(resolve_output_path(config.ui.record), fps)

        self._closed = False
        self._last_status = ""

    # ------------------------------------------------------------------ setup
    def _load_calibration(self):
        try:
            calibration = self.calibration_store.load(self.config.perception.backend)
        except CalibrationError as exc:
            log.warning("ignoring calibration: %s", exc)
            return None
        if calibration is None:
            return None
        if (
            self.config.perception.side in {"left", "right"}
            and calibration.side
            and calibration.side != self.config.perception.side
        ):
            log.warning(
                "calibration was captured for the %s hand but --side=%s is configured",
                calibration.side,
                self.config.perception.side,
            )
        log.info(
            "loaded calibration %s (backend=%s, side=%s)",
            self.calibration_store.path,
            calibration.backend,
            calibration.side,
        )
        return calibration

    # -------------------------------------------------------------------- run
    def run(self, max_frames: int | None = None) -> int:
        frames = 0
        limit = max_frames or self.config.runtime.max_frames or 0
        self.perception.start()
        self.orientation.start()
        self.viewer.launch()
        self.stepper.reset(self._clock())
        self.controller.reset()
        try:
            while True:
                loop_start = self._clock()
                if self._viewer_closed():
                    break
                frame = self.source.read()
                if frame is None:
                    break
                if frame.seq % max(1, self.config.perception.skip) == 0:
                    self.perception.submit(frame)

                now = self._clock()
                health = self.perception.health
                if health.failed:
                    log.error("perception worker failed: %s", health.message)
                    return 1

                result = self.perception.latest_result()
                observation = self._fresh_observation(result, now)
                target = None
                if observation is not None:
                    try:
                        target = self.retargeter.map_observation(observation)
                    except HandGeometryError as exc:
                        log.debug("skipping degenerate observation: %s", exc)

                targets = self.controller.update(target, now)
                self.hand.set_targets(targets)

                sample = self.orientation.latest()
                if (
                    self._auto_follow_armed
                    and not self.following
                    and sample is not None
                    and sample.ready
                ):
                    self._auto_follow_armed = False
                    self._enable_follow(sample)
                macro_changed = False
                if self.following:
                    if sample is None:
                        self.following = False
                        self._last_status = "phone lost; macro pose frozen (press 0 or g)"
                    else:
                        quat = self.macro.update(sample, now, True)
                        if quat is not None:
                            macro_changed = self.hand.set_macro_pose(quat)

                steps = self.stepper.advance(now)
                if steps:
                    for _ in range(steps):
                        mujoco.mj_step(self.model, self.data)
                elif macro_changed:
                    mujoco.mj_forward(self.model, self.data)
                self.hand.update_touch_colors(self.config.sim.touch_force_scale)
                if self._phone_enabled:
                    self.viewer.set_macro_zero(
                        self.macro.zero_hand,
                        self.hand.macro_pivot_world,
                    )
                self.viewer.sync()

                self.telemetry.tick(now)
                self.telemetry.update(result, now)
                key = self._render(frame, observation, result, now)

                frames += 1
                if key in {27, ord("q")}:
                    break
                if key == ord("c"):
                    self._calibrate_with(observation, now)
                elif key == ord("g"):
                    self._toggle_follow()
                elif key == ord("0"):
                    self._zero_phone()
                elif key == ord("r"):
                    self._reset()
                if limit and frames >= limit:
                    break
                self.telemetry.add_process_time(self._clock() - loop_start)
                self._pace(loop_start)
        finally:
            self.close()
        log.info("stopped after %d frames (%s)", frames, self._status_summary())
        return 0

    def _pace(self, loop_start: float) -> None:
        # Camera reads already block at the camera's frame rate; files do not.
        if self.config.capture.source != "video":
            return
        fps = self.source.fps
        if not fps or fps <= 0:
            return
        remaining = 1.0 / fps - (self._clock() - loop_start)
        if remaining > 0:
            self._sleeper(remaining)

    def _viewer_closed(self) -> bool:
        return self.viewer.available and not self.viewer.is_running

    # -------------------------------------------------------------- calibration
    def calibrate_interactive(
        self,
        *,
        timeout_s: float = 30.0,
        stable_s: float = 0.5,
    ) -> bool:
        """Capture a neutral pose after the hand has been held still briefly."""
        self.perception.start()
        self.viewer.launch()
        self.stepper.reset(self._clock())
        deadline = self._clock() + timeout_s
        stable_since: float | None = None
        saved = False
        try:
            while self._clock() < deadline:
                loop_start = self._clock()
                if self._viewer_closed():
                    break
                frame = self.source.read()
                if frame is None:
                    break
                if frame.seq % max(1, self.config.perception.skip) == 0:
                    self.perception.submit(frame)
                now = self._clock()
                health = self.perception.health
                if health.failed:
                    log.error("perception worker failed: %s", health.message)
                    break
                result = self.perception.latest_result()
                observation = self._fresh_observation(result, now)
                key = self._render(
                    frame,
                    observation,
                    result,
                    now,
                    extra=["Hold an open hand still to calibrate"],
                )
                for _ in range(self.stepper.advance(now)):
                    mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
                if key in {27, ord("q")}:
                    break
                if observation is None:
                    stable_since = None
                elif stable_since is None:
                    stable_since = now
                elif now - stable_since >= stable_s:
                    saved = self._calibrate_with(observation, now)
                    break
                self._pace(loop_start)
        finally:
            self.close()
        return saved

    def _calibrate_with(self, observation: HandObservation | None, now: float) -> bool:
        if observation is None or now - observation.capture_ts > CALIBRATION_MAX_AGE_S:
            self._last_status = "calibration failed: no fresh hand observation"
            log.warning(self._last_status)
            return False
        calibration = self.retargeter.calibrate(observation)
        try:
            self.calibration_store.save(calibration)
        except OSError as exc:
            self._last_status = f"calibration failed: {exc}"
            log.error(self._last_status)
            return False
        self.retargeter.neutral = calibration.neutral
        self._last_status = f"calibrated {calibration.side} hand ({calibration.backend})"
        log.info("%s -> %s", self._last_status, self.calibration_store.path)
        return True

    def _enable_follow(self, sample) -> None:
        self.macro.set_zero(sample, self.hand.macro_pose)
        self.following = True
        self._last_status = "phone follow on (relative to current pose)"

    def _toggle_follow(self) -> None:
        if self.following:
            self.following = False
            self._auto_follow_armed = False
            self._last_status = "phone follow off; macro pose frozen"
            return
        sample = self.orientation.latest()
        if sample is None or not sample.ready:
            self._last_status = "phone not ready; check phyphox remote access"
            return
        self._enable_follow(sample)

    def _zero_phone(self) -> bool:
        """Re-zero the relative mapping at the current phone attitude."""
        sample = self.orientation.latest()
        if sample is None or not sample.ready:
            self._last_status = "phone not ready; cannot zero (check phyphox)"
            return False
        self._enable_follow(sample)
        self._auto_follow_armed = False
        self._last_status = "phone zeroed at current attitude"
        pitch, roll, yaw = q_euler_deg(self.macro.zero_hand)
        log.info("phone mapping zeroed (hand p/r/y %+.1f %+.1f %+.1f)", pitch, roll, yaw)
        return True

    def _reset(self) -> None:
        self.controller.reset()
        self.following = False
        self._auto_follow_armed = False
        self.macro.reset()
        self.hand.reset_macro_pose()
        self.hand.reset(self.start_ctrl)
        mujoco.mj_forward(self.model, self.data)
        self._last_status = "reset to start pose"

    # ------------------------------------------------------------------ display
    def _fresh_observation(
        self, result: PerceptionResult | None, now: float
    ) -> HandObservation | None:
        if result is None or not result.observations:
            return None
        if now - result.capture_ts > self.config.perception.max_age_s:
            return None
        return result.observations[0]

    def _render(
        self,
        frame,
        observation: HandObservation | None,
        result: PerceptionResult | None,
        now: float,
        *,
        extra: list[str] | None = None,
    ) -> int | None:
        if not self.config.ui.show and self.recorder is None:
            return None

        preview = frame.image_bgr.copy()
        draw_observation(preview, observation, self.config.ui.draw)
        state = self.controller.state.value
        lines = [f"state {state}"]
        if observation is not None:
            lines.append(
                f"hand {observation.anatomical_side.value} "
                f"(seen {observation.apparent_side.value}) "
                f"{observation.backend} {observation.score:.2f}"
            )
        else:
            lines.append("waiting for hand")
        lines.extend(self.telemetry.hud_lines())
        peaks = self.hand.peak_touch
        lines.append("touch " + " ".join(f"{name} {value:.2f}" for name, value in peaks.items()))
        if self._phone_enabled:
            health = self.orientation.health
            follow = "on" if self.following else "off"
            host = health.host or "-"
            lines.append(
                f"phone {follow} | {health.state} | {host} | "
                f"{health.rate_hz:.0f} Hz | {health.message}"
            )
            pitch, roll, yaw = q_euler_deg(self.macro.zero_hand)
            lines.append(f"zero p/r/y {pitch:+5.1f} {roll:+5.1f} {yaw:+5.1f}")
            lines.append("[g]phone [0]zero [c]alibrate [r]eset [q]uit")
        else:
            lines.append("[c]alibrate [r]eset [q]uit")
        if extra:
            lines.extend(extra)
        if self._last_status:
            lines.append(self._last_status)
        draw_hud(preview, lines)

        key = self.preview.show(preview)
        if self.recorder is not None:
            try:
                self.recorder.write(preview)
            except RecordingError as exc:
                log.error("recording disabled: %s", exc)
                self.recorder = None
        return key

    def _status_summary(self) -> str:
        summary = self.telemetry.summary()
        parts = []
        for key, value in summary.items():
            if value is None:
                continue
            parts.append(f"{key}={value:.1f}")
        return ", ".join(parts)

    # -------------------------------------------------------------------- close
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.source.close()
        except Exception:
            log.exception("error closing frame source")
        try:
            self.perception.close()
        except Exception:
            log.exception("error closing perception")
        try:
            self.orientation.close()
        except Exception:
            log.exception("error closing phone source")
        try:
            self.viewer.close()
        except Exception:
            log.exception("error closing viewer")
        try:
            self.preview.close()
        except Exception:
            log.exception("error closing preview")
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None


# ---------------------------------------------------------------------- CLI glue


def _set(config: AppConfig, section: str, **changes) -> AppConfig:
    current = getattr(config, section)
    updates = {key: value for key, value in changes.items() if value is not None}
    if not updates:
        return config
    return replace(config, **{section: replace(current, **updates)})


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply explicit command-line flags on top of YAML/default configuration."""
    get = lambda name, default=None: getattr(args, name, default)  # noqa: E731

    capture = {}
    source = get("source")
    camera = get("camera")
    if source is not None:
        capture["source"] = source
    if camera is not None:
        if source == "video":
            capture["video"] = camera
        else:
            capture["camera"] = camera
    if get("mirror") is not None:
        capture["mirror"] = bool(get("mirror"))
    config = _set(config, "capture", **capture)

    config = _set(
        config,
        "perception",
        backend=get("backend"),
        side=get("side"),
        pick=get("pick"),
        skip=get("skip"),
        fp16=get("fp16"),
    )
    if get("smooth") is not None:
        config = _set(config, "smoothing", tau=float(get("smooth")))
    config = _set(
        config,
        "sim",
        viewer=get("viewer"),
    )
    config = _set(
        config,
        "ui",
        show=get("show"),
        draw=get("draw"),
        record=get("output"),
    )
    config = _set(
        config,
        "data",
        data_dir=get("data_dir"),
        checkpoint=get("checkpoint"),
        hand_model=get("hand_model"),
        calibration=get("calibration"),
    )
    phone_host = get("phone_host")
    phone_port = get("phone_port")
    if phone_host is not None:
        try:
            parsed_host, parsed_port = parse_host_port(
                phone_host, default_port=phone_port or config.phone.port
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        phone_host = parsed_host
        if phone_port is None:
            phone_port = parsed_port
    config = _set(
        config,
        "phone",
        source=get("phone"),
        host=phone_host,
        port=phone_port,
        auto_discover=get("phone_scan"),
        auto_follow=get("phone_follow"),
        mount_preset=get("phone_mount"),
    )
    config = _set(
        config,
        "runtime",
        max_frames=get("max_frames") if get("max_frames") is not None else None,
        debug=get("debug"),
    )
    config.validate()
    return config


def run_from_args(args: argparse.Namespace, config: AppConfig | None = None) -> int:
    cfg = apply_cli_overrides(config or AppConfig(), args)
    app = TeleopApp(cfg)
    return app.run()


def calibrate_from_args(args: argparse.Namespace, config: AppConfig | None = None) -> int:
    cfg = apply_cli_overrides(config or AppConfig(), args)
    cfg = _set(cfg, "sim", viewer=False)
    app = TeleopApp(cfg)
    try:
        ok = app.calibrate_interactive()
    finally:
        app.close()
    if ok:
        print(f"calibration saved to {app.calibration_store.path}")
        return 0
    print("calibration was not saved (no stable hand observation)")
    return 1
