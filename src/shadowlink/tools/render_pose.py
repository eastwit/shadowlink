"""Render the Shadow Hand for one still image.

This is an offline debugging tool: run perception once on an image, retarget
the observation, hold the resulting actuator targets until the hand settles,
and write a PNG.  It is useful for checking a mapping change without a camera
or a live viewer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from shadowlink.config import AppConfig
from shadowlink.paths import ModelPaths
from shadowlink.perception.base import create_backend, create_detector
from shadowlink.perception.selector import DetectionSelector, filter_by_side
from shadowlink.retarget.calibration import CalibrationStore, Neutral
from shadowlink.retarget.mapper import Retargeter
from shadowlink.sim.model import load_model
from shadowlink.sim.viewer import DEFAULT_CAMERA
from shadowlink.types import FramePacket, HandObservation

SETTLE_STEPS = 1500


def _config_from_args(args) -> AppConfig:
    overrides = []
    if getattr(args, "backend", None):
        overrides.append(f"perception.backend={args.backend}")
    if getattr(args, "side", None):
        overrides.append(f"perception.side={args.side}")
    if getattr(args, "pick", None):
        overrides.append(f"perception.pick={args.pick}")
    if getattr(args, "data_dir", None):
        overrides.append(f"data.data_dir={args.data_dir}")
    if getattr(args, "checkpoint", None):
        overrides.append(f"data.checkpoint={args.checkpoint}")
    if getattr(args, "hand_model", None):
        overrides.append(f"data.hand_model={args.hand_model}")
    return AppConfig().apply_overrides(overrides) if overrides else AppConfig()


def observe_image(
    image: np.ndarray,
    cfg: AppConfig,
    models: ModelPaths,
) -> HandObservation | None:
    """Run the configured detector and backend once on a BGR image."""
    detector = create_detector(cfg.perception, models)
    backend = create_backend(cfg.perception, models)
    try:
        frame = FramePacket(seq=0, capture_ts=time.monotonic(), image_bgr=image)
        detections = filter_by_side(detector.detect(frame), cfg.perception.side)
        detection = DetectionSelector(cfg.perception.pick).select(
            detections, frame.width, frame.height
        )
        if detection is None:
            return None
        observations = backend.estimate(frame, [detection])
    finally:
        backend.close()
        detector.close()
    return observations[0] if observations else None


def render_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: np.ndarray,
    out_path: str | Path,
    *,
    width: int = 640,
    height: int = 480,
    settle_steps: int = SETTLE_STEPS,
) -> Path:
    """Hold ``targets`` until the simulation settles, then render a PNG."""
    data.ctrl[:] = np.asarray(targets, dtype=np.float64)
    for _ in range(int(settle_steps)):
        mujoco.mj_step(model, data)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = DEFAULT_CAMERA["lookat"]
    camera.distance = DEFAULT_CAMERA["distance"]
    camera.azimuth = DEFAULT_CAMERA["azimuth"]
    camera.elevation = DEFAULT_CAMERA["elevation"]

    renderer = mujoco.Renderer(model, int(height), int(width))
    try:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
    finally:
        renderer.close()

    path = Path(out_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image[:, :, ::-1]):
        raise OSError(f"cannot write {path}")
    return path


def describe(metrics, targets, layout) -> list[str]:
    """A compact, human-readable summary of the retargeted pose."""
    lines = []
    for finger in ("index", "middle", "ring", "pinky", "thumb"):
        degrees = "/".join(f"{np.degrees(value):5.1f}" for value in metrics.flex[finger])
        lines.append(f"{finger:7s} flex {degrees} deg")
    spread = "  ".join(
        f"{finger}:{np.degrees(metrics.spread[finger]):5.1f}"
        for finger in ("index", "middle", "ring", "pinky")
    )
    lines.append(f"spread     {spread} deg")
    lines.append(f"opposition {np.degrees(metrics.opposition):.1f} deg")
    values = " ".join(
        f"{spec.semantic}={value:.2f}" for spec, value in zip(layout.specs, targets, strict=True)
    )
    lines.append(f"targets    {values}")
    return lines


def render_pose_cli(args) -> int:
    cfg = _config_from_args(args)
    image_path = Path(args.image).expanduser()
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"cannot read image: {image_path}", file=sys.stderr)
        return 2

    models = ModelPaths.resolve(
        cfg.data.data_dir,
        checkpoint=cfg.data.checkpoint,
        hand_model=cfg.data.hand_model,
    )
    observation = observe_image(image, cfg, models)
    if observation is None:
        print("no hand detected", file=sys.stderr)
        return 1

    model, layout = load_model()
    neutral = Neutral.default()
    calibration_path = getattr(args, "calibration", None)
    if calibration_path:
        calibration = CalibrationStore(calibration_path).load(cfg.perception.backend)
        if calibration is not None:
            neutral = calibration.neutral
    retargeter = Retargeter(cfg.retarget, neutral, layout)
    metrics = retargeter.metrics(observation)
    targets = retargeter.map_metrics(metrics)

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    path = render_target(model, data, targets, args.out)
    for line in describe(metrics, targets, layout):
        print(line)
    print(f"wrote {path}")
    return 0


__all__ = ["describe", "observe_image", "render_pose_cli", "render_target"]
