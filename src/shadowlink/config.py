"""Typed application configuration.

ShadowLink now has one whole-hand control source: a hand-held AprilTag. The
built-in webcam still reconstructs the fingers. ``configs/default.yaml`` mirrors
the dataclass defaults one-to-one.

Values can be overridden with dotted CLI assignments such as
``--set tag.rotation_gain=1.2``. Priority is CLI > YAML > dataclass default.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Literal, Union, get_args, get_origin, get_type_hints

import yaml

DrawMode = Literal["skeleton", "mesh", "none"]
BackendName = Literal["hamer", "mediapipe"]
MacroPivot = Literal["palm", "wrist", "forearm"]
PositionPreset = Literal["front", "palm", "back", "overhead"]
TagMirror = Literal["auto", "on", "off"]


class ConfigError(ValueError):
    """Raised for malformed or inconsistent configuration."""


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    source: Literal["camera", "video"] = "camera"
    camera: str = "1"
    video: str | None = None
    width: int = 0
    height: int = 0
    mirror: bool = True
    api: Literal["any", "dshow", "msmf"] = "any"
    buffer_size: int = 1
    fps: float = 0.0


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    backend: BackendName = "hamer"
    side: Literal["any", "left", "right"] = "any"
    pick: Literal["largest", "center", "track"] = "track"
    skip: int = 1
    max_age_s: float = 0.5
    min_detection_confidence: float = 0.3
    rescale_factor: float = 2.0
    fp16: bool = True
    pipeline: Literal["single", "two_stage"] = "two_stage"


@dataclass(frozen=True, slots=True)
class RetargetConfig:
    flex_gain: float = 1.0
    distal_gain: float = 1.0
    thumb_base_gain: float = 1.7
    thumb_flex_gain: float = 1.1
    thumb_distal_gain: float = 1.15
    spread_gain: float = 0.7
    spread_deadzone: float = 0.03
    spread_clip: float = 0.45
    thumb_opp_gain: float = 0.0


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    kind: Literal["none", "exponential", "one_euro"] = "one_euro"
    tau: float = 0.055
    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0
    max_rate_rad_s: float = 15.0
    return_rate_rad_s: float = 6.0
    lost_timeout_s: float = 0.5
    hold_s: float = 0.15


@dataclass(frozen=True, slots=True)
class SimConfig:
    max_catchup_s: float = 0.05
    reset_stall_s: float = 0.5
    viewer: bool = True
    touch_force_scale: float = 1.0
    start_pose: str = "open hand"


@dataclass(frozen=True, slots=True)
class UIConfig:
    show: bool = True
    draw: DrawMode = "skeleton"
    record: str | None = None
    window_name: str = "ShadowLink"


@dataclass(frozen=True, slots=True)
class DataConfig:
    data_dir: str | None = None
    checkpoint: str | None = None
    hand_model: str | None = None
    calibration: str | None = None


@dataclass(frozen=True, slots=True)
class PositionConfig:
    """Relative tag-to-world mapping shared by translation and rotation."""

    enabled: bool = True
    preset: PositionPreset = "overhead"
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    gain: float = 1.0
    deadzone_m: float = 0.01
    temporal_window: int = 3
    stale_timeout_s: float = 0.5
    auto_zero: bool = True
    min_cutoff: float = 1.2
    beta: float = 1.5
    d_cutoff: float = 1.0
    max_rate_m_s: float = 0.8


@dataclass(frozen=True, slots=True)
class TagConfig:
    """Dedicated AprilTag camera and 6-DoF controller settings."""

    camera: str = "0"
    api: Literal["any", "dshow", "msmf"] = "dshow"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    mirror: TagMirror = "auto"
    show: bool = True
    zero_mode: Literal["sample", "fixed"] = "fixed"
    zero_distance_m: float = 1.0
    hfov_deg: float = 60.0
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    family: str = "tag36h11"
    tag_id: int | None = 5
    size_m: float = 0.057
    quad_decimate: float = 1.0
    refine_edges: bool = True
    nthreads: int = 4
    retry_s: float = 1.0
    stale_timeout_s: float = 0.5
    auto_follow: bool = True
    pivot: MacroPivot = "wrist"
    rotation_gain: float = 1.0
    rotation_deadzone_deg: float = 1.5
    rotation_max_angle_deg: float = 180.0
    rotation_smoothing_tau_s: float = 0.12
    rotation_max_rate_rad_s: float = 3.0
    rotation_jump_reject_deg: float = 35.0
    rotation_jump_confirm_frames: int = 5
    min_decision_margin: float = 10.0
    max_hamming: int = 1
    rotation_invert_x: bool = False
    rotation_invert_y: bool = False
    rotation_invert_z: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_frames: int = 0
    debug: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    capture: CaptureConfig = CaptureConfig()
    perception: PerceptionConfig = PerceptionConfig()
    retarget: RetargetConfig = RetargetConfig()
    smoothing: SmoothingConfig = SmoothingConfig()
    sim: SimConfig = SimConfig()
    ui: UIConfig = UIConfig()
    data: DataConfig = DataConfig()
    position: PositionConfig = PositionConfig()
    tag: TagConfig = TagConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    # ------------------------------------------------------------------ load
    @classmethod
    def from_dict(cls, data: dict) -> AppConfig:
        if not isinstance(data, dict):
            raise ConfigError("configuration root must be a mapping")
        return _build(cls, data, "config")

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        config_path = Path(path).expanduser().resolve()
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read config {config_path}: {exc}") from exc
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
        cfg = cls.from_dict(data)
        cfg = _resolve_relative_paths(cfg, config_path.parent)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        return _to_dict(self)

    def apply_overrides(self, pairs: list[str] | tuple[str, ...]) -> AppConfig:
        cfg = self
        for pair in pairs:
            if "=" not in pair:
                raise ConfigError(f"override {pair!r} must look like section.field=value")
            dotted, raw = pair.split("=", 1)
            parts = dotted.split(".")
            if len(parts) != 2:
                raise ConfigError(f"override {pair!r} must have exactly one dot")
            section_name, field_name = parts
            if section_name not in {f.name for f in fields(AppConfig)}:
                raise ConfigError(f"unknown config section: {section_name}")
            section = getattr(cfg, section_name)
            if not is_dataclass(section):
                raise ConfigError(f"{section_name} is not a config section")
            valid_fields = {f.name for f in fields(section)}
            if field_name not in valid_fields:
                raise ConfigError(f"unknown config key: {dotted}")
            hint = get_type_hints(type(section))[field_name]
            value = _coerce(raw, hint, dotted)
            cfg = replace(cfg, **{section_name: replace(section, **{field_name: value})})
        cfg.validate()
        return cfg

    # -------------------------------------------------------------- validate
    def validate(self) -> None:
        c = self.capture
        p = self.perception
        r = self.retarget
        s = self.smoothing
        sim = self.sim

        if c.source not in ("camera", "video"):
            raise ConfigError("capture.source must be camera or video")
        if c.source == "video" and not c.video:
            raise ConfigError("capture.source=video requires capture.video")
        if c.width < 0 or c.height < 0:
            raise ConfigError("capture width/height must be >= 0")
        if c.buffer_size < 1:
            raise ConfigError("capture.buffer_size must be >= 1")
        if c.fps < 0:
            raise ConfigError("capture.fps must be >= 0")
        if p.skip < 1:
            raise ConfigError("perception.skip must be >= 1")
        if p.max_age_s <= 0:
            raise ConfigError("perception.max_age_s must be > 0")
        if p.rescale_factor <= 1:
            raise ConfigError("perception.rescale_factor must be > 1")
        if not 0.0 <= p.min_detection_confidence <= 1.0:
            raise ConfigError("perception.min_detection_confidence must be in [0, 1]")
        for name in (
            "flex_gain",
            "distal_gain",
            "thumb_base_gain",
            "thumb_flex_gain",
            "thumb_distal_gain",
            "spread_gain",
            "thumb_opp_gain",
        ):
            value = getattr(r, name)
            if not math.isfinite(value) or value < 0:
                raise ConfigError(f"retarget.{name} must be finite and >= 0")
        if r.spread_deadzone < 0 or r.spread_clip <= 0:
            raise ConfigError("retarget spread deadzone/clip must be >= 0 and > 0")
        if s.tau < 0:
            raise ConfigError("smoothing.tau must be >= 0")
        if s.min_cutoff <= 0 or s.d_cutoff <= 0:
            raise ConfigError("smoothing cutoffs must be > 0")
        if s.beta < 0:
            raise ConfigError("smoothing.beta must be >= 0")
        if s.max_rate_rad_s <= 0 or s.return_rate_rad_s <= 0:
            raise ConfigError("smoothing rates must be > 0")
        if s.lost_timeout_s < 0 or s.hold_s < 0:
            raise ConfigError("smoothing timeouts must be >= 0")
        if sim.max_catchup_s <= 0:
            raise ConfigError("sim.max_catchup_s must be > 0")
        if sim.reset_stall_s < 0:
            raise ConfigError("sim.reset_stall_s must be >= 0")
        if sim.touch_force_scale <= 0:
            raise ConfigError("sim.touch_force_scale must be > 0")
        pos = self.position
        if pos.preset not in ("front", "palm", "back", "overhead"):
            raise ConfigError(
                "position.preset must be one of front, palm, back, overhead"
            )
        if not math.isfinite(pos.gain) or pos.gain <= 0:
            raise ConfigError("position.gain must be finite and > 0")
        if not math.isfinite(pos.deadzone_m) or pos.deadzone_m < 0:
            raise ConfigError("position.deadzone_m must be finite and >= 0")
        if pos.temporal_window < 1:
            raise ConfigError("position.temporal_window must be >= 1")
        if not math.isfinite(pos.stale_timeout_s) or pos.stale_timeout_s <= 0:
            raise ConfigError("position.stale_timeout_s must be finite and > 0")
        if pos.min_cutoff <= 0 or pos.d_cutoff <= 0:
            raise ConfigError("position cutoffs must be > 0")
        if pos.beta < 0:
            raise ConfigError("position.beta must be >= 0")
        if not math.isfinite(pos.max_rate_m_s) or pos.max_rate_m_s <= 0:
            raise ConfigError("position.max_rate_m_s must be finite and > 0")

        tag = self.tag
        if not str(tag.camera).strip():
            raise ConfigError("tag.camera must not be empty")
        if c.source == "camera" and str(c.camera) == str(tag.camera):
            raise ConfigError("tag.camera must differ from capture.camera")
        if tag.width <= 0 or tag.height <= 0:
            raise ConfigError("tag.width/height must be > 0")
        if not math.isfinite(tag.fps) or tag.fps <= 0:
            raise ConfigError("tag.fps must be finite and > 0")
        if tag.zero_mode not in ("sample", "fixed"):
            raise ConfigError("tag.zero_mode must be sample or fixed")
        if not math.isfinite(tag.zero_distance_m) or not 0.1 <= tag.zero_distance_m <= 5.0:
            raise ConfigError("tag.zero_distance_m must be in [0.1, 5.0] metres")
        if not 1.0 < tag.hfov_deg < 170.0:
            raise ConfigError("tag.hfov_deg must be between 1 and 170")
        intrinsics = (tag.fx, tag.fy, tag.cx, tag.cy)
        if any(value is not None for value in intrinsics) and not all(
            value is not None for value in intrinsics
        ):
            raise ConfigError("tag.fx/fy/cx/cy must be either all set or all null")
        if tag.fx is not None:
            if tag.fx <= 0 or tag.fy <= 0:
                raise ConfigError("tag.fx/fy must be > 0")
            if not 0 <= tag.cx < tag.width or not 0 <= tag.cy < tag.height:
                raise ConfigError("tag.cx/cy must be inside the configured image")
        if not tag.family.strip():
            raise ConfigError("tag.family must not be empty")
        if tag.tag_id is not None and tag.tag_id < 0:
            raise ConfigError("tag.tag_id must be >= 0 or null")
        if not math.isfinite(tag.size_m) or not 0.001 <= tag.size_m <= 2.0:
            raise ConfigError("tag.size_m must be in [0.001, 2.0] metres")
        if not math.isfinite(tag.quad_decimate) or tag.quad_decimate < 1.0:
            raise ConfigError("tag.quad_decimate must be >= 1")
        if tag.nthreads < 1:
            raise ConfigError("tag.nthreads must be >= 1")
        if not math.isfinite(tag.retry_s) or tag.retry_s <= 0:
            raise ConfigError("tag.retry_s must be finite and > 0")
        if not math.isfinite(tag.stale_timeout_s) or tag.stale_timeout_s <= 0:
            raise ConfigError("tag.stale_timeout_s must be finite and > 0")
        if tag.pivot not in ("palm", "wrist", "forearm"):
            raise ConfigError("tag.pivot must be one of palm, wrist, forearm")
        if not math.isfinite(tag.rotation_gain) or tag.rotation_gain <= 0:
            raise ConfigError("tag.rotation_gain must be finite and > 0")
        if not 0 <= tag.rotation_deadzone_deg < tag.rotation_max_angle_deg <= 180:
            raise ConfigError(
                "tag rotation angles must satisfy 0 <= deadzone < max_angle <= 180"
            )
        if not math.isfinite(tag.rotation_smoothing_tau_s) or tag.rotation_smoothing_tau_s < 0:
            raise ConfigError("tag.rotation_smoothing_tau_s must be finite and >= 0")
        if not math.isfinite(tag.rotation_max_rate_rad_s) or tag.rotation_max_rate_rad_s <= 0:
            raise ConfigError("tag.rotation_max_rate_rad_s must be finite and > 0")
        if not 0 < tag.rotation_jump_reject_deg <= 180:
            raise ConfigError("tag.rotation_jump_reject_deg must be in (0, 180]")
        if tag.rotation_jump_confirm_frames < 1:
            raise ConfigError("tag.rotation_jump_confirm_frames must be >= 1")
        if not math.isfinite(tag.min_decision_margin) or tag.min_decision_margin < 0:
            raise ConfigError("tag.min_decision_margin must be finite and >= 0")
        if not 0 <= tag.max_hamming <= 3:
            raise ConfigError("tag.max_hamming must be in [0, 3]")
        if self.runtime.max_frames < 0:
            raise ConfigError("runtime.max_frames must be >= 0")


# ---------------------------------------------------------------- internals


def _type_name(hint) -> str:
    return getattr(hint, "__name__", str(hint))


def _coerce(value, hint, path: str):
    if hint is type(None):
        if value is not None:
            raise ConfigError(f"{path}: expected null")
        return None

    origin = get_origin(hint)
    args = get_args(hint)

    if origin is Literal:
        allowed = args
        if value not in allowed:
            raise ConfigError(f"{path}: expected one of {allowed}, got {value!r}")
        return value

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: cannot coerce {value!r} ({'; '.join(errors)})")

    if is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return _build(hint, value, path)

    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "1", "yes", "on"}:
            return True
        if isinstance(value, str) and value.lower() in {"false", "0", "no", "off"}:
            return False
        raise ConfigError(f"{path}: expected a boolean, got {value!r}")

    if hint is int:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected an integer, got a boolean")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path}: expected an integer, got {value!r}") from exc
        if isinstance(value, float) and value != result:
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return result

    if hint is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected a number, got a boolean")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path}: expected a number, got {value!r}") from exc

    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    if hint is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return value

    raise ConfigError(f"{path}: unsupported config type {_type_name(hint)}")


def _build(cls, data: dict, path: str):
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for field in fields(cls):
        if field.name in data:
            kwargs[field.name] = _coerce(
                data[field.name], hints[field.name], f"{path}.{field.name}"
            )
    return cls(**kwargs)


def _to_dict(obj):
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


def _resolve_relative_paths(cfg: AppConfig, base: Path) -> AppConfig:
    def absolute(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if path.is_absolute():
            return str(path)
        return str((base / path).resolve())

    data = replace(
        cfg.data,
        data_dir=absolute(cfg.data.data_dir),
        checkpoint=absolute(cfg.data.checkpoint),
        hand_model=absolute(cfg.data.hand_model),
        calibration=absolute(cfg.data.calibration),
    )
    capture = replace(cfg.capture, video=absolute(cfg.capture.video))
    return replace(cfg, data=data, capture=capture)
