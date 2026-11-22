"""Tests for typed configuration, YAML loading, and CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from shadowlink.app import apply_cli_overrides
from shadowlink.cli import _load_config
from shadowlink.config import AppConfig, ConfigError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / "configs" / "default.yaml"


def test_default_yaml_matches_dataclass_defaults():
    loaded = yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))
    assert loaded == AppConfig().to_dict()


def test_dict_round_trip():
    config = AppConfig()
    assert AppConfig.from_dict(config.to_dict()) == config


def test_dotted_overrides_are_typed_and_do_not_mutate_the_original():
    config = AppConfig()
    updated = config.apply_overrides(
        ["retarget.flex_gain=1.25", "perception.skip=3", "sim.viewer=false"]
    )
    assert updated.retarget.flex_gain == pytest.approx(1.25)
    assert updated.perception.skip == 3
    assert updated.sim.viewer is False
    assert config.retarget.flex_gain == pytest.approx(1.0)
    assert config.sim.viewer is True


@pytest.mark.parametrize(
    "pair",
    [
        "retarget",
        "retarget.flex_gain",
        "nope.flex_gain=1",
        "retarget.nope=1",
        "retarget.flex_gain=abc",
        "perception.backend=openpose",
        "phone.source=phyphox",
        "joycon.side=left",
        "capture.depth_sdk=C:/sdk",
        "position.input=rgbd",
    ],
)
def test_bad_or_removed_overrides_are_rejected(pair):
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides([pair])


def test_validation_rejects_inconsistent_values():
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["capture.source=video"])
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["capture.source=rgbd"])
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["perception.skip=0"])
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["smoothing.max_rate_rad_s=0"])
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["capture.api=http"])


def test_yaml_relative_paths_resolve_against_the_file(tmp_path):
    config_path = tmp_path / "nested" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "capture:\n  source: video\n  video: clips/demo.mp4\n"
        "data:\n  data_dir: models\n",
        encoding="utf-8",
    )
    config = AppConfig.from_yaml(config_path)
    assert config.capture.video == str((config_path.parent / "clips" / "demo.mp4").resolve())
    assert config.data.data_dir == str((config_path.parent / "models").resolve())


def test_unknown_yaml_keys_are_rejected(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("capture:\n  source: camera\n  zoom: 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        AppConfig.from_yaml(config_path)


def test_cli_config_precedence(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("perception:\n  backend: hamer\n  skip: 2\n", encoding="utf-8")
    args = argparse.Namespace(config=str(config_path), overrides=["perception.skip=4"])
    config = _load_config(args)
    assert config.perception.backend == "hamer"
    assert config.perception.skip == 4


def test_explicit_cli_flags_beat_the_config_file():
    config = AppConfig.from_dict({"perception": {"backend": "hamer"}})
    args = argparse.Namespace(
        backend="mediapipe",
        fp16=False,
        viewer=False,
        show=False,
        max_frames=7,
    )
    updated = apply_cli_overrides(config, args)
    assert updated.perception.backend == "mediapipe"
    assert updated.perception.fp16 is False
    assert updated.sim.viewer is False
    assert updated.ui.show is False
    assert updated.runtime.max_frames == 7


def test_tag_and_position_defaults():
    config = AppConfig()
    assert config.capture.source == "camera"
    assert config.capture.camera == "1"
    assert config.position.enabled is True
    assert config.position.preset == "overhead"
    assert config.tag.camera == "0"
    assert config.tag.api == "dshow"
    assert config.tag.width == 640
    assert config.tag.height == 480
    assert config.tag.family == "tag36h11"
    assert config.tag.tag_id == 5
    assert config.tag.size_m == pytest.approx(0.057)
    assert config.tag.mirror == "auto"
    assert config.tag.show is True
    assert config.tag.zero_mode == "fixed"
    assert config.tag.zero_distance_m == pytest.approx(1.0)
    assert config.tag.rotation_smoothing_tau_s == pytest.approx(0.12)
    assert config.tag.rotation_max_rate_rad_s == pytest.approx(3.0)
    assert config.tag.rotation_jump_reject_deg == pytest.approx(35.0)
    assert config.tag.rotation_jump_confirm_frames == 5
    assert config.tag.min_decision_margin == pytest.approx(10.0)
    assert config.tag.max_hamming == 1
    assert config.tag.auto_follow is True


@pytest.mark.parametrize(
    "pair",
    [
        "tag.camera=",
        "tag.width=0",
        "tag.height=0",
        "tag.fps=0",
        "tag.hfov_deg=180",
        "tag.fx=900",
        "tag.family= ",
        "tag.tag_id=-1",
        "tag.size_m=0",
        "tag.quad_decimate=0.5",
        "tag.nthreads=0",
        "tag.retry_s=0",
        "tag.stale_timeout_s=0",
        "tag.pivot=elbow",
        "tag.rotation_gain=0",
        "tag.rotation_deadzone_deg=180",
        "tag.rotation_max_angle_deg=0",
        "tag.rotation_smoothing_tau_s=-1",
        "tag.rotation_max_rate_rad_s=0",
        "tag.rotation_jump_reject_deg=0",
        "tag.rotation_jump_reject_deg=181",
        "tag.rotation_jump_confirm_frames=0",
        "tag.min_decision_margin=-1",
        "tag.max_hamming=4",
        "tag.api=serial",
        "tag.mirror=flip",
        "tag.zero_mode=relative",
        "tag.zero_distance_m=0",
        "position.preset=sideways",
        "position.gain=0",
        "position.deadzone_m=-0.1",
        "position.temporal_window=0",
        "position.max_rate_m_s=0",
    ],
)
def test_invalid_tag_and_position_settings_are_rejected(pair):
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides([pair])


def test_tag_camera_must_differ_from_the_hand_camera():
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["tag.camera=1"])
    updated = AppConfig().apply_overrides(["capture.camera=0", "tag.camera=1"])
    assert updated.capture.camera == "0"
    assert updated.tag.camera == "1"


def test_tag_cli_flags_reach_the_config():
    args = argparse.Namespace(
        camera="1",
        tag_camera="0",
        tag_api="msmf",
        tag_family="tag36h11",
        tag_id=0,
        tag_size=0.0456,
        tag_mirror="on",
        tag_zero_mode="sample",
        tag_zero_distance=0.8,
        tag_follow=True,
    )
    updated = apply_cli_overrides(AppConfig(), args)
    assert updated.capture.camera == "1"
    assert updated.tag.camera == "0"
    assert updated.tag.api == "msmf"
    assert updated.tag.tag_id == 0
    assert updated.tag.size_m == pytest.approx(0.0456)
    assert updated.tag.mirror == "on"
    assert updated.tag.zero_mode == "sample"
    assert updated.tag.zero_distance_m == pytest.approx(0.8)
    assert updated.tag.auto_follow is True


def test_position_cli_flags_reach_the_config():
    args = argparse.Namespace(
        position=False,
        position_preset="palm",
        position_gain=1.5,
        position_deadzone=0.02,
    )
    updated = apply_cli_overrides(AppConfig(), args)
    assert updated.position.enabled is False
    assert updated.position.preset == "palm"
    assert updated.position.gain == pytest.approx(1.5)
    assert updated.position.deadzone_m == pytest.approx(0.02)
