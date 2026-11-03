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
    ],
)
def test_bad_overrides_are_rejected(pair):
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides([pair])


def test_validation_rejects_inconsistent_values():
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["capture.source=video"])
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
        "capture:\n  source: video\n  video: clips/demo.mp4\ndata:\n  data_dir: models\n",
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
