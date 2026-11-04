"""Tests for typed configuration, YAML loading, and CLI overrides."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml

from shadowlink.app import apply_cli_overrides
from shadowlink.cli import _load_config
from shadowlink.config import AppConfig, ConfigError
from shadowlink.math3d import q_rotate
from shadowlink.retarget.macro_pose import mount_for_config

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


def test_phone_overrides_and_cli_flags():
    config = AppConfig().apply_overrides(
        [
            "phone.source=phyphox",
            "phone.host=192.0.2.10",
            "phone.gain=1.5",
            "phone.deadzone_deg=2.0",
            "phone.auto_follow=true",
        ]
    )
    assert config.phone.source == "phyphox"
    assert config.phone.host == "192.0.2.10"
    assert config.phone.gain == pytest.approx(1.5)
    args = argparse.Namespace(
        phone="phyphox",
        phone_host="192.0.2.11:8090",
        phone_follow=True,
    )
    updated = apply_cli_overrides(config, args)
    assert updated.phone.host == "192.0.2.11"
    assert updated.phone.port == 8090
    assert updated.phone.auto_follow is True


def test_phone_scan_does_not_require_a_host():
    config = AppConfig().apply_overrides(["phone.source=phyphox", "phone.auto_discover=true"])
    assert config.phone.host is None
    assert config.phone.auto_discover is True


@pytest.mark.parametrize(
    "pair",
    [
        "phone.source=keyboard",
        "phone.mount_preset=elbow",
        "phone.deadzone_deg=180.0",
        "phone.max_angle_deg=0.0",
        "phone.mount_w=nan",
        "phone.gain=-1",
        "phone.stale_timeout_s=0",
        "phone.port=0",
        "phone.poll_hz=0",
    ],
)
def test_invalid_phone_settings_are_rejected(pair):
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides([pair])


def test_zero_mount_quaternion_is_rejected():
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(
            [
                "phone.mount_w=0.0",
                "phone.mount_x=0.0",
                "phone.mount_y=0.0",
                "phone.mount_z=0.0",
            ]
        )


def test_phone_defaults_describe_the_wrist_strap_mount_without_a_clamp():
    phone = AppConfig().phone
    assert phone.pivot == "wrist"
    assert phone.mount_preset == "back"
    assert phone.max_angle_deg == pytest.approx(180.0)
    # The custom values mirror the default preset and are kept for reference.
    assert (phone.mount_w, phone.mount_x, phone.mount_y, phone.mount_z) == pytest.approx(
        (0.0, 0.7071068, 0.0, -0.7071068)
    )


def test_back_mount_maps_phone_axes_onto_the_strap_orientation():
    mount = mount_for_config(AppConfig().phone)
    # Charging port -> elbow (-z), phone top -> fingers, screen -> dorsal (-x).
    np.testing.assert_allclose(q_rotate(mount, [1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(q_rotate(mount, [0.0, 1.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(q_rotate(mount, [0.0, 0.0, 1.0]), [-1.0, 0.0, 0.0], atol=1e-6)


def test_in_hand_mount_is_the_back_mount_flipped_about_the_long_axis():
    config = AppConfig().apply_overrides(["phone.mount_preset=in_hand"])
    mount = mount_for_config(config.phone)
    # Long axis unchanged; width and screen normal reversed.
    np.testing.assert_allclose(q_rotate(mount, [1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(q_rotate(mount, [0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(q_rotate(mount, [0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-6)


def test_phone_mount_cli_flag_reaches_the_config():
    args = argparse.Namespace(phone_mount="in_hand")
    updated = apply_cli_overrides(AppConfig(), args)
    assert updated.phone.mount_preset == "in_hand"


def test_phyphox_requires_a_host_without_discovery():
    with pytest.raises(ConfigError):
        AppConfig().apply_overrides(["phone.source=phyphox"])


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
