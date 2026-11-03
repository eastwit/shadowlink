"""Tests for data, calibration, and output path resolution."""

from __future__ import annotations

from pathlib import Path

from shadowlink import paths


def test_explicit_data_dir_wins_over_the_environment(tmp_path, monkeypatch):
    env_dir = tmp_path / "env"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv(paths.DEFAULT_DATA_ENV, str(env_dir))
    assert paths.resolve_data_dir(explicit) == explicit.resolve()
    assert paths.resolve_data_dir() == env_dir.resolve()


def test_default_data_dir_is_absolute(monkeypatch):
    monkeypatch.delenv(paths.DEFAULT_DATA_ENV, raising=False)
    resolved = paths.resolve_data_dir()
    assert resolved.is_absolute()
    assert resolved == (Path.home() / ".cache" / "shadowlink").resolve()


def test_calibration_path_precedence(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.json"
    env_path = tmp_path / "env.json"
    assert paths.resolve_calibration_path(explicit) == explicit.resolve()
    monkeypatch.setenv(paths.CALIBRATION_ENV, str(env_path))
    assert paths.resolve_calibration_path() == env_path.resolve()
    monkeypatch.delenv(paths.CALIBRATION_ENV)
    default = paths.resolve_calibration_path()
    assert default.is_absolute()
    assert default.name == "calibration.json"


def test_model_paths_derive_from_the_data_dir(tmp_path):
    models = paths.ModelPaths.resolve(tmp_path)
    root = tmp_path.resolve()
    assert models.data_dir == root
    assert models.checkpoint == root / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    assert models.model_config == root / "hamer_ckpts" / "model_config.yaml"
    assert models.mano == root / "data" / "mano" / "MANO_RIGHT.pkl"
    assert models.hand_landmarker == root / "mediapipe" / "hand_landmarker.task"


def test_model_paths_accept_overrides(tmp_path):
    checkpoint = tmp_path / "custom" / "hamer.ckpt"
    hand_model = tmp_path / "custom" / "hand.task"
    models = paths.ModelPaths.resolve(tmp_path, checkpoint=checkpoint, hand_model=hand_model)
    assert models.checkpoint == checkpoint.resolve()
    assert models.model_config == checkpoint.resolve().parent.parent / "model_config.yaml"
    assert models.hand_landmarker == hand_model.resolve()


def test_missing_lists_only_absent_files(tmp_path):
    models = paths.ModelPaths.resolve(tmp_path)
    assert len(models.missing(need_hamer=True)) == 4
    assert len(models.missing(need_hamer=False)) == 1

    models.hand_landmarker.parent.mkdir(parents=True)
    models.hand_landmarker.write_bytes(b"stub")
    assert models.missing(need_hamer=False) == []


def test_resolve_output_path_creates_parents(tmp_path):
    target = tmp_path / "a" / "b" / "clip.mp4"
    resolved = paths.resolve_output_path(target)
    assert resolved == target.resolve()
    assert resolved.parent.is_dir()
