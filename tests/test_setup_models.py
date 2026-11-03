"""Tests for the model setup helpers (no network access)."""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import pytest

from shadowlink.paths import ModelPaths
from shadowlink.tools import setup_models as tools


def namespace(tmp_path, **overrides) -> argparse.Namespace:
    values = {
        "data_dir": str(tmp_path),
        "checkpoint": None,
        "hand_model": None,
        "check_only": False,
        "copy_from": None,
        "force": False,
        "mediapipe_only": False,
        "hamer_only": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def all_model_files(paths: ModelPaths) -> None:
    for path in (paths.checkpoint, paths.model_config, paths.mano, paths.hand_landmarker):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")


def write_tar(path: Path, members) -> None:
    with tarfile.open(path, "w") as tar:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = payload.decode()
                tar.addfile(info)
            else:  # pragma: no cover - defensive
                raise AssertionError(kind)


def test_check_only_reports_missing_files(tmp_path, capsys):
    code = tools.setup_models(namespace(tmp_path, check_only=True))
    output = capsys.readouterr().out
    assert code == 1
    assert "missing" in output
    assert tools.MANO_URL in output


def test_check_only_succeeds_when_everything_is_present(tmp_path, capsys):
    paths = ModelPaths.resolve(tmp_path)
    all_model_files(paths)
    code = tools.setup_models(namespace(tmp_path, check_only=True))
    assert code == 0
    assert "all model files are present" in capsys.readouterr().out


def test_copy_from_copies_known_files_without_overwriting(tmp_path, capsys):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for relative in tools._COPY_FILES:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())

    paths = ModelPaths.resolve(target)
    existing = target / tools._COPY_FILES[0]
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b"old")

    tools._copy_existing(source, paths, force=False)
    assert existing.read_bytes() == b"old"
    assert (target / tools._COPY_FILES[1]).read_bytes() == tools._COPY_FILES[1].encode()

    tools._copy_existing(source, paths, force=True)
    assert existing.read_bytes() == tools._COPY_FILES[0].encode()
    assert "copied" in capsys.readouterr().out


def test_copy_from_rejects_a_file(tmp_path):
    with pytest.raises(SystemExit):
        tools._copy_existing(tmp_path / "missing", ModelPaths.resolve(tmp_path), force=False)


def test_safe_extract_accepts_regular_files(tmp_path):
    archive = tmp_path / "data.tar"
    write_tar(archive, [("root/file.txt", b"hello", "file")])
    target = tmp_path / "out"
    target.mkdir()
    tools._safe_extract(archive, target)
    assert (target / "root" / "file.txt").read_bytes() == b"hello"


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tar"
    write_tar(archive, [("../escape.txt", b"nope", "file")])
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SystemExit):
        tools._safe_extract(archive, target)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extract_rejects_links(tmp_path):
    archive = tmp_path / "link.tar"
    write_tar(archive, [("link", b"/etc/passwd", "symlink")])
    target = tmp_path / "out"
    target.mkdir()
    with pytest.raises(SystemExit):
        tools._safe_extract(archive, target)


def test_find_data_root(tmp_path):
    assert tools._find_data_root(tmp_path) is None
    nested = tmp_path / "_DATA"
    (nested / "hamer_ckpts").mkdir(parents=True)
    assert tools._find_data_root(tmp_path) == nested


def test_download_copies_a_file_url(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"payload")
    destination = tmp_path / "nested" / "destination.bin"
    tools._download(source.as_uri(), destination)
    assert destination.read_bytes() == b"payload"
    assert not destination.with_name(destination.name + ".part").exists()


def test_download_keeps_existing_files_unless_forced(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"new")
    destination = tmp_path / "destination.bin"
    destination.write_bytes(b"old")
    tools._download(source.as_uri(), destination)
    assert destination.read_bytes() == b"old"
    tools._download(source.as_uri(), destination, force=True)
    assert destination.read_bytes() == b"new"
