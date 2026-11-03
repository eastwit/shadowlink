"""Tests for neutral calibration and its JSON persistence."""

from __future__ import annotations

import json

import pytest

from shadowlink.retarget import calibration as calibration_module
from shadowlink.retarget.calibration import (
    SCHEMA_VERSION,
    Calibration,
    CalibrationError,
    CalibrationStore,
    Neutral,
)
from shadowlink.retarget.geometry import ALL_FINGERS, SPREAD_REST, HandMetrics


def make_calibration(backend: str = "test") -> Calibration:
    return Calibration.create(
        backend=backend,
        side="right",
        neutral=Neutral.default(),
    )


def test_default_neutral_covers_every_finger():
    neutral = Neutral.default()
    assert set(neutral.flex) == set(ALL_FINGERS)
    assert neutral.spread == pytest.approx(SPREAD_REST)


def test_neutral_round_trip():
    neutral = Neutral.default()
    assert Neutral.from_dict(neutral.to_dict()) == neutral


def test_neutral_from_metrics():
    metrics = HandMetrics(
        flex={finger: (0.1, 0.2, 0.3) for finger in ALL_FINGERS},
        spread=dict(SPREAD_REST),
        opposition=0.4,
    )
    neutral = Neutral.from_metrics(metrics)
    assert neutral.flex == metrics.flex
    assert neutral.spread == metrics.spread
    assert neutral.opposition == pytest.approx(0.4)


def test_neutral_with_flex_is_a_copy():
    neutral = Neutral.default()
    updated = neutral.with_flex("index", (0.5, 0.5, 0.5))
    assert neutral.flex["index"] == (0.1, 0.1, 0.1)
    assert updated.flex["index"] == (0.5, 0.5, 0.5)


def test_calibration_round_trip_through_disk(tmp_path):
    store = CalibrationStore(tmp_path / "nested" / "calibration.json")
    calibration = make_calibration()
    store.save(calibration)
    assert store.path.exists()
    assert store.load("test") == calibration


def test_load_missing_file_returns_none(tmp_path):
    assert CalibrationStore(tmp_path / "missing.json").load() is None


def test_backend_mismatch_is_ignored_by_default(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    store.save(make_calibration("hamer"))
    assert store.load("mediapipe") is None
    with pytest.raises(CalibrationError):
        store.load("mediapipe", strict=True)
    assert store.load("hamer") is not None


def test_corrupt_json_raises(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationError):
        CalibrationStore(path).load()


@pytest.mark.parametrize("version", [None, 0, SCHEMA_VERSION + 1, "1"])
def test_unsupported_schema_raises(version):
    data = make_calibration().to_dict()
    if version is None:
        data.pop("schema_version")
    else:
        data["schema_version"] = version
    with pytest.raises(CalibrationError):
        Calibration.from_dict(data)


@pytest.mark.parametrize("missing", ["neutral", "flex", "spread", "opposition"])
def test_missing_fields_raise(missing):
    data = make_calibration().to_dict()
    if missing == "neutral":
        data.pop("neutral")
    else:
        data["neutral"].pop(missing)
    with pytest.raises(CalibrationError):
        Calibration.from_dict(data)


def test_invalid_flex_length_raises():
    data = make_calibration().to_dict()
    data["neutral"]["flex"]["index"] = [0.0, 0.0]
    with pytest.raises(CalibrationError):
        Calibration.from_dict(data)


def test_failed_replace_keeps_the_previous_file(tmp_path, monkeypatch):
    store = CalibrationStore(tmp_path / "calibration.json")
    original = make_calibration("hamer")
    store.save(original)
    before = store.path.read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(calibration_module.os, "replace", boom)
    with pytest.raises(OSError):
        store.save(make_calibration("mediapipe"))

    assert store.path.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("*.tmp")) == []


def test_saved_file_is_json_with_metadata(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    store.save(make_calibration("hamer"))
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["backend"] == "hamer"
    assert data["created_at"]
    assert data["neutral"]["opposition"] == pytest.approx(0.0)


def test_delete_removes_the_file(tmp_path):
    store = CalibrationStore(tmp_path / "calibration.json")
    store.save(make_calibration())
    store.delete()
    assert not store.path.exists()
    store.delete()  # missing_ok
