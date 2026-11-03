"""Tests for frame-rate and latency telemetry."""

from __future__ import annotations

import numpy as np
import pytest

from shadowlink.types import PerceptionResult
from shadowlink.ui.telemetry import MetricWindow, RateMeter, Telemetry


def result(seq: int, *, capture_ts: float = 0.0, detect_ms: float = 10.0, infer_ms: float = 20.0):
    return PerceptionResult(
        frame_seq=seq,
        capture_ts=capture_ts,
        finished_ts=capture_ts + 0.03,
        detect_ms=detect_ms,
        infer_ms=infer_ms,
    )


def test_rate_meter():
    meter = RateMeter(alpha=1.0)
    assert meter.tick(0.0) is None
    assert meter.tick(0.1) == pytest.approx(10.0)
    assert meter.tick(0.3) == pytest.approx(5.0)


def test_rate_meter_ignores_non_positive_intervals():
    meter = RateMeter()
    meter.tick(1.0)
    assert meter.tick(1.0) is None
    assert meter.tick(0.5) is None
    assert meter.tick(1.1) == pytest.approx(1.0 / 0.6)


def test_metric_window_statistics():
    window = MetricWindow(size=3)
    assert window.mean is None
    assert window.latest is None
    assert window.percentile(50) is None
    for value in (1.0, 2.0, 3.0, 4.0):
        window.add(value)
    assert len(window) == 3
    assert window.latest == pytest.approx(4.0)
    assert window.mean == pytest.approx(3.0)
    assert window.percentile(50) == pytest.approx(3.0)


def test_metric_window_rejects_invalid_values_and_sizes():
    with pytest.raises(ValueError):
        MetricWindow(0)
    window = MetricWindow()
    window.add(float("nan"))
    window.add(float("inf"))
    assert len(window) == 0


def test_telemetry_deduplicates_by_frame_sequence():
    telemetry = Telemetry()
    telemetry.update(result(1, detect_ms=10.0), now=0.1)
    telemetry.update(result(1, detect_ms=99.0), now=0.2)
    assert len(telemetry.detect_ms) == 1
    assert telemetry.detect_ms.latest == pytest.approx(10.0)
    telemetry.update(result(2, detect_ms=30.0), now=0.2)
    assert len(telemetry.detect_ms) == 2


def test_telemetry_ignores_missing_results():
    telemetry = Telemetry()
    telemetry.update(None, now=1.0)
    assert telemetry.summary()["detect_ms_p50"] is None


def test_age_is_clamped_and_measured_from_capture():
    telemetry = Telemetry()
    telemetry.update(result(1, capture_ts=1.0), now=1.05)
    assert telemetry.age_ms.latest == pytest.approx(50.0, abs=1e-6)
    telemetry.update(result(2, capture_ts=2.0), now=1.9)
    assert telemetry.age_ms.latest == pytest.approx(0.0)


def test_process_time_is_in_milliseconds():
    telemetry = Telemetry()
    telemetry.add_process_time(0.012)
    assert telemetry.process_ms.latest == pytest.approx(12.0)


def test_summary_and_hud_lines():
    telemetry = Telemetry()
    empty = telemetry.summary()
    assert set(empty) == {
        "fps",
        "detect_ms_p50",
        "detect_ms_p95",
        "infer_ms_p50",
        "infer_ms_p95",
        "age_ms_p50",
        "process_ms_p50",
    }
    assert telemetry.hud_lines()[0].startswith("fps")

    telemetry.tick(1.0)
    telemetry.tick(1.1)
    telemetry.update(result(1), now=1.1)
    telemetry.add_process_time(0.005)
    lines = telemetry.hud_lines()
    assert any(line.startswith("detect p50/p95") for line in lines)
    assert any(line.startswith("infer  p50/p95") for line in lines)
    assert any(line.startswith("age    p50") for line in lines)
    assert np.isfinite(telemetry.summary()["process_ms_p50"])
