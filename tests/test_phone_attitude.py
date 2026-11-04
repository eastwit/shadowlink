"""Tests for the Mahony attitude fallback used by raw-IMU phyphox experiments."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shadowlink.math3d import q_rotate
from shadowlink.phone.attitude import Attitude, ImuFusion

GYRO = {"x": "gyroX", "y": "gyroY", "z": "gyroZ"}
ACCEL = {"x": "accelX", "y": "accelY", "z": "accelZ"}
MAG = {"x": "magX", "y": "magY", "z": "magZ"}


def levels(accel=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0)):
    return {
        "gyroX": gyro[0],
        "gyroY": gyro[1],
        "gyroZ": gyro[2],
        "accelX": accel[0],
        "accelY": accel[1],
        "accelZ": accel[2],
    }


def test_reset_from_accel_aligns_gravity():
    flat = Attitude()
    flat.reset_from_accel([0.0, 0.0, 9.81])
    assert flat.initialized
    np.testing.assert_allclose(q_rotate(flat.q, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-9)

    tilted = Attitude()
    tilted.reset_from_accel([0.0, 9.81, 0.0])
    np.testing.assert_allclose(q_rotate(tilted.q, [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-9)


def test_filter_ignores_a_zero_accelerometer_during_reset():
    attitude = Attitude()
    attitude.reset_from_accel([0.0, 0.0, 0.0])
    assert not attitude.initialized
    np.testing.assert_allclose(attitude.q, [1.0, 0.0, 0.0, 0.0])


def test_gyroscope_integrates_a_quarter_turn():
    attitude = Attitude()
    attitude.reset_from_accel([0.0, 0.0, 9.81])
    for _ in range(100):
        attitude.update([0.0, 0.0, 9.81], [0.0, 0.0, math.pi / 2.0], 0.01)
    rotated = q_rotate(attitude.q, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=0.02)
    assert float(np.linalg.norm(attitude.q)) == pytest.approx(1.0)


def test_imu_fusion_waits_for_initialisation_then_returns_gravity_aligned_quaternions():
    fusion = ImuFusion(GYRO, ACCEL)
    assert fusion.names == ["gyroX", "gyroY", "gyroZ", "accelX", "accelY", "accelZ"]
    assert fusion.update(levels(), 0.0) is None
    result = None
    for step in range(1, 30):
        result = fusion.update(levels(), step * 0.01)
    assert result is not None
    np.testing.assert_allclose(q_rotate(result, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-6)


def test_imu_fusion_rejects_incomplete_readings():
    fusion = ImuFusion(GYRO, ACCEL)
    fusion.update(levels(), 0.0)
    for step in range(1, 30):
        fusion.update(levels(), step * 0.01)
    values = levels()
    del values["gyroY"]
    assert fusion.update(values, 0.4) is None


def test_magnetometer_leveling_turns_the_body_x_axis_toward_north():
    fusion = ImuFusion(GYRO, ACCEL, MAG)
    assert fusion.names[-3:] == ["magX", "magY", "magZ"]
    result = None
    for step in range(80):
        values = levels()
        values.update({"magX": 1.0, "magY": 0.0, "magZ": 0.0})
        result = fusion.update(values, step * 0.01)
    assert result is not None
    rotated = q_rotate(result, [1.0, 0.0, 0.0])
    assert rotated[1] > 0.1
    assert float(np.linalg.norm(result)) == pytest.approx(1.0)


def test_fusion_clamps_the_integration_step():
    fusion = ImuFusion(GYRO, ACCEL)
    fusion.update(levels(), 0.0)
    for step in range(1, 30):
        fusion.update(levels(gyro=(0.0, 0.0, 1.0)), step * 0.01)
    result = fusion.update(levels(gyro=(0.0, 0.0, 1.0)), 10.0)
    assert result is not None
    # A 10 s gap is integrated as at most 0.2 s, so this stays a small yaw.
    assert abs(float(q_rotate(result, [1.0, 0.0, 0.0])[1])) < 0.4
