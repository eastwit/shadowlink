"""Diagnostic command for the phyphox phone attitude source."""

from __future__ import annotations

import json
import time

import numpy as np

from shadowlink.math3d import q_conj, q_euler_deg, q_mul, q_to_axis_angle
from shadowlink.phone.client import (
    PhyphoxClient,
    PhyphoxError,
    discover,
    pick_attitude,
    pick_vector,
)
from shadowlink.phone.service import PhyphoxService


def run_phone_test(
    config,
    *,
    seconds: float = 10.0,
    list_buffers: bool = False,
    as_json: bool = False,
) -> int:
    if list_buffers:
        return _list_buffers(config)
    service = PhyphoxService(config)
    service.start()
    zero = None
    last = None
    deadline = time.monotonic() + max(0.0, seconds)
    printed = 0.0
    health = service.health
    try:
        while time.monotonic() < deadline:
            sample = service.latest()
            if sample is not None:
                last = sample
                if zero is None:
                    zero = sample.quat_wxyz.copy()
            now = time.monotonic()
            if now - printed >= 0.5:
                printed = now
                health = service.health
                payload = _payload(sample, zero, health)
                if as_json:
                    print(json.dumps(payload), flush=True)
                else:
                    print(_format(payload), flush=True)
            time.sleep(0.05)
    finally:
        health = service.health
        service.close()

    if last is None:
        print(f"no phone data ({health.state}: {health.message})")
        return 1
    print(f"final: {health.state} ({health.message})")
    return 0


def _list_buffers(config) -> int:
    host = (getattr(config, "host", None) or "").strip()
    if not host:
        if not bool(getattr(config, "auto_discover", False)):
            print("no phone host; pass --host IP[:PORT] or --scan")
            return 2
        hits = discover(int(config.port), min(0.35, float(config.timeout_s)))
        if not hits:
            print("no phyphox device found on the local network")
            return 1
        host, title = hits[0]
        print(f"found {host} ({title})")
    client = PhyphoxClient(host, int(config.port), float(config.timeout_s))
    try:
        payload = client.config()
    except PhyphoxError as exc:
        print(f"cannot read phone configuration: {exc}")
        return 1
    buffers = payload.get("buffers")
    names = (
        [
            entry.get("name")
            for entry in buffers
            if isinstance(entry, dict) and isinstance(entry.get("name"), str)
        ]
        if isinstance(buffers, list)
        else []
    )
    title = payload.get("title")
    print(f"phone: {host}:{int(config.port)}")
    print(f"title: {title if isinstance(title, str) else '-'}")
    print("buffers:")
    for name in names:
        print(f"  {name}")
    if not names:
        print("error: the experiment exposes no buffers")
        return 1

    attitude = pick_attitude(names)
    if attitude is not None:
        print(
            "selected: attitude "
            f"w={attitude.w} x={attitude.x} y={attitude.y} z={attitude.z}"
            + (f" t={attitude.time}" if attitude.time else "")
        )
        return 0
    gyro = pick_vector(names, ("gyro", "gyr"))
    accel = pick_vector(names, ("accel", "acc"))
    if gyro and accel:
        mag = pick_vector(names, ("mag",))
        print(f"selected: imu gyro={gyro} accel={accel}" + (f" mag={mag}" if mag else ""))
        return 0
    print("error: no attitude quaternion (W/X/Y/Z) and no gyroscope+accelerometer buffers")
    print("add an Attitude output to the phyphox experiment")
    return 1


def _payload(sample, zero, health) -> dict:
    payload = {
        "state": health.state,
        "message": health.message,
        "host": health.host,
        "kind": health.kind,
        "title": health.title,
        "measuring": bool(health.measuring),
        "rate_hz": round(float(health.rate_hz), 1),
        "seq": None,
        "quat": None,
        "euler_deg": None,
        "relative_axis": None,
        "relative_angle_deg": None,
    }
    if sample is None:
        return payload
    payload["seq"] = int(sample.seq)
    payload["quat"] = [round(float(value), 5) for value in sample.quat_wxyz]
    payload["euler_deg"] = [round(value, 2) for value in q_euler_deg(sample.quat_wxyz)]
    if zero is not None:
        relative = q_mul(q_conj(zero), sample.quat_wxyz)
        axis, angle = q_to_axis_angle(relative)
        payload["relative_axis"] = [round(float(value), 3) for value in axis]
        payload["relative_angle_deg"] = round(np.degrees(angle), 2)
    return payload


def _format(payload: dict) -> str:
    head = (
        f"{payload['state']:<10} rate {payload['rate_hz']:5.1f} Hz "
        f"{payload['kind'] or '-':<8} {payload['host'] or '-':<21} "
        f"| {payload['message']}"
    )
    if payload["quat"] is None:
        return head
    euler = payload["euler_deg"]
    axis = payload["relative_axis"]
    angle = payload["relative_angle_deg"]
    relative = "n/a"
    if axis is not None:
        relative = f"axis ({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}) angle {angle:+6.1f} deg"
    return (
        f"{head}\n  euler p/r/y ({euler[0]:+6.1f},{euler[1]:+6.1f},{euler[2]:+6.1f}) rel {relative}"
    )
