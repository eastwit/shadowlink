"""Tests for the phyphox transport, buffer discovery, and value validation."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from shadowlink.phone.client import (
    MAX_RESPONSE_BYTES,
    PhyphoxClient,
    PhyphoxError,
    last_value,
    parse_host_port,
    pick_attitude,
    pick_vector,
    quaternion_wxyz,
)

# ------------------------------------------------------------------ addressing


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("192.0.2.10", ("192.0.2.10", 8080)),
        ("192.0.2.10:8090", ("192.0.2.10", 8090)),
        (" http://192.0.2.10:8090/ ", ("192.0.2.10", 8090)),
        ("HTTP://phone.local", ("phone.local", 8080)),
        ("http://[::1]:9000/", ("::1", 9000)),
    ],
)
def test_parse_host_port_accepts_supported_forms(text, expected):
    assert parse_host_port(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "ftp://192.0.2.10",
        "http://user:pass@192.0.2.10",
        "http://192.0.2.10/path",
        "192.0.2.10?scan=1",
        "192.0.2.10#fragment",
        "192.0.2.10:0",
        "192.0.2.10:70000",
        "192.0.2.10:abc",
        None,
    ],
)
def test_parse_host_port_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_host_port(text)


def test_parse_host_port_uses_the_requested_default_port():
    assert parse_host_port("192.0.2.10", default_port=9001) == ("192.0.2.10", 9001)
    with pytest.raises(ValueError):
        parse_host_port("192.0.2.10", default_port=0)


# -------------------------------------------------------------------- buffers


def test_last_value_reads_only_the_newest_finite_number():
    assert last_value({"buffer": [1.0, 2.5]}) == pytest.approx(2.5)
    assert last_value({"buffer": [1.0, float("nan")]}) is None
    assert last_value({"buffer": []}) is None
    assert last_value({"buffer": [True]}) is None
    assert last_value({"buffer": ["2"]}) is None
    assert last_value({"buffer": [1.0]}) == pytest.approx(1.0)
    assert last_value({"last": 4.0}) is None
    assert last_value(None) is None


def test_pick_attitude_prefers_a_group_with_a_time_axis():
    names = [
        "attitudeW",
        "attitudeX",
        "attitudeY",
        "attitudeZ",
        "time",
        "attitudeT",
        "rawW",
        "rawX",
        "rawY",
        "rawZ",
    ]
    picked = pick_attitude(names)
    assert picked is not None
    assert (picked.w, picked.x, picked.y, picked.z) == (
        "attitudeW",
        "attitudeX",
        "attitudeY",
        "attitudeZ",
    )
    assert picked.time == "attitudeT"
    assert picked.names == [
        "attitudeW",
        "attitudeX",
        "attitudeY",
        "attitudeZ",
        "attitudeT",
    ]


def test_pick_attitude_demotes_raw_groups_and_rejects_incomplete_ones():
    picked = pick_attitude(["rawW", "rawX", "rawY", "rawZ", "qW", "qX", "qY", "qZ"])
    assert picked is not None and picked.w == "qW"

    assert pick_attitude(["qW", "qX", "qY"]) is None
    assert pick_attitude(["gyroX", "gyroY", "gyroZ"]) is None
    assert pick_attitude([]) is None


def test_pick_attitude_handles_underscore_time_names():
    picked = pick_attitude(["qW", "qX", "qY", "qZ", "q_time"])
    assert picked is not None and picked.time == "q_time"


def test_pick_vector_selects_shortest_matching_axis_names():
    names = ["gyroscopeX", "gyroX", "gyroY", "gyroZ", "accelX", "accelY", "accelZ"]
    assert pick_vector(names, ("gyro", "gyr")) == {
        "x": "gyroX",
        "y": "gyroY",
        "z": "gyroZ",
    }
    assert pick_vector(names, ("accel", "acc")) == {
        "x": "accelX",
        "y": "accelY",
        "z": "accelZ",
    }
    assert pick_vector(names, ("mag",)) is None
    assert pick_vector(["gyroX", "gyroY"], ("gyro",)) is None


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0, 0.0, 0.0),
        (float("nan"), 0.0, 0.0, 0.0),
        (float("inf"), 0.0, 0.0, 0.0),
        (True, 0.0, 0.0, 0.0),
        (None, 0.0, 0.0, 0.0),
    ],
)
def test_quaternion_validation_rejects_invalid_readings(values):
    assert quaternion_wxyz(*values) is None


def test_quaternion_validation_normalises():
    quat = quaternion_wxyz(2.0, 0.0, 0.0, 0.0)
    assert quat is not None
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])


# ------------------------------------------------------------- loopback HTTP


@contextmanager
def phyphox_server(routes):
    """Serve fixed responses from ``routes`` on a loopback port."""
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            requests.append(self.path)
            route = self.path.split("?", 1)[0]
            response = routes.get(route)
            if response is None:
                status, body = 404, b"{}"
            elif isinstance(response, tuple):
                status, body = response
            else:
                status, body = 200, response
            if not isinstance(body, bytes):
                body = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


CONFIG = {
    "title": "ShadowLink test",
    "buffers": [
        {"name": "attitudeW"},
        {"name": "attitudeX"},
        {"name": "attitudeY"},
        {"name": "attitudeZ"},
    ],
}


def test_client_reads_config_values_and_control():
    routes = {
        "/config": CONFIG,
        "/get": {
            "status": {"measuring": True, "session": "abc"},
            "buffer": {"attitudeW": {"buffer": [0.0, 0.5]}},
        },
        "/control": {"status": {}},
    }
    with phyphox_server(routes) as (port, requests):
        client = PhyphoxClient("127.0.0.1", port, timeout_s=2.0)
        assert client.config()["title"] == "ShadowLink test"
        payload = client.values(["attitudeW", "attitudeX"])
        assert payload["status"]["session"] == "abc"
        client.control("start")

    assert requests == [
        "/config",
        "/get?attitudeW=&attitudeX=",
        "/control?cmd=start",
    ]


def test_client_uses_a_fresh_connection_per_request():
    routes = {"/config": CONFIG}
    with phyphox_server(routes) as (port, requests):
        client = PhyphoxClient("127.0.0.1", port, timeout_s=2.0)
        assert client.config()["title"] == "ShadowLink test"
        assert client.config()["title"] == "ShadowLink test"

    assert len(requests) == 2


@pytest.mark.parametrize(
    ("response", "needle"),
    [
        ((503, b"{}"), "HTTP 503"),
        ((200, b"not json"), "invalid JSON"),
        ((200, b"[]"), "not a JSON object"),
        ((200, b'{"x": NaN}'), "invalid JSON"),
        ((200, b'{"x": Infinity}'), "invalid JSON"),
    ],
)
def test_client_rejects_bad_responses(response, needle):
    with phyphox_server({"/config": response}) as (port, _):
        client = PhyphoxClient("127.0.0.1", port, timeout_s=2.0)
        with pytest.raises(PhyphoxError, match=needle):
            client.config()


def test_client_rejects_an_oversized_response():
    oversized = b"{" + b" " * (MAX_RESPONSE_BYTES + 8)
    with phyphox_server({"/config": (200, oversized)}) as (port, _):
        client = PhyphoxClient("127.0.0.1", port, timeout_s=2.0)
        with pytest.raises(PhyphoxError, match="too large"):
            client.config()


def test_client_reports_an_unreachable_host():
    with phyphox_server({}) as (port, _):
        pass  # the server is now closed; the port should refuse connections
    client = PhyphoxClient("127.0.0.1", port, timeout_s=0.5)
    with pytest.raises(PhyphoxError, match="cannot reach phone"):
        client.config()


def test_client_requires_a_non_empty_host_and_positive_timeout():
    with pytest.raises(ValueError):
        PhyphoxClient("", 8080)
    with pytest.raises(ValueError):
        PhyphoxClient("127.0.0.1", 8080, timeout_s=0.0)
    with pytest.raises(ValueError):
        PhyphoxClient("127.0.0.1", 70000)
