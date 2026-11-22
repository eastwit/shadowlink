"""Tests for the background phyphox service using injected fake clients."""

from __future__ import annotations

import threading
import time

import numpy as np

from shadowlink.config import PhoneConfig
from shadowlink.math3d import q_mul, q_norm
from shadowlink.phone.client import PhyphoxError
from shadowlink.phone.service import PHONE_TWIST, PhyphoxService

ATTITUDE_NAMES = ("attitudeW", "attitudeX", "attitudeY", "attitudeZ", "attitudeT")
QUAT = np.array([0.5, 0.5, 0.5, 0.5])


def wait_for(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class FakeClient:
    def __init__(self, *, measuring: bool = True, session: str = "s1") -> None:
        self.measuring = measuring
        self.session = session
        self.controls: list[str] = []
        self.values_calls = 0
        self.fail = False
        self.time = 0.0

    def config(self) -> dict:
        return {
            "title": "Fake phone",
            "buffers": [{"name": name} for name in ATTITUDE_NAMES],
        }

    def values(self, names) -> dict:
        if self.fail:
            raise PhyphoxError("connection reset")
        self.values_calls += 1
        self.time += 0.01
        axes = {"W": QUAT[0], "X": QUAT[1], "Y": QUAT[2], "Z": QUAT[3]}
        buffer = {}
        for name in names:
            if name == "attitudeT":
                buffer[name] = {"buffer": [self.time]}
            else:
                buffer[name] = {"buffer": [axes[name[-1]]]}
        return {
            "status": {"measuring": self.measuring, "session": self.session},
            "buffer": buffer,
        }

    def control(self, command: str) -> dict:
        self.controls.append(command)
        if command == "start":
            self.measuring = True
        return {"status": {}}


class Factory:
    def __init__(self, client: FakeClient | None = None, **kwargs) -> None:
        self.client = client or FakeClient(**kwargs)
        self.calls = 0
        self.hosts: list[tuple[str, int]] = []

    def __call__(self, host: str, port: int) -> FakeClient:
        self.calls += 1
        self.hosts.append((host, port))
        return self.client


def phone_config(**overrides) -> PhoneConfig:
    values = {
        "source": "phyphox",
        "host": "192.0.2.10",
        "poll_hz": 200.0,
        "timeout_s": 0.2,
        "retry_s": 0.05,
        "stale_timeout_s": 0.3,
    }
    values.update(overrides)
    return PhoneConfig(**values)


def make_service(factory: Factory, config: PhoneConfig | None = None, **kwargs) -> PhyphoxService:
    return PhyphoxService(
        config or phone_config(),
        client_factory=factory,
        discover_fn=kwargs.pop("discover_fn", lambda port, timeout: []),
        **kwargs,
    )


def test_service_publishes_an_attitude_sample_in_the_phone_frame():
    factory = Factory()
    service = make_service(factory)
    try:
        service.start()
        assert wait_for(lambda: service.latest() is not None)
        sample = service.latest()
        assert sample is not None
        assert sample.ready and sample.seq >= 1
        np.testing.assert_allclose(sample.quat_wxyz, q_mul(q_norm(QUAT), PHONE_TWIST), atol=1e-9)
        assert wait_for(lambda: service.health.state == "running")
        assert wait_for(lambda: service.health.rate_hz > 0.0)
        health = service.health
        assert health.host == "192.0.2.10:8080"
        assert health.kind == "attitude"
        assert health.measuring is True
    finally:
        service.close()

    assert factory.calls == 1
    assert not any(t.name == "shadowlink-phone" for t in threading.enumerate())


def test_service_starts_a_paused_experiment():
    client = FakeClient(measuring=False)
    factory = Factory(client)
    service = make_service(factory)
    try:
        service.start()
        assert wait_for(lambda: client.controls == ["start"])
        assert wait_for(lambda: service.latest() is not None)
        assert service.health.measuring is True
    finally:
        service.close()


def test_service_does_not_start_a_paused_experiment_when_disabled():
    client = FakeClient(measuring=False)
    factory = Factory(client)
    service = make_service(factory, phone_config(auto_start=False))
    try:
        service.start()
        assert wait_for(lambda: service.health.measuring is False)
        assert wait_for(lambda: service.health.state in {"connecting", "paused"})
        time.sleep(0.1)
        assert client.controls == []
        assert service.latest() is None
    finally:
        service.close()


def test_service_reloads_configuration_when_the_session_changes():
    client = FakeClient()
    factory = Factory(client)
    service = make_service(factory)
    try:
        service.start()
        assert wait_for(lambda: service.latest() is not None)
        assert factory.calls == 1
        client.session = "s2"
        assert wait_for(lambda: factory.calls >= 2)
        assert wait_for(lambda: service.latest() is not None)
    finally:
        service.close()


def test_service_freezes_and_reconnects_when_polling_fails():
    client = FakeClient()
    factory = Factory(client)
    service = make_service(factory)
    try:
        service.start()
        assert wait_for(lambda: service.latest() is not None)
        client.fail = True
        assert wait_for(lambda: service.health.state == "error")
        assert service.latest() is None or wait_for(lambda: service.latest() is None, 1.0)
        assert wait_for(lambda: factory.calls >= 2)
        client.fail = False
        assert wait_for(lambda: service.latest() is not None)
        assert wait_for(lambda: service.health.state == "running")
    finally:
        service.close()


def test_service_reports_a_missing_host_without_scanning():
    discovered = []
    factory = Factory()
    service = make_service(
        factory,
        phone_config(host=None, auto_discover=False),
        discover_fn=lambda port, timeout: discovered.append(port) or [],
    )
    try:
        service.start()
        assert wait_for(lambda: service.health.state == "error")
        assert "host not configured" in service.health.message
    finally:
        service.close()

    assert discovered == []
    assert factory.calls == 0


def test_service_discovers_a_host_only_when_enabled():
    discovered = []
    factory = Factory()
    service = make_service(
        factory,
        phone_config(host=None, auto_discover=True),
        discover_fn=lambda port, timeout: discovered.append(port) or [("192.0.2.55", "Fake phone")],
    )
    try:
        service.start()
        assert wait_for(lambda: service.latest() is not None)
        assert service.health.host == "192.0.2.55:8080"
    finally:
        service.close()

    assert discovered and discovered[0] == 8080
    assert factory.hosts[0][0] == "192.0.2.55"


def test_service_close_is_idempotent_and_sets_a_stopped_state():
    service = make_service(Factory())
    service.start()
    service.close()
    service.close()
    assert service.health.state == "stopped"
    assert not any(t.name == "shadowlink-phone" for t in threading.enumerate())


def test_latest_returns_none_after_the_stale_timeout():
    factory = Factory()
    service = make_service(factory, phone_config(stale_timeout_s=0.05))
    try:
        service.start()
        assert wait_for(lambda: service.latest() is not None)
        factory.client.fail = True
        assert wait_for(lambda: service.latest() is None, 1.0)
    finally:
        service.close()
