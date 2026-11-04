"""Optional live test against a real phone running phyphox.

Enable with ``SHADOWLINK_TEST_PHONE=1`` and ``SHADOWLINK_TEST_PHONE_HOST`` set
to the phone's IP (``host`` or ``host:port``).  Skipped by default so CI and
normal test runs never need a device.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from shadowlink.config import PhoneConfig
from shadowlink.phone import PhyphoxService
from shadowlink.phone.client import parse_host_port

pytestmark = [
    pytest.mark.phone,
    pytest.mark.skipif(
        os.environ.get("SHADOWLINK_TEST_PHONE") != "1",
        reason="set SHADOWLINK_TEST_PHONE=1 to test a physical phone",
    ),
]


def test_live_phone_produces_finite_attitude_samples():
    address = os.environ.get("SHADOWLINK_TEST_PHONE_HOST", "").strip()
    if not address:
        pytest.skip("set SHADOWLINK_TEST_PHONE_HOST to the phone address")
    host, port = parse_host_port(address)
    config = PhoneConfig(
        source="phyphox",
        host=host,
        port=port,
        poll_hz=30.0,
        auto_start=True,
        stale_timeout_s=1.0,
    )
    service = PhyphoxService(config)
    try:
        service.start()
        deadline = time.monotonic() + 15.0
        sample = None
        while time.monotonic() < deadline:
            sample = service.latest()
            if sample is not None:
                break
            time.sleep(0.05)
    finally:
        service.close()

    assert sample is not None, service.health
    assert sample.ready
    assert np.isfinite(sample.quat_wxyz).all()
    assert float(np.linalg.norm(sample.quat_wxyz)) == pytest.approx(1.0, abs=1e-6)
