"""Background phyphox attitude source with a latest-value slot."""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from shadowlink.math3d import q_mul, q_norm
from shadowlink.phone.attitude import ImuFusion
from shadowlink.phone.client import (
    AttitudeBuffers,
    PhyphoxClient,
    PhyphoxError,
    discover,
    last_value,
    pick_attitude,
    pick_vector,
    quaternion_wxyz,
)

log = logging.getLogger(__name__)

# Android's native body frame is x=right, y=screen-up, z=out of the screen.
# The user-facing phone frame is x=charging port, y=phone width, z=screen
# normal, obtained by a -90 degree rotation about the body z axis.
PHONE_TWIST = np.array(
    [math.cos(-math.pi / 4.0), 0.0, 0.0, math.sin(-math.pi / 4.0)],
    dtype=np.float64,
)
DISCOVER_EVERY_S = 30.0


@dataclass(frozen=True, slots=True)
class PhoneSample:
    """One phone attitude sample in the documented phone body frame."""

    quat_wxyz: np.ndarray
    timestamp: float
    seq: int
    ready: bool
    source: str = "phone"

    def __post_init__(self) -> None:
        quat = q_norm(self.quat_wxyz)
        quat.setflags(write=False)
        object.__setattr__(self, "quat_wxyz", quat)


@dataclass(frozen=True, slots=True)
class PhoneHealth:
    state: str
    message: str = ""
    host: str = ""
    measuring: bool = False
    rate_hz: float = 0.0
    kind: str = ""
    title: str = ""


class PhoneSource(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...

    def latest(self) -> PhoneSample | None: ...

    @property
    def health(self) -> PhoneHealth: ...


class NullPhoneSource:
    """Inert source used when ``phone.source = none``."""

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def latest(self) -> PhoneSample | None:
        return None

    @property
    def health(self) -> PhoneHealth:
        return PhoneHealth(state="off")


class PhyphoxService:
    """Poll one phyphox experiment on a single daemon thread."""

    def __init__(
        self,
        config,
        *,
        client_factory: Callable[[str, int], PhyphoxClient] | None = None,
        discover_fn: Callable[[int, float], list[tuple[str, str]]] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._client_factory = client_factory or (
            lambda host, port: PhyphoxClient(host, port, float(config.timeout_s))
        )
        self._discover_fn = discover_fn or discover
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: PhyphoxClient | None = None
        self._sample: PhoneSample | None = None
        self._health = PhoneHealth(state="idle")
        self._seq = 0
        self._rate_times: deque[float] = deque(maxlen=256)
        self._host = (getattr(config, "host", None) or "").strip()
        self._title = ""
        self._kind = ""
        self._attitude: AttitudeBuffers | None = None
        self._fusion: ImuFusion | None = None
        self._poll_names: list[str] = []
        self._session = None
        self._last_phone_time: float | None = None
        self._measuring = False
        self._error_message: str | None = None
        self._last_discover = float("-inf")

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._sample = None
        self._seq = 0
        self._rate_times.clear()
        self._client = None
        self._host = (getattr(self.config, "host", None) or "").strip()
        self._title = ""
        self._kind = ""
        self._attitude = None
        self._fusion = None
        self._poll_names = []
        self._session = None
        self._last_phone_time = None
        self._measuring = False
        self._error_message = None
        self._last_discover = float("-inf")
        self._set_health(
            PhoneHealth(state="starting", message="starting phone source", host=self._host_label())
        )
        self._thread = threading.Thread(
            target=self._run,
            name="shadowlink-phone",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, float(self.config.timeout_s) + 0.5))
        self._thread = None
        self._set_health(PhoneHealth(state="stopped", host=self._host_label()))

    def latest(self) -> PhoneSample | None:
        with self._lock:
            sample = self._sample
        if sample is None:
            return None
        if self._clock() - sample.timestamp > float(self.config.stale_timeout_s):
            return None
        return sample

    @property
    def health(self) -> PhoneHealth:
        with self._lock:
            return self._health

    # -------------------------------------------------------------- worker
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._client is None and not self._connect():
                    self._stop.wait(max(0.05, float(self.config.retry_s)))
                    continue
                started = self._clock()
                try:
                    self._poll()
                    self._error_message = None
                except Exception as exc:  # noqa: BLE001 - surface via health
                    self._down(exc)
                self._refresh_health()
                period = 1.0 / max(1.0, float(self.config.poll_hz))
                elapsed = self._clock() - started
                self._stop.wait(max(0.0, period - elapsed))
        except Exception as exc:  # noqa: BLE001 - surface via health
            log.exception("phone service failed")
            self._error_message = _short(exc)
            self._set_health(
                PhoneHealth(
                    state="error",
                    message=self._error_message,
                    host=self._host_label(),
                    kind=self._kind,
                    title=self._title,
                )
            )
        finally:
            self._client = None

    def _connect(self, *, allow_rediscovery: bool = True) -> bool:
        if not self._host:
            if not bool(self.config.auto_discover):
                return self._fail("phone host not configured; pass --phone-host or set phone.host")
            if not self._discover_host(force=True):
                return self._fail("no phyphox device found on the local network")
        label = self._host_label()
        self._set_health(
            PhoneHealth(state="connecting", message=f"connecting to {label}", host=label)
        )
        try:
            client = self._client_factory(self._host, int(self.config.port))
            config = client.config()
            names = _buffer_names(config)
            attitude = pick_attitude(names)
            fusion = None
            if attitude is not None:
                kind = "attitude"
                poll_names = attitude.names
            else:
                gyro = pick_vector(names, ("gyro", "gyr"))
                accel = pick_vector(names, ("accel", "acc"))
                if not (gyro and accel):
                    raise PhyphoxError(
                        "no attitude quaternion (W/X/Y/Z) and no gyroscope+accelerometer "
                        "buffers; add an Attitude output in phyphox. Buffers: "
                        + ", ".join(names[:20])
                    )
                fusion = ImuFusion(gyro, accel, pick_vector(names, ("mag",)))
                kind = "imu"
                poll_names = fusion.names

            status = client.values([poll_names[0]]).get("status")
            status = status if isinstance(status, dict) else {}
            measuring = bool(status.get("measuring"))
            session = status.get("session")
            if not measuring and bool(self.config.auto_start):
                with contextlib.suppress(PhyphoxError):
                    client.control("start")

            title = config.get("title")
            self._client = client
            self._attitude = attitude
            self._fusion = fusion
            self._poll_names = poll_names
            self._kind = kind
            self._title = title if isinstance(title, str) else ""
            self._session = session
            self._last_phone_time = None
            self._measuring = measuring
            self._error_message = None
            self._set_health(
                PhoneHealth(
                    state="connecting",
                    message=f"connected to {label} ({kind})",
                    host=label,
                    measuring=measuring,
                    kind=kind,
                    title=self._title,
                )
            )
            return True
        except Exception as exc:  # noqa: BLE001 - report through health
            if (
                allow_rediscovery
                and bool(self.config.auto_discover)
                and self._discover_host(exclude=self._host)
            ):
                log.info("rediscovered phyphox at %s", self._host)
                return self._connect(allow_rediscovery=False)
            return self._fail(f"cannot connect to {label}: {_short(exc)}")

    def _poll(self) -> None:
        client = self._client
        if client is None:
            return
        response = client.values(self._poll_names)
        status = response.get("status")
        status = status if isinstance(status, dict) else {}
        measuring = bool(status.get("measuring"))
        session = status.get("session")
        if session is not None and self._session is not None and session != self._session:
            log.info("phyphox session changed; re-reading experiment configuration")
            self._client = None
            self._attitude = None
            self._fusion = None
            self._poll_names = []
            self._session = session
            self._last_phone_time = None
            return
        if session is not None:
            self._session = session
        self._measuring = measuring
        if not measuring:
            return

        buffers = response.get("buffer")
        if not isinstance(buffers, dict):
            raise PhyphoxError("phone /get response has no buffer object")
        if self._kind == "attitude":
            attitude = self._attitude
            if attitude is None:
                return
            quaternion = quaternion_wxyz(
                last_value(buffers.get(attitude.w)),
                last_value(buffers.get(attitude.x)),
                last_value(buffers.get(attitude.y)),
                last_value(buffers.get(attitude.z)),
            )
            if quaternion is None:
                return
            phone_time = (
                last_value(buffers.get(attitude.time)) if attitude.time is not None else None
            )
            if phone_time is not None:
                if self._last_phone_time is not None and phone_time <= self._last_phone_time:
                    return
                self._last_phone_time = phone_time
        else:
            fusion = self._fusion
            if fusion is None:
                return
            values = {}
            for name in self._poll_names:
                value = last_value(buffers.get(name))
                if value is None:
                    return
                values[name] = value
            quaternion = fusion.update(values, self._clock())
            if quaternion is None:
                return

        now = self._clock()
        quaternion = q_norm(q_mul(quaternion, PHONE_TWIST))
        self._seq += 1
        sample = PhoneSample(
            quat_wxyz=quaternion,
            timestamp=now,
            seq=self._seq,
            ready=True,
        )
        with self._lock:
            self._sample = sample
        self._rate_times.append(now)

    def _down(self, exc: Exception) -> None:
        self._client = None
        self._attitude = None
        self._fusion = None
        self._poll_names = []
        self._error_message = f"phone connection lost: {_short(exc)}"
        log.warning("phone connection lost: %s", exc)

    # ------------------------------------------------------------- helpers
    def _fail(self, message: str) -> bool:
        self._error_message = message
        self._set_health(
            PhoneHealth(
                state="error",
                message=message,
                host=self._host_label(),
                kind=self._kind,
                title=self._title,
            )
        )
        return False

    def _discover_host(self, *, force: bool = False, exclude: str = "") -> bool:
        now = self._clock()
        if not force and now - self._last_discover < DISCOVER_EVERY_S:
            return False
        self._last_discover = now
        try:
            found = self._discover_fn(
                int(self.config.port),
                min(0.35, float(self.config.timeout_s)),
            )
        except Exception as exc:  # noqa: BLE001 - discovery is best effort
            log.debug("phyphox discovery failed: %s", exc)
            return False
        for host, _title in found:
            if host and host != exclude:
                self._host = host
                return True
        return False

    def _refresh_health(self) -> None:
        now = self._clock()
        with self._lock:
            sample = self._sample
        rate = self._rate_hz(now)
        if self._error_message:
            state, message = "error", self._error_message
        elif self._client is None:
            state, message = "connecting", "reconnecting to the phone"
        elif sample is None:
            state, message = "connecting", "waiting for phone data"
        elif not self._measuring:
            state, message = "paused", "measurement paused on the phone"
        elif now - sample.timestamp > float(self.config.stale_timeout_s):
            state, message = "stale", "no fresh samples from the phone"
        else:
            state, message = "running", "following"
        self._set_health(
            PhoneHealth(
                state=state,
                message=message,
                host=self._host_label(),
                measuring=self._measuring,
                rate_hz=rate,
                kind=self._kind,
                title=self._title,
            )
        )

    def _rate_hz(self, now: float) -> float:
        cutoff = now - 1.0
        while self._rate_times and self._rate_times[0] < cutoff:
            self._rate_times.popleft()
        if len(self._rate_times) < 2:
            return 0.0
        span = self._rate_times[-1] - self._rate_times[0]
        return 0.0 if span <= 0 else (len(self._rate_times) - 1) / span

    def _set_health(self, health: PhoneHealth) -> None:
        with self._lock:
            self._health = health

    def _host_label(self) -> str:
        return f"{self._host}:{int(self.config.port)}" if self._host else ""


def _buffer_names(config: dict) -> list[str]:
    buffers = config.get("buffers")
    if not isinstance(buffers, list):
        raise PhyphoxError("/config has no buffer list")
    names = [
        entry.get("name")
        for entry in buffers
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    ]
    if not names:
        raise PhyphoxError("phone experiment has no buffers")
    return names


def _short(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    return text.splitlines()[0][:160]
