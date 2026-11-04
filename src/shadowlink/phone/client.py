"""phyphox Remote Access client and buffer discovery.

The phone runs the third-party `phyphox <https://phyphox.org>`_ app with Remote
Access enabled.  Its small HTTP server exposes the current experiment as JSON;
this module only reads that public interface.  Every request uses a fresh
connection because phyphox's JLHTTP server closes the socket after each reply.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import math
import socket
import urllib.parse
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from shadowlink.math3d import q_norm

DEFAULT_PORT = 8080
MAX_RESPONSE_BYTES = 1 << 20
LOW_PRIORITY = ("in", "off", "raw", "last", "cal", "tmp")


class PhyphoxError(RuntimeError):
    """Raised when the phone cannot be reached or returns invalid data."""


@dataclass(frozen=True, slots=True)
class AttitudeBuffers:
    """phyphox buffer names that together form one ``w, x, y, z`` quaternion."""

    w: str
    x: str
    y: str
    z: str
    time: str | None = None

    @property
    def names(self) -> list[str]:
        names = [self.w, self.x, self.y, self.z]
        if self.time is not None:
            names.append(self.time)
        return names


def _validate_port(port: int) -> int:
    if isinstance(port, bool):
        raise ValueError("port must be an integer")
    try:
        value = int(port)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid port {port!r}") from exc
    if not 1 <= value <= 65535:
        raise ValueError(f"port must be between 1 and 65535, got {value}")
    return value


def parse_host_port(text: str, default_port: int = DEFAULT_PORT) -> tuple[str, int]:
    """Parse ``host``, ``host:port``, or ``http://host:port/``.

    Only plain HTTP is supported.  Paths, credentials, query strings, and
    non-HTTP schemes are rejected so a typo cannot silently talk to the wrong
    service.
    """
    if not isinstance(text, str):
        raise ValueError("phone address must be a string")
    value = text.strip()
    if not value:
        raise ValueError("phone address is empty")
    if "://" in value:
        parts = urllib.parse.urlsplit(value)
        if parts.scheme.lower() != "http":
            raise ValueError("only http:// phone addresses are supported")
    else:
        parts = urllib.parse.urlsplit("//" + value)
    if parts.username is not None or parts.password is not None:
        raise ValueError("phone address must not contain credentials")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("phone address must be host[:port] without a path")
    host = parts.hostname
    if not host:
        raise ValueError(f"cannot parse phone address {text!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError(f"invalid port in phone address {text!r}") from exc
    if port is None:
        port = _validate_port(default_port)
    return host, _validate_port(port)


class PhyphoxClient:
    """Minimal blocking client for phyphox's Remote Access endpoints."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout_s: float = 1.0) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("phone host is empty")
        self.host = host.strip()
        self.port = _validate_port(port)
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("phone timeout must be finite and > 0")

    def config(self) -> dict:
        return self._get("/config")

    def values(self, names: Iterable[str]) -> dict:
        names = list(names)
        if not names:
            raise ValueError("at least one buffer name is required")
        query = "&".join(urllib.parse.quote(str(name), safe="") + "=" for name in names)
        return self._get("/get?" + query)

    def control(self, command: str) -> dict:
        return self._get("/control?cmd=" + urllib.parse.quote(str(command), safe=""))

    def _get(self, path: str) -> dict:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            connection.request(
                "GET",
                path,
                headers={"Accept": "application/json", "Connection": "close"},
            )
            response = connection.getresponse()
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if response.status != 200:
                raise PhyphoxError(f"HTTP {response.status} {response.reason}")
            if len(body) > MAX_RESPONSE_BYTES:
                raise PhyphoxError("phone response is too large")
            try:
                text = body.decode("utf-8")
                payload = json.loads(text, parse_constant=_reject_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise PhyphoxError(f"invalid JSON from phone: {exc}") from exc
            if not isinstance(payload, dict):
                raise PhyphoxError("phone response is not a JSON object")
            return payload
        except PhyphoxError:
            raise
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise PhyphoxError(f"cannot reach phone at {self.host}:{self.port}: {exc}") from exc
        finally:
            connection.close()


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON number {value!r}")


def last_value(entry) -> float | None:
    """Return the newest finite numeric value in one phyphox buffer entry."""
    if not isinstance(entry, dict):
        return None
    values = entry.get("buffer")
    if not isinstance(values, list) or not values:
        return None
    value = values[-1]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def quaternion_wxyz(w, x, y, z):
    """Validate and normalise four scalar buffer readings, or return ``None``."""
    values = []
    for value in (w, x, y, z):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        values.append(number)
    try:
        return q_norm(values)
    except ValueError:
        return None


def pick_attitude(names: Iterable[str]) -> AttitudeBuffers | None:
    """Find one complete ``w, x, y, z`` group, preferring groups with a time axis."""
    low = {name.lower(): name for name in names if isinstance(name, str)}
    groups: dict[str, dict[str, str]] = {}
    for name in names:
        if not isinstance(name, str) or len(name) < 2:
            continue
        lowered = name.lower()
        axis = lowered[-1]
        if axis not in "wxyz":
            continue
        groups.setdefault(lowered[:-1], {})[axis] = name

    best: AttitudeBuffers | None = None
    best_key: tuple[int, int, str] | None = None
    for prefix, axes in groups.items():
        if set(axes) != set("wxyz"):
            continue
        time_name = (
            low.get(prefix + "t")
            or low.get(prefix + "_t")
            or low.get(prefix + "time")
            or low.get(prefix + "_time")
        )
        score = (100 if time_name else 0) - 10 * sum(word in prefix for word in LOW_PRIORITY)
        key = (score, -len(prefix), prefix)
        if best_key is None or key > best_key:
            best_key = key
            best = AttitudeBuffers(
                w=axes["w"],
                x=axes["x"],
                y=axes["y"],
                z=axes["z"],
                time=time_name,
            )
    return best


def pick_vector(names: Iterable[str], keys: Iterable[str]) -> dict[str, str] | None:
    """Find an ``x/y/z`` buffer group whose names contain one of ``keys``."""
    lowered = [(name.lower(), name) for name in names if isinstance(name, str)]
    keys = tuple(key.lower() for key in keys)
    result: dict[str, str] = {}
    for axis in "xyz":
        candidates = []
        for low, original in lowered:
            if not (low.endswith(axis) or low.endswith("_" + axis)):
                continue
            if any(key in low for key in keys):
                candidates.append((len(low), low, original))
        if not candidates:
            return None
        result[axis] = min(candidates)[2]
    return result


def primary_ip() -> str:
    """Best-effort local LAN address without sending application data."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return ""
    finally:
        sock.close()


def discover(
    port: int = DEFAULT_PORT,
    timeout_s: float = 0.3,
    *,
    local_ip: str | None = None,
    probe: Callable[[str], tuple[str, str] | None] | None = None,
    max_workers: int = 128,
) -> list[tuple[str, str]]:
    """Scan the local ``/24`` for phyphox instances (opt-in only)."""
    address = local_ip if local_ip is not None else primary_ip()
    if not address:
        return []
    try:
        network = ipaddress.ip_network(address + "/24", strict=False)
    except ValueError:
        return []

    def default_probe(host: str):
        try:
            config = PhyphoxClient(host, port, timeout_s).config()
        except (PhyphoxError, ValueError):
            return None
        title = config.get("title")
        if "buffers" in config and isinstance(title, str):
            return (host, title)
        return None

    check = probe or default_probe
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        found = list(pool.map(check, (str(host) for host in network.hosts())))
    return sorted((item for item in found if item), key=lambda item: item[0])
