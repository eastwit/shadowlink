"""OpenNI2 depth stream with one background reader thread.

OpenNI setup, teardown, and the DLL search path are owned by the main thread.
The worker only calls ``read_frame`` and publishes an immutable snapshot, so a
blocked read can never hold up the control loop.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from shadowlink.config import CaptureConfig
from shadowlink.rgbd.intrinsics import intrinsics_from_device
from shadowlink.types import DepthIntrinsics

log = logging.getLogger(__name__)

OPENNI_DLL_NAMES = ("OpenNI2.dll", "libOpenNI2.dll")
_CANDIDATE_DIRS = (
    r"C:\Program Files\Orbbec\OpenNI2\Redist",
    r"C:\Program Files\Orbbec\OpenNI2\sdk\libs",
    r"C:\Program Files\Orbbec\OpenNI2",
    r"C:\Program Files (x86)\Orbbec\OpenNI2\Redist",
    r"C:\OpenNI2\Redist",
    r"C:\Program Files\OpenNI2\Redist",
)
WARMUP_FRAMES = 5
_FPS_WINDOW_S = 0.5
_READ_POLL_S = 0.1
_JOIN_TIMEOUT_S = 0.5
_STOP_JOIN_TIMEOUT_S = 1.5


class DepthError(RuntimeError):
    """Raised when the depth sensor cannot be opened or configured."""


@dataclass(frozen=True, slots=True)
class DepthFrame:
    """One depth image in millimetres, in the raw sensor frame."""

    depth_mm: np.ndarray
    intrinsics: DepthIntrinsics
    seq: int
    capture_ts: float

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_mm)
        if depth.ndim != 2:
            raise ValueError(f"expected HxW depth image, got {depth.shape}")
        if depth.dtype != np.uint16:
            raise TypeError(f"expected uint16 depth image, got {depth.dtype}")
        if self.intrinsics.width != depth.shape[1] or self.intrinsics.height != depth.shape[0]:
            raise ValueError("depth intrinsics do not match the image size")
        if not math.isfinite(self.capture_ts):
            raise ValueError("depth timestamp must be finite")
        depth.setflags(write=False)
        object.__setattr__(self, "depth_mm", depth)


@dataclass(frozen=True, slots=True)
class DepthHealth:
    state: str = "idle"  # idle | starting | running | error | stopped
    message: str = ""
    fps: float = 0.0
    frames: int = 0
    errors: int = 0


class _LatestDepth:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: DepthFrame | None = None

    def put(self, item: DepthFrame) -> None:
        with self._lock:
            self._item = item

    def get(self) -> DepthFrame | None:
        with self._lock:
            return self._item

    def clear(self) -> None:
        with self._lock:
            self._item = None


def resolve_depth_sdk(explicit: str | None = None) -> Path:
    """Find a directory containing ``OpenNI2.dll``.

    Resolution order: explicit value, ``OPENNI2_REDIST64``, ``OPENNI2_REDIST``,
    then common Windows installation directories.  User-specific paths must be
    supplied explicitly; nothing is hard-coded.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for name in ("OPENNI2_REDIST64", "OPENNI2_REDIST"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(Path(value) for value in _CANDIDATE_DIRS)

    tried: list[str] = []
    for candidate in candidates:
        tried.append(str(candidate))
        if any((candidate / name).is_file() for name in OPENNI_DLL_NAMES):
            return candidate
    raise DepthError(
        "cannot find OpenNI2.dll; pass --depth-sdk or set OPENNI2_REDIST. "
        f"Tried: {', '.join(tried)}"
    )


def _device_name(device) -> str | None:
    try:
        info = device.get_device_info()
    except Exception:  # pragma: no cover - device dependent
        return None
    name = getattr(info, "name", None)
    if isinstance(name, bytes):
        name = name.decode("utf-8", "replace")
    text = str(name).strip() if name is not None else ""
    return text or None


def _import_openni():
    try:
        from openni import _openni2 as c_api
        from openni import openni2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DepthError(
            "the `openni` package is required for --source rgbd; "
            "install it in the active environment (pip install openni)"
        ) from exc
    return openni2, c_api


class DepthService:
    def __init__(
        self,
        cfg: CaptureConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self._clock = clock
        self._latest = _LatestDepth()
        self._health = DepthHealth()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._openni2 = None
        self._c_api = None
        self._device = None
        self._device_name: str | None = None
        self._stream = None
        self._dll_handle = None
        self._intrinsics: DepthIntrinsics | None = None
        self._seq = 0
        self._started = False
        self._closed = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise DepthError("depth service is closed")
        sdk = resolve_depth_sdk(self.cfg.depth_sdk)
        with self._lock:
            self._health = DepthHealth(state="starting")
        try:
            if os.name == "nt":
                self._dll_handle = os.add_dll_directory(str(sdk))
            self._openni2, self._c_api = _import_openni()
            self._openni2.initialize(str(sdk))
            self._device = self._openni2.Device.open_any()
            self._device_name = _device_name(self._device)
            self._stream = self._device.create_depth_stream()
            try:
                self._stream.set_video_mode(
                    self._c_api.OniVideoMode(
                        pixelFormat=self._openni2.PIXEL_FORMAT_DEPTH_1_MM,
                        resolutionX=int(self.cfg.depth_width),
                        resolutionY=int(self.cfg.depth_height),
                        fps=int(self.cfg.depth_fps),
                    )
                )
            except Exception as exc:  # pragma: no cover - device dependent
                log.warning("depth video mode rejected (%s); using the default mode", exc)
            self._stream.start()
        except DepthError:
            self._cleanup_openni()
            raise
        except Exception as exc:
            self._cleanup_openni()
            raise DepthError(f"cannot open the depth sensor: {exc}") from exc

        self._started = True
        self._stop.clear()
        with self._lock:
            self._health = DepthHealth(state="running")
        self._thread = threading.Thread(
            target=self._read_loop,
            name="shadowlink-depth",
            daemon=True,
        )
        self._thread.start()
        log.info("depth stream started from %s", sdk)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        thread = self._thread
        if thread is not None:
            # The reader wakes every `_READ_POLL_S`, so it exits without needing
            # the stream to be stopped first.  Some OpenNI2 builds do not wake a
            # blocked `read_frame` when the stream is stopped from another
            # thread, so stopping is only the fallback.
            thread.join(timeout=_JOIN_TIMEOUT_S)
            if thread.is_alive():
                stream = self._stream
                if stream is not None:
                    try:
                        stream.stop()
                    except Exception:  # pragma: no cover - best effort
                        log.debug("depth stream stop failed", exc_info=True)
                thread.join(timeout=_STOP_JOIN_TIMEOUT_S)
            if thread.is_alive():
                log.warning("depth thread did not stop; skipping OpenNI2 unload")
                self._close_dll_handle()
                self._thread = None
                self._started = False
                return
        self._cleanup_openni()
        self._close_dll_handle()
        self._thread = None
        self._started = False
        with self._lock:
            if self._health.state != "error":
                self._health = replace(self._health, state="stopped")

    # ----------------------------------------------------------------- reads
    def latest(self) -> DepthFrame | None:
        return self._latest.get()

    @property
    def health(self) -> DepthHealth:
        with self._lock:
            return replace(self._health)

    @property
    def device_name(self) -> str | None:
        return self._device_name

    @property
    def intrinsics(self) -> DepthIntrinsics | None:
        return self._intrinsics

    def _read_loop(self) -> None:
        frames = 0
        published = 0
        errors = 0
        window_start = self._clock()
        window_frames = 0
        fps = 0.0
        stream = self._stream
        try:
            while not self._stop.is_set():
                try:
                    ready = self._openni2.wait_for_any_stream([stream], timeout=_READ_POLL_S)
                    if self._stop.is_set():
                        break
                    if ready is None:
                        continue
                    frame = stream.read_frame()
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    errors += 1
                    with self._lock:
                        self._health = DepthHealth(
                            state="error",
                            message=str(exc),
                            fps=fps,
                            frames=published,
                            errors=errors,
                        )
                    log.error("depth read failed: %s", exc)
                    break

                depth = (
                    np.frombuffer(frame.get_buffer_as_uint16(), np.uint16)[
                        : frame.height * frame.width
                    ]
                    .reshape(frame.height, frame.width)
                    .copy()
                )
                now = self._clock()
                frames += 1
                if frames > WARMUP_FRAMES:
                    depth.setflags(write=False)
                    if self._intrinsics is None:
                        self._intrinsics = intrinsics_from_device(
                            self._stream,
                            frame.width,
                            frame.height,
                            self.cfg,
                        )
                    self._latest.put(
                        DepthFrame(
                            depth_mm=depth,
                            intrinsics=self._intrinsics,
                            seq=self._seq,
                            capture_ts=now,
                        )
                    )
                    self._seq += 1
                    published += 1
                    window_frames += 1

                elapsed = now - window_start
                if elapsed >= _FPS_WINDOW_S:
                    fps = window_frames / elapsed
                    window_start = now
                    window_frames = 0
                    with self._lock:
                        self._health = DepthHealth(
                            state="running",
                            fps=fps,
                            frames=published,
                            errors=errors,
                        )
        finally:
            with self._lock:
                if self._health.state == "running":
                    self._health = replace(self._health, state="stopped")

    # --------------------------------------------------------------- cleanup
    def _cleanup_openni(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("depth stream close failed", exc_info=True)
        device, self._device = self._device, None
        self._device_name = None
        if device is not None:
            try:
                device.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("depth device close failed", exc_info=True)
        openni2, self._openni2 = self._openni2, None
        self._c_api = None
        if openni2 is not None:
            try:
                openni2.unload()
            except Exception:  # pragma: no cover - best effort
                log.debug("OpenNI2 unload failed", exc_info=True)

    def _close_dll_handle(self) -> None:
        handle, self._dll_handle = self._dll_handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:  # pragma: no cover - best effort
                log.debug("DLL directory handle close failed", exc_info=True)
