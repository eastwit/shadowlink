"""Camera, video-file, and synthetic frame sources.

Frames are mirrored exactly once, here, before they reach perception or the
preview.  Everything downstream therefore agrees on the coordinate system, and
``FramePacket.mirrored`` records the decision for handedness correction.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from shadowlink.config import CaptureConfig
from shadowlink.types import FramePacket

log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """Raised when a frame source cannot be opened or configured."""


def _parse_camera(value: str) -> int | str:
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _api_preference(name: str) -> int:
    key = f"CAP_{name.upper()}"
    if name == "any":
        return int(getattr(cv2, "CAP_ANY", 0))
    code = getattr(cv2, key, None)
    if code is None:
        raise CaptureError(f"OpenCV backend {name!r} is not available in this build")
    return int(code)


class OpenCVFrameSource:
    """A ``cv2.VideoCapture``-backed source."""

    def __init__(self, cfg: CaptureConfig, clock: Callable[[], float] = time.monotonic) -> None:
        if cfg.source == "video":
            if not cfg.video:
                raise CaptureError("capture.source=video requires a video path")
            target = cfg.video
        else:
            target = _parse_camera(cfg.camera)

        self.cfg = cfg
        self._clock = clock
        self._seq = 0
        self._mirror = bool(cfg.mirror)
        try:
            self._cap = cv2.VideoCapture(target, _api_preference(cfg.api))
        except cv2.error as exc:
            raise CaptureError(f"cannot open frame source {target!r}: {exc}") from exc
        if not self._cap.isOpened():
            self._cap.release()
            raise CaptureError(
                f"cannot open frame source {target!r} (source={cfg.source}, api={cfg.api})"
            )

        if cfg.width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(cfg.width))
        if cfg.height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(cfg.height))
        if cfg.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, float(cfg.fps))
        if cfg.buffer_size > 0:
            # Not every backend honors this; failure is harmless.
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, float(cfg.buffer_size))
        actual = self.size
        log.info(
            "opened %s (%sx%s, api=%s, mirror=%s)",
            target,
            actual[0] if actual else "?",
            actual[1] if actual else "?",
            cfg.api,
            self._mirror,
        )

    def read(self) -> FramePacket | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if self._mirror:
            frame = cv2.flip(frame, 1)
        frame = np.ascontiguousarray(frame)
        packet = FramePacket(
            seq=self._seq,
            capture_ts=self._clock(),
            image_bgr=frame,
            mirrored=self._mirror,
            source_fps=self.fps,
        )
        self._seq += 1
        return packet

    @property
    def fps(self) -> float | None:
        if self._cap is None:
            return None
        value = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return value if value > 0 else None

    @property
    def size(self) -> tuple[int, int] | None:
        if self._cap is None:
            return None
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return (width, height) if width > 0 and height > 0 else None

    def close(self) -> None:
        cap = getattr(self, "_cap", None)
        if cap is not None:
            cap.release()
            self._cap = None

    def __enter__(self) -> OpenCVFrameSource:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class SyntheticFrameSource:
    """Deterministic frames for tests and headless smoke runs.

    ``frames`` optionally supplies a fixed sequence of BGR images; otherwise a
    moving gradient is generated whenever ``read`` is called.
    """

    def __init__(
        self,
        frames: Sequence[np.ndarray] | None = None,
        *,
        width: int = 64,
        height: int = 48,
        fps: float = 30.0,
        max_frames: int | None = None,
        mirror: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._frames = [np.asarray(frame, dtype=np.uint8).copy() for frame in frames or ()]
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._max_frames = max_frames
        self._mirror = mirror
        self._clock = clock
        self._seq = 0
        self._closed = False

    def read(self) -> FramePacket | None:
        if self._closed:
            return None
        if self._max_frames is not None and self._seq >= self._max_frames:
            return None
        if self._frames:
            frame = self._frames[self._seq % len(self._frames)].copy()
        else:
            frame = self._gradient(self._seq)
        if self._mirror:
            frame = cv2.flip(frame, 1)
        packet = FramePacket(
            seq=self._seq,
            capture_ts=self._clock(),
            image_bgr=frame,
            mirrored=self._mirror,
            source_fps=self._fps,
        )
        self._seq += 1
        return packet

    def _gradient(self, seq: int) -> np.ndarray:
        x = np.linspace(0, 255, self._width, dtype=np.uint8)
        y = np.linspace(0, 255, self._height, dtype=np.uint8)
        frame = np.empty((self._height, self._width, 3), dtype=np.uint8)
        frame[:, :, 0] = x[None, :]
        frame[:, :, 1] = y[:, None]
        frame[:, :, 2] = (seq * 7) % 256
        return frame

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def size(self) -> tuple[int, int] | None:
        if self._frames:
            height, width = self._frames[0].shape[:2]
            return (int(width), int(height))
        return (self._width, self._height)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> SyntheticFrameSource:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def open_source(
    cfg: CaptureConfig, clock: Callable[[], float] = time.monotonic
) -> OpenCVFrameSource:
    return OpenCVFrameSource(cfg, clock=clock)
