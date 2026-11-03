"""OpenCV preview window and video recording."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class UIError(RuntimeError):
    """Raised when a GUI window cannot be created."""


class PreviewWindow:
    """A single OpenCV HighGUI window.

    All calls must happen on the main thread; the perception workers never
    touch OpenCV UI.
    """

    def __init__(self, name: str = "ShadowLink", *, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled
        self._open = False

    def show(self, image: np.ndarray) -> int | None:
        if not self.enabled:
            return None
        if not self._open:
            try:
                cv2.namedWindow(self.name, cv2.WINDOW_AUTOSIZE)
            except cv2.error as exc:
                raise UIError(f"cannot create preview window: {exc}") from exc
            self._open = True
        try:
            cv2.imshow(self.name, image)
        except cv2.error as exc:
            self.enabled = False
            raise UIError(f"cannot show preview: {exc}") from exc
        key = cv2.waitKey(1) & 0xFF
        return None if key == 255 else key

    def close(self) -> None:
        if self._open:
            with contextlib.suppress(cv2.error):
                cv2.destroyWindow(self.name)
            self._open = False


class RecordingError(RuntimeError):
    """Raised when a video writer cannot be opened."""


class VideoRecorder:
    """Lazily opened ``cv2.VideoWriter`` with a codec fallback.

    The writer opens on the first frame because only then is the frame size
    known.  If the requested container/codec combination fails, MJPG in an AVI
    container is tried instead; ``path`` reflects the file actually written.
    """

    def __init__(self, path: str | Path, fps: float) -> None:
        requested = Path(path).expanduser().resolve()
        requested.parent.mkdir(parents=True, exist_ok=True)
        self.path = requested
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._writer: cv2.VideoWriter | None = None
        self.frames = 0

    @property
    def is_open(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        if frame is None or frame.ndim != 3:
            return
        if self._writer is None:
            self._open_writer(frame.shape[1], frame.shape[0])
        self._writer.write(frame)
        self.frames += 1

    def _open_writer(self, width: int, height: int) -> None:
        size = (int(width), int(height))
        attempts: list[tuple[Path, str]] = []
        if self.path.suffix.lower() == ".avi":
            attempts.append((self.path, "MJPG"))
            attempts.append((self.path.with_suffix(".mp4"), "mp4v"))
        else:
            attempts.append((self.path, "mp4v"))
            attempts.append((self.path.with_suffix(".avi"), "MJPG"))
        for candidate, codec in attempts:
            writer = cv2.VideoWriter(
                str(candidate),
                cv2.VideoWriter_fourcc(*codec),
                self.fps,
                size,
            )
            if writer.isOpened():
                self._writer = writer
                self.path = candidate
                log.info("recording to %s (%s, %.1f fps)", candidate, codec, self.fps)
                return
            writer.release()
        raise RecordingError(
            f"no video writer could be opened for {self.path} "
            f"(tried mp4v/MJPG at {size[0]}x{size[1]})"
        )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            log.info("recording finished: %s (%d frames)", self.path, self.frames)

    def __enter__(self) -> VideoRecorder:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
