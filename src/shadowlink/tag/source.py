"""Frame source backed by the AprilTag camera service.

When one camera must serve both the hand/finger perception and the tag, this
adapter exposes the tag service's raw (mirror-corrected) frames as ordinary
``FramePacket`` objects. It never owns or closes the camera itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from shadowlink.tag.service import TagService
from shadowlink.types import FramePacket


class TagFrameSource:
    """Read the latest raw frame published by a :class:`TagService`."""

    def __init__(
        self,
        service: TagService,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.service = service
        self._clock = clock
        self._last_seq = -1

    def read(self) -> FramePacket | None:
        result = self.service.wait_frame(self._last_seq, timeout=0.75)
        if result is None:
            return None
        seq, capture_ts, image = result
        self._last_seq = seq
        return FramePacket(
            seq=seq,
            capture_ts=capture_ts,
            image_bgr=image,
            mirrored=False,
            source_fps=self.fps,
        )

    @property
    def fps(self) -> float | None:
        rate = self.service.health.rate_hz
        return rate if rate > 0 else None

    @property
    def size(self) -> tuple[int, int] | None:
        image = self.service.latest_preview()
        if image is None:
            return None
        return (int(image.shape[1]), int(image.shape[0]))

    def close(self) -> None:
        # The TagService owns the camera and is closed by the app.
        pass
