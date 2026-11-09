"""Match an observation's frame sequence to the depth frame captured with it."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from shadowlink.types import DepthIntrinsics, FramePacket


@dataclass(frozen=True, slots=True)
class DepthRecord:
    """The depth snapshot attached to one colour frame."""

    seq: int
    depth_mm: np.ndarray
    intrinsics: DepthIntrinsics
    depth_ts: float
    color_ts: float
    mirrored: bool
    color_size: tuple[int, int]


class DepthFrameLog:
    """A short, bounded history keyed by colour frame sequence.

    Depth arrays are already read-only and are stored by reference, so adding a
    frame does not copy the image.
    """

    def __init__(self, maxlen: int = 16) -> None:
        if maxlen < 1:
            raise ValueError("maxlen must be >= 1")
        self._records: deque[DepthRecord] = deque(maxlen=int(maxlen))

    def add(self, frame: FramePacket) -> None:
        if frame.depth_mm is None or frame.depth_intrinsics is None or frame.depth_ts is None:
            return
        self._records.append(
            DepthRecord(
                seq=frame.seq,
                depth_mm=frame.depth_mm,
                intrinsics=frame.depth_intrinsics,
                depth_ts=float(frame.depth_ts),
                color_ts=float(frame.capture_ts),
                mirrored=bool(frame.mirrored),
                color_size=(frame.width, frame.height),
            )
        )

    def get(self, seq: int) -> DepthRecord | None:
        for record in reversed(self._records):
            if record.seq == seq:
                return record
        return None

    def clear(self) -> None:
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)
