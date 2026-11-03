"""MediaPipe HandLandmarker adapter.

MediaPipe's handedness label describes the hand *as it appears in the image*:
mirroring the input swaps the label.  We therefore treat the label as the
apparent side and derive the anatomical side from it and
``FramePacket.mirrored``.  This keeps ``--side left|right`` meaningful when the
camera view is mirrored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shadowlink.perception.base import ModelUnavailableError, PerceptionError
from shadowlink.types import FramePacket, HandDetection, Side


class MediaPipeDetector:
    name = "mediapipe"

    def __init__(
        self,
        model_path: str | Path,
        *,
        min_detection_confidence: float = 0.3,
        num_hands: int = 4,
        running_mode: str = "video",
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelUnavailableError(
                "mediapipe is not installed; install it with `pip install mediapipe`"
            ) from exc

        if running_mode not in {"image", "video"}:
            raise ValueError(f"unknown MediaPipe running mode: {running_mode!r}")
        self._mp = mp
        self._running_mode = running_mode
        self._last_timestamp_ms = -1
        path = Path(model_path)
        if not path.exists():
            raise ModelUnavailableError(f"MediaPipe hand model not found: {path}")
        mode = (
            mp.tasks.vision.RunningMode.VIDEO
            if running_mode == "video"
            else mp.tasks.vision.RunningMode.IMAGE
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mode,
            num_hands=int(num_hands),
            min_hand_detection_confidence=float(min_detection_confidence),
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            raise PerceptionError(f"cannot create MediaPipe HandLandmarker: {exc}") from exc

    def detect(self, frame: FramePacket) -> list[HandDetection]:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=frame.rgb_contiguous(),
        )
        try:
            if self._running_mode == "video":
                timestamp_ms = next_timestamp_ms(frame.capture_ts, self._last_timestamp_ms)
                self._last_timestamp_ms = timestamp_ms
                result = self._landmarker.detect_for_video(image, timestamp_ms)
            else:
                result = self._landmarker.detect(image)
        except Exception as exc:  # pragma: no cover - runtime dependent
            raise PerceptionError(f"MediaPipe detection failed: {exc}") from exc

        if not result.hand_landmarks:
            return []

        height, width = frame.image_bgr.shape[:2]
        detections: list[HandDetection] = []
        world_hands = getattr(result, "hand_world_landmarks", None) or []
        for index, landmarks in enumerate(result.hand_landmarks):
            handedness = result.handedness[index] if index < len(result.handedness) else []
            if not handedness:
                continue
            category = handedness[0]
            try:
                apparent = Side.from_mediapipe_label(category.category_name)
            except ValueError:
                continue
            anatomical = anatomical_side(apparent, frame.mirrored)

            points = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
            bbox = _clipped_bbox(points, width, height)
            if bbox is None:
                continue

            world = None
            if index < len(world_hands):
                world_points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in world_hands[index]], dtype=np.float64
                )
                if world_points.shape == (21, 3) and np.isfinite(world_points).all():
                    world = world_points

            detections.append(
                HandDetection(
                    bbox_xyxy=bbox,
                    apparent_side=apparent,
                    anatomical_side=anatomical,
                    score=float(category.score),
                    world_landmarks=world,
                    landmarks_2d=points,
                    frame_seq=frame.seq,
                )
            )
        return detections

    def close(self) -> None:
        landmarker = getattr(self, "_landmarker", None)
        if landmarker is not None:
            landmarker.close()
            self._landmarker = None


def next_timestamp_ms(capture_ts: float, last_ms: int) -> int:
    """Return a strictly increasing millisecond timestamp for VIDEO mode.

    Frames can be dropped by the latest-slot pipeline, and a fast camera can
    deliver two frames in the same millisecond; MediaPipe rejects both cases,
    so clamp to ``last + 1`` when needed.
    """
    timestamp = int(float(capture_ts) * 1000.0)
    return timestamp if timestamp > last_ms else int(last_ms) + 1


def anatomical_side(apparent: Side, mirrored: bool) -> Side:
    """Recover the real hand side from MediaPipe's image-relative label.

    MediaPipe reports the hand as it appears in the pixels it receives.  If we
    mirrored the frame first, the apparent side is the opposite of the
    anatomical side.
    """
    return apparent.flipped() if mirrored else apparent


def _clipped_bbox(points: np.ndarray, width: int, height: int) -> np.ndarray | None:
    xs = points[:, 0]
    ys = points[:, 1]
    x0 = float(np.clip(xs.min(), 0.0, width - 1))
    x1 = float(np.clip(xs.max(), 0.0, width - 1))
    y0 = float(np.clip(ys.min(), 0.0, height - 1))
    y1 = float(np.clip(ys.max(), 0.0, height - 1))
    if x1 - x0 < 2.0 or y1 - y0 < 2.0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float64)
