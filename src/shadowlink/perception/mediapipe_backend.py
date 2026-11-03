"""CPU keypoint backend built on MediaPipe world landmarks."""

from __future__ import annotations

import time
from collections.abc import Sequence

from shadowlink.types import FramePacket, HandDetection, HandObservation


class MediaPipeKeypointBackend:
    """Use ``hand_world_landmarks`` as a metric 3D hand skeleton."""

    name = "mediapipe"
    requires_detections = True
    supports_mesh = False

    def estimate(
        self,
        frame: FramePacket,
        detections: Sequence[HandDetection],
    ) -> list[HandObservation]:
        now = time.monotonic()
        observations: list[HandObservation] = []
        for detection in detections:
            if detection.world_landmarks is None:
                continue
            observations.append(
                HandObservation(
                    keypoints_3d=detection.world_landmarks,
                    apparent_side=detection.apparent_side,
                    anatomical_side=detection.anatomical_side,
                    score=float(detection.score),
                    frame_seq=frame.seq,
                    capture_ts=frame.capture_ts,
                    inference_ts=now,
                    backend=self.name,
                    bbox_xyxy=detection.bbox_xyxy,
                    keypoints_2d=detection.landmarks_2d,
                )
            )
        return observations

    def close(self) -> None:
        return None
