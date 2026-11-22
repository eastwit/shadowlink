"""Background AprilTag camera service.

The tag camera is deliberately independent from ShadowLink's hand camera.
A daemon thread opens the configured UVC device, detects one AprilTag 3 pose
per frame (official AprilTag C library through ``pupil-apriltags``), and
publishes an immutable latest-value sample.  The main teleoperation loop never
blocks on camera I/O or detection.

Astra's RGB stream is horizontally mirrored on this machine.  ``mirror=auto``
tries the normal image first and flips it only when that is required for a
valid tag decode, so the published pose is in the corrected camera frame.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from shadowlink.config import TagConfig
from shadowlink.math3d import q_from_matrix
from shadowlink.tag.types import TagHealth, TagSample

log = logging.getLogger(__name__)

_BACKENDS = {
    "any": cv2.CAP_ANY,
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
}
_MAX_READ_FAILURES = 20

TagPoseTuple = tuple[np.ndarray, np.ndarray, int, int, float, np.ndarray, float]


class TagError(RuntimeError):
    """Raised when the tag camera or AprilTag dependency cannot be used."""


def _camera_target(value: str) -> int | str:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _camera_intrinsics(
    width: int,
    height: int,
    cfg: TagConfig,
) -> np.ndarray:
    if cfg.fx is not None:
        sx = width / float(cfg.width)
        sy = height / float(cfg.height)
        fx = float(cfg.fx) * sx
        fy = float(cfg.fy) * sy
        cx = float(cfg.cx) * sx
        cy = float(cfg.cy) * sy
    else:
        hfov = math.radians(float(cfg.hfov_deg))
        fx = (width / 2.0) / math.tan(hfov / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _mirror_intrinsics(K: np.ndarray, width: int) -> np.ndarray:
    mirrored = K.copy()
    mirrored[0, 2] = (width - 1) - K[0, 2]
    return mirrored


def _params(K: np.ndarray) -> tuple[float, float, float, float]:
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _preview_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    colour: tuple[int, int, int],
) -> None:
    (width, height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        1,
    )
    x, y = org
    cv2.rectangle(
        image,
        (max(0, x - 4), max(0, y - height - 4)),
        (min(image.shape[1] - 1, x + width + 4), y + 4),
        (24, 24, 24),
        -1,
    )
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        1,
        cv2.LINE_AA,
    )


def _open_camera(cfg: TagConfig) -> tuple[cv2.VideoCapture, int, int]:
    api = _BACKENDS.get(str(cfg.api), cv2.CAP_ANY)
    target = _camera_target(cfg.camera)
    cap = cv2.VideoCapture(target, api)
    if not cap.isOpened():
        cap.release()
        raise TagError(
            f"cannot open tag camera {cfg.camera!r} with api={cfg.api}; "
            "close other camera apps or change tag.camera/tag.api"
        )
    try:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except cv2.error:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass

        frame = None
        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            time.sleep(0.02)
        if frame is None:
            raise TagError(f"tag camera {cfg.camera!r} opened but produced no frames")
    except Exception:
        cap.release()
        raise
    height, width = frame.shape[:2]
    return cap, int(width), int(height)


class TagService:
    """One daemon thread owning the AprilTag camera and detector."""

    def __init__(
        self,
        config: TagConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        detector_factory: Callable[[], Any] | None = None,
        camera_factory: Callable[[], tuple[cv2.VideoCapture, int, int]] | None = None,
    ) -> None:
        self.config = config
        self._clock = clock
        self._detector_factory = detector_factory
        self._camera_factory = camera_factory or (lambda: _open_camera(self.config))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample: TagSample | None = None
        self._preview: np.ndarray | None = None
        self._preview_ts = 0.0
        self._health = TagHealth(
            state="idle",
            camera=str(config.camera),
            api=str(config.api),
            mirror=str(config.mirror),
        )
        self._seq = 0
        self._rate_times: deque[float] = deque(maxlen=128)
        self._mirror_state: bool | None = (
            True if config.mirror == "on" else False if config.mirror == "off" else None
        )
        self._errors = 0

    # ------------------------------------------------------------------ API
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._sample = None
            self._preview = None
            self._preview_ts = 0.0
        self._seq = 0
        self._rate_times.clear()
        self._mirror_state = (
            True
            if self.config.mirror == "on"
            else False
            if self.config.mirror == "off"
            else None
        )
        self._set_health(
            TagHealth(
                state="starting",
                message="starting tag camera",
                camera=str(self.config.camera),
                api=str(self.config.api),
                mirror=str(self.config.mirror),
            )
        )
        self._thread = threading.Thread(
            target=self._run,
            name="shadowlink-tag",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.5)
        self._thread = None
        with self._lock:
            self._preview = None
        self._set_health(
            TagHealth(
                state="stopped",
                camera=str(self.config.camera),
                api=str(self.config.api),
                mirror=str(self.config.mirror),
                errors=self._errors,
            )
        )

    def latest(self) -> TagSample | None:
        with self._lock:
            sample = self._sample
        if sample is None or not sample.ready:
            return None
        if self._clock() - sample.timestamp > float(self.config.stale_timeout_s):
            return None
        return sample

    def latest_preview(self) -> np.ndarray | None:
        """Return the newest annotated tag-camera frame (or ``None``)."""
        with self._lock:
            image = self._preview
            timestamp = self._preview_ts
        if image is None or self._clock() - timestamp > 2.0:
            return None
        return image

    @property
    def health(self) -> TagHealth:
        with self._lock:
            return self._health

    # -------------------------------------------------------------- worker
    def _run(self) -> None:
        try:
            if self._detector_factory is None:
                try:
                    from pupil_apriltags import Detector  # noqa: PLC0415 - optional dependency
                except ImportError as exc:
                    raise TagError(
                        "pupil-apriltags is missing; reinstall ShadowLink with "
                        "`pip install -e \".[dev]\"`"
                    ) from exc
                try:
                    detector = Detector(
                        families=self.config.family,
                        nthreads=self.config.nthreads,
                        quad_decimate=self.config.quad_decimate,
                        quad_sigma=0.0,
                        refine_edges=1 if self.config.refine_edges else 0,
                        decode_sharpening=0.25,
                        debug=0,
                    )
                except Exception as exc:  # noqa: BLE001 - invalid family/settings
                    raise TagError(f"cannot create AprilTag detector: {exc}") from exc
            else:
                try:
                    detector = self._detector_factory()
                except Exception as exc:  # noqa: BLE001 - injected factory
                    raise TagError(f"cannot create AprilTag detector: {exc}") from exc

            while not self._stop.is_set():
                cap = None
                try:
                    cap, width, height = self._camera_factory()
                    self._stream(detector, cap, width, height)
                except TagError as exc:
                    self._errors += 1
                    self._fail(str(exc))
                except Exception as exc:  # noqa: BLE001 - report through health
                    self._errors += 1
                    log.exception("tag service failed")
                    self._fail(str(exc))
                finally:
                    if cap is not None:
                        cap.release()
                if not self._stop.is_set():
                    self._stop.wait(max(0.05, float(self.config.retry_s)))
        except TagError as exc:
            self._errors += 1
            self._fail(str(exc))
        finally:
            with self._lock:
                self._sample = None

    def _stream(
        self,
        detector: Any,
        cap: cv2.VideoCapture,
        width: int,
        height: int,
    ) -> None:
        K = _camera_intrinsics(width, height, self.config)
        K_mirror = _mirror_intrinsics(K, width)
        frames = 0
        failed_reads = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads > _MAX_READ_FAILURES:
                    raise TagError("tag camera stopped delivering frames")
                time.sleep(0.01)
                continue
            failed_reads = 0
            frames += 1

            if self._mirror_state is True:
                frame = cv2.flip(frame, 1)
                K_frame = K_mirror
            else:
                K_frame = K
            gray = (
                frame
                if frame.ndim == 2
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            )

            poses = self._detect(detector, gray, K_frame)
            if self._mirror_state is None:
                good = [pose for pose in poses if pose[3] == 0]
                if good:
                    self._mirror_state = False
                    log.info("tag camera orientation: normal")
                elif frames % 2 == 0:
                    mirrored = cv2.flip(gray, 1)
                    mirror_poses = self._detect(detector, mirrored, K_mirror)
                    if any(pose[3] == 0 for pose in mirror_poses):
                        self._mirror_state = True
                        K_frame = K_mirror
                        frame = cv2.flip(frame, 1)
                        gray = mirrored
                        poses = mirror_poses
                        log.info("tag camera image is mirrored; horizontal flip enabled")
            if self._mirror_state is None:
                # Do not drive the hand from hamming>0 guesses before the
                # camera orientation is known.
                self._publish_health(searching=True)
                self._store_preview(frame, K_frame, poses, None)
                continue
            if not poses:
                self._publish_health(searching=True)
                self._store_preview(frame, K_frame, poses, None)
                continue

            # Prefer a clean decode, then the strongest decision margin. This
            # avoids alternating between a real tag and a dimmer reflection.
            best = max(
                poses,
                key=lambda pose: (pose[3] == 0, pose[6], -float(np.linalg.norm(pose[0]))),
            )
            translation, rotation, tag_id, hamming, pose_err, _corners, margin = best
            now = self._clock()
            self._seq += 1
            sample = TagSample(
                translation_m=translation,
                quat_wxyz=q_from_matrix(rotation),
                timestamp=now,
                seq=self._seq,
                tag_id=tag_id,
            )
            self._store_preview(frame, K_frame, poses, best)
            with self._lock:
                self._sample = sample
                self._rate_times.append(now)
                rate = self._rate_hz()
                self._health = TagHealth(
                    state="running",
                    message=(
                        f"id {tag_id} hamming {hamming} margin {margin:.0f} "
                        f"err {pose_err:.2f} mirror {self._mirror_label()}"
                    ),
                    camera=str(self.config.camera),
                    api=str(self.config.api),
                    rate_hz=rate,
                    tag_id=tag_id,
                    mirror=self._mirror_label(),
                    errors=self._errors,
                )

    def _detect(
        self,
        detector: Any,
        gray: np.ndarray,
        K: np.ndarray,
    ) -> list[TagPoseTuple]:
        try:
            results = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=_params(K),
                tag_size=float(self.config.size_m),
            )
        except Exception as exc:  # noqa: BLE001 - detector errors are recoverable
            log.debug("tag detection failed: %s", exc)
            return []
        poses: list[TagPoseTuple] = []
        for result in results:
            tag_id = int(result.tag_id)
            if self.config.tag_id is not None and tag_id != int(self.config.tag_id):
                continue
            hamming = int(result.hamming)
            margin = float(result.decision_margin)
            if hamming > int(self.config.max_hamming):
                continue
            if margin < float(self.config.min_decision_margin):
                continue
            translation = np.asarray(result.pose_t, dtype=np.float64).reshape(3)
            rotation = np.asarray(result.pose_R, dtype=np.float64).reshape(3, 3)
            corners = np.asarray(result.corners, dtype=np.float64).reshape(4, 2)
            if (
                not np.isfinite(translation).all()
                or not np.isfinite(rotation).all()
                or not np.isfinite(corners).all()
            ):
                continue
            poses.append(
                (
                    translation,
                    rotation,
                    tag_id,
                    hamming,
                    float(result.pose_err),
                    corners,
                    margin,
                )
            )
        return poses

    def _store_preview(
        self,
        frame: np.ndarray,
        K: np.ndarray,
        poses: list[TagPoseTuple],
        best: TagPoseTuple | None,
    ) -> None:
        image = (
            frame
            if frame.ndim == 3
            else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        )
        preview = image.copy()
        visual_scale = max(1.0, preview.shape[1] / 960.0)
        outline = max(2, int(round(2 * visual_scale)))
        for pose in poses:
            corners = np.round(pose[5]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(preview, [corners], True, (80, 220, 80), outline, cv2.LINE_AA)
        if best is not None:
            self._draw_pose_axes(preview, best, K, visual_scale)
            center = np.round(best[5].mean(axis=0)).astype(int)
            cv2.circle(
                preview,
                tuple(center),
                max(4, int(round(4 * visual_scale))),
                (0, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            x = max(4, int(best[5][:, 0].min()))
            y = max(16, int(best[5][:, 1].min()) - 8)
            label = f"ID {best[2]}  z {best[0][2]:.2f} m  h {best[3]}"
            _preview_text(preview, label, (x, y), 0.6 * visual_scale, (80, 220, 80))
        if self.config.zero_mode == "fixed":
            self._draw_zero_guide(preview, K, visual_scale)
        status = "tag searching" if best is None else f"tag detected ({len(poses)})"
        _preview_text(
            preview,
            f"{status} | mirror {self._mirror_label()} | {self._rate_hz():.1f} fps",
            (8, max(24, int(round(24 * visual_scale)))),
            0.55 * visual_scale,
            (240, 240, 240),
        )
        max_width = 960
        if preview.shape[1] > max_width:
            scale = max_width / float(preview.shape[1])
            preview = cv2.resize(
                preview,
                (max_width, int(round(preview.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        preview.setflags(write=False)
        with self._lock:
            self._preview = preview
            self._preview_ts = self._clock()

    @staticmethod
    def _draw_pose_axes(
        image: np.ndarray,
        pose: TagPoseTuple,
        K: np.ndarray,
        visual_scale: float = 1.0,
    ) -> None:
        translation, rotation = pose[0], pose[1]
        thickness = max(2, int(round(2 * visual_scale)))
        rvec, _ = cv2.Rodrigues(rotation)
        length = 0.045
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [length, 0.0, 0.0],
                [0.0, length, 0.0],
                [0.0, 0.0, length],
            ],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(
            points,
            rvec,
            translation.reshape(3, 1),
            K,
            np.zeros((5, 1), dtype=np.float64),
        )
        pixels = projected.reshape(-1, 2)
        if not np.isfinite(pixels).all():
            return
        origin = tuple(np.round(pixels[0]).astype(int))
        for end, colour in (
            (pixels[1], (0, 0, 255)),
            (pixels[2], (0, 255, 0)),
            (pixels[3], (255, 0, 0)),
        ):
            cv2.line(
                image,
                origin,
                tuple(np.round(end).astype(int)),
                colour,
                thickness,
                cv2.LINE_AA,
            )

    def _draw_zero_guide(
        self,
        image: np.ndarray,
        K: np.ndarray,
        visual_scale: float = 1.0,
    ) -> None:
        distance = float(self.config.zero_distance_m)
        half = float(self.config.size_m) * 0.5
        square = np.array(
            [
                [-half, -half, distance],
                [half, -half, distance],
                [half, half, distance],
                [-half, half, distance],
            ],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(
            square,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            K,
            np.zeros((5, 1), dtype=np.float64),
        )
        points = np.round(projected.reshape(-1, 2)).astype(int)
        thickness = max(1, int(round(visual_scale)))
        cv2.polylines(
            image,
            [points.reshape(-1, 1, 2)],
            True,
            (255, 255, 0),
            thickness,
            cv2.LINE_AA,
        )
        centre = tuple(np.round(points.mean(axis=0)).astype(int))
        arm = max(12, int(round(12 * visual_scale)))
        cv2.line(
            image,
            (centre[0] - arm, centre[1]),
            (centre[0] + arm, centre[1]),
            (255, 255, 0),
            thickness,
        )
        cv2.line(
            image,
            (centre[0], centre[1] - arm),
            (centre[0], centre[1] + arm),
            (255, 255, 0),
            thickness,
        )
        label_x = max(
            0,
            min(
                image.shape[1] - int(170 * visual_scale),
                centre[0] + int(18 * visual_scale),
            ),
        )
        _preview_text(
            image,
            f"ZERO center {distance:.2f} m",
            (label_x, max(18, centre[1] - int(16 * visual_scale))),
            0.5 * visual_scale,
            (255, 255, 0),
        )

    def _publish_health(self, *, searching: bool) -> None:
        with self._lock:
            self._health = TagHealth(
                state="searching" if searching else self._health.state,
                message="no tag detected" if searching else self._health.message,
                camera=str(self.config.camera),
                api=str(self.config.api),
                rate_hz=0.0 if searching else self._rate_hz(),
                tag_id=self._health.tag_id,
                mirror=self._mirror_label(),
                errors=self._errors,
            )

    def _rate_hz(self) -> float:
        if len(self._rate_times) < 2:
            return 0.0
        span = self._rate_times[-1] - self._rate_times[0]
        return (len(self._rate_times) - 1) / span if span > 1e-9 else 0.0

    def _mirror_label(self) -> str:
        if self._mirror_state is None:
            return "auto"
        return "flipped" if self._mirror_state else "normal"

    def _fail(self, message: str) -> None:
        log.error("tag camera: %s", message)
        with self._lock:
            self._health = TagHealth(
                state="error",
                message=message,
                camera=str(self.config.camera),
                api=str(self.config.api),
                mirror=self._mirror_label(),
                errors=self._errors,
            )

    def _set_health(self, health: TagHealth) -> None:
        with self._lock:
            self._health = health
