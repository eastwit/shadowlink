"""HaMeR monocular 3D hand reconstruction backend.

We deliberately do not call the upstream ``load_hamer()`` helper:

* it imports ``hamer.utils`` (and therefore pyrender) before the platform can be
  configured, which fails on Windows with the default EGL backend;
* it builds a relative ``CACHE_DIR_HAMER`` and relies on the process working
  directory being the HaMeR checkout.

This loader sets ``PYOPENGL_PLATFORM`` for the current platform, resolves every
model path absolutely, removes the training-only pretrained backbone reference,
and constructs the Lightning module with ``init_renderer=False``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from shadowlink.perception.base import ModelUnavailableError, PerceptionError
from shadowlink.perception.hamer_dataset import prepare_batch
from shadowlink.perception.hamer_math import cam_crop_to_full, project_points, scaled_focal_length
from shadowlink.types import FramePacket, HandDetection, HandObservation

log = logging.getLogger(__name__)


class HaMeRBackend:
    name = "hamer"
    requires_detections = True
    supports_mesh = True

    def __init__(
        self,
        checkpoint: str | Path,
        model_config: str | Path,
        mano_path: str | Path,
        *,
        data_dir: str | Path | None = None,
        device: str | None = None,
        fp16: bool = True,
        rescale_factor: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        checkpoint = Path(checkpoint)
        model_config = Path(model_config)
        mano_path = Path(mano_path)
        for path, label in (
            (checkpoint, "checkpoint"),
            (model_config, "model config"),
            (mano_path, "MANO model"),
        ):
            if not path.exists():
                raise ModelUnavailableError(f"HaMeR {label} not found: {path}")

        root = Path(data_dir) if data_dir is not None else mano_path.resolve().parent.parent
        self._clock = clock
        self._torch, self._model, self._cfg, self.device = _load_model(
            checkpoint=checkpoint,
            model_config=model_config,
            mano_path=mano_path,
            data_dir=root,
            device=device,
        )
        self.fp16 = bool(fp16) and self.device.type == "cuda"
        self.rescale_factor = float(rescale_factor)
        if self.rescale_factor <= 0:
            raise ValueError("rescale_factor must be positive")
        self._mesh_edges = _unique_edges(self._model.mano.faces)
        self._closed = False
        log.info(
            "loaded HaMeR on %s (fp16=%s, checkpoint=%s)",
            self.device,
            self.fp16,
            checkpoint,
        )

    # --------------------------------------------------------------- inference
    def estimate(
        self,
        frame: FramePacket,
        detections: Sequence[HandDetection],
    ) -> list[HandObservation]:
        if self._closed:
            raise PerceptionError("HaMeR backend is closed")
        if not detections:
            return []

        torch = self._torch
        boxes = np.stack([detection.bbox_xyxy for detection in detections])
        rights = np.array([detection.apparent_side.is_right for detection in detections])
        batch = prepare_batch(
            self._cfg,
            frame.image_bgr,
            boxes,
            rights,
            rescale_factor=self.rescale_factor,
        )
        device = self.device
        inputs = {
            "img": torch.from_numpy(batch.images).to(device),
            "right": torch.from_numpy(batch.right).to(device),
            "box_center": torch.from_numpy(batch.box_center).to(device),
            "box_size": torch.from_numpy(batch.box_size).to(device),
            "img_size": torch.from_numpy(batch.img_size).to(device),
        }

        try:
            with torch.inference_mode():
                if self.fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = self._model(inputs)
                else:
                    output = self._model(inputs)
        except Exception as exc:  # pragma: no cover - runtime/device dependent
            raise PerceptionError(f"HaMeR inference failed: {exc}") from exc

        pred_cam = output["pred_cam"].float().cpu().numpy()
        keypoints = output["pred_keypoints_3d"].float().cpu().numpy()
        vertices = output["pred_vertices"].float().cpu().numpy()

        # HaMeR was trained on right hands; left-hand crops were mirrored.  Flip
        # the prediction back into the processed image frame so overlays line up
        # and the retargeter can canonicalise using the apparent side.
        multiplier = 2.0 * batch.right - 1.0
        pred_cam[:, 1] *= multiplier
        keypoints[:, :, 0] *= multiplier[:, None]
        vertices[:, :, 0] *= multiplier[:, None]

        img_size = batch.img_size.astype(np.float64)
        focal = scaled_focal_length(
            self._cfg.EXTRA.FOCAL_LENGTH,
            self._cfg.MODEL.IMAGE_SIZE,
            img_size[0],
        )
        cam_t = cam_crop_to_full(
            pred_cam,
            batch.box_center,
            batch.box_size,
            img_size,
            focal,
        )

        now = self._clock()
        observations: list[HandObservation] = []
        for index, detection in enumerate(detections):
            points_3d = np.ascontiguousarray(keypoints[index], dtype=np.float64)
            points_2d = project_points(
                points_3d,
                cam_t[index],
                focal,
                frame.width,
                frame.height,
            )
            observations.append(
                HandObservation(
                    keypoints_3d=points_3d,
                    apparent_side=detection.apparent_side,
                    anatomical_side=detection.anatomical_side,
                    score=float(detection.score),
                    frame_seq=frame.seq,
                    capture_ts=frame.capture_ts,
                    inference_ts=now,
                    backend=self.name,
                    bbox_xyxy=detection.bbox_xyxy,
                    keypoints_2d=points_2d,
                    vertices=np.ascontiguousarray(vertices[index], dtype=np.float32),
                    mesh_edges=self._mesh_edges,
                    camera_translation=np.ascontiguousarray(cam_t[index]),
                    focal_length=float(focal),
                )
            )
        return observations

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._model = None
        if self.device.type == "cuda":
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - driver dependent
                log.debug("CUDA cache cleanup failed", exc_info=True)


# --------------------------------------------------------------------- loader


def _unique_edges(faces) -> np.ndarray:
    """Unique undirected edges of a triangle mesh, for wireframe overlays."""
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise PerceptionError(f"expected (F, 3) mesh faces, got {triangles.shape}")
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]),
        axis=0,
    )
    return np.unique(np.sort(edges, axis=1), axis=0).astype(np.int32)


def _configure_opengl_platform() -> None:
    """Pick a pyrender/OpenGL platform before HaMeR imports it.

    The upstream project forces EGL on every platform; that fails on Windows.
    On Linux EGL remains the right choice for headless inference.
    """
    if "PYOPENGL_PLATFORM" in os.environ:
        return
    os.environ["PYOPENGL_PLATFORM"] = "gl" if os.name == "nt" else "egl"


def _load_model(
    *,
    checkpoint: Path,
    model_config: Path,
    mano_path: Path,
    data_dir: Path,
    device: str | None,
):
    _configure_opengl_platform()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModelUnavailableError(
            "torch is required for the HaMeR backend; install the `hamer` extra"
        ) from exc

    try:
        import hamer.configs as hamer_configs
        from hamer.configs import get_config
        from hamer.models.hamer import HAMER
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModelUnavailableError(
            "the hamer package is not importable; initialise third_party/hamer and "
            "run `pip install -e third_party/hamer --no-deps`"
        ) from exc

    # ``get_config(update_cachedir=True)`` expands MANO paths against this
    # module global, so it must be absolute before the call.
    hamer_configs.CACHE_DIR_HAMER = str(Path(data_dir).resolve())
    try:
        cfg = get_config(str(model_config), update_cachedir=True)
    except Exception as exc:  # pragma: no cover - config dependent
        raise PerceptionError(f"cannot read HaMeR model config {model_config}: {exc}") from exc

    cfg.defrost()
    if "BBOX_SHAPE" not in cfg.MODEL:
        if int(cfg.MODEL.IMAGE_SIZE) != 256:
            raise PerceptionError(
                f"HaMeR ViT checkpoints expect MODEL.IMAGE_SIZE=256 (got {cfg.MODEL.IMAGE_SIZE})"
            )
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    # The backbone is restored from the checkpoint; don't try to load the
    # training-time pretrained weights file.
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    # The checkpoint stores relative MANO paths from the training machine.
    cfg.MANO.MODEL_PATH = str(mano_path)
    mean_params = Path(data_dir) / "data" / "mano_mean_params.npz"
    if mean_params.exists():
        cfg.MANO.MEAN_PARAMS = str(mean_params)
    cfg.freeze()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise PerceptionError("CUDA was requested but is not available")

    try:
        model = HAMER.load_from_checkpoint(
            str(checkpoint),
            strict=False,
            cfg=cfg,
            init_renderer=False,
            map_location="cpu",
        )
    except Exception as exc:  # pragma: no cover - checkpoint dependent
        raise PerceptionError(f"cannot load HaMeR checkpoint {checkpoint}: {exc}") from exc
    model = model.to(torch_device).eval()
    return torch, model, cfg, torch_device
