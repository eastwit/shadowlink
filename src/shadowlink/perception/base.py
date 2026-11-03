"""Perception interfaces, errors, health reporting, and backend factories."""

from __future__ import annotations

from dataclasses import dataclass

from shadowlink.config import PerceptionConfig
from shadowlink.paths import ModelPaths


class PerceptionError(RuntimeError):
    """Raised when perception cannot run."""


class ModelUnavailableError(PerceptionError):
    """Raised when a required model file or optional dependency is missing."""


@dataclass(frozen=True, slots=True)
class WorkerHealth:
    """A snapshot of the perception worker's state."""

    state: str = "idle"  # idle | running | stopped | error
    message: str = ""
    frames: int = 0
    errors: int = 0
    dropped: int = 0

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def failed(self) -> bool:
        return self.state == "error"


def create_detector(cfg: PerceptionConfig, models: ModelPaths):
    from shadowlink.perception.mediapipe_detector import MediaPipeDetector

    if not models.hand_landmarker.exists():
        raise ModelUnavailableError(
            f"MediaPipe hand model not found: {models.hand_landmarker}\n"
            "Run `shadowlink setup-models` or pass --hand-model."
        )
    return MediaPipeDetector(
        models.hand_landmarker,
        min_detection_confidence=cfg.min_detection_confidence,
    )


def create_backend(cfg: PerceptionConfig, models: ModelPaths):
    if cfg.backend == "mediapipe":
        from shadowlink.perception.mediapipe_backend import MediaPipeKeypointBackend

        return MediaPipeKeypointBackend()

    if cfg.backend == "hamer":
        from shadowlink.perception.hamer_backend import HaMeRBackend

        missing = [path for path in (models.checkpoint, models.mano) if not path.exists()]
        if missing:
            joined = "\n  ".join(str(path) for path in missing)
            raise ModelUnavailableError(
                "HaMeR model files are missing:\n  " + joined + "\n"
                "Run `shadowlink setup-models`, then download MANO_RIGHT.pkl from "
                "https://mano.is.tue.mpg.de and place it at the path above."
            )
        return HaMeRBackend(
            checkpoint=models.checkpoint,
            model_config=models.model_config,
            mano_path=models.mano,
            data_dir=models.data_dir,
            fp16=cfg.fp16,
            rescale_factor=cfg.rescale_factor,
        )

    raise PerceptionError(f"unknown perception backend: {cfg.backend!r}")
