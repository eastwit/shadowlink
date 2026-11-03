"""Filesystem locations used by ShadowLink.

Runtime assets (MJCF, meshes, tactile sites) live inside the installed
package.  Model weights and user data live outside the repository, in a data
directory and a per-user configuration directory.  No code in this package
changes the process working directory.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
SHADOW_HAND_DIR = ASSETS_DIR / "shadow_hand"
SCENE_XML = SHADOW_HAND_DIR / "shadowlink_scene.xml"
RIGHT_HAND_XML = SHADOW_HAND_DIR / "right_hand.xml"
KEYFRAMES_XML = SHADOW_HAND_DIR / "keyframes.xml"
TACTILE_DIR = SHADOW_HAND_DIR / "tactile"

DEFAULT_DATA_ENV = "SHADOWLINK_DATA"
CALIBRATION_ENV = "SHADOWLINK_CALIBRATION"


def resolve_data_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the model/data directory.

    Precedence: explicit argument > ``SHADOWLINK_DATA`` > ``~/.cache/shadowlink``.
    The result is always absolute.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(DEFAULT_DATA_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "shadowlink").resolve()


def user_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "ShadowLink"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ShadowLink"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "shadowlink"


def resolve_calibration_path(
    explicit: str | os.PathLike[str] | None = None,
) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(CALIBRATION_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return user_config_dir() / "calibration.json"


@dataclass(frozen=True, slots=True)
class ModelPaths:
    data_dir: Path
    checkpoint: Path
    model_config: Path
    mano: Path
    hand_landmarker: Path

    @classmethod
    def resolve(
        cls,
        data_dir: str | os.PathLike[str] | None = None,
        checkpoint: str | os.PathLike[str] | None = None,
        hand_model: str | os.PathLike[str] | None = None,
    ) -> ModelPaths:
        root = resolve_data_dir(data_dir)
        ckpt = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint
            else root / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
        )
        return cls(
            data_dir=root,
            checkpoint=ckpt,
            model_config=ckpt.parent.parent / "model_config.yaml",
            mano=root / "data" / "mano" / "MANO_RIGHT.pkl",
            hand_landmarker=(
                Path(hand_model).expanduser().resolve()
                if hand_model
                else root / "mediapipe" / "hand_landmarker.task"
            ),
        )

    def missing(self, need_hamer: bool = True) -> list[Path]:
        required = [self.hand_landmarker]
        if need_hamer:
            required = [self.checkpoint, self.model_config, self.mano, *required]
        return [path for path in required if not path.exists()]


def resolve_output_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a CLI output path against the current directory and create parents."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
