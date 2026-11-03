"""Download and verify the external model files.

The repository never ships weights.  This command fetches the public HaMeR
demo archive and the MediaPipe hand landmarker, and reports how to obtain MANO,
which requires registration and must not be redistributed.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from shadowlink.paths import ModelPaths, resolve_data_dir

HAMER_ARCHIVE_URL = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MANO_URL = "https://mano.is.tue.mpg.de"

_COPY_FILES = (
    "hamer_ckpts/checkpoints/hamer.ckpt",
    "hamer_ckpts/model_config.yaml",
    "data/mano/MANO_RIGHT.pkl",
    "mediapipe/hand_landmarker.task",
)


def setup_models(args: argparse.Namespace) -> int:
    data_dir = resolve_data_dir(args.data_dir)
    paths = ModelPaths.resolve(
        data_dir,
        checkpoint=getattr(args, "checkpoint", None),
        hand_model=getattr(args, "hand_model", None),
    )
    print(f"data directory: {data_dir}")

    if args.check_only:
        return _report(paths, strict=True)

    if args.copy_from:
        _copy_existing(Path(args.copy_from).expanduser().resolve(), paths, force=args.force)

    need_hamer = not args.mediapipe_only
    need_mediapipe = not args.hamer_only

    if need_hamer and (args.force or not _hamer_present(paths)):
        _download_hamer(paths.data_dir, force=args.force)
    if need_mediapipe and (args.force or not paths.hand_landmarker.exists()):
        _download(MEDIAPIPE_URL, paths.hand_landmarker, force=args.force)

    return _report(paths, strict=True)


def _hamer_present(paths: ModelPaths) -> bool:
    return paths.checkpoint.exists() and paths.model_config.exists()


def _report(paths: ModelPaths, *, strict: bool) -> int:
    missing = []
    for path in (paths.checkpoint, paths.model_config, paths.mano, paths.hand_landmarker):
        if path.exists():
            print(f"  ok      {path}")
        else:
            print(f"  missing {path}")
            missing.append(path)
    if paths.mano in missing:
        print(
            "MANO is not downloadable automatically. Register at "
            f"{MANO_URL} and place MANO_RIGHT.pkl at:\n  {paths.mano}"
        )
    if not missing:
        print("all model files are present")
        return 0
    return 1 if strict else 0


def _copy_existing(source: Path, paths: ModelPaths, *, force: bool) -> None:
    if not source.is_dir():
        raise SystemExit(f"--copy-from is not a directory: {source}")
    print(f"copying existing data from {source}")
    for relative in _COPY_FILES:
        src = source / relative
        if not src.exists():
            continue
        dst = paths.data_dir / relative
        if dst.exists() and not force:
            print(f"  keep    {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  copied  {src} -> {dst}")


def _download_hamer(data_dir: Path, *, force: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shadowlink-hamer-") as temp_name:
        temp = Path(temp_name)
        archive = temp / "hamer_demo_data.tar.gz"
        _download(HAMER_ARCHIVE_URL, archive, force=True)
        extracted = temp / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        root = _find_data_root(extracted)
        if root is None:
            raise SystemExit("downloaded HaMeR archive does not contain hamer_ckpts/ or data/")
        shutil.copytree(root, data_dir, dirs_exist_ok=True)
    print(f"installed HaMeR demo data into {data_dir}")


def _find_data_root(root: Path) -> Path | None:
    candidates = [root, root / "_DATA", *[p for p in root.iterdir() if p.is_dir()]]
    for candidate in candidates:
        if (candidate / "hamer_ckpts").is_dir() or (candidate / "data").is_dir():
            return candidate
    return None


def _download(url: str, destination: Path, *, force: bool = False) -> None:
    destination = Path(destination)
    if destination.exists() and not force:
        print(f"  keep    {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "shadowlink/0.1"})
    partial = destination.with_name(destination.name + ".part")
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        partial.replace(destination)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"download failed: {exc}\nDownload {url} manually and place it at {destination}."
        ) from exc
    print(f"  saved   {destination} ({destination.stat().st_size / 1e6:.1f} MB)")


def _safe_extract(archive: Path, target: Path) -> None:
    target = target.resolve()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            resolved = (target / member.name).resolve()
            if not resolved.is_relative_to(target):
                raise SystemExit(f"refusing unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise SystemExit(f"refusing archive link: {member.name}")
        tar.extractall(target)
