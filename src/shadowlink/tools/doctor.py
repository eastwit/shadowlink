"""Environment and model diagnostics for bug reports."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

from shadowlink import __version__
from shadowlink.paths import (
    KEYFRAMES_XML,
    SCENE_XML,
    ModelPaths,
    resolve_calibration_path,
    resolve_data_dir,
)

_PACKAGES = (
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("mujoco", "mujoco"),
    ("mediapipe", "mediapipe"),
    ("PyYAML", "yaml"),
    ("torch", "torch"),
    ("pytorch-lightning", "pytorch_lightning"),
)


def doctor(args: argparse.Namespace) -> int:
    report = _collect(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    if args.strict_models and report["models"]["missing"]:
        return 1
    return 0 if report["assets"]["ok"] else 1


def _collect(args: argparse.Namespace) -> dict:
    paths = ModelPaths.resolve(
        args.data_dir,
        checkpoint=args.checkpoint,
        hand_model=args.hand_model,
    )
    missing = [str(path) for path in paths.missing(need_hamer=True)]
    report = {
        "shadowlink": __version__,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {name: _package_version(name, module) for name, module in _PACKAGES},
        "cuda": _cuda_info(),
        "paths": {
            "data_dir": str(resolve_data_dir(args.data_dir)),
            "checkpoint": str(paths.checkpoint),
            "model_config": str(paths.model_config),
            "mano": str(paths.mano),
            "hand_landmarker": str(paths.hand_landmarker),
            "calibration": str(resolve_calibration_path(None)),
        },
        "models": {
            "missing": missing,
            "checkpoint_size_mb": _size_mb(paths.checkpoint),
            "mano_size_mb": _size_mb(paths.mano),
        },
        "assets": _asset_info(),
        "calibration": _calibration_info(),
    }
    return report


def _package_version(name: str, module: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        try:
            imported = importlib.import_module(module)
        except Exception:
            return None
        return str(getattr(imported, "__version__", "unknown"))


def _cuda_info() -> dict:
    try:
        import torch
    except Exception:
        return {"available": False, "reason": "torch not installed"}
    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device": torch.cuda.get_device_name(0) if available else None,
        "torch_cuda": torch.version.cuda,
    }


def _asset_info() -> dict:
    info: dict = {"ok": False, "scene": str(SCENE_XML), "keyframes": str(KEYFRAMES_XML)}
    try:
        from shadowlink.sim.model import load_keyframes, load_model

        model, layout = load_model(SCENE_XML)
        poses = load_keyframes(model, KEYFRAMES_XML)
        info.update(
            {
                "ok": True,
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "nsensor": int(model.nsensor),
                "timestep": float(model.opt.timestep),
                "keyframes": len(poses),
                "actuators": list(layout.names),
            }
        )
    except Exception as exc:  # pragma: no cover - corrupted install
        info["error"] = str(exc)
    return info


def _calibration_info() -> dict:
    path = resolve_calibration_path(None)
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    try:
        from shadowlink.retarget.calibration import CalibrationStore

        calibration = CalibrationStore(path).load()
        if calibration is not None:
            result.update(
                {
                    "backend": calibration.backend,
                    "side": calibration.side,
                    "created_at": calibration.created_at,
                }
            )
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    return round(path.stat().st_size / 1e6, 2)


def _print_report(report: dict) -> None:
    print(f"shadowlink {report['shadowlink']}")
    python = report["python"]
    print(f"python     {python['version']} ({python['executable']})")
    print(f"platform   {python['platform']}")
    print("packages")
    for name, version in report["packages"].items():
        print(f"  {name:20s} {version or 'not installed'}")
    cuda = report["cuda"]
    if cuda.get("available"):
        print(f"cuda       {cuda.get('device')} (torch {cuda.get('torch_cuda')})")
    else:
        print(f"cuda       unavailable ({cuda.get('reason', 'no device')})")
    print("paths")
    for name, value in report["paths"].items():
        print(f"  {name:16s} {value}")
    print("models")
    missing = report["models"]["missing"]
    if missing:
        for path in missing:
            print(f"  missing        {path}")
    else:
        print("  all required model files present")
    print("assets")
    assets = report["assets"]
    if assets.get("ok"):
        print(
            f"  ok             nq={assets['nq']} nu={assets['nu']} "
            f"nsensors={assets['nsensor']} keyframes={assets['keyframes']}"
        )
    else:
        print(f"  error          {assets.get('error', 'unknown')}")
    calibration = report["calibration"]
    if calibration.get("exists"):
        print(
            f"calibration      {calibration.get('backend')} "
            f"{calibration.get('side')} {calibration.get('created_at')}"
        )
    else:
        print("calibration      not saved")
