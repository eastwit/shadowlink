"""Command-line entry point for ShadowLink."""

from __future__ import annotations

import argparse
import logging
import sys

from shadowlink import __version__
from shadowlink.config import AppConfig, ConfigError
from shadowlink.paths import KEYFRAMES_XML, SCENE_XML, ModelPaths, resolve_data_dir


def _common(parser: argparse.ArgumentParser, *, config: bool = True) -> None:
    if config:
        parser.add_argument("--config", type=str, default=None, help="YAML config file")
        parser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="SECTION.FIELD=VALUE",
            help="override one config value (repeatable)",
        )


def _load_config(args: argparse.Namespace) -> AppConfig:
    config_path = getattr(args, "config", None)
    cfg = AppConfig.from_yaml(config_path) if config_path else AppConfig()
    overrides = list(getattr(args, "overrides", None) or [])
    if overrides:
        cfg = cfg.apply_overrides(overrides)
    return cfg


def cmd_demo(args: argparse.Namespace) -> int:
    from shadowlink.sim.model import load_keyframes, load_model
    from shadowlink.sim.pose_demo import run_pose_demo

    cfg = _load_config(args)
    cfg = cfg.apply_overrides([f"sim.viewer={str(args.viewer).lower()}"])
    model, layout = load_model(SCENE_XML)
    poses = load_keyframes(model, KEYFRAMES_XML)
    if args.pose and args.pose != "all":
        wanted = [name.strip() for name in args.pose.split(",") if name.strip()]
        missing = [name for name in wanted if name not in poses]
        if missing:
            raise SystemExit(f"unknown pose(s): {missing}; available: {sorted(poses)}")
        poses = {name: poses[name] for name in wanted}
    import mujoco

    data = mujoco.MjData(model)
    shown = run_pose_demo(
        model,
        data,
        layout,
        poses,
        viewer=cfg.sim.viewer,
        seconds_per_pose=args.seconds_per_pose,
        max_seconds=args.max_seconds,
        touch_force_scale=cfg.sim.touch_force_scale,
    )
    print(f"posed {shown} keyframes")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from shadowlink.sim.model import load_keyframes, load_model

    model, layout = load_model(SCENE_XML)
    poses = load_keyframes(model, KEYFRAMES_XML)
    print(f"model: {SCENE_XML}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nsensor={model.nsensor}")
    print(f"  timestep={model.opt.timestep:g}s")
    print("actuators:")
    for spec in layout.specs:
        print(
            f"  {spec.index:2d} {spec.name:12s} {spec.semantic:5s} "
            f"{spec.kind:6s} [{spec.low:+.4f}, {spec.high:+.4f}] -> {spec.target_name}"
        )
    print(f"keyframes: {len(poses)} ({', '.join(sorted(poses))})")
    data_dir = resolve_data_dir(args.data_dir)
    paths = ModelPaths.resolve(
        data_dir,
        checkpoint=args.checkpoint,
        hand_model=args.hand_model,
    )
    print(f"data dir: {data_dir}")
    missing = paths.missing(need_hamer=True)
    if missing:
        print("missing model files:")
        for path in missing:
            print(f"  - {path}")
        print("run `shadowlink setup-models` for the downloadable files,")
        print("and register at https://mano.is.tue.mpg.de for MANO_RIGHT.pkl.")
        return 1 if args.strict_models else 0
    print("all model files present")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from shadowlink.app import run_from_args

    return run_from_args(args, _load_config(args))


def cmd_calibrate(args: argparse.Namespace) -> int:
    from shadowlink.app import calibrate_from_args

    return calibrate_from_args(args, _load_config(args))


def cmd_phone_test(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from shadowlink.phone.client import parse_host_port
    from shadowlink.tools.phone_test import run_phone_test

    cfg = _load_config(args)
    host = args.host
    port = args.port
    if host is not None:
        try:
            parsed_host, parsed_port = parse_host_port(host, default_port=port or cfg.phone.port)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        host = parsed_host
        if port is None:
            port = parsed_port
    phone = replace(
        cfg.phone,
        source="phyphox",
        host=host if host is not None else cfg.phone.host,
        port=port if port is not None else cfg.phone.port,
        auto_discover=bool(args.scan) or cfg.phone.auto_discover,
    )
    return run_phone_test(
        phone,
        seconds=args.seconds,
        list_buffers=args.list,
        as_json=args.json,
    )


def cmd_setup_models(args: argparse.Namespace) -> int:
    from shadowlink.tools.setup_models import setup_models

    return setup_models(args)


def cmd_generate_tactile(args: argparse.Namespace) -> int:
    from shadowlink.tools.generate_tactile import main as generate_main

    return generate_main(out=args.out, check=args.check)


def cmd_render_pose(args: argparse.Namespace) -> int:
    from shadowlink.tools.render_pose import render_pose_cli

    return render_pose_cli(args)


def cmd_doctor(args: argparse.Namespace) -> int:
    from shadowlink.tools.doctor import doctor

    return doctor(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shadowlink",
        description="Monocular hand teleoperation of a Shadow Hand in MuJoCo.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="cycle through keyframe poses (no camera)")
    _common(demo)
    demo.add_argument("--pose", default="all", help="pose name, comma-separated list, or all")
    demo.add_argument("--seconds-per-pose", type=float, default=1.5)
    demo.add_argument("--max-seconds", type=float, default=0.0, help="0 = until the window closes")
    demo.add_argument("--no-viewer", dest="viewer", action="store_false", default=True)
    demo.set_defaults(func=cmd_demo)

    check = sub.add_parser("check", help="validate MJCF, keyframes, and model files")
    check.add_argument("--data-dir", default=None)
    check.add_argument("--checkpoint", default=None)
    check.add_argument("--hand-model", default=None)
    check.add_argument("--strict-models", action="store_true")
    check.set_defaults(func=cmd_check)

    run = sub.add_parser("run", help="camera/video hand teleoperation")
    _common(run)
    run.add_argument("--source", choices=["camera", "video"], default=None)
    run.add_argument("--camera", default=None, help="camera index, file, or stream URL")
    run.add_argument("--backend", choices=["hamer", "mediapipe"], default=None)
    run.add_argument("--side", choices=["any", "left", "right"], default=None)
    run.add_argument("--pick", choices=["largest", "center", "track"], default=None)
    run.add_argument("--no-mirror", dest="mirror", action="store_false", default=None)
    run.add_argument("--draw", choices=["skeleton", "mesh", "none"], default=None)
    run.add_argument("--skip", type=int, default=None)
    run.add_argument("--smooth", type=float, default=None, help="smoothing time constant (s)")
    run.add_argument("--output", default=None, help="record the preview to this file")
    run.add_argument("--no-show", dest="show", action="store_false", default=None)
    run.add_argument("--no-viewer", dest="viewer", action="store_false", default=None)
    run.add_argument("--max-frames", type=int, default=None)
    run.add_argument("--fp16", dest="fp16", action="store_true", default=None)
    run.add_argument("--no-fp16", dest="fp16", action="store_false", default=None)
    run.add_argument("--data-dir", default=None)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--hand-model", default=None)
    run.add_argument("--calibration", default=None)
    run.add_argument("--phone", choices=["none", "phyphox"], default=None)
    run.add_argument("--phone-host", default=None, help="phone IP or host[:port]")
    run.add_argument("--phone-port", type=int, default=None)
    run.add_argument(
        "--phone-mount",
        choices=["back", "in_hand", "custom"],
        default=None,
        help="phone-to-hand mount: back (strapped to the back of the hand), "
        "in_hand (held screen-up in the palm), or custom (phone.mount_* values)",
    )
    run.add_argument(
        "--phone-scan",
        action="store_true",
        default=None,
        help="scan the local /24 for phyphox when the phone is unreachable",
    )
    run.add_argument(
        "--phone-follow",
        action="store_true",
        default=None,
        help="start following as soon as a ready phone sample arrives",
    )
    run.add_argument("--debug", action="store_true", default=None)
    run.set_defaults(func=cmd_run)

    cal = sub.add_parser("calibrate", help="capture and save a neutral pose")
    _common(cal)
    cal.add_argument("--camera", default=None)
    cal.add_argument("--backend", choices=["hamer", "mediapipe"], default=None)
    cal.add_argument("--side", choices=["any", "left", "right"], default=None)
    cal.add_argument("--pick", choices=["largest", "center", "track"], default=None)
    cal.add_argument("--no-mirror", dest="mirror", action="store_false", default=None)
    cal.add_argument("--draw", choices=["skeleton", "mesh", "none"], default=None)
    cal.add_argument("--skip", type=int, default=None)
    cal.add_argument("--no-show", dest="show", action="store_false", default=None)
    cal.add_argument("--data-dir", default=None)
    cal.add_argument("--checkpoint", default=None)
    cal.add_argument("--hand-model", default=None)
    cal.add_argument("--calibration", default=None)
    cal.add_argument("--debug", action="store_true", default=None)
    cal.set_defaults(func=cmd_calibrate)

    phone_test = sub.add_parser(
        "phone-test",
        help="print live phyphox attitude state and buffers",
    )
    _common(phone_test)
    phone_test.add_argument("--seconds", type=float, default=10.0)
    phone_test.add_argument("--host", default=None, help="phone IP or host[:port]")
    phone_test.add_argument("--port", type=int, default=None)
    phone_test.add_argument("--scan", action="store_true")
    phone_test.add_argument(
        "--list",
        action="store_true",
        help="print the experiment buffers and the selected source, then exit",
    )
    phone_test.add_argument("--json", action="store_true")
    phone_test.set_defaults(func=cmd_phone_test)

    setup = sub.add_parser("setup-models", help="download/check models")
    setup.add_argument("--data-dir", default=None)
    setup.add_argument("--copy-from", default=None)
    setup.add_argument("--check-only", action="store_true")
    setup.add_argument("--mediapipe-only", action="store_true")
    setup.add_argument("--hamer-only", action="store_true")
    setup.add_argument("--force", action="store_true")
    setup.set_defaults(func=cmd_setup_models)

    tactile = sub.add_parser("generate-tactile", help="regenerate tactile site/sensor XML")
    tactile.add_argument("--out", default=None)
    tactile.add_argument("--check", action="store_true")
    tactile.set_defaults(func=cmd_generate_tactile)

    render = sub.add_parser("render-pose", help="render the Shadow Hand for one image")
    render.add_argument("--image", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--backend", choices=["hamer", "mediapipe"], default="mediapipe")
    render.add_argument("--side", choices=["any", "left", "right"], default=None)
    render.add_argument("--pick", choices=["largest", "center", "track"], default=None)
    render.add_argument("--data-dir", default=None)
    render.add_argument("--checkpoint", default=None)
    render.add_argument("--hand-model", default=None)
    render.add_argument("--calibration", default=None)
    render.set_defaults(func=cmd_render_pose)

    doc = sub.add_parser("doctor", help="print environment and model diagnostics")
    doc.add_argument("--data-dir", default=None)
    doc.add_argument("--checkpoint", default=None)
    doc.add_argument("--hand-model", default=None)
    doc.add_argument("--strict-models", action="store_true")
    doc.add_argument("--json", action="store_true")
    doc.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "debug", False):
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - convert expected setup errors to exit code 2
        from shadowlink.capture import CaptureError
        from shadowlink.perception.base import PerceptionError
        from shadowlink.ui.window import UIError

        if isinstance(exc, (CaptureError, PerceptionError, UIError)):
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
