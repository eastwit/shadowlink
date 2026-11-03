# ShadowLink

**Monocular hand teleoperation of a Shadow Hand in MuJoCo.**

ShadowLink reconstructs a human hand from a single RGB camera and retargets its
continuous pose onto the 20 actuated joints of the Shadow Hand E3M5. The Shadow
Hand's 120 fingertip taxels are simulated and visualised in real time. There is
no gesture classifier: joint angles, finger spread, and thumb opposition are
mapped frame by frame, so the robot follows the operator's motion continuously.

[中文说明](README.zh-CN.md)

## Features

- **Two reconstruction backends**
  - **HaMeR** (recommended): monocular 3D hand reconstruction on the GPU,
    21 joints and a MANO mesh, crop-space camera lifted back to the full image.
  - **MediaPipe** (CPU fallback): `hand_world_landmarks` from the Hand Landmarker
    task; no CUDA or MANO download required.
- **Continuous retargeting** using palm-plane geometry: MCP/PIP/DIP flexion,
  finger spread, and thumb opposition. No discrete gestures or presets.
- **Two-stage asynchronous perception**: MediaPipe detection and HaMeR inference
  run in separate threads with latest-slot mailboxes, so the render/physics loop
  never waits for the model.
- **Stable, configurable control**: One-Euro filtering, rate limiting, a
  `waiting → tracking → lost` state machine, and a return-to-neutral ramp
  instead of jumping to zero.
- **Calibration that persists**: capture a neutral open-hand pose; store it per
  backend and validate it across processes.
- **Model-driven actuator layout**: joint names, order, types, and limits are
  read from the MJCF at startup, then checked. No duplicated constants.
- **Tactile simulation**: 120 taxels (5 fingers × 6 × 4) with force-dependent
  blue→red colouring.
- **Headless-friendly**: `--no-show --no-viewer`, synthetic frame sources, and a
  unit suite that needs no GPU, camera, or weights.
- **No path hacks**: no `os.chdir`, no `sys.path.insert`, no hard-coded user
  directories, and all model paths are absolute.

## Requirements

| Component | Version / note |
| --- | --- |
| Python | 3.10 (the combination below is tested on 3.10.22) |
| MuJoCo | 3.16+ |
| MediaPipe | 0.10.14+ |
| OpenCV | 4.8+ |
| HaMeR backend | PyTorch 2.0–2.5, CUDA optional but strongly recommended |
| GPU | Any CUDA GPU; measurements below use an RTX 4060 Laptop |

Windows 11 and Linux are supported. The MuJoCo viewer and OpenCV preview are
optional.

## Installation

### 1. Clone with the HaMeR submodule

```bash
git clone --recurse-submodules <repository-url> ShadowLink
cd ShadowLink
```

If you already cloned without submodules:

```bash
git submodule update --init
```

The pinned submodule is HaMeR at `3a01849` (no recursive ViTPose checkout is
needed for inference).

### 2. Create an environment

```bash
conda create -n shadowlink python=3.10 -y
conda activate shadowlink
```

Install the CUDA build of PyTorch first if you want GPU inference, following the
[official selector](https://pytorch.org/get-started/locally/). Then:

```bash
pip install -e ".[dev,hamer]"
pip install -e third_party/hamer --no-deps
```

`--no-deps` is intentional: HaMeR's `setup.py` pulls training-only packages
(detectron2, mmcv) that are not needed for inference, and its dependency pins
would override the PyTorch build you just installed.

For CPU/MediaPipe-only use:

```bash
pip install -e ".[dev]"
```

### 3. Fetch the models

```bash
shadowlink setup-models
```

This downloads the HaMeR demo checkpoint/config and the MediaPipe Hand Landmarker
task into the data directory. It does **not** download MANO.

MANO is distributed under a non-commercial research licence. Register at
<https://mano.is.tue.mpg.de>, download `MANO_RIGHT.pkl`, and place it at:

```text
<data-dir>/data/mano/MANO_RIGHT.pkl
```

The default data directory is `~/.cache/shadowlink` (Linux) or
`%USERPROFILE%\.cache\shadowlink` (Windows). Override it with `--data-dir` or
`SHADOWLINK_DATA`. An existing HaMeR `_DATA` directory can be reused directly:

```bash
shadowlink setup-models --data-dir D:/models/hamer/_DATA --check-only
```

### 4. Verify

```bash
shadowlink check --data-dir D:/models/hamer/_DATA
shadowlink doctor --data-dir D:/models/hamer/_DATA --strict-models
```

`doctor` reports Python/package versions, CUDA, model files, calibration, and
MJCF/tactile asset integrity.

## Quick start

```bash
# Live camera, HaMeR + MuJoCo viewer
shadowlink run --backend hamer --data-dir D:/models/hamer/_DATA

# CPU-only fallback
shadowlink run --backend mediapipe --data-dir D:/models/hamer/_DATA

# Replay a video without windows
shadowlink run --source video --camera demo.mp4 --backend hamer \
  --no-show --no-viewer --max-frames 120

# No camera: cycle through the 13 keyframe poses
shadowlink demo
```

### Controls

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `c` | capture a neutral pose (hold an open hand still) |
| `r` | reset physics and targets to the start pose |

### Common options

```text
--source camera|video        --camera INDEX_OR_PATH
--backend hamer|mediapipe    --side any|left|right   (anatomical side)
--pick largest|center|track  --skip N
--no-mirror                  --draw skeleton|mesh|none
--smooth SECONDS             --output RECORDING.mp4
--no-show --no-viewer        --max-frames N
--fp16 / --no-fp16           --calibration PATH
--config FILE                --set section.field=value
```

Configuration precedence is **CLI flags / `--set` > `--config` YAML > dataclass
defaults**. `configs/default.yaml` is kept byte-for-byte equivalent to the
default `AppConfig` and is locked by a test.

## How it works

```text
camera/video ──► FramePacket (immutable, mirrored once)
                     │
              ┌──────┴───────┐
              │  detection   │  MediaPipe HandLandmarker (VIDEO mode)
              │  thread      │  side filter + largest/center/IoU selector
              └──────┬───────┘
                     │ latest detection (old frames dropped)
              ┌──────┴───────┐
              │  inference   │  HaMeR crop → checkpoint → 21 joints + mesh
              │  thread      │  (MediaPipe backend reuses world landmarks)
              └──────┬───────┘
                     │ latest PerceptionResult (frame_seq + timestamps)
                     ▼
   retarget (palm frame → flexion/spread/opposition)
                     ▼
   One-Euro filter → slew limit → state machine → 20 actuator targets
                     ▼
   MuJoCo fixed-step physics + 120 touch sensors + viewer/preview
```

Key invariants:

- Frames are frozen dataclasses; the capture buffer is never mutated across
  threads.
- Every result carries `frame_seq`, `capture_ts`, and `finished_ts`, so the app
  can distinguish "no hand in this frame" from "the worker has not produced a
  result yet".
- The main loop only reads the newest result; older results are ignored.
- A perception failure is reported through `WorkerHealth`, logged, and makes the
  process exit non-zero rather than silently freezing on stale data.
- Left hands are mirror-canonicalised for retargeting. The Shadow Hand model is
  right-handed; the left Shadow Hand is not driven in v0.1.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries and the
threading model.

## Calibration

Hold an open hand in front of the camera and press `c` (or run
`shadowlink calibrate`). The neutral flexion/spread/opposition is saved to
`%APPDATA%\ShadowLink\calibration.json` on Windows or
`~/.config/shadowlink/calibration.json` on Linux. Calibration is bound to the
backend and side that produced it; switching backends asks you to calibrate
again instead of silently using the wrong zero.

See [docs/CALIBRATION.md](docs/CALIBRATION.md).

## Performance

Indicative numbers on an RTX 4060 Laptop, 1200×1799 video source, 30 frames:

| Metric | p50 |
| --- | --- |
| MediaPipe detection | 47 ms |
| HaMeR inference (fp16) | 62.5 ms |
| Observation age at consumption | ~180 ms |
| Main loop | 26.9 fps |

Detection and inference overlap across frames; the physics loop advances by
wall-clock time independently. Use `--skip 2` on slower machines, or
`--backend mediapipe` to drop the GPU model entirely. Actual numbers depend on
resolution, crop size, and thermal state. `--no-fp16` was not faster on this GPU.

## Offline tools

```bash
# Retarget one image and render the resulting Shadow Hand pose
shadowlink render-pose --image hand.jpg --out pose.png --backend hamer \
  --data-dir D:/models/hamer/_DATA

# Verify the checked-in tactile XML is reproducible from the meshes
shadowlink generate-tactile --check

# Regenerate it after changing fingertip geometry
shadowlink generate-tactile
```

## Development

```bash
pytest -q                       # no GPU, weights, or camera required
pytest -m "not hamer and not mediapipe" -q
ruff check src tests
ruff format --check src tests
```

Opt-in integration tests:

```bash
# HaMeR checkpoint + MediaPipe + one image
SHADOWLINK_TEST_DATA=D:/models/hamer/_DATA \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m hamer -q

# MediaPipe only
SHADOWLINK_TEST_HAND_MODEL=D:/models/hamer/_DATA/mediapipe/hand_landmarker.task \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m mediapipe -q
```

`tests/test_no_path_hacks.py` enforces that the installed package never changes
the working directory, modifies `sys.path`, or embeds user-specific paths.

Layout:

```text
src/shadowlink/
├── capture.py            camera/video sources, mirroring, synthetic source
├── config.py             dataclass config, YAML, dotted overrides
├── paths.py              package assets, data dir, models, calibration
├── types.py              immutable frames/detections/observations
├── app.py                the only orchestrator
├── cli.py                argparse entry point
├── perception/           MediaPipe detector, HaMeR backend, worker, selector
├── retarget/             geometry, mapping, smoothing, calibration
├── sim/                  MJCF loading, Shadow Hand, fixed-step clock, viewer
├── ui/                   overlay, telemetry, OpenCV window/recorder
├── tools/                setup-models, doctor, tactile generator, render-pose
└── assets/shadow_hand/   MJCF, OBJ meshes, tactile XML, original licences
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common setup
problems.

## Licence and attribution

ShadowLink's own code is MIT licensed. HaMeR is an MIT-licensed submodule;
Shadow Hand assets are Apache-2.0 from MuJoCo Menagerie / Shadow Robot; MediaPipe
is Apache-2.0. MANO and HaMeR checkpoints are **not** redistributed and remain
subject to their original licences. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

If you use the HaMeR backend, please cite the original work:

```bibtex
@inproceedings{pavlakos2023reconstructing,
  title   = {Reconstructing Hands in 3D with Transformers},
  author  = {Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and
             Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
  booktitle = {CVPR},
  year    = {2024}
}
```
