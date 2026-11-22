# ShadowLink

**Track fingers with a built-in camera and drive the whole Shadow Hand pose with a hand-held AprilTag in MuJoCo.**

ShadowLink reconstructs a human hand from a single RGB image and maps continuous
joint motion onto the Shadow Hand E3M5's 20 actuators. The scene contains 120
fingertip tactile taxels coloured by contact force. There is no gesture
classification: every frame computes finger flexion, spread, and thumb
opposition continuously.

The whole-hand base is controlled by a hand-held AprilTag (36h11 / ID 5 / 57 mm
by default). An Astra RGB camera watches the tag while the built-in camera keeps
driving finger perception.

## Features

- **Two finger reconstruction backends**
  - **HaMeR** (recommended): GPU monocular 3D hand reconstruction.
  - **MediaPipe** (CPU fallback): uses `hand_world_landmarks`; no CUDA or MANO.
- **Continuous retargeting** from palm geometry (MCP/PIP/DIP flexion, spread,
  thumb opposition), not discrete gestures.
- **Two-stage asynchronous perception**: detection and HaMeR inference run in
  separate latest-value threads; the physics loop never waits.
- **AprilTag whole-hand 6-DoF**: a dedicated Astra thread publishes tag pose.
  Translation and rotation drive the hand base relative to a zero pose; the
  first detection zeroes automatically. `g`/`p` toggle orientation/translation,
  `0` re-zeros both. Missing tags freeze the hand instead of snapping back.
- **Stable control**: One-Euro filtering, quaternion slerp, slew limiting, and a
  `waiting → tracking → lost` state machine with a rate-limited return.
- **Persistent calibration** for the finger neutral pose, per backend.
- **Tactile simulation**: 120 taxels (5 fingers × 6 × 4), blue→red.
- **Door scene**: a hinged door with a graspable lever handle at palm height.
  Push or pull the handle to swing the door; `r` closes it again.
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
| AprilTag | `pupil-apriltags` (official AprilTag 3 C library binding, core dependency) |
| HaMeR backend | PyTorch 2.0–2.5, CUDA optional but strongly recommended |
| GPU | Any CUDA GPU; measurements below use an RTX 4060 Laptop |
| Cameras | built-in/UVC webcam for fingers + Orbbec Astra RGB for the tag |

Windows 11 and Linux are supported. The MuJoCo viewer and OpenCV preview are
optional.

## Installation

### 1. Clone with the HaMeR submodule

```bash
git clone --recurse-submodules <repo-url> ShadowLink
cd ShadowLink
```

If already cloned without submodules:

```bash
git submodule update --init
```

The submodule is pinned to HaMeR `3a01849`. Recursive ViTPose initialization is
not required for inference.

### 2. Create the environment

```bash
conda create -n shadowlink python=3.10 -y
conda activate shadowlink
```

For GPU inference, install the CUDA-matched PyTorch build first (see the
[official selector](https://pytorch.org/get-started/locally/)), then:

```bash
pip install -e ".[dev,hamer]"
pip install -e third_party/hamer --no-deps
```

`--no-deps` is intentional: HaMeR's `setup.py` pulls training-only packages
(detectron2, mmcv, ...) and may overwrite PyTorch.

CPU/MediaPipe only:

```bash
pip install -e ".[dev]"
```

### 3. Fetch the models

```bash
shadowlink setup-models
```

This downloads the HaMeR demo checkpoint/config and MediaPipe Hand Landmarker;
it does **not** download MANO.

MANO uses a non-commercial research licence. Register at
<https://mano.is.tue.mpg.de>, download `MANO_RIGHT.pkl`, and place it at:

```text
<data-dir>/data/mano/MANO_RIGHT.pkl
```

The default data directory is `~/.cache/shadowlink` (Linux) or
`%USERPROFILE%\.cache\shadowlink` (Windows). Override it with `--data-dir` or
`SHADOWLINK_DATA`.

### 4. Verify

```bash
shadowlink check --data-dir D:/models/hamer/_DATA
shadowlink doctor --data-dir D:/models/hamer/_DATA --strict-models
```

`doctor` checks Python/dependency versions, CUDA, model files, calibration, and
the MJCF/tactile assets.

## Quick start

```bash
# Built-in camera + HaMeR + Astra tag + MuJoCo viewer
shadowlink run --backend hamer --data-dir D:/models/hamer/_DATA

# Two-phone setup: IP Webcam for hands + Plugcam for the tag
shadowlink run --config configs/ipwebcam_plugcam.yaml --backend hamer \
  --data-dir D:/models/hamer/_DATA

# CPU finger fallback
shadowlink run --backend mediapipe --data-dir D:/models/hamer/_DATA

# Replay a video without windows
shadowlink run --source video --camera demo.mp4 --backend hamer \
  --no-show --no-viewer --max-frames 120

# No camera: cycle through the 13 keyframe poses
shadowlink demo
```

Default camera indices:

| Index | Device | Role |
| --- | --- | --- |
| `0` | Astra Pro HD Camera | tag camera |
| `1` | built-in HD Webcam | finger camera |

`configs/ipwebcam_plugcam.yaml` uses the current phone setup: IP Webcam at
`http://192.168.1.109:8080/video` for hands and Plugcam at DirectShow index `1`
for the tag. Update those two values when the phone IP or enumeration changes.

If Windows enumerates cameras differently, change `capture.camera` /
`tag.camera` in the config, or pass `--camera` / `--tag-camera`.
List DirectShow devices:

```bash
python -c "from pygrabber.dshow_graph import FilterGraph; print(list(enumerate(FilterGraph().get_input_devices())))"
```

### Controls

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `c` | capture a finger neutral pose (hold an open hand still) |
| `g` | toggle tag whole-hand orientation follow |
| `0` | re-zero both tag position and orientation at the current tag pose |
| `p` | toggle tag whole-hand translation |
| `r` | reset physics, targets, tag zero, translation, orientation, and the door |

### Common options

```text
--source camera|video        --camera INDEX_OR_PATH
--no-position                --position-preset front|palm|back
--position-gain FLOAT        --position-deadzone M
--tag-camera INDEX_OR_PATH   --tag-api any|dshow|msmf
--tag-family tag36h11        --tag-id N
--tag-size M                 --tag-mirror auto|on|off
--tag-zero-mode sample|fixed --tag-zero-distance M
--tag-show / --no-tag-show   --tag-follow
--backend hamer|mediapipe    --side any|left|right
--pick largest|center|track  --skip N
--no-mirror                  --draw skeleton|mesh|none
--smooth SECONDS             --output RECORDING.mp4
--no-show --no-viewer        --max-frames N
--fp16 / --no-fp16           --calibration PATH
--config FILE                --set section.field=value
```

Priority: **CLI / `--set` > `--config` YAML > dataclass defaults**.
`configs/default.yaml` mirrors `AppConfig` exactly and is locked by a test.

## How it works

```text
built-in camera ──► FramePacket ──► MediaPipe detector ──► HaMeR inference
                                             │
                                             ▼
                              21 joints → flexion/spread/opposition
                                             ▼
                     One-Euro → slew limit → 20 actuator targets

Astra RGB ──► AprilTag thread ──► 6-DoF relative zero → pos + orientation
                                             ▼
                     ShadowHand.set_macro_pose (whole-hand base)
                                             ▼
                     MuJoCo fixed-step physics + 120 taxels + viewer
```

Key invariants:

- Frames are frozen dataclasses; capture buffers are never mutated across threads.
- Results carry `frame_seq`, `capture_ts`, `finished_ts`; the main loop only uses
  the newest result.
- The tag service publishes into an independent latest-value slot; the loop never
  blocks on its camera or detector, and freezes when the sample goes stale.
- Left hands are canonicalised for retargeting. The left Shadow Hand is not
  driven in v0.1.
- Astra's RGB stream is horizontally mirrored; `tag.mirror: auto` corrects it.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for module boundaries and
threading.

## Calibration

Face the camera with an open hand and press `c`, or run:

```bash
shadowlink calibrate --backend hamer --data-dir D:/models/hamer/_DATA
```

The neutral file is stored at `%APPDATA%\ShadowLink\calibration.json` on Windows
or `~/.config/shadowlink/calibration.json` on Linux. It is tied to the backend
and hand side.

## AprilTag whole-hand 6-DoF

- Print or display `src/shadowlink/assets/apriltag/tag36h11_id5.png` **without
  mirroring**.
- `size.md` and the config default are 57 mm (black-border edge).
- A separate **AprilTag Detection** window shows the Astra feed, tag outline,
  coordinate axes, ID, distance, mirror state, and detection rate. Hide it with
  `--no-tag-show`.
- The tag camera defaults to DSHOW at 640×480 (~20 FPS). DSHOW is more reliable
  for Astra than MSMF; `--tag-api msmf` can reach ~30 FPS but may open slowly.
- Translation zero defaults to the **image center at 1.00 m depth**
  (`tag.zero_mode: fixed`, `tag.zero_distance_m: 1.0`). Hold the tag near that
  point for a neutral hand offset. The orientation zero is latched from the
  **first valid tag** and stays absolute for the session; `g` only toggles
  following. `p` toggles translation, `0` explicitly re-zeroes orientation and
  resets fixed translation, and `r` resets everything. Set
  `tag.zero_mode: sample` to use the first tag position instead.
- Initial hand pose: **palm down**, middle finger toward the door.
- Tag axes: **X+** = forward / middle finger, **Y+** = right, **Z+** = palm
  direction, **Z−** = up. Rotation mapping: **green → roll**, **red → pitch**,
  **blue → yaw**, so the three axes stay independent.
- Translation uses the `overhead` preset: tag moving **right** moves the hand
  toward the door (+world x), and tag moving **closer** to the overhead camera
  moves the hand up (+world z).
- There is no spatial clamp: the hand can follow the tag arbitrarily far from
  the zero. If the tag pose is lost for 0.5 s, both position and orientation
  freeze in place instead of returning to zero.
- Hold the tag roughly 0.5–1.2 m from the Astra in even light, near the cyan
  center/1 m guide. See
  [docs/APRILTAG.md](docs/APRILTAG.md) for the printable asset, axis mapping,
  intrinsics, and troubleshooting.

## Performance

RTX 4060 Laptop, 1200×1799 video source, 30 reference frames:

| Metric | p50 |
| --- | --- |
| MediaPipe detection | 47 ms |
| HaMeR inference (fp16) | 62.5 ms |
| Observation age at consumption | about 180 ms |
| Main loop | 26.9 fps |

Detection and inference overlap across frames; the physics loop advances by wall
clock. On slower machines use `--skip 2` or `--backend mediapipe`.

## Offline tools

```bash
shadowlink render-pose --image hand.jpg --out pose.png --backend hamer \
  --data-dir D:/models/hamer/_DATA
shadowlink generate-tactile --check
shadowlink generate-tactile
```

## Development

```bash
pytest -q
pytest -m "not hamer and not mediapipe" -q
ruff check src tests
ruff format --check src tests
```

Opt-in integration tests:

```bash
SHADOWLINK_TEST_DATA=D:/models/hamer/_DATA \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m hamer -q

SHADOWLINK_TEST_HAND_MODEL=D:/models/hamer/_DATA/mediapipe/hand_landmarker.task \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m mediapipe -q
```

`tests/test_no_path_hacks.py` enforces the absence of `os.chdir`, `sys.path`
modifications, and user absolute paths.

Directory layout:

```text
src/shadowlink/
├── capture.py            camera/video sources, mirroring, synthetic frames
├── config.py             dataclass config, YAML, dotted overrides
├── paths.py              package assets, data dir, models, calibration
├── types.py              immutable frames/detections/observations
├── app.py                the only orchestrator
├── cli.py                argparse entry point
├── math3d.py             NumPy-only quaternion helpers
├── perception/           MediaPipe detector, HaMeR backend, worker, selector
├── tag/                  AprilTag camera thread and relative 6-DoF controller
├── retarget/             geometry, mapping, smoothing, calibration
├── sim/                  MJCF loading, Shadow Hand, fixed-step clock, viewer
├── ui/                   overlay, telemetry, OpenCV window/recorder
├── tools/                setup-models, doctor, tactile generator, render-pose
└── assets/shadow_hand/   MJCF, OBJ meshes, tactile XML, original licences
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

## Licence and attribution

ShadowLink's own code is MIT. HaMeR is a MIT submodule; the Shadow Hand assets
come from MuJoCo Menagerie / Shadow Robot (Apache-2.0); MediaPipe is Apache-2.0;
AprilTag is BSD-2-Clause through the `pupil-apriltags` binding. MANO and the
HaMeR checkpoint are **not** redistributed and remain under their own licences.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

When using the HaMeR backend, cite:

```bibtex
@inproceedings{pavlakos2023reconstructing,
  title   = {Reconstructing Hands in 3D with Transformers},
  author  = {Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and
             Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
  booktitle = {CVPR},
  year    = {2024}
}
```
