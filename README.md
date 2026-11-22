# ShadowLink

**Track fingers and drive the whole Shadow Hand from one RGB camera, a hand-held
AprilTag, and an optional phyphox phone for stable tilt.**

ShadowLink reconstructs a human hand from a single RGB image and maps continuous
joint motion onto the Shadow Hand E3M5's 20 actuators. The scene contains 120
fingertip tactile taxels coloured by contact force. There is no gesture
classification: every frame computes finger flexion, spread, and thumb
opposition continuously.

The whole-hand base uses a hand-held AprilTag (36h11 / ID 5 / 57 mm by default)
for translation and stable blue-axis yaw. The two noisier tag tilt axes can be
replaced by a phone running phyphox on the hand. In the recommended setup one
Plugcam/IP Webcam feed is shared by hand perception and tag detection, so no
second camera is required.

## Features

- **Two finger reconstruction backends**
  - **HaMeR** (recommended): GPU monocular 3D hand reconstruction.
  - **MediaPipe** (CPU fallback): uses `hand_world_landmarks`; no CUDA or MANO.
- **Continuous retargeting** from palm geometry (MCP/PIP/DIP flexion, spread,
  thumb opposition), not discrete gestures.
- **Two-stage asynchronous perception**: detection and HaMeR inference run in
  separate latest-value threads; the physics loop never waits.
- **AprilTag + phone whole-hand pose**: the tag supplies translation and
  blue-axis yaw; an optional phyphox phone supplies yaw+pitch for the two noisy
  tilt axes and its screen-normal roll is discarded. `m` cycles eight phone
  mount mappings live. Without a phone, the tag supplies full orientation.
- **One-camera mode**: `capture.source: tag` reuses the tag camera's raw frames
  for HaMeR/MediaPipe, so the same Plugcam/IP Webcam view drives both fingers and
  tag pose.
- **Stable control**: One-Euro filtering, quaternion slerp, slew limiting, and a
  `waiting → tracking → lost` state machine with a rate-limited return.
- **Persistent calibration** for the finger neutral pose, per backend.
- **Tactile simulation**: 120 taxels (5 fingers × 6 × 4), blue→red.
- **Tabletop scene**: a table under the hand with a free 5 cm, 10 g block.
  The hand starts palm-down above the surface; `r` restores the block. Hand and
  block contacts use high friction and stiff `solimp`/`solref` settings.
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
| Cameras | one UVC camera shared by fingers and tag (Plugcam/IP Webcam), or built-in + Astra in two-camera mode |
| Phone (optional) | [phyphox](https://phyphox.org) with Remote Access on the hand for stable tilt |

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
# Recommended: one Plugcam/IP Webcam + phyphox phone + HaMeR + MuJoCo viewer
shadowlink run --config configs/plugcam_hand_tag.yaml --backend hamer \
  --data-dir D:/models/hamer/_DATA

# Two-camera fallback: built-in camera for fingers + Astra RGB for the tag
shadowlink run --backend hamer --data-dir D:/models/hamer/_DATA

# CPU finger fallback
shadowlink run --backend mediapipe --data-dir D:/models/hamer/_DATA

# Replay a video without windows
shadowlink run --source video --camera demo.mp4 --backend hamer \
  --no-show --no-viewer --max-frames 120

# No camera: cycle through the 13 keyframe poses
shadowlink demo
```

Default `configs/default.yaml` (two-camera fallback) indices:

| Index | Device | Role |
| --- | --- | --- |
| `0` | Astra Pro HD Camera | tag camera |
| `1` | built-in HD Webcam | finger camera |

The recommended `configs/plugcam_hand_tag.yaml` uses DirectShow index `1`
(Plugcam) for **both** roles. List your own indices with the command below.

`configs/plugcam_hand_tag.yaml` is the single-camera profile. It sets
`capture.source: tag`, so the `TagService` camera becomes the only frame source:
each raw frame is published once, used by HaMeR/MediaPipe for the fingers, and
used by the AprilTag detector for the hand pose. Point that camera so both the
hand and the tag are visible. The hand-camera setting (`capture.camera`) is
ignored in this mode.

That profile also enables a phone on the hand for the two tag axes that are
least stable. The phone runs [phyphox](https://phyphox.org) with **Remote
Access** enabled; ShadowLink polls it over HTTP (no extra Python package). The
tag supplies yaw from its blue axis, while the phone supplies yaw and pitch;
the phone roll channel (rotation about the screen normal) is deliberately
ignored. Set `phone.host` to the address shown by phyphox:

```yaml
phone:
  source: phyphox
  host: 192.0.2.10
  port: 8080
  axes: yaw_pitch
  mount_preset: yaw_pitch
```

`mount_preset=yaw_pitch` maps the phone's yaw/pitch onto the hand's two non-yaw
axes. The default is the `swap-d` direction combination. If your phone is
mounted differently, press `m` to cycle all eight mappings live or set
`phone.mount_preset`/`mount_w/x/y/z`. The HUD shows the phone rate, tilt state,
and active mount name. `g` still toggles orientation follow, and `0` re-zeroes
tag and phone together. See [docs/PHONE.md](docs/PHONE.md) for details.

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
| `m` | cycle the phone-to-hand mount mapping and re-zero orientation |

The `m` key cycles eight axis/sign mappings while the app keeps running. The
HUD line `phone ... mount <name>` shows the active one; use it if the phone's
yaw/pitch axes feel swapped or inverted.
| `r` | reset physics, targets, tag zero, translation, orientation, and the block |

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
Plugcam/IP Webcam ──► FramePacket ──► MediaPipe detector ──► HaMeR inference
        │                                    │
        │                                    ▼
        │                     21 joints → flexion/spread/opposition
        │                                    ▼
        │                    One-Euro → slew limit → 20 actuator targets
        │
        └──► AprilTag thread ──► translation + blue-axis yaw
                                        │
phyphox phone ──► yaw + pitch ─────────┤
                                        ▼
                    ShadowHand.set_macro_pose (whole-hand base)
                                        ▼
                MuJoCo fixed-step physics + 120 taxels + viewer
```

The built-in + Astra two-camera layout feeds the same two branches through
separate capture threads.

Key invariants:

- Frames are frozen dataclasses; capture buffers are never mutated across threads.
- Results carry `frame_seq`, `capture_ts`, `finished_ts`; the main loop only uses
  the newest result.
- The tag service publishes into an independent latest-value slot; the loop never
  blocks on its camera or detector, and freezes when the sample goes stale.
- The phone service is a separate HTTP polling thread; a stale phone freezes
  orientation while tag translation continues.
- Left hands are canonicalised for retargeting. The simulated Shadow Hand is the
  right hand.
- Phone and UVC streams may be horizontally mirrored; `tag.mirror: auto`
  corrects the tag camera.

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

## Whole-hand pose: AprilTag + phone

- Print or display `src/shadowlink/assets/apriltag/tag36h11_id5.png` **without
  mirroring**.
- `size.md` and the config default are 57 mm (black-border edge).
- A separate **AprilTag Detection** window shows the camera feed, tag outline,
  coordinate axes, ID, distance, mirror state, and detection rate. Hide it with
  `--no-tag-show`.
- The tag camera defaults to DSHOW at 640×480. The Plugcam profile uses
  1920×1080 with `quad_decimate: 2.0`. DSHOW is more reliable than MSMF.
- Translation zero defaults to the **image center at 1.00 m depth**
  (`tag.zero_mode: fixed`, `tag.zero_distance_m: 1.0`). Hold the tag near that
  point for a neutral hand offset. The orientation zero is latched from the
  first valid tag+phone pair and stays absolute for the session; `g` only
  toggles following. `p` toggles translation, `0` explicitly re-zeroes
  orientation and resets fixed translation, and `r` resets everything. Set
  `tag.zero_mode: sample` to use the first tag position instead.
- With `phone.source: phyphox`, the tag contributes only blue-axis yaw while
  the phone supplies the two tilt axes. Press `m` to cycle phone mounts if the
  axes feel swapped or reversed. See [docs/PHONE.md](docs/PHONE.md).
- Initial hand pose: **palm down**, middle finger toward the block (+world x).
- Tag axes: **X+** = forward / middle finger, **Y+** = right, **Z+** = palm
  direction, **Z−** = up. Without a phone, full tag rotation maps **green →
  roll**, **red → pitch**, **blue → yaw**. With a phone, only blue → yaw is
  taken from the tag.
- Translation uses the `overhead` preset: tag moving **right** moves the hand
  toward the block (+world x), and tag moving **closer** to the overhead
  camera moves the hand up (+world z).
- There is no spatial clamp: the hand can follow the tag arbitrarily far from
  the zero. If the tag pose is lost for 0.5 s, both position and orientation
  freeze in place instead of returning to zero.
- Noise rejection: detections prefer `hamming=0` and the highest decision
  margin; a single-frame rotation jump over 35° is rejected until it persists
  for 5 frames. Orientation output uses a 0.12 s slerp and 3 rad/s rate limit.
- Hold the tag roughly 0.5–1.2 m from the tag camera in even light, near the
  cyan center/1 m guide. See
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
├── tag/                  AprilTag camera thread, shared-frame source, controller
├── phone/                phyphox HTTP client, poller, yaw/pitch tilt mapping
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
AprilTag is BSD-2-Clause through the `pupil-apriltags` binding. phyphox is an
optional third-party phone app and is not bundled. MANO and the HaMeR checkpoint
are **not** redistributed and remain under their own licences. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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
