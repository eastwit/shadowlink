# Troubleshooting

Start with:

```bash
shadowlink doctor --data-dir D:/models/hamer/_DATA --strict-models --json
```

## `ModuleNotFoundError: No module named 'hamer'`

The submodule is not initialised or HaMeR is not installed in this environment:

```bash
git submodule update --init
pip install -e third_party/hamer --no-deps
```

Check with `python -c "import hamer; print(hamer.__file__)"`. The path should be
inside this repository's `third_party/hamer`.

## `use_2to3 is invalid` while installing chumpy

`chumpy` 0.70 predates modern setuptools. Install it from source and retry:

```bash
pip install git+https://github.com/mattloper/chumpy
pip install -e ".[dev,hamer]"
```

## `Unsupported PyOpenGL platform` / pyrender/EGL error on Windows

Upstream HaMeR forces `PYOPENGL_PLATFORM=egl`, which is not available on a
default Windows install. ShadowLink sets `gl` before importing HaMeR and loads
the model with `init_renderer=False`, so the renderer is never created.

If you import HaMeR yourself in the same process, set the variable first:

```python
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "gl")  # Linux: "egl"
```

On a headless Linux machine, install the EGL libraries and keep
`PYOPENGL_PLATFORM=egl`. MuJoCo also reads `MUJOCO_GL`; `egl` or `osmesa` are
typical for servers.

## `CUDA was requested but is not available` or torch CPU-only

Install a CUDA build of PyTorch that matches your driver, then reinstall the
extra:

```bash
pip uninstall torch torchvision
# install the wheel from https://pytorch.org/get-started/locally/
pip install -e ".[dev,hamer]" --no-deps  # or install the extra's packages manually
```

`shadowlink doctor` prints `torch.version.cuda` and the device name. Without
CUDA, use `--backend mediapipe` or accept slow HaMeR inference on the CPU.

## HaMeR checkpoint will not load

- Confirm `<data-dir>/hamer_ckpts/checkpoints/hamer.ckpt` and
  `<data-dir>/hamer_ckpts/model_config.yaml` both exist.
- The loader expects `MODEL.IMAGE_SIZE=256`; the public ViT checkpoint satisfies
  this. A mismatched config produces an explicit error.
- `strict=False` is intentional: the training-only backbone weights and renderer
  are not part of inference. Unexpected-key warnings are expected.
- On CPU, expect tens of seconds per frame; this backend is meant for a GPU.

## `MANO model not found`

MANO cannot be downloaded automatically. Register at
<https://mano.is.tue.mpg.de>, download the model, and place `MANO_RIGHT.pkl` at:

```text
<data-dir>/data/mano/MANO_RIGHT.pkl
```

The HaMeR checkpoint is derived from MANO and follows the same non-commercial
licence. Do not commit either file.

## No hand detected

- Make sure the whole hand is visible and reasonably lit.
- `--side` filters the **anatomical** hand, not the mirrored image. With a
  mirrored webcam, your right hand appears on the left of the preview but is
  still `right`.
- Try `--pick largest` (the default `track` needs an initial detection).
- Lower the detector threshold:
  `--set perception.min_detection_confidence=0.2`.
- `--no-mirror` if another layer already mirrors the image.
- For HaMeR, check that the detector found a box in the HUD before blaming the
  checkpoint; an absent box means inference is never called.

## Hand side is swapped

MediaPipe labels handedness as it appears in the processed image. ShadowLink
records both `apparent_side` and `anatomical_side` and flips the latter when the
frame was mirrored. If you feed already-mirrored video through `--source video`,
use `--no-mirror` so the frame is mirrored exactly once.

## Targets are sluggish or jittery

- `--smooth 0.03` reduces lag; `--smooth 0.1` reduces jitter.
- `--set smoothing.max_rate_rad_s=25` allows faster large motions.
- `--skip 1` ensures every frame reaches perception. Increasing `--skip` lowers
  GPU load but makes tracking less responsive.
- On a slow machine, `--backend mediapipe` removes HaMeR entirely.
- If the hand is lost, the controller holds briefly and then ramps back to
  neutral at `smoothing.return_rate_rad_s`.

## Calibration is ignored

The app logs a warning when:

- the file's `schema_version` is not `1`;
- the file is malformed or missing keys;
- it was saved for a different backend (`hamer` vs `mediapipe`);
- `--side` disagrees with the stored side (warning only).

Recalibrate with the backend and side you are actually using. Delete the file or
pass a different `--calibration` path to start fresh.

## Camera will not open

- `--camera 0` is the default index. Try `1`, `2`, ...
- On Windows, try `--set capture.api=dshow` or `msmf`.
- For a video file use `--source video --camera path/to/file.mp4`.
- Close other applications that hold the camera.
- `capture.buffer_size=1` requests a one-frame driver buffer where supported,
  reducing backlog.

## Preview or viewer fails

Use `--no-show --no-viewer` for headless runs. The OpenCV window needs a
desktop session; the MuJoCo passive viewer needs OpenGL. Neither is required for
physics, retargeting, or telemetry.

## Recording produces no file

`VideoRecorder` opens lazily on the first preview frame and falls back from
`mp4v` to MJPG/AVI if the requested codec is unavailable. It raises
`RecordingError` if both fail; check the log for the attempted size/codecs.
Recording requires `--show` or an explicit output that triggers preview
generation; `--no-show --no-viewer` without `--output` does not create frames.

## Windows temp-directory permission errors in pytest

Use a project-local base temp directory:

```bash
pytest --basetemp=.pytest_tmp
```

`.pytest_tmp/` is gitignored. This is a local sandbox/antivirus quirk, not a
test failure.

## Tests are skipped

The default suite skips tests that need weights or a real image. They are
opt-in:

```bash
SHADOWLINK_TEST_DATA=D:/models/hamer/_DATA \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m hamer -q

SHADOWLINK_TEST_HAND_MODEL=D:/models/hamer/_DATA/mediapipe/hand_landmarker.task \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m mediapipe -q
```

## HaMeR inference is slower than the README benchmark

The benchmark uses a 1200×1799 source on an RTX 4060 Laptop. Latency scales with
image area and crop size (`perception.rescale_factor`), and thermal throttling
can halve throughput. Measure your own:

```bash
shadowlink run --source video --camera demo.mp4 --backend hamer \
  --no-show --no-viewer --max-frames 300
```

The summary line reports `detect_ms` and `infer_ms` p50/p95 and observation age.
