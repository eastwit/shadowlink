# Troubleshooting

## `ModuleNotFoundError: No module named 'hamer'`

Install the pinned submodule without its training dependencies:

```bash
git submodule update --init
pip install -e third_party/hamer --no-deps
```

## `use_2to3 is invalid` while installing chumpy

Use a modern `setuptools` and install HaMeR without dependencies as above. If
the error persists, install chumpy from its maintained fork or pin
`setuptools<58` only inside an isolated build environment.

## `Unsupported PyOpenGL platform` / pyrender/EGL error on Windows

HaMeR's mesh renderer needs the optional `pyrender`/PyOpenGL stack. Finger
retargeting and tag control work without it; set `ui.draw: skeleton` or
`--draw skeleton` and avoid the mesh renderer. Alternatively use
`--backend mediapipe`.

## `CUDA was requested but is not available` or torch CPU-only

Install the CUDA-matched PyTorch build from the official selector, then reinstall
`pip install -e ".[dev,hamer]"`. For CPU-only runs use `--backend mediapipe` or
`--no-fp16`.

## HaMeR checkpoint will not load

Check `shadowlink check --data-dir ... --strict-models`. The checkpoint,
`model_config.yaml`, `MANO_RIGHT.pkl`, and the MediaPipe task must all exist.
The MANO model is not downloadable from this project; register at
<https://mano.is.tue.mpg.de>.

## `MANO model not found`

Place `MANO_RIGHT.pkl` at `<data-dir>/data/mano/MANO_RIGHT.pkl`, or pass
`--data-dir` / set `SHADOWLINK_DATA`.

## No hand detected

- Keep the whole hand in frame with even lighting and a plain background.
- Try `--side any` or the correct anatomical side.
- Lower `perception.min_detection_confidence` and try `--skip 1`.
- Try `--backend mediapipe` to isolate model/GPU issues.
- Check the preview; if the camera is mirrored, `capture.mirror` should match the
  physical setup.

## Hand side is swapped

The tracker distinguishes apparent image side from anatomical side. Use
`--side left|right` to force the intended hand. Mirroring the capture once is
the only reflection applied.

## Targets are sluggish or jittery

Adjust `smoothing.max_rate_rad_s`, `smoothing.min_cutoff`,
`smoothing.beta`, and `smoothing.tau`. Use `--smooth SECONDS` for a quick
exponential change. Slower hardware may need `--skip 2`.

## Calibration is ignored

Calibration is tied to the backend and side. If you switch backends or
`--side`, recapture with `c` or `shadowlink calibrate`. The file lives under
`%APPDATA%\ShadowLink` (Windows) or `~/.config/shadowlink` (Linux).

## Camera will not open

Close other apps using the camera and verify the index:

```bash
python -c "from pygrabber.dshow_graph import FilterGraph; print(list(enumerate(FilterGraph().get_input_devices())))"
```

In the default two-camera mode the hand and tag camera indices must differ.
Change `capture.camera` or `tag.camera` in `configs/default.yaml`. With
`capture.source: tag` the tag camera is shared, so `capture.camera` is ignored.
The tag camera defaults to DSHOW; MSMF on Astra can take tens of seconds to open.

## Preview or viewer fails

Use `--no-show` to disable the OpenCV preview or `--no-viewer` to disable the
MuJoCo viewer. This is useful over remote sessions and in CI.

## Recording produces no file

`--output path.mp4` records the annotated preview. If the preview is disabled
with `--no-show`, recording still runs. Check the parent directory is writable.

## AprilTag problems

See [APRILTAG.md](APRILTAG.md) for the full setup.

### HUD says `tag error`

The tag camera could not be opened, or `pupil-apriltags` is not installed.
`pupil-apriltags` is a core dependency now, so reinstall the package
(`pip install -e ".[dev]"`), close apps using the Astra, and check
`tag.camera` / `tag.api`.

### No AprilTag Detection window

It is disabled by `--no-show`, `--no-tag-show`, or `tag.show: false`. The window
appears after the first Astra frame, which can take a few seconds with DSHOW.
Check the `tag` HUD line for `error` and look for a window titled
`AprilTag Detection` in the taskbar.

### HUD says `tag searching`

No valid decode. Check `tag.family`, `tag.tag_id`, and `tag.size_m`; hold the tag
0.5–1.2 m from the Astra in even light, near the cyan center/1 m guide. Astra's
RGB picture is mirrored, so keep `tag.mirror: auto` (or `on`). Never print or
display a mirrored tag.

### Tag is detected but the hand does not move

Look for `tag pos on` / `tag rot on` in the HUD. Press `p` or `g`, then `0` to
reset translation and re-zero orientation. By default translation is measured
from **image center at 1 m depth**, shown by the cyan guide in the AprilTag
Detection window; move the tag away from that point to move the hand. Holding
the tag at the same point produces no motion.

### Directions are reversed or the hand spins around the wrong point

Flip translation with `position.invert_x/y/z`, tag rotation with
`tag.rotation_invert_x/y/z`, or select another `tag.pivot` (`palm`, `wrist`,
`forearm`). Phone tilt axes are tuned live with `m`; the HUD shows the active
mount name. See [PHONE.md](PHONE.md).

### Phone/phyphox problems

- HUD says `phone error` or `connecting`: verify Remote Access with
  `curl http://PHONE:8080/config` from the same Wi-Fi. Check the experiment is
  measuring and `phone.host`/`port` are correct.
- HUD says `phone stale` or orientation freezes: Wi-Fi dropped or the phone
  paused. Tag translation keeps working.
- Orientation never starts: both tag and phone must have a fresh sample to latch
  the zero. Check the tag HUD line too.
- Phone axes swapped or reversed: press `m` to cycle eight mappings; no restart
  is needed.
- Phone rate is low: lower `phone.poll_hz` from 50 to 30, close other traffic,
  and avoid guest Wi-Fi/client isolation.

### The block slips out of the fingers

The hand collision geoms use `friction="2.0 0.05 0.005"`,
`solimp="0.9 0.99 0.0001"`, and `solref="0.004 1"` in `right_hand.xml` /
`left_hand.xml`; the block uses the same contact settings. Increase friction or
stiffness further only if the simulation stays stable, or close the fingers more
with the retargeting gains.

### No 30 FPS from the tag camera

DSHOW on Astra is ~20 FPS at 640×480 and ~10 FPS at 720p. `--tag-api msmf` can
reach ~30 FPS but opening the device may be slow. Keep `tag.quad_decimate: 1.0`
for small tags.

## Windows temp-directory permission errors in pytest

Use a project-local base temp directory:

```bash
pytest --basetemp=.pytest_tmp
```

`.pytest_tmp/` is gitignored. This is a local sandbox/antivirus quirk, not a
test failure.

## Tests are skipped

Opt-in tests need model files or a real image:

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
