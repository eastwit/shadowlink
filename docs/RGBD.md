# RGBD camera: whole-hand translation

ShadowLink can add a second, independent control channel: the **physical
position** of the palm, measured by an RGBD camera, translates the whole
simulated hand. Finger retargeting and the optional phone orientation keep
working unchanged.

The depth stream is used only for position. It does **not** feed the finger
retargeter, and it does not replace the colour camera used by MediaPipe/HaMeR.

## Hardware

The reference device is an **Orbbec Astra Mini S Pro**:

| Stream | Interface | Typical mode |
| --- | --- | --- |
| Depth | OpenNI2 (`openni` Python package + `OpenNI2.dll`) | 640×480 @ 30 Hz, `uint16` millimetres |
| Colour | separate UVC camera through OpenCV DirectShow | 640×480 or the camera's default |

The two sensors are independent. There is no hardware synchronisation and no
factory registration between colour and depth, so the mapping is approximate by
design: landmarks are normalised, projected into the depth image, and sampled
with a small median window. Do not expect millimetre-level colour/depth
alignment.

The Astra has a close-range limit around 0.6 m. Keep the hand between roughly
**0.6 m and 1.2 m** from the sensor with the palm visible to the camera. The
translation mapping is anatomy-relative, so the hand does not need a fixed
orientation.

## Install

The depth SDK is an optional dependency:

```bash
pip install -e ".[rgbd]"
# or, into an existing environment:
pip install "openni>=2.3,<3"
```

`openni` is lazy-imported: a normal camera/video run does not need it.

### Point ShadowLink at `OpenNI2.dll`

The directory you pass must contain `OpenNI2.dll` (Windows) or
`libOpenNI2.dll` (Linux). Resolution order:

1. `--depth-sdk PATH`
2. `capture.depth_sdk` in YAML
3. the `OPENNI2_REDIST64` environment variable
4. the `OPENNI2_REDIST` environment variable
5. common Orbbec installation directories under `Program Files`

ShadowLink never embeds a user-specific path. If none of these work, pass the
directory explicitly:

```bash
shadowlink run --source rgbd --depth-sdk "D:/sdk/OpenNI2/Redist" ...
```

Windows DLL loading is handled with `os.add_dll_directory`; the handle is kept
for the lifetime of the process and released on close.

## Verify the camera first

```bash
shadowlink rgbd-check --camera 1 \
  --depth-sdk "D:/sdk/OpenNI2/Redist" \
  --out rgbd_check.png
```

The command prints:

- the depth device name and stream state/rate;
- the depth image size and intrinsics (`fx`, `fy`, `cx`, `cy`);
- the colour image size;
- the fraction of valid (non-zero) depth pixels;
- the median and maximum colour/depth timestamp skew;
- with `--out`, a side-by-side colour + false-colour depth PNG.

If the colour stream opens but the depth stream does not, check the SDK path and
the USB connection. If both open but the depth is mostly zero, the sensor may be
covered or too close.

## Run

```bash
shadowlink run --source rgbd --camera 1 --backend hamer \
  --depth-sdk "D:/sdk/OpenNI2/Redist" \
  --data-dir D:/models/hamer/_DATA
```

- `--camera` selects the **colour** UVC camera (often `1` when the laptop
  webcam is `0`).
- Position control is enabled by default for `--source rgbd`. Disable it with
  `--no-position` to keep the depth camera open but ignore translation.
- `capture.depth_width`, `depth_height`, and `depth_fps` request the depth mode.
  If the device rejects the mode, the default mode is used and the actual
  intrinsics come from the first received frame.
- If the device exposes FOV properties, the intrinsics are derived from them.
  Otherwise the nominal focal length (≈570 px for 640×480) is used. Override
  with `capture.depth_fx/fy/cx/cy` or `--set capture.depth_fx=...`.

## Controls

| Key | Action |
| --- | --- |
| `p` | toggle position follow on/off; re-enabling re-zeroes at the current palm |
| `r` | reset physics, finger targets, phone orientation, and translation |
| `0` | re-zero the phone orientation only |
| `g` | toggle phone orientation follow only |

The HUD adds lines when position control is active:

```text
pos on | tracking | t +0.01 -0.04 +0.00 m
cam +0.08 +0.03 +0.92 m
depth running 30 Hz
```

`tracking` can also be `waiting` (no usable depth yet) or `frozen` (the hand or
depth stream was lost; the last translation is held, not reset to zero).

## Mapping

The mapping is **relative, 1:1, and anatomy-relative**:

```text
delta_cam    = median(camera palm position over a short window) - reference
delta_cam   *= soft deadzone
delta_local  = A @ delta_cam             # camera → palmar / thumb / distal
delta_world  = gain * sign * (R_hand @ delta_local)
translation  = clamp(anchor_world + delta_world, ±max_offset_m)
               then One-Euro filter, then rate limit
```

- The first usable sample (or pressing `p` after a freeze) sets the reference.
- With `position.hand_relative: true` (the default) the 3D landmarks build a
  palm basis: `A` decomposes the camera displacement into the observed hand's
  palmar normal, thumb side, and distal (fingers) direction. `R_hand` then maps
  those local axes to the simulated hand's current base orientation. Moving your
  hand toward your thumb moves the robot toward its thumb, moving toward your
  middle finger moves it toward its middle finger, and displacing the palm along
  its normal moves the robot along its palm normal — whatever the hand's pose,
  side, or image mirroring. This needs valid 3D keypoints from the backend.
- When the 3D landmarks are degenerate or `hand_relative: false`, the static
  `position.preset` described below is used instead. With the default `palm`
  preset, moving toward the camera pushes along world `+z` (the palm normal at
  rest).
- The anchor is the current world translation, so re-zeroing never makes the
  hand jump.
- Translation is applied in **world space** and is independent of the phone
  rotation. Rotation still uses the configured `phone.pivot`; the rotation
  centre becomes `pivot_world + translation`.
- Losing the hand freezes the translation at its last value. It does not return
  to zero.
- The hand is clamped to `position.max_offset_m` (default 0.25 m) from the
  anchor. The passive viewer starts at distance 0.75 m when position control is
  active so the moving hand stays in view.

### Fallback camera-to-world presets

These presets apply only when `position.hand_relative` is false or the backend
cannot provide usable 3D keypoints. Camera axes: `x` right, `y` down, `z` away
from the sensor.

| `position.preset` | Camera → world | Intended mounting |
| --- | --- | --- |
| `palm` (default) | toward the camera → world +z (palm normal), image up → +x (fingers), image right → -y | Palm toward the camera, fingers up |
| `front` | screen right → world +y, screen up → world +z, away → world -x | Sensor faces the operator, upright; screen-aligned rather than palm-aligned |
| `back` | toward the camera → world -z | Back of the hand toward the camera |

If an axis feels reversed, flip it with `position.invert_x`, `invert_y`, or
`invert_z` (for example `--set position.invert_y=true`). `position.gain` scales
the motion; keep it at `1.0` for physical 1:1. The invert flags and gain apply
to the hand-relative mapping as well, so disable `hand_relative` first when
diagnosing a direction problem.

### Tuning the colour/depth alignment

The colour and depth images may differ in scale, crop, and FOV. Two normalised
offsets compensate:

```bash
--set position.align_offset_x=0.05 --set position.align_offset_y=-0.02
```

Move the hand to a known image location, compare the colour overlay with the
depth preview, and shift the sample until the palm window sits on the hand.
Offsets are in fractions of the image width/height and must lie in
`[-0.5, 0.5]`.

## Relevant configuration

```yaml
capture:
  source: rgbd
  camera: "1"
  depth_sdk: null          # or OPENNI2_REDIST / --depth-sdk
  depth_width: 640
  depth_height: 480
  depth_fps: 30
  depth_fx: null           # override the device FOV if needed
  depth_fy: null
  depth_cx: null
  depth_cy: null
position:
  enabled: true
  preset: palm            # fallback when 3D landmarks are unavailable
  hand_relative: true     # follow the observed thumb / fingers / palm normal
  gain: 1.0
  deadzone_m: 0.01
  max_offset_m: 0.25
  min_depth_mm: 150
  max_depth_mm: 1500
  sample_radius_px: 5
  min_valid_pixels: 8
  min_landmarks: 3
  max_spread_m: 0.12
  temporal_window: 3
  max_depth_skew_s: 0.15
  stale_timeout_s: 0.5
  auto_zero: true
  align_offset_x: 0.0
  align_offset_y: 0.0
  min_cutoff: 1.0
  beta: 0.1
  d_cutoff: 1.0
  max_rate_m_s: 0.8
```

`position.max_depth_skew_s` rejects a depth frame that is too far in time from
the colour frame carrying the landmarks. `position.min_landmarks` and
`position.max_spread_m` reject frames where too few palm landmarks have valid
depth, or where the surviving points are inconsistent.

## How the frame association works

1. `RGBDFrameSource.read()` reads a colour frame and attaches the newest depth
   frame (or returns a colour-only packet while depth is warming up).
2. The app stores that `(seq → depth, intrinsics, timestamps, mirror flag)`
   record in a small bounded log.
3. The MediaPipe detector publishes its selected 2D landmarks as soon as
   detection finishes, before HaMeR inference. The app prefers that view for
   position and looks up the depth record for **its own frame `N`**, not "the
   newest depth now". If the detector view is stale, it falls back to the
   finished observation. This keeps translation updates at detector rate
   (~20 Hz) instead of inference rate (~13 Hz).
4. The colour landmarks (`detector_landmarks_2d`, falling back to
   `keypoints_2d`) are unmirrored, normalised, mapped onto the depth image, and
   sampled.
5. The palm position is filtered, decomposed in the observed palm basis
   (`hand_relative`), and converted into the world translation. The anatomical
   axes come from the latest observation with valid 3D keypoints and are
   re-oriented with the current macro pose every frame, so samples that arrive
   between HaMeR results still follow the hand's anatomy.

HaMeR's own `keypoints_2d` are projected with HaMeR's nominal focal length and
may not align with the real colour pixels, so the position layer prefers the
MediaPipe detector landmarks that both backends pass through.

## Failure behaviour

| Condition | Result |
| --- | --- |
| OpenNI2/SDK/device cannot open at startup | `CaptureError`, exit code 2, actionable message |
| Depth thread fails mid-run | HUD shows `depth error`; translation freezes; finger teleop continues |
| Depth/colour skew exceeds `max_depth_skew_s` | frame skipped; translation freezes |
| Too few valid depth samples or too large a spread | frame skipped; translation freezes |
| Degenerate or non-finite 3D keypoints | that sample uses the static `position.preset` fallback |
| Hand absent longer than `stale_timeout_s` | translation freezes at the last value |
| Press `p` while frozen | re-enables tracking and re-zeroes at the next fresh palm sample |
| Press `r` | translation → 0, orientation → start pose, depth log cleared |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `cannot find OpenNI2.dll` | Pass `--depth-sdk` or set `OPENNI2_REDIST`; the error lists every directory tried |
| `the openni package is required` | `pip install "openni>=2.3,<3"` in the active environment |
| Colour stream opens, depth does not | Check the SDK directory and the USB cable/port; try another USB 3 port |
| `rgbd-check` reports almost no valid depth | Hand/sensor too close (<0.6 m), lens covered, or the device is still warming up |
| Translation moves in the wrong direction | Check that 3D landmarks are available (`hand_relative` needs them); then try `--set position.hand_relative=false` with another `--position-preset`, or flip an axis with `--set position.invert_y=true` |
| Translation is noisy or lags | Reduce `position.min_cutoff`/increase smoothing, or lower `position.max_rate_m_s`; keep the hand inside 0.6–1.2 m |
| Hand drifts while still | Re-zero with `p`; the mapping is relative, so a slow sensor bias accumulates otherwise |
| Hand leaves the viewer | Lower `position.max_offset_m` or run with the default 0.75 m viewer distance |
| Colour landmarks sit on the wrong depth pixel | Adjust `position.align_offset_x/y`; confirm the colour camera index with `rgbd-check` |

## Limitations

- No hardware sync or extrinsic calibration between colour and depth; alignment
  offsets are approximate.
- Sample windows median out holes, but thin fingers and reflective/black
  surfaces can still produce invalid depth.
- Translation is limited to `±max_offset_m` from the last zero, not the full
  workspace.
- Depth is not used for finger retargeting or for left/right handedness.
- Only one OpenNI2 `DepthService` can be active per process.
