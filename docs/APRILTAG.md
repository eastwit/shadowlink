# AprilTag 6-DoF whole-hand control

ShadowLink uses a hand-held AprilTag as the only whole-hand controller. The
built-in webcam keeps driving finger perception; a second camera (an Orbbec
Astra on this machine) watches the tag.

```
built-in camera ──► HaMeR/MediaPipe ──► finger retargeting
Astra RGB ───────► AprilTag 3 ───────► hand base translation + orientation
```

The tag is independent of the hand. Translation uses a fixed reference by
default: **image center at 1.00 m depth** (`tag.zero_mode: fixed`,
`tag.zero_distance_m: 1.0`). Orientation zero is the first detected tag pose, so
a comfortable starting orientation never snaps the hand toward the camera or the
desk. Set `tag.zero_mode: sample` to use the first tag position as the
translation reference instead.

## Hardware and assets

- Tag family: **tag36h11**, id **5**
- Physical size: **57 mm** black-border edge
  (`src/shadowlink/assets/apriltag/size.md`)
- Reference image: `src/shadowlink/assets/apriltag/tag36h11_id5.png`
- Cameras on this machine (DirectShow indices):

| Index | Device | Role |
| --- | --- | --- |
| 0 | Astra Pro HD Camera | tag camera |
| 1 | HD Webcam (built-in) | hand/finger perception |

Print or display the reference image **without mirroring**. The Astra RGB stream
itself is horizontally mirrored; `tag.mirror: auto` detects and corrects that
automatically. A mirror-printed tag, a front-camera preview, or a literal mirror
cannot be decoded.

## Run

`pupil-apriltags` is a core dependency now:

```bash
pip install -e ".[dev]"
shadowlink run --backend mediapipe
```

Use `--backend hamer` (and the HaMeR model files) for GPU finger reconstruction.
The tag path is identical; only finger perception changes.

Camera indices are in `configs/default.yaml`. Override temporarily:

```bash
shadowlink run --camera 1 --tag-camera 0 --tag-api dshow --backend mediapipe
```

The tag camera must differ from `capture.camera`; the config validator rejects a
shared index.

## Detection window

ShadowLink opens a second OpenCV window titled **AprilTag Detection**. It shows
the Astra colour stream with the detected tag outline and X/Y/Z axes, plus ID,
distance, mirror state, and detection rate. A cyan **ZERO center 1.00 m** guide
marks the fixed translation reference; align the tag with it for neutral hand
offset. It is live even when no tag is visible, so you can use it to aim the
camera. Hide it with `--no-tag-show` or
`tag.show: false`; `--no-show` disables both preview windows.

## Controls

| Key | Action |
| --- | --- |
| `g` | toggle tag orientation follow; the first-valid-tag zero is latched and unchanged |
| `p` | toggle tag translation (the fixed reference is unchanged) |
| `0` | explicitly re-zero orientation at the current tag pose and reset fixed translation to zero |
| `r` | reset physics, target, tag zero, translation, orientation, and the cylinder |
| `c` / `q` / `Esc` | calibrate fingers / quit |

The first valid tag orientation is captured as an **absolute, latched
orientation zero** for the session, even if orientation follow is off. `g` only
enables/disables following and never changes that zero. With
`tag.auto_follow: true` (default), following starts immediately at that first
sample. The translation reference is already the fixed center/1 m point, so hold
the tag near that point for a neutral hand offset. If the tag disappears for
`tag.stale_timeout_s` (0.5 s), translation freezes at its last offset and
orientation is held; the hand does not jump back to zero. Press `0` only if you
explicitly want a new zero (and reset translation).

## Mapping

At startup and after `r`, the hand base is set **palm down**, with the middle
finger pointing toward the cylinder (+world x). Translation uses
`position.preset` (`overhead` by default):

```text
tag +x (image right) -> world +x (toward the cylinder)
tag +y (image down)  -> world -y
tag +z (away)        -> world -z  (moving closer -> world +z / up)
t_zero = (0, 0, tag.zero_distance_m)   # image center at 1 m by default
offset = anchor + gain * sign * R_preset @ (t_tag - t_zero)
```

The offset passes through a radial deadzone, One-Euro filter, and slew limiter.
There is **no spatial bound**: the hand can follow the tag arbitrarily far from
the zero. `position.invert_x/y/z` flips world axes.

Orientation uses the tag's own anatomical axes. The tag frame is interpreted as:

```text
tag +X -> forward / right-hand middle finger
tag +Y -> right
tag +Z -> palm direction
tag -Z -> simulation up
```

Those axes are mapped to the Shadow Hand local frame:

```text
tag +X (red)   -> hand +Y (pitch axis)
tag +Y (green) -> hand +Z (roll axis / middle-finger direction)
tag +Z (blue)  -> hand +X (palm-normal / yaw axis)
q_mount = [0, 0.7071068, 0, 0.7071068]
```

The relative rotation is applied in that hand-local frame:

```text
q_rel_tag    = conj(q_tag_zero) ⊗ q_tag_now
q_delta_hand = q_mount ⊗ q_rel_tag ⊗ conj(q_mount)
q_target     = q_hand_at_zero ⊗ q_delta_hand
q_out        = slerp(q_out, q_target, smoothing + rate limit)
```

Rotating the tag about **green** produces hand roll about the middle-finger
axis, **red** produces pitch, and **blue** produces yaw about the palm normal.
This cyclic mapping keeps all three axes independent and leaves yaw on blue.
`g` captures a new orientation zero. `tag.rotation_gain` scales the angle,
`tag.rotation_deadzone_deg` removes small jitter,
`tag.rotation_max_angle_deg` optionally clamps it, and
`tag.rotation_invert_x/y/z` flips a tag axis if the physical mounting makes a
rotation feel reversed. The result drives the forearm mocap body;
`tag.pivot` selects the rotation centre (`wrist` by default).

## Camera settings

| Key | Default | Notes |
| --- | --- | --- |
| `tag.camera` | `"0"` | Astra colour camera |
| `tag.api` | `dshow` | DSHOW opens reliably. MSMF can reach ~30 FPS but sometimes takes tens of seconds to open on Astra |
| `tag.width/height` | `640x480` | lower latency and ~20 FPS |
| `tag.fps` | `30` | requested capture rate |
| `tag.mirror` | `auto` | `auto`, `on`, or `off` |
| `tag.show` | `true` | show the AprilTag detection window (`--no-tag-show`) |
| `tag.zero_mode` | `fixed` | `fixed` = image center at depth; `sample` = first tag position |
| `tag.zero_distance_m` | `1.0` | fixed zero depth in metres |
| `tag.hfov_deg` | `60` | approximate intrinsics when no calibration is supplied |
| `tag.fx/fy/cx/cy` | `null` | optional calibrated pinhole intrinsics; all four or all null |
| `tag.family` | `tag36h11` | AprilTag family |
| `tag.tag_id` | `5` | expected id; null accepts any |
| `tag.size_m` | `0.057` | black-border edge length in metres |
| `tag.quad_decimate` | `1.0` | keep 1.0 for small/far tags |
| `tag.auto_follow` | `true` | zero and follow at the first ready sample |

Pose scale uses `tag.size_m` directly. If the tag was printed with 57 mm as the
whole sheet including the quiet zone, measure the black square and use that
value instead (the reference image's black square is 80% of the sheet). The
default assumes 57 mm is the black-border edge.

## Troubleshooting

- **`tag` HUD says `error`:** camera index is busy or wrong, or
  `pupil-apriltags` is missing. Close other apps using the Astra.
- **HUD says `searching`, no tag:** check `tag.family`/`tag.tag_id`, the black
  square size in pixels, lighting, focus, and that `tag.mirror` is `auto`/`on`.
  The Astra RGB camera is fixed-focus; hold the tag roughly 0.5–1.2 m away,
  near the cyan center/1 m zero guide.
- **Tag found but hand does not move:** check `tag rot/pos on` in the HUD; press
  `g`/`p`, or `0` to re-zero. The tag must be freshly detected.
- **Motion directions reversed:** use `position.invert_x/y/z` for translation and
  `tag.rotation_invert_x/y/z` for rotation.
- **No 30 FPS:** DSHOW on Astra is ~20 FPS at 640×480 and ~10 FPS at 720p.
  `--tag-api msmf` can reach 30 FPS but opening the device may be slow. Lower
  `tag.quad_decimate` for detection robustness, not for frame rate.

This path does not require camera calibration for orientation; translation is
relative and bounded, so an approximate FOV only changes metric scale, which
`position.gain` can trim.
