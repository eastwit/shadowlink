# Whole-hand control: AprilTag + phone

ShadowLink uses a hand-held AprilTag for whole-hand translation and stable
blue-axis yaw. The two noisier tag tilt axes can come from a phone running
[phyphox](https://phyphox.org) on the same hand. Finger perception always comes
from HaMeR or MediaPipe.

The recommended setup uses one camera for everything:

```text
Plugcam/IP Webcam ──► TagService ──► raw frame ──► HaMeR/MediaPipe ──► fingers
                           │
                           ├──────────► AprilTag 3 ──► translation + yaw
                           │
phyphox phone ─────────────┴──────────► yaw + pitch ──► tilt axes
```

The older two-camera layout still works:

```text
built-in camera ──► HaMeR/MediaPipe ──► finger retargeting
Astra RGB ───────► AprilTag 3 ───────► hand base translation + orientation
```

## Hardware and assets

- Tag family: **tag36h11**, id **5**
- Physical size: **57 mm** black-border edge
  (`src/shadowlink/assets/apriltag/size.md`)
- Reference image: `src/shadowlink/assets/apriltag/tag36h11_id5.png`
- Recommended profile: `configs/plugcam_hand_tag.yaml`
  (`capture.source: tag`, DirectShow index `1` for Plugcam on this machine)
- Default two-camera profile: `configs/default.yaml`

| Index | Device | Default role |
| --- | --- | --- |
| 0 | Astra Pro HD Camera | tag camera |
| 1 | HD Webcam / Plugcam | finger camera; also tag camera in single-camera mode |

Print or display the reference image **without mirroring**. UVC and phone
cameras may be horizontally mirrored; `tag.mirror: auto` detects and corrects
that automatically. A mirror-printed tag, a front-camera preview, or a literal
mirror cannot be decoded.

## Run

`pupil-apriltags` is a core dependency:

```bash
pip install -e ".[dev]"
```

Recommended single camera + phone:

```bash
shadowlink run --config configs/plugcam_hand_tag.yaml --backend hamer \
  --data-dir D:/models/hamer/_DATA
```

Two-camera fallback:

```bash
shadowlink run --backend hamer --data-dir D:/models/hamer/_DATA
```

Camera indices are in the YAML profile. Override temporarily:

```bash
shadowlink run --camera 1 --tag-camera 0 --tag-api dshow --backend mediapipe
```

In `capture.source: tag` mode the `capture.camera` value is ignored because the
tag camera is shared. In two-camera mode `tag.camera` must differ from
`capture.camera`; the config validator rejects the same index.

## Detection window

ShadowLink opens a second OpenCV window titled **AprilTag Detection**. It shows
the camera stream with the detected tag outline and X/Y/Z axes, plus ID,
distance, mirror state, and detection rate. A cyan **ZERO center** guide marks
the fixed translation reference; align the tag with it for a neutral hand
offset. It is live even when no tag is visible. Hide it with `--no-tag-show` or
`tag.show: false`; `--no-show` disables both preview windows.

## Controls

| Key | Action |
| --- | --- |
| `g` | toggle whole-hand orientation follow; the latched zero is unchanged |
| `p` | toggle tag translation (the fixed reference is unchanged) |
| `0` | re-zero orientation (and phone tilt) and reset fixed translation |
| `m` | cycle the eight phone mount mappings and re-zero orientation |
| `r` | reset physics, targets, tag zero, translation, orientation, and block |
| `c` / `q` / `Esc` | calibrate fingers / quit |

The first valid tag+phone orientation is captured as an **absolute, latched
orientation zero** for the session, even if orientation follow is off. `g` only
enables/disables following and never changes that zero. With
`tag.auto_follow: true` (default), following starts immediately at that first
sample. The translation reference is the fixed center/1 m point, so hold the tag
near that point for a neutral hand offset. If the tag disappears for
`tag.stale_timeout_s` (0.5 s), translation freezes at its last offset and
orientation is held; the hand does not jump back to zero. Press `0` only if you
explicitly want a new zero.

## Translation mapping

At startup and after `r`, the hand base is set **palm down**, with the middle
finger pointing toward the block (+world x). Translation uses
`position.preset` (`overhead` by default):

```text
tag +x (image right) -> world +x (toward the block)
tag +y (image down)  -> world -y
tag +z (away)        -> world -z  (moving closer -> world +z / up)
t_zero = (0, 0, tag.zero_distance_m)   # image center at 1 m by default
offset = anchor + gain * sign * R_preset @ (t_tag - t_zero)
```

The offset passes through a radial deadzone, One-Euro filter, and slew limiter.
There is **no spatial bound**: the hand can follow the tag arbitrarily far from
the zero. `position.invert_x/y/z` flips world axes.

## Orientation mapping

Tag anatomical axes:

```text
tag +X (red)   -> forward / right-hand middle finger
tag +Y (green) -> right
tag +Z (blue)  -> palm direction
tag -Z         -> simulation up
```

The tag is mounted on the hand with:

```text
tag +X -> hand +Y (pitch)
tag +Y -> hand +Z (roll / middle-finger axis)
tag +Z -> hand +X (palm-normal / yaw axis)
q_mount = [0.5, 0.5, 0.5, 0.5]
```

### Without a phone

The full relative tag rotation drives the hand:

```text
q_rel_tag    = conj(q_tag_zero) ⊗ q_tag_now
q_delta_hand = q_mount ⊗ q_rel_tag ⊗ conj(q_mount)
q_target     = q_hand_at_zero ⊗ q_delta_hand
q_out        = slerp(q_out, q_target, smoothing + rate limit)
```

Rotating the tag about **green** produces hand roll, **red** produces pitch,
and **blue** produces yaw. The detector rejects weak decodes and single-frame
orientation jumps before this mapping.

### With a phyphox phone (`phone.source: phyphox`)

The tag contributes only its blue-axis twist:

```text
q_rel_tag = conj(q_tag_zero) ⊗ q_tag_now
q_tag_yaw = twist(q_rel_tag, tag +Z)
```

The phone contributes yaw+pitch, and its roll channel (rotation about the phone
screen normal, user-frame +Z) is discarded with a swing-twist split. The phone
mount maps phone yaw/pitch onto the hand's two non-yaw axes:

```text
q_phone_tilt = swing(conj(q_phone_zero) ⊗ q_phone_now, phone +Z)
q_target     = q_hand_at_zero ⊗ q_tag_yaw_hand ⊗ q_phone_tilt_hand
```

`phone.mount_preset: yaw_pitch` is the default; the shipped direction
combination is `swap-d`. If the axes are swapped or a direction is reversed,
press **`m`** to cycle all eight mount rotations. Each switch re-zeroes the
orientation, and the HUD shows `phone ... mount <name>`. Set
`phone.mount_preset: custom` plus `mount_w/x/y/z` to persist a custom mapping.

Phone and tag zero are latched together. If the phone is stale, orientation
freezes while tag translation continues. See [PHONE.md](PHONE.md) for phyphox
setup.

Smoothing, deadzone, gain, angle clamp, jump rejection, and rate limiting apply
to the combined target. `tag.pivot` selects the mocap rotation centre (`wrist`
by default).

## Camera settings

| Key | Default | Notes |
| --- | --- | --- |
| `capture.source` | `camera` | `tag` shares the tag camera with finger perception |
| `tag.camera` | `"0"` | Astra in the default profile; `"1"` for Plugcam |
| `tag.api` | `dshow` | DSHOW opens reliably. MSMF may be slower to open |
| `tag.width/height` | `640x480` | `1920x1080` + `quad_decimate: 2.0` in the Plugcam profile |
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
| `tag.quad_decimate` | `1.0` | use `2.0` for 1080p phone streams |
| `tag.max_hamming` | `1` | reject decodes with more corrected bits |
| `tag.min_decision_margin` | `10` | reject weak/false detections |
| `tag.rotation_jump_reject_deg` | `35` | reject a single-frame orientation jump |
| `tag.rotation_jump_confirm_frames` | `5` | accept a jump only after it persists |

Pose scale uses `tag.size_m` directly. If the tag was printed with 57 mm as the
whole sheet including the quiet zone, measure the black square and use that
value instead (the reference image's black square is 80% of the sheet).

## Troubleshooting

- **The hand jumps occasionally:** the detector rejects weak detections and
  single-frame flips, but reflections or another same-ID tag can confuse it.
  Keep the tag away from monitors and check the HUD for `tag rot | rejecting`.
  Lower `tag.rotation_max_rate_rad_s` or raise
  `tag.rotation_jump_confirm_frames`.
- **HUD says `tag error`:** the camera index is busy/wrong, or
  `pupil-apriltags` is missing. Close other apps using the camera.
- **HUD says `searching`:** check family/id, black-square size, lighting, focus,
  and `tag.mirror`. Hold the tag roughly 0.5–1.2 m away.
- **Tag found but the hand does not move:** check `tag rot/pos on`; press
  `g`/`p`, or `0` to re-zero.
- **Translation directions reversed:** use `position.invert_x/y/z`.
- **Phone axes swapped or reversed:** press `m` until correct, then tell the
  project which mount name works.
- **Phone HUD says `connecting` / `stale`:** verify phyphox Remote Access with
  `curl http://PHONE:8080/config` on the same Wi-Fi. See
  [PHONE.md](PHONE.md).
