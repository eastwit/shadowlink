# Architecture

ShadowLink is a single installable package. Its job is narrow: turn one RGB
frame into a 21-joint hand observation, turn that observation into 20 actuator
targets, and advance a MuJoCo model in real time. Everything else exists to make
that loop testable, configurable, and safe to publish.

## Layers

```text
cli.py / __main__.py
        │
        ▼
app.py (TeleopApp) ──────────────────────────────────────────┐
   │            │            │           │          │        │
   ▼            ▼            ▼           ▼          ▼        ▼
capture.py   perception/   retarget/   phone/    position/  sim/ + ui/
   │            │            │           │          │        │
   ▼            ▼            ▼           ▼          ▼        ▼
rgbd/        types.py     geometry.py  client.py  sampling  model.py
depth.py     │            mapper.py    attitude.py controller shadow_hand.py
source.py    │            smoothing.py service.py  association
             │            calibration.py           anatomy.py
             │            macro_pose.py           types.py
```

Dependency rules, enforced by review and tests:

- `types.py` depends only on NumPy (plus the standard library).
- `retarget/` never imports `sim/`; it receives an `ActuatorLayout` through a
  `TYPE_CHECKING`-only import.
- `sim/` never imports `perception/`, `retarget/`, or `phone/`.
- `phone/` never imports `sim/` or `retarget/`; it publishes an immutable
  orientation sample and knows nothing about MuJoCo. It uses only the standard
  library, NumPy, and `math3d.py`.
- `retarget/macro_pose.py` depends only on `math3d.py` and a small sample
  protocol; it never imports `phone/`.
- `position/` depends on `types.py`, `config.py`, the pure geometry helpers in
  `retarget/geometry.py`, and the filters in `retarget/smoothing.py`; it never
  imports `rgbd/`, `sim/`, or `perception/`.
- `rgbd/` depends on `capture.py` for the colour source and `types.py` for the
  depth records. `openni` is imported lazily inside `rgbd/depth.py`; the package
  never imports `sim/` or `position/`.
- `math3d.py` depends only on NumPy and the standard library.
- Only `app.py` orchestrates the layers.
- Only `cli.py` uses `argparse`.
- `hamer` is imported lazily inside `perception/hamer_backend.py`; the phyphox
  transport has no third-party dependency.
- No module calls `os.chdir` or edits `sys.path`.

## Frame path

1. **Capture.** `OpenCVFrameSource` opens the camera or video, applies the
   mirror exactly once, and returns a frozen `FramePacket` with a monotonically
   increasing `seq` and a `time.monotonic()` capture timestamp. The image is
   marked read-only. `SyntheticFrameSource` provides deterministic frames for
   tests.
2. **Submit.** The app submits every `perception.skip`-th frame to
   `PerceptionService`.
3. **Detect.** A detector thread runs MediaPipe, filters by anatomical side, and
   selects one hand (`largest`, `center`, or IoU `track`). If no hand is found it
   publishes an empty result immediately.
4. **Infer.** In `two_stage` mode the selected crop goes to a second latest-slot
   mailbox. The inference thread runs HaMeR and publishes a `PerceptionResult`.
   In `single` mode the detector thread does both stages.
5. **Consume.** The main loop reads the newest result, rejects observations older
   than `perception.max_age_s`, retargets, filters, rate-limits, writes `ctrl`,
   advances physics by wall-clock time, updates touch colours, and draws.

Each stage keeps only the newest item. Slow hardware therefore reduces the
number of processed frames, not the freshness of the control signal. Dropped
frames are counted in `WorkerHealth`.

## Why two stages

MediaPipe detection and HaMeR inference are independent operations on the same
frame. Running them in separate threads lets frame *N*'s inference overlap frame
*N+1*'s detection. The physics loop is unaffected: it never blocks on either
thread.

`frame_seq` on every detection, observation, and result is the identity used to
match the two stages. `capture_ts` and `finished_ts` provide the age and latency
shown in the HUD. There is no shared mutable result object.

## Failure behaviour

Exceptions in either worker are caught at the loop boundary, recorded in
`WorkerHealth(state="error")`, logged with a traceback, and stop the service.
The app observes `health.failed` on its next iteration, logs the message, and
exits with code 1. This is deliberate: continuing to drive the robot from a
stale observation is worse than stopping.

Invalid geometry (NaN keypoints, degenerate palm basis) raises
`HandGeometryError` for that frame only; the app logs it at debug level, skips
the target, and lets the controller hold or return to neutral.

## Retargeting

`retarget/geometry.py` canonicalises the observation into a right-hand frame,
builds an orthonormal palm basis `(forward, side, normal)`, and computes:

- MCP, PIP, and DIP flexion per finger;
- in-palm spread angles for index/middle/ring/pinky;
- thumb opposition relative to the index finger.

All angular differences are wrapped to `[-π, π]`. A degenerate basis or
non-finite input is rejected rather than clipped.

`retarget/mapper.py` converts those metrics into semantic actuator values:
proximal flexion → `*J3`, PIP+DIP → the `*J0` tendon, spread → `*J4` (with
per-finger sign), and thumb base/flexion/opposition → `THJ4/THJ2/THJ1/THJ5`.
`ActuatorLayout.vector()` looks the values up by semantic name and clips them to
the `ctrlrange` read from the MJCF. Reordering actuators in the XML therefore
cannot silently swap fingers.

## Smoothing and target state

`TargetController` owns:

- a One-Euro filter per actuator (or exponential/none, configurable);
- a slew limiter (`max_rate_rad_s`);
- a state machine: `waiting` → `tracking` → `lost`;
- a hold window after tracking is lost, followed by a rate-limited return to the
  neutral target (`return_rate_rad_s`).

Calibration subtracts the neutral metrics before mapping, so an open hand maps
to the neutral actuator vector instead of a hard-coded zero.

## Phone and macro pose

The optional phone path is deliberately separate from finger control.

`phone/client.py` is a small outbound-only HTTP client for the third-party
phyphox app. It reads `/config` for buffer names, `/get?name=...` for the newest
values, and `/control?cmd=start` when a paused experiment should be resumed.
Every request opens a fresh connection because phyphox's server closes the
socket after each reply. Responses are capped at 1 MiB and must be finite JSON
objects. `pick_attitude` finds a complete `w/x/y/z` quaternion group (preferring
one with a time axis); `pick_vector` finds gyroscope/accelerometer/magnetometer
axes when the experiment has no fused attitude.

`phone/attitude.py` is the fallback for raw-IMU experiments: the Mahony filter
integrates gyroscope readings with gravity and optional magnetometer correction.
The primary path uses phyphox's fused rotation vector directly and does not
depend on host-side timing.

`phone/service.py` owns exactly one daemon thread. It polls the configured host
at `poll_hz`, applies a fixed phone-body twist, publishes an immutable
`PhoneSample` into a latest-value slot, re-reads `/config` when the experiment
session changes, and maintains a `PhoneHealth` record. It only discovers hosts
when `auto_discover` is explicitly enabled. It never touches MuJoCo.

`retarget/macro_pose.py` converts a sample into a hand-base quaternion. The
sample attitude at the zero instant defines the world reference; every later
sample is applied relative to it. The fixed mount encodes the anatomical
correspondence between the phone and the hand (screen normal → dorsum, phone
top → middle finger, phone left → hand left):

```text
q_rel    = conj(q_zero) ⊗ q_now             (rotation since the zero)
angle    = 0 if angle < deadzone else min(angle * gain, max_angle)
q_hand   = q_mount ⊗ q_rel ⊗ conj(q_mount)  (phone body → hand base)
q_ref    = q_zero ⊗ conj(q_mount)           (hand pose aligned at zero)
q_target = q_ref ⊗ q_hand = q_now ⊗ conj(q_mount)
q_out    = slerp(q_out, q_target, alpha)    (time constant + rate limit)
```

The result makes the hand's anatomical frame identical to the phone's
anatomical frame at all times, and at the zero instant the hand starts in the
phone's real attitude rather than keeping an arbitrary offset. `max_angle`
defaults to 180°, so the clamp is inactive and the full relative rotation
passes through. `q_mount` comes from `phone.mount_preset`: `back` (default,
phone strapped to the back of the hand), `in_hand` (phone held flat in the
palm with the screen facing away from the palm), or `custom` (the
`mount_w/x/y/z` fields). The two presets differ by a 180° rotation about the
phone's long axis, so only the width and screen-normal axes change sign between
them.

`q_zero` is captured when follow is enabled; the hand reference is derived
from it (`q_zero ⊗ conj(q_mount)`), so enabling follow re-aligns the hand to
the phone instead of preserving its previous pose. When the sample is missing,
not ready, or stale, `update` returns `None` and the base orientation is frozen.

The app writes the result to `ShadowHand.set_macro_pose`, which sets the
`rh_forearm` mocap body. The rotation centre is selected by `phone.pivot` and
captured at rest: `wrist` (default) holds the `rh_wrist` joint fixed, `forearm`
holds the proximal end of the forearm mesh fixed, and `palm` holds the old
palm/finger centroid fixed. The mocap position compensates so the chosen world
point does not move. This changes `nmocap` to 1 but leaves `nq`, `nv`, `nu`, the
keyframes, and the 120 tactile sensors unchanged. If a frame advances zero
physics steps, the app calls `mj_forward` once so the viewer reflects the new
base orientation.

Because the base is kinematic, the hand does not receive reaction forces. The
current scene has no grasped objects; adding one would require a free joint or a
force-aware attachment.

The `g` toggle and `--phone-follow` only enable or disable this layer. The `0`
key re-anchors the world frame at the current phone attitude without toggling,
so the hand is aligned to the phone again. These actions do not modify the
finger retargeter or the `TargetController` state machine.

`MacroPoseController.zero_hand` exposes the aligned hand orientation for the
current zero (`q_zero ⊗ conj(q_mount)`, or the initial orientation before
following starts). Each
frame the app passes it and `ShadowHand.macro_origin_world` (the pivot plus the
RGBD translation) to `Viewer.set_macro_zero`. While synced, the viewer writes three `mjv_connector`
arrows into `user_scn`: red for hand-base `+x`, green for `+y`, blue for `+z`,
offset along the palmar axis so the triad is visible outside the mesh. Clearing
the zero also clears those user geoms. This is display-only and does not touch
the physics state.

## RGBD position

The RGBD path adds whole-hand translation without changing perception or
retargeting. It is enabled only when `capture.source == "rgbd"` and
`position.enabled` is true.

`rgbd/depth.py` owns the OpenNI2 lifecycle. `DepthService.start()` runs on the
main thread: it resolves a directory containing `OpenNI2.dll`, registers it with
`os.add_dll_directory`, initialises OpenNI2, opens the first device, creates a
depth stream, requests the configured mode (falling back to the default mode if
the device rejects it), and starts one daemon thread named `shadowlink-depth`.
The thread waits for a frame with `wait_for_any_stream(..., timeout=0.1)` and
only then calls `read_frame()`, so the stop event is observed within one poll
interval even on builds that do not wake a blocked read. It discards a few
warm-up frames, copies the buffer into a read-only `uint16` array, computes the
intrinsics from the first real frame (device FOV, nominal 570.3 px fallback),
and publishes a `DepthFrame` into a latest-only slot together with
`DepthHealth`. `close()` sets the stop event and joins; stopping the stream is
only a fallback if the thread is still alive, because stopping from another
thread can race with `read_frame`. If the thread still does not exit, the DLL is
deliberately not unloaded; otherwise the stream, device, and DLL directory
handle are released.
OpenNI2 is process-global, so only one `DepthService` may be active.

`rgbd/source.py` wraps a colour source and the depth service. Each `read()`
attaches the newest available depth snapshot to the colour `FramePacket`
(`depth_mm`, `depth_intrinsics`, `depth_ts`). Before the first depth frame it
returns a colour-only packet. Depth startup failure becomes a `CaptureError`,
which the CLI reports with exit code 2.

`position/association.py` stores a short bounded history keyed by colour frame
sequence. The app adds every frame and looks up the record for the sampled
frame `N`. This pairs the landmarks with the depth image from their own capture
instead of "the newest depth now". The app also rejects a pair whose
colour/depth timestamp skew exceeds `position.max_depth_skew_s`.

`PerceptionService.latest_detection()` publishes the detector's selected
landmarks before HaMeR inference finishes. The app prefers that view for palm
sampling (falling back to the finished observation when it is stale), so
translation updates at detector rate rather than inference rate. The
hand-relative anatomical axes are cached from the latest valid 3D observation
and re-oriented with the current mocap quaternion on every iteration.

`position/depth_sampling.py` maps the palm landmarks (`0, 5, 9, 13, 17`) from
the processed colour image back through the mirror, normalises them, and scales
to the depth resolution. Each landmark takes a `(2r+1)²` window; only pixels in
the configured depth range count, and the window depth is the median. Points are
back-projected with

```text
Z = depth_mm / 1000
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

The surviving 3D points are component-wise medianed. Fewer than
`min_landmarks` or a maximum deviation above `max_spread_m` rejects the frame;
background depth is never substituted for a missing sample.

`position/controller.py` converts camera-space motion into a world translation:

```text
delta_cam   = median(camera samples) - reference
delta_cam  *= soft deadzone
delta_local = A @ delta_cam
delta_world = gain * sign * (R_hand @ delta_local)
translation = clip(anchor_world + delta_world, ±max_offset_m)
              → One-Euro filter → slew limit → clip
```

The first usable sample (when `auto_zero` is true) or `zero()` sets the
reference. `zero()` rebases the reference and the anchor so the output does not
jump; pressing `p` while a fresh sample exists takes effect immediately,
otherwise it is deferred to the next update. A missing or stale measurement, or
a rejected depth frame, freezes the output. It never returns to zero on its own.

`position/anatomy.py` builds `A` and `R_hand` from the observation.
`camera_to_local_axes()` canonicalises `HandObservation.keypoints_3d` into a
right-hand frame, builds the palm basis `(normal, -side, forward)` (palmar,
thumb, distal), and mirrors the camera x axis for an anatomical left hand
because the depth displacement is measured on the raw image. `hand_base_rotation()`
then maps those local axes into world space using the current mocap
quaternion. The controller accepts this rotation per sample, so moving toward
the physical thumb drives the simulated thumb in any hand pose. If the
keypoints are degenerate, or `position.hand_relative` is false, the controller
falls back to the static `position.preset`: `palm` maps motion toward the
camera to world `+z` (the palm normal at rest), `front` is screen-aligned, and
`back` is for a hand shown dorsal-side to the camera.

`ShadowHand.set_macro_pose(quat, translation)` writes both values in one mocap
update:

```text
mocap_pos = pivot_world + translation - R(quat) @ pivot_local
```

`quat=None` keeps the current base orientation and `translation=None` keeps the
stored translation. The app therefore calls it once per frame with the phone
quaternion (or `None` when orientation is not following) and the position
controller output. Rotation and translation are independent. The viewer zero
triad is drawn at `macro_origin_world = pivot_world + translation`.

## Simulation

`sim/model.py` loads the scene with `mujoco.MjModel.from_xml_path`. The hand
itself has 24 DOFs (`nq=nv=24`); the scene adds one free joint for the grasp
ball, so the compiled model has `nq=31`, `nv=30`, `nu=20`, `nsensor=120`, and
`timestep=0.002`. It creates an `ActuatorLayout` from `actuator_trntype`,
`actuator_trnid`, and `actuator_ctrlrange`. It also loads the 13 hand keyframes
and converts a keyframe `qpos` to actuator targets through
`data.actuator_length`, so fixed tendons are handled without hard-coded sums;
keyframes store only the hand DOFs and are padded with `qpos0` for the ball.

The scene includes an orange ball on a thin stand at palm height. It is a fully
dynamic free body (0.05 kg) with contact friction, so closing the retargeted
fingers around it produces a real grasp and the tactile sites light up. The
stand, floor, and ball collide; the hand base stays kinematic. Pressing `r`
restores the ball to its stand through `ShadowHand.reset_free_bodies`.

`sim/stepping.py` accumulates wall-clock time and advances at most
`max_catchup_s` of simulation per rendered frame. A stall longer than
`reset_stall_s` resets the accumulator instead of trying to catch up after a
debugger pause.

`sim/shadow_hand.py` wraps actuator writes and reads the 120 touch sensors into
five `(6, 4)` maps. Colours are applied to the site RGBA in place; the viewer
reads them on the next sync.

## Configuration

`config.py` defines frozen dataclasses. `AppConfig.from_yaml` parses YAML,
resolves relative paths against the YAML file's directory, and rejects unknown
keys. `apply_overrides` accepts exactly one dotted key per `--set`. Explicit CLI
flags are applied after YAML, and validation runs once at the end.

`configs/default.yaml` is generated to match the dataclass defaults; a test
fails if they diverge.

## Threading and UI rules

- Perception threads only touch immutable data and their own model/session.
- The phone service is a single thread; it owns HTTP access and the fusion
  state, and publishes only immutable samples.
- The depth service is a single daemon thread that only calls `read_frame()`
  and publishes immutable depth snapshots. OpenNI initialisation and teardown
  happen on the main thread.
- OpenCV HighGUI calls and MuJoCo viewer calls happen only on the main thread.
- Cleanup order is capture (depth, then colour) → perception → phone → viewer →
  preview → recorder.
- The perception, phone, and depth services join their threads with a timeout
  and log any that fail to stop.

## Extension points

- New backend: implement `detect`/`estimate` (see `perception/base.py`) and
  register it in `create_backend`; keep the `HandObservation` contract.
- New detector: return `HandDetection` objects; the worker handles selection and
  filtering.
- New mapping: change `Retargeter.map_metrics`; semantic names keep the XML
  coupling explicit.
- New camera transport: implement the `FrameSource` protocol in `capture.py`.
  A source that attaches `depth_mm`, `depth_intrinsics`, and `depth_ts` to its
  packets participates in position control without further changes.
- New depth provider: implement `start`, `close`, `latest`, `health`,
  `device_name`, and `intrinsics` and pass it to `RGBDFrameSource`.
- New position mapping: extend `position/controller.py`; sampling and the app
  integration stay unchanged.
- New orientation source: implement `start`, `close`, `latest`, and `health`
  (see the `PhoneSource` protocol in `phone/service.py`); the macro layer only
  needs `quat_wxyz`, `timestamp`, and `ready`.
