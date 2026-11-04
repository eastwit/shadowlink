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
app.py (TeleopApp) ─────────────────────────────────┐
   │            │            │           │          │
   ▼            ▼            ▼           ▼          ▼
capture.py   perception/   retarget/   phone/    sim/ + ui/
             │            │           │          │
             ▼            ▼           ▼          ▼
          types.py     geometry.py  client.py   model.py
                       mapper.py    attitude.py shadow_hand.py
                       smoothing.py service.py  stepping.py
                       calibration.py           viewer.py
                       macro_pose.py
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

`retarget/macro_pose.py` converts a sample into a hand-base quaternion:

```text
q_rel    = conj(q_zero) ⊗ q_now             (rotation since the toggle)
angle    = 0 if angle < deadzone else min(angle * gain, max_angle)
q_hand   = q_mount ⊗ q_rel ⊗ conj(q_mount)  (phone body → hand base)
q_target = q_ref_hand ⊗ q_hand              (hand orientation at toggle)
q_out    = slerp(q_out, q_target, alpha)    (time constant + rate limit)
```

`max_angle` defaults to 180°, so the clamp is inactive and the full relative
rotation passes through. `q_mount` comes from `phone.mount_preset`: `back`
(default, phone strapped to the back of the hand), `in_hand` (phone held flat in
the palm with the screen facing away from the palm), or `custom` (the
`mount_w/x/y/z` fields). The two presets differ by a 180° rotation about the
phone's long axis, so only the width and screen-normal axes change sign between
them.

`q_zero` and `q_ref_hand` are captured together when follow is enabled, so
freezing and resuming does not snap the hand back. When the sample is missing,
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
key re-zeros the relative mapping without toggling, using the current phone
attitude and the current hand orientation. These actions do not modify the
finger retargeter or the `TargetController` state machine.

`MacroPoseController.zero_hand` exposes the hand orientation captured with the
zero (`q_ref_hand`, or the initial orientation before following starts). Each
frame the app passes it and `ShadowHand.macro_pivot_world` to
`Viewer.set_macro_zero`. While synced, the viewer writes three `mjv_connector`
arrows into `user_scn`: red for hand-base `+x`, green for `+y`, blue for `+z`,
offset along the palmar axis so the triad is visible outside the mesh. Clearing
the zero also clears those user geoms. This is display-only and does not touch
the physics state.

## Simulation

`sim/model.py` loads the scene with `mujoco.MjModel.from_xml_path`, validates
`nq=nv=24`, `nu=20`, `nsensor=120`, `timestep=0.002`, and creates an
`ActuatorLayout` from `actuator_trntype`, `actuator_trnid`, and
`actuator_ctrlrange`. It also loads the 13 keyframes and converts a keyframe
`qpos` to actuator targets through `data.actuator_length`, so fixed tendons are
handled without hard-coded sums.

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
- OpenCV HighGUI calls and MuJoCo viewer calls happen only on the main thread.
- Cleanup order is capture → perception → phone → viewer → preview → recorder.
- The perception and phone services join their threads with a timeout and log
  any that fail to stop.

## Extension points

- New backend: implement `detect`/`estimate` (see `perception/base.py`) and
  register it in `create_backend`; keep the `HandObservation` contract.
- New detector: return `HandDetection` objects; the worker handles selection and
  filtering.
- New mapping: change `Retargeter.map_metrics`; semantic names keep the XML
  coupling explicit.
- New camera transport: implement the `FrameSource` protocol in `capture.py`.
- New orientation source: implement `start`, `close`, `latest`, and `health`
  (see the `PhoneSource` protocol in `phone/service.py`); the macro layer only
  needs `quat_wxyz`, `timestamp`, and `ready`.
