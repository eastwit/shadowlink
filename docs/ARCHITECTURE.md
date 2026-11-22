# Architecture

ShadowLink turns one RGB frame into a 21-joint hand observation, maps it into 20
actuator targets, and advances a MuJoCo model in real time. A hand-held AprilTag
drives the whole-hand base pose. Everything else exists to make that loop
testable, configurable, and publishable.

## Layers

```text
cli.py / __main__.py
        │
        ▼
app.py (TeleopApp) ──────────────────────────────────────────────┐
   │            │            │          │          │             │
   ▼            ▼            ▼          ▼          ▼             ▼
capture.py   perception/   retarget/  tag/       sim/ + ui/
   │            │            │          │          │
   ▼            ▼            ▼          ▼          ▼
types.py     types.py     geometry.py service.py model.py
             detector.py  mapper.py   controller.py shadow_hand.py
             worker.py    smoothing.py types.py
                          calibration.py
```

Dependency rules:

- `types.py` depends only on NumPy and the standard library.
- `retarget/` never imports `sim/`; it receives an `ActuatorLayout` through a
  `TYPE_CHECKING`-only import.
- `sim/` never imports `perception/`, `retarget/`, or `tag/`.
- `tag/` owns its camera thread and publishes immutable `TagSample` objects.
  `pupil-apriltags` is imported lazily inside `tag/service.py`. The package
  never imports `sim/` or `perception/`; its controller depends on `config.py`,
  `math3d.py`, and the shared filters in `retarget/smoothing.py`.
- `math3d.py` depends only on NumPy and the standard library.
- Only `app.py` orchestrates the layers.
- Only `cli.py` uses `argparse`.
- `hamer` is imported lazily inside `perception/hamer_backend.py`.
- No module calls `os.chdir` or edits `sys.path`.

## Frame path

1. **Capture.** `OpenCVFrameSource` opens a camera or video, applies mirroring
   exactly once, and returns a frozen `FramePacket` with a monotonically
   increasing `seq` and a `time.monotonic()` timestamp. With
   `capture.source: tag`, `TagFrameSource` instead consumes the raw frames that
   `TagService` already captured, so one camera feeds both hand perception and
   tag detection without a second `VideoCapture`. `SyntheticFrameSource`
   provides deterministic frames for tests.
2. **Submit.** The app submits every `perception.skip`-th frame to
   `PerceptionService`.
3. **Detect.** A detector thread runs MediaPipe, filters by anatomical side, and
   selects one hand (`largest`, `center`, or IoU `track`).
4. **Infer.** In `two_stage` mode the selected crop goes to a second latest-slot
   mailbox. The inference thread runs HaMeR and publishes a `PerceptionResult`.
5. **Consume.** The main loop reads the newest result, rejects observations older
   than `perception.max_age_s`, retargets, filters, writes `ctrl`, advances
   physics by wall-clock time, updates touch colours, and draws.

Each stage keeps only the newest item, so slow hardware reduces processed frames
rather than freshness.

## Retargeting

`retarget/geometry.py` canonicalises the observation into a right-hand frame and
builds a palm basis. It computes MCP/PIP/DIP flexion, spread, and thumb
opposition. `retarget/mapper.py` maps those metrics to `*J3`, `*J0`, `*J4`, and
the thumb actuators by semantic name; values are clipped to the MJCF ranges.

`TargetController` owns per-actuator One-Euro filters, a slew limiter, and the
`waiting → tracking → lost` state machine with a rate-limited return to neutral.
The finger neutral pose is calibrated per backend and side.

## AprilTag 6-DoF

`tag/service.py` owns one daemon thread named `shadowlink-tag`. It opens a UVC
camera, detects one AprilTag pose per frame through `pupil-apriltags`, and
publishes an immutable `TagSample` (translation + `wxyz` quaternion + timestamp
+ sequence) into a latest-value slot. It also publishes the corrected raw BGR
frame with a sequence number and wakes `wait_frame()` waiters. When
`capture.source: tag`, `tag/source.py` wraps those frames as `FramePacket`s for
the hand pipeline; otherwise the tag camera stays independent of `capture.py`.
The detector import is lazy; a missing package or camera becomes a `TagHealth`
error and does not stop finger teleoperation. `mirror=auto` tries the normal
frame first and only flips when a valid `hamming=0` decode requires it, using
mirrored principal-point intrinsics for the resulting pose.

`tag/controller.py` owns the relative mapping. At zero it records both the tag
pose and the current hand-base orientation. Translation is a median-windowed
tag displacement through the camera preset, radial deadzone, gain, bounds,
One-Euro filter, and slew limiter. Rotation is:

```text
q_rel_tag    = conj(q_tag_zero) ⊗ q_tag_now
q_delta_hand = q_mount ⊗ q_rel_tag ⊗ conj(q_mount)
q_target     = q_hand_zero ⊗ q_delta_hand
```

`q_mount` is a cyclic tag-to-hand axis map: tag X → hand Y (pitch), tag Y →
hand Z (roll/middle), tag Z → hand X (yaw/palm normal). The rotation is then passed
through deadzone, gain, angle clamp, slerp smoothing, and rate limiting. The
first valid tag orientation is latched as an absolute zero even if follow is
off; `g` toggles following without changing it, `0` explicitly re-zeroes, and
`p` toggles translation. Missing or stale tags freeze both outputs; they never
return to zero. The app writes the resulting quaternion and offset to
`ShadowHand.set_macro_pose`.

## Simulation

`sim/model.py` loads the MJCF and actuator layout. `sim/shadow_hand.py` writes
`data.ctrl`, reads the 120 touch taxels, and sets the forearm mocap body. The
mocap rotation centre is selected by `tag.pivot` (`wrist` by default). The
`FixedStepClock` advances physics by wall time with bounded catch-up. The viewer
shows the zero triad through `Viewer.set_macro_zero`.

## Configuration and threading

`config.py` contains frozen dataclasses with YAML loading and dotted overrides.
`configs/default.yaml` mirrors the defaults and is locked by a test.

The main loop is the only writer of MuJoCo state. Perception, the tag camera, and
the viewer run in separate threads or processes and communicate through
immutable latest-value slots. The app never blocks on a camera or model.

## Failure behaviour

Perception exceptions are reported through `WorkerHealth`, logged, and make the
process exit non-zero. The tag service is optional for finger control: camera or
detector failures are reported in `TagHealth` and the HUD, while the rest of the
app continues. Tag samples older than `tag.stale_timeout_s` are not returned,
which freezes translation and orientation at their last values.
