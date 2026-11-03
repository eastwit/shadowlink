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
app.py (TeleopApp) ──────────────────────────────┐
   │            │            │                   │
   ▼            ▼            ▼                   ▼
capture.py   perception/   retarget/          sim/ + ui/
             │            │                   │
             ▼            ▼                   ▼
          types.py     geometry.py         model.py
                       mapper.py           shadow_hand.py
                       smoothing.py        stepping.py
                       calibration.py      viewer.py
```

Dependency rules, enforced by review and tests:

- `types.py` depends only on NumPy (plus the standard library).
- `retarget/` never imports `sim/`; it receives an `ActuatorLayout` through a
  `TYPE_CHECKING`-only import.
- `sim/` never imports `perception/` or `retarget/`.
- Only `app.py` orchestrates the layers.
- Only `cli.py` uses `argparse`.
- `hamer` is imported lazily inside `perception/hamer_backend.py`.
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
- OpenCV HighGUI calls and MuJoCo viewer calls happen only on the main thread.
- Cleanup order is capture → perception → viewer → recorder.
- The perception service joins its threads with a timeout and logs any that fail
  to stop.

## Extension points

- New backend: implement `detect`/`estimate` (see `perception/base.py`) and
  register it in `create_backend`; keep the `HandObservation` contract.
- New detector: return `HandDetection` objects; the worker handles selection and
  filtering.
- New mapping: change `Retargeter.map_metrics`; semantic names keep the XML
  coupling explicit.
- New camera transport: implement the `FrameSource` protocol in `capture.py`.
