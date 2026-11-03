# Calibration

The retargeter maps *relative* hand motion: an open, relaxed hand should produce
the neutral Shadow Hand pose, not a hard-coded all-zero target. Calibration
records the flexion, spread, and thumb opposition of that neutral pose.

## Capture a neutral pose

Interactive (recommended):

```bash
shadowlink calibrate --backend hamer --data-dir D:/models/hamer/_DATA
```

Hold an open hand still in front of the camera. After the hand has been tracked
continuously for 0.5 s, the app saves the calibration and exits. `q`/`Esc`
cancels. The observation must be fresh (age ≤ 0.25 s) when it is saved.

During a normal `shadowlink run`, press `c` while holding the open hand. The HUD
shows the result; the running controller starts using the new neutral
immediately.

## Storage

Default paths:

| Platform | Path |
| --- | --- |
| Windows | `%APPDATA%\ShadowLink\calibration.json` |
| Linux | `$XDG_CONFIG_HOME/shadowlink/calibration.json` or `~/.config/shadowlink/calibration.json` |
| macOS | `~/Library/Application Support/ShadowLink/calibration.json` |

Override with `--calibration PATH` or `SHADOWLINK_CALIBRATION`. The file is
written atomically: a temporary file in the same directory is flushed, fsynced,
and `os.replace`d over the old file. A failed write leaves the previous
calibration intact.

Example:

```json
{
  "backend": "hamer",
  "created_at": "2026-11-03T12:34:56+00:00",
  "neutral": {
    "flex": {
      "index": [0.09, 0.11, 0.10],
      "middle": [0.10, 0.10, 0.09],
      "ring": [0.11, 0.10, 0.10],
      "pinky": [0.10, 0.12, 0.10],
      "thumb": [0.12, 0.15, 0.10]
    },
    "opposition": 0.0,
    "spread": {
      "index": -0.12,
      "middle": 0.0,
      "ring": 0.1,
      "pinky": 0.2
    }
  },
  "schema_version": 1,
  "side": "right"
}
```

Angles are radians. `flex` holds MCP/PIP/DIP per finger; `spread` is the absolute
in-palm angle of the four non-thumb fingers.

## Validation and compatibility

On load, ShadowLink checks:

- `schema_version == 1`;
- all five fingers have a 3-value `flex` list;
- all four non-thumb fingers have a `spread` value;
- all values are numeric.

A malformed file is reported and ignored (the app falls back to
`Neutral.default()`), rather than aborting a teleoperation session.

The calibration is bound to the backend that produced it. If a file says
`"backend": "hamer"` and you run `--backend mediapipe`, it is ignored with a
warning; calibrate again rather than mixing camera conventions. If `--side` is
explicit and the stored side differs, the app logs a warning.

## Why backend-bound?

HaMeR and MediaPipe produce different keypoint conventions and noise levels.
The neutral flexion measured from a HaMeR mesh is not interchangeable with the
world-landmark angles from MediaPipe. Schema and backend checks make that
explicit instead of silently shifting every actuator target.

## Reset behaviour

`r` resets physics and targets to the configured start pose (`sim.start_pose`,
default `open hand`). It does **not** delete the calibration. To recalibrate,
press `c` again.

## Tuning

If the hand feels too bent or too straight at rest:

1. Recalibrate with a genuinely relaxed open hand.
2. Adjust `retarget.flex_gain` or `distal_gain` with `--set`.
3. For individual fingers, use `--config` with a modified
   `configs/default.yaml`; semantic actuator names are documented in
   [TACTILE.md](TACTILE.md) and the MJCF.

```bash
shadowlink run --set retarget.flex_gain=0.9 --set smoothing.tau=0.07
```

Use `--smooth` as a shortcut for `smoothing.tau`.
