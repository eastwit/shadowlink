# Tactile simulation

The Shadow Hand model ships with 120 touch sensors on the five fingertips:
5 fingers × 6 rows × 4 columns. Each sensor is a MuJoCo `touch` sensor bound to
an ellipsoid `site` placed on the fingertip collision hull.

## Reading touch

`sim/shadow_hand.py` discovers sensors by name (`touch_<finger>_taxel_r<row>_c<col>`)
and exposes:

```python
hand.touch       # {"touch_ff_taxel_r0_c0": 0.0, ...} — 120 floats, newtons
hand.touch_map   # {"ff": (6,4) array, ...} — row 0 nearest the joint
hand.peak_touch  # {"ff": 0.0, ...}
```

Values come from `data.sensordata` after `mj_step`. The row/column layout is
stable and matches the XML order.

## Visualisation

`hand.update_touch_colors(force_scale)` maps each taxel's force to the site
RGBA:

- 0 N → blue (`0.15, 0.65, 1.0`);
- `force_scale` N or more → red (`1.0, 0.0, 0.1`);
- intermediate values interpolate and the site stays semi-transparent.

`sim.touch_force_scale` in the config controls the saturation point. The viewer
must have site group 4 enabled, which `Viewer.launch()` does automatically.

## Generating the sites

The checked-in files under
`src/shadowlink/assets/shadow_hand/tactile/` are generated from the collision
meshes, not hand-written:

```bash
shadowlink generate-tactile --check   # fail if the XML is stale
shadowlink generate-tactile           # rewrite the five *_sites.xml + sensors.xml
```

Algorithm (`tools/generate_tactile.py`):

1. Compile the scene and add a tiny mocap sphere as a probe.
2. For each finger, cast a 6×4 grid of rays from inside the distal link toward
   the fingertip collision geom. Row/column spacing follows a per-finger taper.
3. Project each mesh hit onto the convex collision hull with
   `mj_geomDistance`; reject the hit if the mesh/hull mismatch exceeds 1 mm or
   the hull gap is outside `(0, 1 cm)`.
4. Build a local frame whose z axis is the surface normal, x is the body x axis
   projected into the tangent plane, and y is their cross product.
5. Shrink lateral ellipsoid radii where neighbouring taxels approach each other
   at the tip, so one physical contact is not counted twice.
6. Emit `<site>` elements and matching `<touch>` sensors.

The generator writes LF line endings and compares content after universal
newline translation, so `--check` is stable across Windows and Linux checkouts.

## Assets

```text
assets/shadow_hand/
├── right_hand.xml          # includes tactile/ff_sites.xml ... th_sites.xml
├── left_hand.xml           # no tactile includes in v0.1
├── shadowlink_scene.xml    # the scene loaded by the app
├── tactile/
│   ├── ff_sites.xml ... th_sites.xml
│   └── sensors.xml         # 120 <touch> sensors
└── assets/*.obj            # meshes, Apache-2.0 from MuJoCo Menagerie
```

The left-hand MJCF is kept for reference and parses correctly, but it has no
tactile sites and is not used by the teleoperation app.

## Limitations

- Taxels are point sensors with an ellipsoid visual footprint, not a continuous
  contact patch; force values are MuJoCo touch-sensor outputs, not calibrated
  newtons from a real sensor.
- The generator depends on the collision geometry. Changing a fingertip mesh
  requires rerunning it; `--check` is the guard.
- Left-hand tactile arrays are not provided.
- Rendering requires OpenGL. `shadowlink demo --no-viewer` still simulates and
  updates colours without a window.

## Tests

`tests/test_tactile_assets.py` checks 120 sensors, 24 sites per finger, unit
quaternions, matching names, and that the right-hand XML includes every tactile
file. `tests/test_generate_tactile.py` checks grid completeness, sensor/site
correspondence, and that the committed XML is reproducible.
