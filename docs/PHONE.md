# Phone macro orientation

ShadowLink can rotate the whole Shadow Hand with a phone while the camera keeps
controlling the fingers. The phone runs the third-party
[phyphox](https://phyphox.org) app; ShadowLink is a read-only HTTP client for its
**Remote Access** server. There is no custom phone app and no extra Python
dependency.

The two inputs are independent:

- the camera path drives the 20 finger actuators;
- the phone path drives the `rh_forearm` mocap body (yaw, pitch, roll).

Press `g` to start following, press it again to freeze the macro pose, and press
it again to resume from the current phone attitude. Press `0` at any time to
make the current phone attitude the new zero: the hand keeps its current
orientation and continues (or starts) following. `r` resets everything.

## 1. Prepare the phone

1. Install phyphox (Android or iOS).
2. Open an experiment that records **Attitude**. The built-in *Attitude* /
   *Rotation Vector* experiments are suitable. If your experiment only exposes
   gyroscope and accelerometer channels, ShadowLink falls back to a Mahony
   filter (see [Raw-IMU fallback](#raw-imu-fallback)).
3. Enable **Remote Access** in the experiment menu. phyphox shows an address
   such as `http://192.0.2.10:8080`.
4. Keep the phone and the computer on the same network. Guest Wi-Fi and router
   client isolation will block the connection.

Verify from the computer:

```bash
curl http://192.0.2.10:8080/config
```

The reply is JSON containing a `buffers` list. If that fails, fix the network
before troubleshooting ShadowLink.

## 2. Run teleoperation

```bash
shadowlink run --backend hamer --phone phyphox \
  --phone-host 192.0.2.10 --phone-follow \
  --data-dir D:/models/hamer/_DATA
```

`--phone-host` accepts a bare host, `host:port`, or `http://host:port/`.
`--phone-port` overrides the port. Add `--phone-scan` to let ShadowLink search
the local `/24` when the address is unknown or changes; discovery is off by
default.

Without `--phone-follow`, the HUD shows the phone state and you press `g` to
start following, or `0` to start and re-zero in one step. With `--phone-follow`,
the first ready sample starts following automatically, which is useful in
headless mode where keyboard input is not available.

The HUD line has the form:

```text
phone on | running | 192.0.2.10:8080 | 30 Hz | following
zero p/r/y +12.3 -45.6 +78.9
```

The second line reports the hand-base Euler angles of the current zero. The
passive viewer also draws an RGB triad at that orientation (see
[Zero marker](#zero-marker)).

## 3. Diagnose before driving the robot

```bash
# Live state, rate, quaternion, Euler angles, relative motion
shadowlink phone-test --host 192.0.2.10 --seconds 10

# Show the experiment buffers and the selected attitude/IMU mapping
shadowlink phone-test --host 192.0.2.10 --list

# Machine-readable samples
shadowlink phone-test --host 192.0.2.10 --json
```

If the rate is near zero, the experiment is paused or the Wi-Fi link is bad. If
`--list` reports `selected: imu`, the experiment has no fused quaternion and the
fallback is active.

## Configuration

The `phone:` section of `configs/default.yaml` mirrors `PhoneConfig`:

| Key | Default | Meaning |
| --- | --- | --- |
| `source` | `none` | `none` or `phyphox` |
| `host` | `null` | phone IP or hostname; no personal address is committed |
| `port` | `8080` | phyphox Remote Access port |
| `poll_hz` | `30.0` | polling rate |
| `timeout_s` | `1.0` | per-request HTTP timeout |
| `retry_s` | `1.5` | reconnect interval |
| `auto_start` | `true` | send `/control?cmd=start` when the experiment is paused |
| `auto_discover` | `false` | allow local `/24` discovery |
| `auto_follow` | `false` | start following at the first ready sample |
| `pivot` | `wrist` | rotation centre: `palm`, `wrist`, or `forearm` |
| `stale_timeout_s` | `0.35` | freeze the hand after this gap |
| `mount_preset` | `back` | `back` (strapped), `in_hand` (held screen-up), or `custom` |
| `mount_w/x/y/z` | `0/0.7071068/0/-0.7071068` | custom phone body → hand base rotation, used when the preset is `custom` |
| `gain` | `1.0` | relative angle gain |
| `deadzone_deg` | `1.5` | ignore tiny motion |
| `max_angle_deg` | `180.0` | clamp on relative rotation; `180` means no clamp |
| `smoothing_tau_s` | `0.06` | slerp smoothing time constant |
| `max_rate_rad_s` | `6.0` | angular rate limit |

Example YAML:

```yaml
phone:
  source: phyphox
  host: 192.0.2.10
  port: 8080
  auto_follow: true
  pivot: wrist
  mount_preset: back      # change to in_hand while holding the phone
  # Optional soft limit; 180 (the default) disables the clamp.
  max_angle_deg: 120
```

`--phone-host` and the other flags override YAML; `--set phone.gain=0.8` also
works.

## Rotation pivot

`phone.pivot` chooses the world point that stays fixed while the whole hand
rotates:

- `wrist` (default) keeps the `rh_wrist` joint fixed, so the hand swings out
  from the wrist. This matches a phone strapped at the wrist.
- `forearm` keeps the proximal end of the `rh_forearm` mesh fixed, so the whole
  forearm and hand sweep around that point like a rotation at the elbow. The
  hand travels farther than with `wrist`.
- `palm` keeps the old palm/finger centroid fixed, so the hand turns in place.
  Use it only to compare with older versions.

Select it in YAML or on the command line:

```bash
shadowlink run --backend hamer --phone phyphox --phone-host 192.0.2.10 \
  --set phone.pivot=forearm --data-dir D:/models/hamer/_DATA
```

## Body frame and axis tuning

ShadowLink publishes phone attitude in a documented body frame:

- `x` points out of the charging port (the bottom of the phone in portrait use);
- `y` points across the phone width;
- `z` points out of the screen.

Android's native rotation vector uses x=right, y=screen-up, z=out of the screen,
so the service applies a fixed −90° rotation about z before publishing. This
makes `phone-test` and the macro layer agree on one convention.

The mount quaternion `phone.mount_*` maps that phone frame to the Shadow Hand
base frame. `phone.mount_preset` selects one of two common physical mounts:

| Preset | Physical arrangement | Phone axes in the hand base frame |
| --- | --- | --- |
| `back` (default) | phone strapped to the **back of the hand**: top toward the fingers, charging port toward the elbow, screen facing away from the hand | `+x → −z`, `+y → −y`, `+z → −x` |
| `in_hand` | phone held flat in the **palm**, screen facing away from the palm, top toward the fingers (the usual way to hold a phone while testing) | `+x → −z`, `+y → +y`, `+z → +x` |

The two mounts differ by a 180° rotation about the phone's long axis. As a
result, if rotations about the charging-port axis feel correct but rotations
about the phone width and screen normal feel reversed, you are using the wrong
preset. Switch between `back` and `in_hand`:

```bash
shadowlink run --backend hamer --phone phyphox --phone-host 192.0.2.10 \
  --phone-mount in_hand --data-dir D:/models/hamer/_DATA
```

The `back` preset corresponds to `mount_w/x/y/z = 0/0.7071068/0/-0.7071068`;
the `in_hand` preset corresponds to `0.7071068/0/0.7071068/0`.

Set `mount_preset: custom` to use the `mount_w/x/y/z` fields directly. This is
needed for any other mounting; a 90° mount rotation about an axis is a useful
correction, for example:

```yaml
phone:
  mount_preset: custom
  mount_w: 0.7071068
  mount_z: 0.7071068   # +90° about z
```

Procedure:

1. Run `shadowlink phone-test --host <ip> --seconds 20`.
2. Rotate the phone slowly about one body axis and note the reported relative
   axis-angle.
3. Run with the viewer, press `g`, and compare the hand's motion with the phone.
4. Pick the matching `mount_preset`, or adjust `mount_*` with
   `mount_preset: custom`, until the directions and range feel right.

If the motion jitters, raise `smoothing_tau_s` or lower `max_rate_rad_s`. If it
feels sluggish, lower `smoothing_tau_s`.

## Zero marker

The passive MuJoCo viewer draws a small RGB axis triad at the rotation pivot.
Its orientation is the hand-base orientation captured when the mapping was
zeroed — the pose shown by `zero p/r/y` in the HUD:

- **red** = hand-base `+x` (palmar at rest),
- **green** = hand-base `+y`,
- **blue** = hand-base `+z` (distal, toward the fingers at rest).

The triad is lifted along the palmar axis so it is not hidden inside the hand
mesh. It marks the orientation only; it does not follow the current hand pose.
Press `0` to recapture the marker at the current pose, or `r` to return to the
start pose.

There is no attitude clamp by default: `max_angle_deg` is `180`, so any relative
rotation of the phone is passed through (subject to `gain`, smoothing, and the
rate limit). Lower `max_angle_deg` if you want a softer safety limit.

## Raw-IMU fallback

When the experiment exposes no `w/x/y/z` attitude group, the service looks for
gyroscope (`gyro*`, `gyr*`) and accelerometer (`accel*`, `acc*`) X/Y/Z buffers,
plus an optional magnetometer (`mag*`). It initialises a Mahony filter from about
20 accelerometer samples and integrates gyroscope rates, using the host receipt
time as `dt`.

The fallback is less accurate than the fused rotation vector:

- yaw drifts without a magnetometer;
- startup needs about 0.7 s of still acceleration samples;
- Wi-Fi jitter directly perturbs the integration step, although the step is
  clamped to `[0.001, 0.2] s`.

Prefer an experiment with an **Attitude** output whenever possible.

## Failure behaviour

| Health state | Meaning |
| --- | --- |
| `starting` | thread starting |
| `connecting` | resolving/connecting or waiting for data |
| `running` | fresh samples are arriving |
| `paused` | the experiment is paused on the phone |
| `stale` | no fresh sample within `stale_timeout_s` |
| `error` | unreachable phone or invalid configuration |
| `stopped` | service closed |

`latest()` returns `None` when no sample is available or the newest one is older
than `stale_timeout_s`. The app freezes the macro pose and turns following off;
the finger path keeps running. Press `0` after recovery to re-zero and resume.

Only outbound HTTP is used. ShadowLink never listens on a port, never writes the
phone address to disk, and does not send anything except the documented
`/config`, `/get`, and `/control` requests.

## Security

phyphox Remote Access has **no authentication and no encryption**. Anyone who
can reach the port can read the experiment data and send control commands.

- Use a trusted local network only.
- Do not forward the phyphox port through a router or tunnel it to the internet.
- Stop Remote Access when finished.
- Prefer an explicit `--phone-host` over broad scanning.
- Treat the phone's experiment data and any device identifiers as private.

## Limitations

- One phone is supported. There is no dual-device fusion.
- The macro layer is a 3-DOF kinematic base rotation; it does not apply reaction
  forces to the fingers.
- The current Shadow Hand asset is right-handed; the left hand is not driven in
  v0.1.
- Wi-Fi latency is not compensated. The mapping is relative and smoothed, so a
  short dropout freezes rather than jumps.

## Troubleshooting

See the phone section of [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
