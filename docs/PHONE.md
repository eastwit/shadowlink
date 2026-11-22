# phyphox phone tilt

ShadowLink can read a phone's attitude over
[phyphox](https://phyphox.org) Remote Access and use it for the two hand axes
that are noisy in the AprilTag pose. No phone app needs to be built and no extra
Python package is required: the client is plain HTTP + JSON from the standard
library.

## What the phone controls

The AprilTag remains responsible for:

- whole-hand **translation**;
- hand **yaw** (rotation about the tag's blue axis).

The phone supplies:

- hand **pitch** and **roll**, from the phone's yaw and pitch attitude;
- the phone **roll** channel (rotation about the screen normal, user-frame +Z)
  is discarded.

Tag and phone zeros are latched together at the first moment both are ready.
If the phone sample becomes stale, orientation freezes while tag translation
continues.

## 1. Prepare the phone

1. Install phyphox (Android or iOS).
2. Open an experiment that records **Attitude** / rotation vector. A custom
   experiment with `attW`, `attX`, `attY`, `attZ` buffers works too. If only
   gyroscope + accelerometer channels exist, ShadowLink falls back to a Mahony
   filter.
3. Enable **Remote Access** in the experiment menu. phyphox shows an address
   such as `http://192.0.2.10:8080`.
4. Keep the phone and computer on the same network. Guest Wi-Fi and router
   client isolation block the connection.
5. Verify from the computer:

```bash
curl http://192.0.2.10:8080/config
```

The reply is JSON with a `buffers` list. If this fails, fix the network before
troubleshooting ShadowLink.

## 2. Configure

The recommended profile is `configs/plugcam_hand_tag.yaml`:

```yaml
phone:
  source: phyphox
  host: 192.0.2.10
  port: 8080
  axes: yaw_pitch
  mount_preset: yaw_pitch
  deadzone_deg: 1.0
```

Run:

```bash
shadowlink run --config configs/plugcam_hand_tag.yaml --backend hamer \
  --data-dir D:/models/hamer/_DATA
```

The HUD line shows the phone state, request rate, tilt state, and active mount:

```text
phone running | 192.0.2.10:8080 | 40 Hz | attitude | tilt tracking | mount swap-d
```

## 3. Axis mapping and `m`

`phone.mount_preset: yaw_pitch` maps:

```text
phone screen normal (+Z, roll)  -> hand yaw axis (+X), discarded
phone yaw axis (+X)             -> hand roll axis (+Z)
phone pitch axis (+Y)           -> hand pitch axis (+Y)
```

The shipped direction combination is `swap-d`. If the hand turns in the
opposite direction, or the two phone axes act on the wrong hand axes, press
**`m`**. It cycles eight proper mount rotations:

```text
swap-a  swap-b  swap-c  swap-d  cyclic-a  cyclic-b  cyclic-c  cyclic-d
```

Each press re-zeroes the orientation at the current hand pose. Stop when the
motion feels correct and note the HUD `mount` name. To persist it, either keep
`phone.mount_preset: yaw_pitch` (the default is `swap-d`) or use a custom
quaternion:

```yaml
phone:
  mount_preset: custom
  mount_w: 0.0
  mount_x: -0.7071068
  mount_y: 0.0
  mount_z: 0.7071068
```

## Configuration reference

| Key | Default | Meaning |
| --- | --- | --- |
| `source` | `none` | `none` or `phyphox` |
| `host` | `null` | phone IP or hostname |
| `port` | `8080` | phyphox Remote Access port |
| `poll_hz` | `50.0` | HTTP polling rate |
| `timeout_s` | `1.0` | per-request timeout |
| `retry_s` | `1.5` | reconnect interval |
| `auto_start` | `true` | send `/control?cmd=start` if paused |
| `auto_discover` | `false` | opt-in local `/24` discovery |
| `stale_timeout_s` | `0.35` | freeze orientation after this gap |
| `axes` | `yaw_pitch` | keep phone yaw+pitch; `pitch_roll` is the gravity mode |
| `mount_preset` | `yaw_pitch` | `yaw_pitch`, `back`, `in_hand`, or `custom` |
| `mount_w/x/y/z` | `0/-0.7071068/0/0.7071068` | custom phone→hand mount |
| `gain` | `1.0` | relative angle gain |
| `deadzone_deg` | `1.0` | ignore tiny motion |
| `max_angle_deg` | `180.0` | relative angle clamp |

Dotted overrides work too, for example:

```bash
shadowlink run --set phone.host=192.0.2.10 --set phone.deadzone_deg=2
```

## Troubleshooting

- **`phone error` / `connecting`:** run `curl http://HOST:8080/config`. Check
  that Remote Access is enabled, the experiment is running, and both devices
  are on the same subnet.
- **Phone rate is near zero:** the experiment is paused or Wi-Fi is congested.
  Lower `phone.poll_hz` to 30 if the phone's HTTP server is overloaded.
- **`phone stale` / orientation frozen:** Wi-Fi dropped or the experiment
  stopped. Tag translation still works; orientation resumes when samples return.
- **Axes swapped or reversed:** press `m`; no restart is needed. Note the mount
  name shown in the HUD.
- **Orientation never starts:** both a fresh tag and a fresh phone sample are
  required to latch the zero. Check the tag HUD line as well.

## Security

phyphox Remote Access is unauthenticated plain HTTP. Use it only on a trusted
local network and never expose the port to the internet.
