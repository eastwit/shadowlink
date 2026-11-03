# Third-party notices

ShadowLink contains or depends on the following third-party work. The MIT
license in `LICENSE` applies only to ShadowLink's own source code.

## HaMeR

- Project: https://github.com/geopavlakos/hamer
- License: MIT (Copyright (c) 2023 UC Regents, Georgios Pavlakos)
- Usage: optional `third_party/hamer` git submodule, used as a library for
  3D hand reconstruction. Its source is not copied into this repository.

## MANO hand model

- Project: https://mano.is.tue.mpg.de
- License: Max Planck Institute for Intelligent Systems, non-commercial
  research license. The model, derived checkpoints, and any redistributed
  variants are **not** included in this repository.
- Users must register at the MANO website and download `MANO_RIGHT.pkl`
  themselves. `shadowlink setup-models` only checks for the file.

## Shadow Hand E3M5 description (MJCF and meshes)

- Source: MuJoCo Menagerie / Shadow Robot Company
  https://github.com/google-deepmind/mujoco_menagerie
- License: Apache License 2.0. The original license and attribution files
  are kept at `src/shadowlink/assets/shadow_hand/LICENSE` and
  `src/shadowlink/assets/shadow_hand/README.md`.
- The tactile site/sensor XML files are a derivative of that description.

## MediaPipe

- Project: https://github.com/google-ai-edge/mediapipe
- License: Apache License 2.0.
- The `hand_landmarker.task` model is downloaded separately by the user and is
  not included in this repository.

## MuJoCo

- Project: https://github.com/google-deepmind/mujoco
- License: Apache License 2.0.
