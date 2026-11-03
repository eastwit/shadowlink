# Models and data

ShadowLink never stores weights in the repository. All model files live in a
data directory outside the source tree.

## Data directory

Resolution order:

1. `--data-dir PATH`
2. `SHADOWLINK_DATA` environment variable
3. `~/.cache/shadowlink` (Linux/macOS) or `%USERPROFILE%\.cache\shadowlink`
   (Windows)

Expected layout:

```text
<data-dir>/
├── hamer_ckpts/
│   ├── checkpoints/
│   │   └── hamer.ckpt               # HaMeR ViT checkpoint (~2.7 GB)
│   └── model_config.yaml            # config stored with the checkpoint
├── data/
│   └── mano/
│       └── MANO_RIGHT.pkl           # user-downloaded, not redistributable
└── mediapipe/
    └── hand_landmarker.task         # MediaPipe Hand Landmarker
```

`--checkpoint` and `--hand-model` override individual files. The HaMeR model
config is always resolved as `<checkpoint>/../model_config.yaml`
(`hamer_ckpts/model_config.yaml` in the standard layout).

An existing HaMeR demo installation can be reused without copying 2.6 GB:

```bash
shadowlink doctor --data-dir D:/models/hamer/_DATA
shadowlink run --data-dir D:/models/hamer/_DATA --backend hamer
```

## Installing the pieces

### HaMeR

```bash
git submodule update --init
pip install -e third_party/hamer --no-deps
```

The submodule is pinned to `3a01849`. Do not initialise its `ViTPose`
sub-submodule for inference; the checkpoint uses the local ViT backbone in
`hamer/models/backbones/vit.py`.

ShadowLink deliberately does **not** call upstream `load_hamer()`:

- it imports pyrender/HaMeR utilities before the OpenGL platform can be set,
  which fails on Windows with the default EGL backend;
- it depends on the process working directory for `CACHE_DIR_HAMER`.

`perception/hamer_backend.py` sets `PYOPENGL_PLATFORM` (`gl` on Windows, `egl`
elsewhere) before importing `hamer`, patches `CACHE_DIR_HAMER` to an absolute
path, removes the training-only `BACKBONE.PRETRAINED_WEIGHTS`, fixes `BBOX_SHAPE`
for 256-pixel ViT checkpoints, points `MANO.MODEL_PATH` at the absolute file, and
calls `HAMER.load_from_checkpoint(..., init_renderer=False, strict=False)`.

`chumpy` 0.70 predates modern setuptools and can fail to build with
`use_2to3 is invalid`. If installing the `hamer` extra stops there, install it
from source first and re-run the install:

```bash
pip install git+https://github.com/mattloper/chumpy
pip install -e ".[dev,hamer]"
```

Alternatively build the wheel in a temporary environment with
`pip install "setuptools<58" wheel`.

### HaMeR demo data

`shadowlink setup-models` downloads the public demo archive:

```text
https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
```

The archive is extracted into a temporary empty directory first; member paths
are checked against traversal and symlinks before extraction, so a malicious
archive cannot write outside the data directory.

### MANO

MANO is not downloaded by this project. It is distributed by the Max Planck
Institute under a non-commercial research licence.

1. Register at <https://mano.is.tue.mpg.de>.
2. Download the MANO model archive.
3. Place `MANO_RIGHT.pkl` at `<data-dir>/data/mano/MANO_RIGHT.pkl`.

`setup-models` only checks for the file and prints this path if it is missing.
The HaMeR checkpoint is derived from MANO; treat it as non-redistributable too.

### MediaPipe

```bash
shadowlink setup-models --mediapipe-only
```

or download the float16 Hand Landmarker directly:

```text
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

MediaPipe is also enough for the CPU fallback backend; no MANO or HaMeR
checkpoint is needed with `--backend mediapipe`.

## GPU and PyTorch

The HaMeR backend requires PyTorch. Install a build that matches your CUDA
driver *before* installing the `hamer` extra, for example from the
[PyTorch selector](https://pytorch.org/get-started/locally/). ShadowLink is
tested with PyTorch 2.0.1 + CUDA 11.8 on Windows 11.

`--fp16` enables CUDA autocast inside `torch.inference_mode()`. It is ignored on
CPU. On a 4060 the difference is small; disabling it with `--no-fp16` is a useful
diagnostic if a checkpoint produces NaNs.

Device selection is automatic (`cuda` if available, otherwise `cpu`). There is
no hidden fallback: if CUDA is requested through a future explicit device option
and unavailable, the loader raises an error.

## Version checks

```bash
shadowlink doctor --strict-models
```

Verifies:

- Python, NumPy, OpenCV, MuJoCo, MediaPipe, PyYAML, and (if installed) PyTorch
  versions;
- CUDA availability and device name;
- checkpoint/config/MANO/landmarker presence and sizes;
- MJCF compilation (`nq=nv=24`, `nu=20`, `nsensor=120`, `timestep=0.002`);
- the 13 keyframes and the 120 tactile sites.

`--json` emits machine-readable output for issue reports.

## Why not commit the weights?

- MANO's licence forbids redistribution and limits use to non-commercial
  research.
- HaMeR checkpoints are large and derived from MANO.
- MediaPipe's `.task` file is better fetched from its official URL.

`.gitignore` excludes `*.ckpt`, `*.pkl`, `*.task`, `*.npz`, `_DATA/`, `data/`,
and `models/`. If you fork the project, keep those rules.
