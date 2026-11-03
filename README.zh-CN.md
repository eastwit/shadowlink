# ShadowLink

**用单目 RGB 摄像头连续遥操作 MuJoCo 中的 Shadow Hand。**

ShadowLink 从单张图像重建人手姿态，并把连续的关节运动映射到 Shadow Hand E3M5
的 20 个执行器上。仿真中包含 120 个指腹触觉单元，并实时用颜色显示接触力。
项目**不做手势分类**：每帧都根据关节弯曲、手指张开和拇指对掌连续计算目标，
因此机器人会跟随操作者的实际运动，而不是在预设手势之间切换。

[English](README.md)

## 功能

- **两种重建后端**
  - **HaMeR**（推荐）：GPU 上的单目 3D 手部重建，输出 21 个关节和 MANO 网格，
    并把裁剪空间的相机参数还原到整幅图像。
  - **MediaPipe**（CPU 后备）：使用 Hand Landmarker 的 `hand_world_landmarks`，
    不需要 CUDA 或 MANO。
- **连续重定向**：基于掌面几何计算 MCP/PIP/DIP 弯曲、手指张开和拇指对掌，
  不使用离散手势或预设。
- **两阶段异步感知**：MediaPipe 检测与 HaMeR 推理分别在独立线程中运行，
  每个阶段只保留最新一帧，渲染和物理循环从不等待模型。
- **稳定、可配置的控制**：One-Euro 滤波、速度限幅、`waiting → tracking → lost`
  状态机，以及失手后按速度回到中位，而不是瞬间跳到 0。
- **可持久化的校准**：采集张开手的零点姿态，按后端保存并在进程重启后校验。
- **从模型读取执行器布局**：关节名称、顺序、类型和限位全部来自 MJCF，
  启动时校验，不再维护重复常量。
- **触觉仿真**：120 个触觉点（5 指 × 6 × 4），按力大小从蓝色渐变到红色。
- **适合无头环境**：`--no-show --no-viewer`、合成帧源，以及不需要 GPU、
  摄像头或权重的单元测试。
- **无路径 hack**：不使用 `os.chdir`、`sys.path.insert`，不写死用户目录，
  所有模型路径均为绝对路径。

## 环境要求

| 组件 | 版本 / 说明 |
| --- | --- |
| Python | 3.10（以下组合在 3.10.22 上验证） |
| MuJoCo | 3.16+ |
| MediaPipe | 0.10.14+ |
| OpenCV | 4.8+ |
| HaMeR 后端 | PyTorch 2.0–2.5，建议使用 CUDA |
| GPU | 任意 CUDA GPU；下文数据来自 RTX 4060 Laptop |

支持 Windows 11 和 Linux。MuJoCo viewer 与 OpenCV 预览均可关闭。

## 安装

### 1. 连同 HaMeR 子模块克隆

```bash
git clone --recurse-submodules <仓库地址> ShadowLink
cd ShadowLink
```

如果已经克隆但未初始化子模块：

```bash
git submodule update --init
```

子模块固定为 HaMeR `3a01849`。推理不需要递归初始化其 ViTPose 子模块。

### 2. 创建环境

```bash
conda create -n shadowlink python=3.10 -y
conda activate shadowlink
```

如需 GPU 推理，先按
[PyTorch 官方选择器](https://pytorch.org/get-started/locally/) 安装对应 CUDA
版本的 PyTorch，然后：

```bash
pip install -e ".[dev,hamer]"
pip install -e third_party/hamer --no-deps
```

`--no-deps` 是有意为之：HaMeR 的 `setup.py` 会拉取仅训练需要的 detectron2、
mmcv 等包，并可能覆盖刚安装的 PyTorch 版本。

只用 CPU/MediaPipe 时：

```bash
pip install -e ".[dev]"
```

### 3. 获取模型

```bash
shadowlink setup-models
```

该命令下载 HaMeR demo checkpoint/config 和 MediaPipe Hand Landmarker，**不会**
下载 MANO。

MANO 采用非商业研究许可。请前往 <https://mano.is.tue.mpg.de> 注册、下载
`MANO_RIGHT.pkl`，放到：

```text
<data-dir>/data/mano/MANO_RIGHT.pkl
```

默认数据目录为 `~/.cache/shadowlink`（Linux）或
`%USERPROFILE%\.cache\shadowlink`（Windows）。可用 `--data-dir` 或
`SHADOWLINK_DATA` 覆盖。已有的 HaMeR `_DATA` 目录可以直接复用：

```bash
shadowlink setup-models --data-dir D:/models/hamer/_DATA --check-only
```

### 4. 验证

```bash
shadowlink check --data-dir D:/models/hamer/_DATA
shadowlink doctor --data-dir D:/models/hamer/_DATA --strict-models
```

`doctor` 会检查 Python/依赖版本、CUDA、模型文件、校准文件以及 MJCF/触觉资产。

## 快速开始

```bash
# 摄像头 + HaMeR + MuJoCo viewer
shadowlink run --backend hamer --data-dir D:/models/hamer/_DATA

# 纯 CPU 后备
shadowlink run --backend mediapipe --data-dir D:/models/hamer/_DATA

# 回放视频，不打开任何窗口
shadowlink run --source video --camera demo.mp4 --backend hamer \
  --no-show --no-viewer --max-frames 120

# 无摄像头：循环展示 13 个关键帧姿态
shadowlink demo
```

### 按键

| 按键 | 作用 |
| --- | --- |
| `q` / `Esc` | 退出 |
| `c` | 采集零点姿态（保持张开手不动） |
| `r` | 重置物理状态和目标到起始姿态 |

### 常用参数

```text
--source camera|video        --camera 索引或路径
--backend hamer|mediapipe    --side any|left|right   （真实左右手）
--pick largest|center|track  --skip N
--no-mirror                  --draw skeleton|mesh|none
--smooth 秒                  --output 录像.mp4
--no-show --no-viewer        --max-frames N
--fp16 / --no-fp16           --calibration 路径
--config 文件                --set section.field=value
```

优先级：**CLI 参数 / `--set` > `--config` YAML > dataclass 默认值**。
`configs/default.yaml` 与默认 `AppConfig` 完全一致，并有测试锁定。

## 工作原理

```text
摄像头/视频 ──► FramePacket（不可变，只镜像一次）
                     │
              ┌──────┴───────┐
              │  检测线程     │  MediaPipe HandLandmarker（VIDEO 模式）
              │              │  左右手过滤 + largest/center/IoU 选择
              └──────┬───────┘
                     │ 最新检测（旧帧直接丢弃）
              ┌──────┴───────┐
              │  推理线程     │  HaMeR 裁剪 → checkpoint → 21 关节 + 网格
              │              │  （MediaPipe 后端直接复用 world landmarks）
              └──────┬───────┘
                     │ 最新 PerceptionResult（frame_seq + 时间戳）
                     ▼
   重定向（掌面坐标系 → 弯曲/张开/对掌）
                     ▼
   One-Euro 滤波 → 速度限幅 → 状态机 → 20 维执行器目标
                     ▼
   MuJoCo 固定步长物理 + 120 触觉 + viewer/预览
```

关键约定：

- 帧对象是不可变 dataclass，采集缓冲区不会跨线程被修改。
- 每个结果都带 `frame_seq`、`capture_ts`、`finished_ts`，因此可以区分
  “这一帧没有手”和“worker 尚未产出结果”。
- 主循环只读取最新结果，旧结果直接忽略。
- 感知线程异常会写入 `WorkerHealth`、记录日志，并让进程以非零码退出，
  不会静默停留在过期数据上。
- 左手会被镜像规范化后再重定向。v0.1 不驱动左手 Shadow Hand 模型。

模块边界和线程模型见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 校准

面对摄像头张开手，按 `c`，或运行：

```bash
shadowlink calibrate --backend hamer --data-dir D:/models/hamer/_DATA
```

零点数据保存在 Windows 的 `%APPDATA%\ShadowLink\calibration.json`，或 Linux 的
`~/.config/shadowlink/calibration.json`。校准与后端和左右手绑定；切换后端时会
提示重新校准，而不是悄悄使用错误的零点。

详见 [docs/CALIBRATION.md](docs/CALIBRATION.md)。

## 性能

RTX 4060 Laptop、1200×1799 视频源、30 帧的参考数据：

| 指标 | p50 |
| --- | --- |
| MediaPipe 检测 | 47 ms |
| HaMeR 推理（fp16） | 62.5 ms |
| 消费时的观测年龄 | 约 180 ms |
| 主循环 | 26.9 fps |

检测和推理会跨帧重叠，物理循环按真实时间独立推进。机器较慢时可用
`--skip 2`，或直接用 `--backend mediapipe` 去掉 GPU 模型。实际数据取决于
分辨率、裁剪尺寸和散热状态。在这块 GPU 上 `--no-fp16` 没有更快。

## 离线工具

```bash
# 对单张图片重定向并渲染 Shadow Hand 姿态
shadowlink render-pose --image hand.jpg --out pose.png --backend hamer \
  --data-dir D:/models/hamer/_DATA

# 检查仓库中的触觉 XML 是否可由网格复现
shadowlink generate-tactile --check

# 修改指尖几何后重新生成
shadowlink generate-tactile
```

## 开发

```bash
pytest -q                       # 无需 GPU/权重/摄像头
pytest -m "not hamer and not mediapipe" -q
ruff check src tests
ruff format --check src tests
```

需要模型或真实图片的集成测试是 opt-in：

```bash
# HaMeR checkpoint + MediaPipe + 一张图片
SHADOWLINK_TEST_DATA=D:/models/hamer/_DATA \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m hamer -q

# 仅 MediaPipe
SHADOWLINK_TEST_HAND_MODEL=D:/models/hamer/_DATA/mediapipe/hand_landmarker.task \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m mediapipe -q
```

`tests/test_no_path_hacks.py` 会强制检查安装包中不存在 `os.chdir`、
`sys.path` 修改或用户绝对路径。

目录结构：

```text
src/shadowlink/
├── capture.py            摄像头/视频源、镜像、合成帧源
├── config.py             dataclass 配置、YAML、点号覆盖
├── paths.py              包内资产、数据目录、模型、校准路径
├── types.py              不可变帧/检测/观测
├── app.py                唯一的编排者
├── cli.py                argparse 入口
├── perception/           MediaPipe 检测器、HaMeR 后端、worker、选择器
├── retarget/             几何、映射、平滑、校准
├── sim/                  MJCF 加载、Shadow Hand、固定步长、viewer
├── ui/                   叠加层、遥测、OpenCV 窗口/录像
├── tools/                setup-models、doctor、触觉生成、render-pose
└── assets/shadow_hand/   MJCF、OBJ 网格、触觉 XML、原始许可证
```

常见问题见 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)。

## 许可与署名

ShadowLink 自身代码采用 MIT 许可。HaMeR 是以 MIT 许可引入的子模块；Shadow Hand
资产来自 MuJoCo Menagerie / Shadow Robot，采用 Apache-2.0；MediaPipe 采用
Apache-2.0。MANO 与 HaMeR checkpoint **不随仓库分发**，仍受各自原始许可约束。
详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 引用

使用 HaMeR 后端时，请引用原始工作：

```bibtex
@inproceedings{pavlakos2023reconstructing,
  title   = {Reconstructing Hands in 3D with Transformers},
  author  = {Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and
             Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
  booktitle = {CVPR},
  year    = {2024}
}
```
