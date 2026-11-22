# ShadowLink

**用内置摄像头识别手指，用手持 AprilTag 控制整手位置和朝向，在 MuJoCo 中遥操作 Shadow Hand。**

ShadowLink 从单张 RGB 图像重建人手姿态，并把连续的关节运动映射到 Shadow Hand E3M5
的 20 个执行器上。仿真中包含 120 个指腹触觉单元，并实时用颜色显示接触力。
项目**不做手势分类**：每帧都根据关节弯曲、手指张开和拇指对掌连续计算目标。

整手基座使用一个手持 AprilTag（默认 36h11 / ID 5 / 57 mm）。Astra 彩色相机
跟踪 tag，内置摄像头继续识别手指；tag 的位置和姿态相对于零点驱动整只手。

## 功能

- **两种手指重建后端**
  - **HaMeR**（推荐）：GPU 上的单目 3D 手部重建。
  - **MediaPipe**（CPU 后备）：使用 `hand_world_landmarks`，不需要 CUDA 或 MANO。
- **连续重定向**：基于掌面几何计算 MCP/PIP/DIP 弯曲、手指张开和拇指对掌。
- **两阶段异步感知**：检测与 HaMeR 推理分别在独立线程中运行，物理循环不等待模型。
- **AprilTag 整手 6DoF**：独立 Astra 相机线程；tag 平移和旋转共同驱动整手基座，
  首次检测自动对零，`g`/`p` 分别开关朝向/平移，`0` 重新对零。tag 丢失后冻结。
- **稳定控制**：One-Euro 滤波、四元数 slerp、速度限幅、`waiting → tracking → lost`
  状态机，以及失手后按速度回到中位。
- **可持久化的校准**：采集张开手的零点姿态，按后端保存并在进程重启后校验。
- **触觉仿真**：120 个触觉点（5 指 × 6 × 4），按力大小从蓝色渐变到红色。
- **桌面场景**：手下方有一张桌子，桌面上放着一个可抓取的自由方块。
  手初始为手心朝下、位于桌面上方；按 `r` 复位方块。
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
| AprilTag | `pupil-apriltags`（官方 AprilTag 3 C 库绑定，随基础依赖安装） |
| HaMeR 后端 | PyTorch 2.0–2.5，建议使用 CUDA |
| GPU | 任意 CUDA GPU；下文数据来自 RTX 4060 Laptop |
| 摄像头 | 内置/普通 UVC 摄像头（手指） + Orbbec Astra 彩色流（tag） |

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
`SHADOWLINK_DATA` 覆盖。

### 4. 验证

```bash
shadowlink check --data-dir D:/models/hamer/_DATA
shadowlink doctor --data-dir D:/models/hamer/_DATA --strict-models
```

`doctor` 会检查 Python/依赖版本、CUDA、模型文件、校准文件以及 MJCF/触觉资产。

## 快速开始

```bash
# 内置摄像头 + HaMeR + Astra tag + MuJoCo viewer
shadowlink run --backend hamer --data-dir D:/models/hamer/_DATA

# 单相机：Plugcam 同时提供手势和 tag 画面
shadowlink run --config configs/plugcam_hand_tag.yaml --backend hamer \
  --data-dir D:/models/hamer/_DATA

# 双手机：IP Webcam 做手势 + Plugcam 做 tag
shadowlink run --config configs/ipwebcam_plugcam.yaml --backend hamer \
  --data-dir D:/models/hamer/_DATA

# 纯 CPU 手指后备
shadowlink run --backend mediapipe --data-dir D:/models/hamer/_DATA

# 回放视频，不打开任何窗口
shadowlink run --source video --camera demo.mp4 --backend hamer \
  --no-show --no-viewer --max-frames 120

# 无摄像头：循环展示 13 个关键帧姿态
shadowlink demo
```

默认相机索引：

| 索引 | 设备 | 角色 |
| --- | --- | --- |
| `0` | Astra Pro HD Camera | tag 相机 |
| `1` | 内置 HD Webcam | 手指相机 |

`configs/plugcam_hand_tag.yaml` 是单相机配置：设置 `capture.source: tag` 后，
`TagService` 的相机会成为唯一画面来源。每一帧原始图像会同时送给
HaMeR/MediaPipe 做手指识别和 AprilTag 做整手位姿，因此需要让手和 tag
都出现在同一画面里。这个模式下 `capture.camera` 不再使用。

这个配置还会用手机补齐 tag 最不稳的两个轴。手机运行
[phyphox](https://phyphox.org) 并打开 **Remote Access**，ShadowLink 通过
HTTP 轮询，不需要额外 Python 依赖。tag 的蓝轴提供 yaw，手机提供 yaw 和
pitch，手机 roll（绕屏幕法向量）的数据故意不使用。把 `phone.host` 改成
phyphox 显示的地址：

```yaml
phone:
  source: phyphox
  host: 192.0.2.10
  port: 8080
  axes: yaw_pitch
  mount_preset: yaw_pitch
```

`mount_preset=yaw_pitch` 会把手机的 yaw/pitch 映射到灵巧手除 yaw 外的两个
轴。手机安装方式不同时可改用 `back`、`in_hand`，或自定义 `mount_w/x/y/z`。
HUD 会显示手机状态和采样率；`g` 仍然开关朝向跟随，`0` 会同时重置 tag
和手机的零点。

`configs/ipwebcam_plugcam.yaml` 是原来的双手机配置：IP Webcam
`http://192.168.1.109:8080/video` 做手势，Plugcam 使用 DirectShow 索引 `1`
做 tag。手机 IP 或相机索引变化时改这两个值即可。

如果 Windows 顺序不同，修改配置文件里的 `capture.camera` / `tag.camera`，
或使用 `--camera` / `--tag-camera`。查看设备列表：

```bash
python -c "from pygrabber.dshow_graph import FilterGraph; print(list(enumerate(FilterGraph().get_input_devices())))"
```

### 按键

| 按键 | 作用 |
| --- | --- |
| `q` / `Esc` | 退出 |
| `c` | 采集手指零点姿态（保持张开手不动） |
| `g` | 开关 tag 整手朝向跟随 |
| `0` | 以当前 tag 位姿重新定义位置和朝向零点 |
| `p` | 开关 tag 整手平移 |
| `m` | 循环切换手机到灵巧手的轴向映射，并自动重新锁零 |

`m` 可以在程序运行时循环 8 种轴向/符号组合。HUD 里的
`phone ... mount <名称>` 会显示当前映射；如果手机 yaw/pitch 的轴互换了或
方向反了，直接按 `m` 试到正确为止。
| `r` | 重置物理状态、目标、tag 零点、平移、朝向和方块 |

### 常用参数

```text
--source camera|video        --camera 索引或路径
--no-position                --position-preset front|palm|back
--position-gain FLOAT        --position-deadzone M
--tag-camera 索引/路径        --tag-api any|dshow|msmf
--tag-family tag36h11        --tag-id N
--tag-size M                 --tag-mirror auto|on|off
--tag-zero-mode sample|fixed --tag-zero-distance M
--tag-show / --no-tag-show   --tag-follow
--backend hamer|mediapipe    --side any|left|right
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
内置摄像头 ──► FramePacket ──► MediaPipe 检测线程 ──► HaMeR 推理线程
                                             │
                                             ▼
                              21 关节 → 弯曲/张开/对掌
                                             ▼
                     One-Euro → 速度限幅 → 20 维执行器目标

Astra 彩色流 ──► AprilTag 线程 ──► 6DoF 相对零点 → 位置 + 朝向
                                             ▼
                     ShadowHand.set_macro_pose（整手基座）
                                             ▼
                     MuJoCo 固定步长物理 + 120 触觉 + viewer/预览
```

关键约定：

- 帧对象是不可变 dataclass，采集缓冲区不会跨线程被修改。
- 感知结果带 `frame_seq`、`capture_ts`、`finished_ts`；主循环只读最新结果。
- tag 服务有独立最新值槽位，主循环从不等待相机或检测；tag 过期时冻结。
- 左手会被镜像规范化后再重定向。v0.1 不驱动左手 Shadow Hand 模型。
- Astra 的 RGB 输出是水平镜像的，`tag.mirror: auto` 会自动翻转。

模块边界和线程模型见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 校准

面对摄像头张开手，按 `c`，或运行：

```bash
shadowlink calibrate --backend hamer --data-dir D:/models/hamer/_DATA
```

零点数据保存在 Windows 的 `%APPDATA%\ShadowLink\calibration.json`，或 Linux 的
`~/.config/shadowlink/calibration.json`。校准与后端和左右手绑定。

## AprilTag 整手 6DoF

- 打印/显示 `src/shadowlink/assets/apriltag/tag36h11_id5.png`，**不要镜像**。
- `size.md` 与配置默认值都是 57 mm（按黑色边框边长）。
- 会单独弹出 **AprilTag Detection** 窗口，显示 Astra 画面、tag 轮廓、坐标轴、
  ID、距离、镜像状态和检测帧率；用 `--no-tag-show` 可以关闭。
- 相机默认 DSHOW、640×480，约 20 FPS；DSHOW 比 Astra 的 MSMF 打开更可靠。
  需要更高帧率可试 `--tag-api msmf`，但打开可能很慢。
- 平移零点默认固定为 **画面中心、深度 1.00 m**
  （`tag.zero_mode: fixed`，`tag.zero_distance_m: 1.0`）。把 tag 放在这个位置
  附近，手的平移偏移才接近零。朝向零点锁存为**第一次检测到有效 tag 时的姿态**，
  整个会话不变；`g` 只开关跟随，不会重设零点。`p` 开关平移，`0` 显式重设朝向
  并清空固定平移零点，`r` 全部重置。如果想用第一次 tag 位置作平移零点，设
  `tag.zero_mode: sample`。
- 初始姿态：**手心朝下**，中指指向方块（世界 +x）。
- tag 坐标轴：**X+** = 向前/中指，**Y+** = 向右，**Z+** = 手心朝向，
  **Z−** = 仿真向上。旋转映射为 **绿轴 → 横滚**、**红轴 → 俯仰**、
  **蓝轴 → 左右转**，三个轴相互独立。
- 平移使用 `overhead` 预设：tag 在画面中**向右**移动 → 机械手朝方块方向
  （世界 +x）移动；tag **靠近**头顶摄像头 → 机械手向上（世界 +z）移动。
- 不再限制机械手的位置范围，tag 移动多远，手就跟随多远。tag 位姿丢失超过
  0.5 秒后，位置和朝向都冻结在原地，不会跳回零。
- 抗噪：检测优先使用 `hamming=0` 且 decision margin 最高的 tag；单帧旋转跳变
  超过 35° 会被拒绝，连续 5 帧都一致才接受。朝向输出使用 0.12 s slerp 和
  3 rad/s 限速。
- tag 应在 tag 相机画面内约 0.5–1.2 m，靠近青色中心/1 m 零点标记，光照均匀。
  太远或太斜会漏检。
- 打印、坐标映射、内参和排错见 [docs/APRILTAG.md](docs/APRILTAG.md)。

## 性能

RTX 4060 Laptop、1200×1799 视频源、30 帧的参考数据：

| 指标 | p50 |
| --- | --- |
| MediaPipe 检测 | 47 ms |
| HaMeR 推理（fp16） | 62.5 ms |
| 消费时的观测年龄 | 约 180 ms |
| 主循环 | 26.9 fps |

检测和推理会跨帧重叠，物理循环按真实时间独立推进。机器较慢时可用
`--skip 2`，或直接用 `--backend mediapipe`。

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
SHADOWLINK_TEST_DATA=D:/models/hamer/_DATA \
SHADOWLINK_TEST_IMAGE=example_data/test5.jpg \
pytest -m hamer -q

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
├── math3d.py             纯 NumPy 四元数工具
├── perception/           MediaPipe 检测器、HaMeR 后端、worker、选择器
├── tag/                  AprilTag 相机线程、相对 6DoF 控制器
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
Apache-2.0；AprilTag 采用 BSD-2-Clause，Python 绑定为 `pupil-apriltags`。
MANO 与 HaMeR checkpoint **不随仓库分发**，仍受各自原始许可约束。详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

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
