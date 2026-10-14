# CvSim —— 芹菜收割机视觉对位演示

相机装在收割机构上，**只能左右平移**。视觉在画面里找到待收割芹菜的**根部**，算出它和割刀中心线的横向偏差，驱动平移滑台，直到割刀对准根部。

本仓库交付两样东西：

| 内容 | 路径 |
|---|---|
| 演示视频（仿真相机主视角 + HUD，约 22 s） | `video/celery_align_demo.mp4` |
| 关键部分示例代码（检测→配对→跟踪→行聚类→选目标→PID→状态机） | `sample_code/celery_align.py`（带注释） |
| 同一份代码的无注释版（与带注释版逐行等价，仅去掉注释和文档字符串） | `sample_code/celery_align_nocomment.py` |

仓库里没有任何 GitHub Actions / 定时任务 / 自动监听。

## 视频里发生了什么

场景：一块芹菜田，共 18 行（视野里能看到约 6 行），所有行长势一致；收割前沿以前的芹菜已收割，所以每一行最近的一株就是待收割目标。
画面是**左侧相机**，初始位置在最近两行（L1 / R1）的中间。

| 阶段 | 画面 / 状态 | 代码里对应 |
|---|---|---|
| 0–1.2 s | `INIT` 检测器预热 | — |
| 1.2–5.6 s | `SCAN` 出检测框，下半部 ROI 内检测；根部点投到地面坐标，**行聚类**得到 L3 L2 L1 R1 R2 R3 | `cluster_rows` |
| 5.6–7.1 s | `SELECT` 本相机只负责**自己这一侧最近的一行（L1）**，在该行选根部最接近割刀作业线的一株，红框锁定，HUD 给出 `dx` 和方位（LEFT） | `choose_row` / `pick_target` |
| 7.1–12.6 s | `ALIGN` PID 驱动滑台向左平移，画面视差真实变化，`dX` 逐渐归零 | `LateralController` |
| 12.6 s 起 | `LOCKED` 偏差进入 ±6 mm 并保持，框变绿，输出 `READY` 给收割机构 | `AlignStateMachine` |

**右相机**用同一份代码，只把 `Config.side` 改成 `"R"`，就会对位右侧最近的一行（R1），两套互不干涉。本演示只做左侧。

画面上的 HUD 元素（全部直接叠在相机画面上）：顶栏（相机/帧率/推理耗时/状态/时间）、检测框（`celery` 绿 / `root` 黄，带置信度和跟踪 ID）、DET ROI 虚线、割刀基准线（青色竖线 + 十字）、目标红框 + 根部圆圈、从割刀线指向根部的方位箭头与 `dx`、左下信息块、底部对位指示条、右下事件日志。

## 这个视频是怎么做出来的

- **场景**：three.js（第三方渲染引擎）+ 无头 Chromium 软件渲染。芹菜由程序化建模：带凹槽截面和肋纹的叶柄、分枝叶柄、锯齿小叶（alpha 贴图），按远近分 4 级细节（LOD）实例化；土壤用 Poly Haven 的 PBR 贴图，天空/地平线/环境光用 Poly Haven 的农田 HDRI，另有滴灌带、杂草、阴影与大气雾。
- **检测框**：由 3D 几何投影加噪声合成（模拟 YOLO 输出，含偶发漏检和遮挡），**不是**真的跑了模型。
- **控制**：检测之后的配对、跟踪、行聚类、选目标、PID、状态机都是 `sample_code/celery_align.py` 里的**真实代码**在逐帧运行，相机平移轨迹就是它算出来的。
- **相机成像**：软化/锐化、暗角、传感器噪声、曝光漂移、机体抖动、叶片被微风吹动。

## 重新生成视频

```bash
pip install numpy opencv-python-headless pillow playwright
# 需要 Chromium（playwright install chromium，或使用系统已有的 Chromium）和 ffmpeg
python sim/fetch_assets.py          # 下载 CC0 素材：土壤贴图 + 农田 HDRI（约 33 MB，不入库）
python sim/make_video.py            # 渲染 → video/celery_align_demo.mp4（软件渲染较慢，约 20–40 分钟）
python sim/make_video.py --stills 100,300,520   # 只导出几帧 PNG 调画面
```

渲染结果按相机位置缓存在 `sim/.cache/`，只改 HUD 或后处理时重新合成很快。

## 目录

```
sample_code/celery_align.py   关键部分示例代码（给人讲的那份）
sim/render3d.py               场景布局 + three.js 渲染接口 + 植株屏幕投影
sim/web/                      three.js 场景（index.html / scene.js），vendor/ 为 three.js r170（MIT）
sim/simulate.py               闭环仿真：合成检测 → 真实控制代码 → 滑台动力学
sim/hud.py  sim/postfx.py     HUD 叠加 / 相机成像效果
sim/make_video.py             视频流水线
sim/fetch_assets.py           下载第三方 CC0 素材
video/                        成品视频
```

## 素材与协议

- 土壤 PBR 贴图 "Brown Mud 02"、HDRI "Farm Field"：[Poly Haven](https://polyhaven.com)，CC0。
- three.js：MIT。
