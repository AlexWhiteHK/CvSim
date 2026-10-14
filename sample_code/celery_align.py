"""
celery_align.py —— 芹菜收割机「视觉对位」模块（关键部分示例）

任务：相机装在收割机构上，只能沿 X 方向左右平移（Y/Z 方向固定）。
      视觉负责在画面里找到「待收割芹菜的根部」，算出它与割刀中心线的横向偏差，
      驱动平移滑台，直到割刀中心线对准根部，再通知收割机构下刀。

部署：机器左右各一套「相机 + 滑台 + 割刀」，两套各跑一个实例、互不干涉：
      左侧实例 Config.side = "L"，只对位机器中线左侧最近的一行；
      右侧实例 Config.side = "R"，同一份代码，只对位右侧最近的一行。
      田里不止两行，相机视野里会同时出现多行芹菜，所以需要先做「行聚类」再按 side 选行。

处理流程（每帧）：
    摄像头 → YOLO 检测(整株/根部，仅下半部 ROI) → 植株-根部配对 → 根部亚像素精修 → 多目标跟踪
           → 像素→地面坐标 → 行聚类 + 直线拟合 → 按 side 选行、选目标
           → PID 横向控制 → 滑台速度指令 → 对位稳定后输出 READY

说明：本文件只保留思路中的关键环节，检测模型权重、滑台驱动、PLC 通信为接口占位。
      除 CeleryDetector 外，其余模块不依赖深度学习框架，演示视频里的控制曲线就是由
      这里的 AlignStateMachine / LateralController 实际算出来的。
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("celery_align")


# =============================================================================
# 0. 配置（标定参数 + 控制参数）
# =============================================================================
@dataclass
class Config:
    # --- 相机内参 / 安装位姿（离线标定得到，相机只做 X 向平移，所以高度和俯仰是常量）
    fx: float = 900.0
    fy: float = 900.0
    cx: float = 640.0
    cy: float = 360.0
    cam_height_m: float = 1.12
    cam_pitch_deg: float = 12.0
    ground_y_m: float = 0.13            # 垄面离沟底的高度（根部所在平面）

    # --- 本实例负责哪一侧（左相机 "L" / 右相机 "R"）与垄距
    side: str = "L"
    row_pitch_m: float = 1.2            # 相邻两行的间距
    row_zmax_m: float = 7.0             # 行拟合只用近场目标（越远，像素→地面的误差越大）

    # --- 检测只在画面下半部的 ROI 内跑（降低算力，也避开地平线附近的小目标）
    det_roi_top: int = 330

    # --- 割刀与画面的对应关系（标定得到）
    axis_u: float = 640.0               # 割刀中心线所在的图像列
    harvest_v: float = 540.0            # 割刀作业线所在的图像行
    roi_half_h: int = 70                # 选目标的 ROI 半高 (px)

    # --- 状态机
    scan_frames: int = 120              # SCAN：连续稳定观测的帧数（30fps → 4 s）
    select_frames: int = 45             # SELECT：锁定目标后的确认时间
    min_track_age: int = 12             # 轨迹至少存在这么多帧才算“稳定”
    max_miss: int = 15                  # ALIGN 中目标连续丢失多少帧就放弃

    # --- 控制
    tol_m: float = 0.006                # 对位容差 ±6 mm
    lock_frames: int = 15               # 连续满足容差多少帧判定为 LOCKED
    kp: float = 1.9
    ki: float = 0.25
    kd: float = 0.04
    v_max: float = 0.14                 # 滑台最大速度 m/s
    a_max: float = 0.35                 # 滑台最大加速度 m/s²
    deadband_m: float = 0.0015


# =============================================================================
# 1. 数据结构
# =============================================================================
@dataclass
class Det:
    cls: str                            # celery / root
    conf: float
    box: tuple                          # x1, y1, x2, y2 (px)


@dataclass
class Plant:
    """一株芹菜 = 植株框 + 根部框 + 跟踪 ID"""
    box: tuple
    conf: float
    root_box: Optional[tuple] = None
    root_conf: float = 0.0
    root_uv: Optional[tuple] = None     # 根部像素坐标（精修后）
    ground: Optional[tuple] = None      # 根部地面坐标 (X 横向, Z 向前) m，相机坐标系
    tid: int = -1
    age: int = 0
    miss: int = 0
    row: str = ""                       # 所属行，如 "L1"（左侧第 1 行，离机器中线最近）


# =============================================================================
# 2. 检测（YOLO，TensorRT FP16）
# =============================================================================
class CeleryDetector:
    """两类目标：整株芹菜 / 根部(茎基)。权重为自训练模型；只在画面下半部 ROI 内推理。"""

    NAMES = ("celery", "root")

    def __init__(self, cfg: "Config", weights="weights/celery_root_s.engine", conf=0.35, iou=0.5, imgsz=640):
        from ultralytics import YOLO        # 延迟导入：没装深度学习环境时其余模块仍可用
        self.model = YOLO(weights, task="detect")
        self.top = cfg.det_roi_top
        self.conf, self.iou, self.imgsz = conf, iou, imgsz

    def __call__(self, frame_bgr) -> list[Det]:
        roi = frame_bgr[self.top:]
        r = self.model.predict(roi, imgsz=self.imgsz, conf=self.conf,
                               iou=self.iou, half=True, verbose=False)[0]
        return [Det(self.NAMES[int(c)], float(p), (b[0], b[1] + self.top, b[2], b[3] + self.top))
                for b, p, c in zip(r.boxes.xyxy.cpu().numpy().tolist(),
                                   r.boxes.conf.cpu().numpy().tolist(),
                                   r.boxes.cls.cpu().numpy())]


# =============================================================================
# 3. 植株 ↔ 根部配对
# =============================================================================
def pair_plant_root(dets: list[Det]) -> list[Plant]:
    """根部框的中心必须落在某个植株框的下 35% 内；被遮挡时前方植株的框也可能包住后方的根，
    因此在候选里取「框底边离根部中心最近」的那株。"""
    plants = [Plant(d.box, d.conf) for d in dets if d.cls == "celery"]
    for r in (d for d in dets if d.cls == "root"):
        rx, ry = (r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2
        cands = []
        for p in plants:
            h = p.box[3] - p.box[1]
            if p.box[0] <= rx <= p.box[2] and p.box[3] - 0.35 * h <= ry <= p.box[3] + 0.08 * h:
                cands.append(p)
        if cands:
            best = min(cands, key=lambda p: abs(p.box[3] - ry))
            if best.root_box is None:
                best.root_box, best.root_conf = r.box, r.conf
    for p in plants:                    # 根部点取根部框底边附近（入土点）；没检到根部框的退化为植株框底边中点
        if p.root_box is not None:
            b = p.root_box
            p.root_uv = ((b[0] + b[2]) / 2, b[3] - 0.2 * (b[3] - b[1]))
        else:
            p.root_uv = ((p.box[0] + p.box[2]) / 2, p.box[3])
    return plants


# =============================================================================
# 4. 多目标跟踪（植株数量少、运动只有平移，最近邻 + 距离门限足够；也可换 ByteTrack）
# =============================================================================
class Tracker:
    def __init__(self, gate_px=45.0, max_miss=8, alpha=0.6):
        self.tracks: list[Plant] = []
        self.gate, self.max_miss, self.alpha = gate_px, max_miss, alpha
        self._next = 1

    def update(self, plants: list[Plant]) -> list[Plant]:
        pairs = sorted(((math.dist(t.root_uv, p.root_uv), i, j)
                        for i, t in enumerate(self.tracks) for j, p in enumerate(plants)),
                       key=lambda x: x[0])
        used_t, used_p = set(), set()
        for d, i, j in pairs:                         # 贪心匹配
            if d > self.gate or i in used_t or j in used_p:
                continue
            t, p = self.tracks[i], plants[j]
            p.tid, p.age, p.miss = t.tid, t.age + 1, 0
            a = self.alpha                            # 根部点做指数平滑，抑制检测框抖动
            p.root_uv = (a * p.root_uv[0] + (1 - a) * t.root_uv[0],
                         a * p.root_uv[1] + (1 - a) * t.root_uv[1])
            used_t.add(i), used_p.add(j)
        for j, p in enumerate(plants):                # 新目标
            if j not in used_p:
                p.tid, p.age = self._next, 1
                self._next += 1
        lost = [t for i, t in enumerate(self.tracks) if i not in used_t and t.miss < self.max_miss]
        for t in lost:
            t.miss += 1
        self.tracks = plants + lost
        return plants


# =============================================================================
# 5. 根部亚像素精修（可选，需要图像）
# =============================================================================
def refine_root(frame_bgr, root_box, pad=4):
    """YOLO 框中心会随姿态漂移几个像素，而对位要的是毫米级精度：
    在根部框内分割「淡绿/白色茎基」，取其最底部几行像素的中点作为入土点。"""
    x1, y1, x2, y2 = (int(v) for v in root_box)
    x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    roi = frame_bgr[y1:y2 + pad, x1:x2 + pad]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 20, 120), (55, 160, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (x1 + x2) / 2, y2
    c = max(cnts, key=cv2.contourArea)[:, 0, :]
    bottom = c[c[:, 1] >= c[:, 1].max() - 3]
    return x1 + float(bottom[:, 0].mean()), y1 + float(c[:, 1].max())


# =============================================================================
# 6. 像素 → 地面坐标（针孔模型 + 已知相机高度/俯仰，把射线投到根部所在平面）
# =============================================================================
class GroundProjector:
    def __init__(self, cfg: Config):
        self.c = cfg
        p = math.radians(cfg.cam_pitch_deg)
        self.sinp, self.cosp = math.sin(p), math.cos(p)

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        c = self.c
        du, dv = (u - c.cx) / c.fx, -(v - c.cy) / c.fy
        dy = -self.sinp + dv * self.cosp                 # 射线方向（水平系：x 右 / y 上 / z 前）
        dz = self.cosp + dv * self.sinp
        t = (c.ground_y_m - c.cam_height_m) / dy         # 射线与根部平面相交
        return t * du, t * dz


# =============================================================================
# 7. 行聚类 + 按 side 选行 + 选目标
# =============================================================================
@dataclass
class Row:
    name: str                            # "L1" = 左侧第 1 行（离机器中线最近）
    side: str
    rank: int
    n: int
    a: float = 0.0                       # 行方向： X = a·Z + b（相机坐标系）
    b: float = 0.0
    x_ref: float = 0.0                   # Z = 3 m 处的横向位置
    members: list = field(default_factory=list, repr=False)   # 属于这一行的植株


def cluster_rows(plants: list[Plant], mid_x: float, cfg: Config) -> list[Row]:
    """相机视野里同时有很多行：把近场根部点按横向位置聚类（行距 1.2 m，间隙 > 0.6 m 即分行），
    每行用鲁棒直线拟合，再按相对机器中线(mid_x)分左右、按距离排序编号。
    mid_x：两行中线在相机坐标系下的 X（= -滑台编码器读数）。"""
    pts = [p for p in plants if p.ground is not None and 1.2 < p.ground[1] < cfg.row_zmax_m]
    pts.sort(key=lambda p: p.ground[0])
    groups: list[list[Plant]] = []
    for p in pts:
        if groups and p.ground[0] - groups[-1][-1].ground[0] < 0.5 * cfg.row_pitch_m:
            groups[-1].append(p)
        else:
            groups.append([p])
    rows = []
    for g in (g for g in groups if len(g) >= 3):
        xz = np.array([[p.ground[1], p.ground[0]] for p in g], np.float32)
        vz, vx, z0, x0 = cv2.fitLine(xz, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()   # 抗个别误检
        a = float(vx / vz)
        b = float(x0 - a * z0)
        x_ref = a * 3.0 + b
        rows.append(Row("", "L" if x_ref < mid_x else "R", 0, len(g), a, b, x_ref, g))
    for side in ("L", "R"):              # 按离中线的距离编号：L1, L2, ... / R1, R2, ...
        for rank, r in enumerate(sorted((r for r in rows if r.side == side), key=lambda r: abs(r.x_ref - mid_x)), 1):
            r.rank, r.name = rank, f"{side}{rank}"
            for p in r.members:
                p.row = r.name
    return sorted(rows, key=lambda r: r.x_ref)


def choose_row(rows: list[Row], cfg: Config) -> Optional[Row]:
    """左实例选 L1，右实例选 R1：本相机只负责自己这一侧最近的一行。"""
    return next((r for r in rows if r.side == cfg.side and r.rank == 1), None)


def pick_target(plants: list[Plant], row_name: str, cfg: Config) -> Optional[Plant]:
    """该行里：轨迹稳定、根部落在 ROI 内、且最接近割刀作业线的那一株（收割前沿的第一株）"""
    cands = [p for p in plants if p.row == row_name and p.age >= cfg.min_track_age
             and abs(p.root_uv[1] - cfg.harvest_v) < cfg.roi_half_h]
    return min(cands, key=lambda p: abs(p.root_uv[1] - cfg.harvest_v)) if cands else None


# =============================================================================
# 8. 横向 PID（带死区、积分限幅/抗饱和、速度与加速度限幅）
# =============================================================================
class LateralController:
    def __init__(self, cfg: Config):
        self.c = cfg
        self.reset()

    def reset(self):
        self.i = 0.0
        self.terms = (0.0, 0.0, 0.0)                     # P / I / D 三项（用于显示）
        self.prev_e = None
        self.d_f = 0.0
        self.v = 0.0

    def step(self, err_m: float, dt: float) -> float:
        c = self.c
        e = 0.0 if abs(err_m) < c.deadband_m else err_m
        d_raw = 0.0 if self.prev_e is None else (e - self.prev_e) / dt
        self.d_f += 0.3 * (d_raw - self.d_f)             # 微分低通
        self.prev_e = e
        u = c.kp * e + c.ki * self.i + c.kd * self.d_f
        self.terms = (c.kp * e, c.ki * self.i, c.kd * self.d_f)
        u_sat = float(np.clip(u, -c.v_max, c.v_max))
        if u == u_sat:                                   # 未饱和才累积积分（抗积分饱和）
            self.i = float(np.clip(self.i + e * dt, -0.05, 0.05))
        dv = float(np.clip(u_sat - self.v, -c.a_max * dt, c.a_max * dt))
        self.v += dv
        return self.v


# =============================================================================
# 9. 状态机：INIT → SCAN → SELECT → ALIGN → LOCKED
# =============================================================================
class State(Enum):
    INIT = "INIT"
    SCAN = "SCAN"
    SELECT = "SELECT"
    ALIGN = "ALIGN"
    LOCKED = "LOCKED"


@dataclass
class Command:
    state: State
    v_mps: float = 0.0                   # >0 向右平移，<0 向左平移
    target: Optional[Plant] = None
    err_m: float = 0.0                   # 根部相对割刀中心线的横向偏差（>0 根部在右侧）
    err_px: float = 0.0
    row: Optional[str] = None
    rows: list = field(default_factory=list)
    ready: bool = False                  # LOCKED 稳定后通知收割机构下刀


class AlignStateMachine:
    def __init__(self, cfg: Config):
        self.c = cfg
        self.proj = GroundProjector(cfg)
        self.pid = LateralController(cfg)
        self.state = State.INIT
        self.n = 0                       # 当前状态内的计数器
        self.row: Optional[str] = None
        self.rows: list[Row] = []
        self.tid = -1
        self.events: list[tuple[int, str]] = []
        self.frame_i = 0

    def _log(self, msg):
        self.events.append((self.frame_i, msg))
        log.info(msg)

    def _goto(self, s: State, msg: str = ""):
        self.state, self.n = s, 0
        self._log(f"{s.value}: {msg}" if msg else s.value)

    def _lateral_error(self, tgt: Plant) -> tuple[float, float]:
        """根部相对割刀中心线的横向偏差：(米, 像素)。>0 表示根部在割刀线右侧。"""
        gx_axis, _ = self.proj.pixel_to_ground(self.c.axis_u, tgt.root_uv[1])   # 同一深度处割刀线的 X
        return tgt.ground[0] - gx_axis, tgt.root_uv[0] - self.c.axis_u

    def update(self, plants: list[Plant], dt: float, stage_x: float = 0.0) -> Command:
        """stage_x：滑台编码器读数 (m，相对初始位置，向右为正)"""
        c = self.c
        self.frame_i += 1
        mid_x = -stage_x                                         # 两行中线在相机系下的 X
        for p in plants:                                         # 根部像素 → 地面坐标
            p.ground = self.proj.pixel_to_ground(*p.root_uv)
        cmd = Command(self.state, rows=self.rows, row=self.row)

        if self.state is State.INIT:                             # 等检测器预热、出第一批目标
            if plants:
                self._goto(State.SCAN, f"{len(plants)} plants in ROI")

        elif self.state is State.SCAN:                           # 观测一段时间：行聚类 + 直线拟合
            stable = [p for p in plants if p.age >= c.min_track_age]
            self.rows = cluster_rows(stable, mid_x, c)
            self.n = self.n + 1 if len(self.rows) >= 2 else 0
            if self.n >= c.scan_frames:
                near = [r for r in self.rows if r.rank == 1]
                self._log(f"rows={len(self.rows)}  " + "  ".join(f"{r.name} {r.x_ref - mid_x:+.2f}m" for r in near))
                row = choose_row(self.rows, c)
                if row:
                    self.row = row.name
                    self._goto(State.SELECT, f"unit={c.side} row={row.name}")
                else:
                    self.n = 0

        elif self.state is State.SELECT:                         # 在本侧最近一行里选目标并锁定 track id
            self.rows = cluster_rows([p for p in plants if p.age >= 1], mid_x, c)
            tgt = pick_target(plants, self.row, c)
            if tgt is None:
                self.n = 0
            else:
                if tgt.tid != self.tid:
                    self.tid, self.n = tgt.tid, 0
                    self._log(f"target #{tgt.tid:02d} root=({tgt.root_uv[0]:.0f},{tgt.root_uv[1]:.0f})")
                cmd.target = tgt                                 # 先把偏差算出来给界面提示方位
                cmd.err_m, cmd.err_px = self._lateral_error(tgt)
                self.n += 1
                if self.n >= c.select_frames:
                    self.pid.reset()
                    self._goto(State.ALIGN, "start lateral servo")

        elif self.state in (State.ALIGN, State.LOCKED):
            tgt = next((p for p in plants if p.tid == self.tid), None)
            if tgt is None:                                      # 目标丢失：滑台先刹停，超时回 SCAN
                self.n += 1
                self.pid.reset()
                if self.n > c.max_miss:
                    self._goto(State.SCAN, "target lost")
            else:
                err, cmd.err_px = self._lateral_error(tgt)
                cmd.target, cmd.err_m = tgt, err
                if self.state is State.ALIGN:
                    cmd.v_mps = self.pid.step(err, dt)
                    self.n = self.n + 1 if abs(err) < c.tol_m else 0
                    if self.n >= c.lock_frames:
                        self.pid.reset()
                        self._goto(State.LOCKED, f"err={err * 1000:+.1f} mm -> READY")
                else:                                            # LOCKED：保持，漂出 2.5 倍容差则重新对位
                    self.n = self.n + 1 if abs(err) > 2.5 * c.tol_m else 0
                    cmd.ready = True
                    if self.n > 10:
                        self._goto(State.ALIGN, "drift, re-align")

        cmd.state, cmd.rows, cmd.row = self.state, self.rows, self.row
        if cmd.target is None and self.state in (State.SELECT, State.ALIGN, State.LOCKED):
            cmd.target = next((p for p in plants if p.tid == self.tid), None)
        return cmd


# =============================================================================
# 10. 执行机构接口（占位）
# =============================================================================
class StageDriver:
    """横向滑台：伺服/步进驱动器，Modbus-RTU 或 CAN 速度模式。"""

    def set_velocity(self, v_mps: float):
        ...                              # 例：写速度寄存器 rpm = v / lead * 60

    def position(self) -> float:
        ...                              # 例：读编码器，换算成相对初始位置的 X (m)，向右为正

    def notify_ready(self):
        ...                              # 例：置位 PLC 的 ALIGN_OK 信号，收割机构下刀


# =============================================================================
# 11. 主循环
# =============================================================================
def run(cfg: Config = Config(), cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector, tracker = CeleryDetector(cfg), Tracker()
    machine, stage = AlignStateMachine(cfg), StageDriver()

    t_prev = time.monotonic()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        dt, t_prev = max(now - t_prev, 1e-3), now

        plants = pair_plant_root(detector(frame))
        for p in plants:                 # 先精修根部点，再进跟踪器（平滑的是精修后的点）
            if p.root_box is not None:
                p.root_uv = refine_root(frame, p.root_box)
        plants = tracker.update(plants)

        cmd = machine.update(plants, dt, stage.position())
        stage.set_velocity(cmd.v_mps)
        if cmd.ready:
            stage.notify_ready()

        if cv2.waitKey(1) == 27:         # ESC
            stage.set_velocity(0.0)
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
