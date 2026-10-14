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


@dataclass
class Config:
    fx: float = 900.0
    fy: float = 900.0
    cx: float = 640.0
    cy: float = 360.0
    cam_height_m: float = 1.12
    cam_pitch_deg: float = 12.0
    ground_y_m: float = 0.13

    side: str = "L"
    row_pitch_m: float = 1.2
    row_zmax_m: float = 7.0

    det_roi_top: int = 330

    axis_u: float = 640.0
    harvest_v: float = 540.0
    roi_half_h: int = 70

    scan_frames: int = 120
    select_frames: int = 45
    min_track_age: int = 12
    max_miss: int = 15

    tol_m: float = 0.006
    lock_frames: int = 15
    kp: float = 1.9
    ki: float = 0.25
    kd: float = 0.04
    v_max: float = 0.14
    a_max: float = 0.35
    deadband_m: float = 0.0015


@dataclass
class Det:
    cls: str
    conf: float
    box: tuple


@dataclass
class Plant:
    box: tuple
    conf: float
    root_box: Optional[tuple] = None
    root_conf: float = 0.0
    root_uv: Optional[tuple] = None
    ground: Optional[tuple] = None
    tid: int = -1
    age: int = 0
    miss: int = 0
    row: str = ""


class CeleryDetector:
    NAMES = ("celery", "root")

    def __init__(self, cfg: "Config", weights="weights/celery_root_s.engine", conf=0.35, iou=0.5, imgsz=640):
        from ultralytics import YOLO
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


def pair_plant_root(dets: list[Det]) -> list[Plant]:
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
    for p in plants:
        if p.root_box is not None:
            b = p.root_box
            p.root_uv = ((b[0] + b[2]) / 2, b[3] - 0.2 * (b[3] - b[1]))
        else:
            p.root_uv = ((p.box[0] + p.box[2]) / 2, p.box[3])
    return plants


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
        for d, i, j in pairs:
            if d > self.gate or i in used_t or j in used_p:
                continue
            t, p = self.tracks[i], plants[j]
            p.tid, p.age, p.miss = t.tid, t.age + 1, 0
            a = self.alpha
            p.root_uv = (a * p.root_uv[0] + (1 - a) * t.root_uv[0],
                         a * p.root_uv[1] + (1 - a) * t.root_uv[1])
            used_t.add(i), used_p.add(j)
        for j, p in enumerate(plants):
            if j not in used_p:
                p.tid, p.age = self._next, 1
                self._next += 1
        lost = [t for i, t in enumerate(self.tracks) if i not in used_t and t.miss < self.max_miss]
        for t in lost:
            t.miss += 1
        self.tracks = plants + lost
        return plants


def refine_root(frame_bgr, root_box, pad=4):
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


class GroundProjector:
    def __init__(self, cfg: Config):
        self.c = cfg
        p = math.radians(cfg.cam_pitch_deg)
        self.sinp, self.cosp = math.sin(p), math.cos(p)

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        c = self.c
        du, dv = (u - c.cx) / c.fx, -(v - c.cy) / c.fy
        dy = -self.sinp + dv * self.cosp
        dz = self.cosp + dv * self.sinp
        t = (c.ground_y_m - c.cam_height_m) / dy
        return t * du, t * dz


@dataclass
class Row:
    name: str
    side: str
    rank: int
    n: int
    a: float = 0.0
    b: float = 0.0
    x_ref: float = 0.0
    members: list = field(default_factory=list, repr=False)


def cluster_rows(plants: list[Plant], mid_x: float, cfg: Config) -> list[Row]:
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
        vz, vx, z0, x0 = cv2.fitLine(xz, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        a = float(vx / vz)
        b = float(x0 - a * z0)
        x_ref = a * 3.0 + b
        rows.append(Row("", "L" if x_ref < mid_x else "R", 0, len(g), a, b, x_ref, g))
    for side in ("L", "R"):
        for rank, r in enumerate(sorted((r for r in rows if r.side == side), key=lambda r: abs(r.x_ref - mid_x)), 1):
            r.rank, r.name = rank, f"{side}{rank}"
            for p in r.members:
                p.row = r.name
    return sorted(rows, key=lambda r: r.x_ref)


def choose_row(rows: list[Row], cfg: Config) -> Optional[Row]:
    return next((r for r in rows if r.side == cfg.side and r.rank == 1), None)


def pick_target(plants: list[Plant], row_name: str, cfg: Config) -> Optional[Plant]:
    cands = [p for p in plants if p.row == row_name and p.age >= cfg.min_track_age
             and abs(p.root_uv[1] - cfg.harvest_v) < cfg.roi_half_h]
    return min(cands, key=lambda p: abs(p.root_uv[1] - cfg.harvest_v)) if cands else None


class LateralController:
    def __init__(self, cfg: Config):
        self.c = cfg
        self.reset()

    def reset(self):
        self.i = 0.0
        self.terms = (0.0, 0.0, 0.0)
        self.prev_e = None
        self.d_f = 0.0
        self.v = 0.0

    def step(self, err_m: float, dt: float) -> float:
        c = self.c
        e = 0.0 if abs(err_m) < c.deadband_m else err_m
        d_raw = 0.0 if self.prev_e is None else (e - self.prev_e) / dt
        self.d_f += 0.3 * (d_raw - self.d_f)
        self.prev_e = e
        u = c.kp * e + c.ki * self.i + c.kd * self.d_f
        self.terms = (c.kp * e, c.ki * self.i, c.kd * self.d_f)
        u_sat = float(np.clip(u, -c.v_max, c.v_max))
        if u == u_sat:
            self.i = float(np.clip(self.i + e * dt, -0.05, 0.05))
        dv = float(np.clip(u_sat - self.v, -c.a_max * dt, c.a_max * dt))
        self.v += dv
        return self.v


class State(Enum):
    INIT = "INIT"
    SCAN = "SCAN"
    SELECT = "SELECT"
    ALIGN = "ALIGN"
    LOCKED = "LOCKED"


@dataclass
class Command:
    state: State
    v_mps: float = 0.0
    target: Optional[Plant] = None
    err_m: float = 0.0
    err_px: float = 0.0
    row: Optional[str] = None
    rows: list = field(default_factory=list)
    ready: bool = False


class AlignStateMachine:
    def __init__(self, cfg: Config):
        self.c = cfg
        self.proj = GroundProjector(cfg)
        self.pid = LateralController(cfg)
        self.state = State.INIT
        self.n = 0
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
        gx_axis, _ = self.proj.pixel_to_ground(self.c.axis_u, tgt.root_uv[1])
        return tgt.ground[0] - gx_axis, tgt.root_uv[0] - self.c.axis_u

    def update(self, plants: list[Plant], dt: float, stage_x: float = 0.0) -> Command:
        c = self.c
        self.frame_i += 1
        mid_x = -stage_x
        for p in plants:
            p.ground = self.proj.pixel_to_ground(*p.root_uv)
        cmd = Command(self.state, rows=self.rows, row=self.row)

        if self.state is State.INIT:
            if plants:
                self._goto(State.SCAN, f"{len(plants)} plants in ROI")

        elif self.state is State.SCAN:
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

        elif self.state is State.SELECT:
            self.rows = cluster_rows([p for p in plants if p.age >= 1], mid_x, c)
            tgt = pick_target(plants, self.row, c)
            if tgt is None:
                self.n = 0
            else:
                if tgt.tid != self.tid:
                    self.tid, self.n = tgt.tid, 0
                    self._log(f"target #{tgt.tid:02d} root=({tgt.root_uv[0]:.0f},{tgt.root_uv[1]:.0f})")
                cmd.target = tgt
                cmd.err_m, cmd.err_px = self._lateral_error(tgt)
                self.n += 1
                if self.n >= c.select_frames:
                    self.pid.reset()
                    self._goto(State.ALIGN, "start lateral servo")

        elif self.state in (State.ALIGN, State.LOCKED):
            tgt = next((p for p in plants if p.tid == self.tid), None)
            if tgt is None:
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
                else:
                    self.n = self.n + 1 if abs(err) > 2.5 * c.tol_m else 0
                    cmd.ready = True
                    if self.n > 10:
                        self._goto(State.ALIGN, "drift, re-align")

        cmd.state, cmd.rows, cmd.row = self.state, self.rows, self.row
        if cmd.target is None and self.state in (State.SELECT, State.ALIGN, State.LOCKED):
            cmd.target = next((p for p in plants if p.tid == self.tid), None)
        return cmd


class StageDriver:
    def set_velocity(self, v_mps: float):
        ...

    def position(self) -> float:
        ...

    def notify_ready(self):
        ...


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
        for p in plants:
            if p.root_box is not None:
                p.root_uv = refine_root(frame, p.root_box)
        plants = tracker.update(plants)

        cmd = machine.update(plants, dt, stage.position())
        stage.set_velocity(cmd.v_mps)
        if cmd.ready:
            stage.notify_ready()

        if cv2.waitKey(1) == 27:
            stage.set_velocity(0.0)
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
