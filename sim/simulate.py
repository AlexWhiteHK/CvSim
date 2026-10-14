"""
simulate.py —— 闭环仿真：3D 场景几何 → 合成「检测器输出」→ celery_align 的真实控制逻辑 → 滑台动力学

先把整段轨迹（相机位置、检测、状态、误差…）算出来，再按轨迹渲染画面。
检测框由几何投影 + 噪声合成（模拟 YOLO 输出），其后的配对、跟踪、行聚类、选目标、PID、状态机
全部是 sample_code/celery_align.py 里的真实代码。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sample_code"))
sys.path.insert(0, HERE)

import celery_align as ca                    # noqa: E402
import render3d as r3                        # noqa: E402

FPS = 30
WARMUP_FRAMES = 36                           # 检测器预热：前 ~1.2 s 没有输出
STAGE_TAU = 0.10                             # 滑台速度环一阶滞后 (s)
DET_MAX_DEPTH = 8.5


def make_config(layout, projector):
    """按场景几何标定：割刀作业线 = 目标植株根部所在的图像行（相机只做横向平移，该行不变）"""
    cfg = ca.Config(fx=r3.FOCAL, fy=r3.FOCAL, cx=r3.CX0, cy=r3.CY0, cam_height_m=r3.CAM_H,
                    cam_pitch_deg=12.0, ground_y_m=r3.RIDGE_H, side="L", axis_u=r3.CX0)
    row = min(layout["rows"], key=lambda x: abs(x + r3.ROW_X))      # x = -0.6
    p = next(p for p in layout["plants"] if abs(p["x"] - row) < 0.05 and p["idx"] == 0)
    _, v, _ = r3.project(np.array([p["x"], p["y"], p["z"]]), 0.0)
    cfg.harvest_v = float(round(float(v)))
    return cfg


def _clip_box(b, w=r3.W, h=r3.H):
    return (max(b[0], 0.0), max(b[1], 0.0), min(b[2], w - 1.0), min(b[3], h - 1.0))


def synth_detections(infos, rng, cfg):
    """几何投影 + 噪声 → 模拟 YOLO 输出（植株框 / 根部框 / 置信度 / 偶发漏检 / 遮挡）。只在检测 ROI 内输出。"""
    cand = []
    for k in infos:
        x1, y1, x2, y2 = k["box"]
        if k["depth"] > DET_MAX_DEPTH or k["root"][1] < cfg.det_roi_top or k["root"][1] > r3.H + 30:
            continue
        cb = _clip_box(k["box"])
        w, h = x2 - x1, y2 - y1
        vis = max(cb[2] - cb[0], 0) * max(cb[3] - cb[1], 0) / max(w * h, 1)
        if vis < 0.5 or cb[3] - cb[1] < 20:
            continue
        cand.append((k, cb, w, h))
    dets = []
    for k, cb, w, h in cand:
        d = k["depth"]
        if d > 4.5 and rng.random() < 0.025:                 # 远一点的偶发漏检
            continue
        # 几乎被前方植株完全遮挡的，检测器报不出来
        area = max((cb[2] - cb[0]) * (cb[3] - cb[1]), 1.0)
        cover = 0.0
        for q, qcb, _, _ in cand:
            if q["depth"] < d - 0.1:
                iw = min(cb[2], qcb[2]) - max(cb[0], qcb[0])
                ih = min(cb[3], qcb[3]) - max(cb[1], qcb[1])
                if iw > 0 and ih > 0:
                    cover = max(cover, iw * ih / area)
        if cover > 0.92:
            continue
        j = 0.008 * max(w, h) + 0.6
        box = tuple(float(v + rng.normal(0, j)) for v in cb)
        conf = float(np.clip(0.975 - 0.012 * d + rng.normal(0, 0.012), 0.55, 0.995))
        dets.append(ca.Det("celery", conf, box))
        # 根部：被前方植株挡住时检不到
        ru, rv = k["root"]
        hidden = False
        for q, qcb, qw, qh in cand:
            if q["depth"] < d - 0.15:
                qx1, qy1, qx2, qy2 = q["box"]
                inset = 0.18 * qw
                if qx1 + inset < ru < qx2 - inset and qy1 < rv < qy1 + 0.88 * (qy2 - qy1):
                    hidden = True
                    break
        rb = k["root_box"]
        if not hidden and 0 <= (rb[0] + rb[2]) / 2 < r3.W and rb[3] < r3.H + 4:
            jr = 0.7 + 0.01 * (rb[2] - rb[0])
            box_r = tuple(float(v + rng.normal(0, jr)) for v in rb)
            conf_r = float(np.clip(0.955 - 0.014 * d + rng.normal(0, 0.012), 0.5, 0.99))
            dets.append(ca.Det("root", conf_r, box_r))
    return dets


def simulate(layout, projector, hold_s=7.0, seed=3, max_s=40.0):
    cfg = make_config(layout, projector)
    rng = np.random.default_rng(seed)
    tracker, machine = ca.Tracker(), ca.AlignStateMachine(cfg)
    dt = 1.0 / FPS
    cam_x, v_act = 0.0, 0.0
    frames = []
    t_lock = None
    i = 0
    while i < max_s * FPS:
        infos = projector.screen_infos(cam_x)
        dets = synth_detections(infos, rng, cfg) if i >= WARMUP_FRAMES else []
        plants = tracker.update(ca.pair_plant_root(dets))
        stage_x = cam_x + rng.normal(0, 0.0003)       # 编码器读数（带微小噪声）
        cmd = machine.update(plants, dt, stage_x)
        stable = [p for p in plants if p.age >= cfg.min_track_age]
        rows_vis = ca.cluster_rows(stable, -stage_x, cfg) if stable else []     # 仅用于显示：每帧重算行聚类
        frames.append(dict(
            i=i, t=i * dt, cam_x=cam_x, v_cmd=cmd.v_mps, v_act=v_act, state=cmd.state.value,
            err_m=cmd.err_m, err_px=cmd.err_px, tid=machine.tid, row=machine.row,
            ready=cmd.ready, pid=machine.pid.terms,
            plants=[dict(tid=p.tid, box=p.box, conf=p.conf, root_box=p.root_box, root_conf=p.root_conf,
                         root_uv=p.root_uv, ground=p.ground, age=p.age, row=p.row) for p in plants],
            rows=[dict(name=r.name, side=r.side, rank=r.rank, n=r.n, a=r.a, b=r.b, x_ref=r.x_ref) for r in rows_vis],
            n_dets=(sum(d.cls == "celery" for d in dets), sum(d.cls == "root" for d in dets)),
            target=(dict(tid=cmd.target.tid, conf=cmd.target.conf, root_uv=cmd.target.root_uv,
                         root_conf=cmd.target.root_conf, ground=cmd.target.ground, age=cmd.target.age)
                    if cmd.target else None),
        ))
        v_act += (cmd.v_mps - v_act) * dt / STAGE_TAU
        cam_x += v_act * dt
        if machine.state is ca.State.LOCKED and t_lock is None:
            t_lock = i
        if t_lock is not None and i - t_lock > hold_s * FPS:
            break
        i += 1
    return dict(frames=frames, events=machine.events, cfg=cfg, fps=FPS, t_lock=t_lock)
