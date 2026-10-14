"""
make_video.py —— 生成演示视频（单画面：相机画面 + HUD）。

    python sim/fetch_assets.py                        # 首次：下载 CC0 素材（土壤贴图 + 农田 HDRI）
    python sim/make_video.py                          # 渲染整段视频 → video/celery_align_demo.mp4
    python sim/make_video.py --stills 100,300,500     # 只导出指定帧 PNG（调试画面）

流程：布局 → 闭环仿真（轨迹/检测/状态）→ three.js 渲染（相机位置相同的帧只渲一次，结果缓存）
      → 相机成像效果 → HUD 叠加 → ffmpeg 编码。
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import hud                                   # noqa: E402
import postfx                                # noqa: E402
import render3d as r3                        # noqa: E402
import simulate                              # noqa: E402

CACHE = os.path.join(HERE, ".cache")
_R = None


def _worker_init(layout):
    global _R
    _R = r3.Renderer(layout)


def _worker_render(job):
    key, cam_x = job
    return key, _R.render(cam_x)


def cam_key(x):
    return int(round(x * 5000))          # 0.2 mm 量化：静止/几乎静止的帧共用同一次渲染


def build_meta(sim):
    fps = sim["fps"]
    log = [(0, "camera /dev/video0 1280x720 30fps"), (5, "detector celery_root_s.engine loaded"),
           (24, "warm-up 3 iters ok 14.2ms")]
    for f, m in sim["events"]:
        log.append((f, m.replace("rows:", "rows").replace("SCAN: ", "SCAN ").replace("SELECT: ", "SELECT ")
                    .replace("ALIGN: ", "ALIGN ").replace("LOCKED: ", "LOCKED ")))
        if m.startswith("LOCKED"):
            log.append((f + 3, "ALIGN_OK -> harvester PLC"))
    log.sort(key=lambda x: x[0])

    def clock(t):
        s = 12.0 + t
        return f"2026-10-13 09:41:{int(s):02d}.{int((s % 1) * 1000):03d}"

    return dict(cfg=sim["cfg"], fps=fps, log=log, clock=clock, project=r3.project)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stills", default="")
    ap.add_argument("--stills-dir", default="/tmp/stills")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "video", "celery_align_demo.mp4"))
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--hold", type=float, default=9.0)
    ap.add_argument("--ss", type=float, default=None, help="渲染超采样倍数（默认见 render3d.build_layout）")
    args = ap.parse_args()

    layout = r3.build_layout()
    if args.ss:
        layout["render"]["ss"] = args.ss
    tag = hashlib.md5((json.dumps(layout, sort_keys=True) + open(os.path.join(HERE, "web", "scene.js")).read()).encode()).hexdigest()[:10]
    cdir = os.path.join(CACHE, tag)
    os.makedirs(cdir, exist_ok=True)

    # 1) 取变体顶点采样 → 仿真
    vp_file = os.path.join(CACHE, f"variant_pts_{tag}.json")
    if os.path.exists(vp_file):
        vp = json.load(open(vp_file))
    else:
        lay_small = json.loads(json.dumps(layout))
        lay_small["render"]["ss"] = 0.4
        rr = r3.Renderer(lay_small)
        vp = rr.variant_pts
        rr.close()
        json.dump(vp, open(vp_file, "w"))
    proj = r3.PlantProjector(layout, vp)
    sim = simulate.simulate(layout, proj, hold_s=args.hold)
    meta = build_meta(sim)
    frames = sim["frames"]
    n = len(frames)
    print(f"simulated {n} frames ({n / sim['fps']:.1f}s)", flush=True)
    for f, m in sim["events"]:
        print(f"  [{f / sim['fps']:6.2f}] {m}", flush=True)

    # 2) 渲染需要的唯一相机位置（带磁盘缓存）
    sel = list(range(n))
    if args.stills:
        sel = [min(int(s), n - 1) for s in args.stills.split(",")]
    need = {}
    for i in sel:
        k = cam_key(frames[i]["cam_x"])
        need.setdefault(k, k / 5000.0)
    todo = [(k, x) for k, x in sorted(need.items()) if not os.path.exists(os.path.join(cdir, f"{k}.png"))]
    print(f"unique camera positions: {len(need)}, to render: {len(todo)}", flush=True)
    if todo:
        t0 = time.time()
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.procs, initializer=_worker_init, initargs=(layout,)) as pool:
            for j, (k, img) in enumerate(pool.imap_unordered(_worker_render, todo)):
                cv2.imwrite(os.path.join(cdir, f"{k}.png"), img[..., ::-1])
                if j % 10 == 0:
                    el = time.time() - t0
                    print(f"  rendered {j + 1}/{len(todo)}  elapsed {el:.0f}s  eta {el / (j + 1) * (len(todo) - j - 1):.0f}s", flush=True)

    # 3) 合成：相机效果 + HUD
    imgs = {}

    def base(k):
        if k not in imgs:
            imgs[k] = cv2.imread(os.path.join(cdir, f"{k}.png"))[..., ::-1].copy()
        return imgs[k]

    def make_frame(i):
        fd = frames[i]
        rng = np.random.default_rng(900 + i)
        fd["inf_ms"] = float(np.clip(rng.normal(14.1, 0.5), 12.5, 16.5))
        cam = postfx.apply(base(cam_key(fd["cam_x"])), i, fd["t"], fd["v_act"])
        return hud.compose(cam, fd, meta)

    if args.stills:
        import PIL.Image
        os.makedirs(args.stills_dir, exist_ok=True)
        for i in sel:
            PIL.Image.fromarray(make_frame(i)).save(os.path.join(args.stills_dir, f"f{i:04d}.png"))
            print("still", i, flush=True)
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{r3.W}x{r3.H}",
         "-r", str(sim["fps"]), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], stdin=subprocess.PIPE)
    for i in range(n):
        ff.stdin.write(make_frame(i).tobytes())
        if i % 60 == 0:
            print(f"encode {i}/{n}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("done ->", args.out)


if __name__ == "__main__":
    main()
