"""
render3d.py —— 场景布局 + three.js（无头 Chromium / SwiftShader）渲染接口 + 屏幕投影。

坐标系（Python 侧）：x 向右，y 向上，z 向前（远离相机），单位 m。
相机固定高度/俯仰，只沿 x 平移——和真实机械结构一致。
"""
import base64
import functools
import http.server
import math
import os
import socketserver
import threading

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "web")

# ------------------------------------------------------------------ 相机 / 场景常量
W, H = 1280, 720
FOCAL = 900.0
CX0, CY0 = W / 2.0, H / 2.0
CAM_H = 1.12
PITCH = math.radians(12.0)
SINP, COSP = math.sin(PITCH), math.cos(PITCH)

ROW_X = 0.60
RIDGE_PERIOD = 1.2
RIDGE_H = 0.13
PLANT_SPACING = 0.34
Z_FIRST = 2.20                   # 收割前沿：此前的芹菜已收割
N_VARIANTS = 6
N_ROW_PAIRS = 9                  # 每侧 9 行 → 共 18 行（只有中间两侧最近的行在近景）
Z_MAX = 52.0


def ridge_h(x):
    th = (np.asarray(x, dtype=np.float64) - ROW_X) * (2 * np.pi / RIDGE_PERIOD)
    return RIDGE_H * (0.5 + 0.5 * np.tanh(2.0 * np.cos(th)))


def project(P, cam_x):
    """世界坐标 (...,3) → 像素 u, v 与沿光轴深度"""
    P = np.asarray(P, dtype=np.float64)
    dx = P[..., 0] - cam_x
    dy = P[..., 1] - CAM_H
    dz = P[..., 2]
    depth = -dy * SINP + dz * COSP
    up = dy * COSP + dz * SINP
    d = np.maximum(depth, 1e-3)
    return CX0 + FOCAL * dx / d, CY0 - FOCAL * up / d, depth


# ------------------------------------------------------------------ 布局
def build_layout(seed=2024):
    rng = np.random.default_rng(seed)
    plants = []
    row_xs = []
    for k in range(N_ROW_PAIRS):
        for sgn in (-1, 1):
            row_xs.append(sgn * (ROW_X + RIDGE_PERIOD * k))
    row_xs.sort()
    for row_i, rx in enumerate(row_xs):
        z_min = max(Z_FIRST, (abs(rx) - 1.5) / 0.70)
        z = Z_FIRST + math.ceil((z_min - Z_FIRST) / PLANT_SPACING) * PLANT_SPACING
        idx = 0
        while z < Z_MAX:
            x = rx + rng.normal(0, 0.012)
            zz = z + rng.normal(0, 0.012)
            dist = math.hypot(x, zz)
            lod = 0 if dist < 5.6 else 1 if dist < 11.5 else 2 if dist < 26 else 3
            plants.append(dict(x=float(x), z=float(zz), y=float(ridge_h(x)), yaw=float(rng.uniform(0, 6.283)),
                               scale=float(rng.uniform(0.93, 1.07)), variant=int(rng.integers(0, N_VARIANTS)),
                               lod=lod, row=row_i, idx=idx,
                               tint=[float(v) for v in (rng.uniform(0.92, 1.08), rng.uniform(0.93, 1.07), rng.uniform(0.92, 1.06))]))
            z += PLANT_SPACING
            idx += 1
    tufts = []
    for _ in range(520):
        x = float(rng.uniform(-4.8, 4.8))
        z = float(rng.uniform(0.35, 11.0))
        if z > Z_FIRST - 0.1 and min(abs(x - rx) for rx in row_xs) < 0.16:
            continue
        tufts.append(dict(x=x, z=z, yaw=float(rng.uniform(0, 6.283)), s=float(rng.uniform(0.7, 1.7))))
    tapes = [rx - math.copysign(0.17, rx) for rx in row_xs]
    return dict(
        cam=dict(height=CAM_H, pitch_deg=12.0, focal=FOCAL, w=W, h=H),
        ridge=dict(h=RIDGE_H, period=RIDGE_PERIOD, rowx=ROW_X),
        plants=plants, tufts=tufts, tapes=tapes, rows=row_xs, n_variants=N_VARIANTS,
        render=dict(exposure=0.62, env_intensity=1.25, bg_intensity=1.0, sun=2.6, fog_density=0.0038, fog_gain=0.82,
                    msaa=False, ss=1.5, far_basic=True, far_basic_gain=0.68, shadow_map=2048, soft_shadow=False),
    )


# ------------------------------------------------------------------ 植株屏幕投影（检测框）
class PlantProjector:
    """用 three.js 返回的各变体顶点采样（局部坐标）重建每株芹菜的屏幕包围盒。"""

    ROOT_BOX = np.array([[-0.07, -0.02], [0.07, -0.02], [0.07, 0.10], [-0.07, 0.10]])

    def __init__(self, layout, variant_pts):
        self.layout = layout
        self.vp = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in variant_pts]
        self.plants = [p for p in layout["plants"] if math.hypot(p["x"], p["z"]) < 11.5]

    def _world_pts(self, p):
        loc = self.vp[p["variant"]] * p["scale"]
        c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
        x = loc[:, 0] * c + loc[:, 2] * s
        z = -loc[:, 0] * s + loc[:, 2] * c
        # three.js 侧 X_js = -x_py：局部偏移的 x 分量同样镜像
        return np.stack([p["x"] - x, p["y"] + loc[:, 1], p["z"] + z], -1)

    def screen_infos(self, cam_x):
        out = []
        for p in self.plants:
            pts = self._world_pts(p)
            u, v, d = project(pts, cam_x)
            if d.min() < 0.3:
                continue
            ru, rv, rd = project(np.array([p["x"], p["y"], p["z"]]), cam_x)
            rb = np.array([[p["x"] + dx, p["y"] + dy, p["z"]] for dx, dy in self.ROOT_BOX])
            bu, bv, _ = project(rb, cam_x)
            out.append(dict(box=(float(u.min()), float(v.min()), float(u.max()), float(v.max())),
                            root=(float(ru), float(rv)), depth=float(rd),
                            root_box=(float(bu.min()), float(bv.min()), float(bu.max()), float(bv.max())),
                            row=p["row"], idx=p["idx"], x=p["x"], z=p["z"]))
        return out


# ------------------------------------------------------------------ 渲染器
class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


class Renderer:
    def __init__(self, layout):
        from playwright.sync_api import sync_playwright
        self.layout = layout
        handler = functools.partial(_Quiet, directory=WEB)
        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        port = self.srv.server_address[1]
        self.pw = sync_playwright().start()
        exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
        kw = dict(executable_path=exe) if os.path.exists(exe) else {}
        self.browser = self.pw.chromium.launch(
            args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                  "--ignore-gpu-blocklist", "--enable-webgl", "--no-sandbox"], **kw)
        self.page = self.browser.new_page(viewport={"width": int(W * layout["render"].get("ss", 1)), "height": int(H * layout["render"].get("ss", 1))})
        self.page.on("pageerror", lambda e: print("[page error]", e, flush=True))
        self.page.on("console", lambda m: print("[console]", m.text, flush=True) if m.type == "error" else None)
        self.page.goto(f"http://127.0.0.1:{port}/index.html")
        self.page.wait_for_function("window.ready===true", timeout=60000)
        res = self.page.evaluate("(lay) => window.setup(lay)", layout)
        self.variant_pts = res["variantPts"]
        self.info = res["info"]

    def render(self, cam_x):
        data = self.page.evaluate(f"window.renderFrame({cam_x!r})")
        buf = np.frombuffer(base64.b64decode(data.split(",", 1)[1]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
        if img.shape[1] != W:                                            # 超采样 → 下采样抗锯齿
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def close(self):
        try:
            self.browser.close()
            self.pw.stop()
            self.srv.shutdown()
        except Exception:
            pass
