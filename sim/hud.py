"""
hud.py —— 单画面 HUD：把 CV 检测结果与控制状态直接叠加在相机画面上（工业视觉软件常见的 OSD 风格）。
线条/矩形用 cv2（抗锯齿），文字用 PIL（等宽字体）。画面坐标 = 1280×720 相机像素坐标。
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720

LIME = (60, 255, 60)          # celery 框
YELLOW = (255, 235, 0)        # root 框
RED = (255, 64, 64)           # 目标
CYAN = (0, 220, 255)          # 割刀基准线
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
AMBER = (255, 176, 0)
GREEN = (60, 230, 90)

_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
_fonts = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _fonts:
        _fonts[k] = ImageFont.truetype(_MONOB if bold else _MONO, size)
    return _fonts[k]


def _dash(img, p0, p1, color, th=1, dash=8, gap=6):
    x0, y0 = p0
    x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 1:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    s = 0.0
    while s < L:
        e = min(s + dash, L)
        cv2.line(img, (int(x0 + ux * s), int(y0 + uy * s)), (int(x0 + ux * e), int(y0 + uy * e)), color, th, cv2.LINE_AA)
        s += dash + gap


def _fill(img, x1, y1, x2, y2, color, alpha):
    x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), img.shape[1]), min(int(y2), img.shape[0])
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    roi[:] = (roi * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)


class Overlay:
    """收集文字，最后统一用 PIL 绘制；带底色的标签先用 cv2 画底。"""

    def __init__(self, img):
        self.img = img
        self.texts = []

    def text(self, x, y, s, size=12, color=WHITE, bold=False, anchor="la"):
        self.texts.append((x, y, s, size, color, bold, anchor))

    def tag(self, x, y, s, bg, fg=(0, 0, 0), size=11, bold=False):
        """x, y：标签左下角（贴在框上沿）"""
        w = int(font(size, bold).getlength(s)) + 6
        h = size + 5
        x, y = int(x), int(y)
        y0 = y - h
        if y0 < 28:
            y0 = y + 1                                       # 贴到框内侧，避免遮住顶栏
        cv2.rectangle(self.img, (x, y0), (x + w, y0 + h), bg, -1)
        self.texts.append((x + 3, y0 + h // 2 + 1, s, size, fg, bold, "lm"))

    def flush(self):
        pil = Image.fromarray(self.img)
        d = ImageDraw.Draw(pil)
        for x, y, s, size, color, bold, anchor in self.texts:
            d.text((x, y), s, font=font(size, bold), fill=tuple(color), anchor=anchor)
        return np.asarray(pil)


def compose(feed_rgb, fd, meta):
    cfg = meta["cfg"]
    img = np.ascontiguousarray(feed_rgb)
    ov = Overlay(img)
    au, hv = int(cfg.axis_u), int(cfg.harvest_v)
    t = fd["t"]
    state = fd["state"]
    sel = state in ("SELECT", "ALIGN", "LOCKED")
    tid = fd["tid"] if sel else None
    target_row = fd["row"] if sel else None
    locked = state == "LOCKED"

    # ------------------------------------------------ 顶栏
    _fill(img, 0, 0, W, 26, (0, 0, 0), 0.55)
    ov.text(10, 13, f"CAM-L   {W}x{H}   {meta['fps']:.1f} FPS   INF {fd['inf_ms']:.1f} ms", 13, WHITE, anchor="lm")
    st_col = GREEN if locked else (AMBER if state == "ALIGN" else WHITE)
    ov.text(W // 2, 13, f"STATE: {state}", 14, st_col, True, "mm")
    ov.text(W - 10, 13, meta["clock"](t), 13, WHITE, anchor="rm")

    # ------------------------------------------------ 检测 ROI / 目标 ROI / 割刀基准线
    if state != "INIT":
        _dash(img, (0, cfg.det_roi_top), (W, cfg.det_roi_top), (200, 200, 200), 1, 10, 8)
        ov.text(8, cfg.det_roi_top - 8, "DET ROI", 11, (200, 200, 200), anchor="lm")
    if state in ("SCAN", "SELECT"):
        for yy in (hv - cfg.roi_half_h, hv + cfg.roi_half_h):
            _dash(img, (0, yy), (W, yy), CYAN, 1, 4, 10)
        ov.text(8, hv - cfg.roi_half_h - 8, "TARGET ROI", 11, CYAN, anchor="lm")
    if state != "INIT":
        cv2.line(img, (au, hv - 70), (au, H), CYAN, 1, cv2.LINE_AA)
        cv2.line(img, (au - 20, hv), (au + 20, hv), CYAN, 1, cv2.LINE_AA)
        cv2.circle(img, (au, hv), 9, CYAN, 1, cv2.LINE_AA)
        ov.text(au + 8, hv - 62, "CUTTER AXIS", 11, CYAN, anchor="lm")

    # ------------------------------------------------ 行线（仅最近一对行：L1 为本相机负责的目标行）
    cam_x = fd["cam_x"]
    for r in fd["rows"]:
        if r["rank"] != 1 or state in ("INIT",):
            continue
        mine = r["name"] == cfg.side + "1"
        zs = np.linspace(2.0, 6.5, 12)
        pts = []
        for z in zs:
            X = r["a"] * z + r["b"]
            u, v, _ = meta["project"](np.array([cam_x + X, cfg.ground_y_m, z]), cam_x)
            pts.append((float(u), float(v)))
        col = WHITE if mine else GREY
        for k in range(0, len(pts) - 1, 2):
            if pts[k][1] > cfg.det_roi_top - 4 and pts[k + 1][1] > cfg.det_roi_top - 4:
                cv2.line(img, (int(pts[k][0]), int(pts[k][1])), (int(pts[k + 1][0]), int(pts[k + 1][1])), col, 1, cv2.LINE_AA)
        ov.text(pts[-1][0], pts[-1][1] - 9, r["name"] + (" *" if mine else ""), 12, col, True, "mm")

    # ------------------------------------------------ 检测框
    # OSD 只显示“看得清”的目标：被前方植株遮挡超过 45% 的框不画（控制逻辑仍然使用全部检测结果）
    plist = [p for p in fd["plants"] if p["ground"] is not None and p["age"] >= 3]
    def covered(p):
        x1, y1, x2, y2 = p["box"]
        a = max((x2 - x1) * (y2 - y1), 1.0)
        c = 0.0
        for q in plist:
            if q["ground"][1] < p["ground"][1] - 0.1:
                iw = min(x2, q["box"][2]) - max(x1, q["box"][0])
                ih = min(y2, q["box"][3]) - max(y1, q["box"][1])
                if iw > 0 and ih > 0:
                    c = max(c, iw * ih / a)
        return c
    for p in sorted(plist, key=lambda q: -q["box"][3]):
        if p["tid"] != tid and (covered(p) > 0.45 or p["ground"][1] > 4.6):
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in p["box"])
        is_t = p["tid"] == tid
        dim = sel and (p["row"] != target_row)
        c = RED if is_t else ((0, 140, 0) if dim else LIME)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2 if is_t else 1, cv2.LINE_AA)
        near = p["ground"][1] < 3.6
        if near and not is_t:
            ov.tag(x1, y1, f"celery {p['conf']:.2f} #{p['tid']}", c, size=11)
        if is_t:
            ov.tag(x1, y1, f"TARGET #{p['tid']}  celery {p['conf']:.2f}", (GREEN if locked else RED), (0, 0, 0), 12, True)
        if p["root_box"] is not None:
            rb = p["root_box"]
            rc = (GREEN if locked else RED) if is_t else ((140, 130, 0) if dim else YELLOW)
            cv2.rectangle(img, (int(rb[0]), int(rb[1])), (int(rb[2]), int(rb[3])), rc, 1, cv2.LINE_AA)
            if is_t or near:
                ov.tag(rb[2] + 2, rb[3], f"root {p['root_conf']:.2f}", rc, size=10)
            ru, rv = p["root_uv"]
            cv2.drawMarker(img, (int(ru), int(rv)), rc, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)

    # ------------------------------------------------ 目标指示：割刀线 → 根部
    tg = fd["target"]
    if tg is not None and sel:
        ru, rv = tg["root_uv"]
        ex = ru - au
        tc = GREEN if locked else RED
        cv2.circle(img, (int(ru), int(rv)), 14, tc, 2, cv2.LINE_AA)
        if not locked and abs(ex) > 18:
            sgn = 1 if ex > 0 else -1
            p0 = (int(au + sgn * 14), hv)
            p1 = (int(ru - sgn * 18), hv)
            if abs(p1[0] - p0[0]) > 10:
                cv2.arrowedLine(img, p0, p1, AMBER, 2, cv2.LINE_AA, tipLength=min(0.4, 14 / max(abs(p1[0] - p0[0]), 1)))
                ov.text((p0[0] + p1[0]) // 2, hv - 12, f"dx {ex:+.0f}px  {'LEFT' if ex < 0 else 'RIGHT'}", 13, AMBER, True, "md")
        if locked:
            ov.tag(au + 18, hv - 6, f"ALIGNED  dX {fd['err_m'] * 1000:+.1f}mm", GREEN, (0, 0, 0), 12, True)

    # ------------------------------------------------ 左下信息块
    lines = []
    lines.append(("UNIT", f"{'LEFT' if cfg.side == 'L' else 'RIGHT'}   ROW {fd['row'] or '--'}"))
    lines.append(("STATE", state))
    if tg is not None and sel:
        lines.append(("TARGET", f"#{tg['tid']}  conf {tg['conf']:.2f}"))
        lines.append(("ROOT", f"({tg['root_uv'][0]:.0f}, {tg['root_uv'][1]:.0f}) px"))
        lines.append(("dX", f"{fd['err_px']:+7.1f} px  {fd['err_m'] * 1000:+7.1f} mm"))
    else:
        lines.append(("TARGET", "--"))
        lines.append(("ROOT", "--"))
        lines.append(("dX", "--"))
    arrow = "<<" if fd["v_cmd"] < -1e-3 else (">>" if fd["v_cmd"] > 1e-3 else "  ")
    lines.append(("CMD", f"{fd['v_cmd']:+.3f} m/s {arrow}"))
    lines.append(("STAGE", f"{fd['cam_x'] * 1000:+.1f} mm"))
    bx, by, bw = 8, H - 12 - 17 * len(lines) - 8, 262
    _fill(img, bx, by, bx + bw, H - 8, (0, 0, 0), 0.58)
    for k, (a, b) in enumerate(lines):
        yy = by + 12 + k * 17
        ov.text(bx + 8, yy, a, 12, GREY, anchor="lm")
        col = WHITE
        if a == "dX" and sel:
            col = GREEN if abs(fd["err_m"]) < meta["cfg"].tol_m else AMBER
        if a == "STATE" and locked:
            col = GREEN
        ov.text(bx + 76, yy, b, 12, col, a in ("dX", "STATE"), "lm")

    # ------------------------------------------------ 对位指示条（根部相对割刀线的偏差）
    if tg is not None and sel:
        gw = 200
        gy = H - 34
        cv2.line(img, (W // 2 - gw, gy), (W // 2 + gw, gy), WHITE, 1, cv2.LINE_AA)
        for k in range(-4, 5):
            x = W // 2 + k * gw // 4
            cv2.line(img, (x, gy - (6 if k % 2 == 0 else 3)), (x, gy), WHITE, 1, cv2.LINE_AA)
        tol_px = max(2, int(cfg.tol_m / max(tg["ground"][1], 0.5) * cfg.fx * 0.9))
        cv2.rectangle(img, (W // 2 - tol_px, gy - 8), (W // 2 + tol_px, gy + 2), GREEN, -1)
        e = max(-gw, min(gw, fd["err_px"] * gw / 300.0))
        mx = int(W // 2 + e)
        tri = np.array([[mx, gy + 3], [mx - 6, gy + 14], [mx + 6, gy + 14]], np.int32)
        cv2.fillPoly(img, [tri], GREEN if locked else AMBER, cv2.LINE_AA)
        ov.text(W // 2 - gw - 10, gy, "L", 12, GREY, True, "rm")
        ov.text(W // 2 + gw + 10, gy, "R", 12, GREY, True, "lm")
        ov.text(W // 2, gy - 16, "dX", 11, GREY, anchor="mm")

    # ------------------------------------------------ 右下事件日志
    lg = [l for l in meta["log"] if l[0] <= fd["i"]][-5:]
    if lg:
        lw = 400
        _fill(img, W - lw - 8, H - 12 - 15 * len(lg) - 6, W - 8, H - 8, (0, 0, 0), 0.58)
        for k, (fi, msg) in enumerate(lg):
            yy = H - 12 - 15 * (len(lg) - k) + 8
            tt = fi / meta["fps"]
            ov.text(W - lw, yy, f"{tt:5.1f}s {msg}", 11, WHITE if k == len(lg) - 1 else GREY, anchor="lm")

    if locked:
        cv2.rectangle(img, (1, 1), (W - 2, H - 2), GREEN, 2, cv2.LINE_AA)
    return ov.flush()
