"""
postfx.py —— 模拟工业相机成像：轻微软化/锐化、暗角、传感器噪声、曝光漂移、
机体抖动，以及静止时叶片被微风吹动的细微摆动。
"""
import math

import cv2
import numpy as np

W, H = 1280, 720
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r2 = ((_xx - W / 2) / (W / 2)) ** 2 + ((_yy - H / 2) / (H / 2)) ** 2
VIGNETTE = (1 - 0.26 * _r2 ** 1.15)[..., None].astype(np.float32)


def sway(img, t, mask_img):
    """叶片微风摆动：位移只施加在“绿色植株”区域，幅度约 1 px。"""
    f = mask_img.astype(np.float32)
    g = f[..., 1] - 0.5 * (f[..., 0] + f[..., 2])
    m = cv2.GaussianBlur(np.clip((g - 6) / 30.0, 0, 1), (0, 0), 3.0)
    gust = 1.0 + 0.5 * math.sin(2 * math.pi * 0.21 * t)
    dx = 1.3 * gust * np.sin(2 * math.pi * 0.55 * t + _xx * 0.011 + _yy * 0.007) * m
    dy = 0.5 * gust * np.cos(2 * math.pi * 0.7 * t + _xx * 0.009) * m
    return cv2.remap(img, _xx + dx, _yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply(img_rgb, i, t, v_act):
    rng = np.random.default_rng(50_000 + i)
    img = sway(img_rgb, t, img_rgb)
    amp = 0.25 + 0.9 * abs(v_act) / 0.14                       # 运动时机体抖动更明显
    M = np.float32([[1, 0, rng.normal(0, 0.30 * amp)], [0, 1, rng.normal(0, 0.45 * amp)]])
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    f = img.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(f, (0, 0), 0.65)
    f = soft * 1.22 - cv2.GaussianBlur(soft, (0, 0), 1.5) * 0.22          # 镜头柔化 + 轻微锐化
    f = np.clip(f, 0, 1)
    f = f * f * (3 - 2 * f) * 0.25 + f * 0.75                             # 轻微 S 曲线
    f *= VIGNETTE
    f *= 1.0 + 0.010 * math.sin(t * 0.8)                                   # 自动曝光漂移
    lum = f.mean(-1, keepdims=True)
    sig = 0.005 + 0.011 * (1 - lum)                                        # 暗部噪声更明显
    f += rng.standard_normal(f.shape, dtype=np.float32) * sig
    f += rng.standard_normal((H, W, 1), dtype=np.float32) * 0.003
    return np.clip(f * 255, 0, 255).astype(np.uint8)
