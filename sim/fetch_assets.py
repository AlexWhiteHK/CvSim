"""
fetch_assets.py —— 下载渲染用的第三方 CC0 素材（Poly Haven）到 sim/web/assets/。

  * brown_mud_02  : 深色壤土 PBR 贴图（albedo / normal / roughness）
  * farm_field    : 农田 HDRI（天空、远景地平线与环境光照）

素材协议均为 CC0（https://polyhaven.com/license），见 sim/web/assets/CREDITS.txt。
"""
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "assets")
BASE = "https://dl.polyhaven.org/file/ph-assets"
FILES = {
    "soil_diff.jpg": f"{BASE}/Textures/jpg/2k/brown_mud_02/brown_mud_02_diff_2k.jpg",
    "soil_nor.jpg": f"{BASE}/Textures/jpg/2k/brown_mud_02/brown_mud_02_nor_gl_2k.jpg",
    "soil_rough.jpg": f"{BASE}/Textures/jpg/2k/brown_mud_02/brown_mud_02_rough_2k.jpg",
    "sky.hdr": f"{BASE}/HDRIs/hdr/4k/farm_field_4k.hdr",
}
CREDITS = """Third-party assets (all CC0 1.0, Poly Haven - https://polyhaven.com)
  soil_*.jpg : "Brown Mud 02"  (textures)
  sky.hdr    : "Farm Field"    (HDRI)
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(OUT, name)
        if os.path.exists(dst):
            continue
        print("download", name)
        req = urllib.request.Request(url, headers={"User-Agent": "cvsim-asset-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
    with open(os.path.join(OUT, "CREDITS.txt"), "w") as f:
        f.write(CREDITS)


if __name__ == "__main__":
    main()
