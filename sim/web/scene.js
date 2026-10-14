// scene.js —— 芹菜田 three.js 场景（无头 Chromium + SwiftShader 软件渲染）
//
// 坐标约定：Python 侧 x 向右；three.js 里相机朝 +z、右手系下 +x 在画面左侧，
// 因此这里统一用 X_js = -x_py 做一次镜像（只在布局边界处理）。
import * as THREE from 'three';
import { RGBELoader } from './vendor/RGBELoader.js';

const canvas = document.getElementById('c');
let renderer, scene, camera, sunLight, layout;
const CAM = { h: 1.12, pitch: 12 * Math.PI / 180, focal: 900, w: 1280, hh: 720 };

// ------------------------------------------------------------------ 工具
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const V3 = (x, y, z) => new THREE.Vector3(x, y, z);

class GB {
  constructor() { this.p = []; this.uv = []; this.col = []; this.idx = []; }
  get n() { return this.p.length / 3; }
  v(x, y, z, u, v, c) { this.p.push(x, y, z); this.uv.push(u, v); this.col.push(c[0], c[1], c[2]); return this.n - 1; }
  tri(a, b, c) { this.idx.push(a, b, c); }
  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.p, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    g.setIndex(this.idx);
    g.computeVertexNormals();
    g.computeBoundingBox();
    return g;
  }
}

// ------------------------------------------------------------------ 垄面高度（与 Python 侧保持一致）
let RIDGE = { h: 0.13, period: 1.2, rowx: 0.6 };
function ridgeH(xpy) {
  const th = (xpy - RIDGE.rowx) * (2 * Math.PI / RIDGE.period);
  return RIDGE.h * (0.5 + 0.5 * Math.tanh(2.0 * Math.cos(th)));
}

// ------------------------------------------------------------------ 程序化贴图
function stalkTexture() {
  const c = document.createElement('canvas'); c.width = 256; c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#e9eddc'; g.fillRect(0, 0, 256, 512);
  const r = mulberry32(11);
  // 细纤维
  for (let i = 0; i < 520; i++) {
    const x = r() * 256, w = 0.6 + r() * 1.4;
    const a = 0.04 + r() * 0.14;
    g.strokeStyle = r() < 0.5 ? `rgba(70,95,40,${a})` : `rgba(255,255,255,${a * 1.4})`;
    g.lineWidth = w;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x + (r() - 0.5) * 6, 512); g.stroke();
  }
  // 外侧 6 条肋之间的沟（u = k/12，k = 0..6 对应外侧半圈）
  for (let k = 0; k <= 6; k++) {
    const x = (k / 12) * 256;
    g.strokeStyle = 'rgba(55,85,30,0.38)'; g.lineWidth = 3.2;
    g.beginPath(); g.moveTo(x, 0); g.lineTo(x, 512); g.stroke();
    g.strokeStyle = 'rgba(255,255,255,0.16)'; g.lineWidth = 2;
    g.beginPath(); g.moveTo(x + 4, 0); g.lineTo(x + 4, 512); g.stroke();
  }
  // 内侧凹槽颜色略深
  const gr = g.createLinearGradient(128, 0, 256, 0);
  gr.addColorStop(0, 'rgba(120,150,70,0.0)'); gr.addColorStop(0.5, 'rgba(120,150,70,0.18)'); gr.addColorStop(1, 'rgba(120,150,70,0.0)');
  g.fillStyle = gr; g.fillRect(128, 0, 128, 512);
  // 低频斑驳
  for (let i = 0; i < 90; i++) {
    g.fillStyle = `rgba(90,110,50,${0.02 + r() * 0.05})`;
    g.beginPath(); g.ellipse(r() * 256, r() * 512, 4 + r() * 12, 10 + r() * 40, 0, 0, 7); g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

function leafTexture() {
  const S = 256;
  const c = document.createElement('canvas'); c.width = S; c.height = S;
  const g = c.getContext('2d');
  g.clearRect(0, 0, S, S);
  // 小叶轮廓：楔形基部 + 三裂 + 锯齿
  const half = [
    [0, 250], [30, 222], [58, 192], [86, 168], [100, 138], [78, 120], [52, 112], [34, 98], [38, 70], [60, 52],
    [76, 56], [84, 36], [104, 18], [128, 6],
  ];
  const pts = [];
  half.forEach(p => pts.push([128 - p[0] * 1.0, p[1]]));          // 左半
  pts.push([128, 4]);
  for (let i = half.length - 1; i >= 0; i--) pts.push([128 + half[i][0], half[i][1]]);
  // 锯齿细分
  const out = [];
  const r = mulberry32(5);
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    const n = 4;
    for (let k = 0; k < n; k++) {
      const t = k / n;
      const x = a[0] + (b[0] - a[0]) * t, y = a[1] + (b[1] - a[1]) * t;
      const tooth = (k % 2 === 0 ? 1 : -1) * 3.2;
      const nx = -(b[1] - a[1]), ny = (b[0] - a[0]);
      const L = Math.hypot(nx, ny) || 1;
      out.push([x + nx / L * tooth + (r() - 0.5), y + ny / L * tooth + (r() - 0.5)]);
    }
  }
  g.beginPath(); out.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])); g.closePath();
  g.save(); g.clip();
  const gr = g.createRadialGradient(128, 150, 10, 128, 140, 140);
  gr.addColorStop(0, '#6fa83a'); gr.addColorStop(0.6, '#4f8a2a'); gr.addColorStop(1, '#3b7020');
  g.fillStyle = gr; g.fillRect(0, 0, S, S);
  for (let i = 0; i < 260; i++) {                                  // 叶肉斑驳
    g.fillStyle = `rgba(${r() < 0.5 ? '30,70,15' : '150,190,80'},${0.03 + r() * 0.06})`;
    g.beginPath(); g.ellipse(r() * S, r() * S, 3 + r() * 9, 3 + r() * 9, r() * 3, 0, 7); g.fill();
  }
  // 叶脉
  g.strokeStyle = 'rgba(190,220,120,0.75)'; g.lineWidth = 3;
  g.beginPath(); g.moveTo(128, 250); g.lineTo(128, 10); g.stroke();
  g.lineWidth = 1.8; g.strokeStyle = 'rgba(190,220,120,0.55)';
  [[128, 190, 52, 130], [128, 190, 204, 130], [128, 130, 58, 70], [128, 130, 198, 70], [128, 90, 96, 30], [128, 90, 160, 30]].forEach(l => {
    g.beginPath(); g.moveTo(l[0], l[1]); g.quadraticCurveTo((l[0] + l[2]) / 2, l[1] - 10, l[2], l[3]); g.stroke();
  });
  g.restore();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  t.generateMipmaps = true;
  return t;
}

// ------------------------------------------------------------------ 芹菜几何
const LODS = [
  { seg: 12, rad: 24, lg: [3, 3], br: 4, lf: 3, tube: 5, tseg: 4 },
  { seg: 6, rad: 10, lg: [1, 2], br: 3, lf: 3, tube: 4, tseg: 2 },
  { seg: 3, rad: 6, lg: [1, 1], br: 2, lf: 3, tube: 0, tseg: 1 },
];
const STALK_COLS = [[0.86, 0.90, 0.66], [0.70, 0.84, 0.44], [0.54, 0.74, 0.30], [0.42, 0.64, 0.22]];
function stalkColor(s) {
  const f = s * 3, i = Math.min(2, Math.floor(f)), t = f - i;
  const a = STALK_COLS[i], b = STALK_COLS[i + 1];
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function addTube(gb, pts, radii, radial, color, uvScale = 1) {
  // 简单圆管（细叶柄用）
  const rings = [];
  for (let i = 0; i < pts.length; i++) {
    const T = (i === 0 ? pts[1].clone().sub(pts[0]) : i === pts.length - 1 ? pts[i].clone().sub(pts[i - 1]) : pts[i + 1].clone().sub(pts[i - 1])).normalize();
    let up = Math.abs(T.y) > 0.9 ? V3(1, 0, 0) : V3(0, 1, 0);
    const B = new THREE.Vector3().crossVectors(T, up).normalize();
    const N = new THREE.Vector3().crossVectors(B, T).normalize();
    const ring = [];
    for (let j = 0; j <= radial; j++) {
      const a = j / radial * Math.PI * 2;
      const p = pts[i].clone().addScaledVector(B, Math.cos(a) * radii[i]).addScaledVector(N, Math.sin(a) * radii[i]);
      ring.push(gb.v(p.x, p.y, p.z, j / radial, i / (pts.length - 1) * uvScale, color));
    }
    rings.push(ring);
  }
  for (let i = 0; i < rings.length - 1; i++) for (let j = 0; j < radial; j++) {
    const a = rings[i][j], b = rings[i][j + 1], c = rings[i + 1][j], d = rings[i + 1][j + 1];
    gb.tri(a, c, b); gb.tri(b, c, d);
  }
}

function addLeaflet(gbL, c, along, normal, len, wid, grid, tint, rng) {
  // c: 叶基位置；along: 沿叶长方向；normal: 叶面法向
  const u = along.clone().normalize();
  const nn = normal.clone().sub(u.clone().multiplyScalar(normal.dot(u))).normalize();
  const w = new THREE.Vector3().crossVectors(nn, u).normalize();
  const [nu, nv] = grid;
  const droop = 0.25 + rng() * 0.25, cup = 0.18 + rng() * 0.15;
  const ids = [];
  for (let i = 0; i <= nu; i++) {
    const row = [];
    const s = i / nu;
    for (let j = 0; j <= nv; j++) {
      const t = j / nv, tt = (t - 0.5) * 2;
      const p = c.clone().addScaledVector(u, s * len).addScaledVector(w, tt * 0.5 * wid * (0.35 + 0.65 * Math.sin(Math.PI * Math.min(1, s * 1.15 + 0.1))))
        .addScaledVector(nn, cup * tt * tt * wid * 0.25 - droop * s * s * len * 0.5);
      row.push(gbL.v(p.x, p.y, p.z, t, s, tint));
    }
    ids.push(row);
  }
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = ids[i][j], b = ids[i][j + 1], cc = ids[i + 1][j], d = ids[i + 1][j + 1];
    gbL.tri(a, b, cc); gbL.tri(b, d, cc);
  }
}

function makePlant(variant, lod) {
  const L = LODS[lod];
  const rng = mulberry32(900 + variant * 7);
  const gs = new GB(), gl = new GB();
  const nStalk = 15 + Math.floor(rng() * 4);
  const H = 0.44 + rng() * 0.06;
  const az0 = rng() * 6.283;
  const lean = [(rng() - 0.5) * 0.06, (rng() - 0.5) * 0.06];
  const order = Array.from({ length: nStalk }, (_, i) => i).sort(() => rng() - 0.5);
  const pts3 = [];                                   // 供 Python 计算屏幕包围盒
  for (let j = 0; j < nStalk; j++) {
    const r = (order[j] + rng()) / nStalk;             // 0 内层 → 1 外层
    const az = az0 + 2 * Math.PI * j / nStalk + (rng() - 0.5) * 0.35;
    const tilt0 = 0.03 + 0.20 * r + (rng() - 0.5) * 0.05;
    const bend = 0.06 + 0.30 * r;
    const len = H * (1.0 - 0.18 * r) * (0.95 + rng() * 0.1);
    const r0 = 0.010 + 0.022 * r;
    const w0 = 0.046 + 0.022 * r + (rng() - 0.5) * 0.004;
    const Oh = V3(Math.cos(az), 0, Math.sin(az));
    // 中心线
    const P = [V3(Math.cos(az) * r0, 0, Math.sin(az) * r0)];
    const step = len / L.seg;
    for (let i = 1; i <= L.seg; i++) {
      const s = i / L.seg;
      const til = tilt0 + bend * Math.pow(s, 1.5);
      const d = V3(Math.sin(til) * Math.cos(az) + lean[0] * s, Math.cos(til), Math.sin(til) * Math.sin(az) + lean[1] * s).normalize();
      P.push(P[i - 1].clone().addScaledVector(d, step));
    }
    // 截面
    const rings = [];
    const M = L.rad;
    const shade = 0.78 + 0.34 * (0.5 * r + 0.5 * rng());
    for (let i = 0; i <= L.seg; i++) {
      const s = i / L.seg;
      const T = (i === 0 ? P[1].clone().sub(P[0]) : i === L.seg ? P[i].clone().sub(P[i - 1]) : P[i + 1].clone().sub(P[i - 1])).normalize();
      const N = Oh.clone().sub(T.clone().multiplyScalar(Oh.dot(T))).normalize();
      const B = new THREE.Vector3().crossVectors(T, N).normalize();
      const w = w0 * (1 - 0.50 * Math.pow(s, 1.1)) * (1 + 0.55 * Math.pow(1 - s, 5));
      const t = w * 0.44;
      const col0 = stalkColor(s);
      let k = shade;
      if (s < 0.10) k *= 0.80 + 2.0 * s;                        // 基部带土色
      const col = [col0[0] * k, col0[1] * k, col0[2] * k * (s < 0.10 ? 0.92 : 1)];
      const ring = [];
      for (let m = 0; m <= M; m++) {
        const ph = m / M * Math.PI * 2, cx = Math.cos(ph), sy = Math.sin(ph);
        let x = 0.5 * w * cx, y = 0.5 * t * sy;
        if (sy < 0) y *= (0.18 + 0.82 * cx * cx);                // 内侧凹槽
        else if (lod === 0) {                                     // 外侧 6 条肋
          let prof = 0;
          for (let q = 0; q < 6; q++) { const pk = Math.PI * (q + 0.5) / 6; prof += Math.exp(-Math.pow((ph - pk) / 0.20, 2)); }
          y += 0.5 * t * 0.30 * prof;
        }
        const p = P[i].clone().addScaledVector(B, x).addScaledVector(N, y);
        ring.push(gs.v(p.x, p.y, p.z, m / M, s * len / 0.30, col));
        if (lod === 0 && (m % 3 === 0) && i % 2 === 0) pts3.push(p.x, p.y, p.z);
      }
      rings.push(ring);
    }
    for (let i = 0; i < L.seg; i++) for (let m = 0; m < M; m++) {
      const a = rings[i][m], b = rings[i][m + 1], c = rings[i + 1][m], d = rings[i + 1][m + 1];
      gs.tri(a, c, b); gs.tri(b, c, d);
    }
    // 顶部分枝 + 小叶
    const top = P[L.seg], Tt = top.clone().sub(P[L.seg - 1]).normalize();
    const Nt = Oh.clone().sub(Tt.clone().multiplyScalar(Oh.dot(Tt))).normalize();
    const Bt = new THREE.Vector3().crossVectors(Tt, Nt).normalize();
    for (let b = 0; b < L.br; b++) {
      const spread = L.br === 1 ? 0 : (b / (L.br - 1) - 0.5) * 2;
      const dir = Tt.clone().addScaledVector(Bt, spread * 0.65).addScaledVector(Nt, 0.28 + rng() * 0.25).normalize();
      const bl = 0.075 + rng() * 0.05;
      const bp = [top.clone()];
      for (let i = 1; i <= L.tseg; i++) {
        const f = i / L.tseg;
        const dd = dir.clone().add(V3(0, -0.55 * f, 0)).normalize();
        bp.push(bp[i - 1].clone().addScaledVector(dd, bl / L.tseg));
      }
      if (L.tube > 0) addTube(gs, bp, bp.map((_, i) => 0.0042 * (1 - 0.5 * i / L.tseg)), L.tube, stalkColor(0.9).map(v => v * 0.95));
      const tip = bp[bp.length - 1];
      const endDir = tip.clone().sub(bp[bp.length - 2]).normalize();
      // 顶小叶 + 两侧小叶
      for (let q = 0; q < L.lf; q++) {
        const a = (q - (L.lf - 1) / 2) * 0.85;
        const alongv = endDir.clone().applyAxisAngle(V3(0, 1, 0), a * 0.9).add(V3(0, 0.15 + rng() * 0.2, 0)).normalize();
        const base = q === 1 ? tip.clone() : tip.clone().addScaledVector(endDir, -0.012);
        const nrm = V3((rng() - 0.5) * 0.5, 1, (rng() - 0.5) * 0.5).addScaledVector(Oh, 0.25).normalize();
        const llen = (q === 1 ? 0.112 : 0.090) * (0.85 + rng() * 0.35);
        const lwid = llen * (0.78 + rng() * 0.2);
        const tint = [0.82 + rng() * 0.32, 0.86 + rng() * 0.26, 0.78 + rng() * 0.28];
        addLeaflet(gl, base, alongv, nrm, llen, lwid, L.lg, tint, rng);
        if (lod === 0 && b === 1) {
          pts3.push(base.x + alongv.x * llen, base.y + alongv.y * llen, base.z + alongv.z * llen);
          pts3.push(base.x + alongv.x * llen * 0.5 + 0.03, base.y + alongv.y * llen * 0.5, base.z + alongv.z * llen * 0.5);
          pts3.push(base.x + alongv.x * llen * 0.5 - 0.03, base.y + alongv.y * llen * 0.5, base.z + alongv.z * llen * 0.5);
        }
      }
    }
  }
  return { stalk: gs.build(), leaf: gl.build(), pts: pts3 };
}

function makeFarPlant() {
  // 远景：窄锥形茎束 + 叶冠椭球
  const gs = new GB(), gl = new GB();
  const col = stalkColor(0.15);
  const n = 6;
  const cb = [];
  for (let i = 0; i <= n; i++) cb.push(gs.v(Math.cos(i / n * 6.283) * 0.07, 0, Math.sin(i / n * 6.283) * 0.07, i / n, 0, col));
  const ct = [];
  for (let i = 0; i <= n; i++) ct.push(gs.v(Math.cos(i / n * 6.283) * 0.16, 0.42, Math.sin(i / n * 6.283) * 0.16, i / n, 1, stalkColor(0.6)));
  for (let i = 0; i < n; i++) { gs.tri(cb[i], ct[i], cb[i + 1]); gs.tri(cb[i + 1], ct[i], ct[i + 1]); }
  const lc = [0.9, 1.0, 0.85];
  const rows = 5, cols = 8;
  const ids = [];
  for (let i = 0; i <= rows; i++) {
    const a = i / rows * Math.PI;
    const row = [];
    for (let j = 0; j <= cols; j++) {
      const b = j / cols * Math.PI * 2;
      row.push(gl.v(0.26 * Math.sin(a) * Math.cos(b), 0.52 + 0.22 * Math.cos(a) * 0.8 - 0.04, 0.26 * Math.sin(a) * Math.sin(b), j / cols, i / rows, lc));
    }
    ids.push(row);
  }
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    gl.tri(ids[i][j], ids[i + 1][j], ids[i][j + 1]); gl.tri(ids[i][j + 1], ids[i + 1][j], ids[i + 1][j + 1]);
  }
  return { stalk: gs.build(), leaf: gl.build() };
}

// ------------------------------------------------------------------ 地面
function buildGround() {
  const xs = [];
  for (let x = -60; x < -6; x += 0.6) xs.push(x);
  for (let x = -6; x <= 6; x += 0.04) xs.push(+x.toFixed(3));
  for (let x = 6.6; x <= 60; x += 0.6) xs.push(x);
  const zs = [];
  for (let z = -3; z < 6; z += 0.05) zs.push(+z.toFixed(3));
  for (let z = 6; z < 30; z += 0.2) zs.push(+z.toFixed(3));
  for (let z = 30; z < 120; z += 1.0) zs.push(z);
  for (let z = 120; z <= 700; z += 12) zs.push(z);
  const nx = xs.length, nz = zs.length;
  const pos = new Float32Array(nx * nz * 3), uv = new Float32Array(nx * nz * 2), col = new Float32Array(nx * nz * 3);
  const TILE = 1.3;
  const rnd = mulberry32(3);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
    const idx = k * nx + i;
    const xp = xs[i], z = zs[k];
    const h = ridgeH(xp);
    pos[idx * 3] = -xp; pos[idx * 3 + 1] = h; pos[idx * 3 + 2] = z;
    uv[idx * 2] = xp / TILE; uv[idx * 2 + 1] = z / TILE;
    const f = 0.72 + 0.28 * (h / RIDGE.h);               // 沟底偏湿偏暗，垄顶偏干偏亮
    col[idx * 3] = f * 1.02; col[idx * 3 + 1] = f; col[idx * 3 + 2] = f * 0.98;
  }
  const idxs = [];
  for (let k = 0; k < nz - 1; k++) for (let i = 0; i < nx - 1; i++) {
    const a = k * nx + i, b = a + 1, c = a + nx, d = c + 1;
    idxs.push(a, b, c, b, d, c);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(idxs);
  g.computeVertexNormals();
  // 保证法线朝上
  let ny = 0; const na = g.getAttribute('normal');
  for (let i = 0; i < na.count; i += 997) ny += na.getY(i);
  if (ny < 0) { const ix = g.index.array; for (let i = 0; i < ix.length; i += 3) { const t = ix[i + 1]; ix[i + 1] = ix[i + 2]; ix[i + 2] = t; } g.computeVertexNormals(); }
  return g;
}

function loadTex(url, srgb) {
  return new Promise((res, rej) => new THREE.TextureLoader().load(url, t => {
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 16;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    res(t);
  }, undefined, rej));
}

async function groundMaterial() {
  const [d, n, r] = await Promise.all([loadTex('assets/soil_diff.jpg', true), loadTex('assets/soil_nor.jpg', false), loadTex('assets/soil_rough.jpg', false)]);
  const m = new THREE.MeshStandardMaterial({ map: d, normalMap: n, roughnessMap: r, roughness: 1.0, vertexColors: true, normalScale: new THREE.Vector2(1.2, 1.2) });
  m.onBeforeCompile = (sh) => {
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vW;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvW = (modelMatrix * vec4(transformed,1.0)).xyz;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>
varying vec3 vW;
float h21(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
float vn(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(h21(i),h21(i+vec2(1,0)),f.x), mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x), f.y); }
`).replace('#include <map_fragment>', `
vec4 s1 = texture2D( map, vMapUv );
vec4 s2 = texture2D( map, vMapUv * 0.31 + vec2(0.37, 0.11) );
float mm = vn(vW.xz * 0.9);
vec4 sampledDiffuseColor = mix(s1, s2, 0.30 + 0.25 * mm);
float macro = 0.82 + 0.36 * vn(vW.xz * 0.11) + 0.12 * (vn(vW.xz * 0.6) - 0.5);
sampledDiffuseColor.rgb *= macro;
{
  float xp = -vW.x;
  float t = mod(xp - 0.6, 1.2); float d = min(t, 1.2 - t);
  float zs = mix(34.0, 11.0, smoothstep(8.0, 11.0, abs(xp)));
  float fz = smoothstep(zs, zs + 14.0, vW.z);
  float rowm = (1.0 - smoothstep(0.17, 0.33, d)) * fz;
  vec3 crop = vec3(0.20, 0.31, 0.10) * (0.75 + 0.5 * vn(vW.xz * vec2(2.5, 0.9)));
  sampledDiffuseColor.rgb = mix(sampledDiffuseColor.rgb, crop, rowm * 0.92);
}
diffuseColor *= sampledDiffuseColor;
`);
  };
  return m;
}

// ------------------------------------------------------------------ 杂草
function tuftGeometry() {
  const g = new GB();
  const r = mulberry32(21);
  for (let b = 0; b < 9; b++) {
    const a = r() * 6.283, hh = 0.05 + r() * 0.09, lean = 0.02 + r() * 0.07, w = 0.004 + r() * 0.003;
    const dx = Math.cos(a), dz = Math.sin(a);
    const c0 = [0.30 + r() * 0.1, 0.42 + r() * 0.1, 0.16];
    const seg = 3, ids = [];
    for (let i = 0; i <= seg; i++) {
      const s = i / seg;
      const ww = w * (1 - s * 0.9);
      const px = dx * lean * s * s, pz = dz * lean * s * s, py = hh * s;
      const sx = -dz * ww, sz = dx * ww;
      ids.push([g.v(px - sx, py, pz - sz, 0, s, c0), g.v(px + sx, py, pz + sz, 1, s, c0)]);
    }
    for (let i = 0; i < seg; i++) { g.tri(ids[i][0], ids[i][1], ids[i + 1][0]); g.tri(ids[i][1], ids[i + 1][1], ids[i + 1][0]); }
  }
  return g.build();
}

// ------------------------------------------------------------------ 初始化
async function setup(lay) {
  layout = lay;
  RIDGE = lay.ridge;
  CAM.h = lay.cam.height; CAM.pitch = lay.cam.pitch_deg * Math.PI / 180; CAM.focal = lay.cam.focal; CAM.w = lay.cam.w; CAM.hh = lay.cam.h;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: lay.render.msaa !== false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(Math.round(CAM.w * (lay.render.ss || 1)), Math.round(CAM.hh * (lay.render.ss || 1)), false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = lay.render.exposure;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = lay.render.soft_shadow ? THREE.PCFSoftShadowMap : THREE.PCFShadowMap;

  scene = new THREE.Scene();
  const fov = 2 * Math.atan((CAM.hh / 2) / CAM.focal) * 180 / Math.PI;
  camera = new THREE.PerspectiveCamera(fov, CAM.w / CAM.hh, 0.05, 1500);

  // ---- HDRI（天空 / 地平线 / 环境光）
  const hdr = await new Promise((res, rej) => new RGBELoader().load('assets/sky.hdr', res, undefined, rej));
  hdr.mapping = THREE.EquirectangularReflectionMapping;
  // 找太阳方向、地平线雾色
  const W = hdr.image.width, Hh = hdr.image.height, data = hdr.image.data;
  const toF = (v) => (data instanceof Uint16Array ? THREE.DataUtils.fromHalfFloat(v) : v);
  let best = 0, bx = 0, by = 0;
  for (let y = 0; y < Hh * 0.45; y += 3) for (let x = 0; x < W; x += 3) {
    const i = (y * W + x) * 4;
    const l = toF(data[i]) * 0.3 + toF(data[i + 1]) * 0.59 + toF(data[i + 2]) * 0.11;
    if (l > best) { best = l; bx = x; by = y; }
  }
  const el = (0.5 - by / Hh) * Math.PI, azs = (bx / W - 0.5) * 2 * Math.PI;
  const sunHdri = V3(Math.cos(el) * Math.cos(azs), Math.sin(el), Math.cos(el) * Math.sin(azs));
  let fr = 0, fg = 0, fb = 0, fn = 0;
  for (let y = Math.floor(Hh * 0.5 - Hh * 0.035); y < Math.floor(Hh * 0.5 - Hh * 0.006); y += 2) for (let x = 0; x < W; x += 4) {
    const i = (y * W + x) * 4; fr += toF(data[i]); fg += toF(data[i + 1]); fb += toF(data[i + 2]); fn++;
  }
  const haze = new THREE.Color(fr / fn, fg / fn, fb / fn);
  // 旋转环境：让太阳位于相机左后方（JS 坐标 x>0 为画面左侧）
  const th0 = Math.atan2(sunHdri.z, sunHdri.x);
  const th1 = Math.atan2(-0.80, 0.60);
  const phi = th0 - th1 + (lay.render.env_rot_deg || 0) * Math.PI / 180;
  scene.background = hdr; scene.environment = hdr;
  scene.backgroundRotation.set(0, phi, 0); scene.environmentRotation.set(0, phi, 0);
  scene.backgroundIntensity = lay.render.bg_intensity; scene.environmentIntensity = lay.render.env_intensity;
  const sunDir = V3(Math.cos(el) * 0.6 / Math.hypot(0.6, 0.8), Math.sin(el), Math.cos(el) * -0.8 / Math.hypot(0.6, 0.8));
  sunLight = new THREE.DirectionalLight(0xfff4e0, lay.render.sun);
  sunLight.position.copy(sunDir.clone().multiplyScalar(40)).add(V3(0, 0, 6));
  sunLight.target.position.set(0, 0, 6);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.set(lay.render.shadow_map || 2048, lay.render.shadow_map || 2048);
  const sc = sunLight.shadow.camera;
  sc.left = -11; sc.right = 11; sc.top = 11; sc.bottom = -11; sc.near = 1; sc.far = 90;
  sunLight.shadow.intensity = 0.8; sunLight.shadow.bias = -0.0004; sunLight.shadow.normalBias = 0.012; sunLight.shadow.radius = 2.5;
  scene.add(sunLight, sunLight.target);
  scene.fog = new THREE.FogExp2(haze.clone().multiplyScalar(lay.render.fog_gain || 1.0), lay.render.fog_density);

  // ---- 地面
  const ground = new THREE.Mesh(buildGround(), await groundMaterial());
  ground.receiveShadow = true;
  scene.add(ground);

  // ---- 滴灌带
  const tapeMat = new THREE.MeshStandardMaterial({ color: 0x0b0b0c, roughness: 0.35, metalness: 0.0 });
  for (const xt of lay.tapes) {
    const g = new THREE.CylinderGeometry(0.0085, 0.0085, 80, 8).rotateX(Math.PI / 2);
    const m = new THREE.Mesh(g, tapeMat);
    m.position.set(-xt, ridgeH(xt) + 0.006, 40 + 0.0);
    m.receiveShadow = true;
    scene.add(m);
  }

  // ---- 芹菜实例（按 lod × variant 分组）
  const stalkTex = stalkTexture(), leafTex = leafTexture();
  const stalkMat = new THREE.MeshStandardMaterial({ vertexColors: true, map: stalkTex, bumpMap: stalkTex, bumpScale: 1.2, roughness: 0.42, metalness: 0 });
  const leafMat = new THREE.MeshStandardMaterial({ vertexColors: true, map: leafTex, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.48, metalness: 0, emissive: new THREE.Color(0x16300a), emissiveMap: leafTex });
  const farGain = lay.render.far_basic_gain || 0.55;
  const stalkBasic = new THREE.MeshBasicMaterial({ vertexColors: true, map: stalkTex, color: new THREE.Color(farGain, farGain, farGain) });
  const leafBasic = new THREE.MeshBasicMaterial({ vertexColors: true, map: leafTex, alphaTest: 0.5, side: THREE.DoubleSide, color: new THREE.Color(farGain * 0.8, farGain * 0.8, farGain * 0.8) });
  const nVar = lay.n_variants;
  const groups = new Map();
  lay.plants.forEach((p, i) => { const k = p.lod === 3 ? '3_0' : `${p.lod}_${p.variant}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(p); });
  const variantPts = [];
  const cache = new Map();
  let tris = 0;
  for (const [k, list] of groups) {
    const [lod, variant] = k.split('_').map(Number);
    const ck = k;
    let geo = cache.get(ck);
    if (!geo) { geo = lod === 3 ? makeFarPlant() : makePlant(variant, lod); cache.set(ck, geo); }
    if (lod === 0 && !variantPts[variant]) variantPts[variant] = geo.pts;
    const farBasic = lod >= 2 && lay.render.far_basic;
    const sm = farBasic ? stalkBasic : stalkMat, lm = farBasic ? leafBasic : leafMat;
    for (const [g, mat] of [[geo.stalk, sm], [geo.leaf, lm]]) {
      const im = new THREE.InstancedMesh(g, mat, list.length);
      const o = new THREE.Object3D(); const c = new THREE.Color();
      list.forEach((p, i) => {
        o.position.set(-p.x, p.y, p.z); o.rotation.set(0, p.yaw, 0); o.scale.setScalar(p.scale); o.updateMatrix();
        im.setMatrixAt(i, o.matrix);
        c.setRGB(p.tint[0], p.tint[1], p.tint[2]); im.setColorAt(i, c);
      });
      im.castShadow = lod < 2; im.receiveShadow = lod < 2;
      im.frustumCulled = false;
      scene.add(im);
      tris += (g.index.count / 3) * list.length;
    }
  }
  // ---- 杂草
  if (lay.tufts.length) {
    const tg = tuftGeometry();
    const tm = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, side: THREE.DoubleSide });
    const im = new THREE.InstancedMesh(tg, tm, lay.tufts.length);
    const o = new THREE.Object3D();
    lay.tufts.forEach((t, i) => { o.position.set(-t.x, ridgeH(t.x) - 0.005, t.z); o.rotation.set(0, t.yaw, 0); o.scale.setScalar(t.s); o.updateMatrix(); im.setMatrixAt(i, o.matrix); });
    im.receiveShadow = true; im.frustumCulled = false;
    scene.add(im);
  }
  window.__info = { tris, haze: haze.toArray(), sun: sunDir.toArray(), phi };
  return { variantPts, info: window.__info };
}

function renderFrame(camX) {
  const px = -camX;
  camera.position.set(px, CAM.h, 0);
  camera.up.set(0, 1, 0);
  camera.lookAt(px, CAM.h - Math.sin(CAM.pitch), Math.cos(CAM.pitch));
  renderer.render(scene, camera);
  return canvas.toDataURL('image/png');
}

window.setup = setup;
window.renderFrame = renderFrame;
window.ready = true;
