// CPU ray-cast renderer of a textured room, used to test the SLAM pipeline end to end.
import * as L from '../js/slam/linalg.js';

function hash2(i, j, s) {
  let h = (i * 374761393 + j * 668265263 + s * 1442695041) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

// Procedural texture: colored rectangles of random size on a grid plus soft gradient.
function texture(u, v, seed) {
  const cell = 0.35;
  const i = Math.floor(u / cell), j = Math.floor(v / cell);
  const fu = u / cell - i, fv = v / cell - j;
  const base = [90 + 120 * hash2(i, j, seed), 90 + 120 * hash2(i, j, seed + 1), 90 + 120 * hash2(i, j, seed + 2)];
  const w = 0.25 + 0.6 * hash2(i, j, seed + 3), h = 0.25 + 0.6 * hash2(i, j, seed + 4);
  const ox = 0.05 + (0.9 - w) * hash2(i, j, seed + 5), oy = 0.05 + (0.9 - h) * hash2(i, j, seed + 6);
  if (fu > ox && fu < ox + w && fv > oy && fv < oy + h) {
    const k = hash2(i, j, seed + 7);
    return [30 + 200 * k, 30 + 200 * hash2(i, j, seed + 8), 30 + 200 * hash2(i, j, seed + 9)];
  }
  return base;
}

// Planes: { n: normal, d: offset (n·X = d), seed, axisU, axisV }
export function makeRoom() {
  return [
    { n: [0, 0, 1], d: 4.0, seed: 1, u: [1, 0, 0], v: [0, 1, 0] },     // front wall z=4
    { n: [1, 0, 0], d: 3.0, seed: 2, u: [0, 0, 1], v: [0, 1, 0] },     // right wall x=3
    { n: [-1, 0, 0], d: 3.0, seed: 3, u: [0, 0, 1], v: [0, 1, 0] },    // left wall x=-3
    { n: [0, 1, 0], d: 1.5, seed: 4, u: [1, 0, 0], v: [0, 0, 1] },     // floor y=1.5
    { n: [0, -1, 0], d: 1.5, seed: 5, u: [1, 0, 0], v: [0, 0, 1] },    // ceiling y=-1.5
    { n: [0, 0, -1], d: 2.5, seed: 6, u: [1, 0, 0], v: [0, 1, 0] },    // back wall z=-2.5
  ];
}

/**
 * Render an RGBA frame for camera pose T_cw (R, t) with intrinsics cam.
 */
export function renderRoom(planes, pose, cam, w, h, out = new Uint8ClampedArray(w * h * 4)) {
  const Rt = L.mat3Transpose(pose.R);
  const C = L.poseCenter(pose);
  const dir = new Float64Array(3), dc = new Float64Array(3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      dc[0] = (x + 0.5 - cam.cx) / cam.f; dc[1] = (y + 0.5 - cam.cy) / cam.f; dc[2] = 1;
      L.mat3MulVec(Rt, dc, dir);
      let best = Infinity, bp = null;
      for (const p of planes) {
        const denom = p.n[0] * dir[0] + p.n[1] * dir[1] + p.n[2] * dir[2];
        if (Math.abs(denom) < 1e-9) continue;
        const t = (p.d - (p.n[0] * C[0] + p.n[1] * C[1] + p.n[2] * C[2])) / denom;
        if (t > 1e-6 && t < best) { best = t; bp = p; }
      }
      const k = 4 * (y * w + x);
      if (!bp) { out[k] = out[k + 1] = out[k + 2] = 0; out[k + 3] = 255; continue; }
      const X = [C[0] + best * dir[0], C[1] + best * dir[1], C[2] + best * dir[2]];
      const u = X[0] * bp.u[0] + X[1] * bp.u[1] + X[2] * bp.u[2];
      const v = X[0] * bp.v[0] + X[1] * bp.v[1] + X[2] * bp.v[2];
      const c = texture(u + 10, v + 10, bp.seed);
      // Mild distance shading to mimic lighting.
      const shade = 1 / (1 + 0.03 * best);
      out[k] = c[0] * shade; out[k + 1] = c[1] * shade; out[k + 2] = c[2] * shade; out[k + 3] = 255;
    }
  }
  return out;
}

// Ground-truth camera path: sideways sweep with gentle yaw, then forward motion.
export function cameraPose(k) {
  const yaw = 0.35 * Math.sin(k * 0.025);
  const pitch = 0.08 * Math.sin(k * 0.04);
  const C = [0.9 * Math.sin(k * 0.03), 0.12 * Math.sin(k * 0.05), 0.5 * (1 - Math.cos(k * 0.02))];
  const R = L.mat3Mul(L.so3Exp([pitch, 0, 0]), L.so3Exp([0, yaw, 0])); // R_cw
  const t = L.scale3(L.mat3MulVec(R, C), -1);
  return L.poseCreate(R, t);
}

// Umeyama similarity alignment of point sets (3xN as flat arrays). Returns {s, R, t, rmse}.
export function alignSimilarity(src, dst, n) {
  const cs = [0, 0, 0], cd = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { cs[k] += src[3 * i + k] / n; cd[k] += dst[3 * i + k] / n; }
  const H = new Float64Array(9);
  let varS = 0;
  for (let i = 0; i < n; i++) {
    const a = [src[3 * i] - cs[0], src[3 * i + 1] - cs[1], src[3 * i + 2] - cs[2]];
    const b = [dst[3 * i] - cd[0], dst[3 * i + 1] - cd[1], dst[3 * i + 2] - cd[2]];
    varS += (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / n;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) H[3 * r + c] += b[r] * a[c] / n;
  }
  const { U, S, V } = L.svd3(H);
  const D = L.mat3Identity();
  const R0 = L.mat3Mul(U, L.mat3Transpose(V));
  if (L.mat3Det(R0) < 0) D[8] = -1;
  const R = L.mat3Mul(L.mat3Mul(U, D), L.mat3Transpose(V));
  const s = (S[0] + S[1] + S[2] * D[8]) / varS;
  const t = L.sub3(cd, L.scale3(L.mat3MulVec(R, cs), s));
  let se = 0;
  for (let i = 0; i < n; i++) {
    const p = L.add3(L.scale3(L.mat3MulVec(R, src.subarray(3 * i, 3 * i + 3)), s), t);
    se += Math.pow(p[0] - dst[3 * i], 2) + Math.pow(p[1] - dst[3 * i + 1], 2) + Math.pow(p[2] - dst[3 * i + 2], 2);
  }
  return { s, R, t, rmse: Math.sqrt(se / n) };
}
