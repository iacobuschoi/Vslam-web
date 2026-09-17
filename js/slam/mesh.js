// Textured keyframe meshes: 2D Delaunay triangulation of the points seen in a keyframe,
// lifted to 3D and filtered so triangles do not bridge depth discontinuities.
import Delaunator from './vendor/delaunator.js';
import { poseApply } from './linalg.js';

/**
 * @param {Float64Array|Float32Array} pts2 pixel coordinates in the keyframe (2n)
 * @param {Float64Array|Float32Array} pts3 world coordinates (3n)
 * @param {number} n
 * @param {{R:Float64Array,t:Float64Array}} pose keyframe pose (camera-from-world)
 * @param {number} w image width
 * @param {number} h image height
 * @param {object} [opts]
 * @returns {{positions: Float32Array, uvs: Float32Array, indices: Uint32Array, vertexCount: number, triangleCount: number}|null}
 */
export function buildKeyframeMesh(pts2, pts3, n, pose, w, h, opts = {}) {
  if (n < 3) return null;
  const maxEdge2D = opts.maxEdge2D ?? 0.25 * Math.max(w, h);
  const maxDepthRatio = opts.maxDepthRatio ?? 0.28;
  const minCosView = opts.minCosView ?? 0.12;
  const minArea2D = opts.minArea2D ?? 2.0;
  const coords = new Float64Array(2 * n);
  for (let i = 0; i < 2 * n; i++) coords[i] = pts2[i];
  let del;
  try { del = new Delaunator(coords); } catch (_) { return null; }
  const tri = del.triangles;
  const depth = new Float64Array(n);
  const p = new Float64Array(3);
  for (let i = 0; i < n; i++) {
    poseApply(pose, pts3.subarray(3 * i, 3 * i + 3), p);
    depth[i] = p[2];
  }
  // Camera center in world coordinates.
  const C = new Float64Array(3);
  const R = pose.R, t = pose.t;
  C[0] = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
  C[1] = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
  C[2] = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
  const keep = [];
  const maxEdge2D2 = maxEdge2D * maxEdge2D;
  for (let k = 0; k < tri.length; k += 3) {
    const a = tri[k], b = tri[k + 1], c = tri[k + 2];
    if (depth[a] <= 0 || depth[b] <= 0 || depth[c] <= 0) continue;
    // 2D checks: sliver / oversized triangles.
    const ax = pts2[2 * a], ay = pts2[2 * a + 1], bx = pts2[2 * b], by = pts2[2 * b + 1], cx = pts2[2 * c], cy = pts2[2 * c + 1];
    const area2 = Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
    if (area2 < 2 * minArea2D) continue;
    const e2ab = (ax - bx) ** 2 + (ay - by) ** 2, e2bc = (bx - cx) ** 2 + (by - cy) ** 2, e2ca = (cx - ax) ** 2 + (cy - ay) ** 2;
    if (e2ab > maxEdge2D2 || e2bc > maxEdge2D2 || e2ca > maxEdge2D2) continue;
    // 3D checks: edge length relative to depth (depth discontinuities) and grazing angle.
    const A = pts3.subarray(3 * a, 3 * a + 3), B = pts3.subarray(3 * b, 3 * b + 3), Cc = pts3.subarray(3 * c, 3 * c + 3);
    const meanDepth = (depth[a] + depth[b] + depth[c]) / 3;
    const lim = maxDepthRatio * meanDepth;
    const dAB = Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]);
    const dBC = Math.hypot(B[0] - Cc[0], B[1] - Cc[1], B[2] - Cc[2]);
    const dCA = Math.hypot(Cc[0] - A[0], Cc[1] - A[1], Cc[2] - A[2]);
    if (dAB > lim || dBC > lim || dCA > lim) continue;
    // Normal vs. viewing direction.
    const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2];
    const vx = Cc[0] - A[0], vy = Cc[1] - A[1], vz = Cc[2] - A[2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const nn = Math.hypot(nx, ny, nz);
    if (nn < 1e-12) continue;
    const gx = (A[0] + B[0] + Cc[0]) / 3 - C[0], gy = (A[1] + B[1] + Cc[1]) / 3 - C[1], gz = (A[2] + B[2] + Cc[2]) / 3 - C[2];
    const gn = Math.hypot(gx, gy, gz);
    const cosv = Math.abs((nx * gx + ny * gy + nz * gz) / (nn * gn + 1e-18));
    if (cosv < minCosView) continue;
    keep.push(a, b, c);
  }
  if (keep.length === 0) return null;
  // Compact vertices.
  const remap = new Int32Array(n).fill(-1);
  let m = 0;
  for (const i of keep) if (remap[i] < 0) remap[i] = m++;
  const positions = new Float32Array(3 * m), uvs = new Float32Array(2 * m);
  for (let i = 0; i < n; i++) {
    const j = remap[i];
    if (j < 0) continue;
    positions[3 * j] = pts3[3 * i]; positions[3 * j + 1] = pts3[3 * i + 1]; positions[3 * j + 2] = pts3[3 * i + 2];
    uvs[2 * j] = pts2[2 * i] / w; uvs[2 * j + 1] = pts2[2 * i + 1] / h;
  }
  const indices = new Uint32Array(keep.length);
  for (let k = 0; k < keep.length; k++) indices[k] = remap[keep[k]];
  return { positions, uvs, indices, vertexCount: m, triangleCount: keep.length / 3 };
}
