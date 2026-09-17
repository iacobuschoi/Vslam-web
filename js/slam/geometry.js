// Multi-view geometry: essential matrix estimation, pose recovery, triangulation, PnP, pose refinement.
// Image points given to these functions are in normalized camera coordinates unless stated otherwise.

import {
  jacobiEigenSym, eigenColumn, svd3, mat3Mul, mat3Transpose, mat3Det, mat3MulVec, so3Exp,
  poseCreate, poseApply, poseCenter, solveLinear, orthonormalizeRotation, svd, cross3, norm3, dot3,
} from './linalg.js';

// ---------- Normalization ----------

// Hartley normalization of 2D points (arrays xs: Float64Array [x0,y0,x1,y1,...], idx: indices to use).
export function normalizeTransform(xs, idx) {
  const n = idx.length;
  let cx = 0, cy = 0;
  for (let k = 0; k < n; k++) { cx += xs[2 * idx[k]]; cy += xs[2 * idx[k] + 1]; }
  cx /= n; cy /= n;
  let md = 0;
  for (let k = 0; k < n; k++) md += Math.hypot(xs[2 * idx[k]] - cx, xs[2 * idx[k] + 1] - cy);
  md /= n;
  const s = md > 1e-12 ? Math.SQRT2 / md : 1;
  // T = [s 0 -s cx; 0 s -s cy; 0 0 1]
  return new Float64Array([s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1]);
}

// ---------- Essential matrix ----------

// Eight-point algorithm for a subset idx of correspondences (x1 -> x2). Returns E (3x3) or null.
export function eightPoint(x1, x2, idx) {
  const n = idx.length;
  if (n < 8) return null;
  const T1 = normalizeTransform(x1, idx), T2 = normalizeTransform(x2, idx);
  const AtA = new Float64Array(81);
  const row = new Float64Array(9);
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    const u1 = T1[0] * x1[2 * i] + T1[2], v1 = T1[4] * x1[2 * i + 1] + T1[5];
    const u2 = T2[0] * x2[2 * i] + T2[2], v2 = T2[4] * x2[2 * i + 1] + T2[5];
    row[0] = u2 * u1; row[1] = u2 * v1; row[2] = u2;
    row[3] = v2 * u1; row[4] = v2 * v1; row[5] = v2;
    row[6] = u1; row[7] = v1; row[8] = 1;
    for (let a = 0; a < 9; a++) {
      const ra = row[a];
      for (let b = a; b < 9; b++) AtA[a * 9 + b] += ra * row[b];
    }
  }
  for (let a = 0; a < 9; a++) for (let b = 0; b < a; b++) AtA[a * 9 + b] = AtA[b * 9 + a];
  const { vectors } = jacobiEigenSym(AtA, 9);
  const f = eigenColumn(vectors, 9, 0);
  // Denormalize: F = T2^T F' T1
  const Fn = Float64Array.from(f);
  const F = mat3Mul(mat3Mul(mat3Transpose(T2), Fn), T1);
  // Project onto the essential manifold: singular values (1, 1, 0).
  const { U, V } = svd3(F);
  const D = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 0]);
  const E = mat3Mul(mat3Mul(U, D), mat3Transpose(V));
  return E;
}

// Sampson distance (squared, in normalized units) for correspondence i.
export function sampsonError(E, x1, x2, i) {
  const a = x1[2 * i], b = x1[2 * i + 1], c = x2[2 * i], d = x2[2 * i + 1];
  const Ex0 = E[0] * a + E[1] * b + E[2], Ex1 = E[3] * a + E[4] * b + E[5], Ex2 = E[6] * a + E[7] * b + E[8];
  const Etx0 = E[0] * c + E[3] * d + E[6], Etx1 = E[1] * c + E[4] * d + E[7];
  const num = c * Ex0 + d * Ex1 + Ex2;
  const den = Ex0 * Ex0 + Ex1 * Ex1 + Etx0 * Etx0 + Etx1 * Etx1;
  return den > 1e-20 ? (num * num) / den : Infinity;
}

/**
 * RANSAC essential matrix. x1, x2: Float64Array normalized coords (n pairs).
 * thresh: Sampson error threshold (normalized units squared, e.g. (px/f)^2).
 * Returns { E, inliers: Uint8Array, nInliers } or null.
 */
export function findEssentialRansac(x1, x2, n, thresh, rng, maxIter = 400, minIter = 200) {
  if (n < 8) return null;
  const all = new Int32Array(n);
  for (let i = 0; i < n; i++) all[i] = i;
  let bestE = null, bestInliers = null, bestCount = 0, bestScore = -Infinity;
  const sample = new Int32Array(8);
  let iters = maxIter;
  for (let it = 0; it < iters; it++) {
    // Draw 8 distinct indices.
    for (let k = 0; k < 8; k++) {
      let v, dup;
      do {
        v = Math.floor(rng() * n); dup = false;
        for (let j = 0; j < k; j++) if (sample[j] === v) { dup = true; break; }
      } while (dup);
      sample[k] = v;
    }
    const E = eightPoint(x1, x2, sample);
    if (!E) continue;
    let count = 0, score = 0;
    for (let i = 0; i < n; i++) {
      const e = sampsonError(E, x1, x2, i);
      if (e < thresh) { count++; score += thresh - e; }
    }
    if (score > bestScore) {
      bestScore = score; bestCount = count; bestE = E;
      // Adaptive iteration count.
      const w = count / n;
      const denom = Math.log(Math.max(1 - Math.pow(w, 8), 1e-12));
      const need = Math.ceil(Math.log(0.01) / denom);
      iters = Math.max(minIter, Math.min(maxIter, need));
    }
  }
  if (!bestE || bestCount < 8) return null;
  // Refit on all inliers.
  const inl = [];
  for (let i = 0; i < n; i++) if (sampsonError(bestE, x1, x2, i) < thresh) inl.push(i);
  const E2 = eightPoint(x1, x2, Int32Array.from(inl));
  let E = bestE;
  if (E2) {
    let c2 = 0;
    for (let i = 0; i < n; i++) if (sampsonError(E2, x1, x2, i) < thresh) c2++;
    if (c2 >= bestCount) E = E2;
  }
  const inliers = new Uint8Array(n);
  let nInliers = 0;
  for (let i = 0; i < n; i++) if (sampsonError(E, x1, x2, i) < thresh) { inliers[i] = 1; nInliers++; }
  return { E, inliers, nInliers };
}

// Decompose E into 4 candidate (R, t) pairs. t has unit norm.
export function decomposeEssential(E) {
  const { U, V } = svd3(E);
  if (mat3Det(U) < 0) for (let i = 0; i < 9; i++) U[i] = -U[i];
  if (mat3Det(V) < 0) for (let i = 0; i < 9; i++) V[i] = -V[i];
  const W = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 1]);
  const Wt = mat3Transpose(W);
  const Vt = mat3Transpose(V);
  const R1 = mat3Mul(mat3Mul(U, W), Vt);
  const R2 = mat3Mul(mat3Mul(U, Wt), Vt);
  const t = new Float64Array([U[2], U[5], U[8]]);
  const tn = new Float64Array([-t[0], -t[1], -t[2]]);
  return [
    poseCreate(R1, t), poseCreate(R1, tn), poseCreate(R2, t), poseCreate(R2, tn),
  ];
}

// ---------- Triangulation ----------

/**
 * Linear (DLT) triangulation from N views.
 * poses: array of {R,t} camera-from-world; xs: array of [x, y] normalized coords.
 * Returns world point Float64Array(3) or null.
 */
export function triangulateN(poses, xs) {
  const n = poses.length;
  const AtA = new Float64Array(16);
  const r0 = new Float64Array(4), r1 = new Float64Array(4);
  for (let k = 0; k < n; k++) {
    const R = poses[k].R, t = poses[k].t;
    const x = xs[k][0], y = xs[k][1];
    // rows: x*P[2] - P[0], y*P[2] - P[1]
    r0[0] = x * R[6] - R[0]; r0[1] = x * R[7] - R[1]; r0[2] = x * R[8] - R[2]; r0[3] = x * t[2] - t[0];
    r1[0] = y * R[6] - R[3]; r1[1] = y * R[7] - R[4]; r1[2] = y * R[8] - R[5]; r1[3] = y * t[2] - t[1];
    for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) AtA[a * 4 + b] += r0[a] * r0[b] + r1[a] * r1[b];
  }
  const { vectors } = jacobiEigenSym(AtA, 4);
  const v = eigenColumn(vectors, 4, 0);
  if (Math.abs(v[3]) < 1e-12) return null;
  return new Float64Array([v[0] / v[3], v[1] / v[3], v[2] / v[3]]);
}

export function triangulate2(P1, P2, x1, y1, x2, y2) {
  return triangulateN([P1, P2], [[x1, y1], [x2, y2]]);
}

// Angle (radians) between the rays from two camera centers to X.
export function parallaxAngle(C1, C2, X) {
  const a = new Float64Array([X[0] - C1[0], X[1] - C1[1], X[2] - C1[2]]);
  const b = new Float64Array([X[0] - C2[0], X[1] - C2[1], X[2] - C2[2]]);
  const d = dot3(a, b) / (norm3(a) * norm3(b) + 1e-18);
  return Math.acos(Math.max(-1, Math.min(1, d)));
}

// Squared reprojection error in normalized units for world point X under pose P.
export function reprojError2(P, X, x, y) {
  const p = poseApply(P, X);
  if (p[2] <= 1e-9) return Infinity;
  const dx = p[0] / p[2] - x, dy = p[1] / p[2] - y;
  return dx * dx + dy * dy;
}

/**
 * Choose the correct (R,t) among the essential decompositions by cheirality.
 * x1,x2 normalized coords; inliers mask. reprojThresh: normalized (squared) threshold.
 * Returns { pose, points: Array<{i, X}>, nGood, medianParallax, ratio }.
 */
export function selectPoseFromEssential(E, x1, x2, n, inliers, reprojThresh, minParallaxRad = 0.0) {
  const cands = decomposeEssential(E);
  return selectPoseFromCandidates(cands, x1, x2, n, inliers, reprojThresh, minParallaxRad);
}


/**
 * Choose the correct (R,t) among candidate poses by cheirality / reprojection / parallax.
 * Returns { pose, points: Array<{i, X, parallax}>, nGood, medianParallax, ratio }.
 */
export function selectPoseFromCandidates(cands, x1, x2, n, inliers, reprojThresh, minParallaxRad = 0.0) {
  const P1 = poseCreate();
  const C1 = new Float64Array(3);
  let best = null;
  const results = [];
  for (const P2 of cands) {
    const C2 = poseCenter(P2);
    const pts = [];
    const parallaxes = [];
    let nGood = 0;
    for (let i = 0; i < n; i++) {
      if (!inliers[i]) continue;
      const X = triangulate2(P1, P2, x1[2 * i], x1[2 * i + 1], x2[2 * i], x2[2 * i + 1]);
      if (!X || !isFinite(X[0]) || !isFinite(X[1]) || !isFinite(X[2])) continue;
      const z1 = X[2];
      const p2 = poseApply(P2, X);
      if (z1 <= 0 || p2[2] <= 0) continue;
      const e1 = reprojError2(P1, X, x1[2 * i], x1[2 * i + 1]);
      const e2 = reprojError2(P2, X, x2[2 * i], x2[2 * i + 1]);
      if (e1 > reprojThresh || e2 > reprojThresh) continue;
      const par = parallaxAngle(C1, C2, X);
      parallaxes.push(par);
      if (par >= minParallaxRad) { nGood++; pts.push({ i, X, parallax: par }); }
    }
    parallaxes.sort((a, b) => a - b);
    const medianParallax = parallaxes.length ? parallaxes[parallaxes.length >> 1] : 0;
    const res = { pose: P2, points: pts, nGood, medianParallax };
    results.push(res);
    if (!best || nGood > best.nGood) best = res;
  }
  if (!best) return null;
  results.sort((a, b) => b.nGood - a.nGood);
  const second = results.length > 1 ? results[1].nGood : 0;
  best.ratio = best.nGood > 0 ? second / best.nGood : 1;
  return best;
}

// ---------- Homography ----------

// Four-point (or more) DLT homography x2 ~ H x1 for the index subset idx.
export function homographyDLT(x1, x2, idx) {
  const n = idx.length;
  if (n < 4) return null;
  const T1 = normalizeTransform(x1, idx), T2 = normalizeTransform(x2, idx);
  const AtA = new Float64Array(81);
  const r0 = new Float64Array(9), r1 = new Float64Array(9);
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    const u1 = T1[0] * x1[2 * i] + T1[2], v1 = T1[4] * x1[2 * i + 1] + T1[5];
    const u2 = T2[0] * x2[2 * i] + T2[2], v2 = T2[4] * x2[2 * i + 1] + T2[5];
    r0[0] = -u1; r0[1] = -v1; r0[2] = -1; r0[3] = 0; r0[4] = 0; r0[5] = 0; r0[6] = u2 * u1; r0[7] = u2 * v1; r0[8] = u2;
    r1[0] = 0; r1[1] = 0; r1[2] = 0; r1[3] = -u1; r1[4] = -v1; r1[5] = -1; r1[6] = v2 * u1; r1[7] = v2 * v1; r1[8] = v2;
    for (let a = 0; a < 9; a++) for (let b = 0; b < 9; b++) AtA[a * 9 + b] += r0[a] * r0[b] + r1[a] * r1[b];
  }
  const { vectors } = jacobiEigenSym(AtA, 9);
  const hn = eigenColumn(vectors, 9, 0);
  // H = T2^-1 H' T1
  const s2 = T2[0];
  const T2inv = new Float64Array([1 / s2, 0, -T2[2] / s2, 0, 1 / s2, -T2[5] / s2, 0, 0, 1]);
  const H = mat3Mul(mat3Mul(T2inv, Float64Array.from(hn)), T1);
  // Normalize scale so that the Frobenius norm is sqrt(3) (roughly rotation-like).
  let fro = 0;
  for (let i = 0; i < 9; i++) fro += H[i] * H[i];
  const sc = Math.sqrt(3 / Math.max(fro, 1e-30));
  for (let i = 0; i < 9; i++) H[i] *= sc;
  if (H[8] < 0) for (let i = 0; i < 9; i++) H[i] = -H[i];
  return H;
}

function mat3Inverse(A) {
  const det = mat3Det(A);
  if (Math.abs(det) < 1e-18) return null;
  const inv = new Float64Array(9);
  inv[0] = (A[4] * A[8] - A[5] * A[7]) / det; inv[1] = (A[2] * A[7] - A[1] * A[8]) / det; inv[2] = (A[1] * A[5] - A[2] * A[4]) / det;
  inv[3] = (A[5] * A[6] - A[3] * A[8]) / det; inv[4] = (A[0] * A[8] - A[2] * A[6]) / det; inv[5] = (A[2] * A[3] - A[0] * A[5]) / det;
  inv[6] = (A[3] * A[7] - A[4] * A[6]) / det; inv[7] = (A[1] * A[6] - A[0] * A[7]) / det; inv[8] = (A[0] * A[4] - A[1] * A[3]) / det;
  return inv;
}

// Transfer error of x1 -> x2 through H (squared, normalized units).
function transferError2(H, x1, y1, x2, y2) {
  const w = H[6] * x1 + H[7] * y1 + H[8];
  if (Math.abs(w) < 1e-12) return Infinity;
  const px = (H[0] * x1 + H[1] * y1 + H[2]) / w, py = (H[3] * x1 + H[4] * y1 + H[5]) / w;
  const dx = px - x2, dy = py - y2;
  return dx * dx + dy * dy;
}

/**
 * RANSAC homography. Errors are symmetric transfer errors; both directions must be below thresh.
 * Returns { H, inliers, nInliers, score } or null.
 */
export function findHomographyRansac(x1, x2, n, thresh, rng, maxIter = 300, minIter = 150) {
  if (n < 4) return null;
  const sample = new Int32Array(4);
  let bestH = null, bestScore = -Infinity, bestCount = 0;
  let iters = maxIter;
  for (let it = 0; it < iters; it++) {
    for (let k = 0; k < 4; k++) {
      let v, dup;
      do {
        v = Math.floor(rng() * n); dup = false;
        for (let j = 0; j < k; j++) if (sample[j] === v) { dup = true; break; }
      } while (dup);
      sample[k] = v;
    }
    const H = homographyDLT(x1, x2, sample);
    if (!H) continue;
    const Hinv = mat3Inverse(H);
    if (!Hinv) continue;
    let count = 0, score = 0;
    for (let i = 0; i < n; i++) {
      const e1 = transferError2(H, x1[2 * i], x1[2 * i + 1], x2[2 * i], x2[2 * i + 1]);
      const e2 = transferError2(Hinv, x2[2 * i], x2[2 * i + 1], x1[2 * i], x1[2 * i + 1]);
      if (e1 < thresh && e2 < thresh) { count++; score += (thresh - e1) + (thresh - e2); }
    }
    if (score > bestScore) {
      bestScore = score; bestCount = count; bestH = H;
      const w = count / n;
      const denom = Math.log(Math.max(1 - Math.pow(w, 4), 1e-12));
      const need = Math.ceil(Math.log(0.01) / denom);
      iters = Math.max(minIter, Math.min(maxIter, need));
    }
  }
  if (!bestH || bestCount < 4) return null;
  // Refit on inliers.
  const Hinv0 = mat3Inverse(bestH);
  const inl = [];
  for (let i = 0; i < n; i++) {
    const e1 = transferError2(bestH, x1[2 * i], x1[2 * i + 1], x2[2 * i], x2[2 * i + 1]);
    const e2 = transferError2(Hinv0, x2[2 * i], x2[2 * i + 1], x1[2 * i], x1[2 * i + 1]);
    if (e1 < thresh && e2 < thresh) inl.push(i);
  }
  let H = bestH;
  const H2 = homographyDLT(x1, x2, Int32Array.from(inl));
  if (H2) {
    const H2inv = mat3Inverse(H2);
    if (H2inv) {
      let c2 = 0;
      for (let i = 0; i < n; i++) {
        const e1 = transferError2(H2, x1[2 * i], x1[2 * i + 1], x2[2 * i], x2[2 * i + 1]);
        const e2 = transferError2(H2inv, x2[2 * i], x2[2 * i + 1], x1[2 * i], x1[2 * i + 1]);
        if (e1 < thresh && e2 < thresh) c2++;
      }
      if (c2 >= bestCount) H = H2;
    }
  }
  const Hinv = mat3Inverse(H);
  const inliers = new Uint8Array(n);
  let nInliers = 0, score = 0;
  for (let i = 0; i < n; i++) {
    const e1 = transferError2(H, x1[2 * i], x1[2 * i + 1], x2[2 * i], x2[2 * i + 1]);
    const e2 = transferError2(Hinv, x2[2 * i], x2[2 * i + 1], x1[2 * i], x1[2 * i + 1]);
    if (e1 < thresh && e2 < thresh) { inliers[i] = 1; nInliers++; score += (thresh - e1) + (thresh - e2); }
  }
  return { H, inliers, nInliers, score };
}

/**
 * Decompose a homography (normalized coordinates, H = R + t n^T / d) into up to 8 (R, t) candidates
 * following Faugeras & Lustman. Returns array of poses (t normalized); empty if degenerate.
 */
export function decomposeHomography(H) {
  const { U, S, V } = svd3(H);
  const d1 = S[0], d2 = S[1], d3 = S[2];
  if (d1 / d2 < 1.00001 || d2 / d3 < 1.00001) return [];
  const s = mat3Det(U) * mat3Det(V);
  const Vt = mat3Transpose(V);
  const aux1 = Math.sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
  const aux3 = Math.sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
  const x1s = [aux1, aux1, -aux1, -aux1];
  const x3s = [aux3, -aux3, aux3, -aux3];
  const out = [];
  // Case d' > 0
  const auxSt = Math.sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);
  const ct = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
  const sts = [auxSt, -auxSt, -auxSt, auxSt];
  for (let i = 0; i < 4; i++) {
    const Rp = new Float64Array([ct, 0, -sts[i], 0, 1, 0, sts[i], 0, ct]);
    const R = mat3Mul(mat3Mul(U, Rp), Vt);
    for (let k = 0; k < 9; k++) R[k] *= s;
    const tp = new Float64Array([x1s[i], 0, -x3s[i]]);
    const t = mat3MulVec(U, tp);
    const nt = norm3(t) || 1;
    out.push(poseCreate(R, [t[0] / nt, t[1] / nt, t[2] / nt]));
  }
  // Case d' < 0
  const auxSp = Math.sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);
  const cp = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
  const sps = [auxSp, auxSp, -auxSp, -auxSp];
  for (let i = 0; i < 4; i++) {
    const Rp = new Float64Array([cp, 0, sps[i], 0, -1, 0, sps[i], 0, -cp]);
    const R = mat3Mul(mat3Mul(U, Rp), Vt);
    for (let k = 0; k < 9; k++) R[k] *= s;
    const tp = new Float64Array([x1s[i], 0, x3s[i]]);
    const t = mat3MulVec(U, tp);
    const nt = norm3(t) || 1;
    out.push(poseCreate(R, [t[0] / nt, t[1] / nt, t[2] / nt]));
  }
  return out;
}

export function selectPoseFromHomography(H, x1, x2, n, inliers, reprojThresh, minParallaxRad = 0.0) {
  const cands = decomposeHomography(H);
  if (!cands.length) return null;
  return selectPoseFromCandidates(cands, x1, x2, n, inliers, reprojThresh, minParallaxRad);
}

// ---------- PnP ----------

/**
 * DLT PnP from >= 6 correspondences. Xs: Float64Array world points (3n), xs: normalized (2n), idx: indices.
 * Returns pose {R, t} or null.
 */
export function pnpDLT(Xs, xs, idx) {
  const n = idx.length;
  if (n < 6) return null;
  // Normalize world points.
  let cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < n; k++) { const i = idx[k]; cx += Xs[3 * i]; cy += Xs[3 * i + 1]; cz += Xs[3 * i + 2]; }
  cx /= n; cy /= n; cz /= n;
  let md = 0;
  for (let k = 0; k < n; k++) { const i = idx[k]; md += Math.hypot(Xs[3 * i] - cx, Xs[3 * i + 1] - cy, Xs[3 * i + 2] - cz); }
  md /= n;
  const sw = md > 1e-12 ? Math.sqrt(3) / md : 1;
  const Ti = normalizeTransform(xs, idx);
  const AtA = new Float64Array(144);
  const r0 = new Float64Array(12), r1 = new Float64Array(12);
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    const X = (Xs[3 * i] - cx) * sw, Y = (Xs[3 * i + 1] - cy) * sw, Z = (Xs[3 * i + 2] - cz) * sw;
    const u = Ti[0] * xs[2 * i] + Ti[2], v = Ti[4] * xs[2 * i + 1] + Ti[5];
    r0.fill(0); r1.fill(0);
    r0[0] = X; r0[1] = Y; r0[2] = Z; r0[3] = 1; r0[8] = -u * X; r0[9] = -u * Y; r0[10] = -u * Z; r0[11] = -u;
    r1[4] = X; r1[5] = Y; r1[6] = Z; r1[7] = 1; r1[8] = -v * X; r1[9] = -v * Y; r1[10] = -v * Z; r1[11] = -v;
    for (let a = 0; a < 12; a++) for (let b = 0; b < 12; b++) AtA[a * 12 + b] += r0[a] * r0[b] + r1[a] * r1[b];
  }
  const { vectors } = jacobiEigenSym(AtA, 12);
  const p = eigenColumn(vectors, 12, 0);
  // P' (3x4) in normalized frames: x' = P' X'. Undo: x = Ti^-1 P' Tw X, with Tw = [sw I, -sw c; 0 1].
  const Pn = [p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11]];
  // Multiply by Tw on the right: columns 0..2 scaled by sw, column 3 = P'[:,3] - sw*(P'[:,0:3] c)
  const P = new Float64Array(12);
  for (let r = 0; r < 3; r++) {
    P[4 * r] = Pn[4 * r] * sw; P[4 * r + 1] = Pn[4 * r + 1] * sw; P[4 * r + 2] = Pn[4 * r + 2] * sw;
    P[4 * r + 3] = Pn[4 * r + 3] - sw * (Pn[4 * r] * cx + Pn[4 * r + 1] * cy + Pn[4 * r + 2] * cz);
  }
  // Ti^-1 = [1/s 0 cx; 0 1/s cy; 0 0 1]
  const is = 1 / Ti[0], icx = -Ti[2] * is, icy = -Ti[5] * is;
  const Pf = new Float64Array(12);
  for (let c = 0; c < 4; c++) {
    Pf[c] = is * P[c] + icx * P[8 + c];
    Pf[4 + c] = is * P[4 + c] + icy * P[8 + c];
    Pf[8 + c] = P[8 + c];
  }
  // Fix sign so that most points have positive depth.
  let pos = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    const z = Pf[8] * Xs[3 * i] + Pf[9] * Xs[3 * i + 1] + Pf[10] * Xs[3 * i + 2] + Pf[11];
    if (z > 0) pos++;
  }
  if (pos * 2 < n) for (let i = 0; i < 12; i++) Pf[i] = -Pf[i];
  const M = new Float64Array([Pf[0], Pf[1], Pf[2], Pf[4], Pf[5], Pf[6], Pf[8], Pf[9], Pf[10]]);
  const { S } = svd3(M);
  const scale = (S[0] + S[1] + S[2]) / 3;
  if (!(scale > 1e-12)) return null;
  const R = orthonormalizeRotation(M);
  const t = new Float64Array([Pf[3] / scale, Pf[7] / scale, Pf[11] / scale]);
  return poseCreate(R, t);
}

// Reprojection error in pixels for pose P, world point X, pixel observation (u, v). Camera (f, cx, cy).
export function reprojErrorPx(P, X, u, v, f, cx, cy) {
  const p = poseApply(P, X);
  if (p[2] <= 1e-9) return Infinity;
  const du = f * p[0] / p[2] + cx - u, dv = f * p[1] / p[2] + cy - v;
  return Math.sqrt(du * du + dv * dv);
}

/**
 * Refine a pose with robust Gauss-Newton / Levenberg-Marquardt on pixel reprojection errors.
 * Xs: Float64Array (3n) world points; uv: Float64Array (2n) pixel observations.
 * Returns { pose, inliers: Uint8Array, nInliers, rmse }.
 */
export function refinePose(init, Xs, uv, n, cam, opts = {}) {
  const iters = opts.iters ?? 10;
  const huber = opts.huber ?? 2.5;
  const inlierThresh = opts.inlierThresh ?? 3.0;
  const mask = opts.mask ?? null;
  const f = cam.f, cx = cam.cx, cy = cam.cy;
  let R = Float64Array.from(init.R), t = Float64Array.from(init.t);
  const H = new Float64Array(36), b = new Float64Array(6);
  const J0 = new Float64Array(6), J1 = new Float64Array(6);
  let lambda = 1e-3;
  let lastChi2 = Infinity;
  const evalChi2 = (R_, t_) => {
    let chi = 0;
    for (let i = 0; i < n; i++) {
      if (mask && !mask[i]) continue;
      const X0 = Xs[3 * i], X1 = Xs[3 * i + 1], X2 = Xs[3 * i + 2];
      const px = R_[0] * X0 + R_[1] * X1 + R_[2] * X2 + t_[0];
      const py = R_[3] * X0 + R_[4] * X1 + R_[5] * X2 + t_[1];
      const pz = R_[6] * X0 + R_[7] * X1 + R_[8] * X2 + t_[2];
      if (pz <= 1e-6) { chi += huber * huber * 4; continue; }
      const ru = f * px / pz + cx - uv[2 * i], rv = f * py / pz + cy - uv[2 * i + 1];
      const e = Math.sqrt(ru * ru + rv * rv);
      chi += e <= huber ? e * e : huber * (2 * e - huber);
    }
    return chi;
  };
  lastChi2 = evalChi2(R, t);
  for (let it = 0; it < iters; it++) {
    H.fill(0); b.fill(0);
    for (let i = 0; i < n; i++) {
      if (mask && !mask[i]) continue;
      const X0 = Xs[3 * i], X1 = Xs[3 * i + 1], X2 = Xs[3 * i + 2];
      const px = R[0] * X0 + R[1] * X1 + R[2] * X2 + t[0];
      const py = R[3] * X0 + R[4] * X1 + R[5] * X2 + t[1];
      const pz = R[6] * X0 + R[7] * X1 + R[8] * X2 + t[2];
      if (pz <= 1e-6) continue;
      const iz = 1 / pz, iz2 = iz * iz;
      const ru = f * px * iz + cx - uv[2 * i], rv = f * py * iz + cy - uv[2 * i + 1];
      const e = Math.sqrt(ru * ru + rv * rv);
      const w = e <= huber ? 1 : huber / e;
      // Jacobian rows w.r.t. [dt(3), dtheta(3)] for left-multiplicative update.
      J0[0] = f * iz; J0[1] = 0; J0[2] = -f * px * iz2;
      J0[3] = -f * px * py * iz2; J0[4] = f + f * px * px * iz2; J0[5] = -f * py * iz;
      J1[0] = 0; J1[1] = f * iz; J1[2] = -f * py * iz2;
      J1[3] = -f - f * py * py * iz2; J1[4] = f * px * py * iz2; J1[5] = f * px * iz;
      for (let a = 0; a < 6; a++) {
        b[a] += w * (J0[a] * ru + J1[a] * rv);
        for (let c = a; c < 6; c++) H[a * 6 + c] += w * (J0[a] * J0[c] + J1[a] * J1[c]);
      }
    }
    for (let a = 0; a < 6; a++) for (let c = 0; c < a; c++) H[a * 6 + c] = H[c * 6 + a];
    let improved = false;
    for (let attempt = 0; attempt < 5; attempt++) {
      const Hd = Float64Array.from(H);
      for (let a = 0; a < 6; a++) Hd[a * 6 + a] += lambda * (H[a * 6 + a] + 1e-9);
      const nb = new Float64Array(6);
      for (let a = 0; a < 6; a++) nb[a] = -b[a];
      const d = solveLinear(Hd, nb, 6);
      if (!d) { lambda *= 10; continue; }
      const Rd = so3Exp([d[3], d[4], d[5]]);
      const R2 = mat3Mul(Rd, R);
      const t2 = mat3MulVec(Rd, t);
      t2[0] += d[0]; t2[1] += d[1]; t2[2] += d[2];
      const chi = evalChi2(R2, t2);
      if (chi < lastChi2) {
        const step = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3] + d[4] * d[4] + d[5] * d[5];
        R = R2; t = t2; lastChi2 = chi; lambda = Math.max(lambda * 0.3, 1e-6); improved = true;
        if (step < 1e-12) it = iters;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  // Keep R exactly on SO(3); tiny numerical drift otherwise compounds through the motion model.
  R = orthonormalizeRotation(R);
  const pose = poseCreate(R, t);
  const inliers = new Uint8Array(n);
  let nInliers = 0, se = 0;
  for (let i = 0; i < n; i++) {
    if (mask && !mask[i]) continue;
    const e = reprojErrorPx(pose, Xs.subarray(3 * i, 3 * i + 3), uv[2 * i], uv[2 * i + 1], f, cx, cy);
    if (e < inlierThresh) { inliers[i] = 1; nInliers++; se += e * e; }
  }
  return { pose, inliers, nInliers, rmse: nInliers ? Math.sqrt(se / nInliers) : Infinity };
}

/**
 * RANSAC PnP (DLT hypotheses + robust refinement).
 * Xs: world points (3n), uv: pixels (2n), cam {f,cx,cy}.
 * Returns { pose, inliers, nInliers } or null.
 */
export function pnpRansac(Xs, uv, n, cam, rng, opts = {}) {
  const thresh = opts.thresh ?? 4.0;
  const maxIter = opts.maxIter ?? 200;
  const sampleSize = Math.min(opts.sampleSize ?? 6, n);
  if (n < 6) return null;
  const f = cam.f, cx = cam.cx, cy = cam.cy;
  // Normalized coordinates for DLT.
  const xs = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) { xs[2 * i] = (uv[2 * i] - cx) / f; xs[2 * i + 1] = (uv[2 * i + 1] - cy) / f; }
  const sample = new Int32Array(sampleSize);
  let best = null, bestCount = 0;
  let iters = maxIter;
  for (let it = 0; it < iters; it++) {
    for (let k = 0; k < sampleSize; k++) {
      let v, dup;
      do {
        v = Math.floor(rng() * n); dup = false;
        for (let j = 0; j < k; j++) if (sample[j] === v) { dup = true; break; }
      } while (dup);
      sample[k] = v;
    }
    const P = pnpDLT(Xs, xs, sample);
    if (!P) continue;
    let count = 0;
    for (let i = 0; i < n; i++) {
      if (reprojErrorPx(P, Xs.subarray(3 * i, 3 * i + 3), uv[2 * i], uv[2 * i + 1], f, cx, cy) < thresh) count++;
    }
    if (count > bestCount) {
      bestCount = count; best = P;
      const w = count / n;
      const denom = Math.log(Math.max(1 - Math.pow(w, sampleSize), 1e-12));
      const need = Math.ceil(Math.log(0.01) / denom);
      iters = Math.max(20, Math.min(maxIter, need));
    }
  }
  if (!best || bestCount < 6) return null;
  // Refine on the preliminary inliers, then recompute the inlier set.
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (reprojErrorPx(best, Xs.subarray(3 * i, 3 * i + 3), uv[2 * i], uv[2 * i + 1], f, cx, cy) < thresh) mask[i] = 1;
  }
  let r = refinePose(best, Xs, uv, n, cam, { mask, iters: 10, huber: thresh * 0.6, inlierThresh: thresh });
  const r2 = refinePose(r.pose, Xs, uv, n, cam, { iters: 8, huber: thresh * 0.6, inlierThresh: thresh });
  if (r2.nInliers >= r.nInliers) r = r2;
  return { pose: r.pose, inliers: r.inliers, nInliers: r.nInliers, rmse: r.rmse };
}
