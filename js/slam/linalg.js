// Small dense linear algebra helpers for the SLAM pipeline.
// Matrices are row-major Float64Arrays. 3x3 matrices have 9 entries.

export function vec3(x = 0, y = 0, z = 0) { return new Float64Array([x, y, z]); }
export function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function norm3(a) { return Math.sqrt(dot3(a, a)); }
export function sub3(a, b, out = new Float64Array(3)) { out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2]; return out; }
export function add3(a, b, out = new Float64Array(3)) { out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2]; return out; }
export function scale3(a, s, out = new Float64Array(3)) { out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s; return out; }
export function cross3(a, b, out = new Float64Array(3)) {
  const x = a[1] * b[2] - a[2] * b[1];
  const y = a[2] * b[0] - a[0] * b[2];
  const z = a[0] * b[1] - a[1] * b[0];
  out[0] = x; out[1] = y; out[2] = z; return out;
}
export function normalize3(a, out = new Float64Array(3)) {
  const n = norm3(a) || 1; out[0] = a[0] / n; out[1] = a[1] / n; out[2] = a[2] / n; return out;
}

export function mat3Identity(out = new Float64Array(9)) { out.fill(0); out[0] = out[4] = out[8] = 1; return out; }
export function mat3Copy(A, out = new Float64Array(9)) { out.set(A); return out; }
export function mat3Mul(A, B, out = new Float64Array(9)) {
  const a0 = A[0], a1 = A[1], a2 = A[2], a3 = A[3], a4 = A[4], a5 = A[5], a6 = A[6], a7 = A[7], a8 = A[8];
  const b0 = B[0], b1 = B[1], b2 = B[2], b3 = B[3], b4 = B[4], b5 = B[5], b6 = B[6], b7 = B[7], b8 = B[8];
  out[0] = a0 * b0 + a1 * b3 + a2 * b6; out[1] = a0 * b1 + a1 * b4 + a2 * b7; out[2] = a0 * b2 + a1 * b5 + a2 * b8;
  out[3] = a3 * b0 + a4 * b3 + a5 * b6; out[4] = a3 * b1 + a4 * b4 + a5 * b7; out[5] = a3 * b2 + a4 * b5 + a5 * b8;
  out[6] = a6 * b0 + a7 * b3 + a8 * b6; out[7] = a6 * b1 + a7 * b4 + a8 * b7; out[8] = a6 * b2 + a7 * b5 + a8 * b8;
  return out;
}
export function mat3Transpose(A, out = new Float64Array(9)) {
  const a1 = A[1], a2 = A[2], a5 = A[5];
  out[0] = A[0]; out[4] = A[4]; out[8] = A[8];
  out[1] = A[3]; out[3] = a1; out[2] = A[6]; out[6] = a2; out[5] = A[7]; out[7] = a5;
  return out;
}
export function mat3MulVec(A, v, out = new Float64Array(3)) {
  const x = v[0], y = v[1], z = v[2];
  out[0] = A[0] * x + A[1] * y + A[2] * z;
  out[1] = A[3] * x + A[4] * y + A[5] * z;
  out[2] = A[6] * x + A[7] * y + A[8] * z;
  return out;
}
export function mat3TMulVec(A, v, out = new Float64Array(3)) { // A^T v
  const x = v[0], y = v[1], z = v[2];
  out[0] = A[0] * x + A[3] * y + A[6] * z;
  out[1] = A[1] * x + A[4] * y + A[7] * z;
  out[2] = A[2] * x + A[5] * y + A[8] * z;
  return out;
}
export function mat3Det(A) {
  return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) + A[2] * (A[3] * A[7] - A[4] * A[6]);
}
export function mat3Scale(A, s, out = new Float64Array(9)) { for (let i = 0; i < 9; i++) out[i] = A[i] * s; return out; }
export function skew3(v, out = new Float64Array(9)) {
  out[0] = 0; out[1] = -v[2]; out[2] = v[1];
  out[3] = v[2]; out[4] = 0; out[5] = -v[0];
  out[6] = -v[1]; out[7] = v[0]; out[8] = 0;
  return out;
}

// SO(3) exponential map (Rodrigues). w: rotation vector.
export function so3Exp(w, out = new Float64Array(9)) {
  const theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
  const theta = Math.sqrt(theta2);
  let a, b;
  if (theta < 1e-8) { a = 1 - theta2 / 6; b = 0.5 - theta2 / 24; }
  else { a = Math.sin(theta) / theta; b = (1 - Math.cos(theta)) / theta2; }
  const wx = w[0], wy = w[1], wz = w[2];
  // R = I + a*[w]x + b*[w]x^2
  out[0] = 1 + b * (-wy * wy - wz * wz); out[1] = -a * wz + b * wx * wy; out[2] = a * wy + b * wx * wz;
  out[3] = a * wz + b * wx * wy; out[4] = 1 + b * (-wx * wx - wz * wz); out[5] = -a * wx + b * wy * wz;
  out[6] = -a * wy + b * wx * wz; out[7] = a * wx + b * wy * wz; out[8] = 1 + b * (-wx * wx - wy * wy);
  return out;
}

// SO(3) logarithm map: rotation matrix -> rotation vector.
export function so3Log(R, out = new Float64Array(3)) {
  const tr = R[0] + R[4] + R[8];
  let c = (tr - 1) * 0.5;
  if (c > 1) c = 1; else if (c < -1) c = -1;
  const theta = Math.acos(c);
  if (theta < 1e-7) {
    out[0] = 0.5 * (R[7] - R[5]); out[1] = 0.5 * (R[2] - R[6]); out[2] = 0.5 * (R[3] - R[1]);
    return out;
  }
  if (Math.PI - theta < 1e-4) {
    // Near pi: axis from the symmetric part (R + I) / 2 = a a^T for unit axis a.
    const S = [(R[0] + 1) * 0.5, (R[4] + 1) * 0.5, (R[8] + 1) * 0.5];
    let k = 0; if (S[1] > S[k]) k = 1; if (S[2] > S[k]) k = 2;
    const ak = Math.sqrt(Math.max(S[k], 0));
    const a = new Float64Array(3);
    a[k] = ak;
    if (k === 0) { a[1] = (R[1] + R[3]) * 0.25 / ak; a[2] = (R[2] + R[6]) * 0.25 / ak; }
    else if (k === 1) { a[0] = (R[1] + R[3]) * 0.25 / ak; a[2] = (R[5] + R[7]) * 0.25 / ak; }
    else { a[0] = (R[2] + R[6]) * 0.25 / ak; a[1] = (R[5] + R[7]) * 0.25 / ak; }
    normalize3(a, a);
    out[0] = a[0] * theta; out[1] = a[1] * theta; out[2] = a[2] * theta;
    return out;
  }
  const s = theta / (2 * Math.sin(theta));
  out[0] = s * (R[7] - R[5]); out[1] = s * (R[2] - R[6]); out[2] = s * (R[3] - R[1]);
  return out;
}

export function rotationAngle(R) {
  let c = (R[0] + R[4] + R[8] - 1) * 0.5;
  if (c > 1) c = 1; else if (c < -1) c = -1;
  return Math.acos(c);
}

// Symmetric eigen decomposition with cyclic Jacobi rotations.
// A: n x n symmetric (row-major). Returns { values (ascending), vectors (row-major, column j = eigenvector j) }.
export function jacobiEigenSym(Ain, n) {
  const A = Float64Array.from(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, diag = 0;
    for (let p = 0; p < n; p++) {
      diag += A[p * n + p] * A[p * n + p];
      for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    }
    if (off <= 1e-30 * (diag + 1e-300) || off < 1e-300) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) { // columns
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) { // rows
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => A[a * n + a] - A[b * n + b]);
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    const src = order[j];
    values[j] = A[src * n + src];
    for (let i = 0; i < n; i++) vectors[i * n + j] = V[i * n + src];
  }
  return { values, vectors };
}

export function eigenColumn(vectors, n, j, out = new Float64Array(n)) {
  for (let i = 0; i < n; i++) out[i] = vectors[i * n + j];
  return out;
}

// Smallest eigenvector of a symmetric matrix (used for null-space solutions).
export function smallestEigenvector(A, n, out = new Float64Array(n)) {
  const { vectors } = jacobiEigenSym(A, n);
  return eigenColumn(vectors, n, 0, out);
}

// One-sided Jacobi SVD (Hestenes) for an m x n matrix with m >= n.
// Returns { U (m x n), S (n, descending), V (n x n) } with A = U diag(S) V^T.
export function svd(Ain, m, n) {
  const U = Float64Array.from(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 80; sweep++) {
    let rotated = false;
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        let alpha = 0, beta = 0, gamma = 0;
        for (let k = 0; k < m; k++) {
          const ui = U[k * n + i], uj = U[k * n + j];
          alpha += ui * ui; beta += uj * uj; gamma += ui * uj;
        }
        if (Math.abs(gamma) <= 1e-15 * Math.sqrt(alpha * beta) || Math.abs(gamma) < 1e-300) continue;
        rotated = true;
        const zeta = (beta - alpha) / (2 * gamma);
        const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = c * t;
        for (let k = 0; k < m; k++) {
          const ui = U[k * n + i], uj = U[k * n + j];
          U[k * n + i] = c * ui - s * uj;
          U[k * n + j] = s * ui + c * uj;
        }
        for (let k = 0; k < n; k++) {
          const vi = V[k * n + i], vj = V[k * n + j];
          V[k * n + i] = c * vi - s * vj;
          V[k * n + j] = s * vi + c * vj;
        }
      }
    }
    if (!rotated) break;
  }
  const S = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let k = 0; k < m; k++) s += U[k * n + j] * U[k * n + j];
    s = Math.sqrt(s);
    S[j] = s;
    if (s > 1e-300) for (let k = 0; k < m; k++) U[k * n + j] /= s;
  }
  // Sort descending.
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => S[b] - S[a]);
  const U2 = new Float64Array(m * n), V2 = new Float64Array(n * n), S2 = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const src = order[j];
    S2[j] = S[src];
    for (let k = 0; k < m; k++) U2[k * n + j] = U[k * n + src];
    for (let k = 0; k < n; k++) V2[k * n + j] = V[k * n + src];
  }
  return { U: U2, S: S2, V: V2 };
}

// SVD of a 3x3 matrix, completing U to a proper orthonormal basis when rank-deficient.
export function svd3(A) {
  const r = svd(A, 3, 3);
  const U = r.U, S = r.S;
  const tol = Math.max(S[0] * 1e-9, 1e-300);
  const col = (M, j) => new Float64Array([M[j], M[3 + j], M[6 + j]]);
  const setCol = (M, j, v) => { M[j] = v[0]; M[3 + j] = v[1]; M[6 + j] = v[2]; };
  if (S[2] <= tol) {
    if (S[1] <= tol) {
      // rank <= 1: build any orthonormal completion.
      const u0 = S[0] > tol ? col(U, 0) : new Float64Array([1, 0, 0]);
      let tmp = Math.abs(u0[0]) < 0.9 ? new Float64Array([1, 0, 0]) : new Float64Array([0, 1, 0]);
      const u1 = normalize3(cross3(u0, tmp));
      setCol(U, 0, u0); setCol(U, 1, u1);
    }
    const u2 = normalize3(cross3(col(U, 0), col(U, 1)));
    setCol(U, 2, u2);
  }
  // Same for V (columns of V are orthonormal by construction; nothing to do).
  return r;
}

// Solve A x = b (n x n) with Gaussian elimination and partial pivoting. Returns null if singular.
export function solveLinear(Ain, bin, n) {
  const A = Float64Array.from(Ain);
  const b = Float64Array.from(bin);
  for (let c = 0; c < n; c++) {
    let piv = c, best = Math.abs(A[c * n + c]);
    for (let r = c + 1; r < n; r++) { const v = Math.abs(A[r * n + c]); if (v > best) { best = v; piv = r; } }
    if (best < 1e-14) return null;
    if (piv !== c) {
      for (let k = 0; k < n; k++) { const t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; }
      const t = b[c]; b[c] = b[piv]; b[piv] = t;
    }
    const inv = 1 / A[c * n + c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r * n + c] * inv;
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * x[k];
    x[r] = s / A[r * n + r];
  }
  return x;
}

// Project a near-rotation matrix onto SO(3).
export function orthonormalizeRotation(M, out = new Float64Array(9)) {
  const { U, V } = svd3(M);
  const Vt = mat3Transpose(V);
  const R = mat3Mul(U, Vt);
  if (mat3Det(R) < 0) {
    // Flip the sign of the last column of U.
    U[2] = -U[2]; U[5] = -U[5]; U[8] = -U[8];
    mat3Mul(U, Vt, R);
  }
  out.set(R);
  return out;
}

// Rigid pose T_cw: p_c = R p_w + t.
export function poseCreate(R = mat3Identity(), t = new Float64Array(3)) { return { R: Float64Array.from(R), t: Float64Array.from(t) }; }
export function poseClone(P) { return { R: Float64Array.from(P.R), t: Float64Array.from(P.t) }; }
export function poseInverse(P, out = poseCreate()) {
  mat3Transpose(P.R, out.R);
  const t = mat3MulVec(out.R, P.t);
  out.t[0] = -t[0]; out.t[1] = -t[1]; out.t[2] = -t[2];
  return out;
}
// out = A ∘ B  (apply B first, then A)
export function poseCompose(A, B, out = poseCreate()) {
  const R = mat3Mul(A.R, B.R);
  const t = mat3MulVec(A.R, B.t);
  out.R.set(R);
  out.t[0] = t[0] + A.t[0]; out.t[1] = t[1] + A.t[1]; out.t[2] = t[2] + A.t[2];
  return out;
}
export function poseApply(P, X, out = new Float64Array(3)) {
  const x = X[0], y = X[1], z = X[2];
  out[0] = P.R[0] * x + P.R[1] * y + P.R[2] * z + P.t[0];
  out[1] = P.R[3] * x + P.R[4] * y + P.R[5] * z + P.t[1];
  out[2] = P.R[6] * x + P.R[7] * y + P.R[8] * z + P.t[2];
  return out;
}
export function poseCenter(P, out = new Float64Array(3)) { // camera center in world coordinates: -R^T t
  mat3TMulVec(P.R, P.t, out);
  out[0] = -out[0]; out[1] = -out[1]; out[2] = -out[2];
  return out;
}

// Deterministic PRNG (mulberry32) for RANSAC sampling.
export function makeRng(seed = 12345) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function median(values) {
  if (values.length === 0) return 0;
  const arr = Array.from(values).sort((a, b) => a - b);
  const mid = arr.length >> 1;
  return arr.length % 2 ? arr[mid] : 0.5 * (arr[mid - 1] + arr[mid]);
}
