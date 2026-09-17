import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as L from '../js/slam/linalg.js';

function approx(a, b, eps = 1e-9) { assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`); }
function matApprox(A, B, eps = 1e-9) { for (let i = 0; i < A.length; i++) approx(A[i], B[i], eps); }

test('so3 exp/log roundtrip', () => {
  const w = new Float64Array([0.3, -0.2, 0.9]);
  const R = L.so3Exp(w);
  approx(L.mat3Det(R), 1, 1e-12);
  matApprox(L.so3Log(R), w, 1e-10);
  // near pi
  const w2 = L.scale3(L.normalize3(new Float64Array([1, 2, 3])), Math.PI - 1e-6);
  matApprox(L.so3Log(L.so3Exp(w2)), w2, 1e-5);
  const w3 = new Float64Array([1e-10, 0, 0]);
  matApprox(L.so3Log(L.so3Exp(w3)), w3, 1e-12);
});

test('jacobi eigen decomposition', () => {
  const n = 5;
  const M = new Float64Array(n * n);
  const rng = L.makeRng(7);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const v = rng() * 2 - 1; M[i * n + j] = v; M[j * n + i] = v; }
  const { values, vectors } = L.jacobiEigenSym(M, n);
  for (let j = 0; j < n; j++) {
    const v = L.eigenColumn(vectors, n, j);
    for (let i = 0; i < n; i++) {
      let s = 0; for (let k = 0; k < n; k++) s += M[i * n + k] * v[k];
      approx(s, values[j] * v[i], 1e-9);
    }
    if (j > 0) assert.ok(values[j] >= values[j - 1]);
  }
});

test('svd reconstructs matrix and handles rank deficiency', () => {
  const rng = L.makeRng(3);
  const m = 7, n = 4;
  const A = new Float64Array(m * n);
  for (let i = 0; i < A.length; i++) A[i] = rng() * 2 - 1;
  const { U, S, V } = L.svd(A, m, n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    let s = 0; for (let k = 0; k < n; k++) s += U[i * n + k] * S[k] * V[j * n + k];
    approx(s, A[i * n + j], 1e-9);
  }
  for (let k = 1; k < n; k++) assert.ok(S[k] <= S[k - 1]);
  // rank-2 3x3 (an essential-like matrix)
  const E = new Float64Array([0, -1, 0, 1, 0, 0, 0, 0, 0]);
  const r = L.svd3(E);
  approx(r.S[2], 0, 1e-12);
  // U must be orthonormal with a completed third column
  const Ut = L.mat3Transpose(r.U);
  matApprox(L.mat3Mul(Ut, r.U), L.mat3Identity(), 1e-9);
});

test('solveLinear', () => {
  const A = new Float64Array([4, 1, 2, 1, 3, 0, 2, 0, 5]);
  const b = new Float64Array([1, 2, 3]);
  const x = L.solveLinear(A, b, 3);
  matApprox(L.mat3MulVec(A, x), b, 1e-10);
  assert.equal(L.solveLinear(new Float64Array([1, 2, 2, 4]), new Float64Array([1, 2]), 2), null);
});

test('pose compose/inverse', () => {
  const A = L.poseCreate(L.so3Exp([0.1, 0.2, 0.3]), [1, 2, 3]);
  const B = L.poseCreate(L.so3Exp([-0.4, 0.1, 0.2]), [0.5, -1, 2]);
  const X = new Float64Array([0.3, -0.7, 2.5]);
  const AB = L.poseCompose(A, B);
  matApprox(L.poseApply(AB, X), L.poseApply(A, L.poseApply(B, X)), 1e-12);
  const Ainv = L.poseInverse(A);
  matApprox(L.poseApply(Ainv, L.poseApply(A, X)), X, 1e-12);
  const C = L.poseCenter(A);
  matApprox(L.poseApply(A, C), [0, 0, 0], 1e-12);
});

test('orthonormalizeRotation', () => {
  const R = L.so3Exp([0.5, -0.3, 0.2]);
  const noisy = Float64Array.from(R, (v) => v + 0.01);
  const R2 = L.orthonormalizeRotation(noisy);
  approx(L.mat3Det(R2), 1, 1e-10);
  matApprox(L.mat3Mul(L.mat3Transpose(R2), R2), L.mat3Identity(), 1e-10);
});
