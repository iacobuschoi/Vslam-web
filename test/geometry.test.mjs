import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as L from '../js/slam/linalg.js';
import * as G from '../js/slam/geometry.js';

const cam = { f: 400, cx: 240, cy: 180 };

function makeScene(rng, nPts = 200) {
  const Xs = new Float64Array(3 * nPts);
  for (let i = 0; i < nPts; i++) {
    Xs[3 * i] = (rng() - 0.5) * 6;
    Xs[3 * i + 1] = (rng() - 0.5) * 4;
    Xs[3 * i + 2] = 3 + rng() * 5;
  }
  return Xs;
}

function project(P, Xs, n, noise = 0, rng = null) {
  const xn = new Float64Array(2 * n), uv = new Float64Array(2 * n);
  for (let i = 0; i < n; i++) {
    const p = L.poseApply(P, Xs.subarray(3 * i, 3 * i + 3));
    let u = cam.f * p[0] / p[2] + cam.cx, v = cam.f * p[1] / p[2] + cam.cy;
    if (noise) { u += (rng() - 0.5) * 2 * noise; v += (rng() - 0.5) * 2 * noise; }
    uv[2 * i] = u; uv[2 * i + 1] = v;
    xn[2 * i] = (u - cam.cx) / cam.f; xn[2 * i + 1] = (v - cam.cy) / cam.f;
  }
  return { xn, uv };
}

test('essential matrix RANSAC recovers relative pose with outliers', () => {
  const rng = L.makeRng(42);
  const n = 200;
  const Xs = makeScene(rng, n);
  const P1 = L.poseCreate();
  const Rtrue = L.so3Exp([0.02, -0.15, 0.01]);
  const ttrue = new Float64Array([0.6, 0.05, 0.1]);
  const P2 = L.poseCreate(Rtrue, ttrue);
  const v1 = project(P1, Xs, n, 0.3, rng), v2 = project(P2, Xs, n, 0.3, rng);
  // Add outliers.
  for (let i = 0; i < 40; i++) { const k = Math.floor(rng() * n); v2.xn[2 * k] = (rng() - 0.5); v2.xn[2 * k + 1] = (rng() - 0.5); }
  const thresh = Math.pow(1.5 / cam.f, 2);
  const res = G.findEssentialRansac(v1.xn, v2.xn, n, thresh, L.makeRng(1));
  assert.ok(res, 'no result');
  assert.ok(res.nInliers > 130, `inliers ${res.nInliers}`);
  const best = G.selectPoseFromEssential(res.E, v1.xn, v2.xn, n, res.inliers, Math.pow(4 / cam.f, 2));
  assert.ok(best.nGood > 120, `good ${best.nGood}`);
  assert.ok(best.ratio < 0.5, `ratio ${best.ratio}`);
  // Compare rotation and translation direction.
  const dR = L.mat3Mul(best.pose.R, L.mat3Transpose(Rtrue));
  assert.ok(L.rotationAngle(dR) < 0.01, `rotation error ${L.rotationAngle(dR)}`);
  const tn = L.normalize3(ttrue);
  const cosang = L.dot3(best.pose.t, tn);
  assert.ok(cosang > 0.995, `translation dir cos ${cosang}`);
  assert.ok(best.medianParallax > 0.03, `parallax ${best.medianParallax}`);
});

test('triangulation is accurate', () => {
  const rng = L.makeRng(5);
  const n = 50;
  const Xs = makeScene(rng, n);
  const P1 = L.poseCreate(L.so3Exp([0.1, 0.0, 0.0]), [0.2, 0, 0]);
  const P2 = L.poseCreate(L.so3Exp([0.0, -0.2, 0.05]), [-0.8, 0.1, 0.3]);
  const P3 = L.poseCreate(L.so3Exp([0.05, 0.2, 0.0]), [0.7, -0.2, 0.1]);
  const a = project(P1, Xs, n), b = project(P2, Xs, n), c = project(P3, Xs, n);
  for (let i = 0; i < n; i++) {
    const X2 = G.triangulate2(P1, P2, a.xn[2 * i], a.xn[2 * i + 1], b.xn[2 * i], b.xn[2 * i + 1]);
    const X3 = G.triangulateN([P1, P2, P3], [[a.xn[2 * i], a.xn[2 * i + 1]], [b.xn[2 * i], b.xn[2 * i + 1]], [c.xn[2 * i], c.xn[2 * i + 1]]]);
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(X2[k] - Xs[3 * i + k]) < 1e-6);
      assert.ok(Math.abs(X3[k] - Xs[3 * i + k]) < 1e-6);
    }
  }
});

test('PnP DLT + RANSAC + refinement recover pose', () => {
  const rng = L.makeRng(11);
  const n = 120;
  const Xs = makeScene(rng, n);
  const Rtrue = L.so3Exp([0.2, -0.3, 0.1]);
  const ttrue = new Float64Array([0.4, -0.2, 0.5]);
  const P = L.poseCreate(Rtrue, ttrue);
  const v = project(P, Xs, n, 0.5, rng);
  // DLT on all clean points
  const idx = new Int32Array(n); for (let i = 0; i < n; i++) idx[i] = i;
  const Pd = G.pnpDLT(Xs, v.xn, idx);
  assert.ok(Pd);
  assert.ok(L.rotationAngle(L.mat3Mul(Pd.R, L.mat3Transpose(Rtrue))) < 0.02);
  assert.ok(L.norm3(L.sub3(Pd.t, ttrue)) < 0.1);
  // outliers
  for (let i = 0; i < 30; i++) { const k = Math.floor(rng() * n); v.uv[2 * k] = rng() * 480; v.uv[2 * k + 1] = rng() * 360; }
  const r = G.pnpRansac(Xs, v.uv, n, cam, L.makeRng(3), { thresh: 4 });
  assert.ok(r, 'ransac failed');
  assert.ok(r.nInliers > 80, `inliers ${r.nInliers}`);
  assert.ok(L.rotationAngle(L.mat3Mul(r.pose.R, L.mat3Transpose(Rtrue))) < 0.005, 'rot');
  assert.ok(L.norm3(L.sub3(r.pose.t, ttrue)) < 0.02, `t err ${L.norm3(L.sub3(r.pose.t, ttrue))}`);
});

test('refinePose converges from a perturbed initial pose', () => {
  const rng = L.makeRng(21);
  const n = 100;
  const Xs = makeScene(rng, n);
  const Rtrue = L.so3Exp([-0.1, 0.25, 0.05]);
  const ttrue = new Float64Array([0.1, 0.3, -0.2]);
  const P = L.poseCreate(Rtrue, ttrue);
  const v = project(P, Xs, n, 0.3, rng);
  const init = L.poseCreate(L.mat3Mul(L.so3Exp([0.03, -0.04, 0.02]), Rtrue), [ttrue[0] + 0.1, ttrue[1] - 0.05, ttrue[2] + 0.08]);
  const r = G.refinePose(init, Xs, v.uv, n, cam, { iters: 15 });
  assert.ok(r.nInliers > 90, `inliers ${r.nInliers}`);
  assert.ok(L.rotationAngle(L.mat3Mul(r.pose.R, L.mat3Transpose(Rtrue))) < 0.003, 'rot');
  assert.ok(L.norm3(L.sub3(r.pose.t, ttrue)) < 0.02, 't');
  assert.ok(r.rmse < 0.6, `rmse ${r.rmse}`);
});

test('homography RANSAC + decomposition recover planar motion', () => {
  const rng = L.makeRng(77);
  const n = 150;
  // Points on the plane z = 3 + 0.2x - 0.1y  (normal n, distance d)
  const Xs = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {
    const x = (rng() - 0.5) * 5, y = (rng() - 0.5) * 3.5;
    Xs[3 * i] = x; Xs[3 * i + 1] = y; Xs[3 * i + 2] = 3 + 0.2 * x - 0.1 * y;
  }
  const P1 = L.poseCreate();
  const Rtrue = L.so3Exp([0.03, 0.12, -0.02]);
  const ttrue = new Float64Array([0.5, 0.1, 0.15]);
  const P2 = L.poseCreate(Rtrue, ttrue);
  const v1 = project(P1, Xs, n, 0.3, rng), v2 = project(P2, Xs, n, 0.3, rng);
  for (let i = 0; i < 20; i++) { const k = Math.floor(rng() * n); v2.xn[2 * k] = rng() - 0.5; v2.xn[2 * k + 1] = rng() - 0.5; }
  const thresh = Math.pow(2 / cam.f, 2);
  const res = G.findHomographyRansac(v1.xn, v2.xn, n, thresh, L.makeRng(2));
  assert.ok(res, 'no homography');
  assert.ok(res.nInliers > 110, `inliers ${res.nInliers}`);
  const best = G.selectPoseFromHomography(res.H, v1.xn, v2.xn, n, res.inliers, Math.pow(4 / cam.f, 2), 0.5 * Math.PI / 180);
  assert.ok(best, 'no pose');
  assert.ok(best.nGood > 100, `good ${best.nGood}`);
  assert.ok(best.ratio < 0.75, `ratio ${best.ratio}`);
  const dR = L.mat3Mul(best.pose.R, L.mat3Transpose(Rtrue));
  assert.ok(L.rotationAngle(dR) < 0.02, `rotation error ${L.rotationAngle(dR)}`);
  const cosang = L.dot3(best.pose.t, L.normalize3(ttrue));
  assert.ok(cosang > 0.99, `translation dir cos ${cosang}`);
});
