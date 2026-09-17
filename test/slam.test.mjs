import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Slam, State } from '../js/slam/slam.js';
import * as L from '../js/slam/linalg.js';
import { makeRoom, renderRoom, cameraPose, alignSimilarity } from './synth-room.mjs';

test('SLAM initializes and tracks through a synthetic room', () => {
  const w = 480, h = 360;
  const slam = new Slam(w, h, { fovDeg: 75, fbCheck: true });
  const cam = slam.cam;
  const planes = makeRoom();
  const frames = 160;
  const est = [], gt = [], stamps = [];
  let initFrame = -1, lostCount = 0, procTotal = 0;
  const kfCounts = [];
  for (let k = 0; k < frames; k++) {
    const P = cameraPose(k);
    const rgba = renderRoom(planes, P, cam, w, h);
    const res = slam.processFrame(rgba);
    procTotal += res.stats.procMs;
    if (res.state === State.TRACKING && initFrame < 0) initFrame = k;
    if (res.state === State.LOST) lostCount++;
    if (res.pose) {
      const pose = L.poseCreate(res.pose.subarray(0, 9), res.pose.subarray(9, 12));
      const C = L.poseCenter(pose), Cg = L.poseCenter(P);
      est.push(...C); gt.push(...Cg); stamps.push(k);
    }
    kfCounts.push(res.stats.keyframes);
  }
  const final = slam.buildResult(false, 0);
  console.log(`init at frame ${initFrame}, lost frames ${lostCount}, map points ${final.stats.mapPoints}, keyframes ${final.stats.keyframes}, avg proc ${(procTotal / frames).toFixed(1)} ms`);
  assert.ok(initFrame >= 0 && initFrame < 60, `initialized late: ${initFrame}`);
  assert.equal(final.state, State.TRACKING);
  assert.ok(final.stats.mapPoints > 400, `map points ${final.stats.mapPoints}`);
  const n = est.length / 3;
  const a = alignSimilarity(Float64Array.from(est), Float64Array.from(gt), n);
  // Path extent for a relative error measure.
  let ext = 0;
  for (let i = 0; i < n; i++) ext = Math.max(ext, Math.hypot(gt[3 * i] - gt[0], gt[3 * i + 1] - gt[1], gt[3 * i + 2] - gt[2]));
  console.log(`trajectory rmse after similarity alignment: ${a.rmse.toFixed(4)} (path extent ${ext.toFixed(3)}), scale ${a.s.toFixed(3)}`);
  assert.ok(a.rmse < 0.06 * ext, `trajectory error too large: ${a.rmse} vs extent ${ext}`);
  // Map points should lie near the true room surfaces (check distance to nearest plane after alignment).
  const snap = slam.buildMapSnapshot();
  let onSurface = 0;
  for (let i = 0; i < snap.count; i++) {
    const X = L.add3(L.scale3(L.mat3MulVec(a.R, snap.positions.subarray(3 * i, 3 * i + 3)), a.s), a.t);
    let dmin = Infinity;
    for (const p of planes) dmin = Math.min(dmin, Math.abs(p.n[0] * X[0] + p.n[1] * X[1] + p.n[2] * X[2] - p.d));
    if (dmin < 0.15) onSurface++;
  }
  console.log(`map points within 15cm of a wall: ${onSurface}/${snap.count}`);
  assert.ok(onSurface / snap.count > 0.8, `too many off-surface points: ${onSurface}/${snap.count}`);
});
