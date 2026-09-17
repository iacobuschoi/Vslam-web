// Monocular visual SLAM pipeline:
//   FAST corners -> pyramidal KLT tracking -> essential-matrix initialization ->
//   PnP pose tracking -> keyframe triangulation -> BRIEF-based relocalization.

import { Pyramid, rgbaToGray, boxBlur } from './image.js';
import { detectFast, selectGridCorners } from './fast.js';
import { KLTTracker } from './klt.js';
import * as G from './geometry.js';
import * as L from './linalg.js';
import * as B from './brief.js';

export const State = Object.freeze({ INIT: 'INIT', TRACKING: 'TRACKING', LOST: 'LOST' });

const DEG = Math.PI / 180;

export class Slam {
  constructor(width, height, opts = {}) {
    this.w = width; this.h = height;
    this.opts = Object.assign({
      fovDeg: 75,             // diagonal field of view
      maxFeatures: 350,
      fastThreshold: 20,
      pyramidLevels: Math.max(width, height) >= 400 ? 4 : 3,
      kltWinRadius: 7,
      fbCheck: true,
      targetDepth: 2.0,       // scene scale after initialization (median depth in world units)
      minInitTracks: 60,
      initParallaxFrac: 0.05, // median pixel parallax (fraction of the longer side) required before trying to initialize
      minKFInterval: 3,
      kfParallaxFrac: 0.06,
      kfBaselineFrac: 0.06,
      kfRotationDeg: 8,
      relocalize: true,
      maxRelocTrain: 3000,
      kltPredict: false,      // seed KLT with motion-model predictions (can lock onto repetitive texture)
    }, opts);
    this.scale = Math.max(width, height) / 480;
    this.setFov(this.opts.fovDeg);
    this.cellSize = Math.max(8, Math.sqrt((width * height) / this.opts.maxFeatures));
    this.gray = new Float32Array(width * height);
    this.blur = null;
    this.scoreMap = new Float32Array(width * height);
    this.pyrCur = new Pyramid(width, height, this.opts.pyramidLevels);
    this.pyrPrev = new Pyramid(width, height, this.opts.pyramidLevels);
    this.klt = new KLTTracker({ winRadius: this.opts.kltWinRadius, fbCheck: this.opts.fbCheck });
    this.rng = L.makeRng(1234);
    this.reset();
  }

  setFov(deg) {
    this.opts.fovDeg = deg;
    const diag = Math.hypot(this.w, this.h);
    const f = (diag / 2) / Math.tan((deg * DEG) / 2);
    this.cam = { f, cx: this.w / 2, cy: this.h / 2 };
  }

  reset() {
    this.state = State.INIT;
    this.tracks = [];
    this.keyframes = [];
    this.mapPoints = [];
    this.nextTrackId = 0;
    this.nextMpId = 0;
    this.nextKfId = 0;
    this.frameIndex = -1;
    this.hasPrev = false;
    this.pose = L.poseCreate();
    this.velocity = L.poseCreate();
    this.lastPoseOk = false;
    this.initKF = null;
    this.refKF = null;
    this.framesSinceKF = 0;
    this.lastInitAttempt = -10;
    this.lostFrames = 0;
    this.medianDepth = this.opts.targetDepth;
    this.mapVersion = 0;
    this.relocTrain = null;
    this.relocTrainVersion = -1;
    this.hint = '';
    this.lastInliers = 0;
    this.lastParallax = 0;
    this.trajectory = [];
  }

  // ---------- Public entry point ----------

  /**
   * Process one RGBA frame. Returns a result object (see buildResult).
   */
  processFrame(rgba) {
    const t0 = now();
    rgbaToGray(rgba, this.w, this.h, this.gray);
    const tmp = this.pyrPrev; this.pyrPrev = this.pyrCur; this.pyrCur = tmp;
    this.pyrCur.build(this.gray);
    this.curRgba = rgba;
    this.blur = null;
    this.frameIndex++;
    let mapChanged = false;

    if (!this.hasPrev) {
      this.hasPrev = true;
      this.startInit();
    } else {
      this.trackFeatures();
      switch (this.state) {
        case State.INIT: mapChanged = this.doInit(); break;
        case State.TRACKING: mapChanged = this.doTracking(); break;
        case State.LOST: mapChanged = this.doRelocalize(); break;
      }
    }
    if (this.lastPoseOk) {
      const C = L.poseCenter(this.pose);
      this.trajectory.push(C[0], C[1], C[2]);
      if (this.trajectory.length > 3 * 20000) this.trajectory.splice(0, 3 * 5000);
    }
    return this.buildResult(mapChanged, now() - t0);
  }

  // ---------- Initialization ----------

  startInit() {
    this.tracks = [];
    this.mapPoints = [];
    this.keyframes = [];
    this.nextMpId = 0;
    this.nextKfId = 0;
    this.trajectory = [];
    this.lastPoseOk = false;
    this.initKF = { id: this.nextKfId++, index: this.frameIndex, pose: L.poseCreate(), nTracked: 0 };
    this.detectNewFeatures(this.initKF);
    this.state = State.INIT;
    this.hint = 'move';
    this.mapVersion++;
  }

  doInit() {
    const n = this.tracks.length;
    if (n < this.opts.minInitTracks) { this.startInit(); return true; }
    const disp = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const tr = this.tracks[i], o = tr.obs[0];
      disp[i] = Math.hypot(tr.x - o.x, tr.y - o.y);
    }
    const med = L.median(disp);
    this.lastParallax = med;
    const maxDim = Math.max(this.w, this.h);
    if (med < this.opts.initParallaxFrac * maxDim) { this.hint = 'move'; return false; }
    if (this.frameIndex - this.lastInitAttempt < 2) return false;
    this.lastInitAttempt = this.frameIndex;

    const { f, cx, cy } = this.cam;
    const x1 = new Float64Array(2 * n), x2 = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const tr = this.tracks[i], o = tr.obs[0];
      x1[2 * i] = (o.x - cx) / f; x1[2 * i + 1] = (o.y - cy) / f;
      x2[2 * i] = (tr.x - cx) / f; x2[2 * i + 1] = (tr.y - cy) / f;
    }
    // Estimate both an essential matrix and a homography; keep whichever reconstructs more points.
    const reprojThresh = Math.pow((4 * this.scale) / f, 2);
    const minPar = 0.5 * DEG;
    const threshE = Math.pow((1.5 * this.scale) / f, 2);
    const resE = G.findEssentialRansac(x1, x2, n, threshE, this.rng, 300);
    const threshH = Math.pow((2.0 * this.scale) / f, 2);
    const resH = G.findHomographyRansac(x1, x2, n, threshH, this.rng, 300);
    let best = null;
    if (resE && resE.nInliers >= 30) {
      const b = G.selectPoseFromEssential(resE.E, x1, x2, n, resE.inliers, reprojThresh, minPar);
      if (b) { b.model = 'E'; b.nInliers = resE.nInliers; best = b; }
    }
    if (resH && resH.nInliers >= 30) {
      const b = G.selectPoseFromHomography(resH.H, x1, x2, n, resH.inliers, reprojThresh, minPar);
      if (b && (!best || b.nGood > best.nGood)) { b.model = 'H'; b.nInliers = resH.nInliers; best = b; }
    }
    if (!best) { this.hint = 'move'; return false; }
    this.lastInitInfo = { model: best.model, nGood: best.nGood, ratio: best.ratio, parallaxDeg: best.medianParallax / DEG, nInliers: best.nInliers };
    if (best.nGood < 40 || best.nGood < 0.6 * best.nInliers || best.ratio > 0.75) { this.hint = 'move'; return false; }
    if (best.medianParallax < 1.5 * DEG) { this.hint = 'move'; return false; }

    // Fix the scale so that the median depth equals targetDepth.
    const depths = best.points.map((p) => p.X[2]);
    const medDepth = L.median(depths);
    const s = this.opts.targetDepth / Math.max(medDepth, 1e-6);
    const pose2 = L.poseCreate(best.pose.R, L.scale3(best.pose.t, s));

    const KF0 = this.initKF;
    const KF1 = { id: this.nextKfId++, index: this.frameIndex, pose: L.poseClone(pose2), nTracked: 0 };
    this.keyframes = [KF0, KF1];
    const good = new Map();
    for (const p of best.points) good.set(p.i, p.X);
    const keep = [];
    for (let i = 0; i < n; i++) {
      const tr = this.tracks[i];
      const X = good.get(i);
      if (!X) continue;
      const Xs = new Float64Array([X[0] * s, X[1] * s, X[2] * s]);
      const mp = this.createMapPoint(Xs, tr.x, tr.y);
      mp.obs.push({ kf: KF0, xn: x1[2 * i], yn: x1[2 * i + 1] });
      mp.obs.push({ kf: KF1, xn: x2[2 * i], yn: x2[2 * i + 1] });
      tr.mp = mp;
      tr.obs.push({ kf: KF1, x: tr.x, y: tr.y });
      keep.push(tr);
    }
    this.tracks = keep;
    this.pose = pose2;
    this.velocity = L.poseCreate();
    this.lastPoseOk = true;
    this.medianDepth = this.opts.targetDepth;
    this.computeDescriptorsForTracks();
    this.detectNewFeatures(KF1);
    KF1.nTracked = keep.length;
    this.refKF = KF1;
    this.framesSinceKF = 0;
    this.state = State.TRACKING;
    this.hint = '';
    this.lastInliers = keep.length;
    this.mapVersion++;
    return true;
  }

  // ---------- Feature tracking ----------

  trackFeatures() {
    const n = this.tracks.length;
    if (n === 0) return;
    const pts = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) { pts[2 * i] = this.tracks[i].x; pts[2 * i + 1] = this.tracks[i].y; }
    let guess = null;
    if (this.opts.kltPredict && this.state === State.TRACKING && this.lastPoseOk) {
      // Predict feature positions with the constant-velocity motion model.
      const pred = L.poseCompose(this.velocity, this.pose);
      const { f, cx, cy } = this.cam;
      guess = new Float32Array(pts);
      const dxs = [], dys = [];
      const p = new Float64Array(3);
      for (let i = 0; i < n; i++) {
        const mp = this.tracks[i].mp;
        if (!mp || mp.bad) continue;
        L.poseApply(pred, mp.X, p);
        if (p[2] <= 1e-6) continue;
        const u = f * p[0] / p[2] + cx, v = f * p[1] / p[2] + cy;
        if (u < 0 || v < 0 || u >= this.w || v >= this.h) continue;
        const dx = u - pts[2 * i], dy = v - pts[2 * i + 1];
        if (Math.hypot(dx, dy) > 0.25 * Math.max(this.w, this.h)) continue;
        guess[2 * i] = u; guess[2 * i + 1] = v;
        dxs.push(dx); dys.push(dy);
      }
      if (dxs.length >= 5) {
        const mdx = L.median(dxs), mdy = L.median(dys);
        for (let i = 0; i < n; i++) {
          if (this.tracks[i].mp && !this.tracks[i].mp.bad) continue;
          guess[2 * i] = pts[2 * i] + mdx; guess[2 * i + 1] = pts[2 * i + 1] + mdy;
        }
      } else guess = null;
    }
    const r = this.klt.track(this.pyrPrev, this.pyrCur, pts, n, guess);
    const keep = [];
    for (let i = 0; i < n; i++) {
      const tr = this.tracks[i];
      if (!r.status[i]) continue;
      tr.x = r.next[2 * i]; tr.y = r.next[2 * i + 1];
      tr.age++;
      keep.push(tr);
    }
    this.tracks = keep;
  }

  // ---------- Tracking ----------

  doTracking() {
    let mapChanged = false;
    const { f, cx, cy } = this.cam;
    const mpTracks = [];
    for (const tr of this.tracks) if (tr.mp && !tr.mp.bad) mpTracks.push(tr);
    const n = mpTracks.length;
    if (n < 10) return this.setLost();
    const Xs = new Float64Array(3 * n), uv = new Float64Array(2 * n);
    for (let i = 0; i < n; i++) {
      const tr = mpTracks[i];
      Xs[3 * i] = tr.mp.X[0]; Xs[3 * i + 1] = tr.mp.X[1]; Xs[3 * i + 2] = tr.mp.X[2];
      uv[2 * i] = tr.x; uv[2 * i + 1] = tr.y;
    }
    const pred = this.lastPoseOk ? L.poseCompose(this.velocity, this.pose) : this.pose;
    const inlierThresh = 3 * this.scale;
    const refineOpts = { iters: 10, huber: 2.5 * this.scale, inlierThresh };
    let r = G.refinePose(pred, Xs, uv, n, this.cam, refineOpts);
    if (r.nInliers < 0.8 * n && this.lastPoseOk) {
      // The motion model may be off; also try from the last pose.
      const rb = G.refinePose(this.pose, Xs, uv, n, this.cam, refineOpts);
      if (rb.nInliers > r.nInliers) r = rb;
    }
    const firstInliers = r.nInliers;
    if (r.nInliers < Math.max(12, 0.5 * n)) {
      const r2 = G.pnpRansac(Xs, uv, n, this.cam, this.rng, { thresh: 4 * this.scale, maxIter: 150 });
      if (r2 && r2.nInliers > r.nInliers) {
        const r3 = G.refinePose(r2.pose, Xs, uv, n, this.cam, { iters: 6, huber: 2.5 * this.scale, inlierThresh });
        r = r3.nInliers >= r2.nInliers ? r3 : r2;
      }
    }
    this.lastDebug = { n, firstInliers, inliers: r.nInliers, rmse: r.rmse };
    if (r.nInliers < 12) return this.setLost();

    // Accept the pose.
    const newPose = r.pose;
    if (this.lastPoseOk) {
      this.velocity = L.poseCompose(newPose, L.poseInverse(this.pose));
      L.orthonormalizeRotation(this.velocity.R, this.velocity.R);
    } else this.velocity = L.poseCreate();
    this.pose = newPose;
    this.lastPoseOk = true;
    this.lastInliers = r.nInliers;

    // Reject outlier tracks; drop weakly supported map points.
    const depths = [];
    const p = new Float64Array(3);
    for (let i = 0; i < n; i++) {
      const tr = mpTracks[i];
      if (!r.inliers[i]) {
        tr.dead = true;
        tr.mp.outliers = (tr.mp.outliers || 0) + 1;
        if (tr.mp.obs.length <= 2) { tr.mp.bad = true; mapChanged = true; }
        continue;
      }
      L.poseApply(this.pose, tr.mp.X, p);
      depths.push(p[2]);
      tr.mp.lastSeen = this.frameIndex;
    }
    if (depths.length >= 10) this.medianDepth = L.median(depths);
    this.tracks = this.tracks.filter((t) => !t.dead);

    // Keyframe decision.
    this.framesSinceKF++;
    const ref = this.refKF;
    let needKF = false;
    if (this.framesSinceKF >= this.opts.minKFInterval) {
      const maxDim = Math.max(this.w, this.h);
      const nInl = r.nInliers;
      if (nInl < 0.7 * ref.nTracked || nInl < 40) needKF = true;
      if (!needKF) {
        const disp = [];
        for (const tr of this.tracks) { const o = tr.obs[tr.obs.length - 1]; disp.push(Math.hypot(tr.x - o.x, tr.y - o.y)); }
        const medDisp = L.median(disp);
        if (medDisp > this.opts.kfParallaxFrac * maxDim) needKF = true;
      }
      if (!needKF) {
        const C = L.poseCenter(this.pose), Cr = L.poseCenter(ref.pose);
        const baseline = L.norm3(L.sub3(C, Cr));
        if (baseline / Math.max(this.medianDepth, 1e-6) > this.opts.kfBaselineFrac) needKF = true;
        const dR = L.mat3Mul(this.pose.R, L.mat3Transpose(ref.pose.R));
        if (L.rotationAngle(dR) > this.opts.kfRotationDeg * DEG) needKF = true;
      }
    }
    if (needKF) { this.createKeyframe(); mapChanged = true; }
    this.hint = '';
    return mapChanged;
  }

  setLost() {
    this.state = State.LOST;
    this.lastPoseOk = false;
    this.lostFrames = 0;
    this.tracks = [];
    this.hint = 'lost';
    return false;
  }

  // ---------- Mapping ----------

  createMapPoint(X, px, py) {
    const rgba = this.curRgba;
    let r = 200, g = 200, b = 200;
    if (rgba) {
      const xi = Math.min(Math.max(Math.round(px), 0), this.w - 1), yi = Math.min(Math.max(Math.round(py), 0), this.h - 1);
      const k = 4 * (yi * this.w + xi);
      r = rgba[k]; g = rgba[k + 1]; b = rgba[k + 2];
    }
    const mp = { id: this.nextMpId++, X: Float64Array.from(X), r, g, b, obs: [], desc: null, bad: false, outliers: 0, lastSeen: this.frameIndex };
    this.mapPoints.push(mp);
    return mp;
  }

  createKeyframe() {
    const { f, cx, cy } = this.cam;
    const KF = { id: this.nextKfId++, index: this.frameIndex, pose: L.poseClone(this.pose), nTracked: 0 };
    this.keyframes.push(KF);
    const reprojThresh = 2.5 * this.scale;
    const minParallax = 1.0 * DEG;
    for (const tr of this.tracks) {
      tr.obs.push({ kf: KF, x: tr.x, y: tr.y });
      const xn = (tr.x - cx) / f, yn = (tr.y - cy) / f;
      if (tr.mp) {
        if (tr.mp.bad) { tr.mp = null; tr.obs = [{ kf: KF, x: tr.x, y: tr.y }]; continue; }
        const mp = tr.mp;
        mp.obs.push({ kf: KF, xn, yn });
        const m = mp.obs.length;
        if (m === 3 || m === 4 || m === 6 || m === 8 || m === 12) this.retriangulate(mp, reprojThresh);
        continue;
      }
      const first = tr.obs[0];
      if (first.kf === KF || tr.obs.length < 2) continue;
      const xn1 = (first.x - cx) / f, yn1 = (first.y - cy) / f;
      // Parallax between the viewing rays.
      const r1 = L.mat3TMulVec(first.kf.pose.R, [xn1, yn1, 1]);
      const r2 = L.mat3TMulVec(KF.pose.R, [xn, yn, 1]);
      const cosang = L.dot3(r1, r2) / (L.norm3(r1) * L.norm3(r2));
      const ang = Math.acos(Math.max(-1, Math.min(1, cosang)));
      if (ang < minParallax) continue; // keep as a candidate
      let X = G.triangulate2(first.kf.pose, KF.pose, xn1, yn1, xn, yn);
      if (!X || !this.checkPoint(X, tr.obs, reprojThresh)) { tr.dead = true; continue; }
      if (tr.obs.length > 2) {
        const poses = tr.obs.map((o) => o.kf.pose);
        const xs = tr.obs.map((o) => [(o.x - cx) / f, (o.y - cy) / f]);
        const X2 = G.triangulateN(poses, xs);
        if (X2 && this.checkPoint(X2, tr.obs, reprojThresh)) X = X2;
      }
      const mp = this.createMapPoint(X, tr.x, tr.y);
      for (const o of tr.obs) mp.obs.push({ kf: o.kf, xn: (o.x - cx) / f, yn: (o.y - cy) / f });
      tr.mp = mp;
    }
    this.tracks = this.tracks.filter((t) => !t.dead);
    this.computeDescriptorsForTracks();
    this.detectNewFeatures(KF);
    let nTracked = 0;
    for (const tr of this.tracks) if (tr.mp) nTracked++;
    KF.nTracked = nTracked;
    this.refKF = KF;
    this.framesSinceKF = 0;
    this.mapVersion++;
  }

  // Check a world point against pixel observations (positive depth and reprojection error).
  checkPoint(X, obs, thresh) {
    const { f, cx, cy } = this.cam;
    const p = new Float64Array(3);
    for (const o of obs) {
      L.poseApply(o.kf.pose, X, p);
      if (p[2] <= 1e-6) return false;
      const du = f * p[0] / p[2] + cx - o.x, dv = f * p[1] / p[2] + cy - o.y;
      if (du * du + dv * dv > thresh * thresh) return false;
    }
    // Reject absurdly far points (relative to the scene scale).
    const C = L.poseCenter(obs[obs.length - 1].kf.pose);
    if (L.norm3(L.sub3(X, C)) > 40 * this.medianDepth) return false;
    return true;
  }

  retriangulate(mp, thresh) {
    const obs = mp.obs.slice(-8);
    const poses = obs.map((o) => o.kf.pose);
    const xs = obs.map((o) => [o.xn, o.yn]);
    const X = G.triangulateN(poses, xs);
    if (!X) return;
    const { f, cx, cy } = this.cam;
    const pxObs = obs.map((o) => ({ kf: o.kf, x: o.xn * f + cx, y: o.yn * f + cy }));
    if (this.checkPoint(X, pxObs, thresh)) mp.X.set(X);
  }

  detectNewFeatures(KF) {
    const w = this.w, h = this.h, cell = this.cellSize;
    const cellsX = Math.ceil(w / cell), cellsY = Math.ceil(h / cell);
    const occupied = new Uint8Array(cellsX * cellsY);
    let occ = 0;
    for (const tr of this.tracks) {
      const c = ((tr.y / cell) | 0) * cellsX + ((tr.x / cell) | 0);
      if (!occupied[c]) { occupied[c] = 1; occ++; }
    }
    const empty = occupied.length - occ;
    if (empty <= 0) return;
    const border = 10;
    let corners = detectFast(this.gray, w, h, this.opts.fastThreshold, border, this.scoreMap);
    let sel = selectGridCorners(corners, w, h, cell, occupied);
    if (sel.length < 0.5 * empty && this.opts.fastThreshold > 8) {
      for (const p of sel) occupied[((p[1] / cell) | 0) * cellsX + ((p[0] / cell) | 0)] = 1;
      corners = detectFast(this.gray, w, h, Math.max(6, this.opts.fastThreshold * 0.5), border, this.scoreMap);
      sel = sel.concat(selectGridCorners(corners, w, h, cell, occupied));
    }
    const room = Math.max(0, Math.round(this.opts.maxFeatures * 1.4) - this.tracks.length);
    if (sel.length > room) { sel.sort((a, b) => b[2] - a[2]); sel.length = room; }
    for (const p of sel) {
      this.tracks.push({ id: this.nextTrackId++, x: p[0], y: p[1], mp: null, obs: [{ kf: KF, x: p[0], y: p[1] }], age: 0, dead: false });
    }
  }

  ensureBlur() {
    if (!this.blur) this.blur = boxBlur(this.gray, this.w, this.h, 2, this._blurBuf, this._blurTmp);
    return this.blur;
  }

  computeDescriptorsForTracks() {
    if (!this.opts.relocalize) return;
    if (!this._blurBuf) { this._blurBuf = new Float32Array(this.w * this.h); this._blurTmp = new Float32Array(this.w * this.h); }
    const blur = this.ensureBlur();
    for (const tr of this.tracks) {
      if (!tr.mp || tr.mp.bad) continue;
      const a = B.keypointAngle(this.gray, this.w, this.h, tr.x, tr.y);
      const d = B.computeDescriptor(blur, this.w, this.h, tr.x, tr.y, a);
      if (d) tr.mp.desc = Uint32Array.from(d);
    }
  }

  // ---------- Relocalization ----------

  getRelocTrain() {
    if (this.relocTrain && this.relocTrainVersion === this.mapVersion) return this.relocTrain;
    const mps = [];
    for (let i = this.mapPoints.length - 1; i >= 0 && mps.length < this.opts.maxRelocTrain; i--) {
      const mp = this.mapPoints[i];
      if (!mp.bad && mp.desc) mps.push(mp);
    }
    const desc = new Uint32Array(mps.length * B.DESC_WORDS);
    for (let i = 0; i < mps.length; i++) desc.set(mps[i].desc, i * B.DESC_WORDS);
    this.relocTrain = { mps, desc };
    this.relocTrainVersion = this.mapVersion;
    return this.relocTrain;
  }

  doRelocalize() {
    this.lostFrames++;
    this.hint = 'lost';
    if (!this.opts.relocalize) return false;
    if (this.lostFrames % 2 !== 0) return false;
    const train = this.getRelocTrain();
    if (train.mps.length < 20) return false;
    const w = this.w, h = this.h;
    if (!this._blurBuf) { this._blurBuf = new Float32Array(w * h); this._blurTmp = new Float32Array(w * h); }
    const blur = this.ensureBlur();
    const corners = detectFast(this.gray, w, h, this.opts.fastThreshold, B.MIN_BORDER, this.scoreMap);
    const sel = selectGridCorners(corners, w, h, Math.max(6, this.cellSize * 0.6), null, 600);
    const query = new Uint32Array(sel.length * B.DESC_WORDS);
    const kps = [];
    for (const p of sel) {
      const a = B.keypointAngle(this.gray, w, h, p[0], p[1]);
      const d = B.computeDescriptor(blur, w, h, p[0], p[1], a);
      if (!d) continue;
      query.set(d, kps.length * B.DESC_WORDS);
      kps.push(p);
    }
    if (kps.length < 12) return false;
    const matches = B.matchDescriptors(query, kps.length, train.desc, train.mps.length, 64, 0.8);
    if (matches.length < 12) return false;
    const m = matches.length;
    const Xs = new Float64Array(3 * m), uv = new Float64Array(2 * m);
    for (let i = 0; i < m; i++) {
      const mp = train.mps[matches[i][1]], kp = kps[matches[i][0]];
      Xs[3 * i] = mp.X[0]; Xs[3 * i + 1] = mp.X[1]; Xs[3 * i + 2] = mp.X[2];
      uv[2 * i] = kp[0]; uv[2 * i + 1] = kp[1];
    }
    const r = G.pnpRansac(Xs, uv, m, this.cam, this.rng, { thresh: 5 * this.scale, maxIter: 200 });
    if (!r || r.nInliers < 15) return false;
    // Success: rebuild tracks from the inlier matches and start a new keyframe.
    this.pose = r.pose;
    this.velocity = L.poseCreate();
    this.lastPoseOk = true;
    this.tracks = [];
    const used = new Set();
    for (let i = 0; i < m; i++) {
      if (!r.inliers[i]) continue;
      const mp = train.mps[matches[i][1]];
      if (used.has(mp)) continue;
      used.add(mp);
      const kp = kps[matches[i][0]];
      this.tracks.push({ id: this.nextTrackId++, x: kp[0], y: kp[1], mp, obs: [], age: 0, dead: false });
    }
    this.lastInliers = r.nInliers;
    this.createKeyframe();
    this.state = State.TRACKING;
    this.hint = '';
    return true;
  }

  // ---------- Output ----------

  buildResult(mapChanged, procMs) {
    const n = this.tracks.length;
    const features = new Float32Array(3 * n);
    let nMp = 0;
    for (let i = 0; i < n; i++) {
      const tr = this.tracks[i];
      features[3 * i] = tr.x; features[3 * i + 1] = tr.y;
      features[3 * i + 2] = tr.mp ? 1 : 0;
      if (tr.mp) nMp++;
    }
    let pose = null;
    if (this.lastPoseOk) {
      pose = new Float64Array(12);
      pose.set(this.pose.R, 0); pose.set(this.pose.t, 9);
    }
    let nMap = 0;
    for (const mp of this.mapPoints) if (!mp.bad) nMap++;
    const result = {
      state: this.state,
      hint: this.hint,
      pose,
      features,
      stats: {
        frame: this.frameIndex,
        tracks: n,
        trackedPoints: nMp,
        inliers: this.lastInliers,
        mapPoints: nMap,
        keyframes: this.keyframes.length,
        procMs,
        parallax: this.lastParallax,
        medianDepth: this.medianDepth,
      },
      mapChanged,
    };
    if (mapChanged) result.map = this.buildMapSnapshot();
    return result;
  }

  buildMapSnapshot() {
    let count = 0;
    for (const mp of this.mapPoints) if (!mp.bad) count++;
    const positions = new Float32Array(3 * count), colors = new Uint8Array(3 * count);
    let k = 0;
    for (const mp of this.mapPoints) {
      if (mp.bad) continue;
      positions[3 * k] = mp.X[0]; positions[3 * k + 1] = mp.X[1]; positions[3 * k + 2] = mp.X[2];
      colors[3 * k] = mp.r; colors[3 * k + 1] = mp.g; colors[3 * k + 2] = mp.b;
      k++;
    }
    const kfs = new Float32Array(12 * this.keyframes.length);
    this.keyframes.forEach((kf, i) => { kfs.set(kf.pose.R, 12 * i); kfs.set(kf.pose.t, 12 * i + 9); });
    return { positions, colors, count, keyframes: kfs, keyframeCount: this.keyframes.length, version: this.mapVersion };
  }

  // PLY export of the current map (ASCII).
  exportPly() {
    const lines = [];
    let count = 0;
    for (const mp of this.mapPoints) if (!mp.bad) count++;
    lines.push('ply', 'format ascii 1.0', `element vertex ${count}`,
      'property float x', 'property float y', 'property float z',
      'property uchar red', 'property uchar green', 'property uchar blue', 'end_header');
    for (const mp of this.mapPoints) {
      if (mp.bad) continue;
      lines.push(`${mp.X[0].toFixed(4)} ${mp.X[1].toFixed(4)} ${mp.X[2].toFixed(4)} ${mp.r} ${mp.g} ${mp.b}`);
    }
    return lines.join('\n') + '\n';
  }
}

function now() {
  return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
}
