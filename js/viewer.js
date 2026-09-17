// Three.js viewer for the point-cloud map, camera trajectory and keyframe frustums.
import * as THREE from 'three';
import { OrbitControls } from '../lib/OrbitControls.js';

function makeDiscTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.7, 'rgba(255,255,255,1)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  return tex;
}

export class MapViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0f14);
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 500);
    this.camera.position.set(0.8, 1.2, 1.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0, -2);
    this.controls.minDistance = 0.05;
    this.controls.maxDistance = 200;
    this.userInteracted = false;
    this.controls.addEventListener('start', () => { this.userInteracted = true; });

    // CV convention (x right, y down, z forward) -> Three (y up, z backward)
    this.root = new THREE.Group();
    this.root.rotation.x = Math.PI;
    this.scene.add(this.root);

    this.follow = false;
    this.pointSize = 0.03;

    // Point cloud
    this.capacity = 0;
    this.pointCount = 0;
    this.pointsGeom = new THREE.BufferGeometry();
    this.pointsMat = new THREE.PointsMaterial({ size: this.pointSize, vertexColors: true, sizeAttenuation: true, map: makeDiscTexture(), alphaTest: 0.5, transparent: false });
    this.points = new THREE.Points(this.pointsGeom, this.pointsMat);
    this.points.frustumCulled = false;
    this.root.add(this.points);
    this._ensureCapacity(20000);

    // Trajectory
    this.trajCapacity = 30000;
    this.trajCount = 0;
    this.trajPositions = new Float32Array(3 * this.trajCapacity);
    this.trajGeom = new THREE.BufferGeometry();
    this.trajGeom.setAttribute('position', new THREE.BufferAttribute(this.trajPositions, 3).setUsage(THREE.DynamicDrawUsage));
    this.trajGeom.setDrawRange(0, 0);
    this.traj = new THREE.Line(this.trajGeom, new THREE.LineBasicMaterial({ color: 0x4fd1c5 }));
    this.traj.frustumCulled = false;
    this.root.add(this.traj);

    // Keyframe frustums
    this.kfGeom = new THREE.BufferGeometry();
    this.kfLines = new THREE.LineSegments(this.kfGeom, new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.6 }));
    this.kfLines.frustumCulled = false;
    this.root.add(this.kfLines);

    // Current camera frustum
    this.camFrustum = new THREE.LineSegments(this._frustumGeometry(0.22), new THREE.LineBasicMaterial({ color: 0xfacc15 }));
    this.camFrustum.matrixAutoUpdate = false;
    this.camFrustum.visible = false;
    this.root.add(this.camFrustum);

    // Origin axes (small) and a subtle grid for orientation
    const axes = new THREE.AxesHelper(0.25);
    this.root.add(axes);
    this.grid = new THREE.GridHelper(20, 40, 0x1f2937, 0x1a2230);
    this.grid.position.y = -0.01;
    this.grid.visible = false;
    this.scene.add(this.grid);

    this.aspect = 4 / 3;
    this.hfov = 60 * Math.PI / 180;
    this._tmpM = new THREE.Matrix4();
    this._tmpV = new THREE.Vector3();
    this.lastCamPos = null;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  _frustumGeometry(depth) {
    const w = depth * Math.tan(this.hfov ? this.hfov / 2 : 0.5), h = w / (this.aspect || 4 / 3);
    const c = [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]];
    const verts = [];
    for (let i = 0; i < 4; i++) { verts.push(0, 0, 0, ...c[i]); verts.push(...c[i], ...c[(i + 1) % 4]); }
    // "up" indicator triangle (in CV coordinates y is down, so draw it above the top edge: -y)
    verts.push(-w * 0.5, -h, depth, 0, -h * 1.6, depth, 0, -h * 1.6, depth, w * 0.5, -h, depth);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    return g;
  }

  setCameraIntrinsics(width, height, fovDegDiagonal) {
    this.aspect = width / height;
    const diag = Math.hypot(width, height);
    const f = (diag / 2) / Math.tan((fovDegDiagonal * Math.PI / 180) / 2);
    this.hfov = 2 * Math.atan((width / 2) / f);
    this.camFrustum.geometry.dispose();
    this.camFrustum.geometry = this._frustumGeometry(0.22);
  }

  _ensureCapacity(n) {
    if (n <= this.capacity) return;
    let cap = Math.max(20000, this.capacity);
    while (cap < n) cap *= 2;
    const pos = new Float32Array(3 * cap), col = new Uint8Array(3 * cap);
    if (this.positions) { pos.set(this.positions.subarray(0, 3 * this.pointCount)); col.set(this.colors.subarray(0, 3 * this.pointCount)); }
    this.positions = pos; this.colors = col; this.capacity = cap;
    this.pointsGeom.setAttribute('position', new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.pointsGeom.setAttribute('color', new THREE.BufferAttribute(col, 3, true).setUsage(THREE.DynamicDrawUsage));
    this.pointsGeom.setDrawRange(0, this.pointCount);
  }

  setMap(positions, colors, count) {
    this._ensureCapacity(count);
    this.positions.set(positions.subarray(0, 3 * count));
    this.colors.set(colors.subarray(0, 3 * count));
    this.pointCount = count;
    const pa = this.pointsGeom.getAttribute('position'), ca = this.pointsGeom.getAttribute('color');
    pa.needsUpdate = true; ca.needsUpdate = true;
    this.pointsGeom.setDrawRange(0, count);
    this.pointsGeom.computeBoundingSphere();
  }

  setKeyframes(kfs, n) {
    const depth = 0.1;
    const w = depth * Math.tan(this.hfov / 2), h = w / this.aspect;
    const verts = new Float32Array(n * 16 * 3);
    const c = [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]];
    let k = 0;
    const R = new Float64Array(9), t = new Float64Array(3), C = new Float64Array(3);
    const toWorld = (p) => [ // world = R^T (p - t)
      R[0] * (p[0] - t[0]) + R[3] * (p[1] - t[1]) + R[6] * (p[2] - t[2]),
      R[1] * (p[0] - t[0]) + R[4] * (p[1] - t[1]) + R[7] * (p[2] - t[2]),
      R[2] * (p[0] - t[0]) + R[5] * (p[1] - t[1]) + R[8] * (p[2] - t[2]),
    ];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < 9; j++) R[j] = kfs[12 * i + j];
      for (let j = 0; j < 3; j++) t[j] = kfs[12 * i + 9 + j];
      const apex = toWorld([0, 0, 0]);
      const cw = c.map(toWorld);
      for (let e = 0; e < 4; e++) {
        verts.set(apex, k); k += 3; verts.set(cw[e], k); k += 3;
        verts.set(cw[e], k); k += 3; verts.set(cw[(e + 1) % 4], k); k += 3;
      }
    }
    this.kfGeom.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    this.kfGeom.setDrawRange(0, n * 16);
  }

  // pose: Float64Array(12) [R(9) row-major, t(3)] camera-from-world, or null.
  setCameraPose(pose) {
    if (!pose) { this.camFrustum.visible = false; return; }
    const R = pose, t = pose.subarray(9, 12);
    // camera-to-world: [R^T | -R^T t]
    const cx = -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]);
    const cy = -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]);
    const cz = -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]);
    this._tmpM.set(
      R[0], R[3], R[6], cx,
      R[1], R[4], R[7], cy,
      R[2], R[5], R[8], cz,
      0, 0, 0, 1);
    this.camFrustum.matrix.copy(this._tmpM);
    this.camFrustum.visible = true;
    // Trajectory
    if (this.trajCount < this.trajCapacity) {
      const last = this.trajCount ? this.trajPositions.subarray(3 * (this.trajCount - 1), 3 * this.trajCount) : null;
      if (!last || Math.hypot(last[0] - cx, last[1] - cy, last[2] - cz) > 1e-4) {
        this.trajPositions[3 * this.trajCount] = cx; this.trajPositions[3 * this.trajCount + 1] = cy; this.trajPositions[3 * this.trajCount + 2] = cz;
        this.trajCount++;
        this.trajGeom.getAttribute('position').needsUpdate = true;
        this.trajGeom.setDrawRange(0, this.trajCount);
      }
    }
    if (this.follow) {
      const p = this._tmpV.set(cx, cy, cz);
      this.root.localToWorld(p);
      if (this.lastCamPos) {
        const d = p.clone().sub(this.lastCamPos);
        this.controls.target.add(d);
        this.camera.position.add(d);
      } else {
        this.controls.target.copy(p);
      }
      this.lastCamPos = p.clone();
    }
  }

  setFollow(on) {
    this.follow = on;
    this.lastCamPos = null;
  }

  setPointSize(s) {
    this.pointSize = s;
    this.pointsMat.size = s;
  }

  clear() {
    this.userInteracted = false;
    this.pointCount = 0;
    this.pointsGeom.setDrawRange(0, 0);
    this.trajCount = 0;
    this.trajGeom.setDrawRange(0, 0);
    this.kfGeom.setDrawRange(0, 0);
    this.camFrustum.visible = false;
    this.lastCamPos = null;
  }

  // Fit the view to the current point cloud (or the origin if empty).
  fitView() {
    if (this.pointCount < 10) {
      this.controls.target.set(0, 0, -2);
      this.camera.position.set(0.8, 1.2, 1.6);
      return;
    }
    // Robust bounds: use percentiles to ignore outliers.
    const n = this.pointCount;
    const xs = new Float32Array(n), ys = new Float32Array(n), zs = new Float32Array(n);
    for (let i = 0; i < n; i++) { xs[i] = this.positions[3 * i]; ys[i] = this.positions[3 * i + 1]; zs[i] = this.positions[3 * i + 2]; }
    xs.sort(); ys.sort(); zs.sort();
    const lo = Math.floor(n * 0.05), hi = Math.floor(n * 0.95);
    const center = new THREE.Vector3((xs[lo] + xs[hi]) / 2, (ys[lo] + ys[hi]) / 2, (zs[lo] + zs[hi]) / 2);
    const size = Math.max(xs[hi] - xs[lo], ys[hi] - ys[lo], zs[hi] - zs[lo], 0.5);
    this.root.localToWorld(center);
    const dist = (size / 2) / Math.tan((this.camera.fov * Math.PI / 180) / 2) * 1.3;
    this.controls.target.copy(center);
    const dir = new THREE.Vector3(0.6, 0.7, 0.9).normalize();
    this.camera.position.copy(center).add(dir.multiplyScalar(dist));
    this.lastCamPos = null;
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth, h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
