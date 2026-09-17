// Synthetic textured room rendered with Three.js, used as a stand-in camera (demo mode / tests).
import * as THREE from 'three';

function makeTexture(seed, w = 512, h = 512, base = '#9aa4b2') {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  let s = seed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  ctx.fillStyle = base;
  ctx.fillRect(0, 0, w, h);
  // Soft gradient
  const g = ctx.createLinearGradient(0, 0, w, h);
  g.addColorStop(0, 'rgba(255,255,255,0.15)'); g.addColorStop(1, 'rgba(0,0,0,0.25)');
  ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  // Random rectangles / posters
  for (let i = 0; i < 45; i++) {
    const rw = 20 + rnd() * 120, rh = 20 + rnd() * 120;
    ctx.fillStyle = `hsl(${Math.floor(rnd() * 360)}, ${40 + rnd() * 50}%, ${30 + rnd() * 50}%)`;
    ctx.fillRect(rnd() * (w - rw), rnd() * (h - rh), rw, rh);
  }
  // Text labels add distinctive corners
  ctx.fillStyle = 'rgba(20,20,20,0.9)';
  ctx.font = 'bold 40px sans-serif';
  for (let i = 0; i < 8; i++) ctx.fillText(String.fromCharCode(65 + Math.floor(rnd() * 26)) + Math.floor(rnd() * 90), rnd() * (w - 80), 40 + rnd() * (h - 40));
  // Fine noise
  const img = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < img.data.length; i += 4) { const nz = (rnd() - 0.5) * 24; img.data[i] += nz; img.data[i + 1] += nz; img.data[i + 2] += nz; }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class DemoSource {
  constructor(width, height, fovDegDiagonal) {
    this.width = width; this.height = height;
    this.canvas = document.createElement('canvas');
    this.canvas.width = width; this.canvas.height = height;
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x202020);
    const diag = Math.hypot(width, height);
    const f = (diag / 2) / Math.tan((fovDegDiagonal * Math.PI / 180) / 2);
    const vfov = 2 * Math.atan((height / 2) / f) * 180 / Math.PI;
    this.camera = new THREE.PerspectiveCamera(vfov, width / height, 0.05, 100);
    this.buildRoom();
    this.target = new THREE.WebGLRenderTarget(width, height, { depthBuffer: true });
    this.pixels = new Uint8Array(width * height * 4);
    this.t0 = null;
    this.speed = 1.0;
    this.frame = 0;
  }

  buildRoom() {
    const W = 6, H = 3.2, D = 7;
    const mats = [];
    for (let i = 0; i < 6; i++) {
      const tex = makeTexture(1000 + i * 77, 512, 512, ['#8fa1b5', '#a7b6a1', '#b8a58d', '#9ba3ad', '#8e9bab', '#b0a08c'][i]);
      // Negative x-repeat un-mirrors textures seen from inside the box (BackSide).
      tex.repeat.set(-(i === 2 || i === 3 ? 2 : 1.5), 1);
      mats.push(new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide }));
    }
    const room = new THREE.Mesh(new THREE.BoxGeometry(W, H, D), mats);
    this.scene.add(room);
    // Furniture-like boxes
    const boxes = [
      [1.2, 0.8, 0.8, 1.6, -H / 2 + 0.4, -2.2],
      [0.8, 1.6, 0.6, -2.4, -H / 2 + 0.8, -1.5],
      [1.8, 0.5, 0.9, -0.6, -H / 2 + 0.25, -2.6],
      [0.6, 1.2, 0.6, 2.4, -H / 2 + 0.6, 0.8],
      [1.0, 0.9, 0.5, -2.3, -H / 2 + 0.45, 2.2],
    ];
    boxes.forEach((b, i) => {
      const tex = makeTexture(5000 + i * 13, 256, 256, '#c9b79c');
      const m = new THREE.Mesh(new THREE.BoxGeometry(b[0], b[1], b[2]), new THREE.MeshBasicMaterial({ map: tex }));
      m.position.set(b[3], b[4], b[5]);
      this.scene.add(m);
    });
  }

  // Camera path (seconds). Sideways translation first so initialization has parallax.
  updateCamera(time) {
    const t = time * this.speed;
    const x = 0.9 * Math.sin(t * 0.55);
    const y = 0.12 * Math.sin(t * 0.8);
    const z = 0.7 * Math.sin(t * 0.23);
    const yaw = 0.55 * Math.sin(t * 0.3) + 0.15 * t * 0.1;
    const pitch = 0.12 * Math.sin(t * 0.47);
    this.camera.position.set(x, y, z);
    this.camera.rotation.set(0, 0, 0);
    this.camera.rotateY(yaw);
    this.camera.rotateX(pitch);
    // Looking towards -z (Three convention); the room is centered at origin.
  }

  // Returns { data: Uint8ClampedArray, width, height } of the current frame (top-down rows).
  grab(nowMs) {
    if (this.t0 === null) this.t0 = nowMs;
    this.updateCamera((nowMs - this.t0) / 1000);
    this.renderer.setRenderTarget(this.target);
    this.renderer.render(this.scene, this.camera);
    this.renderer.readRenderTargetPixels(this.target, 0, 0, this.width, this.height, this.pixels);
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
    // Flip vertically into a fresh buffer.
    const w = this.width, h = this.height;
    const out = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) out.set(this.pixels.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    this.frame++;
    return { data: out, width: w, height: h };
  }

  dispose() {
    this.target.dispose();
    this.renderer.dispose();
  }
}
