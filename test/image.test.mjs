import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Pyramid, bilinear, boxBlur } from '../js/slam/image.js';
import { detectFast, selectGridCorners } from '../js/slam/fast.js';
import { KLTTracker } from '../js/slam/klt.js';
import { makeRng } from '../js/slam/linalg.js';

// Synthetic textured image: smooth random blobs + sharp rectangles.
function makeTexture(w, h, seed = 1) {
  const rng = makeRng(seed);
  const img = new Float32Array(w * h);
  const rects = [];
  for (let i = 0; i < 60; i++) rects.push([rng() * w, rng() * h, 10 + rng() * 40, 10 + rng() * 40, rng() * 255]);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 128 + 40 * Math.sin(x * 0.05) * Math.cos(y * 0.07);
    for (const r of rects) if (x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]) v = r[4];
    img[y * w + x] = v;
  }
  return img;
}

function shiftImage(src, w, h, dx, dy) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = bilinear(src, w, h, x - dx, y - dy);
  return out;
}

test('FAST finds corners of rectangles', () => {
  const w = 160, h = 120;
  const img = makeTexture(w, h);
  const c = detectFast(img, w, h, 20, 8);
  assert.ok(c.count > 30, `found ${c.count}`);
  const sel = selectGridCorners(c, w, h, 20, null);
  assert.ok(sel.length > 10);
});

test('KLT tracks a subpixel shift', () => {
  const w = 200, h = 150;
  const img = makeTexture(w, h, 5);
  const dx = 3.3, dy = -2.6;
  const img2 = shiftImage(img, w, h, dx, dy);
  const A = new Pyramid(w, h, 3).build(img);
  const B = new Pyramid(w, h, 3).build(img2);
  const c = detectFast(img, w, h, 20, 20);
  const sel = selectGridCorners(c, w, h, 16, null);
  const pts = new Float32Array(sel.length * 2);
  sel.forEach((p, i) => { pts[2 * i] = p[0]; pts[2 * i + 1] = p[1]; });
  const klt = new KLTTracker({ fbCheck: true });
  const r = klt.track(A, B, pts, sel.length);
  let ok = 0, errSum = 0;
  for (let i = 0; i < sel.length; i++) {
    if (!r.status[i]) continue;
    ok++;
    errSum += Math.hypot(r.next[2 * i] - pts[2 * i] - dx, r.next[2 * i + 1] - pts[2 * i + 1] - dy);
  }
  assert.ok(ok > sel.length * 0.7, `tracked ${ok}/${sel.length}`);
  assert.ok(errSum / ok < 0.15, `mean err ${errSum / ok}`);
});

test('KLT tracks a large shift via the pyramid', () => {
  const w = 240, h = 180;
  const img = makeTexture(w, h, 9);
  const dx = 14, dy = 9;
  const img2 = shiftImage(img, w, h, dx, dy);
  const A = new Pyramid(w, h, 4).build(img);
  const B = new Pyramid(w, h, 4).build(img2);
  const c = detectFast(img, w, h, 20, 30);
  const sel = selectGridCorners(c, w, h, 16, null);
  const pts = new Float32Array(sel.length * 2);
  sel.forEach((p, i) => { pts[2 * i] = p[0]; pts[2 * i + 1] = p[1]; });
  const klt = new KLTTracker({ fbCheck: true });
  const r = klt.track(A, B, pts, sel.length);
  let ok = 0, good = 0;
  for (let i = 0; i < sel.length; i++) {
    if (!r.status[i]) continue;
    ok++;
    const e = Math.hypot(r.next[2 * i] - pts[2 * i] - dx, r.next[2 * i + 1] - pts[2 * i + 1] - dy);
    if (e < 0.5) good++;
  }
  assert.ok(good > sel.length * 0.5, `good ${good}/${sel.length} (tracked ${ok})`);
});

test('boxBlur preserves constant image', () => {
  const w = 20, h = 10;
  const img = new Float32Array(w * h).fill(77);
  const b = boxBlur(img, w, h, 2);
  for (const v of b) assert.ok(Math.abs(v - 77) < 1e-4);
});

test('KLT is robust to a global brightness change', () => {
  const w = 200, h = 150;
  const img = makeTexture(w, h, 13);
  const dx = 2.4, dy = 1.7;
  const shifted = shiftImage(img, w, h, dx, dy);
  const img2 = new Float32Array(w * h);
  for (let i = 0; i < img2.length; i++) img2[i] = Math.min(255, shifted[i] * 1.15 + 25); // brighter exposure
  const A = new Pyramid(w, h, 3).build(img);
  const B = new Pyramid(w, h, 3).build(img2);
  const c = detectFast(img, w, h, 20, 20);
  const sel = selectGridCorners(c, w, h, 16, null);
  const pts = new Float32Array(sel.length * 2);
  sel.forEach((p, i) => { pts[2 * i] = p[0]; pts[2 * i + 1] = p[1]; });
  const klt = new KLTTracker({ fbCheck: true });
  const r = klt.track(A, B, pts, sel.length);
  let ok = 0, good = 0;
  for (let i = 0; i < sel.length; i++) {
    if (!r.status[i]) continue;
    ok++;
    if (Math.hypot(r.next[2 * i] - pts[2 * i] - dx, r.next[2 * i + 1] - pts[2 * i + 1] - dy) < 0.3) good++;
  }
  assert.ok(ok > sel.length * 0.6, `tracked ${ok}/${sel.length}`);
  assert.ok(good > ok * 0.8, `accurate ${good}/${ok}`);
});
