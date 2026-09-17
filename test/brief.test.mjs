import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeDescriptor, keypointAngle, matchDescriptors, hamming, DESC_WORDS } from '../js/slam/brief.js';
import { boxBlur, bilinear } from '../js/slam/image.js';
import { detectFast, selectGridCorners } from '../js/slam/fast.js';
import { makeRng } from '../js/slam/linalg.js';

function makeTexture(w, h, seed = 1) {
  const rng = makeRng(seed);
  const img = new Float32Array(w * h);
  const rects = [];
  for (let i = 0; i < 80; i++) rects.push([rng() * w, rng() * h, 8 + rng() * 40, 8 + rng() * 40, rng() * 255]);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let v = 128 + 50 * Math.sin(x * 0.08) * Math.cos(y * 0.05);
    for (const r of rects) if (x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]) v = r[4];
    img[y * w + x] = v;
  }
  return img;
}

function rotateImage(src, w, h, angle, cx, cy) {
  const out = new Float32Array(w * h);
  const c = Math.cos(angle), s = Math.sin(angle);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = x - cx, dy = y - cy;
    const sx = c * dx + s * dy + cx, sy = -s * dx + c * dy + cy;
    out[y * w + x] = (sx >= 0 && sy >= 0 && sx < w - 1 && sy < h - 1) ? bilinear(src, w, h, sx, sy) : 0;
  }
  return out;
}

test('descriptors match across a rotated image', () => {
  const w = 240, h = 180;
  const img = makeTexture(w, h, 3);
  const angle = 0.5;
  const img2 = rotateImage(img, w, h, angle, w / 2, h / 2);
  const b1 = boxBlur(img, w, h, 2), b2 = boxBlur(img2, w, h, 2);
  const c1 = selectGridCorners(detectFast(img, w, h, 20, 20), w, h, 12, null);
  const c2 = selectGridCorners(detectFast(img2, w, h, 20, 20), w, h, 12, null);
  const d1 = new Uint32Array(c1.length * DESC_WORDS), d2 = new Uint32Array(c2.length * DESC_WORDS);
  const k1 = [], k2 = [];
  for (const p of c1) { const a = keypointAngle(img, w, h, p[0], p[1]); const d = computeDescriptor(b1, w, h, p[0], p[1], a); if (d) { d1.set(d, k1.length * DESC_WORDS); k1.push(p); } }
  for (const p of c2) { const a = keypointAngle(img2, w, h, p[0], p[1]); const d = computeDescriptor(b2, w, h, p[0], p[1], a); if (d) { d2.set(d, k2.length * DESC_WORDS); k2.push(p); } }
  const matches = matchDescriptors(d2, k2.length, d1, k1.length, 70, 0.85);
  // Verify geometric consistency: map k1 through the rotation and compare.
  const c = Math.cos(angle), s = Math.sin(angle);
  let correct = 0;
  for (const [q, t] of matches) {
    const p = k1[t];
    const dx = p[0] - w / 2, dy = p[1] - h / 2;
    const rx = c * dx - s * dy + w / 2, ry = s * dx + c * dy + h / 2;
    if (Math.hypot(rx - k2[q][0], ry - k2[q][1]) < 3) correct++;
  }
  assert.ok(matches.length >= 15, `matches ${matches.length}`);
  assert.ok(correct / matches.length > 0.6, `precision ${correct}/${matches.length}`);
});

test('hamming distance basics', () => {
  const a = new Uint32Array(DESC_WORDS), b = new Uint32Array(DESC_WORDS);
  assert.equal(hamming(a, 0, b, 0), 0);
  b[0] = 0xFFFFFFFF; b[3] = 0x0F;
  assert.equal(hamming(a, 0, b, 0), 36);
});
