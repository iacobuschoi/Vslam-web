// Rotated BRIEF descriptors (256 bits) with intensity-centroid orientation, used for relocalization.

import { makeRng } from './linalg.js';

const PATCH_RADIUS = 15;
const PATTERN_LIMIT = 13; // keep sampling positions within the rotated patch

// Deterministic Gaussian-distributed sampling pattern: 256 pairs of (x1,y1,x2,y2).
function buildPattern() {
  const rng = makeRng(0xB51EF);
  const gauss = () => {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const sigma = PATCH_RADIUS / 2.5;
  const pat = new Float32Array(256 * 4);
  for (let i = 0; i < 256 * 4; i++) {
    let v = gauss() * sigma;
    if (v > PATTERN_LIMIT) v = PATTERN_LIMIT; else if (v < -PATTERN_LIMIT) v = -PATTERN_LIMIT;
    pat[i] = v;
  }
  return pat;
}
const PATTERN = buildPattern();

// Circular mask for orientation (row extents for each dy).
const UMAX = (() => {
  const r = PATCH_RADIUS;
  const u = new Int32Array(r + 1);
  for (let v = 0; v <= r; v++) u[v] = Math.floor(Math.sqrt(r * r - v * v));
  return u;
})();

export const DESC_WORDS = 8;
export const MIN_BORDER = PATCH_RADIUS + 3;

// Intensity centroid orientation (radians) at integer pixel (x, y).
export function keypointAngle(img, w, h, x, y) {
  x = Math.round(x); y = Math.round(y);
  const r = PATCH_RADIUS;
  if (x < r + 1 || y < r + 1 || x >= w - r - 1 || y >= h - r - 1) return 0;
  let m01 = 0, m10 = 0;
  const c = y * w + x;
  for (let u = -r; u <= r; u++) m10 += u * img[c + u];
  for (let v = 1; v <= r; v++) {
    const d = UMAX[v];
    let vsum = 0;
    for (let u = -d; u <= d; u++) {
      const plus = img[c + v * w + u], minus = img[c - v * w + u];
      vsum += plus - minus;
      m10 += u * (plus + minus);
    }
    m01 += v * vsum;
  }
  return Math.atan2(m01, m10);
}

/**
 * Compute a 256-bit descriptor at (x, y) on a (blurred) image with the given orientation.
 * @returns {Uint32Array|null} 8 words, or null if too close to the border
 */
export function computeDescriptor(img, w, h, x, y, angle, out = new Uint32Array(DESC_WORDS)) {
  const xi = Math.round(x), yi = Math.round(y);
  if (xi < MIN_BORDER || yi < MIN_BORDER || xi >= w - MIN_BORDER || yi >= h - MIN_BORDER) return null;
  const c = Math.cos(angle), s = Math.sin(angle);
  const base = yi * w + xi;
  let word = 0, bit = 0;
  out.fill(0);
  for (let i = 0; i < 256; i++) {
    const k = i * 4;
    const ax = PATTERN[k], ay = PATTERN[k + 1], bx = PATTERN[k + 2], by = PATTERN[k + 3];
    const rax = Math.round(c * ax - s * ay), ray = Math.round(s * ax + c * ay);
    const rbx = Math.round(c * bx - s * by), rby = Math.round(s * bx + c * by);
    const va = img[base + ray * w + rax], vb = img[base + rby * w + rbx];
    if (va < vb) word |= (1 << bit);
    bit++;
    if (bit === 32) { out[i >> 5] = word >>> 0; word = 0; bit = 0; }
  }
  return out;
}

function popcount32(v) {
  v = v - ((v >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
}

// Hamming distance between descriptor a (offset ao in words) and b (offset bo).
export function hamming(a, ao, b, bo) {
  let d = 0;
  for (let i = 0; i < DESC_WORDS; i++) d += popcount32((a[ao + i] ^ b[bo + i]) >>> 0);
  return d;
}

/**
 * Match query descriptors against train descriptors (brute force, ratio test).
 * Both are Uint32Array with DESC_WORDS words per descriptor.
 * Returns array of [queryIdx, trainIdx, distance].
 */
export function matchDescriptors(query, nq, train, nt, maxDist = 60, ratio = 0.8) {
  const matches = [];
  const bestForTrain = new Int32Array(nt).fill(-1);
  const bestDistTrain = new Int32Array(nt).fill(999);
  const qBest = new Int32Array(nq), qDist = new Int32Array(nq);
  for (let q = 0; q < nq; q++) {
    let b1 = 999, b2 = 999, bi = -1;
    const qo = q * DESC_WORDS;
    for (let t = 0; t < nt; t++) {
      const d = hamming(query, qo, train, t * DESC_WORDS);
      if (d < b1) { b2 = b1; b1 = d; bi = t; }
      else if (d < b2) b2 = d;
    }
    qBest[q] = bi; qDist[q] = b1;
    if (bi < 0 || b1 > maxDist || b1 > ratio * b2) { qBest[q] = -1; continue; }
    if (b1 < bestDistTrain[bi]) { bestDistTrain[bi] = b1; bestForTrain[bi] = q; }
  }
  for (let q = 0; q < nq; q++) {
    const t = qBest[q];
    if (t >= 0 && bestForTrain[t] === q) matches.push([q, t, qDist[q]]);
  }
  return matches;
}
