// Image utilities: grayscale conversion, pyramids, gradients, blur, bilinear sampling.
// Images are Float32Array (row-major, values 0..255).

export function rgbaToGray(rgba, w, h, out = new Float32Array(w * h)) {
  const n = w * h;
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = 0.299 * rgba[j] + 0.587 * rgba[j + 1] + 0.114 * rgba[j + 2];
  }
  return out;
}

// Downsample by 2 with a [1 2 1]/4 binomial kernel in both directions.
export function pyrDown(src, w, h, out) {
  const w2 = w >> 1, h2 = h >> 1;
  if (!out) out = new Float32Array(w2 * h2);
  for (let y = 0; y < h2; y++) {
    const sy = 2 * y;
    const y0 = Math.max(sy - 1, 0) * w, y1 = sy * w, y2 = Math.min(sy + 1, h - 1) * w;
    for (let x = 0; x < w2; x++) {
      const sx = 2 * x;
      const x0 = Math.max(sx - 1, 0), x1 = sx, x2 = Math.min(sx + 1, w - 1);
      const r0 = src[y0 + x0] + 2 * src[y0 + x1] + src[y0 + x2];
      const r1 = src[y1 + x0] + 2 * src[y1 + x1] + src[y1 + x2];
      const r2 = src[y2 + x0] + 2 * src[y2 + x1] + src[y2 + x2];
      out[y * w2 + x] = (r0 + 2 * r1 + r2) * 0.0625;
    }
  }
  return out;
}

// Central-difference gradients (scaled by 0.5).
export function gradients(img, w, h, gx, gy) {
  if (!gx) gx = new Float32Array(w * h);
  if (!gy) gy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ym = Math.max(y - 1, 0) * w, yp = Math.min(y + 1, h - 1) * w, row = y * w;
    for (let x = 0; x < w; x++) {
      const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
      gx[row + x] = 0.5 * (img[row + xp] - img[row + xm]);
      gy[row + x] = 0.5 * (img[yp + x] - img[ym + x]);
    }
  }
  return { gx, gy };
}

export class Pyramid {
  constructor(w, h, levels) {
    this.levels = levels;
    this.width = []; this.height = [];
    this.img = []; this.gx = []; this.gy = [];
    let lw = w, lh = h;
    for (let l = 0; l < levels; l++) {
      this.width.push(lw); this.height.push(lh);
      this.img.push(new Float32Array(lw * lh));
      this.gx.push(new Float32Array(lw * lh));
      this.gy.push(new Float32Array(lw * lh));
      lw >>= 1; lh >>= 1;
    }
  }
  // Fill from a level-0 gray image (Float32Array of w*h).
  build(gray) {
    this.img[0].set(gray);
    for (let l = 1; l < this.levels; l++) {
      pyrDown(this.img[l - 1], this.width[l - 1], this.height[l - 1], this.img[l]);
    }
    for (let l = 0; l < this.levels; l++) {
      gradients(this.img[l], this.width[l], this.height[l], this.gx[l], this.gy[l]);
    }
    return this;
  }
}

// Bilinear sample with clamping. Caller must ensure x,y are inside [0, w-1] x [0, h-1] for accuracy.
export function bilinear(img, w, h, x, y) {
  let x0 = Math.floor(x), y0 = Math.floor(y);
  let fx = x - x0, fy = y - y0;
  if (x0 < 0) { x0 = 0; fx = 0; } else if (x0 >= w - 1) { x0 = w - 2; fx = 1; }
  if (y0 < 0) { y0 = 0; fy = 0; } else if (y0 >= h - 1) { y0 = h - 2; fy = 1; }
  const i = y0 * w + x0;
  const a = img[i], b = img[i + 1], c = img[i + w], d = img[i + w + 1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

// Separable box blur with radius r (window 2r+1), used before computing BRIEF descriptors.
export function boxBlur(src, w, h, r, out = new Float32Array(w * h), tmp = new Float32Array(w * h)) {
  const win = 2 * r + 1;
  // horizontal
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[row + Math.min(Math.max(x, 0), w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = s / win;
      const xa = Math.min(Math.max(x - r, 0), w - 1), xb = Math.min(x + r + 1, w - 1);
      s += src[row + xb] - src[row + xa];
    }
  }
  // vertical
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(Math.max(y, 0), h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / win;
      const ya = Math.min(Math.max(y - r, 0), h - 1), yb = Math.min(y + r + 1, h - 1);
      s += tmp[yb * w + x] - tmp[ya * w + x];
    }
  }
  return out;
}
