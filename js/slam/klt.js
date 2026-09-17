// Pyramidal Lucas-Kanade feature tracker (forward-additive with template gradients),
// with an optional forward-backward consistency check.

import { Pyramid } from './image.js';

export class KLTTracker {
  constructor(opts = {}) {
    this.winRadius = opts.winRadius ?? 7;
    this.maxIter = opts.maxIter ?? 25;
    this.eps = opts.eps ?? 0.03;
    this.minEigPerPixel = opts.minEig ?? 0.5;
    this.maxError = opts.maxError ?? 40;
    this.fbCheck = opts.fbCheck ?? true;
    this.fbThreshold = opts.fbThreshold ?? 1.5;
    const win = 2 * this.winRadius + 1;
    this._tI = new Float32Array(win * win);
    this._tX = new Float32Array(win * win);
    this._tY = new Float32Array(win * win);
  }

  // Build the template (intensity + gradients) from image A at (px, py). Returns [gxx, gxy, gyy].
  _buildTemplate(A, L, px, py) {
    const r = this.winRadius;
    const w = A.width[L], h = A.height[L];
    const imgA = A.img[L], gxA = A.gx[L], gyA = A.gy[L];
    const tI = this._tI, tX = this._tX, tY = this._tY;
    const x0 = Math.floor(px), y0 = Math.floor(py);
    const fx = px - x0, fy = py - y0;
    const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    let gxx = 0, gxy = 0, gyy = 0, k = 0, tSum = 0, gxSum = 0, gySum = 0;
    if (x0 - r >= 0 && y0 - r >= 0 && x0 + r + 1 < w && y0 + r + 1 < h) {
      for (let j = -r; j <= r; j++) {
        let idx = (y0 + j) * w + (x0 - r);
        for (let i = -r; i <= r; i++, idx++, k++) {
          const iv = imgA[idx] * w00 + imgA[idx + 1] * w10 + imgA[idx + w] * w01 + imgA[idx + w + 1] * w11;
          const gx = gxA[idx] * w00 + gxA[idx + 1] * w10 + gxA[idx + w] * w01 + gxA[idx + w + 1] * w11;
          const gy = gyA[idx] * w00 + gyA[idx + 1] * w10 + gyA[idx + w] * w01 + gyA[idx + w + 1] * w11;
          tI[k] = iv; tX[k] = gx; tY[k] = gy;
          gxx += gx * gx; gxy += gx * gy; gyy += gy * gy;
          tSum += iv; gxSum += gx; gySum += gy;
        }
      }
    } else {
      for (let j = -r; j <= r; j++) {
        const ya = Math.min(Math.max(y0 + j, 0), h - 1) * w, yb = Math.min(Math.max(y0 + j + 1, 0), h - 1) * w;
        for (let i = -r; i <= r; i++, k++) {
          const xa = Math.min(Math.max(x0 + i, 0), w - 1), xb = Math.min(Math.max(x0 + i + 1, 0), w - 1);
          const iv = imgA[ya + xa] * w00 + imgA[ya + xb] * w10 + imgA[yb + xa] * w01 + imgA[yb + xb] * w11;
          const gx = gxA[ya + xa] * w00 + gxA[ya + xb] * w10 + gxA[yb + xa] * w01 + gxA[yb + xb] * w11;
          const gy = gyA[ya + xa] * w00 + gyA[ya + xb] * w10 + gyA[yb + xa] * w01 + gyA[yb + xb] * w11;
          tI[k] = iv; tX[k] = gx; tY[k] = gy;
          gxx += gx * gx; gxy += gx * gy; gyy += gy * gy;
          tSum += iv; gxSum += gx; gySum += gy;
        }
      }
    }
    this._tMean = tSum / k; this._gxSum = gxSum; this._gySum = gySum;
    return [gxx, gxy, gyy];
  }

  // Accumulate b = sum (T - B(q)) * gradT over the window at (qx, qy) in image B, compensating for a
  // brightness offset between the two windows (robust to auto-exposure changes). Returns [bx, by, meanAbsErr].
  _residual(B, L, qx, qy, offsetGuess = 0) {
    const r = this.winRadius;
    const w = B.width[L], h = B.height[L];
    const imgB = B.img[L];
    const tI = this._tI, tX = this._tX, tY = this._tY;
    const x0 = Math.floor(qx), y0 = Math.floor(qy);
    const fx = qx - x0, fy = qy - y0;
    const v00 = (1 - fx) * (1 - fy), v10 = fx * (1 - fy), v01 = (1 - fx) * fy, v11 = fx * fy;
    let bx = 0, by = 0, e = 0, k = 0, iSum = 0;
    if (x0 - r >= 0 && y0 - r >= 0 && x0 + r + 1 < w && y0 + r + 1 < h) {
      for (let j = -r; j <= r; j++) {
        let idx = (y0 + j) * w + (x0 - r);
        for (let i = -r; i <= r; i++, idx++, k++) {
          const iv = imgB[idx] * v00 + imgB[idx + 1] * v10 + imgB[idx + w] * v01 + imgB[idx + w + 1] * v11;
          const d = tI[k] - iv;
          bx += d * tX[k]; by += d * tY[k]; e += Math.abs(d - offsetGuess); iSum += iv;
        }
      }
    } else {
      for (let j = -r; j <= r; j++) {
        const ya = Math.min(Math.max(y0 + j, 0), h - 1) * w, yb = Math.min(Math.max(y0 + j + 1, 0), h - 1) * w;
        for (let i = -r; i <= r; i++, k++) {
          const xa = Math.min(Math.max(x0 + i, 0), w - 1), xb = Math.min(Math.max(x0 + i + 1, 0), w - 1);
          const iv = imgB[ya + xa] * v00 + imgB[ya + xb] * v10 + imgB[yb + xa] * v01 + imgB[yb + xb] * v11;
          const d = tI[k] - iv;
          bx += d * tX[k]; by += d * tY[k]; e += Math.abs(d - offsetGuess); iSum += iv;
        }
      }
    }
    // Offset between template and current window means; remove its contribution from b.
    const offset = this._tMean - iSum / k;
    bx -= offset * this._gxSum; by -= offset * this._gySum;
    this._lastOffset = offset;
    return [bx, by, e];
  }

  /**
   * Track one point from pyramid A to pyramid B.
   * @returns {boolean} success; on success out = [x, y, error]
   */
  trackPoint(A, B, x, y, out, guessDx = 0, guessDy = 0) {
    const r = this.winRadius, win = 2 * r + 1, area = win * win;
    const levels = A.levels;
    const scaleTop = 1 / (1 << (levels - 1));
    let dx = guessDx * scaleTop, dy = guessDy * scaleTop;
    let err = 0;
    for (let L = levels - 1; L >= 0; L--) {
      const w = A.width[L], h = A.height[L];
      const s = 1 / (1 << L);
      const px = x * s, py = y * s;
      // The point itself must be inside the image (allow the window to be clamped at borders).
      const margin = L === 0 ? 2 : 1;
      if (px < margin || py < margin || px > w - 1 - margin || py > h - 1 - margin) {
        if (L === 0) return false;
        dx *= 2; dy *= 2; continue;
      }
      const [gxx, gxy, gyy] = this._buildTemplate(A, L, px, py);
      const det = gxx * gyy - gxy * gxy;
      const tr = gxx + gyy;
      const minEig = 0.5 * (tr - Math.sqrt(Math.max(tr * tr - 4 * det, 0)));
      if (minEig / area < this.minEigPerPixel || det < 1e-12) {
        if (L === 0) return false;
        dx *= 2; dy *= 2; continue;
      }
      const iDet = 1 / det;
      this._lastOffset = 0;
      for (let it = 0; it < this.maxIter; it++) {
        const qx = px + dx, qy = py + dy;
        if (qx < margin || qy < margin || qx > w - 1 - margin || qy > h - 1 - margin) return false;
        const [bx, by, e] = this._residual(B, L, qx, qy, this._lastOffset);
        const ddx = (gyy * bx - gxy * by) * iDet;
        const ddy = (-gxy * bx + gxx * by) * iDet;
        dx += ddx; dy += ddy;
        err = e / area;
        if (ddx * ddx + ddy * ddy < this.eps * this.eps) break;
      }
      if (L > 0) { dx *= 2; dy *= 2; }
    }
    if (err > this.maxError) return false;
    const nx = x + dx, ny = y + dy;
    if (nx < 2 || ny < 2 || nx > A.width[0] - 3 || ny > A.height[0] - 3) return false;
    out[0] = nx; out[1] = ny; out[2] = err;
    return true;
  }

  /**
   * Track many points. pts: Float32Array [x0,y0,x1,y1,...].
   * @returns {{next: Float32Array, status: Uint8Array, err: Float32Array}}
   */
  track(A, B, pts, n, guess = null) {
    const next = new Float32Array(n * 2);
    const status = new Uint8Array(n);
    const err = new Float32Array(n);
    const out = new Float64Array(3);
    const back = new Float64Array(3);
    for (let i = 0; i < n; i++) {
      const x = pts[2 * i], y = pts[2 * i + 1];
      const gdx = guess ? guess[2 * i] - x : 0, gdy = guess ? guess[2 * i + 1] - y : 0;
      let ok = this.trackPoint(A, B, x, y, out, gdx, gdy);
      if (ok && this.fbCheck) {
        ok = this.trackPoint(B, A, out[0], out[1], back, x - out[0], y - out[1]);
        if (ok) {
          const ex = back[0] - x, ey = back[1] - y;
          if (ex * ex + ey * ey > this.fbThreshold * this.fbThreshold) ok = false;
        }
      }
      status[i] = ok ? 1 : 0;
      if (ok) { next[2 * i] = out[0]; next[2 * i + 1] = out[1]; err[i] = out[2]; }
      else { next[2 * i] = x; next[2 * i + 1] = y; }
    }
    return { next, status, err };
  }
}

export { Pyramid };
