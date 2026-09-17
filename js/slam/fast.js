// FAST-9 corner detector with non-maximum suppression and grid-based selection.

const CIRCLE = [
  [0, -3], [1, -3], [2, -2], [3, -1], [3, 0], [3, 1], [2, 2], [1, 3],
  [0, 3], [-1, 3], [-2, 2], [-3, 1], [-3, 0], [-3, -1], [-2, -2], [-1, -3],
];

/**
 * Detect FAST-9 corners.
 * @param {Float32Array} img gray image
 * @param {number} w
 * @param {number} h
 * @param {number} threshold intensity threshold
 * @param {number} border keep corners at least this far from the image border
 * @param {Float32Array} [scoreMap] optional w*h scratch buffer (zeroed by this function)
 * @returns {{x:Int32Array,y:Int32Array,score:Float32Array,count:number}}
 */
export function detectFast(img, w, h, threshold, border = 8, scoreMap = null) {
  if (!scoreMap || scoreMap.length !== w * h) scoreMap = new Float32Array(w * h);
  else scoreMap.fill(0);
  const off = new Int32Array(16);
  for (let i = 0; i < 16; i++) off[i] = CIRCLE[i][1] * w + CIRCLE[i][0];
  const b = Math.max(border, 4);
  const d = new Float32Array(16);
  let count = 0;
  const candX = [], candY = [];
  for (let y = b; y < h - b; y++) {
    for (let x = b; x < w - b; x++) {
      const idx = y * w + x;
      const p = img[idx];
      const hi = p + threshold, lo = p - threshold;
      // Quick rejection with the 4 compass points.
      const p0 = img[idx + off[0]], p4 = img[idx + off[4]], p8 = img[idx + off[8]], p12 = img[idx + off[12]];
      let nb = (p0 > hi) + (p4 > hi) + (p8 > hi) + (p12 > hi);
      let nd = (p0 < lo) + (p4 < lo) + (p8 < lo) + (p12 < lo);
      if (nb < 2 && nd < 2) continue;
      for (let i = 0; i < 16; i++) d[i] = img[idx + off[i]] - p;
      // Look for 9 contiguous brighter or darker pixels (circular).
      let isCorner = false;
      for (let sign = 1; sign >= -1 && !isCorner; sign -= 2) {
        let run = 0, maxRun = 0;
        for (let i = 0; i < 32; i++) {
          const v = d[i & 15] * sign;
          if (v > threshold) { run++; if (run > maxRun) maxRun = run; if (maxRun >= 9) break; }
          else run = 0;
        }
        if (maxRun >= 9) {
          isCorner = true;
          let s = 0;
          for (let i = 0; i < 16; i++) { const v = d[i] * sign; if (v > threshold) s += v - threshold; }
          scoreMap[idx] = s;
        }
      }
      if (isCorner) { candX.push(x); candY.push(y); count++; }
    }
  }
  // Non-maximum suppression (3x3).
  const outX = new Int32Array(count), outY = new Int32Array(count), outS = new Float32Array(count);
  let n = 0;
  for (let i = 0; i < count; i++) {
    const x = candX[i], y = candY[i], idx = y * w + x, s = scoreMap[idx];
    if (s < scoreMap[idx - 1] || s < scoreMap[idx + 1] || s < scoreMap[idx - w] || s < scoreMap[idx + w] ||
      s < scoreMap[idx - w - 1] || s < scoreMap[idx - w + 1] || s < scoreMap[idx + w - 1] || s < scoreMap[idx + w + 1]) continue;
    outX[n] = x; outY[n] = y; outS[n] = s; n++;
  }
  return { x: outX.subarray(0, n), y: outY.subarray(0, n), score: outS.subarray(0, n), count: n };
}

/**
 * Pick the strongest corner per grid cell, skipping occupied cells.
 * occupied: Uint8Array of cellsX*cellsY (1 = cell already has a feature).
 * Returns array of [x, y, score].
 */
export function selectGridCorners(corners, w, h, cellSize, occupied, maxCount = Infinity) {
  const cellsX = Math.ceil(w / cellSize), cellsY = Math.ceil(h / cellSize);
  const best = new Int32Array(cellsX * cellsY).fill(-1);
  for (let i = 0; i < corners.count; i++) {
    const cx = (corners.x[i] / cellSize) | 0, cy = (corners.y[i] / cellSize) | 0;
    const c = cy * cellsX + cx;
    if (occupied && occupied[c]) continue;
    if (best[c] < 0 || corners.score[i] > corners.score[best[c]]) best[c] = i;
  }
  const out = [];
  for (let c = 0; c < best.length; c++) {
    const i = best[c];
    if (i >= 0) out.push([corners.x[i], corners.y[i], corners.score[i]]);
  }
  if (out.length > maxCount) {
    out.sort((a, b) => b[2] - a[2]);
    out.length = maxCount;
  }
  return out;
}
