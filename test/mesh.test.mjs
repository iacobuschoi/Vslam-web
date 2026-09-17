import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildKeyframeMesh } from '../js/slam/mesh.js';
import * as L from '../js/slam/linalg.js';

test('keyframe mesh triangulates a plane and does not bridge a depth discontinuity', () => {
  const w = 480, h = 360, f = 400, cx = 240, cy = 180;
  const pose = L.poseCreate();
  // Left half: plane at depth 2; right half: plane at depth 4 (a step in depth at u = 240).
  const pts2 = [], pts3 = [];
  for (let v = 20; v < h - 20; v += 20) {
    for (let u = 20; u < w - 20; u += 20) {
      const z = u < 240 ? 2 : 4;
      pts2.push(u + (u * 7 % 3), v + (v * 5 % 3));
      const px = pts2[pts2.length - 2], py = pts2[pts2.length - 1];
      pts3.push((px - cx) / f * z, (py - cy) / f * z, z);
    }
  }
  const n = pts2.length / 2;
  const mesh = buildKeyframeMesh(Float64Array.from(pts2), Float64Array.from(pts3), n, pose, w, h);
  assert.ok(mesh && mesh.triangleCount > 100, 'mesh built');
  // No triangle should mix vertices from the two depth levels.
  let mixed = 0;
  for (let k = 0; k < mesh.indices.length; k += 3) {
    const zs = [0, 1, 2].map((j) => mesh.positions[3 * mesh.indices[k + j] + 2]);
    if (Math.max(...zs) - Math.min(...zs) > 1) mixed++;
  }
  assert.equal(mixed, 0, `${mixed} triangles bridge the depth step`);
  // UVs inside [0,1].
  for (const v of mesh.uvs) assert.ok(v >= 0 && v <= 1);
});
