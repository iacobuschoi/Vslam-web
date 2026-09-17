// Render the synthetic room to a Y4M video for Chromium's fake camera (--use-file-for-fake-video-capture).
// Usage: node test/make-y4m.mjs out.y4m [frames] [width] [height]
import { writeFileSync, openSync, writeSync, closeSync } from 'node:fs';
import { makeRoom, renderRoom, cameraPose } from './synth-room.mjs';

const out = process.argv[2] || 'test/output/room.y4m';
const frames = Number(process.argv[3] || 240);
const w = Number(process.argv[4] || 640), h = Number(process.argv[5] || 480);
const diag = Math.hypot(w, h);
const cam = { f: (diag / 2) / Math.tan((75 * Math.PI / 180) / 2), cx: w / 2, cy: h / 2 };
const planes = makeRoom();
const fd = openSync(out, 'w');
writeSync(fd, `YUV4MPEG2 W${w} H${h} F30:1 Ip A1:1 C420jpeg\n`);
const rgba = new Uint8ClampedArray(w * h * 4);
const Y = new Uint8Array(w * h), U = new Uint8Array((w / 2) * (h / 2)), V = new Uint8Array((w / 2) * (h / 2));
for (let k = 0; k < frames; k++) {
  // Slow the ground-truth path down (fake capture plays at 30 fps).
  renderRoom(planes, cameraPose(k * 0.6), cam, w, h, rgba);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = 4 * (y * w + x);
    const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2];
    Y[y * w + x] = Math.max(0, Math.min(255, Math.round(0.299 * r + 0.587 * g + 0.114 * b)));
    if ((x & 1) === 0 && (y & 1) === 0) {
      const j = (y / 2) * (w / 2) + x / 2;
      U[j] = Math.max(0, Math.min(255, Math.round(-0.169 * r - 0.331 * g + 0.5 * b + 128)));
      V[j] = Math.max(0, Math.min(255, Math.round(0.5 * r - 0.419 * g - 0.081 * b + 128)));
    }
  }
  writeSync(fd, 'FRAME\n');
  writeSync(fd, Y); writeSync(fd, U); writeSync(fd, V);
}
closeSync(fd);
console.log(`wrote ${out}: ${frames} frames ${w}x${h}`);
