// End-to-end check of the camera code path: Chromium's fake camera plays a synthetic room video
// (test/make-y4m.mjs) through getUserMedia, and the app must initialize and build a map from it.
import { createRequire } from 'node:module';
import { execSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  try { return require('playwright'); } catch (_) { /* fall through */ }
  return require(path.join(execSync('npm root -g').toString().trim(), 'playwright'));
}
const { chromium } = loadPlaywright();
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = process.env.E2E_OUT || path.join(root, 'test', 'output');
mkdirSync(outDir, { recursive: true });
const video = path.join(outDir, 'room.y4m');
if (!existsSync(video)) execSync(`${process.execPath} ${path.join(root, 'test', 'make-y4m.mjs')} ${video} 240 640 480`, { stdio: 'inherit' });

const port = 8123 + Math.floor(Math.random() * 1000);
const server = spawn(process.execPath, [path.join(root, 'server.js'), String(port)], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 500));
const durationSec = Number(process.env.E2E_SECONDS || 20);
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    args: [
      '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
      '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
      `--use-file-for-fake-video-capture=${video}`,
    ],
  });
  const context = await browser.newContext({ viewport: { width: 1280, height: 800 }, permissions: ['camera'] });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(`http://localhost:${port}/`, { waitUntil: 'load' });
  const t0 = Date.now();
  let last = null, best = null;
  while (Date.now() - t0 < durationSec * 1000) {
    await page.waitForTimeout(1000);
    last = await page.evaluate(() => {
      const a = window.__vslam;
      const r = a.lastResult;
      return { mode: a.mode, ready: a.workerReady, frames: a.frameCounter, proc: [a.procW, a.procH], video: [a.videoW, a.videoH],
        state: r && r.state, hint: r && r.hint, stats: r && r.stats, fps: a.fps.value, overlay: document.getElementById('startOverlay').classList.contains('hidden') };
    });
    if (last.stats && (!best || last.stats.mapPoints > best.stats.mapPoints)) best = last;
    console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s]`, JSON.stringify(last));
  }
  await page.screenshot({ path: path.join(outDir, 'camera-final.png') });
  await page.locator('#videoWrap').screenshot({ path: path.join(outDir, 'camera-pip.png') });
  if (errors.length) console.log('browser errors:\n' + errors.join('\n'));
  const ok = last.mode === 'camera' && best && best.state === 'TRACKING' && best.stats.mapPoints > 200;
  console.log(ok ? 'E2E CAMERA OK' : 'E2E CAMERA FAILED');
  await browser.close();
  server.kill();
  process.exit(ok && errors.length === 0 ? 0 : 1);
} catch (err) {
  console.error(err);
  if (browser) await browser.close();
  server.kill();
  process.exit(1);
}
