// End-to-end check in headless Chromium: runs the demo scene through the full web app
// (worker + Three.js viewer) and verifies that the SLAM initializes and builds a map.
// Requires Playwright (npm i -g playwright && npx playwright install chromium).
import { createRequire } from 'node:module';
import { execSync, spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
function loadPlaywright() {
  try { return require('playwright'); } catch (_) { /* fall through */ }
  const globalRoot = execSync('npm root -g').toString().trim();
  return require(path.join(globalRoot, 'playwright'));
}
const { chromium } = loadPlaywright();

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = 8123 + Math.floor(Math.random() * 1000);
const outDir = process.env.E2E_OUT || path.join(root, 'test', 'output');
mkdirSync(outDir, { recursive: true });

const server = spawn(process.execPath, [path.join(root, 'server.js'), String(port)], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 500));

const durationSec = Number(process.env.E2E_SECONDS || 25);
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await page.goto(`http://localhost:${port}/?demo=1`, { waitUntil: 'load' });
  const t0 = Date.now();
  let last = null;
  while (Date.now() - t0 < durationSec * 1000) {
    await page.waitForTimeout(1000);
    last = await page.evaluate(() => {
      const a = window.__vslam;
      const r = a.lastResult;
      return r ? { state: r.state, hint: r.hint, stats: r.stats, frames: a.frameCounter, results: a.resultCounter, fps: a.fps.value, mode: a.mode, ready: a.workerReady } : { mode: a.mode, ready: a.workerReady, frames: a.frameCounter };
    });
    const sec = ((Date.now() - t0) / 1000).toFixed(0);
    console.log(`[${sec}s]`, JSON.stringify(last));
    if (sec === '8' || sec === '16') await page.screenshot({ path: path.join(outDir, `demo-${sec}s.png`) });
  }
  await page.screenshot({ path: path.join(outDir, 'demo-final.png') });
  await page.locator('#videoWrap').screenshot({ path: path.join(outDir, 'demo-pip.png') });
  const overlayPixels = await page.evaluate(() => {
    const c = document.getElementById('overlay');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let n = 0;
    for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
    return n;
  });
  console.log('overlay painted pixels:', overlayPixels);
  if (overlayPixels < 100) errors.push('feature overlay is empty');
  if (errors.length) console.log('browser errors:\n' + errors.join('\n'));
  const ok = last && last.state === 'TRACKING' && last.stats.mapPoints > 300 && last.stats.keyframes >= 3;
  console.log(ok ? 'E2E OK' : 'E2E FAILED');
  await browser.close();
  server.kill();
  process.exit(ok && errors.length === 0 ? 0 : 1);
} catch (err) {
  console.error(err);
  if (browser) await browser.close();
  server.kill();
  process.exit(1);
}
