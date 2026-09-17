// VSLAM Web: camera capture, worker orchestration, UI and 3D view.
import { MapViewer } from './viewer.js';
import { DemoSource } from './demo-scene.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

const app = {
  mode: null,          // 'camera' | 'demo'
  stream: null,
  facing: 'environment',
  worker: null,
  workerReady: false,
  busy: false,
  running: false,
  procW: 0, procH: 0,
  procCanvas: null, procCtx: null,
  settings: {
    fov: 75,
    procMax: 480,
    pointSize: 0.03,
    follow: false,
    fbCheck: true,
    relocalize: true,
    showKeyframes: true,
    showTrajectory: true,
    viewMode: 'mesh',      // 'mesh' | 'points' | 'both'
    meshing: true,
    texScale: 1,           // 1 = processing resolution, 2 = double (camera mode only)
    maxMeshes: 120,
  },
  viewer: null,
  demo: null,
  lastResult: null,
  map: null,
  keyframes: null,
  state: 'INIT',
  fps: { count: 0, last: performance.now(), value: 0 },
  procMs: 0,
  lastStatsUpdate: 0,
  frameCounter: 0,
  resultCounter: 0,
};
window.__vslam = app;
app.exportGlb = () => app.viewer.exportGLB();
app.setViewMode = (m) => setViewMode(m);

const els = {
  view3d: $('view3d'), video: $('video'), overlay: $('overlay'), videoWrap: $('videoWrap'),
  statusPill: $('statusPill'), hint: $('hint'), stats: $('stats'), toast: $('toast'),
  startOverlay: $('startOverlay'), startMsg: $('startMsg'),
  btnStart: $('btnStart'), btnStartDemo: $('btnStartDemo'),
  btnReset: $('btnReset'), btnExport: $('btnExport'), btnFlip: $('btnFlip'), btnFit: $('btnFit'),
  btnFollow: $('btnFollow'), btnSettings: $('btnSettings'), btnDemo: $('btnDemo'), btnCamera: $('btnCamera'),
  settings: $('settings'), btnCloseSettings: $('btnCloseSettings'),
  fovRange: $('fovRange'), fovValue: $('fovValue'), resSelect: $('resSelect'),
  sizeRange: $('sizeRange'), sizeValue: $('sizeValue'), fbCheck: $('fbCheck'), relocCheck: $('relocCheck'),
  kfCheck: $('kfCheck'), trajCheck: $('trajCheck'),
  btnView: $('btnView'), btnGlb: $('btnGlb'), meshCheck: $('meshCheck'), texSelect: $('texSelect'), maxMeshSelect: $('maxMeshSelect'),
};

// ---------- Settings persistence ----------
function loadSettings() {
  try {
    const raw = localStorage.getItem('vslam-settings');
    if (raw) Object.assign(app.settings, JSON.parse(raw));
  } catch (_) { /* ignore */ }
  if (params.get('fov')) app.settings.fov = +params.get('fov');
  if (params.get('res')) app.settings.procMax = +params.get('res');
  if (['mesh', 'points', 'both'].includes(params.get('view'))) app.settings.viewMode = params.get('view');
}
function saveSettings() {
  try { localStorage.setItem('vslam-settings', JSON.stringify(app.settings)); } catch (_) { /* ignore */ }
}

// ---------- UI helpers ----------
let toastTimer = null;
function toast(msg, ms = 3500) {
  els.toast.textContent = msg;
  els.toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), ms);
}

const STATUS_TEXT = {
  INIT: '초기화 중',
  TRACKING: '추적 중',
  LOST: '추적 실패',
};
const HINT_TEXT = {
  move: '카메라를 옆으로 천천히 움직이세요 (회전만 하면 초기화되지 않습니다)',
  lost: '이전에 스캔한 곳을 다시 비추면 자동으로 복구됩니다. 안 되면 리셋하세요.',
  '': '',
};

function setStatus(state, hint) {
  if (state !== app.state) {
    app.state = state;
    els.statusPill.textContent = STATUS_TEXT[state] || state;
    els.statusPill.className = 'pill ' + state.toLowerCase();
  }
  const text = HINT_TEXT[hint] ?? hint ?? '';
  if (els.hint.textContent !== text) els.hint.textContent = text;
}

function updateStats(res) {
  const now = performance.now();
  if (now - app.lastStatsUpdate < 200) return;
  app.lastStatsUpdate = now;
  const s = res.stats;
  els.stats.innerHTML = [
    `<span>처리 <b>${app.fps.value.toFixed(0)}</b> fps</span>`,
    `<span>연산 <b>${s.procMs.toFixed(0)}</b> ms</span>`,
    `<span>포인트 <b>${s.mapPoints}</b></span>`,
    `<span>키프레임 <b>${s.keyframes}</b></span>`,
    `<span>삼각형 <b>${app.viewer.meshTriangles}</b></span>`,
    `<span>추적점 <b>${s.inliers}</b>/${s.tracks}</span>`,
  ].join('');
}

function drawOverlay(features) {
  const ctx = els.overlay.getContext('2d');
  const w = els.overlay.width, h = els.overlay.height;
  ctx.clearRect(0, 0, w, h);
  if (!features) return;
  const n = features.length / 3;
  ctx.lineWidth = 1.5;
  for (const [status, color] of [[0, 'rgba(250, 204, 21, 0.9)'], [1, 'rgba(74, 222, 128, 0.95)']]) {
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      if (features[3 * i + 2] !== status) continue;
      const x = features[3 * i], y = features[3 * i + 1];
      ctx.moveTo(x + 3, y);
      ctx.arc(x, y, 3, 0, Math.PI * 2);
    }
    ctx.stroke();
  }
}

// ---------- Worker ----------
function initWorker() {
  if (app.worker) app.worker.terminate();
  app.worker = new Worker('./js/slam-worker.js', { type: 'module' });
  app.workerReady = false;
  app.busy = false;
  app.worker.onmessage = (e) => {
    const msg = e.data;
    switch (msg.type) {
      case 'ready':
        app.workerReady = true;
        app.busy = false;
        break;
      case 'result':
        onResult(msg);
        break;
      case 'reset':
        app.busy = false;
        break;
      case 'error':
        app.busy = false;
        console.error('SLAM worker error:', msg.message);
        toast('SLAM 오류가 발생해 재시작했습니다.');
        break;
      case 'export':
        downloadText(msg.ply, `vslam-map-${timestamp()}.ply`);
        break;
    }
  };
  app.worker.onerror = (err) => {
    console.error('worker failed', err);
    toast('워커 로드 실패: ' + (err.message || err));
  };
}

function workerOptions() {
  return {
    fovDeg: app.settings.fov,
    fbCheck: app.settings.fbCheck,
    relocalize: app.settings.relocalize,
    meshing: app.settings.meshing,
  };
}

function onResult(res) {
  app.busy = false;
  app.lastResult = res;
  app.resultCounter++;
  app.fps.count++;
  const now = performance.now();
  if (now - app.fps.last >= 1000) {
    app.fps.value = app.fps.count * 1000 / (now - app.fps.last);
    app.fps.count = 0; app.fps.last = now;
  }
  setStatus(res.state, res.hint);
  drawOverlay(res.features);
  app.viewer.setCameraPose(res.pose);
  if (res.mesh) {
    let tex;
    if (app.mode === 'camera' && app.settings.texScale > 1 && app.texCanvas) {
      const c = document.createElement('canvas');
      c.width = app.texCanvas.width; c.height = app.texCanvas.height;
      c.getContext('2d').drawImage(app.texCanvas, 0, 0);
      tex = c;
    } else {
      tex = { rgba: res.mesh.rgba, width: res.mesh.width, height: res.mesh.height };
    }
    app.viewer.addKeyframeMesh(res.mesh, tex);
  }
  if (res.map) {
    app.map = res.map;
    app.viewer.setMap(res.map.positions, res.map.colors, res.map.count);
    app.viewer.setKeyframes(res.map.keyframes, res.map.keyframeCount);
    // Auto-fit the view while the user has not taken control of the camera.
    if (res.map.count > 0 && !app.viewer.userInteracted && (!app.fitDone || res.map.keyframeCount <= 12 || res.map.keyframeCount % 10 === 0)) {
      app.viewer.fitView();
      app.fitDone = true;
    }
  }
  updateStats(res);
}

// ---------- Frame sources ----------
function computeProcSize(vw, vh) {
  const maxSide = app.settings.procMax;
  const scale = Math.min(1, maxSide / Math.max(vw, vh));
  let pw = Math.round(vw * scale / 2) * 2, ph = Math.round(vh * scale / 2) * 2;
  pw = Math.max(pw, 64); ph = Math.max(ph, 64);
  return [pw, ph];
}

function setupProcessing(pw, ph) {
  app.procW = pw; app.procH = ph;
  if (!app.procCanvas) app.procCanvas = document.createElement('canvas');
  app.procCanvas.width = pw; app.procCanvas.height = ph;
  app.procCtx = app.procCanvas.getContext('2d', { willReadFrequently: true });
  els.overlay.width = pw; els.overlay.height = ph;
  els.videoWrap.style.aspectRatio = `${pw} / ${ph}`;
  app.viewer.setCameraIntrinsics(pw, ph, app.settings.fov);
  app.workerReady = false;
  app.busy = false;
  app.fitDone = false;
  app.worker.postMessage({ type: 'init', width: pw, height: ph, options: workerOptions() });
  app.viewer.clear();
  app.map = null;
}

async function startCamera(facing) {
  stopSources();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(window.isSecureContext ? '이 브라우저는 카메라 API를 지원하지 않습니다.' : 'HTTPS(또는 localhost)에서만 카메라를 사용할 수 있습니다.');
  }
  const constraints = {
    audio: false,
    video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  };
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    // Fall back to any camera.
    stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
  }
  app.stream = stream;
  app.facing = facing;
  els.video.srcObject = stream;
  els.video.style.display = '';
  if (app.demo) { app.demo.canvas.remove(); }
  await new Promise((resolve, reject) => {
    const onMeta = () => { els.video.removeEventListener('loadedmetadata', onMeta); resolve(); };
    els.video.addEventListener('loadedmetadata', onMeta);
    setTimeout(() => reject(new Error('카메라 메타데이터 대기 시간 초과')), 8000);
  });
  await els.video.play();
  const vw = els.video.videoWidth, vh = els.video.videoHeight;
  app.videoW = vw; app.videoH = vh;
  const [pw, ph] = computeProcSize(vw, vh);
  setupProcessing(pw, ph);
  app.mode = 'camera';
  app.running = true;
  els.startOverlay.classList.add('hidden');
  els.btnFlip.disabled = false;
  toast(`카메라 ${vw}×${vh} → 처리 ${pw}×${ph}`);
}

function startDemo() {
  stopSources();
  const maxSide = app.settings.procMax;
  const pw = maxSide, ph = Math.round(maxSide * 3 / 4 / 2) * 2;
  app.demo = new DemoSource(pw, ph, app.settings.fov);
  els.video.style.display = 'none';
  app.demo.canvas.id = 'demoCanvas';
  els.videoWrap.insertBefore(app.demo.canvas, els.overlay);
  setupProcessing(pw, ph);
  app.mode = 'demo';
  app.running = true;
  els.startOverlay.classList.add('hidden');
  els.btnFlip.disabled = true;
  toast('데모 모드: 가상의 방을 스캔합니다');
}

function stopSources() {
  app.running = false;
  if (app.stream) { app.stream.getTracks().forEach((t) => t.stop()); app.stream = null; }
  els.video.srcObject = null;
  if (app.demo) { app.demo.dispose(); app.demo.canvas.remove(); app.demo = null; }
}

function grabFrame() {
  if (app.mode === 'camera') {
    if (els.video.readyState < 2) return null;
    const vw = els.video.videoWidth, vh = els.video.videoHeight;
    if (vw && vh && (vw !== app.videoW || vh !== app.videoH)) {
      // Stream size changed (e.g. device rotation): rebuild the processing pipeline.
      app.videoW = vw; app.videoH = vh;
      const [pw, ph] = computeProcSize(vw, vh);
      if (pw !== app.procW || ph !== app.procH) { setupProcessing(pw, ph); return null; }
    }
    app.procCtx.drawImage(els.video, 0, 0, app.procW, app.procH);
    if (app.settings.texScale > 1 && app.settings.meshing) {
      // Higher-resolution copy of the same frame, used as the mesh texture if this frame becomes a keyframe.
      const tw = app.procW * app.settings.texScale, th = app.procH * app.settings.texScale;
      if (!app.texCanvas) app.texCanvas = document.createElement('canvas');
      if (app.texCanvas.width !== tw || app.texCanvas.height !== th) { app.texCanvas.width = tw; app.texCanvas.height = th; app.texCtx = app.texCanvas.getContext('2d'); }
      app.texCtx.drawImage(els.video, 0, 0, tw, th);
    }
    return app.procCtx.getImageData(0, 0, app.procW, app.procH);
  }
  if (app.mode === 'demo') return app.demo.grab(performance.now());
  return null;
}

function loop() {
  requestAnimationFrame(loop);
  if (app.running && app.workerReady && !app.busy && !document.hidden) {
    const img = grabFrame();
    if (img) {
      app.busy = true;
      app.frameCounter++;
      app.worker.postMessage({ type: 'frame', buffer: img.data.buffer, width: img.width, height: img.height, time: performance.now() }, [img.data.buffer]);
    }
  }
  app.viewer.render();
}

// ---------- Actions ----------
function resetMap() {
  if (!app.worker) return;
  app.worker.postMessage({ type: 'reset' });
  app.viewer.clear();
  app.map = null;
  app.fitDone = false;
  toast('맵을 초기화했습니다');
}

function timestamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function downloadText(text, name) {
  const blob = new Blob([text], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 1000);
}

async function exportGlb() {
  if (app.viewer.meshes.length === 0) { toast('저장할 메시가 아직 없습니다'); return; }
  try {
    toast('GLB 파일을 만드는 중...');
    const buf = await app.viewer.exportGLB();
    const blob = new Blob([buf], { type: 'model/gltf-binary' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `vslam-map-${timestamp()}.glb`;
    document.body.appendChild(a); a.click();
    setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 1000);
    toast(`GLB 저장 완료 (${(blob.size / 1048576).toFixed(1)} MB, 메시 ${app.viewer.meshes.length}개)`);
  } catch (err) {
    console.error(err);
    toast('GLB 저장 실패: ' + (err.message || err));
  }
}

function setViewMode(mode) {
  app.settings.viewMode = mode;
  app.viewer.setViewMode(mode);
  const label = { mesh: '🖼 사진', points: '· 점', both: '🖼 사진+점' }[mode] || mode;
  els.btnView.textContent = label;
  saveSettings();
}

function exportPly() {
  const m = app.map;
  if (!m || m.count === 0) { toast('저장할 포인트가 아직 없습니다'); return; }
  const lines = ['ply', 'format ascii 1.0', `element vertex ${m.count}`,
    'property float x', 'property float y', 'property float z',
    'property uchar red', 'property uchar green', 'property uchar blue', 'end_header'];
  for (let i = 0; i < m.count; i++) {
    lines.push(`${m.positions[3 * i].toFixed(4)} ${m.positions[3 * i + 1].toFixed(4)} ${m.positions[3 * i + 2].toFixed(4)} ${m.colors[3 * i]} ${m.colors[3 * i + 1]} ${m.colors[3 * i + 2]}`);
  }
  downloadText(lines.join('\n') + '\n', `vslam-map-${timestamp()}.ply`);
  toast(`${m.count}개 포인트를 PLY로 저장했습니다`);
}

async function flipCamera() {
  const next = app.facing === 'environment' ? 'user' : 'environment';
  try { await startCamera(next); } catch (err) { showStartOverlay(err.message); }
}

function showStartOverlay(message) {
  els.startMsg.textContent = message || '';
  els.startOverlay.classList.remove('hidden');
}

function applySettingsToUI() {
  const s = app.settings;
  els.fovRange.value = s.fov; els.fovValue.textContent = `${s.fov}°`;
  els.resSelect.value = String(s.procMax);
  els.sizeRange.value = s.pointSize; els.sizeValue.textContent = s.pointSize.toFixed(3);
  els.fbCheck.checked = s.fbCheck;
  els.relocCheck.checked = s.relocalize;
  els.kfCheck.checked = s.showKeyframes;
  els.trajCheck.checked = s.showTrajectory;
  els.btnFollow.classList.toggle('active', s.follow);
  els.meshCheck.checked = s.meshing;
  els.texSelect.value = String(s.texScale);
  els.maxMeshSelect.value = String(s.maxMeshes);
  app.viewer.setMaxMeshes(s.maxMeshes);
  setViewMode(s.viewMode);
  app.viewer.setPointSize(s.pointSize);
  app.viewer.setFollow(s.follow);
  app.viewer.kfLines.visible = s.showKeyframes;
  app.viewer.traj.visible = s.showTrajectory;
}

function bindUI() {
  els.btnStart.addEventListener('click', async () => {
    els.startMsg.textContent = '카메라 권한을 요청하는 중...';
    try { await startCamera(app.facing); } catch (err) { showStartOverlay('카메라를 열 수 없습니다: ' + err.message); }
  });
  els.btnStartDemo.addEventListener('click', () => startDemo());
  els.btnDemo.addEventListener('click', () => startDemo());
  els.btnCamera.addEventListener('click', async () => {
    try { await startCamera(app.facing); } catch (err) { showStartOverlay('카메라를 열 수 없습니다: ' + err.message); }
  });
  els.btnReset.addEventListener('click', resetMap);
  els.btnExport.addEventListener('click', exportPly);
  els.btnFlip.addEventListener('click', flipCamera);
  els.btnFit.addEventListener('click', () => app.viewer.fitView());
  els.btnFollow.addEventListener('click', () => {
    app.settings.follow = !app.settings.follow;
    app.viewer.setFollow(app.settings.follow);
    els.btnFollow.classList.toggle('active', app.settings.follow);
    saveSettings();
  });
  els.btnSettings.addEventListener('click', () => els.settings.classList.toggle('hidden'));
  els.btnView.addEventListener('click', () => {
    const order = ['mesh', 'points', 'both'];
    setViewMode(order[(order.indexOf(app.settings.viewMode) + 1) % order.length]);
  });
  els.btnGlb.addEventListener('click', exportGlb);
  els.meshCheck.addEventListener('change', () => {
    app.settings.meshing = els.meshCheck.checked; saveSettings();
    app.worker.postMessage({ type: 'options', options: { meshing: app.settings.meshing } });
  });
  els.texSelect.addEventListener('change', () => { app.settings.texScale = +els.texSelect.value; saveSettings(); });
  els.maxMeshSelect.addEventListener('change', () => { app.settings.maxMeshes = +els.maxMeshSelect.value; app.viewer.setMaxMeshes(app.settings.maxMeshes); saveSettings(); });
  els.btnCloseSettings.addEventListener('click', () => els.settings.classList.add('hidden'));
  els.videoWrap.addEventListener('click', () => els.videoWrap.classList.toggle('large'));

  els.fovRange.addEventListener('input', () => { els.fovValue.textContent = `${els.fovRange.value}°`; });
  els.fovRange.addEventListener('change', () => {
    app.settings.fov = +els.fovRange.value;
    saveSettings();
    app.viewer.setCameraIntrinsics(app.procW, app.procH, app.settings.fov);
    app.worker.postMessage({ type: 'options', options: { fovDeg: app.settings.fov }, reset: true });
    app.viewer.clear(); app.map = null; app.fitDone = false;
    if (app.mode === 'demo') startDemo();
    toast('시야각을 변경해 맵을 초기화했습니다');
  });
  els.resSelect.addEventListener('change', () => {
    app.settings.procMax = +els.resSelect.value;
    saveSettings();
    if (app.mode === 'camera') {
      const [pw, ph] = computeProcSize(els.video.videoWidth, els.video.videoHeight);
      setupProcessing(pw, ph);
    } else if (app.mode === 'demo') startDemo();
  });
  els.sizeRange.addEventListener('input', () => {
    app.settings.pointSize = +els.sizeRange.value;
    els.sizeValue.textContent = app.settings.pointSize.toFixed(3);
    app.viewer.setPointSize(app.settings.pointSize);
    saveSettings();
  });
  els.fbCheck.addEventListener('change', () => {
    app.settings.fbCheck = els.fbCheck.checked; saveSettings();
    app.worker.postMessage({ type: 'options', options: { fbCheck: app.settings.fbCheck } });
  });
  els.relocCheck.addEventListener('change', () => {
    app.settings.relocalize = els.relocCheck.checked; saveSettings();
    app.worker.postMessage({ type: 'options', options: { relocalize: app.settings.relocalize } });
  });
  els.kfCheck.addEventListener('change', () => { app.settings.showKeyframes = els.kfCheck.checked; app.viewer.kfLines.visible = els.kfCheck.checked; saveSettings(); });
  els.trajCheck.addEventListener('change', () => { app.settings.showTrajectory = els.trajCheck.checked; app.viewer.traj.visible = els.trajCheck.checked; saveSettings(); });

  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    if (e.key === 'r') resetMap();
    else if (e.key === 'f') app.viewer.fitView();
    else if (e.key === 'e') exportPly();
    else if (e.key === 'g') exportGlb();
    else if (e.key === 'v') els.btnView.click();
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) app.busy = false; });
}

// ---------- Boot ----------
async function boot() {
  loadSettings();
  app.viewer = new MapViewer(els.view3d);
  initWorker();
  bindUI();
  applySettingsToUI();
  requestAnimationFrame(loop);
  if (params.get('demo') === '1') { startDemo(); return; }
  if (params.get('autostart') === '0') { showStartOverlay('시작 버튼을 누르면 카메라가 켜집니다.'); return; }
  try {
    await startCamera(app.facing);
  } catch (err) {
    console.warn('camera auto-start failed:', err);
    let msg = '카메라를 자동으로 켤 수 없습니다: ' + (err && err.message ? err.message : err);
    if (!window.isSecureContext) msg = '카메라는 HTTPS(또는 localhost)에서만 동작합니다. 서버를 HTTPS로 열거나 데모 모드를 사용하세요.';
    else if (err && err.name === 'NotAllowedError') msg = '카메라 권한이 거부되었습니다. 브라우저 주소창의 권한 설정에서 허용한 뒤 다시 시도하세요.';
    else if (err && err.name === 'NotFoundError') msg = '사용 가능한 카메라를 찾지 못했습니다. 데모 모드로 체험할 수 있습니다.';
    showStartOverlay(msg);
  }
}

boot();
