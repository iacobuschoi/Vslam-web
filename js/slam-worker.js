// Web Worker entry: runs the SLAM pipeline off the main thread.
import { Slam } from './slam/slam.js';

let slam = null;
let options = {};

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

self.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init': {
      options = Object.assign({}, msg.options || {});
      slam = new Slam(msg.width, msg.height, options);
      post({ type: 'ready', width: msg.width, height: msg.height, cam: slam.cam });
      break;
    }
    case 'frame': {
      if (!slam) return;
      if (msg.width !== slam.w || msg.height !== slam.h) {
        slam = new Slam(msg.width, msg.height, options);
      }
      const rgba = new Uint8ClampedArray(msg.buffer);
      let res;
      try {
        res = slam.processFrame(rgba);
      } catch (err) {
        post({ type: 'error', message: String(err && err.stack || err) });
        slam.reset();
        return;
      }
      const transfer = [res.features.buffer];
      if (res.pose) transfer.push(res.pose.buffer);
      if (res.map) transfer.push(res.map.positions.buffer, res.map.colors.buffer, res.map.keyframes.buffer);
      post({ type: 'result', ...res, time: msg.time }, transfer);
      break;
    }
    case 'reset': {
      if (!slam) return;
      slam.reset();
      post({ type: 'reset' });
      break;
    }
    case 'options': {
      if (!slam) return;
      Object.assign(options, msg.options || {});
      if (msg.options && msg.options.fovDeg !== undefined) slam.setFov(msg.options.fovDeg);
      if (msg.options && msg.options.fbCheck !== undefined) slam.klt.fbCheck = !!msg.options.fbCheck;
      if (msg.options && msg.options.relocalize !== undefined) slam.opts.relocalize = !!msg.options.relocalize;
      if (msg.reset) { slam.reset(); post({ type: 'reset' }); }
      break;
    }
    case 'export': {
      if (!slam) return;
      post({ type: 'export', ply: slam.exportPly() });
      break;
    }
  }
};
