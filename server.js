#!/usr/bin/env node
// Minimal static file server for VSLAM Web (no dependencies).
// Usage: node server.js [port]
// HTTPS: put key.pem and cert.pem next to this file (or set SSL_KEY / SSL_CERT), and the server switches to HTTPS.
// Cameras only work on HTTPS or localhost, so use HTTPS when opening the page from a phone.
import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.argv[2] || process.env.PORT || 8080);
const keyPath = process.env.SSL_KEY || path.join(root, 'key.pem');
const certPath = process.env.SSL_CERT || path.join(root, 'cert.pem');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
  '.ply': 'application/octet-stream', '.md': 'text/markdown; charset=utf-8', '.txt': 'text/plain; charset=utf-8',
};

function handler(req, res) {
  let urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  if (urlPath.endsWith('/')) urlPath += 'index.html';
  const filePath = path.normalize(path.join(root, urlPath));
  if (!filePath.startsWith(root)) { res.writeHead(403); res.end('Forbidden'); return; }
  fs.stat(filePath, (err, st) => {
    if (err || !st.isFile()) { res.writeHead(404); res.end('Not found'); return; }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Content-Length': st.size,
      'Cache-Control': 'no-cache',
    });
    fs.createReadStream(filePath).pipe(res);
  });
}

let server, proto;
if (fs.existsSync(keyPath) && fs.existsSync(certPath)) {
  server = https.createServer({ key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) }, handler);
  proto = 'https';
} else {
  server = http.createServer(handler);
  proto = 'http';
}
server.listen(port, '0.0.0.0', () => {
  console.log(`VSLAM Web server (${proto}) listening on port ${port}`);
  console.log(`  local:   ${proto}://localhost:${port}/`);
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) {
    for (const i of ifaces[name]) {
      if (i.family === 'IPv4' && !i.internal) console.log(`  network: ${proto}://${i.address}:${port}/`);
    }
  }
  if (proto === 'http') {
    console.log('  note: phones need HTTPS for the camera. Create key.pem/cert.pem (see README) to enable HTTPS.');
  }
});
