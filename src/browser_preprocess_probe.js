// src/browser_preprocess_probe.js
//
// WasteLens Iteration 9 - LOOP 2/3: run the SAME source images through the
// browser's REAL preprocessing path and export exactly what the runtime sees,
// so src/preprocess_compare.py can measure decode/resize/normalise/prediction
// differences against the Keras path.
//
// The probe page replicates web/index.html runInference() verbatim:
//   tf.browser.fromPixels(img) -> resizeBilinear([224,224]) -> div(127.5).sub(1)
//   -> model.predict(...)      (production model, /web/model)
// and additionally compares Chrome's canvas decode against the uint8 RGB that
// TensorFlow decodes for the same file (scratch/prep_probe/<name>.py.rgb).
//
// Output: scratch/prep_probe/browser_probe.json
//   [{name,w,h,py_w,py_h,dims_match,decode_diff{max_abs,mean_abs,
//     pct_pixels_differing},tensor_b64,bins,reject,top,margin,verdict}, ...]
//
// Run from repo root (http server must serve the repo root on 8123):
//   node src/browser_preprocess_probe.js [--port 8123] [--chrome <path>]

'use strict';
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const REPO = path.resolve(__dirname, '..');
const PROBE_DIR = path.join(REPO, 'scratch', 'prep_probe');
const OUT = path.join(PROBE_DIR, 'browser_probe.json');
const PORT = process.argv.includes('--port')
  ? process.argv[process.argv.indexOf('--port') + 1] : '8123';
const chromeIdx = process.argv.indexOf('--chrome');
const CHROME_CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  (process.env.LOCALAPPDATA || '') + '/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];

function fail(msg) { console.error('prep_probe: ' + msg); process.exit(2); }

// --- probe page --------------------------------------------------------------
function probeHtml() {
  return `<!doctype html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<title>WasteLens preprocessing probe</title></head><body><pre id="log"></pre>
<script>
const THRESHOLD = 0.0702, MIN_TOP = 0.60, MIN_MARGIN = 0.50;
let model = null, manifest = null;
function log(s){ document.getElementById('log').textContent += s + '\\n'; }
function verdictOf(bins, reject){
  let best = 0, second = 1;
  if (bins[1] > bins[0]) { best = 1; second = 0; }
  for (let i = 2; i < bins.length; i++) {
    if (bins[i] > bins[best]) { second = best; best = i; }
    else if (bins[i] > bins[second]) { second = i; }
  }
  const top = bins[best], margin = top - bins[second];
  if (reject >= THRESHOLD) return {state:'unsupported', top, margin};
  if (top < MIN_TOP - 1e-9 || margin + 1e-9 < MIN_MARGIN)
    return {state:'uncertain', top, margin};
  return {state:'supported', top, margin};
}
function b64FromF32(arr){
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let s = ''; const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH)
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  return btoa(s);
}
async function init(){
  manifest = await (await fetch('/scratch/prep_probe/manifest.json')).json();
  model = await tf.loadLayersModel('/web/model/model.json');
  model.predict(tf.zeros([1,224,224,3]));       // warm-up (same as production)
  window.__ready = true;
  log('ready: ' + manifest.images.length + ' images; tfjs ' + tf.version.tfjs);
}
async function probeOne(name){
  const e = manifest.images.find(m => m.name === name);
  if (!e) return {name: name, error: 'unknown image'};
  const img = new Image();
  img.src = e.image;
  await img.decode();                            // browser JPEG decode
  const W = img.naturalWidth, H = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  ctx.drawImage(img, 0, 0);                      // canvas rasterisation
  const rgba = ctx.getImageData(0, 0, W, H).data;
  const pyrgb = new Uint8Array(
    await (await fetch(e.pyrgb)).arrayBuffer());
  let maxD = 0, sumD = 0, nD = 0, n = 0;
  const lenOk = (pyrgb.length === W * H * 3);
  if (lenOk) {
    for (let i = 0, p = 0; i < rgba.length; i += 4, p += 3) {
      for (let c = 0; c < 3; c++) {
        const d = Math.abs(rgba[i + c] - pyrgb[p + c]);
        if (d > maxD) maxD = d;
        sumD += d; n++; if (d > 0) nD++;
      }
    }
  }
  // --- the production path, verbatim ---
  const t0 = tf.browser.fromPixels(img);
  const resized = t0.resizeBilinear([224, 224]);
  const norm = resized.div(127.5).sub(1);
  const f32 = norm.dataSync();
  const outs = model.predict(norm.expandDims(0));
  const bins = Array.from(await outs[0].data());
  const reject = (await outs[1].data())[0];
  const v = verdictOf(bins, reject);
  const res = {
    name: name, w: W, h: H, py_w: e.pyw, py_h: e.pyh,
    dims_match: (W === e.pyw && H === e.pyh),
    decode_diff: {
      max_abs: maxD,
      mean_abs: n ? (sumD / n) : null,
      pct_pixels_differing: n ? (100 * nD / n) : null,
      n_compared: n, python_bytes_len_ok: lenOk
    },
    tensor_b64: b64FromF32(f32),
    bins: bins.map(x => +x.toFixed(6)),
    reject: +reject.toFixed(6),
    top: +v.top.toFixed(6), margin: +v.margin.toFixed(6),
    verdict: v.state
  };
  [t0, resized, norm, outs[0], outs[1]].forEach(t => t && t.dispose());
  return res;
}
init().catch(e => { window.__error = String((e && e.stack) || e); });
</script></body></html>`;
}

// --- minimal CDP client ------------------------------------------------------
let ws = null, msgId = 0;
const pending = new Map();

function send(method, params = {}, sessionId) {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('CDP timeout: ' + method)),
                             60000);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
}

function findChrome() {
  if (chromeIdx > 0 && fs.existsSync(process.argv[chromeIdx + 1])) {
    return process.argv[chromeIdx + 1];
  }
  return CHROME_CANDIDATES.find(p => fs.existsSync(p)) || null;
}

async function startChrome() {
  const exe = findChrome();
  if (!exe) fail('no Chrome/Edge found; pass --chrome <path>');
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'wl_prep_'));
  const child = spawn(exe, [
    '--headless=new', '--remote-debugging-port=0',
    '--user-data-dir=' + profile, '--no-first-run', '--disable-gpu',
    '--no-sandbox', 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
  const wsUrl = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Chrome did not start')),
                             45000);
    child.stderr.on('data', (d) => {
      const m = String(d).match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) { clearTimeout(timer); resolve(m[1]); }
    });
    child.on('exit', (code) => reject(new Error('chrome exited ' + code)));
  });
  return { child, profile, wsUrl };
}

async function main() {
  if (!fs.existsSync(path.join(PROBE_DIR, 'manifest.json'))) {
    fail('scratch/prep_probe/manifest.json missing - run '
         + 'src/preprocess_compare.py dump-inputs first');
  }
  const manifest = JSON.parse(
    fs.readFileSync(path.join(PROBE_DIR, 'manifest.json'), 'utf8'));
  fs.writeFileSync(path.join(PROBE_DIR, 'index.html'), probeHtml());

  const pageUrl = `http://127.0.0.1:${PORT}/scratch/prep_probe/index.html`;
  const smoke = await fetch(
    `http://127.0.0.1:${PORT}/scratch/prep_probe/manifest.json`)
    .then(r => r.ok).catch(() => false);
  if (!smoke) fail(`http server not serving the repo on port ${PORT}`);

  const { child, profile, wsUrl } = await startChrome();
  let sid = null;
  try {
    ws = new WebSocket(wsUrl);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
    ws.onmessage = (m) => {
      const msg = JSON.parse(m.data);
      if (msg.id && pending.has(msg.id)) {
        const p = pending.get(msg.id);
        pending.delete(msg.id);
        clearTimeout(p.timer);
        if (msg.error) p.reject(new Error(msg.error.message));
        else p.resolve(msg.result);
      }
    };
    const target = await send('Target.createTarget', { url: pageUrl });
    const attached = await send('Target.attachToTarget',
      { targetId: target.targetId, flatten: true });
    sid = attached.sessionId;
    await send('Runtime.enable', {}, sid);

    // wait for the probe page to finish init() (model + tfjs loaded)
    let ready = false;
    for (let i = 0; i < 120 && !ready; i++) {
      const r = await send('Runtime.evaluate', {
        expression: 'window.__error ? ("ERR:" + window.__error) : '
                    + '(window.__ready === true ? "READY" : "wait")',
        returnByValue: true }, sid);
      const v = r.result.value;
      if (v && v.startsWith('ERR:')) fail(v.slice(4));
      if (v === 'READY') ready = true;
      else await new Promise(res => setTimeout(res, 500));
    }
    if (!ready) fail('probe page never became ready');

    const results = [];
    for (const entry of manifest.images) {
      const r = await send('Runtime.evaluate', {
        expression: 'probeOne(' + JSON.stringify(entry.name) + ')',
        awaitPromise: true, returnByValue: true }, sid);
      if (r.exceptionDetails) {
        results.push({ name: entry.name,
                       error: r.exceptionDetails.exception?.description
                              || r.exceptionDetails.text });
        console.log('ERROR ' + entry.name);
        continue;
      }
      const rec = r.result.value;
      results.push(rec);
      console.log(`${rec.name}: ${rec.w}x${rec.h} ` +
        `decode max=${rec.decode_diff.max_abs} ` +
        `mean=${rec.decode_diff.mean_abs?.toFixed(3)} ` +
        `diff%=${rec.decode_diff.pct_pixels_differing?.toFixed(2)} | ` +
        `top=${rec.top} margin=${rec.margin} reject=${rec.reject} ` +
        `-> ${rec.verdict}`);
    }

    const out = {
      generated_utc: new Date().toISOString(),
      page: pageUrl,
      tfjs_note: 'production path verbatim: fromPixels -> resizeBilinear '
                 + '(224,224) -> div(127.5).sub(1) -> model.predict',
      threshold: 0.0702,
      results,
    };
    fs.writeFileSync(OUT, JSON.stringify(out));
    console.log(`wrote ${OUT} (${results.length} images)`);
  } finally {
    try { child.kill(); } catch {}
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
  }
}

if (require.main === module) {
  main().catch(e => { console.error('prep_probe: ' + e); process.exit(1); });
}
module.exports = { probeHtml };
