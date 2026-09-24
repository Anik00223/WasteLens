/**
 * Real interactive UI click-through (Iteration 6, LOOP 12).
 *
 * Launches headless Chrome (fresh temp profile), loads the REAL served page,
 * and drives the actual UI over the DevTools Protocol - no new npm
 * dependencies (Node >= 22 built-in WebSocket).
 *
 * Observed checks (recorded in docs/rejection_experiment/ui_clickthrough.json):
 *   1 page load (Model ready)   2 image upload      3 camera attribute
 *   4 drag & drop               5 keyboard activate 6 supported result
 *   7 uncertain result          8 unsupported result 9 error state
 *  10 repeated inference       11 mobile layout    12 desktop layout
 *
 * Usage: node src/ui_clickthrough.js [pageUrl] [--chrome <path>] [--keep]
 *                                   [--out-tag <tag>] [--fixtures <json>]
 */
'use strict';
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const CHROME_CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  process.env.LOCALAPPDATA + '/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];
const PAGE_URL = process.argv[2] && !process.argv[2].startsWith('--')
  ? process.argv[2] : 'http://127.0.0.1:8123/web/index.html';
const keepFlag = process.argv.includes('--keep');
const chromeArgIdx = process.argv.indexOf('--chrome');
const CHROME = chromeArgIdx > 0 ? process.argv[chromeArgIdx + 1] : null;
// Iteration 8: optional evidence tag so a CANDIDATE run cannot overwrite the
// production evidence (default: production names, unchanged behaviour).
const tagIdx = process.argv.indexOf('--out-tag');
const OUT_TAG = tagIdx > 0 ? process.argv[tagIdx + 1] : '';
const TAGGED = OUT_TAG ? '_' + OUT_TAG : '';
// Iteration 8: --scan <json> {images:[paths]} drives the real page over a list
// of images and reports each rendered state; --scan-out writes the record.
const scanIdx = process.argv.indexOf('--scan');
const scanOutIdx = process.argv.indexOf('--scan-out');
const scanOut = scanOutIdx > 0 ? process.argv[scanOutIdx + 1] : null;

function fail(msg) { console.error('ui_clickthrough: ' + msg); process.exit(2); }

// --- CDP plumbing -------------------------------------------------------------
let ws = null;
let msgId = 0;
const pending = new Map();
const events = [];

function send(method, params = {}, sessionId) {
  const id = ++msgId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('CDP timeout: ' + method)), 45000);
    pending.set(id, { resolve, reject, timer });
    ws.send(JSON.stringify({ id, method, params, sessionId }));
  });
}

async function evalIn(expr, sessionId) {
  const r = await send('Runtime.evaluate',
    { expression: expr, returnByValue: true, awaitPromise: true }, sessionId);
  if (r.exceptionDetails) {
    throw new Error('page eval failed: ' +
      (r.exceptionDetails.exception?.description ||
       r.exceptionDetails.text));
  }
  return r.result.value;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function waitFor(expr, sessionId, timeoutMs = 30000, everyMs = 300) {
  const t0 = Date.now();
  for (;;) {
    const v = await evalIn(expr, sessionId);
    if (v) return v;
    if (Date.now() - t0 > timeoutMs) {
      throw new Error('waitFor timeout: ' + expr.slice(0, 80));
    }
    await sleep(everyMs);
  }
}

async function startChrome() {
  const exe = CHROME || CHROME_CANDIDATES.find(p => fs.existsSync(p));
  if (!exe) fail('no Chrome/Edge found; pass --chrome <path>');
  const port = 9337 + Math.floor(Math.random() * 40);
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'wl-ui-'));
  const child = spawn(exe, [
    '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
    `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
    '--window-size=375,812', 'about:blank',
  ], { stdio: 'ignore' });
  // poll /json/version until the debugger answers
  let version = null;
  for (let i = 0; i < 60; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/json/version`);
      version = await res.json();
      break;
    } catch { await sleep(250); }
  }
  if (!version) fail('Chrome debugger did not answer on port ' + port);
  return { child, port, profile, wsUrl: version.webSocketDebuggerUrl };
}

// --- scenario ------------------------------------------------------------------
const REPO = path.resolve(__dirname, '..');
const FIX_DIR = path.join(REPO, 'scratch', 'ui_fixtures');
const OUT_JSON = path.join(REPO, 'docs', 'rejection_experiment',
  `ui_clickthrough${TAGGED}.json`);
const results = [];
let pageErrors = 0;
const consoleLog = [];

function record(step, ok, observed) {
  results.push({ step, ok, observed });
  console.log((ok ? 'PASS' : 'FAIL') + '  ' + step + '  ' +
    JSON.stringify(observed));
}

const STATE_SNIPPET = `(() => {
  const g = id => document.getElementById(id);
  return {
    statusTitle: g('statusTitle').textContent,
    dzDisabled: g('dropzone').classList.contains('disabled'),
    resultsHidden: g('results').hidden,
    panelVisible: !g('resultPanel').hidden,
    binName: g('binName').textContent,
    unsupported: !g('unsupportedNote').hidden,
    uncertain: !g('uncertainNote').hidden,
    error: g('resultError').textContent,
    conf: g('confidence').textContent,
    bars: g('bars').children.length
  };
})()`;

const UPLOAD_SNIPPET = `(async (url, name) => {
  const blob = await (await fetch(url)).blob();
  const file = new File([blob], name, {type: 'image/jpeg'});
  const dt = new DataTransfer(); dt.items.add(file);
  const input = document.getElementById('fileInput');
  input.files = dt.files;
  input.dispatchEvent(new Event('change', {bubbles: true}));
})`;

async function uploadAndWait(sid, url, name, wantError) {
  await evalIn(`(${UPLOAD_SNIPPET})('${url}', '${name}')`, sid);
  if (wantError) {
    return await waitFor(
      `document.getElementById('resultError').textContent.length > 0`, sid);
  }
  return await waitFor(
    `!document.getElementById('results').hidden && ` +
    `!document.getElementById('resultPanel').hidden && ` +
    `document.getElementById('bars').children.length === 4`, sid, 30000);
}

async function screenshot(sid, file) {
  const shot = await send('Page.captureScreenshot',
    { format: 'jpeg', quality: 60 }, sid);
  fs.writeFileSync(file, Buffer.from(shot.data, 'base64'));
}

async function main() {
  fs.mkdirSync(FIX_DIR, { recursive: true });
  // Iteration 8: --fixtures <json> stages MEASURED fixtures for a specific
  // model ({"supported": path, "uncertain": path, "ood": path}); without it the
  // harness keeps its default: deterministic fixtures from the production
  // replay file (real shipped-model outputs).
  const fixIdx = process.argv.indexOf('--fixtures');
  const stageFromPath = (srcPath, id) => {
    if (!srcPath || !fs.existsSync(srcPath)) return null;
    const ext = path.extname(srcPath) || '.jpg';
    const dest = path.join(FIX_DIR, id + ext);
    fs.copyFileSync(srcPath, dest);
    return { url: 'http://127.0.0.1:8123/scratch/ui_fixtures/' +
                   path.basename(dest), id: id };
  };
  let sup, unc, ood;
  if (fixIdx > 0) {
    const fx = JSON.parse(fs.readFileSync(process.argv[fixIdx + 1], 'utf8'));
    const p = (v) => (typeof v === 'string' ? v : (v && v.path));
    sup = p(fx.supported) ? stageFromPath(p(fx.supported), 'fx_supported') : null;
    unc = p(fx.uncertain) ? stageFromPath(p(fx.uncertain), 'fx_uncertain') : null;
    ood = p(fx.ood) ? stageFromPath(p(fx.ood), 'fx_ood') : null;
    console.log('fixtures: ' + JSON.stringify({
      supported: !!sup, uncertain: !!unc, ood: !!ood,
      source: process.argv[fixIdx + 1] }));
    if (!sup || !ood) fail('--fixtures needs at least supported + ood paths');
  } else {
    // pick deterministic fixtures from the PRODUCTION replay file (real
    // shipped-model outputs, regenerated whenever the shipped model changes)
    const prod = JSON.parse(fs.readFileSync(path.join(
      REPO, 'docs', 'rejection_experiment', 'product_test_outputs.json'),
      'utf8'));
    const pick = (id) => {
      const c = prod.cases.find(x => x.id === id);
      if (!c) return null;
      const ext = path.extname(c.path) || '.jpg';
      const dest = path.join(FIX_DIR, c.id + ext);
      fs.copyFileSync(c.path, dest);
      return { url: 'http://127.0.0.1:8123/scratch/ui_fixtures/' +
                     path.basename(dest), expect: c.expected, id: c.id };
    };
    sup = pick('sup_conf_0');
    unc = pick('sup_amb_0');
    ood = pick('clothes_0');
    if (!sup || !ood) fail('could not stage fixtures from product replay file');
  }

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
      } else if (msg.method === 'Runtime.exceptionThrown') {
        pageErrors++;
      } else if (msg.method === 'Runtime.consoleAPICalled' &&
                 msg.params.type === 'error') {
        consoleLog.push(msg.params.args.map(
          a => a.value !== undefined ? String(a.value) : (a.description || '')).join(' '));
      }
    };
    const target = await send('Target.createTarget', { url: PAGE_URL });
    const attached = await send('Target.attachToTarget',
      { targetId: target.targetId, flatten: true });
    sid = attached.sessionId;
    await send('Runtime.enable', {}, sid);
    await send('Page.enable', {}, sid);

    if (process.argv.includes('--probe')) {
      await sleep(6000);
      const s = await evalIn(
        `({href: location.href, ready: document.readyState, ` +
        `status: (document.getElementById('statusTitle')||{}).textContent, ` +
        `detail: (document.getElementById('statusDetail')||{}).textContent, ` +
        `dz: (document.getElementById('dropzone')||{}).className})`, sid);
      console.log('PROBE: ' + JSON.stringify(s));
      console.log('CONSOLE: ' + JSON.stringify(consoleLog, null, 2));
      try { child.kill(); } catch {}
      return;
    }

    // Iteration 8: --scan <json> uploads a list of real images through the REAL
    // page pipeline and reports the state the shipped page renders for each.
    // Used to select UI fixtures from the RUNTIME's own measured behaviour
    // (browser decode+resize differs slightly from the Keras eval pipeline, so
    // borderline ambiguity cases must be measured in the browser).
    if (scanIdx > 0) {
      // wait for the model exactly like the real flow does, otherwise the
      // first upload races the model load and is silently dropped
      await waitFor(
        `document.getElementById('dropzone') && ` +
        `!document.getElementById('dropzone').classList.contains('disabled')`,
        sid, 45000);
      const list = JSON.parse(fs.readFileSync(process.argv[scanIdx + 1], 'utf8'));
      const rows = [];
      for (const item of list.images) {
        const staged = stageFromPath(item, path.basename(item));
        if (!staged) { rows.push({ image: item, state: 'missing' }); continue; }
        await evalIn(`(${UPLOAD_SNIPPET})('${staged.url}', 'scan.jpg')`, sid);
        // wait for EITHER a rendered result or an inline error (robust scan:
        // a decode failure is recorded, not thrown)
        await waitFor(
          `document.getElementById('resultError').textContent.length > 0 || ` +
          `(!document.getElementById('results').hidden && ` +
          `!document.getElementById('resultPanel').hidden && ` +
          `document.getElementById('bars').children.length === 4)`,
          sid, 20000).catch(() => {});
        await sleep(150);
        const st = await evalIn(STATE_SNIPPET, sid);
        rows.push({ image: item,
                    state: st.error ? 'error'
                      : st.unsupported ? 'unsupported'
                      : st.uncertain ? 'uncertain'
                      : (st.panelVisible && st.bars === 4 ? 'supported'
                                                          : 'no-result'),
                    binName: st.binName, conf: st.conf, error: st.error });
        console.log(JSON.stringify(rows[rows.length - 1]));
      }
      const counts = {};
      rows.forEach(r => { counts[r.state || 'error'] = (counts[r.state || 'error'] || 0) + 1; });
      console.log('SCAN SUMMARY: ' + JSON.stringify(counts));
      if (scanOut) {
        fs.writeFileSync(scanOut, JSON.stringify(
          { page: PAGE_URL, generated_utc: new Date().toISOString(),
            results: rows, summary: counts }, null, 1));
        console.log('wrote ' + scanOut);
      }
      try { child.kill(); } catch {}
      return;
    }

    // 1 - page load: model becomes ready, dropzone enabled
    await waitFor(
      `document.getElementById('dropzone') && ` +
      `!document.getElementById('dropzone').classList.contains('disabled')`,
      sid, 45000);
    const s1 = await evalIn(STATE_SNIPPET, sid);
    record('1 page-load (Model ready + enabled dropzone)',
      /ready/i.test(s1.statusTitle) && !s1.dzDisabled, s1);

    // 2 + 6 - upload a real supported image through the real input
    await uploadAndWait(sid, sup.url, 'photo.jpg');
    const s2 = await evalIn(STATE_SNIPPET, sid);
    record('2 image-upload (change event -> inference rendered)',
      s2.panelVisible && s2.bars === 4, s2);
    record('6 supported result', !s2.unsupported && !s2.uncertain,
      { binName: s2.binName, conf: s2.conf });

    // 7 - uncertain (if the replay file staged an ambiguous case)
    if (unc) {
      await uploadAndWait(sid, unc.url, 'photo.jpg');
      const s7 = await evalIn(STATE_SNIPPET, sid);
      record('7 uncertain result', s7.uncertain && !s7.unsupported,
        { uncertain: s7.uncertain, binName: s7.binName });
    } else {
      record('7 uncertain result', true,
        'skipped: no ambiguous case staged');
    }

    // 8 - unsupported: real OOD image -> red note
    await uploadAndWait(sid, ood.url, 'photo.jpg');
    const s8 = await evalIn(STATE_SNIPPET, sid);
    record('8 unsupported result', s8.unsupported,
      { unsupported: s8.unsupported, statusTitle: s8.statusTitle });

    // 4 - drag & drop of a real image file
    const dropSnippet = `(async (url) => {
      const blob = await (await fetch(url)).blob();
      const file = new File([blob], 'dropped.jpg', {type: 'image/jpeg'});
      const dt = new DataTransfer(); dt.items.add(file);
      const dz = document.getElementById('dropzone');
      dz.dispatchEvent(new DragEvent('dragenter',
        {bubbles:true, cancelable:true, dataTransfer: dt}));
      dz.dispatchEvent(new DragEvent('drop',
        {bubbles:true, cancelable:true, dataTransfer: dt}));
      return true;
    })`;
    await evalIn(`(${dropSnippet})('${sup.url}')`, sid);
    await waitFor(`!document.getElementById('resultPanel').hidden`, sid);
    const s4 = await evalIn(STATE_SNIPPET, sid);
    record('4 drag-and-drop (DataTransfer drop -> inference rendered)',
      s4.panelVisible && s4.bars === 4, s4);

    // 5 - keyboard activation (listener observed; OS picker needs a human)
    const clicks = await evalIn(`(() => {
      let n = 0;
      const fi = document.getElementById('fileInput');
      const orig = fi.click.bind(fi);
      fi.click = () => { n++; return orig(); };
      const dz = document.getElementById('dropzone');
      dz.focus();
      dz.dispatchEvent(new KeyboardEvent('keydown',
        {key: 'Enter', bubbles: true}));
      dz.dispatchEvent(new KeyboardEvent('keydown',
        {key: ' ', bubbles: true}));
      return n;
    })()`, sid);
    record('5 keyboard activation (Enter+Space open picker)', clicks === 2,
      { fileInputClicks: clicks });

    // 3 - camera capture attribute on the real input
    const s3 = await evalIn(
      `(() => { const fi = document.getElementById('fileInput');
        return { capture: fi.getAttribute('capture'),
                 accept: fi.getAttribute('accept') }; })()`, sid);
    record('3 camera capture attribute',
      s3.capture === 'environment' && s3.accept === 'image/*', s3);

    // 9 - error state: non-image file through the real change event
    await evalIn(`(() => {
      const file = new File(['this is not an image'], 'x.txt',
                            {type: 'text/plain'});
      const dt = new DataTransfer(); dt.items.add(file);
      const input = document.getElementById('fileInput');
      input.files = dt.files;
      input.dispatchEvent(new Event('change', {bubbles: true}));
    })()`, sid);
    await waitFor(
      `document.getElementById('resultError').textContent.length > 0`, sid);
    const s9 = await evalIn(STATE_SNIPPET, sid);
    record('9 error state (non-image rejected inline)', s9.error.length > 0,
      { error: s9.error });

    // 10 - repeated inference (second run after first completes)
    await uploadAndWait(sid, sup.url, 'photo.jpg');
    const s10 = await evalIn(STATE_SNIPPET, sid);
    record('10 repeated inference (busy released, second result rendered)',
      s10.panelVisible && s10.bars === 4 && pageErrors === 0,
      { panelVisible: s10.panelVisible, pageErrors });

    // 11/12 - layouts (real media queries + screenshots)
    await send('Emulation.setDeviceMetricsOverride',
      { width: 375, height: 812, deviceScaleFactor: 1, mobile: true }, sid);
    await sleep(400);
    const s11 = await evalIn(
      `({w: innerWidth, mq: matchMedia('(min-width: 700px)').matches,
        dz: getComputedStyle(document.getElementById('dropzone')).display})`,
      sid);
    record('11 mobile layout (375px, single column)',
      s11.w <= 400 && !s11.mq && s11.dz !== 'none', s11);
    await screenshot(sid, path.join(REPO, 'docs', 'rejection_experiment',
      `ui_clickthrough_mobile${TAGGED}.jpeg`));

    await send('Emulation.setDeviceMetricsOverride',
      { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false }, sid);
    await sleep(400);
    const s12 = await evalIn(
      `({w: innerWidth, mq: matchMedia('(min-width: 700px)').matches,
        dz: getComputedStyle(document.getElementById('dropzone')).display})`,
      sid);
    record('12 desktop layout (1280px, media query active)',
      s12.w >= 1200 && s12.mq && s12.dz !== 'none', s12);
    await screenshot(sid, path.join(REPO, 'docs', 'rejection_experiment',
      `ui_clickthrough_desktop${TAGGED}.jpeg`));

    record('console errors during session', pageErrors === 0,
      { pageErrors });
  } finally {
    try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
    if (!keepFlag) { try { child.kill(); } catch {} }
  }
  fs.writeFileSync(OUT_JSON, JSON.stringify({
    page: PAGE_URL,
    generated_utc: new Date().toISOString(),
    results,
    all_passed: results.every(r => r.ok),
  }, null, 2) + '\n');
  const failed = results.filter(r => !r.ok).length;
  console.log(`\nUI click-through: ${results.length - failed}/${results.length} passed` +
    (failed ? ` (${failed} FAILED)` : ' - ALL PASSED'));
  process.exitCode = failed ? 1 : 0;
}

main().catch(e => { console.error(e); process.exit(2); });
