const tf = require('../node_modules/@tensorflow/tfjs');
tf.setBackend('cpu');
console.log('backend:', tf.getBackend());

const path = require('path');
const fs = require('fs');

// CLI: node src/validate_reject_in_browser.js <modelDirOrURL> [--parity <dir>] [--perf]
const ARGS = process.argv.slice(2);
const FLAGS = {};
for (let i = 0; i < ARGS.length; i++) {
  if (ARGS[i] === '--parity') FLAGS.parity = ARGS[++i];
  else if (ARGS[i] === '--perf') FLAGS.perf = true;
}
const MODEL_REF = ARGS[0] || 'scratch/validate_reject_export';
const IS_URL = /^https?:\/\//.test(MODEL_REF);
const PARITY_ATOL = 2e-5; // measured baseline diff ~5e-6; 4x headroom (float32)

// Pure-JS tfjs cannot fetch file:// URLs in modern Node (undici has no file
// scheme handler), so load the model.json + weight shards manually and hand
// tfjs a complete ModelArtifacts object. Shard paths appear in weightsManifest
// order and are contiguous, so Buffer.concat is a valid weightData buffer.
function loadLocalLayersModel(modelDir) {
  const j = JSON.parse(fs.readFileSync(path.join(modelDir, 'model.json'), 'utf8'));
  const specs = [];
  const shards = [];
  for (const entry of j.weightsManifest) {
    for (const w of entry.weights) specs.push(w);
    for (const p of entry.paths) shards.push(fs.readFileSync(path.join(modelDir, p)));
  }
  console.log('model.json: format=%s  weight shards=%d  total bytes=%d',
      j.format, shards.length, shards.reduce((a, b) => a + b.length, 0));
  return tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: j.modelTopology,
    format: j.format,
    generatedBy: j.generatedBy,
    convertedBy: j.convertedBy,
    weightSpecs: specs,
    weightData: Buffer.concat(shards),
  }));
}

async function loadModel(modelRef) {
  if (IS_URL) return tf.loadLayersModel(modelRef);
  return loadLocalLayersModel(path.resolve(modelRef));
}

function structuralChecks(model) {
  let failed = 0;
  const gate = (name, ok, extra) => {
    console.log((ok ? 'PASS ' : 'FAIL ') + name + (extra ? '  ' + extra : ''));
    if (!ok) failed++;
  };
  gate('inputs == 1', model.inputs.length === 1);
  gate('outputs == 2 (dual-head)', model.outputs.length === 2,
       'names: ' + JSON.stringify(model.outputs.map(o => o.name)));
  gate('bins output is 4-unit',
       model.outputs[0].shape[model.outputs[0].shape.length - 1] === 4);
  gate('reject output is 1-unit',
       model.outputs[1].shape[model.outputs[1].shape.length - 1] === 1);

  const x = tf.tensor4d(new Float32Array(1 * 224 * 224 * 3), [1, 224, 224, 3]);
  const t0 = performance.now();
  const preds = model.predict(x);
  const dt = performance.now() - t0;
  const bins = Array.from(preds[0].dataSync());
  const rej = Array.from(preds[1].dataSync());
  console.log('predict() returned in', dt.toFixed(1), 'ms');
  console.log('bins output (4)   =', bins.map(v => v.toFixed(5)));
  console.log('reject output (1) =', rej[0].toFixed(6));
  gate('bins is valid 4-probability distribution',
       bins.length === 4 && bins.every(v => v >= 0 && v <= 1) &&
       Math.abs(bins.reduce((a, b) => a + b, 0) - 1) < 1e-4);
  gate('reject in [0,1]', rej[0] >= 0 && rej[0] <= 1);
  x.dispose(); preds[0].dispose(); preds[1].dispose();
  return failed;
}

async function runParity(model, parityDir) {
  const payload = JSON.parse(fs.readFileSync(path.join(parityDir, 'parity.json'), 'utf8'));
  const raw = fs.readFileSync(path.join(parityDir, 'inputs.f32'));
  const n = payload.n;
  if (raw.length !== n * 224 * 224 * 3 * 4) {
    throw new Error('inputs.f32 size mismatch: ' + raw.length + ' bytes for n=' + n);
  }
  const f32 = new Float32Array(raw.buffer, raw.byteOffset, n * 224 * 224 * 3);
  const kb = payload.keras.bins;
  const kr = payload.keras.reject;
  let maxDb = 0, maxDr = 0;
  for (let i = 0; i < n; i++) {
    const x = tf.tensor4d(f32.slice(i * 224 * 224 * 3, (i + 1) * 224 * 224 * 3),
                          [1, 224, 224, 3]);
    const preds = model.predict(x);
    const b = Array.from(preds[0].dataSync());
    const r = preds[1].dataSync()[0];
    for (let j = 0; j < 4; j++) maxDb = Math.max(maxDb, Math.abs(b[j] - kb[i][j]));
    maxDr = Math.max(maxDr, Math.abs(r - kr[i]));
    console.log('  [%2d] %-34s maxdbins=%.2e  dreject=%.2e',
                i, payload.input_names[i], maxDb, maxDr);
    x.dispose(); preds[0].dispose(); preds[1].dispose();
  }
  console.log('PARITY max |dbins| = %.3e   max |dreject| = %.3e   (atol %.0e)',
              maxDb, maxDr, PARITY_ATOL);
  const ok = maxDb <= PARITY_ATOL && maxDr <= PARITY_ATOL;
  console.log((ok ? 'PASS' : 'FAIL') + ' python-tfjs parity within atol');
  return ok ? 0 : 1;
}

async function runPerf(model) {
  const x = tf.tensor4d(new Float32Array(1 * 224 * 224 * 3), [1, 224, 224, 3]);
  const t1 = performance.now();
  const p = model.predict(x);
  await p[0].data(); await p[1].data();
  const first = performance.now() - t1;
  p[0].dispose(); p[1].dispose();
  let warm = 0;
  const runs = 10;
  for (let i = 0; i < runs; i++) {
    const t = performance.now();
    const q = model.predict(x);
    await q[0].data(); await q[1].data();
    q[0].dispose(); q[1].dispose();
    warm += performance.now() - t;
  }
  x.dispose();
  let bytes = 0;
  if (!IS_URL) {
    for (const f of fs.readdirSync(MODEL_REF)) {
      bytes += fs.statSync(path.join(MODEL_REF, f)).size;
    }
  }
  console.log(JSON.stringify({
    model_ref: MODEL_REF, first_predict_ms: Math.round(first),
    warm_avg_ms: Math.round(warm / runs), warm_runs: runs,
    total_artifact_bytes: IS_URL ? null : bytes}));
  return 0;
}

async function main() {
  console.log('loading TF.js layers model from', MODEL_REF);
  const model = await loadModel(MODEL_REF);
  let failed = structuralChecks(model);
  if (FLAGS.parity) failed += await runParity(model, FLAGS.parity);
  if (FLAGS.perf) failed += await runPerf(model);
  if (failed === 0) {
    console.log('OK - browser-runtime multi-output validation passed');
  } else {
    console.log('FAILED - ' + failed + ' check(s) failed');
  }
  process.exitCode = failed > 0 ? 1 : 0;
}

main().catch(e => { console.error(e); process.exit(1); });
