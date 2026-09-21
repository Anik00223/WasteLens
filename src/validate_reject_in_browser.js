const tf = require('../node_modules/@tensorflow/tfjs');
tf.setBackend('cpu');
console.log('backend:', tf.getBackend());

const path = require('path');
const fs = require('fs');
const MODEL_DIR = path.resolve(process.argv[2] || 'scratch/validate_reject_export');

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

async function main() {
  console.log('loading TF.js layers model from', MODEL_DIR);
  const model = await loadLocalLayersModel(MODEL_DIR);
  console.log('model.inputs.length     =', model.inputs.length);
  console.log('model.outputs.length    =', model.outputs.length);
  console.log('output names:', model.outputs.map(o => o.name));

  // Synthetic 1x224x224x3 input in the exact web preprocessing range (x/127.5)-1
  const x = tf.tensor4d(new Float32Array(1*224*224*3), [1,224,224,3]);
  const t0 = performance.now();
  const preds = model.predict(x);
  const dt = performance.now() - t0;
  console.log('predict() returned in', dt.toFixed(1), 'ms');
  console.log('preds.length            =', preds.length);

  const bins = Array.from(preds[0].dataSync());
  const rej  = Array.from(preds[1].dataSync());
  console.log('bins output (4)         =', bins.map(v=>v.toFixed(5)));
  console.log('reject output (1)       =', rej[0].toFixed(6));
  console.log('bins sum                =', bins.reduce((a,b)=>a+b,0).toFixed(6));
  console.log('reject in [0,1]         =', rej[0]>=0 && rej[0]<=1);

  const isValidDist = bins.length===4 && bins.every(v=>v>=0 && v<=1) && Math.abs(bins.reduce((a,b)=>a+b,0)-1)<1e-4;
  console.log('bins is valid dist      =', isValidDist);

  x.dispose(); preds[0].dispose(); preds[1].dispose();
  console.log('OK - browser-runtime multi-output inference proof passed');
}

main().catch(e => { console.error(e); process.exit(1); });
