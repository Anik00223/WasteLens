# §13 Browser-runtime export proof (Variant A dual-head model)

**Goal:** prove the WasteLens TF.js export pipeline can carry a *multi-output*
model (4-bin classification + 1-dim rejection) and that it runs and predicts
correctly in a JS runtime (`@tensorflow/tfjs` layers backend, CPU).

**Artifacts (this proof):**
- Checkpoint: `models/checkpoints/wastelens_rej_best.keras` (Variant A best-by-val_loss)
- Export dir: `scratch/validate_reject_export/` (regenerable:
  `py -3.13 -W ignore src/export_tfjs.py --model models/checkpoints/wastelens_rej_best.keras --out scratch/validate_reject_export`)
- Harness: `src/validate_reject_in_browser.js`
  (run from repo root: `node src/validate_reject_in_browser.js`)

## Export shape

- `model.json`: format `layers-model`, 8 layers, named outputs
  `bins -> predictions(4)` / `reject -> reject(1)`
- 3 shards: 4,194,304 + 4,194,304 + 1,960,212 bytes = 10,348,820 B
- `labels.json`: `recyclable / organic / hazardous / general trash`

## Exporter fix shipped with this proof

Keras 3 serializes **named (dict) outputs** as a dict in
`config.output_layers` (`{"bins": [...], "reject": [...]}`), while
tfjs-layers containers only accept a flat list of `[name, nodeIndex,
tensorIndex]` triplets. Single-output models always got a list, so the
Iteration-3 export path was validated - the dual-head export tripped
`TypeError: Object is not iterable` in `Container.fromConfig`.
`src/export_tfjs.py::_wrap_io_triplet` now converts the dict form to a
triplet list in insertion order (Python dicts keep insertion order and
`json.dumps` preserves it, so the output order is the model's output order).

## Harness notes

Pure-JS `@tensorflow/tfjs` cannot fetch `file://` URLs in modern Node
(undici has no file-scheme handler), so the harness loads `model.json` +
shards with `fs` and hands tfjs a complete `ModelArtifacts` via
`tf.io.fromMemory` (shards are contiguous in manifest order, so
`Buffer.concat` in path order is a valid `weightData` buffer).

## Runtime output (Node 22, `@tensorflow/tfjs` CPU backend)

```
backend: cpu
model.inputs.length     = 1
model.outputs.length    = 2
output names: [ 'predictions/predictions', 'reject/reject' ]
predict() returned in 3288.6 ms        <- cold; 769.5 ms warm
bins output (4)         = [ '0.99997', '0.00000', '0.00003', '0.00000' ]
reject output (1)       = 0.684148
bins sum                = 1.000000
reject in [0,1]         = true
bins is valid dist      = true
OK - browser-runtime multi-output inference proof passed
```

Input: all-zeros 1x224x224x3 (the exact web preprocessing range is (x/127.5)-1;
zeros map to -1, the lower bound).

## Parity vs Keras (same zeros input)

| head    | Keras                      | TF.js CPU                  | max abs diff |
|---------|----------------------------|----------------------------|--------------|
| bins    | [0.99997, 0, 3e-05, 0]     | [0.99997, 0, 3e-05, 0]     | 5.0e-06      |
| reject  | 0.684152                   | 0.684148                   | 3.8e-06      |

Differences are float32 noise - the web pipeline preserves the model's
numerics. The multi-output export contract is proven end-to-end in a JS
runtime, de-risking a future Outcome A ship of the frozen dual-head model.
