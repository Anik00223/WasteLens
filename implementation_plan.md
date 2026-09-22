# Implementation Plan — Iteration 5: Integrate and Ship the Verified Frozen Rejection Model

## Overview

Wire the approved Variant B dual-head checkpoint (`models/checkpoints/wastelens_rej_frozen_best.keras`) into the actual browser app in `web/index.html`, add a third **unsupported** result state driven by the calibrated rejection head, extend the browser/validation harnesses, and ship via a single reviewed commit to `main`. No product redesign, no retraining, no label-order change.

**Baseline (verified 2026-09-21):** `main ≡ origin/main ≡ f20c2d6`, working tree clean. `web/model/` currently holds the OLD **single-head** export (model.json 213,970 B, 7 layers, one `predictions(4)` output); the Variant B dual-head export (model.json 215,184 B, 8 layers, named outputs `bins`/`reject`) has only ever been produced into `scratch/validate_reject_export/` from the **Variant A** checkpoint. Exporting Variant B into `web/model/` is therefore a required first step.

**Verified facts this plan relies on (not assumptions):**

1. **Reject-output semantics** — `src/train_rejection.py`, `assemble()` (line ~354): `y_rej = [0] * n_s + [1] * n_u`. The `reject` head outputs **P(unsupported)**: higher = more likely outside scope. The web must treat `reject >= THRESHOLD` as unsupported.
2. **Calibrated threshold** — `docs/rejection_experiment/rejection_metrics_frozen.json`, operating point `"0.01"`: **reject ≥ 0.8774** (val-frozen 1% FRR point), measured on test: **84.26% OOD detection @ 1.38% FRR**. This is the threshold `docs/rejection_experiment/outcome_decision.md` ("What ships") fixes for shipping. The 5% FRR point (0.5432; 96.08% @ 6.65%) is documented but NOT shipped.
3. **Rejection priority over uncertain** — correct per the experiment: `eval_rejection.py` defines the operating point purely on the rejection head (FRR measured on supported test only, detection on unsupported only); the top/margin rule is only ever evaluated on non-rejected inputs. Decision layer order: unsupported (rejection first) → uncertain (existing rule) → supported.
4. **Existing ambiguity rule** (kept independent) — `web/index.html`: `UNCERTAIN = { MIN_TOP: 0.60, MIN_MARGIN: 0.50 }` inside pure `classify(probs)`.
5. **tfjs version parity** — CDN `@tensorflow/tfjs@4.22.0` == installed `node_modules/@tensorflow/tfjs` 4.22.0. No dependency work needed.
6. **Preprocessing contract** — unchanged: resizeBilinear 224×224 → `(x / 127.5) - 1`; `LABELS = ['recyclable','organic','hazardous','general trash']` matches `train.BINS` and `labels.json`.
7. **Known weak spot** — collages: 57.8% detection @5% FRR (Variant B, documented). The unsupported state must not claim perfect OOD detection; the existing scope disclosure stays.
8. **User decisions (this session):** the "49-check validator" is an external checklist — all existing in-repo validators must keep passing and the full check list is enumerated in the iteration report; §13 uses NO new dependencies (HTTP-serve + Node tfjs harness against the served URL + a manual interactive checklist).

## Types

No formal type system (vanilla JS + Python). Introduced data shapes:

- **Verdict state (JS, in `web/index.html`):** string enum `'supported' | 'uncertain' | 'unsupported'`.
- **`decideVerdict(probs, rejectProb)` return (JS object):** `{ state: 'supported'|'uncertain'|'unsupported', best: int, second: int, top: float, margin: float, reject: float }` — pure, DOM-free, so it can be extracted and unit-tested.
- **`REJECT_THRESHOLD = 0.8774`** (JS const inside the marked decision block, next to `UNCERTAIN`), with a comment citing `rejection_metrics_frozen.json` op point `"0.01"`.
- **Parity artifact set (files):** `inputs.f32` — raw little-endian float32, C-order, `N × 1 × 224 × 224 × 3`, values ALREADY preprocessed to `[-1,1]` (exact model input); `parity.json` — `{model, created_utc, n, input_shape: [1,224,224,3], keras: {bins: number[N][4], reject: number[N][1]}}`.
- **Product-test artifact (JSON):** `{model, created_utc, reject_threshold, cases: [{id, group, path, ground_truth, expected: 'supported'|'uncertain'|'unsupported'|null, bins: number[4], reject: number}]}`.
- **Perf report (harness stdout JSON):** `{model_ref, load_ms, first_predict_ms, warm_avg_ms, warm_runs, model_json_bytes, shard_bytes}`.

## Files

**New files:**
- `src/make_parity_inputs.py` — builds deterministic parity inputs (synthetic + optional real from the dataset cache), runs Keras on them, writes `inputs.f32` + `parity.json` (schema in Types). CLI: `--model` (default `models/checkpoints/wastelens_rej_frozen_best.keras`), `--out` (default `scratch/parity_frozen`), `--real-per-group N` (default 8).
- `src/product_test_outputs.py` — deterministic selection of product-test samples (Testing §C) from `docs/rejection_experiment/eval_sets.json` + the `%TEMP%\wastelens_rejection_data` cache; runs the Variant B Keras model; writes `docs/rejection_experiment/product_test_outputs.json`. CLI: `--model`, `--seed 42`.
- `src/test_decision_logic.js` — Node test runner; extracts the marked decision block from `web/index.html`, unit-tests `decideVerdict` boundaries, then replays every case in `product_test_outputs.json` through `decideVerdict` and asserts states. Exits non-zero on failure.
- `docs/rejection_experiment/shipping_report.md` — Iteration 5 report artifact: what shipped, decision logic, measured numbers (parity, perf before/after, OOD product tests), full enumerated check list, remaining limitations.
- `scratch/perf_baseline_singlehead/` (NOT committed) — copy of the current `web/model/` taken **before** overwriting, for the performance "before" measurement.

**Modified files:**
- `web/model/model.json`, `web/model/group1-shard{1,2,3}of3.bin` — replaced by the Variant B dual-head export (git tracks these binaries; ~10.3 MB total, same ballpark as today). `web/model/labels.json` — byte-identical content re-written by the exporter (no diff expected).
- `web/index.html` — core integration (see Functions). CSS: add `.chip.unsupported` and `.unsupported-note` using the existing `--danger` palette; keep `.chip.uncertain` (warning) visually distinct. Copy updates: ready-status message, footer, and the stale SCOPE comment above `UNCERTAIN` (the claim "NO softmax threshold can detect OOD" is superseded by the model-side rejection head; the margin rule now covers ambiguity only).
- `src/validate_reject_in_browser.js` — model-ref may be a local dir **or an `http(s)://` URL** (tf.loadLayersModel fetches; shards resolve relative to the URL); `--parity <dir>` mode (Python-vs-TF.js parity, tolerance atol 2e-5, justified below); `--perf` mode (load/first/warm timing + artifact sizes); existing structural checks become hard gates (exit non-zero).
- `src/validate_realworld.py` — add `--model` (default unchanged `models/checkpoints/wastelens_ep09.keras`) and `--tag` (output suffix); dual-head-aware prediction (Keras dict outputs: take `out["bins"]`, capture `out["reject"]` when present); report gains a reject column and an OOD-probe rejection section when a dual-head model is evaluated; writes `docs/realworld_validation{tag}.md/json`.
- `models/tfjs_model/` — archive copy of the new export (model.json + shards + labels.json; currently holds only `labels.json` + `.gitkeep`).
- `README.md` — Status section: Variant B shipped, dual-head architecture, calibrated threshold, measured OOD performance, collage limitation, `uncertain` vs `unsupported` distinction.
- `docs/report.md` — append "## 6. Shipped rejection model (Iteration 5)" (file header says "edit freely"; do not touch computed sections 3/4).
- `docs/rejection_experiment/outcome_decision.md` — "What ships" updated: `web/model/` now carries the dual-head export; threshold 0.8774 active in the UI.

**Deleted/moved:** none. **Config:** none (thresholds live in `web/index.html`).

## Functions

**`web/index.html` (all changes in the existing IIFE):**
- `runInference(img)` — MODIFY: return `model.predict(...)` as-is; it is now a **2-element Tensor array** (`[binsTensor, rejectTensor]`). Still inside `tf.tidy`; tidy preserves returned tensor containers.
- `handleFile(file)` — MODIFY: replace the single `probsTensor.data()` with `Promise.all([outs[0].data(), outs[1].data()])`, dispose both tensors, call `renderResult(Array.from(binData), rejectData[0])`.
- `classify(probs)` — KEEP unchanged (pure argmax/margin logic), but move it — with `UNCERTAIN`, new `REJECT_THRESHOLD`, and `decideVerdict` — between marker comments: `// === DECISION-BEGIN (pure logic; extracted verbatim by src/test_decision_logic.js) ===` and `// === DECISION-END ===`.
- `decideVerdict(probs, rejectProb)` — NEW (inside the markers): rejection gate first (`rejectProb >= REJECT_THRESHOLD` → `'unsupported'`), else existing `classify()` uncertainty → `'uncertain'`/`'supported'`. Returns the verdict object from Types.
- `renderResult(probs, rejectProb)` — MODIFY: `var verdict = decideVerdict(probs, rejectProb)`; branch on `verdict.state`:
  - `supported`: exactly today's render (accent chip, predicted-highlight bars).
  - `uncertain`: exactly today's render (warning chip + note).
  - `unsupported`: bin-label text → "Result"; chip text "Unsupported" with class `unsupported` (danger); confidence number = `(verdict.reject*100).toFixed(1)+'%'` with caption "rejection confidence"; show `els.unsupportedNote`; render all four bars **without** the predicted highlight (muted); hide `uncertainNote`; keep `scope-note` visible.
  - Copy for `unsupportedNote` (no overclaiming): "This image appears to be outside WasteLens's supported single-item waste classification scope. Non-waste objects, cluttered or multi-object scenes, and unsupported item types may be rejected. Rejection is not perfect — some unsupported images may still be classified, and occasional supported photos may be flagged. Photograph one item at a time on a plain background for best results."
- `init()` — MODIFY ready-status detail: "MobileNetV2 + 4-bin head + rejection head loaded from web/model/ · all inference runs locally."
- `els` map — ADD `unsupportedNote`.
- Footer — ADD: "images that appear outside the supported single-item scope are flagged unsupported (imperfect; ~84% of unsupported inputs caught at ~1.4% false rejections in the validation experiment)".
- Keep untouched: file-type guard, loading/error handling, camera `capture` attribute on `fileInput`, drag/drop, keyboard activation, `hidden` result flow.

**`src/validate_reject_in_browser.js`:**
- `loadLocalLayersModel(dir)` — KEEP (fs + `tf.io.fromMemory`); used for local dirs.
- `main()` — MODIFY: `argv[2]` may be a dir or URL; dispatch on `--parity`/`--perf`; structural checks (backend, inputs=1, outputs=2, named `predictions/predictions` + `reject/reject`, bins valid dist, reject in [0,1]) exit non-zero on failure.
- NEW `runParity(model, parityDir)` — read `inputs.f32` + `parity.json`; slice into `tf.tensor4d` inputs; predict both heads; per-input and max abs diffs for bins and reject; PASS iff max diffs ≤ 2e-5 (justification: measured parity on identical input was 5.0e-6 bins / 3.8e-6 reject; 2e-5 leaves ~4× headroom for float32 accumulation-order differences between oneDNN (Keras CPU) and the XLA/Eigen TF.js CPU backend).
- NEW `runPerf(model, ref)` — load ms, first predict ms, warm avg over 10 runs, artifact byte sizes; prints one JSON line.

**`src/validate_realworld.py`:**
- Prediction call site — dual-head-aware: `out = model.predict(...)`; `probs = out["bins"] if isinstance(out, dict) else out`; `rej = out["reject"].ravel() if isinstance(out, dict) else None`.
- `render_md(payload)` — ADD reject column + per-probe rejection rows when `rej` is present; when a dual-head model is evaluated, add a "model-side rejection" section for the 5 synthetic probes (report actual counts — never fabricate) while the margin-rule analysis stays for continuity.
- `main()` — ADD argparse `--model`, `--tag`.

**`src/make_parity_inputs.py`** — NEW `main()` + helpers: `build_synthetic_inputs()` (5 deterministic images: zeros, 16px checkerboard, horizontal gradient, seeded gaussian noise, flat mid-gray; built directly in `[-1,1]` model-input space), `select_real_inputs(data, n)` (sorted-path stride sampling from `eval_sets.json` `supported_test` and `val_unsup`), `keras_predict(model, paths)` (reuse `tr.make_weighted_ds(...)` images-only mapping exactly as `src/eval_rejection.py::batches` does, guaranteeing identical preprocessing), `write_artifacts(out_dir, inputs, outputs)`.

**`src/product_test_outputs.py`** — NEW `main()` + `select_cases(data, seed)` (rules in Testing §C) + `write_json(out, payload)`.

**`src/test_decision_logic.js`** — NEW `extractDecisionBlock()` (regex between markers + `new Function`), `runUnitCases()`, `runProductCases()`, `main()` with PASS/FAIL table and exit code.

**`src/export_tfjs.py`** — NO code changes (already supports multi-output; `--out` is parameterized).

## Classes

None — the repo has no classes in `web/` or `src/` scripts; this plan introduces none (plain functions + constants only, matching existing conventions).

## Dependencies

- **None new.** Node harness uses `node_modules/@tensorflow/tfjs` 4.22.0 (installed; matches the CDN pin 4.22.0 in `web/index.html`). Python side uses installed TF 2.21 / numpy / scikit-learn. `package.json` already pins `@tensorflow/tfjs-node ^4.22.0` (its tfjs transitive dep is what the harness requires; the native binding remains unusable on this machine — documented in Iteration 4 — so the pure-JS backend stays).
- Optional (no-op unless it installs cleanly): pin `"@tensorflow/tfjs": "4.22.0"` explicitly in `package.json` for clarity. NEVER commit `node_modules/`.

## Testing

**A. Export + structural gates (hard gates, exit non-zero on failure)**
1. Export Variant B → `web/model/` and archive → `models/tfjs_model/`; assert `format: layers-model`, 2 outputs (`predictions/predictions`, `reject/reject`), 3 shards, `labels.json` == BINS order, and no unintended `web/index.html` diff from the export itself.
2. Harness structural run: `node src/validate_reject_in_browser.js web/model` — all Iteration-4 checks pass on the SHIPPED artifact.

**B. Python ↔ TF.js parity** — `py -3.13 -W ignore src/make_parity_inputs.py` then `node src/validate_reject_in_browser.js web/model --parity scratch/parity_frozen`. Inputs: 5 synthetic + up to 16 real (8 supported-test + 8 val-unsup when the `%TEMP%` cache exists; the script skips real inputs with a clear message otherwise). Report per-input and max abs diffs for bins and reject; tolerance atol 2e-5.

**C. Decision-layer tests** — `node src/test_decision_logic.js`:
- Unit boundaries: `reject=0.8774` → unsupported; `reject=0.87739` with dominant bins → supported; `top=0.55` → uncertain; `top=0.9, margin=0.3` → uncertain; ambiguous bins + `reject=0.95` → unsupported (priority check); exact `MIN_TOP/MIN_MARGIN` boundaries ±1e-9.
- Product cases (real Variant B Keras outputs, deterministic selection, seed 42):
  - 8 supported-confident (2 per bin; argmax correct AND top ≥ 0.95) → expected `supported`.
  - up to 3 supported-ambiguous (top < 0.6 or margin < 0.5; `trash223.jpg` is a known case: top 67.4%, margin 43.7%) → expected `uncertain`.
  - 3 clothes + 3 shoes + 3 non-waste (val-unsup, highest reject scores — all three sources ≥ 94% mean detection in Iteration 4) → expected `unsupported`.
  - 5 collages → `expected: null` — assert only decision-layer consistency and LOG the state; ground truth is documented weak (57.8% @5% FRR). Never assert perfect collage rejection.
  - Ground truth per case is recorded in the JSON; high-margin unsupported cases must match their ground truth.

**D. Regression (existing validators must keep passing)**
1. `py -3.13 -W ignore src/validate_realworld.py` (default ep09) — numbers must match the existing `docs/realworld_validation.md` (only the timestamp changes).
2. `py -3.13 -W ignore src/validate_realworld.py --model models/checkpoints/wastelens_rej_frozen_best.keras --tag frozen` → `docs/realworld_validation_frozen.md`: 32 in-distribution FRR ≈ 1/32 (consistent with 1.38%); synthetic OOD probes now caught by the rejection head (report actual counts).
3. `py -3.13 -m py_compile src/*.py`; single-output exporter regression re-run (`wastelens_ep09.keras` → `scratch/regress_single_output`, `output_layers` must remain a flat list).
4. Existing rejection eval artifacts (`rejection_metrics_frozen.json`, `rejection_report_frozen.md`) stay the source of truth — no retraining, no re-eval.

**E. Performance** — harness `--perf` on `scratch/perf_baseline_singlehead` (BEFORE — captured in step 1, prior to overwriting `web/model/`) and on `web/model` (AFTER); report load/first/warm + artifact sizes. Only investigate material regressions (per brief).

**F. HTTP + manual UI (no new dependencies, per user decision)**
- `py -3.13 -m http.server 8000 --directory web` (background); `GET http://localhost:8000/index.html` → 200; run the harness against `http://localhost:8000/model/model.json` (same tfjs loading path as the browser) with parity inputs.
- Manual interactive checklist (recorded in `shipping_report.md` for the user to run): model-ready status, file input, camera capture, drag/drop, keyboard (Tab/Enter/Space), supported state, uncertain state, unsupported state (e.g., a photo of a person/room), error state (non-image file), desktop + mobile layout, footer disclosure.

**Full check list enumerated in `shipping_report.md`** (the user's "49-check" list is an external artifact — this report reproduces it): 2 export checks, 5 structural harness checks, 2 parity gates × (5 synthetic + up to 16 real) inputs, ~8 decision unit cases, 19–22 product cases, 2 realworld-validation runs, 1 exporter regression, 2 perf runs, 2 HTTP checks, ~12 manual UI checks, 3 git-state checks.

## Implementation Order

1. **Preflight:** confirm `git status` clean at `f20c2d6`; `py -3.13 -m py_compile src/*.py`; copy `web/model/` → `scratch/perf_baseline_singlehead/`; run harness `--perf` on the single-head baseline (the "before" number) BEFORE overwriting `web/model/`.
2. **Export Variant B:** `py -3.13 -W ignore src/export_tfjs.py --model models/checkpoints/wastelens_rej_frozen_best.keras --out web/model`; copy artifacts to `models/tfjs_model/`; verify topology (2 named outputs) and labels.
3. **Structural harness gate:** `node src/validate_reject_in_browser.js web/model` — all checks green on the shipped artifact.
4. **`web/index.html` integration:** markers + `REJECT_THRESHOLD` + `decideVerdict`; dual-output read in `handleFile`; 3-state `renderResult` + unsupported CSS/copy; status/footer/comment updates. (Largest change; done before parity so served-page tests exercise final code.)
5. **Parity:** write `src/make_parity_inputs.py`, run it; add `--parity` to the harness; run locally; record max diffs.
6. **Decision tests:** write `src/product_test_outputs.py` (run against the Keras frozen checkpoint), `src/test_decision_logic.js`; run; investigate any expectation mismatch (never silently relax ground-truth assertions).
7. **Real-world validator:** dual-head-aware `src/validate_realworld.py` (`--model`/`--tag`); run baseline (must match prior numbers) and frozen runs.
8. **Performance "after":** harness `--perf` on `web/model`; assemble the before/after table.
9. **Docs:** README status, `docs/report.md` §6, `outcome_decision.md` "What ships".
10. **HTTP verification:** serve `web/`, GET 200, harness against the served URL with parity; capture outputs.
11. **Write `docs/rejection_experiment/shipping_report.md`** with the full enumerated check list and measured numbers.
12. **Git (mandatory):** `git status` + `git diff` review (expected changes ONLY: `web/index.html`, `web/model/*`, `models/tfjs_model/*`, `README.md`, `docs/report.md`, `docs/rejection_experiment/{outcome_decision.md, product_test_outputs.json, shipping_report.md, realworld_validation_frozen.md, realworld_validation_frozen.json}`, `src/{validate_reject_in_browser.js, validate_realworld.py, make_parity_inputs.py, product_test_outputs.py, test_decision_logic.js}`; NO `node_modules/`, NO `scratch/`, NO logs, NO checkpoints); commit `Ship dual-head OOD rejection model`; `git push origin main`; verify `git status` clean and `git rev-parse HEAD` == `git rev-parse origin/main` (final state: `local main ≡ origin/main ≡ <commit>`, working tree clean).
