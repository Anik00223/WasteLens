# WasteLens — Iteration 5 shipping report

Model shipped: `models/checkpoints/wastelens_rej_frozen_best.keras`
(**Variant B**) exported to TensorFlow.js in `web/model/`.
Baseline: `main ≡ origin/main ≡ f20c2d6` → first commit of this iteration
`bfeaede` ("Ship dual-head OOD rejection model"), second commit (this
document + verification completion) "Verify shipped dual-head model: docs,
validators, shipping report". All numbers below are real measurements from
this iteration's runs — none are invented.

## 1. Model outputs verified

- `web/model/model.json`: **exactly 2 outputs** — `predictions/predictions`
  (4-unit softmax = bins) and `reject/reject` (1-unit sigmoid = P(unsupported)).
- Label order unchanged: `recyclable`, `organic`, `hazardous`,
  `general trash` (`web/model/labels.json` = `train.BINS`).
- Preprocessing unchanged: 224 × 224, resizeBilinear, `(x / 127.5) - 1`,
  client-side inference.
- Reject-output semantics verified from `src/train_rejection.py::assemble`
  (`y_rej = [0]*supported + [1]*unsupported`): **higher = more likely out of
  scope** — the web treats `reject >= 0.8774` as unsupported.

## 2. Decision logic (shipped, pure, unit-tested)

`decideVerdict(probs, rejectProb)` in `web/index.html` between
`=== DECISION-BEGIN/END ===` markers (extracted verbatim by
`src/test_decision_logic.js`):

```
rejection gate first:  reject >= 0.8774 (REJECT_THRESHOLD)  -> unsupported
else existing rule:    top < 0.60 or margin < 0.50 (unchanged) -> uncertain
else:                                                          supported
```

- Threshold 0.8774 = val-frozen 1% FRR operating point from
  `rejection_metrics_frozen.json` op `"0.01"`; NOT invented or tuned here.
- Rejection has priority over uncertainty — matches how the operating point
  was validated (FRR/detection measured per head; top/margin rule evaluated
  only on non-rejected inputs).
- Three states are never collapsed: `unsupported` (red, scope-language,
  honest wording — no "definitely not waste"/"100% wrong" claims),
  `uncertain` (amber, existing copy), `supported` (normal dark-green card).

## 3. Python ↔ TensorFlow.js parity (LOOP 8)

21 deterministic inputs (5 synthetic + 16 real supported/OOD from the
experiment's own eval sets; `src/make_parity_inputs.py`), identical tensors
through Keras Variant B and TF.js 4.22.0:

- max |Δbins| = **3.636e-6**, max |Δreject| = **2.593e-6**, tolerance
  atol **2e-5** (justified: prior browser proof measured ~5e-6, pure
  float32 noise). **PASS — executed twice: local dir and served HTTP URL
  (identical results), exit 0.**

## 4. Product-output tests on real images (LOOP 10)

`src/product_test_outputs.py` (seed/stride-deterministic selections from
`eval_sets.json` + cache) → 33 cases replayed through the *extracted shipped
decision function* (`src/test_decision_logic.js`):

| group | n | measured result |
|---|---:|---|
| supported_confident | 8 | 8/8 `supported` |
| supported_ambiguous | 1 | 1/1 `uncertain` |
| ood_clothes | 3 | 3/3 `unsupported` (top reject scorers) |
| ood_shoes | 3 | 3/3 `unsupported` (top reject scorers) |
| ood_nonwaste | 3 | 3/3 `unsupported` (top reject scorers) |
| ood_collage | 5 | 5/5 `unsupported` (top reject scorers — see §7) |
| ood_*_miss (genuine below-threshold) | 5 | pass through as `supported`/`uncertain` (reject 0.83–0.875) — honest misses |
| synth_ood (flat probes) | 5 | 0/5 rejected (reject 0.003–0.228) — NOT in calibration data |

Selection note: OOD hit cases are the highest-reject samples of each source
(so they demonstrate the state, not the aggregate rate); aggregate rates are
§6. Miss cases are true below-threshold partition (`first_n` fixed this
iteration — see §10). Collage is never asserted as "must reject".

## 5. Real-world validation (LOOP 11)

`src/validate_realworld.py` is now dual-head aware (`--model/--tag`) and
records `reject` + three-state `state` per record:

- **Baseline run (single-head `wastelens_ep09.keras`):** all prior numbers
  reproduced exactly — diff vs committed artifact is timestamp-only
  (15:02 → 18:58), including 1/32 uncertain and 0/5 probes flagged.
- **Frozen run (Variant B)** → `docs/rejection_experiment/
  realworld_validation_frozen.{json,md}`: identical bin predictions (bins
  head is the frozen copy: 30/32 top ≥ 95%, 1 uncertain), all 32 supported
  reject scores < 0.8774 (max 0.334), probes 0/5 rejected — explicitly
  documented as *not covered by calibration* (calibration sources: clothes,
  shoes, nonwaste, collage).

## 6. Measured OOD performance (Iteration 4, shipped unchanged)

From `rejection_metrics_frozen.json` (test split, threshold frozen on val):

- **84.26% OOD detection @ 1.38% false-rejection rate** at the shipped
  threshold 0.8774; rejection **AUROC 0.9900**, AUPRC 0.9937.
- Per source at the shipped threshold: clothes 84.5%, shoes 74.5%,
  nonwaste 95.5%, **collage 21.1%**.
- At the looser (NOT shipped) 5% FRR point (thr 0.5432): 96.1% overall;
  collage 57.8%.
- Classification unchanged: test accuracy **0.9822**, macro-F1 **0.9742** —
  exactly the baseline (freeze proof max_abs_diff 1.19e-07).

## 7. Collage behavior

Multi-item collages remain the documented weak spot: 21.1% detection at
the shipped threshold (57.8% at the 5% FRR point). In the product tests the
5 collage cases are the top-reject samples of the source and do reject —
this must NOT be read as typical collage behavior. Aggregate performance in
§6 is the number of record. Input-side multi-item heuristics (Iterations
2–3) remain active as the complementary layer.

## 8. Performance (§11 of the brief)

TF.js 4.22.0 CPU backend in Node (no tfjs-node; indicative of a browser CPU
path), 10 warm runs, same machine:

| metric | before (single-head) | after (Variant B dual-head) |
|---|---:|---:|
| model load | 76 ms | 60 ms |
| first inference | 923 ms | 920 ms |
| warm average (10 runs) | 1240 ms | 1189 ms |
| artifacts total | 10,561,905 B | 10,564,166 B (**+2,261 B, +0.02%**) |

No material regression: load/first are effectively identical, warm is
−4.1% (the dual-head run was *faster* within run-to-run noise on this CPU
backend), and size grew 2.3 KB (one extra dense unit + graph bookkeeping).
Not investigated further, per plan. Note: the BEFORE run exits 1 because
the single-head baseline legitimately fails the dual-head structural gate
— perf metrics are still emitted; only the AFTER run is expected exit 0.

## 9. HTTP + page runtime (LOOP 12)

- `py -3.13 -m http.server 8000 --directory web`:
  `GET /index.html → 200` (23,589 B), `GET /model/model.json → 200`
  (215,203 B).
- Harness against `http://localhost:8000/model/model.json`: 6/6 structural
  gates + parity **PASS**, exit 0 — the same loading path the browser uses
  (tfjs 4.22.0 == CDN 4.22.0).
- Inline page script: `node --check` clean; all `src/*.py` compile.

## 10. Bugs found and fixed this iteration (LOOP 15)

1. **Warm-up crash (would have broken the page):** dual-head `predict()`
   returns an *array* of tensors; the ready-path warm-up called
   `warm.data()` on it → TypeError → status stuck loading. Fixed to unwrap
   + `Promise.all` (`web/index.html`).
2. **`first_n` partition bug** (`src/product_test_outputs.py`): everything
   after the nth hit landed in "rest", so `*_miss` groups could contain
   above-threshold items. Fixed to a true predicate partition; artifact
   regenerated (34 → 33 cases; misses now genuinely below threshold).
3. **Perf `load_ms` missing** and printf-style `%.2e` literals printing raw
   in harness output — fixed (`src/validate_reject_in_browser.js`).

## 11. Full enumerated check list (real counts — no "49-check" artifact exists)

| # suite | checks | result |
|---|---:|---|
| Export/topology gates (exit 0, 2 outputs, label order, preprocessing) | 4 | all pass |
| Structural browser-runtime harness (inputs=1, outputs=2, bins 4-unit, reject 1-unit, bins valid, reject in [0,1]) | 6 × 2 runs (local + URL) | 12/12 pass |
| Parity gates (max\|Δbins\|, max\|Δreject\| ≤ 2e-5 over 21 inputs) | 2 × 2 runs | 4/4 pass |
| Decision unit tests (threshold at/±ε, uncertain boundaries, priority, argmax, malformed inputs, fields) | 18 | 18 pass |
| Product replay cases (real model outputs through extracted shipped logic) | 33 | 33 pass |
| Real-world validation runs (baseline reproduce + frozen) | 2 runs (37 records + 5 probes each) | pass |
| `py_compile` (all `src/*.py`) | 10 files | pass |
| Page inline script + harness syntax (`node --check`) | 2 | pass |
| Perf runs (before/after, load+first+warm+bytes) | 2 | pass (after: exit 0; before: metrics captured, exits 1 by design — single-head model under the dual-head gate) |
| HTTP (index 200, model.json 200, URL parity) | 3 | pass |
| **Automated total** | **85 assertions + 2 validation runs** | **all pass** |
| Manual UI checklist (§12) | 12 | 12 code-verified; interactive click-through pending human |

Run history: decision suite **51/51** (18 unit + 33 product), baseline
real-world diff **timestamp-only**, frozen real-world **pass**, harness
`OK - browser-runtime multi-output validation passed` on both executions
(local dir + served URL).

## 12. Manual UI checklist (LOOP 13 — actual observations)

No browser automation was added (per brief: no Puppeteer/Chrome deps).
Each item has runtime or code-level evidence from the served page; items
marked *pending* need a human click-through (this checklist is the record):

| # | item | observation (code/runtime-verified) | interactive |
|---:|---|---|---|
| 1 | model loading | warm-up bug fixed; status → "Model ready" only after both tensors resolve; model.json 200 over HTTP | pending |
| 2 | image upload | `fileInput` change → `handleFile` guard (`modelReady && !busy && file.type image/*`) | pending |
| 3 | camera capture | `capture="environment"` attribute on the file input (mobile camera) | pending |
| 4 | drag/drop | `dragenter/over/leave/drop` listeners + window-level guards present | pending |
| 5 | keyboard | `tabindex=0 role=button` + `keydown` Enter/Space → `fileInput.click()` | pending |
| 6 | supported card | decision tests 8/8 supported + render branch (accent chip, highlighted bars) | pending |
| 7 | uncertain card | unit + real ambiguous case pass; existing amber copy unchanged | pending |
| 8 | unsupported card | 14 real rejections replay to `unsupported`; red card, scope-honest copy | pending |
| 9 | error state | non-image `file.type` → inline error, no inference attempted | pending |
| 10 | repeated inference | `busy` flag blocks re-entry; released in `finally` | pending |
| 11 | mobile layout | base CSS is the mobile layout; `@media (min-width: 700px)` upgrades | pending |
| 12 | desktop layout | same media query: wider grid, centered card | pending |

## 13. Files changed

**Commit 1 `bfeaede`** (15 files, already pushed): `web/index.html`
(dual-output read, `decideVerdict`, 3-state result UI), `web/model/*` +
archived `models/tfjs_model/*` (Variant B TF.js export),
`src/{validate_reject_in_browser.js, make_parity_inputs.py,
product_test_outputs.py, test_decision_logic.js}`, `implementation_plan.md`,
`docs/rejection_experiment/product_test_outputs.json`, baseline refresh of
`docs/realworld_validation.{json,md}`.

**Commit 2** (this change set): `src/validate_realworld.py` (dual-head
`--model/--tag` + reject columns/section 4b), `src/product_test_outputs.py`
(`first_n` partition fix), `src/validate_reject_in_browser.js` (perf
`load_ms` + output formatting), `web/index.html` (warm-up crash fix),
`README.md` (three-state status), `docs/report.md` (§6), 
`docs/rejection_experiment/outcome_decision.md` ("Browser integration"),
`docs/rejection_experiment/realworld_validation_frozen.{json,md}` (new),
`docs/rejection_experiment/shipping_report.md` (this file).

**NOT committed:** `node_modules/`, `scratch/` (perf baseline copy, parity
inputs, logs, extracted script), `__pycache__/`, model checkpoints.

## 14. Remaining limitations (stated exactly — do not overstate)

- Rejection is **not perfect**: measured **84.3% OOD detection @ 1.38%
  FRR** overall (AUROC 0.9900); per source at the shipped threshold:
  clothes 84.5%, shoes 74.5%, nonwaste 95.5%, **collage 21.1%**.
- ~**1.4% of supported images are falsely flagged** as unsupported at the
  shipped threshold (by calibration design — the 1% FRR target landed at
  1.38% on test).
- The 5 flat synthetic probes are **not rejected (0/5)** — flat fields were
  never in the rejection calibration sources, so uncalibrated OOD types
  can still pass through as `supported` with confident bins. The in-panel
  single-item scope disclosure remains for exactly this reason.
- `uncertain` ≠ `unsupported`: the first answers "which bin?", the second
  "in scope?". Neither state claims the image is "definitely not waste",
  "100% incorrect", or "detected as non-waste" — the head *rejects*, it
  does not prove objective truth.
- Interactive browser click-through (§12) is pending a human; automated
  coverage exercises the identical tfjs runtime (4.22.0 == CDN) against the
  served model over HTTP.

## 15. Recommendation for Iteration 6

**Variant C** (the sketched follow-up in `outcome_decision.md`): freeze the
backbone only, unfreeze fc_256 + rejection head while keeping the bins
head's loss weight high — targets the weakest source (**collage**, 21.1% @
shipped threshold) while protecting the exactly-preserved 0.9822 bins
accuracy. Gate it with the same harness / parity / product-test suite —
they are model-agnostic except for the threshold constant
(`REJECT_THRESHOLD` must be re-frozen from val and updated in
`web/index.html`, `src/product_test_outputs.py`, and the tests together).
Secondary backlog: (a) add flat synthetic probes + cluttered scenes to the
rejection calibration set so uncalibrated-OOD statements are measured, not
assumed; (b) complete the interactive §12 click-through and record results.
