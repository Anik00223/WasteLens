# WasteLens — Iteration 7: protocol-gated Variant C checkpoint selection

Production model: **Variant B** (unchanged). Variant C was re-trained with an
identical configuration and its checkpoint chosen by a **pre-registered rule**.
Result: **Outcome B — the gate fails on classification, Variant B stays
production.** The post-hoc diagnostics (clearly marked below) show the
objective itself can reach the classification tolerance at the final epoch,
which becomes the Iteration-8 lead.

---

## 1. Pre-registered checkpoint rule (LOOP 2 — fixed before training)

Rule text (identical to the Iteration-7 brief):

> Select the lowest validation-loss checkpoint among epochs where the
> validation bin accuracy is at least 98% of the Variant B baseline.

Implementation: `src/select_varc_checkpoint.py` (constants registered in code
*before* the run: `BASELINE_ACC = 0.9822`, `FLOOR_RATIO = 0.98`).

- floor = 0.9822 × 0.98 = **0.962556** (computed in code, printed at runtime)
- "validation bin accuracy" = top-1 accuracy of the bins head on the
  **supported validation images only** (`eval_sets.json → supported_val`,
  1,231 images, labels from the same seed-42 split). The training-history
  column `val_bins_sparse_categorical_accuracy` is computed over the *mixed*
  validation batch (supported + unsupported rows, unsupported rows carry no
  valid bin target) and is therefore **not** a bin-accuracy measurement; it is
  recorded but not used for selection. This metric definition was written into
  the script before the run.
- "validation loss" = the per-epoch `val_loss` recorded in the training
  history CSV (the quantity the trainer's ModelCheckpoint monitors).
- The **test set is never touched** by the selector.
- No epoch was chosen manually and no test metric was inspected before the
  decision.

## 2. Training configuration (LOOP 3–5 — identical to Iteration 6, one tag)

| item | value |
|---|---|
| architecture | MobileNetV2 backbone **frozen**; `fc_256` + bins head + reject head trainable |
| trainable / frozen params | **329,221 / 2,257,984** (tensors: fc_256/kernel+bias, predictions/kernel+bias, reject/kernel+bias) |
| diffs vs Iteration 6 | **none** — same script, same `--bins-weight 2.0`; only `--out-tag 7` isolates the run's artifacts |
| seed | 42 |
| epochs | 10 (`models/checkpoints/wastelens_rej_varc7_01..10.keras` + `_best`, all kept until selection) |
| optimizer / LR / batch | Adam(1e-3) / 1e-3 / 32 (unchanged from Variant A/B) |
| loss weights | bins **2.0** / reject **1.0** |
| dataset | exactly the Iteration-6 protocol: seed 42, dedup, 70/15/15 supported split, clothes/shoes (md5-deduped), CIFAR-10 non-waste, collages from disjoint train pools, synthetic probes; **no test example in training** |
| images | train 14,218 · val 3,165 |
| training time | **1,524.78 s** |
| best-by-val_loss epoch (trainer monitor) | 3 (val_loss 0.069174) |

Config artifact: `variantc7_training_config.json`; full history:
`variantc7_history.csv`.

## 3. Selection record (LOOP 6 — applied mechanically, no test data)

Per-epoch supported-val metrics (all recorded *before* any test evaluation,
`variantc7_selection.json`):

| epoch | val_loss | sup-val bins acc | sup-val macro-F1 | mean reject (sup-val) | floor 0.962556 |
|---:|---:|---:|---:|---:|:--|
| 1 | 0.079480 | 0.9659 | 0.9533 | 0.0161 | PASS |
| 2 | 0.099479 | 0.9634 | 0.9504 | 0.0203 | PASS |
| 3 | **0.069174** | 0.9756 | 0.9668 | 0.0206 | PASS |
| 4 | 0.087057 | 0.9643 | 0.9510 | 0.0249 | PASS |
| 5 | 0.132014 | 0.9805 | 0.9702 | 0.0040 | PASS |
| 6 | 0.091167 | 0.9781 | 0.9686 | 0.0076 | PASS |
| 7 | 0.114518 | 0.9764 | 0.9662 | 0.0089 | PASS |
| 8 | 0.134587 | 0.9764 | 0.9663 | 0.0056 | PASS |
| 9 | 0.110461 | 0.9740 | 0.9672 | 0.0098 | PASS |
| 10 | 0.114210 | 0.9789 | 0.9691 | 0.0085 | PASS |

**Selected: epoch 3** (`wastelens_rej_varc7_03.keras`) — lowest val_loss
(0.069174) among the eligible epochs (all 10 passed the floor). Reproducible
from the saved history + checkpoints by re-running the selector.

## 4. Test evaluation of the selected checkpoint (LOOP 7)

### Classification (untouched test, 1,147 supported images)
accuracy **0.9667** · macro-F1 **0.9536**

| bin | P | R | F1 |
|---|---:|---:|---:|
| recyclable | 0.9914 | 0.9607 | 0.9758 |
| organic | 0.9608 | 0.9932 | 0.9767 |
| hazardous | 0.9189 | 0.9577 | 0.9379 |
| general trash | 0.8655 | 0.9904 | 0.9238 |

confusion matrix (rows=true, cols=pred; order recyclable, organic, hazardous,
general trash): `[[806,6,11,16],[1,147,0,0],[6,0,136,0],[0,0,1,103]]`

### Rejection (thresholds frozen on VAL, evaluated on untouched test)
AUROC **0.9995** (AUPRC 0.9997)

| FRR target | val-frozen thr | OOD detect | test FRR |
|---|---:|---:|---:|
| 1% | 0.5287 | 0.9943 | 0.0154 |
| 5% | 0.0923 | 0.9979 | 0.0535 |
| 10% | 0.0264 | 1.0000 | 0.1160 |
| 15% | 0.0134 | 1.0000 | 0.1598 |
| 20% | 0.0067 | 1.0000 | 0.2084 |

Per-source OOD detection @5% FRR target — **collage reported separately, never
folded into the aggregate**: clothes 1.0000 · shoes 0.9966 · **collage 0.9778**
· nonwaste 0.9987. @1% FRR target: clothes 0.99875 · shoes 0.9832 ·
**collage 0.9556** · nonwaste 0.99867.

At **Variant B's shipped threshold 0.8774** (never re-tuned for Variant C,
`variantc7_shipped_threshold.json`): test FRR 0.49%, OOD 98.76%, clothes
99.38% · shoes 96.98% · **collage 91.11%** · nonwaste 99.73%.

Synthetic probes: reject scores 0.1203 / 0.7441 / 0.0782 / 0.9914 / 0.5933 →
3/5 above the candidate's 1% point 0.1811, 1/5 above 0.8774 (Variant B: 0/5).
Still uncalibrated synthetic data; recorded, not asserted.

## 5. Variant B reference re-verified (LOOP 1)

`rejection_metrics_frozen.json` re-read this iteration: acc 0.9822, macro-F1
0.9742, AUROC 0.9900, OOD @1% FRR 0.8426 (test FRR 0.0138), collage @5% FRR
0.5778, collage @0.8774 0.2111 — matches the Iterations 4–6 record.

**Floor-reference measurement (new):** Variant B's *own* supported-val bins
accuracy is **0.9829** (macro-F1 0.9773) on the same 1,231 images the selector
uses. The registered floor 0.962556 is therefore only 97.9% of the baseline's
val accuracy — it permits a checkpoint up to ~2.0pt below the baseline on the
very split the rule measures.

## 6. Adoption gate (LOOP 9 — criteria fixed before the run)

| # | condition | result | verdict |
|---|---|---|---|
| 1 | bin accuracy within the registered tolerance (±0.5pt) | 0.9667 vs 0.9822 = **−1.55pt** | **FAIL** |
| 2 | macro-F1 within the registered tolerance | 0.9536 vs 0.9742 = **−2.06pt** | **FAIL** |
| 3 | OOD detection improves meaningfully | 99.43% vs 84.26% @1% FRR | pass |
| 4 | collage detection improves meaningfully | 97.78% vs 57.78% @5% FRR (91.1% vs 21.1% at 0.8774) | pass |
| 5 | false rejection within target | 1.54% @1% target (0.49% at shipped thr) | pass |
| 6 | export works | TF.js export, dual outputs, label order verified | pass |
| 7 | Python↔TF.js parity | PASS local & HTTP (2.563e-6 / 1.639e-6 vs atol 2e-5) | pass |

**Outcome B.** Two classification gates fail, so Variant C is **not promoted**;
Variant B remains production and `web/model/` was not touched. Standards were
not lowered in either direction: the collage gain (gate 4) does not buy an
exemption, and the classification failure is not excused by the rejection win.

## 7. Post-hoc diagnostics (disclosed; NOT eligible for adoption)

After the gate decision was recorded, the neighbouring checkpoints were
measured on test to answer "is this selection, or the objective?" (same
disclosure practice as Iteration 6's ep08):

| checkpoint | why measured | test acc | macro-F1 | AUROC | OOD @1% FRR | test FRR | collage @5% FRR |
|---|---|---:|---:|---:|---:|---:|---:|
| ep3 (selected) | pre-registered rule | 0.9667 | 0.9536 | 0.9995 | 0.9943 | 0.0154 | 0.9778 |
| ep5 | highest sup-val acc (0.9805) | 0.9732 | 0.9595 | 0.9995 | 0.9959 | 0.0178 | 0.9889 |
| ep6 | lowest val_loss under a 0.9762 floor | 0.9773 | 0.9658 | 0.9996 | 0.9948 | 0.0146 | 0.9889 |
| ep10 | final epoch of the fixed budget | **0.9789** | **0.9698** | 0.9995 | 0.9943 | 0.0130 | 0.9889 |

Observations (diagnostic only):
- Bins accuracy **recovers monotonically after ep3** while rejection stays
  ~99%: the degradation is *selection*, not the objective.
- ep10 would satisfy gates 1–5 (accuracy −0.33pt, macro-F1 −0.44pt, both inside
  the 0.5pt tolerance) with collage @5% 98.89%. It was **not** the
  pre-registered choice, so it cannot be adopted in this iteration — promoting
  it now would be exactly the post-hoc selection the brief forbids.
- ep5 (the val-accuracy maximum, the "classification-first" rule) does *not*
  transfer best to test (0.9732) — val accuracy is a noisy proxy at this scale.
- **Run-to-run variance:** the Iteration-6 and Iteration-7 runs share seed,
  config and data, yet ep3 scored 0.9611 (iter 6) vs 0.9667 (iter 7), and
  iter-6 ep08 reached 0.9813. Single-seed results carry ≈±0.5pt noise (TF
  oneDNN/thread scheduling), which matters when the tolerance is 0.5pt.

## 8. Threshold calibration (LOOP 10)

Variant C was not promoted, so the production threshold is unchanged
(0.8774, Variant B's val-frozen 1% point). For the record, the selected
checkpoint's own thresholds (§4 table) were frozen on **validation only** and
then evaluated on the untouched test set; the candidate's 1% point is
**0.1811** (test FRR 1.54%). No test data was used to choose any threshold.

The rule that produced those thresholds (same rule as Variants A/B, unchanged):
for each target FRR, take the smallest reject score whose supported-**val**
false-rejection rate is ≤ target, freeze it, and apply it to the untouched
test set. Validation FRR at the 1% point = 0.0098 (12/1231); test FRR = 1.54%.
Calibration stability note: the candidate's reject distribution is sharply
bimodal (mean sup-val reject 0.0206, most unsupported scores >0.9), so the
1%-FRR threshold lands at 0.5287 — mid-gap — rather than at a fragile
percentile edge.

## 9. Export and browser parity (LOOP 11–12)

- export: `src/export_tfjs.py --model …varc7_03.keras --out scratch/tfjs_varc7`
  → `model.json` (215,194 B) + 3 shards + `labels.json` (order
  `['recyclable', 'organic', 'hazardous', 'general trash']`, unchanged);
  outputs `predictions/predictions` (4-class softmax) and `reject/reject`
  (sigmoid); preprocessing untouched (raw [-1,1], 224×224).
- harness structural gates: 6/6 pass; **parity PASS twice** — local artifacts
  and HTTP-served (`http://127.0.0.1:8123/scratch/tfjs_varc7/model.json`):
  max|Δbins| **2.563e-6**, max|Δreject| **1.639e-6** ≤ atol 2e-5.
- perf (tfjs-cpu, Node): load 68 ms, first predict 697 ms, warm 668 ms.
- `web/model/` was **not** modified; `scratch/` remains gitignored.

## 10. Product replay (LOOP 13)

| replay | predictions | threshold | result |
|---|---|---:|---|
| production (default file) | Variant B, 51 checks | 0.8774 | **51 passed, 0 failed** |
| candidate @ shipped threshold | varc7_03 | 0.8774 | **46 passed, 0 failed** |
| candidate @ own 1% point | varc7_03 | 0.1811 | 43 passed, **2 failed** |

The two failures in row 3 are the known threshold-coupling artifact, not a
logic defect: `synth_ood_1` (reject 0.7441) and `synth_ood_4` (0.5933) are
"unsupported" at the candidate's own 0.1811 point but the shipped page logic
correctly decides "supported" at 0.8774. A promoted candidate must ship with
its own re-frozen threshold constant. Priority order
`unsupported → uncertain → supported` verified in rows 1–2; 24/24 supported
replay images → "supported" (0 false rejects), 22/23 recyclable-source replay
cases rejected at 0.8774 (1 measured shoes miss recorded). Collage is treated
as an improvement target, never as a guaranteed rejection class.

## 11. Regression results (LOOP 14 — actual checks, enumerated)

1. `py_compile` × 9 python files → exit 0
2. `node --check` × 3 JS tools → exit 0
3. production decision suite `node src/test_decision_logic.js` → **51/51**
4. production browser harness (local `web/model`) → OK, parity 3.636e-6 / 2.593e-6
5. production browser harness over HTTP + perf → OK, load 118 ms / first 717 ms / warm 714 ms
6. HTTP smoke: `web/index.html` 200, `web/model/model.json` 200
7. candidate export structural gates + parity (local & HTTP) → PASS
8. candidate product replay @ shipped threshold → 46/46
9. real-browser UI click-through (`src/ui_clickthrough.js`, headless Chrome) →
   **13/13** (page load, upload, camera attribute, drag&drop, keyboard,
   supported/uncertain/unsupported, error state, repeated inference, mobile
   375px, desktop 1280px, 0 console errors)
10. four full harness evaluations completed (varc7, e5, e6, e10)
11. selector re-run (`select_varc_checkpoint.py … --out scratch/selection_recheck.json`)
    → per-epoch metrics **identical** to the saved record, floor 0.962556,
    **selected epoch 3 again** (determinism of the pre-registered rule verified)
12. CLI wiring check: `train_rejection_variantc.py --help` renders the fixed
    `--model/--out` help; smoke run with `--out-tag 7smoke --out
    scratch/smoke_best.keras` → checkpoint written to the requested path,
    unfreeze proof passed, 10/10 smoke epochs ok, artifacts deleted afterwards

No test was weakened, removed, or re-tuned.

## 12. Comparison (measured only)

| Model | Bin Accuracy | Macro-F1 | OOD @1% FRR | OOD @5% FRR | Shipped-threshold OOD | Collage @5% FRR | Collage @0.8774 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Variant A (iter 4/5, joint FT) | 0.9716 | 0.9599 | 0.9969 | 0.9990 | — (own thr 0.3101) | 1.0000 | — |
| **Variant B (production)** | **0.9822** | **0.9742** | **0.8426** | **0.9608** | 0.8426 @1.38% FRR | 0.5778 | 0.2111 |
| Variant C, pre-registered ep3 | 0.9667 | 0.9536 | 0.9943 | 0.9979 | 0.9876 @0.49% FRR | 0.9778 | 0.9111 |
| Variant C, ep10 (post-hoc only) | 0.9789 | 0.9698 | 0.9943 | 0.9985 | — | 0.9889 | — |

## 13. What changed / did anything break?

Changed: the Iteration-7 run's artifacts, the pre-registered selector tool, and
an `--out-tag` argument on the Variant C trainer. Production model, page,
threshold, `web/model/`, and the four-bin contract are **unchanged and
re-verified** (`web/index.html` still `REJECT_THRESHOLD = 0.8774`,
`MODEL_URL = 'model/model.json'`, labels order unchanged). Nothing shipped
broke; no failing model/browser/product test is being carried forward.

Small fixes made during review (do not affect any recorded measurement):
- `train_rejection_variantc.py` accepted `--model`/`--out` but ignored them
  (a defect inherited from Iteration 6). They are now honored; defaults still
  reproduce the original names, and the Iteration-7 run was launched with
  defaults only, so its artifacts are unaffected. Validated with a smoke run.
- Note for readers of the auto-generated evidence: `rejection_report_varc7*.md`
  carry the evaluator's static title line ("Iteration 4") — the model path on
  line 3 is the authoritative identifier (`…varc7_03/_05/_06/_10.keras`).

## 14. Recommendation for Iteration 8

1. **Pre-register a no-selection rule** — "fixed 10-epoch budget, final epoch
   is the candidate" — and re-run the identical configuration. Rationale: the
   rejection head converges by ep3 while the bins head recovers monotonically;
   ep10 measured 0.9789 / 0.9698 (inside tolerance) with collage @5% 98.89%.
   If the final-epoch checkpoint reproduces inside tolerance, ship it **with
   its own val-frozen threshold** and the coordinated constant update in
   `web/index.html`, `src/product_test_outputs.py` and the tests.
2. **Fix the floor reference:** the rule should compare against Variant B's
   *same-split* val accuracy (0.9829, newly measured) with an explicit,
   pre-registered tolerance — e.g. `sup-val acc ≥ 0.9829 − 0.25pt` **and**
   `sup-val macro-F1 ≥ 0.9773 − 0.25pt` — instead of 98% of a test metric,
   which permits a ~2pt val gap.
3. **Reduce selection noise before shipping:** either make the run
   deterministic (TF_DETERMINISTIC_OPS=1, single-threaded intra-op) or require
   the winning configuration to reproduce on a second seed; ±0.5pt run-to-run
   noise is the same size as the tolerance.
4. **Keep a classification-safe fallback ready:** a reject-head-only variant
   trained on collage-augmented data (bins head bit-frozen) cannot regress
   gate 1 at all and would isolate how much collage rejection is achievable
   inside the frozen feature space that Variant B already uses.

