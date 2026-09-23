# WasteLens Iteration 6 — Variant C experiment report

*All numbers below are real measurements on this machine (py 3.13.1, TF
2.21.0 CPU, seed 42, identical datasets/splits to Iterations 4–5). No
number is invented or estimated.*

## 1. Question

Can partially unfreezing the shared classifier layers improve rejection of
multi-object/collage inputs while preserving the four-bin classification
performance (0.9822 / macro-F1 0.9742)?

## 2. Variant C definition (LOOP 3)

- MobileNetV2 backbone: **frozen** (bit-identical to the shipped baseline —
  re-asserted and verified before training).
- Trainable: `fc_256` (327,936) + `predictions` (1,028) + `reject` (257)
  = **329,221 trainable / 2,257,984 frozen parameters**.
- Four-bin head: intact (same layer, labels, ordering, preprocessing,
  input size, browser architecture — unchanged).
- **Changed hyperparameter (the only one vs Variant A):** loss weight
  `bins: 2.0` (A used 1.0) with `reject: 1.0`. Optimizer Adam(1e-3),
  batch 32, 10 epochs, monitor val_loss — all inherited unchanged.

## 3. Training configuration (LOOP 5)

From `variantc_training_config.json`: seed 42 (`tf.keras.utils.set_random_seed`),
10 epochs, 14,218 train / 3,165 val images (same assemble as A/B), elapsed
1742.6 s, best epoch **3** by val_loss (0.0751). Pre-training unfreeze proof:
backbone weights bit-identical to the base checkpoint; untrained bins output
max|Δ| 1.19e-07 vs baseline (one-ULP float noise, same magnitude as
Variant B's freeze proof).

## 4. Fresh baseline (LOOP 2)

Re-ran the shipped Variant B through `src/eval_rejection.py` this iteration
(`rejection_metrics_c6_baseline.json`): **reproduces Iterations 4–5 exactly**
— acc 0.9822, macro-F1 0.9742, AUROC 0.9900, OOD @1% FRR 84.26% (test FRR
1.38%), @5% 96.08%, @10% 98.40%, @15% 99.23%, @20% 99.43%; collage @5% FRR
57.78%; at the shipped threshold 0.8774 → collage 21.11%. No stale numbers
were used anywhere in this comparison.

## 5. Variant C results (LOOP 6) — protocol-selected checkpoint

`models/checkpoints/wastelens_rej_varc_best.keras` = epoch 3 (best
val_loss, the same selection convention used for A and B).

**Classification (supported test, 1,233 images):**

| Bin | P | R | F1 |
|---|---|---|---|
| recyclable | 0.9950 | 0.9535 | 0.9738 |
| organic | 0.8605 | 1.0000 | 0.9250 |
| hazardous | 0.9195 | 0.9648 | 0.9416 |
| general trash | 0.9259 | 0.9615 | 0.9434 |

accuracy **0.9611** (baseline 0.9822, −2.11pt), macro-F1 **0.9460**
(baseline 0.9742, −2.82pt). Confusion matrix in
`rejection_metrics_varc.json`; the organic bin collapsed toward high recall
/ low precision.

**Rejection:** AUROC **0.9996** (AUPRC 0.9997); thresholds frozen on VAL,
evaluated on TEST (LOOP 7 protocol):

| FRR target | thr | OOD detect | test FRR |
|---|---|---|---|
| 0.01 | 0.3125 | 0.9943 | 0.0146 |
| 0.05 | 0.0396 | 0.9985 | 0.0600 |
| 0.10 | 0.0109 | 1.0000 | 0.1184 |
| 0.15 | 0.0051 | 1.0000 | 0.1671 |
| 0.20 | 0.0025 | 1.0000 | 0.2198 |

**By OOD source @5% FRR:** clothes 100.0%, shoes 99.66%, nonwaste 99.87%,
**collage 98.89%**. At Variant C's shipped threshold (0.3125): collage
94.44%, clothes 100%, shoes 98.32%, nonwaste 99.87%.

**Synthetic probes (never trained on):** flat color field and horizontal
gradient rejected; gray noise, checkerboard, blank frame not rejected —
2/5 at C's threshold (Variant B: 0/5). Uncalibrated-OOD limitation remains.

## 6. Post-hoc diagnostic (disclosed, NOT pre-registered)

Variant C's val_bins_acc oscillated across epochs (0.57–0.92) while
val_loss monotonically improved (rejection loss dominating the sum), so the
pre-registered best-val_loss rule selected an epoch whose bins head had
temporarily degraded. For characterization only, epoch 8 (highest
val_bins_acc) was evaluated post-hoc — a selection rule that would have
peeked at validation bins accuracy, i.e. not one we allow ourselves for an
adoption decision: acc 0.9813 (−0.09pt, within the 0.5pt tolerance),
macro-F1 0.9725 (−0.17pt), AUROC 0.9997, OOD @1% FRR 99.64%,
**collage 100.0% @5% FRR** (`rejection_metrics_varc_ep08.json`). This shows
the joint-training objective *can* hold the classifier while learning
rejection — but only with a checkpoint-selection rule the protocol does not
permit, so it cannot be the basis for shipping in this iteration.

## 7. Adoption gate (LOOP 8) — applied to the protocol-selected model

| # | gate | result |
|---|---|---|
| 1 | bins accuracy no material regression from 0.9822 | **FAIL** (0.9611, −2.11pt > 0.5pt tolerance) |
| 2 | macro-F1 no material regression | **FAIL** (0.9460, −2.82pt) |
| 3 | OOD detection improves meaningfully | pass (99.43% vs 84.26% @1% FRR) |
| 4 | collage detection improves meaningfully | pass (98.89% vs 57.78% @5% FRR) |
| 5 | false rejection within target | pass (1.46% @1% target; 6.00% @5%) |
| 6 | browser export viable | pass (TF.js export verified, see §8) |
| 7 | Python↔TF.js parity | pass (max|Δbins| 3.28e-6, |Δreject| 4.71e-6 ≤ 2e-5) |

**Outcome B: Variant C is NOT adopted.** Gates 3–7 all passed and the
rejection gains are real and large, but the two classification gates fail
beyond the pre-registered 0.5pt tolerance — the same failure mode as
Variant A (0.9716), despite the 2.0 bins loss weight. Production remains
Variant B; `web/model/` was never modified.

## 8. Browser proof for Variant C (LOOP 10 — experiment artifacts only)

Exported to `scratch/tfjs_varc/` (production `web/model/` untouched):
dual outputs verified (bins 4-unit + reject 1-unit), labels.json order
unchanged, 10,564,086 B across 4 artifacts; harness
`src/validate_reject_in_browser.js` against the exported artifacts: all
structural gates + **parity PASS** (local and over HTTP:
max|Δbins| 3.278e-6, max|Δreject| 4.709e-6, atol 2e-5), exit 0.

## 9. Product replay (LOOP 11)

`src/product_test_outputs.py` gained `--model/--threshold/--out` (defaults
unchanged) and ran against the ep3 checkpoint at its own val-frozen
threshold 0.3125 → `scratch/product_test_outputs_varc_ep3.json`: supported
24/24 confident, clothes 3/3, shoes 3/3, nonwaste 3/3, **collage 5/5**
rejected in-sample, synth probes 2/5. Decision-layer replay through the
shipped page logic (threshold 0.8774) fails 1 synth case — the expected
coupling artifact: a new model must ship with its OWN re-frozen threshold
updated in `web/index.html`, the replay generator, and the tests together.
The production replay file is untouched and the production suite still
passes 51/51.

## 10. Manual UI verification (LOOP 12 — Iteration-5 gap closed)

New tool `src/ui_clickthrough.js` (headless Chrome + DevTools Protocol,
zero new npm dependencies; native WebSocket) drove the REAL served page:

- **13/13 checks PASSED** (`ui_clickthrough.json`, screenshots
  `ui_clickthrough_mobile.jpeg` / `ui_clickthrough_desktop.jpeg`):
  page-load ready state, real file-input change→inference, camera
  `capture="environment"`, DataTransfer drag&drop, keyboard Enter/Space
  (2 picker activations observed), supported result (rendered bin+bars),
  uncertain result (amber note shown), unsupported result (red note shown
  for a real OOD image), inline error for non-image files, repeated
  inference after busy-release, mobile 375px and desktop 1280px layouts
  with correct media-query state, **0 console errors / 0 page exceptions**.
- Residual human-only scope (stated honestly): the OS file-picker dialog
  itself and a physical phone camera open — the listener side
  (`input.click`) was verified to fire on Enter/Space; every in-page
  behavior is now machine-verified against the real served page.

## 11. Regression results (LOOP 14) — all pass, enumerated

py_compile on the 6 touched/created Python files (exit 0); `node --check`
on 3 JS files; production decision suite **51/51**; production browser
harness local + HTTP `OK` (parity max|Δbins| 3.636e-6, |Δreject| 2.593e-6);
HTTP smoke 200/200 (page + production model.json); Variant C export gates +
parity PASS (local + HTTP); fresh baseline re-run matches Iteration-5
frozen numbers exactly; `git status` shows only intended files (models/,
scratch/, caches gitignored; no node_modules/logs/temp files).

## 12. A / B / C comparison (LOOP 9 — measured numbers only)

| Model | Bin Accuracy | Macro-F1 | OOD @ 5% FRR | Shipped-threshold OOD | Collage @ 5% FRR | Browser |
|---|---:|---:|---:|---:|---:|---|
| Variant A | 0.9716 | 0.9599 | 99.90% | 99.69% @ 1.30% FRR | 100.0% | never exported |
| Variant B | **0.9822** | **0.9742** | 96.08% | 84.26% @ 1.38% FRR | 57.8% | **Yes (shipped)** |
| Variant C (ep3, protocol) | 0.9611 | 0.9460 | 99.85% | 99.43% @ 1.46% FRR | 98.9% | verified, not shipped |
| Variant C (ep8, post-hoc) | 0.9813 | 0.9725 | 99.85% | 99.64% @ 1.46% FRR | 100.0% | verified, not shipped |

## 13. Recommendation for Iteration 7

Variant C's ep8 diagnostic shows the failure is a **checkpoint-selection**
problem, not an objective problem: monotone val_loss (dominated by the easy
rejection head) picked an epoch where the bins head was in transient
degradation, while a later epoch sat within tolerance. Pre-register (before
training) a compound selection rule — e.g. "best val_loss among checkpoints
with val_bins_acc ≥ 0.98 × baseline", or early stopping on a val
bins-accuracy floor — retrain with the same single-knob change (bins weight
2.0), and re-run gates 1–7. If a protocol-selected Variant C then passes,
ship it with its own re-frozen threshold and the coordinated constant
update in `web/index.html` / `src/product_test_outputs.py` / the tests; the
gate tooling (harness, parity, replay, decision tests, UI click-through) is
now model-agnostic and ready.
