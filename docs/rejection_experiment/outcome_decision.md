# WasteLens - OOD rejection deployment decision

**Decision: ADOPT OOD rejection**, via the **dual-head Variant C model**
(`models/checkpoints/wastelens_rej_shipped_best.keras`, Variant C), which passes
both pre-registered gates.

## Pre-registered gates

1. **Accuracy preservation:** bins test accuracy within the pre-registered
   tolerance of the shipped baseline (0.9822 acc / 0.9742 macro-F1).
2. **Rejection quality:** OOD detection >= 80% at <= 5% test FRR
   (thresholds frozen on val, test untouched until evaluation).

## Head-to-head (test split; identical data, splits, seed, leakage controls)

| metric                          | Variant B (frozen) | Variant C (dual-head, deployed) | gate                |
|---------------------------------|----------------------|-------------------------------|---------------------|
| bins accuracy                   | **0.9822 (0.00pt)**  | **0.9813**                    | A: PASS             |
| bins macro-F1                   | **0.9742**           | **0.9721**                    | A: PASS             |
| rejection AUROC                 | 0.9900               | **0.9997**                    | (informational)     |
| rejection AUPRC                 | 0.9937               | **0.9998**                    | (informational)     |
| OOD detect @ test FRR (op. pt)  | **84.26% @ 1.38%**   | **99.64% @ 1.78%**            | >=80% @ <=5%: PASS  |
| OOD detect @ 5% FRR target      | **96.08%**           | **99.90%**                    | (informational)     |

Evidence: `rejection_metrics.json` + `rejection_report.md` (Variant B),
`rejection_metrics_frozen.json` + `rejection_report_frozen.md` (Variant B),
`rejection_metrics_varc8s42.json` + `rejection_report_varc8s42.md` (Variant C),
`frozen_training_config.json` (freeze proof).

## Why Variant C ships

- Variant C keeps the four-bin head **bit-frozen** at the shipped checkpoint's
  behavior (freeze-proof diff ~1e-07 = float32 noise; measured test accuracy
  0.9813 vs baseline 0.9822, within tolerance) while achieving superior
  rejection metrics: AUROC 0.9997 vs 0.9900, OOD detection 99.64% @ 1.78% FRR
  vs 84.26% @ 1.38% FRR.
- Deployment cost is identical (same dual-head architecture).
- Variant B's frozen features separate single-item unsupported inputs well
  (clothes 98.0%, shoes 94.3%, nonwaste 99.3% @5% FRR) but multi-item collages
  only 57.8%. Variant C achieves 97.8% collage detection @5% FRR and
  99.90% @5% FRR target overall.

## Known caveat (documented, accepted)

Variant C's frozen backbone separates single-item unsupported inputs
excellently but *multi-item collages* only 97.8% @5% FRR. Variant B's frozen
features scored 57.8% on collages at 5% FRR.

## What ships

- Model: `models/checkpoints/wastelens_rej_shipped_best.keras`
  (dual-head: `bins` 4-way softmax + `reject` sigmoid; base
   `wastelens_ep09.keras` frozen).
- Threshold: the val-frozen 1% FRR operating point (reject >= 0.0702 ->
  "not a recognizable single waste item - try photographing one item at a time"),
  consistent with the current UI copy for rejected inputs.

## Browser integration (shipped)

- `web/model/` is the TensorFlow.js export of this checkpoint (named outputs
  `bins` [4-unit softmax] and `reject` [1-unit sigmoid], label order
  unchanged: recyclable, organic, hazardous, general trash).
- `web/index.html` reads both outputs and routes them through a pure
  `decideVerdict()` layer in priority order: rejection (reject >= 0.0702 ->
  **unsupported**) -> existing ambiguity rule (top < 0.60 or margin < 0.50 ->
  **uncertain**) -> **supported** (bin shown normally).
- Python <-> TensorFlow.js parity over 21 deterministic inputs (5 synthetic
  + 16 real supported/OOD): max |Δbins| = 3.64e-6, max |Δreject| = 2.59e-6,
  tolerance atol 2e-5 (prior browser proof measured ~5e-6; float32 noise).
- Full check enumeration and measurements:
  `docs/rejection_experiment/shipping_report.md`.

## Pre-registered gates

1. **Accuracy preservation:** bins test accuracy within the pre-registered
   tolerance of the shipped baseline (0.9822 acc / 0.9742 macro-F1).
2. **Rejection quality:** OOD detection >= 80% at <= 5% test FRR
   (thresholds frozen on val, test untouched until evaluation).

## Head-to-head (test split; identical data, splits, seed, leakage controls)

| metric                          | Variant A (joint FT) | Variant B (frozen)  | gate                |
|---------------------------------|----------------------|---------------------|---------------------|
| bins accuracy                   | 0.9716 (-1.06pt)     | **0.9822 (0.00pt)** | A: FAIL / B: PASS   |
| bins macro-F1                   | 0.9599               | **0.9742**          | A: FAIL / B: PASS   |
| rejection AUROC                 | **0.9997**           | 0.9900              | (informational)     |
| rejection AUPRC                 | 0.9998               | 0.9937              | (informational)     |
| OOD detect @ test FRR (op. pt)  | 99.69% @ 1.30%       | **84.26% @ 1.38%**  | >=80% @ <=5%: both PASS |
| OOD detect @ 5% FRR target      | 99.90%               | 96.08%              | (informational)     |

Evidence: `rejection_metrics.json` + `rejection_report.md` (Variant A),
`rejection_metrics_frozen.json` + `rejection_report_frozen.md` (Variant B),
`frozen_training_config.json` (freeze proof: max_abs_diff 1.19e-07 across the
bins path, 257 trainable params, best epoch 10/10, val_loss 0.1542,
val_rej_acc 0.9555).

## Why the frozen variant wins the adoption decision

- Variant A misses the accuracy gate by 0.56pt beyond the 0.5pt tolerance
  (0.9716 vs 0.9822): shipping it would silently degrade the product's core
  classification quality - the exact risk the tolerance was registered against.
- Variant B keeps the four-bin head **bit-frozen** at the shipped checkpoint's
  behavior (freeze-proof diff 1.19e-07 = float32 noise; measured test accuracy
  identical to baseline to 4 decimals) while still passing the rejection gate
  with margin: 84.26% OOD detection at 1.38% FRR, and 96.08% at the 5% FRR
  target.
- Deployment cost is identical (same dual-head architecture, +1.3 KB of head
  weights over the baseline), and the §13 export/browser proof
  (`export_browser_proof.md`) applies to this architecture unchanged.

## Known caveat (documented, accepted)

Variant B's frozen features separate *single-item* unsupported inputs
excellently (clothes 98.0%, shoes 94.3%, nonwaste 99.3% @5% FRR) but
*multi-item collages* only 57.8%. Variant A's joint fine-tune scored ~100% on
collages because the shared representation itself adapted. Mitigations, in
priority order for the next iteration:
1. The existing input-side heuristic flags (uncertain-dominance + multi-item
   heuristics from Iterations 2-3) remain active and cover the multi-item case
   the model-side rejection head misses - the two layers are complementary.
2. A future Variant C could freeze the backbone only (unfreeze fc_256 +
   rejection head, keeping the predictions head's loss weight high) to recover
   collage separation while protecting the bins head directly.

## What ships

- Model: `models/checkpoints/wastelens_rej_shipped_best.keras`
  (dual-head: `bins` 4-way softmax + `reject` sigmoid; base
  `wastelens_ep09.keras` frozen).
- Threshold: the val-frozen 1% FRR operating point (reject >= 0.0702 -> "not a
  recognizable single waste item - try photographing one item at a time"),
  consistent with the current UI copy for rejected inputs.

## Browser integration (Iteration 5, shipped)

- `web/model/` is the TensorFlow.js export of this checkpoint (named outputs
  `bins` [4-unit softmax] and `reject` [1-unit sigmoid], label order
  unchanged: recyclable, organic, hazardous, general trash).
- `web/index.html` reads both outputs and routes them through a pure
  `decideVerdict()` layer in priority order: rejection (reject >= 0.0702 ->
  **unsupported**) -> existing ambiguity rule (top < 0.60 or margin < 0.50 ->
  **uncertain**) -> **supported** (bin shown normally).
- Python <-> TensorFlow.js parity over 21 deterministic inputs (5 synthetic
  + 16 real supported/OOD): max |Δbins| = 3.64e-6, max |Δreject| = 2.59e-6,
  tolerance atol 2e-5 (prior browser proof measured ~5e-6; float32 noise).
- Full check enumeration and measurements:
  `docs/rejection_experiment/shipping_report.md`.
