# WasteLens — Iteration 11: PRE-REGISTERED classification-head-only adaptation protocol

Written and committed **before any Iteration-11 training run and before any
Iteration-11 candidate metric exists** (baseline commit `ae1410f`).

Production is untouched by this iteration's design: the shipped
Iteration-8 Variant C checkpoint
(`models/checkpoints/wastelens_rej_shipped_best.keras`, md5
`8735019226412c4d67b0a17269dc38ad`) and the rejection threshold
**0.0702** remain the production constants.

## 0. Question and relationship to Iterations 9/10

* Iteration 9 (held-out benchmark `fresh_eval_protocol.md`): fresh supported
  accuracy **0.3968**, macro-F1 0.2546, OOD detection 0.6744 @ 0.3333 FRR.
* Iteration 10 (protocol `iteration10_adaptation_protocol.md`): unfreezing
  `fc_256` + both heads produced **0.5238 / 0.5397** fresh accuracy with all
  three collapse bins improving, but re-calibrated the rejection head as a
  side effect (fresh OOD 0.6047 seed 42; fresh FRR 0.1905 / 0.3016) and
  failed the pre-registered 0.55 gate → decision
  **Improvement-but-fail**, production kept.
* Iteration 11 asks a deliberately narrow isolation question:

  > **Can the fresh-domain classification problem be improved without
  > changing the learned rejection representation at all?**

  The rejection pathway (backbone → gap → dropouts → fc_256 → `reject`) is
  frozen, so any measured rejection change is a bug in this experiment's
  isolation, not an adaptation trade-off. LOOP 12 tests that directly.

## 1. Graph facts (measured, not assumed)

`scratch/_graph_probe.py` (run at `ae1410f`) on the shipped checkpoint:

```
image (None,224,224,3)
  -> mobilenetv2_1.00_224   (None,7,7,1280)   [nested Functional, frozen]
  -> gap                    (None,1280)
  -> dropout_1 (0.2)        (None,1280)
  -> fc_256  (256, relu)    (None,256)
  -> dropout_2 (0.2)        (None,256)   <-- keras_tensor_325
       -> predictions (4, softmax)   (None,4)
       -> reject      (1, sigmoid)   (None,1)
```

* Both heads consume the **same** tensor: the `dropout_2` output
  (`same input tensor for both heads: True`).
* Shipped tensor inventory: **6 trainable** (`fc_256/{bias,kernel}`,
  `predictions/{bias,kernel}`, `reject/{bias,kernel}`) + **260 non-trainable**
  (backbone, including all BatchNorm moving statistics).
* 8 top-level layers; the backbone is a nested model, so its BatchNorm
  layers live inside `mobilenetv2_1.00_224` and stay in inference mode
  (`train.py: base.trainable = False`).

## 2. The single changed variable: the trainable set

| group | tensors | params |
|---|---|---|
| **trainable (the experiment)** | `predictions/kernel` (4×256), `predictions/bias` (4) | **1,028** |
| frozen | backbone (260 tensors incl. BN stats), `gap`, `dropout_1`, `dropout_2`, `fc_256/kernel` (256×1280), `fc_256/bias`, `reject/kernel` (256×1), `reject/bias` | 2,590,000+ |

Everything else stays byte-identical to Iteration 10 (§4), because the
experiment is only meaningful if the trainable set is the sole changed
variable.

## 3. Dataset freeze (no data changes)

| item | frozen value |
|---|---|
| `docs/rejection_experiment/adaptation_set_manifest.json` | md5 `acae0933aca123c835c1e55396af182a` (94 entries: 60 adapt-supported, 28 adapt-ood, 6 adapt-multi) |
| `scratch/adaptation_set/` | 94 files |
| mix / split code path | imported from `src/train_adaptation.py` (`adapt_rows`, `split_adapt`, `upsample`, `build_mixed_arrays`, `pack_rows`, `make_ds`) — **not reimplemented** |
| original pools | `docs/rejection_experiment/eval_sets.json`, unchanged |
| fresh benchmark | `docs/rejection_experiment/fresh_set_manifest.json` md5 `99540f56efb95e2c26b2b6cc9414ff5d` (131 images + 5 synthetic probes), never modified |
| leakage audit | must reproduce `adaptation_eval_audit.json` byte-for-byte **except** `generated_utc` (checked mechanically); a hit aborts |

The trainer asserts both md5 locks and the file count before loading the
model; the evaluator re-runs the audit and diffs it against the committed
Iteration-10 audit.

## 4. Training protocol (registered before training)

| item | value |
|---|---|
| seeds | **42** and **43** (both evaluated; neither discarded, no winner-picking) |
| epochs | **6** fixed budget, no early stopping |
| candidate | **epoch 6 (final epoch)**, copied to `_final`/`_best` (copied, never selected) |
| optimizer / lr | Adam, **1e-3** (identical to Iteration 10) |
| batch size | 32 |
| loss | `{"bins": sparse_ce, "reject": binary_ce}`, weights **2.0 / 1.0** (identical to Iteration 10); the reject term contributes exactly **zero** gradient because every reject-path tensor is frozen |
| augmentation | **none** (Iteration 10 had none either: decode → resize 224 → MobileNetV2 preprocess) |
| data mix | replayed 3:1 original:adapt, per-role, seed-independent (same as Iteration 10) |
| class weights | bins inverse-frequency on the mixed supported pool; rejection weights inverse-frequency on the packed pool (same formula) |
| determinism | `TF_DETERMINISTIC_OPS=1` + `enable_op_determinism()` + `set_random_seed(seed)` |
| validation | Iteration-8 original val pool (fit/val history) + adaptation val split (**monitoring only**, never selection) |
| test data | never touched during training |
| checkpoints | scratch only: `models/checkpoints/wastelens_rej_head11s{SEED}_{01..06,final,best}.keras` |

## 5. Warm-start / freeze proof (run and recorded per seed)

Before training, against the shipped checkpoint:

1. checkpoint md5 equals the production constant,
2. trainable tensor list is exactly
   `["predictions/bias", "predictions/kernel"]` (1,028 params) and every
   other tensor is non-trainable,
3. SHA-256 digest of **all 260 frozen tensors** recorded
   (per-tensor `numpy().tobytes()` hash + an aggregate),
4. bins **and** reject outputs on 16 real original-train images are
   bit-exact vs the shipped model (max abs diff 0.0 expected; > 1e-6 aborts).

After training, the same digests are recomputed and **must be identical**
(frozen weights, BatchNorm moving statistics and rejection head unchanged);
only `predictions/{kernel,bias}` may differ. Both proofs are written to
`head11s{SEED}_freeze_proof.json`.

## 6. Pre-registered gates (computed mechanically; no post-hoc moves)

| Gate | Rule |
|---|---|
| **G1 existing benchmark** | original supported-test accuracy ≥ **0.9763** (0.9813 − 0.5pt) **and** macro-F1 ≥ **0.9671** (0.9721 − 0.5pt) |
| **G2 fresh classification** | fresh supported accuracy ≥ **0.5500** |
| **G3 fresh class coverage** | every class reported; fresh recall ≥ the Iteration-10 adaptation baseline **taken as the min over the two Iteration-10 seeds**, frozen here: recyclable ≥ **0.7143**, organic ≥ **0.3750**, hazardous ≥ **0.4000**, general trash ≥ **0.2727** |
| **G4 rejection preservation** | original AUROC ≥ **0.99466**, original OOD detection @0.0702 ≥ **0.99139**, original ID FRR ≤ **0.02284** (all = production ∓ 0.5pt); fresh OOD detection ≥ **0.66940** (production 0.6744 − 0.5pt); fresh AUROC ≥ **0.74770** (production 0.7527 − 0.5pt) |
| **G5 fresh FRR + isolation** | fresh FRR @0.0702 ≤ **0.33830** (production 0.3333 + 0.5pt) **and** the LOOP-12 isolation test passes: max abs reject-probability difference over fresh 131 + original test pools ≤ **1e-6** and **0** changed reject verdicts at 0.0702 |
| **G6 browser** | scratch TFJS export: 4-softmax + 1-sigmoid, label order, preprocessing, topology; Python↔TF.js parity atol ≤ **2e-5** on bins and reject (local **and** HTTP); headless UI click-through on the scratch page (production HTML, threshold unchanged) with zero console errors; `web/model/` untouched |
| **G7 reproducibility** | **both** seeds must satisfy G2 and every other gate numerically; both reported as mean ± spread; disagreement on G2 = Outcome B |

**G5 reference disclosure (pre-registered, before any measurement).**
Iteration 10's candidates re-calibrated the rejection head, which is what
produced their *lower* fresh FRR (0.1905 / 0.3016) — and also their *changed*
fresh OOD (0.6047 / 0.6977). Iteration 11 freezes that pathway on purpose, so
its fresh FRR is production's by construction. The gate is therefore
referenced to the production baseline (0.3333), and the Iteration-10 FRR
values are still shown in the comparison table as an observation. No
threshold is calibrated or changed in this iteration (production 0.0702 is
used verbatim).


## 7. Outcome vocabulary (decided by the numbers, not by preference)

* **A — Ship-eligible:** both seeds pass G1–G6 (⇒ G7 satisfied) → the
  seed-42 candidate may be promoted in the promotion step (§8).
* **B — Improvement-but-fail:** fresh supported accuracy ≥ **0.4600** for
  both seeds (production + ~4 images of the 63-image supported set) but at
  least one gate fails → production unchanged, improvement documented.
* **C — No meaningful improvement:** fresh supported accuracy < **0.4600**
  for either seed → the classification-head-only approach is rejected;
  Variant C stays production.

## 8. Promotion step (only under Outcome A)

Export the seed-42 candidate to `web/model/`, update the shipped alias,
re-run the full regression (LOOP 18) and commit as
"Ship classification-only domain adaptation". Threshold stays **0.0702**
(no calibration). Under B or C nothing in `web/` or `models/checkpoints/`
production paths changes.

## 9. What must not happen

No threshold tuning. No dataset changes (manifest md5 is asserted). No
checkpoint selection, no seed cherry-picking, no best-epoch selection. No
test-metric peeking during training. No `web/model/` or shipped-alias writes
before the gates are computed. No re-running with different hyperparameters
after seeing results — a follow-up becomes Iteration 12.

## 10. Evidence files

* `docs/rejection_experiment/head11s{42,43}_history.csv`
* `docs/rejection_experiment/head11s{42,43}_training_config.json`
* `docs/rejection_experiment/head11s{42,43}_freeze_proof.json`
* `docs/rejection_experiment/head11s{42,43}_eval.json`, `_gates.json`, `_report.md`
* `docs/rejection_experiment/head11_rejection_equivalence.json`
* `docs/rejection_experiment/head11_audit.json` (must equal the Iteration-10 audit up to `generated_utc`)
* `docs/rejection_experiment/head11_seed_consistency.json`, `iteration11_decision.md`

**Status: registered at baseline commit `ae1410f`, before training.**
