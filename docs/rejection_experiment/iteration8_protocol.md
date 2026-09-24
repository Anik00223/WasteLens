# WasteLens — Iteration 8 PRE-REGISTERED PROTOCOL
## Fixed-budget final-epoch Variant C (seed 42 + seed 43 reproduction)

Registered before any Iteration-8 training run. Machine-readable constants live
in `src/fixed_budget_gate.py` (single source of truth); this document is the
text of record. **Nothing here may be edited after training starts.**

---

## 1. Objective

Test whether Variant C's strong rejection behaviour survives when checkpoint
selection is **completely removed**: train a fixed 10-epoch budget and take
**epoch 10** exactly, on two seeds, and apply a pre-registered gate. Iterations
6–7 showed the Variant C objective reaches collage detection ≈98% but that the
accuracy outcome depended heavily on which epoch was picked (and, it turns out,
on an unseeded weight initialization). This iteration removes both sources of
freedom.

## 2. Frozen elements (unchanged from Iterations 6–7)

| element | value |
|---|---|
| architecture | MobileNetV2 backbone frozen; `fc_256`, 4-bin head, reject head trainable |
| loss weights | bins **2.0** / reject **1.0** |
| optimizer / LR / batch | Adam(1e-3) / 1e-3 / 32 |
| epochs | **10 (fixed budget, no early stopping)** |
| labels, input size, preprocessing | unchanged (`web/index.html` LABELS order) |
| dataset | `train_rejection.build_datasets()`, **data seed 42**: same supported split, clothes/shoes (md5-deduped), CIFAR-10 non-waste, collages from disjoint train pools, 5 synthetic probes; test splits never trained on |
| base checkpoint | `models/checkpoints/wastelens_ep09.keras` |

The dataset seed stays 42 for **both** seeds so every run sees byte-identical
data; the seed variable (42 / 43) controls **model initialization, dropout and
training-time shuffling only**.

## 3. Selection rule — none

* fixed budget: **10 epochs**
* candidate checkpoint: **epoch 10**
* no early stopping
* no best-epoch selection, no "best val_loss", no manual pick
* no selection using test metrics (test is not touched until the candidate is
  frozen)
* no selection using later validation observations

## 4. Reproducibility requirement (LOOP 3)

* `TF_DETERMINISTIC_OPS=1` set before TensorFlow import
* `tf.config.experimental.enable_op_determinism()` at import time
* `tf.keras.utils.set_random_seed(seed)` called **before** model construction
  (Iteration 6/7 seeded after construction, which left the weight init
  unseeded — this is the technical adjustment the reproducibility requirement
  demands and the documented cause of the Iteration-6/7 run-to-run spread)
* seed 42 is the registered primary run; seed 43 is a **reproducibility check**
  under the identical protocol, never a "pick the better one" option

## 5. Gate constants (registered, computed in code)

| constant | value |
|---|---|
| `REGISTERED_BASELINE_VAL_ACC` | 0.9829 (Variant B supported-val, Iter-7 measurement) |
| `REGISTERED_TOLERANCE_PP` | 0.25 percentage points |
| `MIN_VAL_ACC` | 0.9829 − 0.0025 = **0.9804** |
| `BASELINE_TEST` | accuracy 0.9822, macro-F1 0.9742 |
| `TEST_ACC_TOLERANCE` / `TEST_F1_TOLERANCE` | 0.005 / 0.005 (the project's existing 0.5pt tolerance) |
| `MIN_OOD_AT_1PCT` | 0.90 (Variant B: 0.8426) |
| `MIN_COLLAGE_AT_5PCT` | 0.75 (Variant B: 0.5778) |
| `EPOCH_BUDGET` / `CANDIDATE_EPOCH` | 10 / 10 |

**Validation gate (per seed):** `val_bins_acc >= 0.9804`. If a seed fails, that
seed fails — no epoch substitution, no retry with another seed.

**Test-side conditions:** accuracy ≥ 0.9822 − 0.005 **and** macro-F1 ≥
0.9742 − 0.005; OOD @1% FRR ≥ 0.90; collage @5% FRR ≥ 0.75.

**Reproducibility condition:** **both** seeds must clear the validation floor.
Test numbers are reported as mean ± spread; rejection is reported per seed and
as mean. A weak seed is never discarded and the stronger seed is never shipped
alone.

## 6. Measurement protocol (unchanged from Variants A/B/C)

* classification: supported test split (1,147 images), accuracy, macro-F1,
  per-bin P/R/F1, confusion matrix
* rejection: AUROC; thresholds frozen on **supported validation only** at the
  1/5/10/15/20% FRR operating points (same rule as Variants A/B), applied to
  the untouched test set; test FRR reported alongside
* OOD sources reported separately: clothes, shoes, non-waste, collages,
  synthetic probes (collage never folded into the aggregate)
* candidate's own calibrated threshold; `0.8774` (Variant B) is **not** reused
  for Variant C

## 7. Decision matrix (registered)

### Outcome A — ship
Both seeds clear the validation floor **and** the test-side conditions pass for
both seeds. Then: promote the protocol-defined epoch-10 candidate, freeze its
validation-derived threshold, update production constants together
(`web/index.html` threshold, `web/model/`, `src/product_test_outputs.py`,
tests), rerun full production verification (parity, replay, UI, regression).

### Outcome B — reject
A seed fails the validation floor or the test-side conditions → Variant B stays
production; Variant C evidence is preserved; production files untouched.

### Outcome C — promising but inconclusive
Classification passes for both seeds but rejection behaviour is unstable across
seeds (e.g. one seed's rejection is materially weaker) → do not ship; document
the instability; retain Variant B.

No criterion in this document may be changed after seeing results, and no
production file may be modified before the gate passes.

## 8. Pipeline validation performed before registration

Pre-training checks use **smoke / reduced-epoch probes only** (no test-set
access, no candidate selection): syntax compilation, a 1-epoch probe to confirm
the deterministic pipeline runs and to size the runtime, and an artifact-tag
probe to confirm run isolation. Probe artifacts are deleted and are not
evidence.

Signed off in the repository in the commit that introduced this file
(`git log --follow docs/rejection_experiment/iteration8_protocol.md`), which
was created before the seed-42 training was launched. The results commit that
follows records both hashes.
