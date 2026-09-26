# Iteration 12 — Distillation-Constrained Fresh-Domain Adaptation

**Status: registered BEFORE any Iteration-12 training (protocol file written
first; no Iteration-12 checkpoint exists at registration time).**

Baseline commit: `f970680` (Iteration 11 evaluated: Outcome B —
fresh accuracy 59.52% mean, original accuracy fell to ~96.1% mean).

Production remains the Iteration-8 Variant C model:
`models/checkpoints/wastelens_rej_shipped_best.keras`
(md5 `8735019226412c4d67b0a17269dc38ad`), `REJECT_THRESHOLD = 0.0702`.

---

## 1. The one experimental change (LOOP 2)

Iteration 11 showed head-only adaptation moves the fresh benchmark from
39.68% → 59.52% but drags the original test accuracy from 98.13% → ~96.1%
(mean of two seeds). Iteration 12 adds **exactly one** thing to the
Iteration-11 recipe:

> **production-model distillation (knowledge distillation from the frozen
> shipped teacher) added to the classification adaptation objective.**

Everything else is byte-for-byte the Iteration-11 recipe:

| Item | Value (unchanged unless noted) |
|---|---|
| seeds | 42 and 43 (both reported, no cherry-picking) |
| epochs | 6; candidate = **FINAL epoch** (no selection of any kind) |
| dataset | Iteration-11 adaptation set (manifest md5 `acae0933aca123c835c1e55396af182a`, 94 files) and fresh benchmark (md5 `99540f56efb95e2c26b2b6cc9414ff5d`, 131 images) — untouched |
| split | same deterministic seed-42 70/15/15 stratified original split + same adaptation train/val split (`train_adaptation.split_adapt`) |
| mix | same seed-independent 3:1 original:adapt replay (`build_mixed_arrays`) with the same class/rejection weights |
| optimizer | Adam(learning_rate = 1e-3) |
| batch size | 32 (`train.BATCH_SIZE`) |
| trainable set | `predictions/kernel` + `predictions/bias` only (1,028 params) |
| backbone / trunk | frozen MobileNetV2 incl. BatchNorm moving statistics, gap, dropouts, `fc_256` |
| rejection head | **frozen** (`reject/kernel`, `reject/bias`) — never updated, never distilled |
| loss weights | bins 2.0, reject 1.0 |
| threshold | 0.0702, no calibration |
| determinism | `TF_DETERMINISTIC_OPS=1`, `enable_op_determinism()`, reseed before fit (Iteration-8/11 pattern) |

**Prohibitions:** no threshold tuning; no dataset changes; no checkpoint or
seed selection; no λ/temperature change after training starts; no test-set
peeking; no `web/model/` or shipped-alias writes before gates are computed;
no post-hoc gate moves — a follow-up becomes Iteration 13.

## 2. Teacher and student (LOOP 3)

* **Teacher**: the shipped Variant C checkpoint, loaded fresh, then
  `layer.trainable = False` on every layer; zero trainable tensors; md5 must
  equal `8735019226412c4d67b0a17269dc38ad`. The teacher is NEVER optimized
  (it is outside the student's optimizer entirely).
* **Student**: warm start from the SAME shipped checkpoint with the
  Iteration-11 trainable set (bin head only). Student and teacher therefore
  start as the same function; any later divergence is attributable to
  training alone.
* **Not distilled**: the rejection output. The reject head and its whole
  input trunk are frozen, so the rejection pathway is constant by
  construction (Iteration 11 verified max|Δreject| = 0 exactly).
* Teacher forward pass: `training=False` (inference mode, dropout off).
  Student forward pass: `training=True` (exactly as in Iteration 11).
  Teacher outputs are wrapped in `stop_gradient`.

## 3. Distillation objective — exact formulation (LOOP 4)

Registered constants, fixed before training and never changed:

```
λ = 0.5        (distillation weight)
T = 2          (distillation temperature)
```

For an image `x` with student bin probabilities `p_s = softmax head(x)` (4
classes) and teacher bin probabilities `p_t`:

```
q(p, T)   = p^(1/T) / Σ_j p_j^(1/T)                  (temperature distribution)
```

Because the head is a softmax, `log p = z − logsumexp(z)` for logits `z`, so
`softmax(log p / T) = softmax(z / T)`: the temperature distribution computed
from probabilities is EXACTLY the Hinton temperature-softmax of the logits
(the per-row constant cancels inside softmax). Probabilities are clipped to
`[1e-12, 1]` before the power/log.

```
KL_i      = Σ_j q(p_t,i, T) · ( log q(p_t,i, T) − log q(p_s,i, T) )   ≥ 0
L_kd      = T² · Σ_i m_i·KL_i / (Σ_i m_i + ε)          (T² = 4, Hinton scaling)
m_i       = 1 if image i is an ORIGINAL-domain training row
            (original train-split supported + original unsup: clothes, shoes,
             collage, nonwaste), else 0
L_ce      = 2.0 · SWCE(bins) + 1.0 · BCE(reject)        (Iteration-11 terms,
            sample-weighted exactly as compiled; the reject term contributes
            zero gradient because reject head + trunk are frozen)
L_total   = L_ce + λ · L_kd                             (λ = 0.5, T = 2)
```

Notes, registered up front:

* Distillation applies **only to original-domain rows** (LOOP 3): the fresh
  (adaptation) rows keep plain ground-truth CE — distilling the teacher on
  fresh rows would fight the very adaptation being tested.
* Original unsupported rows (sw_bins = 0) still receive the KD term: they
  carry no ground truth, but anchoring the teacher's behaviour there is part
  of "do not move the original boundary".
* At initialization KL ≈ 0 (student ≡ teacher modulo dropout noise in the
  student's training-mode forward); the term only bites as the student
  diverges on original images.
* The optimizer still updates ONLY the 1,028 bin-head parameters; the KD
  gradient flows through the student's bin head like the CE gradient.

## 4. Warm-start / freeze proof (LOOP 5)

Before training, per seed, recorded to `head12s{SEED}_freeze_proof.json`:

1. student checkpoint md5 == shipped md5 (production constant),
2. teacher checkpoint md5 == shipped md5 and teacher trainable tensors == 0,
3. trainable set is exactly `["predictions/bias", "predictions/kernel"]`
   (1,028 params); all other tensors non-trainable (264 frozen tensors,
   SHA-256 per tensor + aggregate),
4. bins **and** reject outputs bit-exact (max abs diff 0 expected) on 16 real
   original-train images vs the shipped model (student = teacher = shipped),
5. after training the same 264 digests must be byte-identical (only
   `predictions/{kernel,bias}` may differ); reject output drift must be
   ≤ 1e-6 (expected 0.0); any frozen-tensor change aborts the run.

## 5. Dataset (LOOP 6)

The Iteration-11 dataset, asserted before the model loads
(`train_distill_adaptation.lock_dataset`):

* adaptation manifest md5 `acae0933aca123c835c1e55396af182a`, 94 files,
* fresh manifest md5 `99540f56efb95e2c26b2b6cc9414ff5d` (provenance only —
  training never opens a fresh file),
* original split/eval sets: `eval_sets.json` + `adaptation_eval_audit.json`
  re-audited by the evaluator and required identical to the committed
  Iteration-10 audit except `generated_utc` (§3-style drift check).

No fresh/original/unsupported example may be added, removed or relabelled.

## 6. Training (LOOP 7)

* Seed 42 first: exactly 6 epochs, then seed 43 with the identical command
  except `--seed 43`. Candidate = epoch 6 file, copied to
  `models/checkpoints/wastelens_rej_head12s{SEED}_final.keras`
  (scratch checkpoints only; `_best` copy is a mechanical copy of epoch 6,
  never a selection).
* Recorded per seed: history CSV with `loss`, `kd_loss`, bins/reject
  components, train/val metrics, per-epoch adaptation-val monitoring,
  trainable parameter count, full config JSON (λ, T, formulation, dataset
  locks, determinism), checkpoint md5s, freeze proofs, wall time.
* No early stopping, no LR schedule (constant 1e-3), no batch changes.

## 7. Pre-registered gates (LOOP 12) — computed mechanically

| Gate | Rule (literal) |
|---|---|
| **G1 original accuracy** | original test accuracy ≥ **0.9763** (production 0.9813 − 0.5pt) |
| **G2 original macro-F1** | original test macro-F1 ≥ **0.9671** (0.9721 − 0.5pt) |
| **G3 fresh supported accuracy** | ≥ **0.5500** on the untouched 63 supported fresh images |
| **G4 fresh class coverage** | per-class fresh recall ≥ the Iteration-11 two-seed minimum minus 5pt ("material collapse" = >5pt drop): recyclable ≥ **0.7119**, organic ≥ **0.3875**, hazardous ≥ **0.4833**, general trash ≥ **0.4045** |
| **G5 rejection preservation** | original AUROC ≥ **0.99466**, original OOD detect @0.0702 ≥ **0.99139**, original ID FRR ≤ **0.02284**, fresh FRR @0.0702 ≤ **0.33830**, fresh AUROC ≥ **0.74770**, AND the isolation test: max abs reject-prob difference over fresh 131 + original test pools ≤ **1e-6**, **0** changed reject verdicts, frozen digests byte-identical in the saved file |
| **G6 fresh OOD** | fresh OOD detection @0.0702 ≥ **0.66940** (production 0.6744 − 0.5pt) |
| **G7 reproducibility** | **both** seeds satisfy G1–G4 (core classification gates); disagreement on G3 = automatic fail of G7 |
| **G8 browser** | scratch TFJS export (4-softmax + 1-sigmoid, label order `recyclable, organic, hazardous, general trash`, topology) + Python↔TF.js parity atol ≤ **2e-5** on bins and reject, **local and HTTP** |

Measured but not gated (reported honestly): ambiguous-item behaviour,
multi-object detection, non-waste/clothes/shoes/scenes OOD roles, teacher–
student divergence on original images, fresh per-class P/R/F1 + confusion,
original confusion matrix, seed mean ± spread.

## 8. Outcome vocabulary (LOOP 20)

* **A — Pass (promotion-eligible):** G1–G8 all pass for both seeds
  (fresh ≥ 55%, original within tolerance, fresh OOD preserved, rejection
  identical, browser parity) → eligible for the promotion step (production
  swap remains a separate, explicitly verified action).
* **B — Improvement but fail:** fresh accuracy ≥ **0.4600** for both seeds
  but at least one gate fails → keep production unchanged, document.
* **C — No meaningful improvement:** fresh accuracy < **0.4600** for either
  seed → reject the distillation approach, keep production unchanged.

## 9. Evidence files (LOOPs 8–19)

* `docs/rejection_experiment/iteration12_distillation_protocol.md` (this file)
* `head12s{42,43}_history.csv`, `_training_config.json`, `_freeze_proof.json`
* `head12s{42,43}_eval.json`, `_gates.json`, `_report.md`
* `head12_rejection_equivalence.json`, `head12_audit.json`
* `head12_browser.json` (export + local/HTTP parity evidence)
* `ui_clickthrough_head12s42.json` (+ desktop/mobile screenshots)
* `head12_seed_consistency.json`, `iteration12_decision.md`
  (includes the Production vs Iter-10 vs Iter-11 vs Iter-12 comparison table)

## 10. Git (LOOPs 21–24)

Review (`git status`, `git diff --stat`, `git diff`) before committing:
no raw images, no temp checkpoints, no scratch exports, no logs, no caches,
no `web/model/` modifications. Commit message: **"Evaluate
distillation-constrained domain adaptation"** (or **"Ship
distillation-constrained domain adaptation"** only under Outcome A after a
full passing gate), then `git push origin main` and verify
local ≡ origin/main with a clean tree.

**Status: registered before training; λ = 0.5 and T = 2 are immutable from
this point.**

