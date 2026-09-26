# WasteLens — Iteration 10: PRE-REGISTERED fresh-domain adaptation protocol

Written **before** any adaptation image is downloaded and before any
candidate metric is computed.

`src/build_adaptation_set.py` implements exactly this procedure; the
adaptation images never enter the repository (only
`docs/rejection_experiment/adaptation_set_manifest.json` — source URLs,
labels, licences, SHA-256 — is committed).

This keeps LOOP 6 (register before training) honest: the evaluation
below cannot move the protocol, because the protocol is committed first.

## 0. Relationship to Iteration 9 (non-contamination statement)

* `docs/rejection_experiment/fresh_eval_protocol.md` still defines the
  **final held-out benchmark** (131 images + 5 synthetic probes;
  locked Iteration-10 baseline: supported acc **0.3968**, macro-F1
  **0.2546**, AUROC **0.7527**, OOD detection **0.6744 @ 0.3333 FRR**,
  multi-object detection **0.4667**).
* The adaptation set in §1 **extends the same Wikimedia Commons source
  to NEW files only**. Every adaptation file is:
  1. taken from the same pre-registered categories, but at
     **rank offset ≥ 50 past every fresh file already selected**
     (guaranteed by title-sorted enumeration + skipping any title
     present in `fresh_set_manifest.json`),
  2. SHA-256 checked against **all 131 fresh files** (any hit is a
     leakage violation and aborts),
  3. SHA-256 checked against the **training corpus + CIFAR PNG cache**
     (same procedure as `src/build_fresh_set.py`).
* The final fresh benchmark files are **never opened, read, or used**
  in any training/validation path. §3 proves this mechanically:
  the audit asserts zero path/SHA-256 overlap between adaptation
  train/val lists and the 131 fresh files.

## 1. Adaptation categories (new files only)

| split role | Commons category | N new | label |
|---|---|---:|---|
| adapt-supported | Category:Food waste | 8 | organic |
| adapt-supported | Category:Compost | 8 | organic |
| adapt-supported | Category:Cardboard boxes | 8 | recyclable |
| adapt-supported | Category:Plastic bottles | 8 | recyclable |
| adapt-supported | Category:Lead-acid batteries | 6 | hazardous |
| adapt-supported | Category:Button cells | 6 | hazardous |
| adapt-supported | Category:Litter | 8 | general trash |
| adapt-supported | Category:Paper towels | 8 | general trash |
| adapt-ood | Category:Toy robots | 6 | unsupported |
| adapt-ood | Category:Computer keyboards | 6 | unsupported |
| adapt-ood | Category:Shirts | 6 | unsupported |
| adapt-ood | Category:Shoes | 6 | unsupported |
| adapt-ood | Category:Parks | 4 | unsupported (scene) |
| adapt-ood | Category:Streets | 4 | unsupported (scene) |
| adapt-multi | Category:Beach litter | 6 | unsupported (multi-object) |
| adapt-multi | Category:Flea markets | 6 | unsupported (multi-object) |

Expected totals: **adapt-supported 60** (organic 16 / recyclable 16 /
hazardous 12 / general 16), **adapt-ood 32**, **adapt-multi 12**
(**104 new files max**; a category that returns fewer usable files keeps
what exists, recorded in the manifest — never substituted).

All geometry, media-type, title-exclusion, thumbnail (1280px), and
provenance rules are inherited unchanged from
`docs/rejection_experiment/fresh_eval_protocol.md` §2.
## 2. Candidate training (the single controlled experiment)

* **Architecture:** exact Iteration-8 dual-head contract — MobileNetV2
  backbone frozen, `fc_256` + bins head + reject head trainable; labels,
  input size, preprocessing, four-bin order unchanged.
* **Warm start:** `models/checkpoints/wastelens_rej_shipped_best.keras`
  (Variant C seed-42 epoch-10), never the base single-head checkpoint.
* **Batch composition (the variable under test):** every training batch
  draws from a replayed mix —
  original Variant-C train pool **plus** the adaptation pool —
  with adaptation rows upsampled so the model sees fresh-domain signal
  without drowning the original distribution:
  * supported rows: original-train-supported : adapt-supported ≈ **3:1**,
  * unsupported rows: original-train-unsupported : adapt-{ood,multi} ≈ **3:1**,
  * bins-head inverse-frequency weights recomputed on the mixed
    supported-train pool (same formula as `src/train_rejection.py`),
  * rejection-head weights stay inverse-frequency on the mixed pool.
* **Hyperparameters:** Adam(1e-3), batch 32, **6 epochs** (shorter than
  the 10-epoch Iteration-8 budget because this is fine-tuning, not
  from-scratch head training), TF deterministic ops on, op determinism
  enabled best-effort.
* **Seeds:** primary **seed 42**; reproducibility **seed 43** uses the
  identical mix (a different shuffle/init seed only, same files).
* **Checkpoint rule (fixed-budget, exactly as Iteration 8):**
  candidate = **final epoch (= epoch 6)**. No early stopping, no
  best-epoch selection, no test-metric selection.
* **Artifacts (scratch / gitignored, never production paths):**
  `models/checkpoints/wastelens_rej_adapt10s{42,43}_*.keras` +
  `*_best.keras` alias = the final-epoch file (copied, not selected).

## 3. Validation design (leakage-proof by construction)

* **Original-benchmark validation:** unchanged Iteration-8 val pools
  (`eval_sets.json` supported_val + val_unsup) — used for the val gate
  and for freezing the candidate's own threshold.
* **Adaptation validation:** a deterministic 80/20 split of the
  adaptation pool (seed 42, stratified by role) used ONLY to monitor
  fine-tuning (val loss / adapt-supported accuracy); it is **not** used
  for checkpoint selection or threshold freezing.
* **Final test (untouched until evaluation):**
  original test pools **and** the full 131-image fresh benchmark.
* **Mechanical audit:** before training, `src/eval_adaptation.py`
  asserts zero path/SHA-256 overlap between
  `{adapt-train, adapt-val}` and `{fresh all 131, original test pools,
  CIFAR test batch}` and writes the audit into
  `adaptation_eval_audit.json`.

## 4. Pre-registered gates (no post-hoc moves)

Production threshold stays **0.0702** during the first candidate
evaluation (no threshold rescue — LOOP 12).

| Gate | Rule (computed in `src/fixed_budget_gate.py`) |
|---|---|
| G1 original classification | test acc ≥ **0.9763** (baseline 0.9813 − 0.5pt) **and** macro-F1 ≥ **0.9671** |
| G2 original rejection | AUROC ≥ **0.9950** and OOD detection @ production threshold ≥ **0.95** on original test pools |
| G3 fresh improvement | fresh supported accuracy ≥ **0.5500** (baseline 0.3968 + ~15pt) **and** improvement in ≥ 2 of the 3 collapsed bins (organic, hazardous, general) vs baseline |
| G4 no false-rejection blowup | fresh FRR @ 0.0702 ≤ **0.45** (baseline 0.3333 + ~12pt tolerance) |
| G5 seed consistency | **both** seeds satisfy G1–G4 independently; reported mean + spread; neither seed discarded |
| G6 product + browser | `bins` 4-softmax + `reject` sigmoid, labels order, preprocessing; parity atol ≤ 2e-5 (local + HTTP); decision suite + replay pass |

Promotion ships the **seed-42** candidate with its own val-frozen
threshold and the coordinated constant update; seed 43 is the
reproducibility witness, never the cherry-picked winner.

## 5. What must not happen

No training on any of the 131 fresh files. No threshold tuning to
rescue the candidate. No checkpoint substitution after results. No
production-path writes during experimentation. No claim beyond the
measured 63-image supported sample and the stated per-role breakdowns.
