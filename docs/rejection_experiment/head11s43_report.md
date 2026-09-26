# Iteration 11 - classification-head-only adaptation - seed 43

- protocol: `iteration11_head_adaptation_protocol.md` (registered at baseline `ae1410f`, protocol committed pre-training `19c30bc`)
- candidate: `models\checkpoints\wastelens_rej_head11s43_final.keras` (md5 `f794062272d5d1dc02f6a55c94d522f8`)
- warm start: shipped checkpoint - only `predictions/kernel` + `predictions/bias` (1,028 params) trainable; 264 frozen tensors
- threshold: **0.0702** unchanged (no calibration)
- dataset audit identical to Iteration-10 (except timestamp): **True**

## 1. Gates (pre-registered §6)

| Gate | Measured | Rule | Pass |
|---|---|---|---|
| G1 existing | acc 0.9578, macro-F1 0.9408 | acc >= 0.9763 and F1 >= 0.9671 | FAIL |
| G2 fresh | acc 0.5873 | >= 0.5500 | PASS |
| G4 rejection | orig AUROC 0.99966, orig OOD 0.99639, orig ID FRR 0.01784, fresh OOD 0.67442, fresh AUROC 0.75268 | production ∓0.5pt | PASS |
| G5 FRR + isolation | fresh FRR 0.33333, max |Δreject| 0.000e+00, flips 0 | FRR <= 0.33830 AND max|Δ| <= 1e-6 AND 0 flips AND frozen file identical | PASS |
| G6 browser/parity | scored | export + atol<=2e-5 parity + UI click-through | PASS |
| G7 two-seed | decided in --summary | both seeds pass | PENDING |

### G3 fresh class coverage

| Bin | Fresh recall | Bar (Iter-10 min) | In-run baseline | Pass |
|---|---|---|---|---|
| recyclable | 0.7619 | 0.7143 | 0.9524 | PASS |
| organic | 0.5000 | 0.3750 | 0.0625 | PASS |
| hazardous | 0.5333 | 0.4000 | 0.2667 | PASS |
| general trash | 0.4545 | 0.2727 | 0.0000 | PASS |

**G3 overall: PASS**; G1-G5 composite: **FAIL**

## 2. LOOP-12 rejection isolation

- fresh 131 one-at-a-time: max |Δreject| = 0.000e+00, bit-exact = True, reject verdict flips = 0
- original pools: supported_test: max|d| 0.000e+00, flips 0; unsup_clothes: max|d| 0.000e+00, flips 0; unsup_shoes: max|d| 0.000e+00, flips 0; unsup_collage: max|d| 0.000e+00, flips 0; unsup_nonwaste: max|d| 0.000e+00, flips 0 (overall incl. fresh: max|d| 0.000e+00, flips 0)
- file-level frozen digests identical: True (264 tensors)
- bins argmax changes on fresh (expected - this is the trained head): 80; three-way verdict changes: 23
- **pass: True**

## 3. Existing + fresh benchmark numbers

| Metric | Candidate | In-run shipped baseline | Locked |
|---|---|---|---|
| orig accuracy | 0.9578 | 0.9813 | 0.9813 |
| orig macro-F1 | 0.9408 | 0.9721 | 0.9721 |
| orig AUROC | 0.99966 | 0.99966 | 0.99966 |
| orig OOD detect @0.0702 | 0.99639 | 0.99639 | - |
| fresh supported acc (63) | 0.5873 | 0.3968 | 0.3968 |
| fresh macro-F1 | 0.5653 | 0.2546 | 0.2546 |
| fresh AUROC | 0.75268 | 0.75268 | - |
| fresh FRR @0.0702 | 0.33333 | 0.33333 | 0.33333 |
| fresh OOD detection | 0.67442 | 0.67442 | 0.67442 |

Fresh accuracy deltas vs Iteration 10 (seed: delta): 42: +0.0635, 43: +0.0476 (Iteration 10 scored 0.5238 / 0.5397; outcome B bar 0.4600).

## 4. Notes

- The reject head is frozen, so rejection metrics are production's by construction; the in-run baseline column re-measures the shipped model in the same process to prove the harness did not move.
- Threshold 0.0702 is untouched; no checkpoint selection: the candidate is the FINAL epoch.

