# WasteLens — Iteration 10 adaptation candidate, seed 42

Candidate: `models\checkpoints\wastelens_rej_adapt10s42_final.keras` (md5 `260586f7640230f5d78f461e61f414dd`)
Baseline (re-measured in this run): `models\checkpoints\wastelens_rej_shipped_best.keras`
Evaluated: 2026-09-26 15:31:28 UTC | production threshold **0.0702** (no rescue, protocol §4)

## 0. Leakage audit (protocol §3)

adapt-train 75, adapt-val 19, fresh 131, original-test 3171
path hits **0**, SHA-256 hits vs fresh **0**, MD5 hits vs original test pools **0**, manifest integrity failures **0** → clean = **True**

## 1. Pre-registered gates (protocol §4)

| Gate | Rule | Measured | Pass |
|---|---|---|---|
| G1 | test acc >= 0.9763 and macro-F1 >= 0.9671 | `{'orig_accuracy': 0.9813, 'orig_macro_f1': 0.9736}` | **PASS** |
| G2 | AUROC >= 0.995 and OOD detect >= 0.95 @ 0.0702 | `{'auroc': 0.9997, 'ood_detect': 0.9964}` | **PASS** |
| G3 | fresh supported acc >= 0.55 and >= 2 collapse bins improved | `{'fresh_accuracy': 0.5238, 'fresh_macro_f1': 0.4832, 'bins_improved': 3}` | **FAIL** |
| G4 | fresh FRR <= 0.45 @ 0.0702 | `{'fresh_frr': 0.1905}` | **PASS** |

All G1–G4 pass: **False**

## 2. Original benchmark (test pools)

| Metric | Candidate | Baseline (locked) | In-run baseline |
|---|---|---|---|
| accuracy | 0.9813 | 0.9813 | 0.9813 |
| macro-F1 | 0.9736 | 0.9721 | 0.9721 |
| rejection AUROC | 0.99969 | 0.99966 | 0.99966 |
| OOD detect @0.0702 | 0.9964 | n/a | 0.9964 |
| ID FRR @0.0702 | 0.0154 | n/a | 0.0178 |

Per-source OOD detection @0.0702 (candidate): {"clothes": 0.9988, "shoes": 0.9933, "collage": 0.9667, "nonwaste": 0.9987}

## 3. Fresh benchmark (131 files, untouched)

| Metric | Candidate | Locked baseline | In-run baseline |
|---|---|---|---|
| supported acc (63) | 0.5238 | 0.3968 | 0.3968 |
| macro-F1 | 0.4832 | 0.2546 | 0.2546 |
| FRR @0.0702 | 0.1905 | 0.3333 | 0.3333 |
| OOD detect @0.0702 | 0.6047 | 0.6744 | 0.6744 |
| multi-object detect | 0.6000 | 0.4667 | 0.4667 |

### Per-bin recall (fresh supported)

| Bin | Candidate | Locked baseline | Delta |
|---|---|---|---|
| recyclable | 0.8571 | 0.9524 | -0.0952 |
| organic | 0.3750 | 0.0625 | +0.3125 |
| hazardous | 0.4000 | 0.2667 | +0.1333 |
| general trash | 0.2727 | 0.0000 | +0.2727 |

### Fresh per-role rejection @0.0702

| Role | n | detected | rate |
|---|---|---|---|
| ood-objects | 15 | 10 | 0.6667 |
| ood-clothes | 8 | 6 | 0.7500 |
| ood-shoes | 8 | 4 | 0.5000 |
| ood-scenes | 12 | 6 | 0.5000 |

### Ambiguous supported items

{"count": 10, "verdicts": {"supported": 7, "uncertain": 1, "unsupported": 2}, "mean_top_prob": 0.9434519648551941, "mean_margin": 0.9022999141173116}

