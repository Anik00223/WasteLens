# WasteLens — Iteration 10 adaptation candidate, seed 43

Candidate: `models\checkpoints\wastelens_rej_adapt10s43_final.keras` (md5 `2006cfa04614961c801550784ca7bd0b`)
Baseline (re-measured in this run): `models\checkpoints\wastelens_rej_shipped_best.keras`
Evaluated: 2026-09-26 15:56:06 UTC | production threshold **0.0702** (no rescue, protocol §4)

## 0. Leakage audit (protocol §3)

adapt-train 75, adapt-val 19, fresh 131, original-test 3171
path hits **0**, SHA-256 hits vs fresh **0**, MD5 hits vs original test pools **0**, manifest integrity failures **0** → clean = **True**

## 1. Pre-registered gates (protocol §4)

| Gate | Rule | Measured | Pass |
|---|---|---|---|
| G1 | test acc >= 0.9763 and macro-F1 >= 0.9671 | `{'orig_accuracy': 0.9773, 'orig_macro_f1': 0.9672}` | **PASS** |
| G2 | AUROC >= 0.995 and OOD detect >= 0.95 @ 0.0702 | `{'auroc': 0.9997, 'ood_detect': 0.9974}` | **PASS** |
| G3 | fresh supported acc >= 0.55 and >= 2 collapse bins improved | `{'fresh_accuracy': 0.5397, 'fresh_macro_f1': 0.5152, 'bins_improved': 3}` | **FAIL** |
| G4 | fresh FRR <= 0.45 @ 0.0702 | `{'fresh_frr': 0.3016}` | **PASS** |

All G1–G4 pass: **False**

## 2. Original benchmark (test pools)

| Metric | Candidate | Baseline (locked) | In-run baseline |
|---|---|---|---|
| accuracy | 0.9773 | 0.9813 | 0.9813 |
| macro-F1 | 0.9672 | 0.9721 | 0.9721 |
| rejection AUROC | 0.99970 | 0.99966 | 0.99966 |
| OOD detect @0.0702 | 0.9974 | n/a | 0.9964 |
| ID FRR @0.0702 | 0.0203 | n/a | 0.0178 |

Per-source OOD detection @0.0702 (candidate): {"clothes": 1.0, "shoes": 0.9899, "collage": 0.9889, "nonwaste": 0.9987}

## 3. Fresh benchmark (131 files, untouched)

| Metric | Candidate | Locked baseline | In-run baseline |
|---|---|---|---|
| supported acc (63) | 0.5397 | 0.3968 | 0.3968 |
| macro-F1 | 0.5152 | 0.2546 | 0.2546 |
| FRR @0.0702 | 0.3016 | 0.3333 | 0.3333 |
| OOD detect @0.0702 | 0.6977 | 0.6744 | 0.6744 |
| multi-object detect | 0.6000 | 0.4667 | 0.4667 |

### Per-bin recall (fresh supported)

| Bin | Candidate | Locked baseline | Delta |
|---|---|---|---|
| recyclable | 0.7143 | 0.9524 | -0.2381 |
| organic | 0.5625 | 0.0625 | +0.5000 |
| hazardous | 0.4667 | 0.2667 | +0.2000 |
| general trash | 0.2727 | 0.0000 | +0.2727 |

### Fresh per-role rejection @0.0702

| Role | n | detected | rate |
|---|---|---|---|
| ood-objects | 15 | 12 | 0.8000 |
| ood-clothes | 8 | 6 | 0.7500 |
| ood-shoes | 8 | 5 | 0.6250 |
| ood-scenes | 12 | 7 | 0.5833 |

### Ambiguous supported items

{"count": 10, "verdicts": {"supported": 3, "uncertain": 2, "unsupported": 5}, "mean_top_prob": 0.8607996046543122, "mean_margin": 0.759014225117167}

