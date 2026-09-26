# WasteLens — Iteration 10 decision: fresh-domain adaptation

Protocol: `docs/rejection_experiment/iteration10_adaptation_protocol.md` (registered before any adaptation image was downloaded).
Generated: 2026-09-26 15:57:44 UTC

## 1. Candidates and seeds

| Seed | Candidate | md5 | G1 | G2 | G3 | G4 | G1–G4 |
|---|---|---|---|---|---|---|---|
| 42 | `wastelens_rej_adapt10s42_final.keras` | `260586f76402` | PASS | PASS | FAIL | PASS | FAIL |
| 43 | `wastelens_rej_adapt10s43_final.keras` | `2006cfa04614` | PASS | PASS | FAIL | PASS | FAIL |

## 2. Measured metrics (both seeds, mean ± spread)

| Metric | Mean | Spread | Seed 42 | Seed 43 |
|---|---|---|---|---|
| orig_accuracy | 0.9793 | 0.0041 | 0.9813 | 0.9773 |
| orig_macro_f1 | 0.9704 | 0.0064 | 0.9736 | 0.9672 |
| orig_auroc | 0.9997 | 0.0000 | 0.9997 | 0.9997 |
| orig_ood_detect | 0.9969 | 0.0010 | 0.9964 | 0.9974 |
| fresh_accuracy | 0.5317 | 0.0159 | 0.5238 | 0.5397 |
| fresh_macro_f1 | 0.4992 | 0.0319 | 0.4832 | 0.5152 |
| fresh_frr | 0.2460 | 0.1111 | 0.1905 | 0.3016 |

### Fresh per-bin recall (`organic`, `hazardous`, `general trash` are the Iteration-9 collapse bins)

| Seed | bins improved | fresh acc | fresh FRR |
|---|---|---|---|
| 42 | 3/3 | 0.5238 | 0.1905 |
| 43 | 3/3 | 0.5397 | 0.3016 |

## 3. Decision (protocol §4 vocabulary, applied mechanically)

**Improvement-but-fail**

Fresh-domain accuracy improved, but at least one gate failed: no promotion - the protocol is not renegotiated after the measurement.

- G5 seed consistency: both seeds satisfy G1–G4 = **False**; neither seed was discarded or swapped.
- G6 browser parity: separate step: TFJS export to scratch + browser_preprocess_probe.js parity (atol <= 2e-5)

## 4. Interpretation (computed from the measured numbers)

- Fresh supported accuracy **0.5238** vs locked baseline **0.3968** (**+12.70pt**), macro-F1 0.2546 → 0.4832.
- Fresh false-rejection rate 0.3333 → 0.1905; multi-object detection 0.4667 → 0.6000; fresh OOD detection 0.6744 → 0.6047.
- Collapse-bin recalls: organic 0.0625 → 0.3750 (+31.25pt), hazardous 0.2667 → 0.4000 (+13.33pt), general trash 0.0000 → 0.2727 (+27.27pt).
- Original-domain cost: accuracy 0.9813 → 0.9813, macro-F1 0.9721 → 0.9736, AUROC 0.99966 → 0.99969 (G1/G2 hold: the fix did not buy fresh-domain accuracy with original-domain performance).
- Read-out: controlled fresh-domain adaptation is the first measured intervention that moves real-world accuracy on the untouched benchmark; it misses only the pre-registered G3 accuracy bar, so the honest outcome is Improvement-but-fail, not Ship.

## 5. Production state

- Shipped model unchanged: `models/checkpoints/wastelens_rej_shipped_best.keras` / `web/model/`.
  Verified at summary time: md5 `8735019226412c4d67b0a17269dc38ad` — byte-identical to the Iteration-8 release file (`wastelens_rej_varc8s42_10.keras`); `web/model/` was never rewritten.
- Production rejection threshold unchanged: **0.0702**.
- No threshold tuning, no checkpoint substitution: the gates were computed with the pre-registered constants only.
- Adaptation images never entered the repository; the leakage audit is `docs/rejection_experiment/adaptation_eval_audit.json` (zero path/SHA-256/MD5 hits).


## 6. Evidence files

- `docs/rejection_experiment/adapt10s42_eval.json`, `adapt10s42_gates.json`, `adapt10s42_report.md`, `adapt10s42_history.csv`, `adapt10s42_training_config.json`
- `docs/rejection_experiment/adapt10s43_eval.json`, `adapt10s43_gates.json`, `adapt10s43_report.md`, `adapt10s43_history.csv`, `adapt10s43_training_config.json`
- `docs/rejection_experiment/adaptation_eval_audit.json`, `adapt10_seed_consistency.json`
