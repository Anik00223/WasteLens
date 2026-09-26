# Iteration 11 decision - classification-head-only adaptation

- decision: **B - Improvement-but-fail**
- reason: fresh supported accuracy >= 0.46 for both seeds (meaningful improvement) but at least one gate fails -> production unchanged, improvement documented; the protocol is not renegotiated after the measurement.
- protocol: `docs/rejection_experiment/iteration11_head_adaptation_protocol.md` §7 vocabulary, decided by the numbers

| Seed | G1 | G2 | G3 | G4 | G5 | G6 | fresh acc | fresh macro-F1 | fresh FRR |
|---|---|---|---|---|---|---|---|---|---|
| 42 | FAIL | PASS | PASS | PASS | PASS | PASS | 0.6032 | 0.5733 | 0.33333 |
| 43 | FAIL | PASS | PASS | PASS | PASS | PASS | 0.5873 | 0.5653 | 0.33333 |

## Seed consistency (G7)

- orig_accuracy: mean 0.9611 ± 0.0065 ({"42": 0.9643146796431468, "43": 0.9578264395782644})
- orig_macro_f1: mean 0.9449 ± 0.0083 ({"42": 0.949099813230248, "43": 0.9407613554698019})
- orig_auroc: mean 0.9997 ± 0.0000 ({"42": 0.9996601876333407, "43": 0.9996601876333407})
- fresh_accuracy: mean 0.5952 ± 0.0159 ({"42": 0.6031746031746031, "43": 0.5873015873015873})
- fresh_macro_f1: mean 0.5693 ± 0.0080 ({"42": 0.5733030990173847, "43": 0.5652591973244147})
- fresh_frr: mean 0.3333 ± 0.0000 ({"42": 0.3333333333333333, "43": 0.3333333333333333})

- both seeds pass G1-G6: **False**; G6 status: scored; G2 disagreement: False

## Iteration-10 / production reference (observations)

- Iteration-10 fresh accuracy: {"42": 0.5238095238095238, "43": 0.5396825396825397} (bar for Outcome B: 0.46)
- Iteration-10 fresh FRR: {"42": 0.19047619047619047, "43": 0.30158730158730157} - lower than production because Iteration 10 re-calibrated the reject head; Iteration 11 freezes it by design
- production fresh FRR 0.3333 (bar 0.3383), threshold 0.0702 unchanged

## Production state

- shipped checkpoint md5: `8735019226412c4d67b0a17269dc38ad`
- web/model rewritten: False
- no promotion unless Outcome A triggers §8; production artifacts and the 0.0702 threshold are read-only throughout Iteration 11

