# WasteLens — Fresh Real-World Evaluation Report (Iteration 9)

**Evaluation Date:** 2026-09-25 17:06:30 UTC  
**Model Under Test:** `models\checkpoints\wastelens_rej_shipped_best.keras` (Shipped Variant C, Seed 42, Epoch 10)  
**Shipped Rejection Threshold:** `0.0702`  
**Dataset:** Fresh 131 real-world images from untouched classes (0 overlap with train/val/test)

---

## 1. Classification Performance (Supported Single Items)

Evaluated on **63** fresh supported single-item photographs across all 4 bins.

* **Accuracy:** `0.3968` (25/63)
* **Macro-F1:** `0.2546`

### Per-Bin Performance
| Bin | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| **recyclable** | 0.3636 | 0.9524 | 0.5263 | 21 |
| **organic** | 0.5000 | 0.0625 | 0.1111 | 16 |
| **hazardous** | 0.6667 | 0.2667 | 0.3810 | 15 |
| **general trash** | 0.0000 | 0.0000 | 0.0000 | 11 |

### Confusion Matrix
(Rows = ground truth, Columns = predicted: recyclable, organic, hazardous, general trash)
```
[[20, 0, 1, 0], [15, 1, 0, 0], [10, 1, 4, 0], [10, 0, 1, 0]]
```

---

## 2. Rejection Performance at Shipped Threshold (0.0702)

* **Rejection AUROC:** `0.7527`
* **Rejection AUPRC:** `0.6717`
* **False Rejection Rate (FRR) on Supported Waste:** `33.33%` (21/63)
* **Overall OOD Detection Rate:** `67.44%` (29/43)

### OOD Category Breakdown
| Category | N | Detected | Detection Rate | Mean Reject Score |
|---|---:|---:|---:|---:|
| **Non-waste objects** (keyboards, bikes, robots) | 15 | 9 | 60.0% | 0.4487 |
| **Clothes** (shirts) | 8 | 6 | 75.0% | 0.6601 |
| **Shoes** | 8 | 4 | 50.0% | 0.5017 |
| **Environmental scenes** (parks, streets) | 12 | 10 | 83.3% | 0.4878 |

---

## 3. Real Multi-Object Scene Evaluation (LOOP 9)

Evaluated on **15** real, natural photographs of multi-object scenes (beach litter, flea markets, cluttered waste).

* **Detection Rate @ Shipped Threshold (0.0702):** `46.7%` (7/15)
* **Mean Rejection Score:** `0.2554`
* **Final Verdict Breakdown:**
  * Unsupported: 7
  * Uncertain: 0
  * Supported: 8

---

## 4. Ambiguous Supported Items Evaluation

Evaluated on **10** visually challenging waste items (paper towels, pizza boxes, cigarette butts).

* **Final Verdict Breakdown:**
  * Supported: 8
  * Uncertain (Ambiguous triggered): 0
  * Unsupported (False Rejection): 2
* **Mean Top Probability:** `0.9888`
* **Mean Margin:** `0.9777`

---

## 5. Synthetic Flat Probes Evaluation

* **Probes Rejected:** 3/5
  * `gray noise (static)`: reject = 0.0541 -> supported
  * `flat color field (#161b23)`: reject = 0.5775 -> unsupported
  * `checkerboard 8px`: reject = 0.0062 -> supported
  * `horizontal gradient`: reject = 0.9999 -> unsupported
  * `blank white frame`: reject = 0.1331 -> unsupported

---

## 6. Operating-Point Analysis (LOOP 10)

| Operating Point | Target FRR | Threshold | Measured FRR | Measured OOD Detection | Measured Multi-Object Detection |
|---|---:|---:|---:|---:|---:|
| **Shipped Point** | ~1% | `0.0702` | `33.33%` | `67.44%` | `46.7%` |
| **Fresh 1% FRR** | 1% | `0.9997` | `0.00%` | `11.63%` | `0.00%` |
| **Fresh 2% FRR** | 2% | `0.9997` | `1.59%` | `11.63%` | `0.00%` |
| **Fresh 5% FRR** | 5% | `0.9976` | `4.76%` | `18.60%` | `0.00%` |
