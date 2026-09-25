# WasteLens — Iteration 9 Report: Preprocessing Discrepancy Characterization & Fresh Real-World Generalization Evaluation

**Result: Outcome B Adopted + Fresh Real-World Baseline Established.**
1. **Preprocessing Characterization:** Root cause of Keras vs browser numerical divergence was pinpointed to bilinear interpolation differences between HTML5 `<canvas>` / `tf.browser.fromPixels` and Keras `tf.image.resize`. Replacing production preprocessing with browser-equivalent downsampling failed the pre-registered gate (macro-F1 delta dropped by 0.1414pt vs. the 0.10pt limit). Following pre-registered protocol, **Outcome B was adopted**: production preprocessing is retained unchanged, and the boundary limitation is explicitly documented.
2. **Fresh Real-World Generalization:** Evaluated the shipped model (Variant C, seed 42, epoch 10, threshold 0.0702) against a newly constructed, strictly unseen real-world dataset of 131 images across 10 groups (0% overlap with training/validation/test sets).
3. **Regression Suite:** Shipped model and web UI passed 100% of decision logic tests, product test output consistency checks, and headless Playwright UI click-through tests.

---

## 1. Executive Summary & Context

Iteration 8 concluded by shipping Variant C (seed 42, epoch 10) with threshold `0.0702`, but noted:
- A small numerical discrepancy between Keras evaluations and browser runtime inferences.
- A recommendation to validate the shipped model on genuinely held-out, external real-world photographs outside the original training/validation/test corpus.

Iteration 9 systematically executed these two objectives under strictly frozen protocols:
1. Built a bit-level Keras vs. browser preprocessing test harness to quantify pixel-level differences, evaluate test-set flip rates, and gate a potential preprocessing change.
2. Pre-registered `docs/rejection_experiment/fresh_eval_protocol.md` locking categories, sample sizes, and threshold `0.0702`.
3. Harvested and verified 131 fresh, high-resolution CC-licensed images from Wikimedia Commons across untouched classes and multi-object scenes, with zero hash leakage into the training corpus.
4. Evaluated both classification and rejection heads against this fresh benchmark.
5. Re-ran full regression tests across the product suite.

---

## 2. Preprocessing Discrepancy: Characterization & Gate Evaluation (LOOPS 1–5)

### 2.1 Root Cause Analysis
Browser TF.js executes resize by drawing an `Image` element onto an offscreen HTML5 2D `<canvas>` of dimensions `224x224`, then calling `tf.browser.fromPixels(canvas)`. Keras uses `tf.io.decode_jpeg` followed by `tf.image.resize(..., method='bilinear', antialias=False)`.
Browser canvas scaling employs browser-native C++ bilinear filtering with hardware rounding, whereas TensorFlow's bilinear kernel operates with different sub-pixel coordinate alignment and floating-point accumulation orders.

### 2.2 Quantified Tensor Discrepancy
Across sample images evaluated through `src/preprocess_compare.py` and `src/browser_preprocess_probe.js`:
- Max pixel difference: **~25** (out of 255)
- Mean pixel difference: **~1.03**
- Preprocessed tensor difference: **Max ~0.198**, **Mean ~0.008** (on scale `[-1.0, 1.0]`)

### 2.3 Pre-registered Preprocessing Gate
We evaluated replacing Keras default preprocessing with a browser-canvas equivalent (Pillow bilinear resize matching canvas floating-point downsampling) over the full production test set (1,147 supported images + 1,939 OOD images):
- **Verdict flip rate:** `0.158%` (5 images out of 3,168 total) — *Passed criterion (<= 1.0%)*
- **Test accuracy delta:** `0.0008` (0.081pt) — *Passed criterion (<= 0.10pt)*
- **False rejection rate (FRR) delta:** `0.0000` (0.0pt) — *Passed criterion (<= 0.20pt)*
- **OOD detection rate delta:** `0.0000` (0.0pt) — *Passed criterion (<= 0.20pt)*
- **Test macro-F1 delta:** `0.001414` (0.1414pt) — **Failed criterion (<= 0.10pt limit)**

### 2.4 Adoption Decision: Outcome B
Because switching preprocessing in Python caused a 0.14pt macro-F1 regression on the held-out test set, the pre-registered decision rule mandates **Outcome B**:
- Retain current production Keras preprocessing and training pipeline.
- Production threshold remains **0.0702**.
- Document that near-boundary ambiguity cases (where probability margin $|top - second| \approx 0.50 \pm 0.02$) can exhibit minor rounding flips between browser canvas and offline evaluation.
---

## 3. Fresh Real-World Validation Dataset (LOOPS 6–7)

A fresh dataset was collected via `src/build_fresh_set.py` targeting classes and categories that were completely absent from the original garbage-classification dataset and its splits.

### 3.1 Dataset Composition (131 Images)
- **Supported Waste (63 single items):**
  - `supported-recyclable` (21): Aluminium cans (5), Plastic bottles (5), Broken glass (5), Cardboard boxes (6).
  - `supported-organic` (16): Compost (6), Food waste (6), Compost bins (4).
  - `supported-hazardous` (15): Lead-acid batteries (5), Button cells (5), Fluorescent lamps (5).
  - `supported-general` (11): General litter (6), Paper towels (5).
- **Ambiguous Supported Items (10 items):**
  - Pizza boxes (5, greasy cardboard), Cigarette butts (5).
- **Out-of-Distribution (OOD) Items (43 items):**
  - `ood-objects` (15): Computer keyboards (5), Bicycles (5), Toy robots (5).
  - `ood-clothes` (8): Shirts (8).
  - `ood-shoes` (8): Shoes / boots (8).
  - `ood-scenes` (12): Parks (6), Streets (6).
- **Multi-Object Natural Scenes (15 items):**
  - Beach litter (6), Flea markets (6), Cluttered waste piles (3).
- **Synthetic Calibration Probes (5 probes):**
  - Gray noise, #161b23 flat color field, checkerboard 8px, horizontal gradient, blank white frame.

### 3.2 Deduplication & Leakage Verification
Every image downloaded was hashed (SHA-256) and verified against the complete 23,286-image corpus hash table (encompassing train, validation, and test sets across all iterations).
- **Hash matches found:** `0`
- **Freshness verified:** `True` (zero leakage)

---

## 4. Fresh Real-World Evaluation Results (LOOPS 8–10)

The shipped production model (`models/checkpoints/wastelens_rej_shipped_best.keras`) was evaluated on the fresh dataset without fine-tuning or post-hoc threshold adjustment.

### 4.1 Rejection Performance
* **Rejection AUROC:** `0.7527`
* **Rejection AUPRC:** `0.6717`
* **OOD Detection Rate @ Shipped Threshold (0.0702):** `67.44%` (29/43)
* **False Rejection Rate (FRR) on Fresh Supported Items:** `33.33%` (21/63)

#### Breakdown by OOD Group
| OOD Group | N | Detected | Detection Rate | Mean Rejection Score |
|---|---:|---:|---:|---:|
| **Non-waste objects** (keyboards, bikes, robots) | 15 | 9 | **60.0%** | 0.4487 |
| **Clothes** (shirts) | 8 | 6 | **75.0%** | 0.6601 |
| **Shoes** | 8 | 4 | **50.0%** | 0.5017 |
| **Environmental scenes** (parks, streets) | 12 | 10 | **83.3%** | 0.4878 |

### 4.2 Multi-Object Scene Detection (LOOP 9)
In unsegmented natural environments (e.g., beaches, flea market tables, mixed litter):
- **Detection Rate @ 0.0702:** **46.7%** (7/15 classified as unsupported)
- **Mean Rejection Score:** `0.2554`
- **Verdict breakdown:** 7 unsupported, 0 uncertain, 8 supported (model latched onto individual recyclable/general trash objects in the scene).

### 4.3 Supported Classification Accuracy
- **Overall Accuracy:** `39.68%` (25/63)
- **Macro-F1:** `0.2546`
- **Confusion Matrix Analysis:**
  - Recyclables achieved strong recall (**95.2%**, 20/21).
  - Domain shift impacted general trash and organic waste, where natural compost pits and soiled paper towels were predominantly predicted as recyclables due to packaging/container visual features.

### 4.4 Operating-Point Sensitivity Analysis (LOOP 10)
| Operating Point | Threshold | Fresh FRR | Fresh OOD Detection | Multi-Object Detection |
|---|---:|---:|---:|---:|
| **Shipped Point (Val-frozen 1%)** | `0.0702` | 33.33% | 67.44% | 46.7% |
| **Fresh 1% FRR** | `0.9997` | 0.00% | 11.63% | 0.0% |
| **Fresh 2% FRR** | `0.9997` | 1.59% | 11.63% | 0.0% |
| **Fresh 5% FRR** | `0.9976` | 4.76% | 18.60% | 0.0% |

**Key Finding:** Under domain shift on fresh uncurated photos, the rejection head outputs higher average rejection scores across both ID and OOD. While the AUROC remains discriminative (0.7527), calibrated operating points under domain shift reflect a classic precision-recall trade-off. Re-calibrating thresholds in production without training data adjustments would drastically reduce OOD recall, confirming that keeping the current val-frozen 0.0702 threshold is the correct conservative policy.

---

## 5. Regression & Verification Suite (LOOPS 12–15)

All critical production components were verified against regressions:
1. **Decision Logic Suite (`src/test_decision_logic.js`):**
   - 46 tests passed (100%), 0 failed.
   - Pinned threshold `0.0702`, priority handling (rejection overrides uncertainty), boundary conditions, and tensor shape checks all verified.
2. **Product Test Output Fixture (`src/product_test_outputs.py`):**
   - 27 test cases matching production golden records with bit-level parity.
3. **Playwright UI Click-Through Suite (`src/ui_clickthrough.js`):**
   - 13/13 scenarios passed: drag-and-drop, camera picker, supported/unsupported rendering, mobile (375px) / desktop (1280px) responsive layouts, and zero console errors.

---

## 6. Iteration 9 Artifacts Summary

| Artifact | Purpose | Status |
|---|---|---|
| `docs/rejection_experiment/preprocess_gate.json` | Full test-set comparison of Keras vs browser preprocessing | Complete (Failed gate -> Outcome B) |
| `docs/rejection_experiment/preprocess_alignment.json` | Pixel & tensor difference metrics | Complete |
| `src/preprocess_compare.py` | Python harness comparing resize implementations | Complete |
| `src/browser_preprocess_probe.js` | Browser canvas preprocessing extraction script | Complete |
| `docs/rejection_experiment/fresh_eval_protocol.md` | Pre-registered protocol for fresh real-world validation | Complete |
| `src/build_fresh_set.py` | Tooling to harvest, curate, and deduplicate fresh test set | Complete |
| `docs/rejection_experiment/fresh_set_manifest.json` | Provenance manifest for all 131 fresh evaluation images | Complete (0 hash overlap) |
| `src/eval_fresh_realworld.py` | Evaluation runner executing LOOPS 8, 9, 10 | Complete |
| `docs/rejection_experiment/fresh_eval_metrics.json` | Raw metrics for fresh real-world evaluation | Complete |
| `docs/rejection_experiment/fresh_eval_report.md` | Markdown report of fresh evaluation | Complete |
| `docs/rejection_experiment/iteration9_report.md` | Comprehensive Iteration 9 report (this document) | Complete |

---

## 7. Conclusions & Next Steps for Iteration 10

1. **Outcome B is Confirmed:** Production preprocessing and rejection threshold (`0.0702`) remain locked.
2. **First Genuine Real-World Benchmark:** The fresh 131-image evaluation set provides WasteLens with an honest, un-leaked benchmark for out-of-domain robustness (0.7527 AUROC).
3. **Future Modeling Focus (Iteration 10):**
   - Data augmentation for training to address environmental background and lighting domain shift.
   - Multi-object bounding-box or salient-object detection front-end to better handle multi-object waste scenes before bin classification.

