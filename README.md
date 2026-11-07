# WasteLens — AI-Powered Waste Classification with Dual-Head OOD Rejection

[![Tests](https://img.shields.io/badge/Python%20Tests-42%20passed-brightgreen.svg)]()
[![JS Tests](https://img.shields.io/badge/JS%20Decision%20Tests-60%20passed-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-MobileNetV2%20Dual--Head-blue.svg)]()
[![Modes](https://img.shields.io/badge/Modes-Single%20Item%20%7C%20Multi--Waste-purple.svg)]()
[![Inference](https://img.shields.io/badge/Inference-Client--Side%20TensorFlow.js-orange.svg)]()

> **B.Tech AI & ML Capstone Project**<br>
> **Official Implementation:** MobileNetV2 + TensorFlow.js 4.22.0 client-side inference + Dual-head Out-of-Distribution (OOD) rejection + Multi-waste candidate region analysis.<br>
> **Repository:** [https://github.com/Anik00223/WasteLens](https://github.com/Anik00223/WasteLens)

---

## 1. Project Overview

**WasteLens** is an intelligent waste image classification system designed to make automated recycling practical and reliable. Unlike conventional image classifiers that force every input into a pre-defined waste category, WasteLens combines transfer learning with an integrated **Out-of-Distribution (OOD) rejection mechanism** and **multi-waste candidate region analysis**.

The application operates **100% client-side** inside the browser using TensorFlow.js. No images are uploaded to a remote server, ensuring privacy, instant inference, and offline operability.

The application supports **two operational modes**:
1. **Single Item (Default):** Holistic image evaluation with dual-head OOD rejection (`REJECT_THRESHOLD = 0.0702`) and ambiguity detection.
2. **Multi-Region Analysis (Experimental Feature):** Multi-item analysis using an 8-window fixed spatial proposal grid, single-pass batched `cropAndResize` inference, per-window dual-head OOD filtering, and IoU duplicate suppression. Evaluates coarse candidate regions; not an end-to-end trained object detector. Substantially less reliable on dense clutter and overlapping objects.

In both modes, the system assigns one of **three distinct verdict states**:
- **Supported:** Confident classification into one of the four supported waste streams.
- **Uncertain:** The input is recognized as waste, but classification confidence or margin is ambiguous.
- **Unsupported:** The input is detected as outside the supported waste scope (e.g., non-waste objects, complex cluttered scenes, synthetic patterns).

---

## 2. Problem Statement

Standard Convolutional Neural Networks (CNNs) equipped with softmax output heads suffer from **overconfidence on out-of-distribution data**: they routinely map arbitrary non-waste objects, pets, household furniture, or complex street scenes to a waste bin with 99%+ statistical confidence. In a real-world municipal or automated sorting pipeline, this creates severe contamination in recycling bins.

WasteLens addresses this failure mode by integrating a dedicated OOD rejection head and calibrated ambiguity margins directly into the model architecture.

---

## 3. Project Objectives

1. **Accurate 4-Category Waste Classification:** Classify single-item waste photographs into Recyclable, Organic, Hazardous, and General Trash.
2. **Built-in Out-of-Distribution Rejection:** Reject non-waste inputs, textiles, ambiguous clutter, and multi-object scenes rather than assigning a false classification.
3. **Calibrated Decision Layer:** Produce three explicit user-facing states (`Supported`, `Uncertain`, `Unsupported`) with honest, calibrated decision boundaries.
4. **Zero-Backend Browser Inference:** Convert and deploy the trained model directly in the browser via TensorFlow.js 4.22.0.

---

## 4. Official Model Architecture

The official model is built upon **MobileNetV2** (ImageNet pre-trained) with a frozen feature extraction backbone and a dual-head output structure.

```
Input Image (224 x 224 x 3)
   │
   ▼
MobileNetV2 Backbone (Frozen, ImageNet weights)
   │
   ▼
GlobalAveragePooling2D
   │
   ▼
Dropout (0.2)
   │
   ▼
Dense (256 units, ReLU)  [name: "fc_256"]
   │
   ▼
Dropout (0.2)  [name: "dropout_2"]
   ├───► Dense (4 units, Softmax) [name: "predictions"]  --> Classification Probabilities
   └───► Dense (1 unit, Sigmoid)  [name: "reject"]       --> OOD Rejection Score
```

### Architecture Specifications
- **Input Dimensions:** `(224, 224, 3)`
- **Backbone:** MobileNetV2 (depth multiplier 1.0, frozen weights)
- **Shared Dense Layer:** 256 units with ReLU activation
- **Trainable Parameters:** 328,964
- **Dual Outputs:**
  1. `predictions`: 4-class softmax probability distribution over waste categories.
  2. `reject`: Sigmoid output where $P(\text{unsupported}) \in [0, 1]$.

> *Note on Legacy Experiments:* Earlier experiments exploring ConvNeXt-XL or PyTorch backends are preserved in `notebooks/` as experimental historical artifacts. The official, verified, and shipped architecture is the MobileNetV2 dual-head pipeline.

---

## 5. Class Mapping and Taxonomy

The 4-class taxonomy is mapped from the primary source dataset:

| Class Index | Waste Category | Source Dataset Classes (Mostafa Ablal) |
|:---:|:---|:---|
| **0** | **Recyclable** | Cardboard, Paper, Plastic, Metal, Brown Glass, Green Glass, White Glass |
| **1** | **Organic** | Biological waste (food waste, fruit scraps) |
| **2** | **Hazardous** | Household batteries |
| **3** | **General Trash** | Non-recyclable miscellaneous waste |

*Dropped classes:* `clothes` and `shoes` were excluded from the classification taxonomy to prevent label ambiguity and were repurposed as negative OOD evaluation sets.

Strict class array ordering: `["recyclable", "organic", "hazardous", "general trash"]`.

---

## 6. Image Preprocessing

Training and browser inference maintain strict mathematical parity:
1. Decode image to RGB.
2. Bilinear resize to $224 \times 224$.
3. Normalize pixel intensities to the range $[-1.0, 1.0]$:
   $$\text{normalized} = \frac{x}{127.5} - 1.0$$
4. Browser runtime implementation:
   ```javascript
   var pixels = tf.browser.fromPixels(img);
   var resized = tf.image.resizeBilinear(pixels, [224, 224]);
   var normalized = resized.div(127.5).sub(1);
   ```
Parity between TensorFlow/Keras and TensorFlow.js has been numerically verified ($\Delta_{\text{max}} < 3.64 \times 10^{-6}$, pure float32 noise).

---

## 7. Dataset and Data Splits

The primary dataset used is the **Garbage Classification Dataset** by Mostafa Ablal (Kaggle).

### Split Counts (Stratified Seed 42)
- **Training Set:** 5,749 images (70%)
- **Validation Set:** 1,231 images (15%)
- **Test Set (Held-out & Locked):** 1,233 images (15%)
- **Total Supported Dataset:** 8,213 images

To address the severe dataset imbalance (~8:1 ratio between Recyclable and General Trash), training utilized inverse-frequency class weights:
$$\text{Recyclable: } 0.37, \quad \text{Organic: } 2.09, \quad \text{Hazardous: } 2.17, \quad \text{General Trash: } 2.95$$

---

## 8. Dual-Head OOD Rejection Methodology

### Rejection Training Data
The rejection head was trained on:
1. **In-Distribution Supported Waste ($y_{\text{rej}} = 0$):** Training split of the 8,213 supported waste images.
2. **Out-of-Distribution Inputs ($y_{\text{rej}} = 1$):**
   - Dropped dataset classes (`clothes`, `shoes`), rigorously MD5-deduplicated against supported splits.
   - Non-waste natural images from CIFAR-10 (disjoint official splits).
   - Multi-item synthetic collages generated strictly from training-split waste photos.

### Decision Hierarchy and Thresholds
The shipped operating threshold is frozen at:
$$\text{REJECT\_THRESHOLD} = 0.0702$$
This threshold was selected at a validation operating point targeting a 1% false rejection rate on in-distribution data.

The decision layer operates in strict priority order:
1. **Rejection Gate:** If $\text{score}_{\text{reject}} \ge 0.0702 \implies \mathbf{Unsupported}$
2. **Ambiguity Gate:** If $\text{score}_{\text{top}} < 0.60$ or $(\text{score}_{\text{top}} - \text{score}_{\text{second}}) < 0.50 \implies \mathbf{Uncertain}$
3. **Acceptance:** Otherwise $\implies \mathbf{Supported}$

---

## 9. Evaluation Results

### 9.1 Classification Performance (Held-out Test Set)
- **Test Accuracy:** 98.13% (shipped dual-head checkpoint)
- **Macro-Averaged F1-Score:** 0.9721

| Waste Class | Precision | Recall | F1-Score | Test Support |
|:---|:---:|:---:|:---:|:---:|
| **Recyclable** | 0.9823 | 0.9928 | 0.9876 | 839 |
| **Organic** | 0.9864 | 0.9797 | 0.9831 | 148 |
| **Hazardous** | 0.9710 | 0.9437 | 0.9571 | 142 |
| **General Trash** | 0.9800 | 0.9423 | 0.9608 | 104 |

### 9.2 OOD Rejection Performance (Test Split)
- **Rejection AUROC:** `0.9997`
- **Rejection AUPRC:** `0.9998`
- **Overall OOD Detection Rate:** `99.64%` at `1.78%` False Rejection Rate (FRR).
- **Per-Source OOD Detection at Shipped Threshold (0.0702):**
  - Clothes: `99.9%`
  - Shoes: `99.0%`
  - Non-waste objects (CIFAR-10): `99.9%`
  - Multi-item collages: `97.8%`

---

## 10. Web Application Features & Operational Modes

The frontend is implemented as a standalone static application in `web/index.html` (zero server dependencies):

### 10.1 Conceptual Architecture: Three Distinct Levels

To maintain technical honesty, WasteLens strictly delineates three distinct levels of image analysis:

1. **Official Single-Item Classification (Default):**
   - The primary, fully verified application mode.
   - Holistic image evaluation using MobileNetV2 with calibrated dual-head OOD rejection (`REJECT_THRESHOLD = 0.0702`) and ambiguity margins.
   - Produces three discrete states: `Supported`, `Uncertain`, `Unsupported`.
2. **Experimental Multi-Region Analysis (Candidate Proposals MVP):**
   - An experimental client-side heuristic examining fixed spatial regions in scenes containing multiple waste items.
   - Evaluates a deterministic grid of 8 fixed spatial proposal windows via batched `tf.image.cropAndResize`.
   - Filters non-waste windows with the frozen 0.0702 rejection threshold and suppresses overlapping duplicates via IoU.
   - Displays a persistent experimental notice in the UI: *"Experimental Multi-Waste Analysis: This mode examines several fixed image regions. It may duplicate, miss or incorrectly classify objects. For the most reliable results, use Single Item mode."*
   - Rectangles represent **fixed spatial proposals**, not trained object-detection bounding boxes. Does not reliably separate individual objects.
3. **True Object Detection (Future Work — Unimplemented):**
   - End-to-end bounding box regression and object localization (e.g., YOLO, SSD, Faster R-CNN).
   - WasteLens does **not** implement a trained object detector; it relies on image-level classification over candidate proposals.

### 10.2 Client-Side Reliability
- **Privacy & Speed:** Model and weight shards run 100% locally in the browser with WebGL acceleration (fallback to CPU/Wasm).
- **Responsive Controls:** Instant mode switching between Single Item and Multi-Region without re-uploading the photograph.
- **Client Safeguards:** Checks against `file://` protocol misuse, missing weight shards, and invalid image formats.

---

## 11. Repository Structure

```
WasteLens/
├── data/                       # Dataset manifests and metadata
├── docs/                       # Project documentation and experiment reports
│   ├── report.md               # Detailed academic technical report
│   ├── dataset_research.md     # Primary dataset research and class mapping
│   └── rejection_experiment/   # OOD training logs, metrics, and decisions
├── models/
│   └── checkpoints/            # Keras model checkpoints (.keras)
├── notebooks/                  # Colab training and legacy exploration notebooks
├── src/                        # Core Python scripts and test utilities
│   ├── train.py                # Single-head baseline trainer
│   ├── train_rejection_variantc.py # Shipped dual-head trainer
│   ├── evaluate.py             # Test set evaluator
│   ├── export_tfjs.py          # Keras to TF.js converter
│   └── test_decision_logic.js  # JavaScript unit tests
├── tests/                      # Automated test suite (Pytest)
│   ├── test_application_contract.py # Shards, labels, and contract validation
│   ├── test_backup_iter12.py
│   ├── test_iter12_preflight.py
│   ├── test_materialize_manifest.py
│   ├── test_notebook_structure.py
│   └── test_restore_teacher_ckpt.py
├── web/                        # Official client-side web application
│   ├── index.html              # Frontend application
│   └── model/                  # Exported TensorFlow.js model files
│       ├── model.json          # Layer topology and weights manifest
│       ├── group1-shard*.bin   # Model weight shards (3 shards)
│       └── labels.json         # Class labels in strict order
├── PROJECT_BRIEF.md            # Initial project specification
└── README.md                   # Complete master documentation
```

---

## 12. Local Setup and Execution

### Prerequisites
- Python 3.10+
- Modern Web Browser (Chrome, Edge, Firefox, or Safari)
- Node.js (optional, for running JS test suites)

### 1. Clone Repository
```bash
git clone https://github.com/Anik00223/WasteLens.git
cd WasteLens
```

### 2. Run Test Suites
Verify model integrity, manifests, contract rules, and multi-waste NMS logic:
```bash
# Run all Python unit tests (42 tests)
pytest -v

# Run JavaScript decision and multi-waste logic tests (60 tests)
node src/test_decision_logic.js

# Verify browser model multi-output execution
node src/validate_reject_in_browser.js web/model
```

### 3. Launch Local Demo Server
Browsers require an HTTP server to load WebAssembly and model weight shards:
```bash
python -m http.server 8000 --directory web
```
Open your browser and navigate to: `http://localhost:8000`

---

## 13. Google Colab Demonstration Guide

To run a live interactive demonstration using Google Colab's cloud environment:

1. Open Google Colab and start a fresh Python 3 runtime.
2. In a notebook code cell, execute:
   ```python
   # 1. Clone the repository
   !git clone https://github.com/Anik00223/WasteLens.git
   %cd /content/WasteLens

   # 2. Start the background static server for web/
   import subprocess
   import time
   server = subprocess.Popen(["python3", "-m", "http.server", "8000", "--directory", "web"])
   time.sleep(2)

   # 3. Generate Colab live web preview URL
   from google.colab.output import eval_js
   print("Live WasteLens Demo URL:")
   print(eval_js("google.colab.kernel.proxyPort(8000)"))
   ```
3. Click the generated proxy link to test images directly in your browser.

---

## 14. Documented Real-World Limitations

### 14.1 Qualitative Manual Demonstration Results (Single-Item Mode)

In addition to benchmark metrics on held-out test splits, informal qualitative testing was performed during live demonstrations:

| Input | Observed Result | Assessment | Failure Mode / Note |
|---|---|---|---|
| **Plastic bottle** | Recyclable, 100% | Accepted | Correct in-distribution classification |
| **Banana peel** | Organic 95.2%, rejection 89.1% | False rejection | Domain shift: background/lighting elevated rejection above 0.0702 threshold |
| **Rusty battery-like objects** | Recyclable 96.8% | Potential misclassification | Visual overlap with recyclable metal cylinders; ground-truth verification needed |
| **Mixed household waste** | Unsupported, rejection 97.0% | Rejected | Correct single-item scope enforcement on cluttered scenes |
| **Shoe** | Unsupported, rejection 99.3% | Rejected | Correct OOD rejection of unsupported apparel |
| **Mobile phone** | Recyclable, 100% | False acceptance | Rejection failure: glass/metal textures triggered confident recyclable classification |

### 14.2 Multi-Waste Real-World Validation Audit (14-Image Benchmark)

To assess the practical utility and limitations of the Multi-Region Analysis mode, a headless browser audit was conducted across 14 diverse real-world images spanning five categories:

| ID | Category | Scene Description | Single-Item Result | Multi-Region Analysis Result | Assessment |
|---|---|---|---|---|---|
| **A1** | Single Item | Centered 12-oz aluminium can | Recyclable 100.0% | 5 Regions (Center, TL, TR, BL, BR) | Quadrant duplication |
| **A2** | Single Item | Plastic bottle filling frame | Recyclable 100.0% | 1 Region (Full Scene) | Clean single detection |
| **A3** | Single Item | Banana peel on natural surface | Unsupported (reject 82.6%) | 0 Regions (all windows OOD) | Domain shift false rejection |
| **A4** | Single Item | Button cell batteries | Recyclable 91.8% | 3 Regions (BR, BL, TC) | Regional duplication / misclassification |
| **B1** | Two Items | Two separated soda cans | Unsupported (reject 14.4%) | 3 Regions (Center, TL: Can 1, TR: Can 2) | Coarse quadrant crops; includes redundant bridging center |
| **B2** | Two Items | Bowl & bottle on ledge | Unsupported (reject 9.5%) | 3 Regions (BR: Bottle, BL: Bowl, Center) | Coarse regional proposals; background excluded |
| **C1** | 3–5 Items | Litter on ground | Recyclable 100.0% | 4 Regions (BL, BR, TR, Center) | Coarse itemization |
| **C2** | 3–5 Items | Cardboard boxes on street | Unsupported (reject 76.4%) | 0 Regions (all windows OOD) | Domain shift (street asphalt) |
| **D1** | Mixed Clutter | Beach litter on sand | Unsupported (reject 70.5%) | 2 Regions (TL, TR) | Background rejected, partial items detected |
| **D2** | Mixed Clutter | Indoor mixed waste pile | Recyclable 100.0% | 1 Region (Full Scene) | Overly coarse full-scene fallback |
| **E1** | Non-Waste OOD | Shoes on plain background | Recyclable 100.0% | 1 Region (Full Scene) | False acceptance (leather/fabric) |
| **E2** | Non-Waste OOD | Folded cotton shirt | Recyclable 100.0% | 1 Region (Full Scene) | False acceptance (fabric texture) |
| **E3** | Non-Waste OOD | Computer keyboard | Unsupported (reject 97.3%) | 1 Region (TL crop) | Rejection leakage in crop window |
| **E4** | Non-Waste OOD | Outdoor park landscape | Unsupported (reject 97.6%) | 0 Regions (8/8 windows rejected) | Clean 100% OOD rejection |

**Overall Audit Finding:** **READY WITH LIMITATIONS** (Partially Useful). Multi-Region Analysis evaluates coarse candidate regions across the scene where Single-Item mode rejects the entire frame. However, because it relies on an experimental fixed-window heuristic rather than learned object detection, it exhibits five major operational limitations:
1. **Quadrant Duplication on Centered Objects:** When an item occupies the image center, multiple non-overlapping quadrant crops capture parts of the same item. Because disjoint quadrants have $\text{IoU} < 0.30$, IoU suppression cannot eliminate them without risking suppression of legitimate multi-item layouts.
2. **Bridging Center Proposals:** In two-item scenes, the center proposal window ($[0.2, 0.2, 0.8, 0.8]$) frequently encompasses parts of both items, generating a third redundant proposal.
3. **Domain Shift False Rejections:** Natural outdoor surfaces (dirt, gravel, asphalt) trigger high rejection scores ($> 0.0702$) across all proposal windows, resulting in zero candidate regions even when waste is visible.
4. **False Acceptance on Regional Crops:** Crop windows containing non-waste items with plastic or metallic textures can trigger false acceptance as recyclable.
5. **Dense Clutter and Overlapping Objects:** The fixed 8-window spatial grid lacks fine multi-scale anchors. In scenes with dense clutter or physically overlapping items, multiple items fall into the same candidate crop, causing the model to predict only the visually dominant item or fall back to an overly coarse full-scene prediction.

*Key Takeaway:* The candidate rectangles indicate **fixed spatial proposal windows**, not trained bounding-box regressions. For mission-critical single-item sorting, users should always rely on Single-Item mode.

### 14.3 Analysis of Underlying Limitations

1. **Domain Shift & False Rejections (Banana Peel Observation):**
   The training dataset consists predominantly of single objects photographed on plain, uniform backgrounds. When tested on real-world mobile photographs with complex countertop or kitchen textures, feature drift can cause elevated rejection scores (e.g., a real banana peel scoring `reject = 89.1%` despite `organic = 95.2%`). Independent evaluation on fresh real-world images demonstrated a 33.3% false rejection rate at the strict 0.0702 threshold.
2. **Negative Space Holes & False Acceptances (Mobile Phone Observation):**
   Unseen non-waste manufactured items containing glass, polished plastic, and aluminum frames (such as mobile phones) activate low-level edge and texture detectors similar to recyclable glass and metal containers. Because the OOD training data cannot cover all non-waste objects, the model falsely accepts them as recyclable with high confidence.
3. **Metal vs. Battery Confusion (Rusty Cylinders):**
   Recyclable metal cans and hazardous cylindrical batteries share visual features (cylindrical aspect ratio, metallic reflections, rust/oxidation). Because `recyclable` comprises 68% of the training distribution versus 11.5% for `hazardous`, the model can bias towards predicting rusty cylindrical batteries as recyclable.
4. **Synthetic Texture Probes:**
   Certain flat synthetic color fields and high-frequency uniform noise patterns fail to trigger rejection, as the convolutional filters respond primarily to edge and shape gradients.

---

## 15. Future Work

- **In-the-Wild Domain Adaptation:** Fine-tune the feature representations using diverse background augmentations and mobile phone datasets (e.g., RealWaste dataset).
- **Object Detection Integration:** Integrate a lightweight client-side detector (such as YOLOv8-nano) to crop and localize multiple individual waste items in cluttered scenes prior to classification.
- **Dynamic Rejection Calibration:** Implement temperature scaling and user-adjustable sensitivity sliders in the UI for specialized sorting contexts.
