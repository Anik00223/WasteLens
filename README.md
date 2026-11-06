# WasteLens — AI-Powered Waste Classification with Dual-Head OOD Rejection

[![Tests](https://img.shields.io/badge/Python%20Tests-39%20passed-brightgreen.svg)]()
[![JS Tests](https://img.shields.io/badge/JS%20Decision%20Tests-46%20passed-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-MobileNetV2%20Dual--Head-blue.svg)]()
[![Inference](https://img.shields.io/badge/Inference-Client--Side%20TensorFlow.js-orange.svg)]()

> **B.Tech AI & ML Capstone Project**<br>
> **Official Implementation:** MobileNetV2 + TensorFlow.js 4.22.0 client-side inference + Dual-head Out-of-Distribution (OOD) rejection.<br>
> **Repository:** [https://github.com/Anik00223/WasteLens](https://github.com/Anik00223/WasteLens)

---

## 1. Project Overview

**WasteLens** is an intelligent waste image classification system designed to make automated recycling practical and reliable. Unlike conventional image classifiers that force every input into a pre-defined waste category, WasteLens combines transfer learning with an integrated **Out-of-Distribution (OOD) rejection mechanism**.

The application operates **100% client-side** inside the browser using TensorFlow.js. No images are uploaded to a remote server, ensuring privacy, instant inference, and offline operability.

The system evaluates incoming images and assigns one of **three distinct verdict states**:
- **Supported:** Confident classification into one of the four supported waste streams.
- **Uncertain:** The input is recognized as waste, but classification confidence or margin is ambiguous.
- **Unsupported:** The input is detected as outside the supported single-item waste scope (e.g., non-waste objects, complex cluttered scenes, synthetic patterns).

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

## 10. Web Application Features

The frontend is implemented as a standalone static application in `web/index.html`:
- **Direct Local Inference:** Model and weight shards are fetched and evaluated locally in the browser.
- **Three Result States:**
  - **Supported:** Displays the classified bin, confidence meter, and per-class probability breakdown.
  - **Uncertain:** Amber warning card advising the user that the item is ambiguous or boundary-line.
  - **Unsupported:** Red rejection card notifying the user that the image is outside the supported single-item waste scope.
- **Input Support:** Drag-and-drop file upload, file browser selector, and mobile camera capture.
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
Verify model integrity, manifests, and contract rules:
```bash
# Run all Python unit tests (39 tests)
pytest -v

# Run JavaScript decision logic tests (46 tests)
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

1. **Domain Shift on In-the-Wild Photos (Banana Peel Observation):**
   The training dataset consists predominantly of single objects photographed on plain, uniform backgrounds. When tested on real-world mobile photographs with complex countertop or kitchen textures, feature drift can cause elevated rejection scores (e.g., a real banana peel scoring `reject = 89.1%` despite `organic = 95.2%`). Independent evaluation on fresh real-world images demonstrated a 33.3% false rejection rate at the strict 0.0702 threshold.
2. **Metal vs. Battery Confusion (Rusty Cylinders):**
   Recyclable metal cans and hazardous cylindrical batteries share visual features (cylindrical aspect ratio, metallic reflections, rust/oxidation). Because `recyclable` comprises 68% of the training distribution versus 11.5% for `hazardous`, the model can bias towards predicting rusty cylindrical batteries as recyclable.
3. **Synthetic Texture Probes:**
   Certain flat synthetic color fields and high-frequency uniform noise patterns fail to trigger rejection, as the convolutional filters respond primarily to edge and shape gradients.

---

## 15. Future Work

- **In-the-Wild Domain Adaptation:** Fine-tune the feature representations using diverse background augmentations and mobile phone datasets (e.g., RealWaste dataset).
- **Object Detection Integration:** Integrate a lightweight client-side detector (such as YOLOv8-nano) to crop and localize multiple individual waste items in cluttered scenes prior to classification.
- **Dynamic Rejection Calibration:** Implement temperature scaling and user-adjustable sensitivity sliders in the UI for specialized sorting contexts.
