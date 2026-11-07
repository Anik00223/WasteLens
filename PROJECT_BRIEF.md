# WasteLens — Project Brief

> **Repository:** https://github.com/Anik00223/WasteLens.git (branch `main`, local working directory: `WasteLens`)
> **Status:** Project brief written before any implementation work.

## 1. Project Name

**WasteLens**

## 2. Goal

A browser-based waste/garbage image classifier. The user either uploads an image or takes a photo using their device camera. The app classifies the image into waste categories and displays a confidence score alongside the prediction.

**Target categories (initial taxonomy):**
- Recyclable
- Organic
- Hazardous
- General trash

## 3. Constraints

- **College ML project.** The model must use **transfer learning** (e.g., a pretrained **MobileNetV2** or **ResNet18** backbone with a new classification head) — **not** training from scratch.
- **Deployment target: TensorFlow.js.** The trained model is converted and runs **entirely client-side in the browser**.
- **No backend server.** All inference happens in the browser; the demo is static.

## 4. Success Criteria

1. **Working end-to-end demo:** camera capture or file upload → model prediction → predicted category with confidence score shown to the user.
2. **Per-class evaluation:** precision, recall, and F1 reported **per class** — not just a single overall accuracy figure.
3. **Weak-class analysis:** a clear, written explanation of any weak-performing class (what it gets confused with, and a plausible reason why).

## 5. Explicitly Out of Scope (for now)

- Native mobile app (iOS/Android, React Native, etc.)
- Backend API or server-side inference
- User accounts, authentication, or cloud storage
- Anything beyond a single-page web demo

## 6. Official Architecture, Operational Modes, and Evaluation

### 6.1 Official Architecture
- **Official Backbone:** MobileNetV2 (frozen ImageNet weights, 224x224x3 input).
- **Dual-Head Output:**
  - 4-class classification head (`predictions`, softmax): `recyclable`, `organic`, `hazardous`, `general trash`.
  - Out-of-Distribution rejection head (`reject`, sigmoid): $P(\text{unsupported}) \in [0, 1]$.
- **Preprocessing:** Bilinear resize to $224 \times 224$ and normalization strictly to $[-1.0, 1.0]$ via $(x / 127.5) - 1.0$.
- **Deployment:** 100% client-side in-browser inference via TensorFlow.js 4.22.0 (`web/index.html`). No server-side ML inference or backend dependencies.

### 6.2 Dual Operational Modes
1. **Single Item Mode (Default):**
   - The primary and recommended operational mode.
   - Evaluates the whole image holistically using the dual-head MobileNetV2 architecture.
   - Enforces a calibrated three-state decision hierarchy:
     - **Supported:** Confident classification into one of the four supported categories ($\text{reject} < 0.0702$, top $\ge 0.60$, margin $\ge 0.50$).
     - **Uncertain:** The input is recognized within waste scope, but confidence or classification margin is ambiguous.
     - **Unsupported:** The input is rejected as non-waste, complex clutter, or out-of-distribution ($\text{reject} \ge 0.0702$).
2. **Multi-Region Analysis Mode (Experimental Feature):**
   - An experimental client-side heuristic examining fixed spatial regions in scenes containing multiple waste items.
   - Evaluates a deterministic grid of 8 fixed spatial proposal windows (full scene, center, and overlapping quadrants) via a single batched `tf.image.cropAndResize` forward pass.
   - Evaluates each proposal window against the frozen $0.0702$ rejection threshold and suppresses overlapping duplicates via IoU.
   - **Technical Honesty:** These rectangles represent **fixed spatial proposals**, not trained object-detection bounding boxes. Does not reliably separate individual objects; WasteLens does not implement trained bounding-box regression (e.g., YOLO or SSD).
   - **Documented Experimental Limitations:**
     - May produce duplicate regions on single centered items (quadrants have $\text{IoU} < 0.30$).
     - May produce bridging proposals across physically separated items.
     - Prone to missing objects or generating false rejections on complex outdoor textures (asphalt/soil).
     - Substantially less reliable on dense clutter and overlapping objects.

### 6.3 Evaluation Summary
- **Held-out Test Set Classification:** 98.13% accuracy, 0.9721 macro-F1 score across 1,233 held-out test images.
- **OOD Rejection Performance:** 99.64% OOD detection rate at 1.78% False Rejection Rate (AUROC: 0.9997, AUPRC: 0.9998) at the calibrated $0.0702$ threshold.
- **Future Scope:** Training or integrating a lightweight end-to-end detector (such as YOLOv8-nano) on annotated multi-object waste datasets for pixel-accurate bounding box regression.
