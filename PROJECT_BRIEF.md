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
