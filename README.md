# WasteLens

A browser-based waste/garbage image classifier. Upload or take a photo, and the model (running entirely client-side via TensorFlow.js) classifies it into a waste category and shows a confidence score.

See [PROJECT_BRIEF.md](PROJECT_BRIEF.md) for the full project brief, constraints, and success criteria.

## Repository Structure

```
data/raw/              # Original, unmodified training images (not committed if large)
data/processed/        # Train/val/test splits after preprocessing
notebooks/             # Exploratory analysis and dataset evaluation notebooks
src/                   # Training / evaluation / conversion scripts
  train.py             # Transfer-learning training (MobileNetV2 / ResNet18 backbone)
  evaluate.py          # Per-class precision / recall / F1 evaluation
  export_tfjs.py       # Convert trained Keras model to TensorFlow.js format
models/checkpoints/    # Keras checkpoints during training
models/tfjs_model/     # Final converted TensorFlow.js model artifacts
web/                   # Single-page web demo
  index.html           # Demo page (camera capture / upload → prediction → confidence)
  model/               # TensorFlow.js model files loaded by the demo
docs/report.md         # Project report (methodology, per-class results, weak-class analysis)
```

## Status

MVP complete. The MobileNetV2 (frozen) + 4-bin head is trained (test accuracy
0.9822, macro-F1 0.9742 — see [docs/report.md](docs/report.md)), exported to
TensorFlow.js, and served by the single-page demo in `web/`. Predictions whose
top bin is not clearly dominant are flagged **uncertain** in the demo
(motivated by the real-world testing findings in docs/report.md, section 4.1).

## Out of Scope (for now)

Mobile app, backend API, user accounts — anything beyond a single-page web demo.
