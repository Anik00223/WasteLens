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

Skeleton only — no implementation logic yet.

## Out of Scope (for now)

Mobile app, backend API, user accounts — anything beyond a single-page web demo.
