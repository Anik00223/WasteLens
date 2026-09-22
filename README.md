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

MVP complete — **shipping the dual-head OOD rejection model (Iteration 5)**.
The MobileNetV2 (frozen) backbone serves two heads exported to TensorFlow.js:
the 4-bin classifier (test accuracy 0.9822, macro-F1 0.9742 — see
[docs/report.md](docs/report.md)) and a calibrated rejection head. The demo
renders three distinct result states:

- **supported** — a confident bin prediction, shown normally;
- **uncertain** — within scope but the bin prediction is ambiguous
  (existing rule: top < 0.60 or top−runner-up margin < 0.50);
- **unsupported** — the rejection head scores the image as outside the
  supported single-item waste scope at the calibrated threshold
  (`reject >= 0.8774`).

At that threshold the rejection head measured **84.3% OOD detection at a
1.38% false-rejection rate** on the held-out test split (rejection AUROC
0.9900; see [docs/rejection_experiment/outcome_decision.md](docs/rejection_experiment/outcome_decision.md)).
Rejection is **not perfect**: multi-item collages are the documented weak
spot (21.1% detection at the shipped threshold), and ~1.4% of supported
images are flagged — so the results panel keeps the single-item-photo scope
disclosure. `uncertain` (bin ambiguity) and `unsupported` (out of scope)
are deliberately separate states with distinct copy.

Real-inference validation: `src/validate_realworld.py` →
`docs/realworld_validation.md` (single-head baseline) and
`docs/rejection_experiment/realworld_validation_frozen.md` (shipped dual-head).

## Out of Scope (for now)

Mobile app, backend API, user accounts — anything beyond a single-page web demo.
