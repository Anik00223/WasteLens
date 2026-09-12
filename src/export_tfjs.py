# src/export_tfjs.py
#
# PURPOSE:
#   Convert the trained Keras model to TensorFlow.js format for
#   client-side browser deployment (no backend server).
#
#   - Reads the trained model from models/checkpoints/.
#   - Exports layers/model artifacts to models/tfjs_model/ (and a copy
#     to web/model/ for the demo to load).
#   - CRITICAL: browser-side preprocessing in web/index.html must exactly
#     match training preprocessing (input size 224x224, normalization
#     scheme used by the chosen backbone).
#
# STATUS: Implemented (writes labels.json from train.BINS; the TF.js
#   conversion itself runs wherever the `tensorflowjs` package is
#   installed - see the Colab export path; local install is blocked by
#   dependency conflicts).
#
"""WasteLens TF.js export script.

Writes labels.json from the training bin order (train.BINS) and converts
the trained .keras checkpoint to TF.js layers-model format.

Run from the repo root, in an environment with `tensorflowjs` installed
(e.g. Colab - the local Python 3.13 env cannot take the install without
downgrading packaging/tensorflow-hub, so it is intentionally absent here):
    python src/export_tfjs.py                  # checkpoint -> web/model/
    python src/export_tfjs.py --model <path>   # custom checkpoint
    python src/export_tfjs.py --out <dir>      # custom output dir
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import train as wl  # single source of truth for BINS order - never retyped

DEFAULT_MODEL = Path("models/checkpoints/wastelens_ep09.keras")
DEFAULT_OUT = Path("web/model")
ARCHIVE_DIR = Path("models/tfjs_model")

# Hard guard: fail loudly instead of silently mislabeling predictions.
EXPECTED_BINS = ["recyclable", "organic", "hazardous", "general trash"]

INSTALL_CMD = "python -m pip install tensorflowjs"

def write_labels_json(out_dir: Path, archive_dir: Path) -> list[Path]:
    """Write labels.json (train.BINS order) to both output dirs."""
    labels = list(wl.BINS)
    if labels != EXPECTED_BINS:
        raise ValueError(
            f"train.BINS order changed to {labels} - refusing to write "
            f"labels.json (expected {EXPECTED_BINS}). Fix the mapping before "
            "exporting or web predictions will be mislabeled."
        )
    payload = json.dumps(labels, indent=2) + "\n"
    written: list[Path] = []
    for target in (out_dir, archive_dir):
        target.mkdir(parents=True, exist_ok=True)
        path = target / "labels.json"
        path.write_text(payload, encoding="utf-8")
        written.append(path)
    return written


def mirror_artifacts(src_dir: Path, dst_dir: Path) -> list[Path]:
    """Copy model.json + weight shards from src_dir to the archive dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    mirrored: list[Path] = []
    for artifact in sorted(src_dir.iterdir()):
        if artifact.is_file() and artifact.name != "labels.json":
            dest = dst_dir / artifact.name
            shutil.copy2(artifact, dest)
            mirrored.append(dest)
    if not mirrored:
        raise RuntimeError(f"No model artifacts found to mirror in {src_dir}")
    return mirrored

def convert_model(model_path: Path, out_dir: Path) -> None:
    """Load the .keras checkpoint and convert to TF.js layers-model."""
    try:
        import tensorflowjs as tfjs
    except ImportError:
        print(
            "ERROR: the `tensorflowjs` package is not installed, so the "
            "model cannot be converted on this machine.",
            file=sys.stderr,
        )
        print(f"Fix with:  {INSTALL_CMD}", file=sys.stderr)
        print(
            "Note: on this repo's Python 3.13 env that install backtracks "
            "and wants to DOWNGRADE packaging/tensorflow-hub - do NOT accept "
            "that. Run this script in Colab (or another env with "
            "tensorflowjs already present) instead.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    import tensorflow as tf

    print(f"[1/3] Loading checkpoint: {model_path}")
    model = tf.keras.models.load_model(model_path)
    n_outputs = model.output_shape[-1]
    if n_outputs != len(wl.BINS):
        raise ValueError(
            f"Checkpoint has {n_outputs} outputs but train.BINS has "
            f"{len(wl.BINS)} entries - refusing to export (predictions "
            "would be mislabeled)."
        )
    print(f"      loaded: {model.name} "
          f"in {model.input_shape} -> out {model.output_shape}")
    print(f"[2/3] Converting to TF.js layers-model format -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tfjs.converters.save_keras_model(model, str(out_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WasteLens model to TF.js")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="checkpoint path (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="TF.js output dir (default: %(default)s)")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model}\n"
            "Copy it from Colab Drive (MyDrive/WasteLens/models/checkpoints/) "
            "into models/checkpoints/, or pass --model <path>."
        )
    convert_model(args.model, args.out)  # fails fast here if no tensorflowjs
    written = write_labels_json(args.out, ARCHIVE_DIR)
    written += mirror_artifacts(args.out, ARCHIVE_DIR)

    print("[3/3] SUCCESS - TF.js export complete. Files written:")
    for path in sorted(written):
        size = path.stat().st_size
        print(f"      {path}  ({size:,} bytes)")
    print("labels.json order (must match web/index.html LABELS): "
          f"{list(wl.BINS)}")


if __name__ == "__main__":
    main()




