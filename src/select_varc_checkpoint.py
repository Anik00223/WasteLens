# src/select_varc_checkpoint.py
#
# WasteLens Iteration 7 - PRE-REGISTERED checkpoint-selection rule.
#
# THE RULE (fixed before any Iteration-7 training was started; identical to
# the text in the Iteration-7 brief):
#
#   Select the lowest-validation-loss checkpoint among epochs where the
#   validation BIN ACCURACY is at least 98% of the Variant B baseline:
#
#       floor = 0.9822 * 0.98 = 0.962556
#
# METRIC DEFINITION (also fixed in advance):
#   "validation bin accuracy" = top-1 accuracy of the bins head on the
#   SUPPORTED validation images only (eval_sets.json -> supported_val).
#   The training-history column val_bins_sparse_categorical_accuracy is
#   computed over the MIXED validation batch (supported + unsupported rows,
#   where unsupported rows have no valid bin target) and is therefore NOT a
#   bin-accuracy measurement; it is recorded but NOT used for selection.
#   "validation loss" = the per-epoch val_loss recorded in the training
#   history CSV (the quantity the trainer's ModelCheckpoint monitors).
#
# The TEST set is never touched by this script.
#
# Usage (from repo root), after a full training run:
#   py -3.13 -W ignore src/select_varc_checkpoint.py \
#       --history docs/rejection_experiment/variantc7_history.csv \
#       --epochs 10 \
#       --out docs/rejection_experiment/variantc7_selection.json

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

import train as wl
import train_rejection as tr

# --- PRE-REGISTERED CONSTANTS (do not change after seeing test results) ----
BASELINE_ACC = 0.9822        # Variant B baseline (registered in the brief)
FLOOR_RATIO = 0.98           # "at least 98% of the Variant B baseline"
EVAL_SETS_JSON = Path("docs/rejection_experiment/eval_sets.json")


def supported_val_bins_metrics(model, paths: list[str]) -> dict:
    """Bins-head accuracy + macro-F1 + mean reject on SUPPORTED val images.

    Ground-truth labels come from the same seed-42 split (train.py), so the
    measurement is exact; no test image is ever loaded.
    """
    root = wl.download_dataset()
    splits = wl.stratified_split(wl.remap_to_bins(wl.scan_images(root)))
    label_of = {str(p): int(l) for p, l in splits["val"]}
    y_true = np.array([label_of[p] for p in paths], dtype=int)
    n = len(paths)
    ds = tr.make_weighted_ds(
        np.array(paths), np.zeros(n, np.int32), np.zeros(n, np.float32),
        np.ones(n, np.float32), np.ones(n, np.float32), training=False)
    ds = ds.map(lambda x, y, w: x)
    out = model.predict(ds, verbose=0)
    y_pred = out["bins"].argmax(axis=1)
    per_class_f1 = []
    for c in range(len(wl.BINS)):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per_class_f1.append(
            2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "supported_val_bins_acc": float(np.mean(y_pred == y_true)),
        "supported_val_macro_f1": float(np.mean(per_class_f1)),
        "supported_val_mean_reject": float(np.mean(out["reject"].ravel())),
        "n_images": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply the PRE-REGISTERED Iteration-7 checkpoint rule: "
                    "lowest val_loss among epochs with supported-val bins "
                    "acc >= baseline*0.98")
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("models/checkpoints"))
    ap.add_argument("--pattern", default="wastelens_rej_varc7_{:02d}.keras")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--baseline-acc", type=float, default=BASELINE_ACC,
                    help="registered Variant B baseline (default 0.9822)")
    ap.add_argument("--floor-ratio", type=float, default=FLOOR_RATIO)
    args = ap.parse_args()

    floor = round(args.baseline_acc * args.floor_ratio, 6)
    print(f"PRE-REGISTERED RULE: lowest val_loss among epochs with "
          f"supported-val bins acc >= {args.baseline_acc} * "
          f"{args.floor_ratio} = {floor}")

    rows = list(csv.DictReader(args.history.open()))
    assert len(rows) >= args.epochs, "history shorter than --epochs"
    data = json.loads(EVAL_SETS_JSON.read_text())
    val_paths = data["supported_val"]

    per_epoch = []
    for ep in range(1, args.epochs + 1):
        ckpt = args.ckpt_dir / args.pattern.format(ep)
        hist = rows[ep - 1]
        entry = {
            "epoch": ep,
            "checkpoint": str(ckpt),
            "val_loss": float(hist["val_loss"]),
            "history_val_bins_acc_mixed": float(
                hist["val_bins_sparse_categorical_accuracy"]),
            "exists": ckpt.exists(),
        }
        if ckpt.exists():
            model = tf.keras.models.load_model(ckpt)
            entry.update(supported_val_bins_metrics(model, val_paths))
            ok = entry["supported_val_bins_acc"] >= floor
            print(f"  ep{ep:02d}  val_loss={entry['val_loss']:.6f}  "
                  f"sup-val acc={entry['supported_val_bins_acc']:.4f}  "
                  f"macroF1={entry['supported_val_macro_f1']:.4f}  "
                  f"meanRej={entry['supported_val_mean_reject']:.4f}  "
                  f"floor={'PASS' if ok else 'fail'}")
        else:
            print(f"  ep{ep:02d}  checkpoint MISSING: {ckpt}")
        per_epoch.append(entry)

    eligible = [e for e in per_epoch
                if e["exists"] and e["supported_val_bins_acc"] >= floor]
    record = {
        "rule": "lowest val_loss among epochs with supported-val bins acc "
                f">= {args.baseline_acc} * {args.floor_ratio}",
        "floor": floor,
        "per_epoch": per_epoch,
        "eligible_epochs": [e["epoch"] for e in eligible],
        "selected_epoch": None,
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"),
    }
    if not eligible:
        record["decision"] = ("NO epoch satisfied the pre-registered floor; "
                              "Variant B stays production (Outcome B/C)")
        print("\nNO eligible epoch - Variant B stays production.")
    else:
        best = min(eligible, key=lambda e: (e["val_loss"], e["epoch"]))
        record["selected_epoch"] = best["epoch"]
        record["selected_checkpoint"] = best["checkpoint"]
        record["selection_reason"] = (
            f"epoch {best['epoch']}: lowest val_loss "
            f"({best['val_loss']:.6f}) among epochs meeting the floor "
            f"(supported-val bins acc {best['supported_val_bins_acc']:.4f} "
            f">= {floor}); eligible epochs were "
            f"{[e['epoch'] for e in eligible]}")
        print(f"\nSELECTED epoch {best['epoch']} -> {best['checkpoint']}")
        print(f"  reason: {record['selection_reason']}")

    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
