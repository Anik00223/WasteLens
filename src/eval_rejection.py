# src/eval_rejection.py - part 1/3: header + helpers.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

import train as wl
import train_rejection as tr

FRR_POINTS = (0.01, 0.05, 0.10, 0.15, 0.20)
OUT_DIR = tr.OUT_DIR
EVAL_JSON = OUT_DIR / "rejection_metrics.json"
REPORT_MD = OUT_DIR / "rejection_report.md"


def batches(model, paths, batch=64):
    """Predict bins+reject for a path list; returns (probs, rej)."""
    probs, rej = [], []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        n = len(chunk)
        ds = tr.make_weighted_ds(
            np.array(chunk), np.zeros(n, np.int32), np.zeros(n, np.float32),
            np.ones(n, np.float32), np.ones(n, np.float32), training=False)
        ds = ds.map(lambda x, y, w: x)  # images only
        out = model.predict(ds, verbose=0)
        probs.append(out["bins"])
        rej.append(out["reject"].ravel())
    return np.concatenate(probs), np.concatenate(rej)
def evaluate_bins(model, sup):
    paths = [r[0] for r in sup]
    y_true = np.array([r[1] for r in sup])
    probs, _ = batches(model, paths)
    y_pred = probs.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "per_bin": {b: {"p": float(prec[i]), "r": float(rec[i]),
                        "f1": float(f1[i])} for i, b in enumerate(wl.BINS)},
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "baseline": {"accuracy": 0.9822, "macro_f1": 0.9742}}


def group_scores(model, group):
    return {src: batches(model, plist)[1] for src, plist in group.items()}
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate dual-head model")
    ap.add_argument("--model", type=Path,
                    default=tr.CKPT_DIR / "wastelens_rej_best.keras")
    ap.add_argument("--tag", default="",
                    help="suffix for the output files, so a second variant "
                         "never overwrites another's evidence (e.g. 'frozen' "
                         "-> rejection_metrics_frozen.json)")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    eval_json = OUT_DIR / f"rejection_metrics{suffix}.json"
    report_md = OUT_DIR / f"rejection_report{suffix}.md"
    data = json.loads(tr.EVAL_SETS_JSON.read_text())
    model = tf.keras.models.load_model(args.model)
    bins = evaluate_bins(model, data["supported_test"])
    sup_val_rej = batches(model, data["supported_val"])[1]
    sup_test_rej = batches(model, [r[0]
                                   for r in data["supported_test"]])[1]
    val_u = group_scores(model, data["val_unsup"])
    test_u = group_scores(model, data["test_unsup"])
    frozen = {f"{f:.2f}": float(np.quantile(sup_val_rej, 1.0 - f))
              for f in FRR_POINTS}
    op_table = {}
    for k, t in frozen.items():
        det = {s: float(np.mean(v >= t)) for s, v in test_u.items()}
        all_ood = np.concatenate(list(test_u.values()))
        op_table[k] = {"threshold": t, "per_source": det,
                       "ood_detect": float(np.mean(all_ood >= t)),
                       "id_frr": float(np.mean(sup_test_rej >= t))}
    y_sc = np.concatenate([sup_test_rej] + list(test_u.values()))
    y_tr = np.concatenate(
        [np.zeros_like(sup_test_rej)] +
        [np.ones_like(v) for v in test_u.values()])
    auroc = float(roc_auc_score(y_tr, y_sc))
    pr, rc, _ = precision_recall_curve(y_tr, y_sc)
    metrics = {"model": str(args.model), "bins": bins,
               "rejection": {"auroc": auroc, "auprc": float(auc(rc, pr)),
                             "operating_points": op_table,
                             "val_per_source_mean": {
                                 s: float(np.mean(v))
                                 for s, v in val_u.items()}}}
    eval_json.write_text(json.dumps(metrics, indent=2) + "\n")
    L = ["# WasteLens Iteration 4 - rejection experiment report", "",
         f"Model: `{args.model}`", "",
         "## 4-bin classification (supported test)",
         f"accuracy={bins['accuracy']:.4f} (baseline 0.9822), "
         f"macro-F1={bins['macro_f1']:.4f} (baseline 0.9742)", "",
         "| Bin | P | R | F1 |", "|---|---|---|---|"]
    for i, b in enumerate(wl.BINS):
        d = bins["per_bin"][b]
        L.append(f"| {b} | {d['p']:.4f} | {d['r']:.4f} | {d['f1']:.4f} |")
    L += ["", "## Rejection (thresholds frozen on VAL)",
          f"AUROC={metrics['rejection']['auroc']:.4f} "
          f"AUPRC={metrics['rejection']['auprc']:.4f}", "",
          "| FRR target | thr | OOD detect | test FRR |",
          "|---|---|---|---|"]
    for k, op in op_table.items():
        L.append(f"| {k} | {op['threshold']:.4f} | {op['ood_detect']:.4f} | "
                 f"{op['id_frr']:.4f} |")
    L += ["", "### Per-source OOD detection @5% FRR target",
          "| source | detect |", "|---|---|"]
    for s, v in op_table["0.05"]["per_source"].items():
        L.append(f"| {s} | {v:.4f} |")
    L.append("")
    report_md.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

