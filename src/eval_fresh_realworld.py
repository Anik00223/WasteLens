# src/eval_fresh_realworld.py
#
# PURPOSE:
#   Evaluate the SHIPPED Variant C model on the genuinely fresh real-world
#   dataset built by src/build_fresh_set.py, exactly implementing LOOP 8,
#   LOOP 9, and LOOP 10 of Iteration 9.
#
#   Adheres to docs/rejection_experiment/fresh_eval_protocol.md:
#   - Shipped rejection threshold: 0.0702 (frozen, never tuned on fresh data)
#   - Ambiguity rule: top < 0.60 or margin < 0.50
#   - Reports classification on supported-confident, rejection across OOD,
#     multi-object scenes separately, and operating-point analysis.
#
# Run from repo root:
#   py -3.13 -W ignore src/eval_fresh_realworld.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from validate_realworld import ood_probes

MANIFEST_PATH = Path("docs/rejection_experiment/fresh_set_manifest.json")
MODEL_PATH = Path("models/checkpoints/wastelens_rej_shipped_best.keras")
OUT_METRICS = Path("docs/rejection_experiment/fresh_eval_metrics.json")
OUT_REPORT = Path("docs/rejection_experiment/fresh_eval_report.md")

SHIPPED_THRESHOLD = 0.0702
MIN_TOP = 0.60
MIN_MARGIN = 0.50

BINS = ["recyclable", "organic", "hazardous", "general trash"]


def preprocess_image(path: Path) -> np.ndarray:
    raw = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (224, 224), method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 127.5 - 1.0
    return img.numpy()


def decide_verdict(bins_p: list[float], reject_p: float, threshold: float) -> str:
    if reject_p >= threshold:
        return "unsupported"
    top = max(bins_p)
    margin = top - sorted(bins_p)[-2]
    if top < MIN_TOP - 1e-9 or margin + 1e-9 < MIN_MARGIN:
        return "uncertain"
    return "supported"


def compute_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0

    tpr = np.cumsum(y_true_sorted) / n_pos
    fpr = np.cumsum(1 - y_true_sorted) / n_neg
    
    auroc = float(np.trapezoid(tpr, fpr))
    prec = np.cumsum(y_true_sorted) / np.arange(1, len(y_true) + 1)
    auprc = float(np.sum(prec * y_true_sorted) / n_pos)
    return auroc, auprc
def main() -> None:
    print(f"[1/4] Loading manifest {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = manifest["entries"]

    print(f"[2/4] Loading shipped model {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"[3/4] Running inference on {len(items)} fresh images + 5 synthetic probes")
    scored_items: list[dict[str, Any]] = []
    
    for item in items:
        p = Path("scratch/fresh_set") / item["local_name"]
        arr = preprocess_image(p)
        x = np.expand_dims(arr, 0)
        out = model(x, training=False)
        bins_p = [float(v) for v in out["bins"].numpy()[0]]
        rej_p = float(out["reject"].numpy().ravel()[0])
        v_shipped = decide_verdict(bins_p, rej_p, SHIPPED_THRESHOLD)

        scored_items.append({
            **item,
            "bins": bins_p,
            "reject": rej_p,
            "verdict_shipped": v_shipped,
            "pred_bin": BINS[int(np.argmax(bins_p))],
            "top_prob": float(max(bins_p)),
            "margin": float(max(bins_p) - sorted(bins_p)[-2]),
        })

    # Synthetic probes
    synth_scored = []
    for name, arr01 in ood_probes():
        arr = (arr01.astype(np.float32) / 0.5) - 1.0  # [0,1] -> [-1,1]
        x = np.expand_dims(arr, 0)
        out = model(x, training=False)
        bins_p = [float(v) for v in out["bins"].numpy()[0]]
        rej_p = float(out["reject"].numpy().ravel()[0])
        v = decide_verdict(bins_p, rej_p, SHIPPED_THRESHOLD)
        synth_scored.append({
            "name": name,
            "bins": bins_p,
            "reject": rej_p,
            "verdict_shipped": v,
        })

    print("[4/4] Computing metrics across categories")
    
    # Split into groups
    supported_singles = [it for it in scored_items if it["label"] in BINS]
    ambiguous_items = [it for it in scored_items if it["group"] == "ambiguous"]
    ood_items = [it for it in scored_items if it["group"].startswith("ood-")]
    multi_items = [it for it in scored_items if it["group"] == "multi-object"]

    # --- Classification on supported single items ---
    y_true_sup = [BINS.index(it["label"]) for it in supported_singles]
    y_pred_sup = [int(np.argmax(it["bins"])) for it in supported_singles]
    
    n_sup = len(supported_singles)
    acc = sum(t == p for t, p in zip(y_true_sup, y_pred_sup)) / n_sup

    per_bin = {}
    f1_list = []
    cm = [[0]*4 for _ in range(4)]
    for t, p in zip(y_true_sup, y_pred_sup):
        cm[t][p] += 1
    
    for i, b_name in enumerate(BINS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(4) if r != i)
        fn = sum(cm[i][c] for c in range(4) if c != i)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_bin[b_name] = {"precision": prec, "recall": rec, "f1": f1, "count": sum(cm[i])}
        f1_list.append(f1)
    macro_f1 = sum(f1_list) / len(f1_list)

    # --- Rejection Metrics ---
    y_id_rej = np.array([it["reject"] for it in supported_singles])
    y_ood_rej = np.array([it["reject"] for it in ood_items])
    
    y_eval = np.concatenate([np.zeros(len(y_id_rej)), np.ones(len(y_ood_rej))])
    score_eval = np.concatenate([y_id_rej, y_ood_rej])
    auroc, auprc = compute_auroc_auprc(y_eval, score_eval)

    frr_shipped = float(np.mean(y_id_rej >= SHIPPED_THRESHOLD))
    ood_detect_shipped = float(np.mean(y_ood_rej >= SHIPPED_THRESHOLD))

    ood_by_role = {}
    for r in ["ood-objects", "ood-clothes", "ood-shoes", "ood-scenes"]:
        subset = [it for it in ood_items if it["group"] == r]
        ood_by_role[r] = {
            "count": len(subset),
            "detected": sum(it["reject"] >= SHIPPED_THRESHOLD for it in subset),
            "detection_rate": float(np.mean([it["reject"] >= SHIPPED_THRESHOLD for it in subset])),
            "mean_reject": float(np.mean([it["reject"] for it in subset])),
        }

    # Multi-object scenes (LOOP 9)
    multi_rej = np.array([it["reject"] for it in multi_items])
    multi_detect_shipped = float(np.mean(multi_rej >= SHIPPED_THRESHOLD))
    multi_details = {
        "count": len(multi_items),
        "detected": int(np.sum(multi_rej >= SHIPPED_THRESHOLD)),
        "detection_rate": multi_detect_shipped,
        "mean_reject": float(np.mean(multi_rej)),
        "verdicts": {v: sum(it["verdict_shipped"] == v for it in multi_items) for v in ["supported", "uncertain", "unsupported"]},
        "top_pred_bins": {b: sum(it["pred_bin"] == b for it in multi_items) for b in BINS},
    }

    # Ambiguous supported items
    amb_details = {
        "count": len(ambiguous_items),
        "verdicts": {v: sum(it["verdict_shipped"] == v for it in ambiguous_items) for v in ["supported", "uncertain", "unsupported"]},
        "mean_top_prob": float(np.mean([it["top_prob"] for it in ambiguous_items])),
        "mean_margin": float(np.mean([it["margin"] for it in ambiguous_items])),
        "mean_reject": float(np.mean([it["reject"] for it in ambiguous_items])),
    }

    synth_details = {
        "count": len(synth_scored),
        "detected": sum(p["reject"] >= SHIPPED_THRESHOLD for p in synth_scored),
        "items": synth_scored,
    }

    # --- Operating-Point Analysis (LOOP 10) ---
    operating_points = {}
    for target_frr in [0.01, 0.02, 0.05]:
        sorted_id = np.sort(y_id_rej)
        max_fr = int(np.floor(target_frr * len(sorted_id)))
        if max_fr == 0:
            t_cand = float(sorted_id[-1]) + 1e-5
        else:
            t_cand = float(sorted_id[-max_fr])
        
        actual_frr = float(np.mean(y_id_rej >= t_cand))
        actual_ood = float(np.mean(y_ood_rej >= t_cand))
        actual_multi = float(np.mean(multi_rej >= t_cand))
        operating_points[f"frr_{int(target_frr*100)}pct"] = {
            "target_frr": target_frr,
            "threshold": t_cand,
            "actual_frr": actual_frr,
            "ood_detection": actual_ood,
            "multi_detection": actual_multi,
        }

    metrics: dict[str, Any] = {
        "evaluation_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "model": str(MODEL_PATH),
        "shipped_threshold": SHIPPED_THRESHOLD,
        "classification": {
            "dataset": "fresh supported single-item waste",
            "n_images": n_sup,
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "per_bin": per_bin,
            "confusion_matrix": cm,
        },
        "rejection_shipped_threshold": {
            "threshold": SHIPPED_THRESHOLD,
            "id_n": len(supported_singles),
            "ood_n": len(ood_items),
            "auroc": auroc,
            "auprc": auprc,
            "false_rejection_rate": frr_shipped,
            "overall_ood_detection": ood_detect_shipped,
            "per_role": ood_by_role,
        },
        "multi_object_evaluation": multi_details,
        "ambiguous_evaluation": amb_details,
        "synthetic_probes": synth_details,
        "operating_points": operating_points,
        "detailed_predictions": scored_items,
    }

    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics to {OUT_METRICS}")

    report = f"""# WasteLens — Fresh Real-World Evaluation Report (Iteration 9)

**Evaluation Date:** {metrics['evaluation_utc']}  
**Model Under Test:** `{MODEL_PATH}` (Shipped Variant C, Seed 42, Epoch 10)  
**Shipped Rejection Threshold:** `{SHIPPED_THRESHOLD}`  
**Dataset:** Fresh 131 real-world images from untouched classes (0 overlap with train/val/test)

---

## 1. Classification Performance (Supported Single Items)

Evaluated on **{n_sup}** fresh supported single-item photographs across all 4 bins.

* **Accuracy:** `{acc:.4f}` ({sum(t == p for t, p in zip(y_true_sup, y_pred_sup))}/{n_sup})
* **Macro-F1:** `{macro_f1:.4f}`

### Per-Bin Performance
| Bin | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| **recyclable** | {per_bin['recyclable']['precision']:.4f} | {per_bin['recyclable']['recall']:.4f} | {per_bin['recyclable']['f1']:.4f} | {per_bin['recyclable']['count']} |
| **organic** | {per_bin['organic']['precision']:.4f} | {per_bin['organic']['recall']:.4f} | {per_bin['organic']['f1']:.4f} | {per_bin['organic']['count']} |
| **hazardous** | {per_bin['hazardous']['precision']:.4f} | {per_bin['hazardous']['recall']:.4f} | {per_bin['hazardous']['f1']:.4f} | {per_bin['hazardous']['count']} |
| **general trash** | {per_bin['general trash']['precision']:.4f} | {per_bin['general trash']['recall']:.4f} | {per_bin['general trash']['f1']:.4f} | {per_bin['general trash']['count']} |

### Confusion Matrix
(Rows = ground truth, Columns = predicted: recyclable, organic, hazardous, general trash)
```
{cm}
```

---

## 2. Rejection Performance at Shipped Threshold (0.0702)

* **Rejection AUROC:** `{auroc:.4f}`
* **Rejection AUPRC:** `{auprc:.4f}`
* **False Rejection Rate (FRR) on Supported Waste:** `{frr_shipped*100:.2f}%` ({int(np.sum(y_id_rej >= SHIPPED_THRESHOLD))}/{len(y_id_rej)})
* **Overall OOD Detection Rate:** `{ood_detect_shipped*100:.2f}%` ({int(np.sum(y_ood_rej >= SHIPPED_THRESHOLD))}/{len(y_ood_rej)})

### OOD Category Breakdown
| Category | N | Detected | Detection Rate | Mean Reject Score |
|---|---:|---:|---:|---:|
| **Non-waste objects** (keyboards, bikes, robots) | {ood_by_role['ood-objects']['count']} | {ood_by_role['ood-objects']['detected']} | {ood_by_role['ood-objects']['detection_rate']*100:.1f}% | {ood_by_role['ood-objects']['mean_reject']:.4f} |
| **Clothes** (shirts) | {ood_by_role['ood-clothes']['count']} | {ood_by_role['ood-clothes']['detected']} | {ood_by_role['ood-clothes']['detection_rate']*100:.1f}% | {ood_by_role['ood-clothes']['mean_reject']:.4f} |
| **Shoes** | {ood_by_role['ood-shoes']['count']} | {ood_by_role['ood-shoes']['detected']} | {ood_by_role['ood-shoes']['detection_rate']*100:.1f}% | {ood_by_role['ood-shoes']['mean_reject']:.4f} |
| **Environmental scenes** (parks, streets) | {ood_by_role['ood-scenes']['count']} | {ood_by_role['ood-scenes']['detected']} | {ood_by_role['ood-scenes']['detection_rate']*100:.1f}% | {ood_by_role['ood-scenes']['mean_reject']:.4f} |

---

## 3. Real Multi-Object Scene Evaluation (LOOP 9)

Evaluated on **{multi_details['count']}** real, natural photographs of multi-object scenes (beach litter, flea markets, cluttered waste).

* **Detection Rate @ Shipped Threshold (0.0702):** `{multi_details['detection_rate']*100:.1f}%` ({multi_details['detected']}/{multi_details['count']})
* **Mean Rejection Score:** `{multi_details['mean_reject']:.4f}`
* **Final Verdict Breakdown:**
  * Unsupported: {multi_details['verdicts']['unsupported']}
  * Uncertain: {multi_details['verdicts']['uncertain']}
  * Supported: {multi_details['verdicts']['supported']}

---

## 4. Ambiguous Supported Items Evaluation

Evaluated on **{amb_details['count']}** visually challenging waste items (paper towels, pizza boxes, cigarette butts).

* **Final Verdict Breakdown:**
  * Supported: {amb_details['verdicts']['supported']}
  * Uncertain (Ambiguous triggered): {amb_details['verdicts']['uncertain']}
  * Unsupported (False Rejection): {amb_details['verdicts']['unsupported']}
* **Mean Top Probability:** `{amb_details['mean_top_prob']:.4f}`
* **Mean Margin:** `{amb_details['mean_margin']:.4f}`

---

## 5. Synthetic Flat Probes Evaluation

* **Probes Rejected:** {synth_details['detected']}/{synth_details['count']}
"""
    details_synth = ""
    for p in synth_scored:
        details_synth += f"  * `{p['name']}`: reject = {p['reject']:.4f} -> {p['verdict_shipped']}\n"

    report += details_synth
    report += f"""
---

## 6. Operating-Point Analysis (LOOP 10)

| Operating Point | Target FRR | Threshold | Measured FRR | Measured OOD Detection | Measured Multi-Object Detection |
|---|---:|---:|---:|---:|---:|
| **Shipped Point** | ~1% | `{SHIPPED_THRESHOLD:.4f}` | `{frr_shipped*100:.2f}%` | `{ood_detect_shipped*100:.2f}%` | `{multi_details['detection_rate']*100:.1f}%` |
| **Fresh 1% FRR** | 1% | `{operating_points['frr_1pct']['threshold']:.4f}` | `{operating_points['frr_1pct']['actual_frr']*100:.2f}%` | `{operating_points['frr_1pct']['ood_detection']*100:.2f}%` | `{operating_points['frr_1pct']['multi_detection']*100:.2f}%` |
| **Fresh 2% FRR** | 2% | `{operating_points['frr_2pct']['threshold']:.4f}` | `{operating_points['frr_2pct']['actual_frr']*100:.2f}%` | `{operating_points['frr_2pct']['ood_detection']*100:.2f}%` | `{operating_points['frr_2pct']['multi_detection']*100:.2f}%` |
| **Fresh 5% FRR** | 5% | `{operating_points['frr_5pct']['threshold']:.4f}` | `{operating_points['frr_5pct']['actual_frr']*100:.2f}%` | `{operating_points['frr_5pct']['ood_detection']*100:.2f}%` | `{operating_points['frr_5pct']['multi_detection']*100:.2f}%` |
"""

    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Saved report to {OUT_REPORT}")


if __name__ == "__main__":
    main()

