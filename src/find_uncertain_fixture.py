# src/find_uncertain_fixture.py
#
# WasteLens Iteration 8 helper: find real supported-test images whose MEASURED
# verdict under a CANDIDATE model (its own calibrated threshold) is "uncertain"
# per the shipped page rule (reject < threshold AND (top < 0.60 OR margin <
# 0.50)). Used to stage a genuine uncertain-state fixture for the candidate UI
# click-through instead of assuming Variant-B-era ambiguity transfers.
#
# Usage:
#   py -3.13 -W ignore src/find_uncertain_fixture.py \
#       --model models/checkpoints/wastelens_rej_varc8s42_10.keras \
#       --threshold 0.0702 --limit 3 \
#       --out scratch/varc8s42_uncertain_fixtures.json

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

import train as wl
import train_rejection as tr

MIN_TOP = 0.60
MIN_MARGIN = 0.50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    data = json.loads(tr.EVAL_SETS_JSON.read_text())
    rows = sorted(data["supported_test"], key=lambda r: r[0])
    model = tf.keras.models.load_model(args.model)
    hits: list[dict] = []
    n_checked = 0
    for i in range(0, len(rows), 64):
        chunk = rows[i:i + 64]
        paths = [r[0] for r in chunk]
        n = len(paths)
        ds = tr.make_weighted_ds(
            np.array(paths), np.zeros(n, np.int32), np.zeros(n, np.float32),
            np.ones(n, np.float32), np.ones(n, np.float32),
            training=False).map(lambda x, y, w: x)
        out = model.predict(ds, verbose=0)
        bins, rej = out["bins"], out["reject"].ravel()
        for (path, y), b, r in zip(chunk, bins, rej):
            n_checked += 1
            top = float(np.max(b))
            margin = float(top - np.sort(b)[-2])
            if r < args.threshold and (top < MIN_TOP or margin < MIN_MARGIN):
                hits.append({
                    "path": path, "true_bin": wl.BINS[y],
                    "bins": [float(v) for v in b], "reject": float(r),
                    "top": top, "margin": margin,
                    "verdict": "uncertain"})
                print(f"  uncertain[{len(hits)}/{args.limit}] {path} "
                      f"top={top:.4f} margin={margin:.4f} reject={r:.4f}")
                if len(hits) >= args.limit:
                    break
        if len(hits) >= args.limit:
            break
    rec = {"model": str(args.model), "threshold": float(args.threshold),
           "rule": f"reject < threshold AND (top < {MIN_TOP} OR "
                   f"margin < {MIN_MARGIN})",
           "checked_images": n_checked,
           "supported_test_size": len(rows),
           "n_found": len(hits), "fixtures": hits,
           "generated_utc": datetime.now(timezone.utc).strftime(
               "%Y-%m-%d %H:%M:%S UTC")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"checked {n_checked}/{len(rows)} images, found {len(hits)} "
          f"uncertain -> {args.out}")


if __name__ == "__main__":
    main()
