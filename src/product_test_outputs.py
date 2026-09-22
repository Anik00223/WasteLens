# src/product_test_outputs.py
#
# PURPOSE:
#   Deterministically select real product-test cases from the Iteration-4 eval
#   sets, run them through the SHIPPED Variant B Keras model, and write
#   docs/rejection_experiment/product_test_outputs.json for
#   src/test_decision_logic.js to replay through the exact web decision logic.
#
#   Expected states come from the MEASURED model outputs (never assumptions):
#   - supported_confident: supported-test images the measured model treats as
#     confident+non-rejected -> 'supported' (skipped false-rejections logged).
#   - supported_ambiguous: supported-test images with ambiguous bins and no
#     rejection -> 'uncertain'.
#   - clothes/shoes/nonwaste: highest-scoring rejection cases -> 'unsupported';
#     measured misses (reject < threshold) are recorded with expected=null so
#     the known detection rates (84.5% / 74.5% / 95.5% @ the shipped point)
#     are visible instead of hidden.
#   - collage: sampled as-is with expected=null - Iteration 4 measured only
#     21.1% collage detection at this operating point; never asserted perfect.
#   - synth_ood: the same 5 deterministic probes as src/validate_realworld.py,
#     with the measured rule's verdict as expected (logged either way).
#
# Run from the repo root:
#   py -3.13 -W ignore src/product_test_outputs.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

import train as wl
import train_rejection as tr
from validate_realworld import ood_probes, predict_array

EVAL_SETS_JSON = Path("docs/rejection_experiment/eval_sets.json")
OUT_JSON = Path("docs/rejection_experiment/product_test_outputs.json")
REJECT_THRESHOLD = 0.8774   # shipped operating point (op "0.01", val-frozen)
MIN_TOP = 0.60              # existing web ambiguity rule (unchanged)
MIN_MARGIN = 0.50


def rule(bins_row, reject) -> str:
    """Mirror of web/index.html decideVerdict (same constants)."""
    b = list(map(float, bins_row))
    top = max(b)
    margin = top - sorted(b)[-2]
    if float(reject) >= REJECT_THRESHOLD:
        return "unsupported"
    if top < MIN_TOP - 1e-9 or margin + 1e-9 < MIN_MARGIN:
        return "uncertain"
    return "supported"


def predict_batch(model, paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    n = len(paths)
    ds = tr.make_weighted_ds(
        np.array(paths), np.zeros(n, np.int32), np.zeros(n, np.float32),
        np.ones(n, np.float32), np.ones(n, np.float32), training=False)
    ds = ds.map(lambda x, y, w: x)
    out = model.predict(ds, verbose=0)
    return out["bins"], out["reject"].ravel()


def first_n(cases: list[dict], pred, n: int) -> tuple[list[dict], list[dict]]:
    """Split cases into the first n satisfying pred, and the rest."""
    hit, rest = [], []
    for c in cases:
        (hit if len(hit) < n and pred(c) else rest).append(c)
    return hit, rest


def main() -> None:
    data = json.loads(EVAL_SETS_JSON.read_text())
    model = tf.keras.models.load_model(
        Path("models/checkpoints/wastelens_rej_frozen_best.keras"))
    print("[1/4] Loaded Variant B checkpoint")

    cases: list[dict] = []

    # --- supported test: bins + rejection measured in one pass ---
    sup = sorted(data["supported_test"], key=lambda r: r[0])
    stride = max(1, len(sup) // 24)
    sample = sup[::stride][:24]
    bins_s, rej_s = predict_batch(model, [r[0] for r in sample])
    confident, ambiguous, false_rej = [], [], []
    for (path, y), b, r in zip(sample, bins_s, rej_s):
        st = rule(b, r)
        item = {"path": path, "bins": [float(v) for v in b],
                "reject": float(r), "true_bin": wl.BINS[y]}
        if r >= REJECT_THRESHOLD:
            false_rej.append(item)          # measured false rejection (1.38%)
        elif st == "uncertain":
            ambiguous.append(item)
        else:
            confident.append(item)
    conf_n = min(8, len(confident))
    amb_n = min(3, len(ambiguous))
    cases += [{"id": f"sup_conf_{i}", "group": "supported_confident",
               "path": it["path"], "ground_truth": it["true_bin"],
               "expected": "supported", **{k: it[k] for k in ("bins", "reject")}}
              for i, it in enumerate(confident[:conf_n])]
    cases += [{"id": f"sup_amb_{i}", "group": "supported_ambiguous",
               "path": it["path"], "ground_truth": it["true_bin"],
               "expected": "uncertain", **{k: it[k] for k in ("bins", "reject")}}
              for i, it in enumerate(ambiguous[:amb_n])]
    print(f"      supported sample 24: {len(confident)} confident, "
          f"{len(ambiguous)} ambiguous, {len(false_rej)} false-rejected")

    # --- unsupported sources: measured top scorers; misses kept as null ---
    for src, n_want in (("clothes", 3), ("shoes", 3), ("nonwaste", 3), ("collage", 5)):
        paths = sorted(data["val_unsup"][src])
        stride = max(1, len(paths) // 40)
        sample_u = paths[::stride][:40]
        bins_u, rej_u = predict_batch(model, sample_u)
        scored = sorted(zip(sample_u, bins_u, rej_u), key=lambda t: -t[2])
        items = [{"path": p, "bins": [float(v) for v in b], "reject": float(r)}
                 for p, b, r in scored]
        hit, rest = first_n(items, lambda it: it["reject"] >= REJECT_THRESHOLD, n_want)
        cases += [{"id": f"{src}_{i}", "group": f"ood_{src}",
                   "path": it["path"], "ground_truth": "unsupported",
                   "expected": "unsupported", **{k: it[k] for k in ("bins", "reject")}}
                  for i, it in enumerate(hit)]
        # measured misses: expected=null, state logged, never asserted
        cases += [{"id": f"{src}_miss_{i}", "group": f"ood_{src}_miss",
                   "path": it["path"], "ground_truth": "unsupported",
                   "expected": None, **{k: it[k] for k in ("bins", "reject")}}
                  for i, it in enumerate(rest[: 5 - len(hit)])]
        print(f"      {src}: {len(hit)}/{n_want} above threshold in sample "
              f"({len(rest[: 5 - len(hit)])} measured misses recorded)")

    # --- the 5 deterministic synthetic OOD probes (same as prior iterations) ---
    from validate_realworld import _to_model_input
    print("      synth probes:")
    for i, (name, arr01) in enumerate(ood_probes()):
        x = _to_model_input(arr01)
        full = model(x, training=False)
        bins_p = [float(v) for v in full["bins"].numpy()[0]]
        reject_p = float(full["reject"].numpy().ravel()[0])
        cases.append({
            "id": f"synth_ood_{i}", "group": "synth_ood",
            "path": name, "ground_truth": "unsupported",
            "expected": "unsupported" if reject_p >= REJECT_THRESHOLD else None,
            "bins": bins_p, "reject": reject_p})
        print(f"        {name}: reject={reject_p:.4f} -> "
              f"{'unsupported' if reject_p >= REJECT_THRESHOLD else 'NOT rejected (measured)'}")

    OUT_JSON.write_text(json.dumps({
        "model": "models/checkpoints/wastelens_rej_frozen_best.keras",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "reject_threshold": REJECT_THRESHOLD,
        "cases": cases}, indent=1) + "\n", encoding="utf-8")
    print(f"[4/4] wrote {len(cases)} cases -> {OUT_JSON}")


if __name__ == "__main__":
    main()
