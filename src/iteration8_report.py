# src/iteration8_report.py
#
# WasteLens Iteration 8 - aggregation + OUTCOME mapping for the pre-registered
# fixed-budget protocol. Written BEFORE any seed result existed; it reads only
# the gate records and the evaluation artifacts, applies the constants from
# src/fixed_budget_gate.py, and emits docs/rejection_experiment/
# iteration8_results.json.
#
# OUTCOME PRECEDENCE (fixed in advance; the protocol defines the three
# outcomes, this is the deterministic mapping used to pick one):
#   1. any seed fails the validation floor OR the test classification
#      tolerance            -> outcome "B" (reject; classification unstable /
#                              below the registered requirement)
#   2. else, any seed fails a registered rejection bar (OOD @1% FRR >= 0.90,
#      collage @5% FRR >= 0.75) -> outcome "C" (promising but inconclusive;
#                              rejection unstable across seeds)
#   3. else                   -> outcome "A" (both seeds pass everything)
# No seed is discarded, averaged away, or replaced by the stronger one.
#
# Usage:
#   py -3.13 -W ignore src/iteration8_report.py \
#       --seed-spec 42=<gate_s42.json>=<rejection_metrics_varc8s42.json> \
#       --seed-spec 43=<gate_s43.json>=<rejection_metrics_varc8s43.json> \
#       --out docs/rejection_experiment/iteration8_results.json

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fixed_budget_gate import (BASELINE_COLLAGE_AT_5PCT, BASELINE_OOD_AT_1PCT,
                               MIN_COLLAGE_AT_5PCT, MIN_OOD_AT_1PCT,
                               REGISTERED_BASELINE_VAL_ACC, RULE, test_gate)


def load_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("=")
    if len(parts) != 3:
        raise SystemExit(f"--seed-spec must be label=gate_json=metrics_json, "
                         f"got {spec!r}")
    return parts[0], Path(parts[1]), Path(parts[2])


def summarise(values: list[float]) -> dict:
    a = np.array(values, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max()), "spread": float(a.max() - a.min()),
            "values": [float(v) for v in a]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Iteration-8 aggregation")
    ap.add_argument("--seed-spec", action="append", required=True,
                    help="label=gate_json=metrics_json (repeat per seed)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    per_seed = {}
    for spec in args.seed_spec:
        label, gate_path, metrics_path = load_spec(spec)
        gate = json.loads(gate_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        op = metrics["rejection"]["operating_points"]
        op01, op05 = op["0.01"], op["0.05"]
        tg = test_gate(metrics["bins"], ood_at_1pct=op01["ood_detect"],
                       collage_at_5pct=op05["per_source"]["collage"])
        per_seed[label] = {
            "gate_json": str(gate_path), "metrics_json": str(metrics_path),
            "model": metrics["model"],
            "validation": gate["validation"],
            "test": tg,
            "auroc": float(metrics["rejection"]["auroc"]),
            "auprc": float(metrics["rejection"]["auprc"]),
            "ood_at_1pct_frr": float(op01["ood_detect"]),
            "ood_at_5pct_frr": float(op05["ood_detect"]),
            "collage_at_1pct_frr": float(op01["per_source"]["collage"]),
            "collage_at_5pct_frr": float(op05["per_source"]["collage"]),
            "per_source_ood_at_5pct": op05["per_source"],
            "calibration": {
                "threshold_at_1pct": float(op01["threshold"]),
                "val_frr_at_1pct": float(op01["val_frr"]),
                "test_frr_at_1pct": float(op01["id_frr"]),
                "threshold_at_5pct": float(op05["threshold"]),
                "val_frr_at_5pct": float(op05["val_frr"]),
                "test_frr_at_5pct": float(op05["id_frr"]),
                "protocol": "same as Variants A/B/C: threshold = val quantile "
                            "at each FRR target, frozen on supported-val, "
                            "applied unchanged to the untouched test set",
            },
            "per_bin": metrics["bins"]["per_bin"],
            "confusion_matrix": metrics["bins"]["confusion_matrix"],
        }

    labels = list(per_seed)
    val_pass = {s: bool(per_seed[s]["validation"]["pass"]) for s in labels}
    cls_pass = {s: bool(per_seed[s]["test"]["classification_pass"])
                for s in labels}
    rej_bar = {s: bool(per_seed[s]["test"]["ood_pass"] and
                       per_seed[s]["test"]["collage_pass"]) for s in labels}

    both_val = all(val_pass.values())
    if not both_val or not all(cls_pass.values()):
        outcome, why = "B", ("classification below the registered requirement "
                             "or unstable across seeds -> reject Variant C, "
                             "keep Variant B")
    elif not all(rej_bar.values()):
        outcome, why = "C", ("classification passes for both seeds but the "
                             "registered rejection/collage bar is not met for "
                             "every seed -> promising but inconclusive, do not "
                             "ship")
    else:
        outcome, why = "A", ("both seeds clear the validation floor and all "
                             "registered test-side conditions -> eligible to "
                             "ship the protocol-defined epoch-10 candidate")

    record = {
        "protocol": "Iteration 8 fixed-budget final-epoch Variant C",
        "rule": RULE,
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"),
        "baselines": {
            "variant_b_val_acc": REGISTERED_BASELINE_VAL_ACC,
            "variant_b_ood_at_1pct_frr": BASELINE_OOD_AT_1PCT,
            "variant_b_collage_at_5pct_frr": BASELINE_COLLAGE_AT_5PCT,
            "registered_min_ood_at_1pct": MIN_OOD_AT_1PCT,
            "registered_min_collage_at_5pct": MIN_COLLAGE_AT_5PCT,
        },
        "seed_checks": {"validation_floor_pass": val_pass,
                        "test_classification_pass": cls_pass,
                        "rejection_bar_pass": rej_bar},
        "per_seed": per_seed,
        "seed_consistency": {
            "both_seeds_pass_validation_floor": bool(both_val),
            "test_accuracy": summarise(
                [per_seed[s]["test"]["test_accuracy"] for s in labels]),
            "test_macro_f1": summarise(
                [per_seed[s]["test"]["test_macro_f1"] for s in labels]),
            "auroc": summarise([per_seed[s]["auroc"] for s in labels]),
            "ood_at_1pct_frr": summarise(
                [per_seed[s]["ood_at_1pct_frr"] for s in labels]),
            "collage_at_5pct_frr": summarise(
                [per_seed[s]["collage_at_5pct_frr"] for s in labels]),
            "note": "no seed is discarded; spread is reported, not averaged "
                    "away",
        },
        "outcome": outcome,
        "outcome_reason": why,
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"{'metric':30s}" + "".join(f"{('seed ' + s):>14s}"
                                      for s in labels))
    for key in ("auroc", "ood_at_1pct_frr", "collage_at_5pct_frr"):
        print(f"{key:30s}" + "".join(
            f"{per_seed[s][key]:14.4f}" for s in labels))
    for key in ("test_accuracy", "test_macro_f1"):
        print(f"{key:30s}" + "".join(
            f"{per_seed[s]['test'][key]:14.4f}" for s in labels))
    print(f"{'val_floor_pass':30s}" +
          "".join(f"{str(val_pass[s]):>14s}" for s in labels))
    print(f"OUTCOME {outcome}: {why}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
