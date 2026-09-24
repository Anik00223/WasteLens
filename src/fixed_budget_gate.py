# src/fixed_budget_gate.py
#
# WasteLens Iteration 8 - PRE-REGISTERED protocol constants and gate.
#
# THE PROTOCOL (fixed BEFORE any Iteration-8 training; text of record in
# docs/rejection_experiment/iteration8_protocol.md):
#
#   * fixed budget: 10 epochs, no early stopping, NO checkpoint selection;
#   * the candidate is ALWAYS epoch 10 of its run;
#   * validation classification gate, applied per seed:
#         val_bins_acc (supported-val, same split as Variant B's 0.9829)
#         >= 0.9829 - 0.25pp  =  0.9804
#   * test classification tolerance vs Variant B (project tolerance, the same
#     0.5pt used by the Iteration-4/5 and Iteration-6/7 gates):
#         test acc >= 0.9822 - 0.005  and  macro-F1 >= 0.9742 - 0.005
#   * "meaningfully better rejection" (registered operating points):
#         OOD detection @1% FRR >= 0.90      (Variant B: 0.8426)
#         collage detection @5% FRR >= 0.75 (Variant B: 0.5778)
#   * reproducibility: BOTH seeds must clear the validation floor; test
#     numbers are reported as mean +/- spread; a weak seed is never discarded
#     or swapped for the stronger one.
#
# The constants below are the single source of truth; the md/json artifacts
# are generated from them. Nothing here may be edited after training starts.
#
# Usage:
#   py -3.13 -W ignore src/fixed_budget_gate.py --model <candidate ckpt> \
#       --seed-label 42 --out docs/rejection_experiment/iteration8_gate_s42.json

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

import train_rejection as tr
from select_varc_checkpoint import supported_val_bins_metrics

# --- PRE-REGISTERED CONSTANTS (frozen before training) ----------------------
REGISTERED_BASELINE_VAL_ACC = 0.9829   # Variant B, supported-val (Iter-7 meas.)
REGISTERED_TOLERANCE_PP = 0.25         # percentage points
MIN_VAL_ACC = REGISTERED_BASELINE_VAL_ACC - REGISTERED_TOLERANCE_PP / 100.0
BASELINE_TEST = {"accuracy": 0.9822, "macro_f1": 0.9742}
TEST_ACC_TOLERANCE = 0.005             # project tolerance (0.5pt), unchanged
TEST_F1_TOLERANCE = 0.005
BASELINE_OOD_AT_1PCT = 0.8426
MIN_OOD_AT_1PCT = 0.90
BASELINE_COLLAGE_AT_5PCT = 0.5778
MIN_COLLAGE_AT_5PCT = 0.75
EPOCH_BUDGET = 10
CANDIDATE_EPOCH = 10

RULE = (
    "Fixed budget: 10 epochs, no early stopping, NO selection - the candidate "
    "is epoch 10. Validation classification gate: supported-val bins accuracy "
    f">= {REGISTERED_BASELINE_VAL_ACC} - {REGISTERED_TOLERANCE_PP}pp = "
    f"{MIN_VAL_ACC} (floor computed in code). Test tolerance vs Variant B: "
    f"accuracy >= {BASELINE_TEST['accuracy']} - {TEST_ACC_TOLERANCE} and "
    f"macro-F1 >= {BASELINE_TEST['macro_f1']} - {TEST_F1_TOLERANCE}. "
    f"Rejection must improve: OOD @1% FRR >= {MIN_OOD_AT_1PCT} "
    f"(B: {BASELINE_OOD_AT_1PCT}) and collage @5% FRR >= "
    f"{MIN_COLLAGE_AT_5PCT} (B: {BASELINE_COLLAGE_AT_5PCT}). BOTH seeds must "
    "clear the validation floor; no seed is discarded or swapped."
)


def validation_gate(val_metrics: dict) -> dict:
    """Apply the pre-registered validation classification floor."""
    acc = float(val_metrics["supported_val_bins_acc"])
    passed = acc >= MIN_VAL_ACC
    return {
        "supported_val_bins_acc": acc,
        "supported_val_macro_f1": float(val_metrics["supported_val_macro_f1"]),
        "supported_val_mean_reject": float(
            val_metrics["supported_val_mean_reject"]),
        "n_images": int(val_metrics["n_images"]),
        "floor": MIN_VAL_ACC,
        "baseline": REGISTERED_BASELINE_VAL_ACC,
        "tolerance_pp": REGISTERED_TOLERANCE_PP,
        "pass": bool(passed),
        "margin_pp": round((acc - MIN_VAL_ACC) * 100.0, 4),
        "verdict": ("PASS - candidate may be evaluated on test"
                    if passed else
                    "FAIL - seed fails the registration; Variant B stays "
                    "production (no epoch substitution allowed)"),
    }


def test_gate(test_metrics: dict, ood_at_1pct: float,
              collage_at_5pct: float) -> dict:
    """Apply the pre-registered test-side conditions (classification,
    rejection, collage)."""
    acc = float(test_metrics["accuracy"])
    f1 = float(test_metrics["macro_f1"])
    cls_ok = (acc >= BASELINE_TEST["accuracy"] - TEST_ACC_TOLERANCE and
              f1 >= BASELINE_TEST["macro_f1"] - TEST_F1_TOLERANCE)
    ood_ok = float(ood_at_1pct) >= MIN_OOD_AT_1PCT
    col_ok = float(collage_at_5pct) >= MIN_COLLAGE_AT_5PCT
    return {
        "test_accuracy": acc, "test_macro_f1": f1,
        "accuracy_delta_pp": round((acc - BASELINE_TEST["accuracy"]) * 100, 4),
        "macro_f1_delta_pp": round((f1 - BASELINE_TEST["macro_f1"]) * 100, 4),
        "classification_pass": bool(cls_ok),
        "ood_at_1pct_frr": float(ood_at_1pct),
        "ood_pass": bool(ood_ok),
        "collage_at_5pct_frr": float(collage_at_5pct),
        "collage_pass": bool(col_ok),
        "pass": bool(cls_ok and ood_ok and col_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply the PRE-REGISTERED Iteration-8 fixed-budget gate "
                    "to one seed's epoch-10 candidate (validation side).")
    ap.add_argument("--model", type=Path, required=True,
                    help="candidate checkpoint (must be the fixed epoch 10)")
    ap.add_argument("--seed-label", required=True,
                    help="seed label, e.g. '42' or '43'")
    ap.add_argument("--epoch", type=int, default=CANDIDATE_EPOCH,
                    help=f"candidate epoch (default {CANDIDATE_EPOCH}; fixed)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.epoch != CANDIDATE_EPOCH:
        raise SystemExit(f"protocol fixes the candidate epoch at "
                         f"{CANDIDATE_EPOCH}; got {args.epoch}")

    print(f"PRE-REGISTERED RULE: {RULE}")
    data = json.loads(tr.EVAL_SETS_JSON.read_text())
    model = tf.keras.models.load_model(args.model)
    val_metrics = supported_val_bins_metrics(model, data["supported_val"])
    gate = validation_gate(val_metrics)
    record = {
        "protocol": "Iteration 8 fixed-budget final-epoch Variant C",
        "rule": RULE,
        "constants": {
            "baseline_val_acc": REGISTERED_BASELINE_VAL_ACC,
            "tolerance_pp": REGISTERED_TOLERANCE_PP,
            "min_val_acc": MIN_VAL_ACC,
            "baseline_test": BASELINE_TEST,
            "test_acc_tolerance": TEST_ACC_TOLERANCE,
            "test_f1_tolerance": TEST_F1_TOLERANCE,
            "min_ood_at_1pct": MIN_OOD_AT_1PCT,
            "min_collage_at_5pct": MIN_COLLAGE_AT_5PCT,
            "epoch_budget": EPOCH_BUDGET,
            "candidate_epoch": CANDIDATE_EPOCH,
        },
        "seed_label": str(args.seed_label),
        "model": str(args.model),
        "validation": gate,
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"seed {args.seed_label}: sup-val acc="
          f"{gate['supported_val_bins_acc']:.4f} "
          f"macroF1={gate['supported_val_macro_f1']:.4f} "
          f"floor={gate['floor']:.4f} -> "
          f"{'PASS' if gate['pass'] else 'FAIL'} "
          f"({gate['margin_pp']:+.4f}pp)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
