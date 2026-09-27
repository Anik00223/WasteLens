# src/eval_head_adaptation.py
#
# WasteLens Iteration 11/12 - LOOP 10/11/12: evaluator for a
# classification-head-only candidate (Iteration 11: plain CE objective;
# Iteration 12: distillation-constrained objective - the evaluator is
# objective-agnostic: it scores a SAVED candidate file either way).
#
# Implements docs/rejection_experiment/iteration11_head_adaptation_protocol.md
# (and, via --protocol/--tag/--gates overrides, the Iteration-12 protocol):
#   * dataset-drift check: re-runs the Iteration-10 leakage audit and asserts it
#     is identical to the committed one except `generated_utc` (§3),
#   * existing-benchmark evaluation (G1 + rejection preservation part of G4),
#   * fresh-benchmark evaluation (G2/G3/G4/G5),
#   * the LOOP-12 REJECTION EQUIVALENCE test: candidate vs shipped rejection
#     probabilities over fresh 131 + original test pools; protocol §6 requires
#     max |Δreject| <= 1e-6 and 0 changed reject verdicts at 0.0702,
#   * writes {tag}s{seed}_eval.json / _gates.json / _report.md and, with
#     --summary, {tag}_seed_consistency.json + iteration{nn}_decision.md
#     (defaults: tag=head11, protocol=iteration11 file, gates=head11 table).
#
# Iteration-12 gate overrides (--gates head12_gates.json) implement the
# literal protocol §7 table: G1/G2/G3 identical; G4 splits into G4 (class
# coverage, Iter-11 min - 5pt) + G5/G6 (rejection bars) with G6 fresh OOD
# as its own gate; G7 both-seeds G1-G4; G8 browser parity. Metric code is
# shared; only the bar table and artifact names change.
#
# METRIC CODE IS REUSED, NOT REWRITTEN: eval_adaptation.candidate_metrics
# (which itself calls eval_rejection / eval_fresh_realworld definitions).
# The threshold stays 0.0702 everywhere (protocol §6 - no threshold rescue).

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

import eval_adaptation as ea
import eval_fresh_realworld as fr
import eval_rejection as er
import train_adaptation as ta
import train_head_adaptation as th
import train_rejection as tr

OUT_DIR = tr.OUT_DIR
SHIPPED = Path("models/checkpoints/wastelens_rej_shipped_best.keras")
SHIPPED_THRESHOLD = ea.SHIPPED_THRESHOLD          # 0.0702, production constant
ADAPT10_SEEDS = (42, 43)

# Default gate table = pre-registered Iteration-11 gates (protocol §6).
# --gates head12_gates.json swaps in the literal Iteration-12 table (§7);
# see resolve_gate_table() below. Numbers are literal in both tables.
GATE = {
    "G1": {"orig_acc_min": 0.9763, "orig_macro_f1_min": 0.9671},
    "G2": {"fresh_acc_min": 0.5500},
    "G3": {"min_recall": {"recyclable": 0.7143, "organic": 0.3750,
                          "hazardous": 0.4000, "general trash": 0.2727},
           "baseline_rule": "min over the two Iteration-10 seeds"},
    "G4": {"orig_auroc_min": 0.99466, "orig_ood_detect_min": 0.99139,
           "orig_id_frr_max": 0.02284, "fresh_ood_detect_min": 0.66940,
           "fresh_auroc_min": 0.74770},
    "G5": {"fresh_frr_max": 0.33830, "reject_max_abs_diff": 1e-6,
           "reject_verdict_changes_max": 0},
}
# Literal Iteration-12 gate table (task spec + protocol §7): G1 original
# accuracy, G2 original macro-F1, G3 fresh acc, G4 class coverage
# (Iter-11 two-seed min, minus 5pt), G5 rejection, G6 fresh OOD as its
# own gate, G7 both-seeds G1-G4, G8 browser parity (scored from evidence).
GATE_HEAD12 = {
    "G1": {"orig_acc_min": 0.9763},
    "G2": {"orig_macro_f1_min": 0.9671},
    "G3": {"fresh_acc_min": 0.5500},
    "G4": {"min_recall": {"recyclable": 0.7119, "organic": 0.3875,
                          "hazardous": 0.4833, "general trash": 0.4045},
           "baseline_rule": "Iteration-11 two-seed minimum minus 5pt"},
    "G5": {"orig_auroc_min": 0.99466, "orig_ood_detect_min": 0.99139,
           "orig_id_frr_max": 0.02284, "fresh_auroc_min": 0.74770,
           "fresh_frr_max": 0.33830, "reject_max_abs_diff": 1e-6,
           "reject_verdict_changes_max": 0},
    "G6": {"fresh_ood_detect_min": 0.66940},
}


def resolve_gate_table(name: str | None) -> dict:
    """Select the pre-registered gate table by --gates flag (no post-hoc edits)."""
    if name in (None, "", "head11_gates.json"):
        return GATE
    if name == "head12_gates.json":
        return GATE_HEAD12
    raise SystemExit(f"unknown --gates table {name!r} (expected "
                     "head11_gates.json | head12_gates.json)")
MEANINGFUL_IMPROVEMENT_MIN = 0.4600               # protocol §7
ITER10_FRR_OBSERVED = {"42": 0.19047619047619047, "43": 0.30158730158730157}
# Iteration-10 fresh supported accuracy per seed (adapt10_seed_consistency.json)
ITER10_FRESH_ACC = {"42": 0.5238095238095238, "43": 0.5396825396825397}


def audit_iter11(tag: str = "head11") -> tuple[dict, bool]:
    """Re-run the leakage audit and prove the dataset did not move (§3).

    `eval_adaptation.run_audit()` rewrites the committed audit file, so its
    bytes are saved first and restored afterwards: no iteration may change
    an Iteration-10 artifact. The fresh audit is written to
    `{tag}_audit.json` with the comparison result attached.
    """
    committed_path = OUT_DIR / "adaptation_eval_audit.json"
    committed_bytes = committed_path.read_bytes()
    try:
        fresh = ea.run_audit()
    finally:
        committed_path.write_bytes(committed_bytes)   # keep the commit clean
    strip = lambda d: {k: v for k, v in d.items() if k != "generated_utc"}
    identical = strip(fresh) == strip(json.loads(committed_bytes))
    fresh["iteration11_check"] = {
        "identical_to_iteration10_audit_except_timestamp": bool(identical),
        "note": "protocol §3: the adaptation dataset and benchmarks must be "
                "unchanged between Iteration 10 and Iteration 11/12",
    }
    (OUT_DIR / f"{tag}_audit.json").write_text(
        json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    if not identical:
        raise SystemExit("dataset drift: the audit differs from "
                         "the committed Iteration-10 audit (see "
                         f"docs/rejection_experiment/{tag}_audit.json)")
    print(f"[audit] clean={fresh['clean']} "
          f"identical_to_iteration10={identical}")
    return fresh, identical


def fresh_paths() -> list[Path]:
    manifest = json.loads(ea.FRESH_MANIFEST.read_text(encoding="utf-8"))
    return [ea.FRESH_DIR / it["local_name"] for it in manifest["entries"]]


def original_pool_paths() -> dict[str, list[str]]:
    """Original test pools exactly as eval_rejection defines them."""
    data = json.loads(tr.EVAL_SETS_JSON.read_text(encoding="utf-8"))
    pools = {"supported_test": [r[0] for r in data["supported_test"]]}
    for src, plist in data["test_unsup"].items():
        pools[f"unsup_{src}"] = list(plist)
    return pools


# --- LOOP 12: rejection equivalence -----------------------------------------

def rejection_equivalence(model_path: Path, seed: int,
                          tag: str = "head11",
                          gate: dict | None = None,
                          protocol: str = "iteration11_head_adaptation_protocol.md §6 G5") -> dict:
    """Prove the frozen rejection pathway did not move, at file level.

    Compares candidate vs shipped `reject` probabilities (and bins, for
    information) over the fresh 131 files and every original test pool.
    Head11 §6/G5 (head12 §7/G5): max |Δreject| <= 1e-6, 0 verdict flips.
    """
    gate = gate if gate is not None else GATE
    cand = tf.keras.models.load_model(model_path)
    shipped = tf.keras.models.load_model(SHIPPED)
    thr = SHIPPED_THRESHOLD
    manifest = json.loads(ea.FRESH_MANIFEST.read_text(encoding="utf-8"))
    fresh_rows = []
    for item in manifest["entries"]:
        arr = fr.preprocess_image(ea.FRESH_DIR / item["local_name"])
        x = np.expand_dims(arr, 0)
        a = cand(x, training=False)
        b = shipped(x, training=False)
        fresh_rows.append({
            "local_name": item["local_name"], "group": item["group"],
            "bins_cand": [float(v) for v in a["bins"].numpy()[0]],
            "bins_shipped": [float(v) for v in b["bins"].numpy()[0]],
            "reject_cand": float(a["reject"].numpy().ravel()[0]),
            "reject_shipped": float(b["reject"].numpy().ravel()[0]),
        })
    pred_c = np.array([r["reject_cand"] for r in fresh_rows])
    pred_s = np.array([r["reject_shipped"] for r in fresh_rows])
    fresh_flips = [r["local_name"] for r in fresh_rows
                   if (r["reject_cand"] >= thr) != (r["reject_shipped"] >= thr)]
    fresh_verdict_flips = [
        r["local_name"] for r in fresh_rows
        if fr.decide_verdict(r["bins_cand"], r["reject_cand"], thr)
        != fr.decide_verdict(r["bins_shipped"], r["reject_shipped"], thr)]

    pools = original_pool_paths()
    pool_summary, diffs = {}, [np.abs(pred_c - pred_s)]
    flips_all = len(fresh_flips)
    for name, paths in pools.items():
        pc, rc = er.batches(cand, paths)
        ps, rs = er.batches(shipped, paths)
        d = np.abs(rc - rs)
        pool_summary[name] = {
            "n": len(paths), "reject_max_abs_diff": float(d.max()),
            "reject_mean_abs_diff": float(d.mean()),
            "reject_verdict_changes_at_threshold": int(
                np.sum((rc >= thr) != (rs >= thr))),
            "bins_argmax_changes": int(np.sum(pc.argmax(1) != ps.argmax(1))),
        }
        diffs.append(d)
        flips_all += pool_summary[name]["reject_verdict_changes_at_threshold"]
    all_d = np.concatenate(diffs)
    max_all = float(all_d.max())

    # File-level freeze check: the saved candidate must carry byte-identical
    # frozen tensors (proves the .keras round trip, not just the in-memory run).
    digests = th.sha256_weights(cand.weights)
    proof_path = OUT_DIR / f"{tag}s{seed}_freeze_proof.json"
    file_ok, file_detail = None, "freeze proof file not found"
    if proof_path.exists():
        before = json.loads(proof_path.read_text(encoding="utf-8"))["before"]
        bad = sorted(p for p, h in before["frozen_digests"].items()
                     if digests.get(p) != h)
        file_ok = not bad
        file_detail = {
            "frozen_tensors_in_file": len(before["frozen_digests"]),
            "mismatches": bad,
            "frozen_aggregate_sha256_in_file": th.digest_aggregate(
                {p: h for p, h in digests.items()
                 if p not in th.TRAINABLE_PATHS}),
            "before_aggregate": before["frozen_aggregate_sha256"],
        }
    result = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": f"docs/rejection_experiment/{protocol}",
        "candidate": str(model_path), "candidate_md5": tr.md5_file(model_path),
        "shipped": str(SHIPPED), "threshold": thr,
        "fresh": {
            "n": len(fresh_rows),
            "reject_max_abs_diff": float(np.abs(pred_c - pred_s).max()),
            "reject_mean_abs_diff": float(np.abs(pred_c - pred_s).mean()),
            "reject_bit_exact": bool(np.array_equal(pred_c, pred_s)),
            "reject_verdict_changes_at_threshold": len(fresh_flips),
            "reject_verdict_change_examples": fresh_flips[:10],
            "bins_argmax_changes": int(sum(
                int(np.argmax(r["bins_cand"])) != int(np.argmax(
                    r["bins_shipped"])) for r in fresh_rows)),
            "three_way_verdict_changes": len(fresh_verdict_flips),
            "three_way_verdict_change_examples": fresh_verdict_flips[:10],
            "rows": fresh_rows,
        },
        "original_pools": pool_summary,
        "overall": {"reject_max_abs_diff": max_all,
                    "reject_mean_abs_diff": float(all_d.mean()),
                    "reject_verdict_changes_at_threshold": flips_all},
        "file_level_freeze_check": {"pass": file_ok, "detail": file_detail},
        "pass": bool(max_all <= gate["G5"]["reject_max_abs_diff"]
                     and flips_all == gate["G5"]["reject_verdict_changes_max"]
                     and file_ok is True),
        "rule": f"max |Δreject| <= {gate['G5']['reject_max_abs_diff']}, "
                f"{gate['G5']['reject_verdict_changes_max']} changed reject "
                f"verdicts, frozen tensors byte-identical in the saved file",
        "notes": "Fresh rows are scored one image at a time (the baseline's "
                 "exact preprocessing); original pools use "
                 "eval_rejection.batches. 0.0 is the expected Δ: the reject "
                 "path is frozen and every input is identical.",
    }
    (OUT_DIR / f"{tag}_rejection_equivalence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[equivalence] max |delta(reject)| = "
          f"{result['overall']['reject_max_abs_diff']:.3e}, reject verdict "
          f"changes = {flips_all}, fresh bit-exact = "
          f"{result['fresh']['reject_bit_exact']}, file freeze = {file_ok} "
          f"-> pass={result['pass']}")
    return result


# --- Pre-registered gates (§6) ----------------------------------------------

def browser_gate(tag: str = "head11") -> dict:
    """G6/G8 scored from the export/parity/UI evidence file (written later)."""
    path = OUT_DIR / f"{tag}_browser.json"
    if not path.exists():
        return {"status": "pending",
                "rule": "TFJS export + atol<=2e-5 parity (local + HTTP) + "
                        "headless UI click-through, zero console errors",
                "pass": None,
                "note": "run the export/parity/UI step, then re-run --summary"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {**data, "status": "scored", "pass": bool(data.get("pass"))}


def gates_for(cand: dict, base: dict, equiv: dict,
              gate: dict | None = None, tag: str = "head11") -> dict:
    """Pre-registered gates for one seed (browser gate from evidence).

    Head11 table: G1 orig acc+F1, G2 fresh acc, G3 class recall, G4
    rejection+freshOOD, G5 FRR+isolation, G6 browser evidence.
    Head12 table: G1 orig acc, G2 orig F1, G3 fresh acc, G4 class recall,
    G5 rejection+FRR+isolation, G6 fresh OOD (own gate), G8 browser
    evidence lands in the same G6 evidence slot (renamed in the report).
    """
    gate = gate if gate is not None else GATE
    is12 = "G6" in gate and "fresh_ood_detect_min" in gate["G6"]
    o, f = cand["original"], cand["fresh"]
    ob, fb = base["original"], base["fresh"]
    c, r = o["classification"], o["rejection"]
    fc, frj = f["classification"], f["rejection"]
    if is12:
        g1 = {"orig_accuracy": c["accuracy"],
              "pass": bool(c["accuracy"] >= gate["G1"]["orig_acc_min"]),
              "rule": f"orig acc >= {gate['G1']['orig_acc_min']}"}
        g2 = {"orig_macro_f1": c["macro_f1"],
              "pass": bool(c["macro_f1"] >= gate["G2"]["orig_macro_f1_min"]),
              "rule": f"orig macro-F1 >= {gate['G2']['orig_macro_f1_min']}"}
        g3 = {"fresh_accuracy": fc["accuracy"],
              "fresh_macro_f1": fc["macro_f1"],
              "pass": bool(fc["accuracy"] >= gate["G3"]["fresh_acc_min"]),
              "rule": f"fresh supported acc >= {gate['G3']['fresh_acc_min']}"}
        gkey_cov, gkey_rej = "G4", "G5"
    else:
        g1 = {"orig_accuracy": c["accuracy"], "orig_macro_f1": c["macro_f1"],
              "pass": bool(c["accuracy"] >= gate["G1"]["orig_acc_min"]
                           and c["macro_f1"] >= gate["G1"]["orig_macro_f1_min"]),
              "rule": f"orig acc >= {gate['G1']['orig_acc_min']} and macro-F1 "
                      f">= {gate['G1']['orig_macro_f1_min']}"}
        g2 = {"fresh_accuracy": fc["accuracy"],
              "fresh_macro_f1": fc["macro_f1"],
              "pass": bool(fc["accuracy"] >= gate["G2"]["fresh_acc_min"]),
              "rule": f"fresh supported acc >= {gate['G2']['fresh_acc_min']}"}
        gkey_cov, gkey_rej = "G3", "G4"
    per_bin = {}
    for b, bar in gate[gkey_cov]["min_recall"].items():
        rec = fc["per_bin"][b]["recall"]
        per_bin[b] = {"fresh_recall": rec, "bar": bar,
                      "in_run_baseline_recall":
                          fb["classification"]["per_bin"][b]["recall"],
                      "frozen_iteration10_baseline_recall":
                          ea.LOCKED["fresh_bin_recall"].get(b),
                      "pass": bool(rec >= bar)}
    g_cov = {"per_bin": per_bin,
             "pass": bool(all(v["pass"] for v in per_bin.values())),
             "rule": gate[gkey_cov].get(
                 "baseline_rule",
                 "every fresh class recall >= bar (no class may collapse)")}
    g_rej = {"orig_auroc": r["auroc"],
             "orig_ood_detect": r["ood_detect_at_production_threshold"],
             "orig_id_frr": r["id_frr_at_production_threshold"],
             "fresh_auroc": frj["auroc"],
             "pass": bool(
                 r["auroc"] >= gate[gkey_rej]["orig_auroc_min"]
                 and r["ood_detect_at_production_threshold"]
                 >= gate[gkey_rej]["orig_ood_detect_min"]
                 and r["id_frr_at_production_threshold"]
                 <= gate[gkey_rej]["orig_id_frr_max"]
                 and frj["auroc"] >= gate[gkey_rej]["fresh_auroc_min"]),
             "rule": "rejection preservation vs production ∓0.5pt: "
                     f"orig AUROC >= {gate[gkey_rej]['orig_auroc_min']}, "
                     f"orig OOD detect >= "
                     f"{gate[gkey_rej]['orig_ood_detect_min']}, orig ID FRR "
                     f"<= {gate[gkey_rej]['orig_id_frr_max']}, fresh AUROC >= "
                     f"{gate[gkey_rej]['fresh_auroc_min']}"}
    if is12:
        g3, g4 = g3, g_cov
        g5 = {**g_rej,
              "fresh_frr": frj["false_rejection_rate"],
              "fresh_frr_bar": gate["G5"]["fresh_frr_max"],
              "fresh_frr_pass": bool(
                  frj["false_rejection_rate"]
                  <= gate["G5"]["fresh_frr_max"]),
              "isolation_pass": equiv["pass"],
              "max_abs_reject_diff":
                  equiv["overall"]["reject_max_abs_diff"],
              "reject_verdict_changes_at_threshold":
                  equiv["overall"]["reject_verdict_changes_at_threshold"],
              "file_level_frozen_identical":
                  equiv["file_level_freeze_check"]["pass"],
              "iteration10_frr_observed_only": ITER10_FRR_OBSERVED}
        g5["pass"] = bool(g5["fresh_frr_pass"] and equiv["pass"])
        g5["rule"] = (
            f"fresh FRR <= {gate['G5']['fresh_frr_max']} @0.0702 AND "
            f"isolation: max |Δreject| <= "
            f"{gate['G5']['reject_max_abs_diff']} with 0 changed reject "
            "verdicts; orig/fresh-AUROC bars as in G5")
        g6 = {"fresh_ood_detect": frj["overall_ood_detection"],
              "fresh_ood_detect_bar": gate["G6"]["fresh_ood_detect_min"],
              "pass": bool(frj["overall_ood_detection"]
                            >= gate["G6"]["fresh_ood_detect_min"]),
              "rule": f"fresh OOD detect >= "
                      f"{gate['G6']['fresh_ood_detect_min']} @0.0702"}
        g8_evidence = browser_gate(tag)
        g6 = {**g6, "browser_evidence": g8_evidence}
    else:
        g3 = g_cov
        g4 = {**g_rej,
              "fresh_ood_detect": frj["overall_ood_detection"],
              "pass": bool(g_rej["pass"] and frj["overall_ood_detection"]
                           >= gate["G4"]["fresh_ood_detect_min"])}
        g4["rule"] += (f", fresh OOD detect >= "
                       f"{gate['G4']['fresh_ood_detect_min']}")
        g5_frr_ok = bool(frj["false_rejection_rate"]
                         <= gate["G5"]["fresh_frr_max"])
        g5 = {"fresh_frr": frj["false_rejection_rate"],
              "fresh_frr_bar": gate["G5"]["fresh_frr_max"],
              "fresh_frr_pass": g5_frr_ok,
              "isolation_pass": equiv["pass"],
              "max_abs_reject_diff": equiv["overall"]["reject_max_abs_diff"],
              "reject_verdict_changes_at_threshold":
                  equiv["overall"]["reject_verdict_changes_at_threshold"],
              "file_level_frozen_identical":
                  equiv["file_level_freeze_check"]["pass"],
              "iteration10_frr_observed_only": ITER10_FRR_OBSERVED,
              "pass": bool(g5_frr_ok and equiv["pass"]),
              "rule": f"fresh FRR <= {gate['G5']['fresh_frr_max']} @0.0702 "
                      f"AND isolation: max |Δreject| <= "
                      f"{gate['G5']['reject_max_abs_diff']} with 0 changed "
                      "reject verdicts"}
        g6 = browser_gate(tag)
    remeasured = {"orig_accuracy": ob["classification"]["accuracy"],
                  "orig_macro_f1": ob["classification"]["macro_f1"],
                  "orig_auroc": ob["rejection"]["auroc"],
                  "fresh_accuracy": fb["classification"]["accuracy"],
                  "fresh_frr": fb["rejection"]["false_rejection_rate"]}
    core = (g1, g2, g3, g4, g5, g6) if is12 else (g1, g2, g3, g4, g5)
    out = {
        "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6,
        "G7": {"pass": None,
               "rule": "both seeds must satisfy G1-G4 (head12) / every "
                       "other gate (head11); decided in --summary"},
        "all_g1_g5_pass": bool(all(g["pass"] for g in core)),
        "in_run_baseline_remeasurement": {
            **remeasured,
            "consistency_vs_locked": {
                k: {"locked": ea.LOCKED[k], "remeasured": m,
                    "abs_diff": abs(m - ea.LOCKED[k])}
                for k, m in remeasured.items()},
        },
        "fresh_accuracy_delta_vs_locked":
            fc["accuracy"] - ea.LOCKED["fresh_accuracy"],
        "fresh_accuracy_delta_vs_iteration10": {
            str(s): fc["accuracy"] - ITER10_FRESH_ACC[str(s)]
            for s in ADAPT10_SEEDS},
    }
    return out


# --- Per-seed report --------------------------------------------------------

# --- Per-seed report --------------------------------------------------------

def report_md_iter11(seed: int, cand: dict, base: dict, gates: dict,
                     equiv: dict, audit: dict, tag: str = "head11",
                     protocol: str = "iteration11_head_adaptation_protocol.md") -> str:
    o, f = cand["original"], cand["fresh"]
    ob, fb = base["original"], base["fresh"]
    yn = lambda b: "PASS" if b else "FAIL"
    fd = equiv["file_level_freeze_check"]["detail"]
    ftensors = fd["frozen_tensors_in_file"] if isinstance(fd, dict) else "n/a"
    audit_ok = audit["iteration11_check"][
        "identical_to_iteration10_audit_except_timestamp"]
    iter_no = "12 - distillation-constrained" if tag == "head12" else "11"
    L = [f"# Iteration {iter_no} - classification-head-only adaptation - seed {seed}",
         "",
         f"- protocol: `{protocol}`",
         f"- candidate: `{cand['model']}` (md5 `{cand['md5']}`)",
         "- warm start: shipped checkpoint - only `predictions/kernel` + "
         "`predictions/bias` (1,028 params) trainable; 264 frozen tensors",
         f"- threshold: **{SHIPPED_THRESHOLD}** unchanged (no calibration)",
         "- dataset audit identical to Iteration-10 (except timestamp): "
         f"**{audit_ok}**",
         "", "## 1. Gates (pre-registered)", "",
         "| Gate | Measured | Rule | Pass |", "|---|---|---|---|",
         f"| G1 existing | acc {gates['G1'].get('orig_accuracy', float('nan')):.4f}, macro-F1 "
         f"{gates['G1'].get('orig_macro_f1', gates['G2'].get('orig_macro_f1', float('nan'))):.4f} | acc >= 0.9763 and F1 >= 0.9671"
         f" | {yn(gates['G1']['pass'] and gates['G2']['pass'] if tag == 'head12' else gates['G1']['pass'])} |",
         f"| G2 fresh | acc {gates['G3' if tag == 'head12' else 'G2']['fresh_accuracy']:.4f} | >= 0.5500 "
         f"| {yn(gates['G3' if tag == 'head12' else 'G2']['pass'])} |",
         f"| G4 rejection | orig AUROC {gates['G5' if tag == 'head12' else 'G4']['orig_auroc']:.5f}, orig "
         f"OOD {gates['G5' if tag == 'head12' else 'G4']['orig_ood_detect']:.5f}, orig ID FRR "
         f"{gates['G5' if tag == 'head12' else 'G4']['orig_id_frr']:.5f}, fresh AUROC "
         f"{gates['G5' if tag == 'head12' else 'G4']['fresh_auroc']:.5f} | production ∓0.5pt "
         f"| {yn(gates['G5' if tag == 'head12' else 'G4']['pass'])} |",
         f"| G5 FRR + isolation | fresh FRR {gates['G5']['fresh_frr']:.5f}, "
         f"max |Δreject| {gates['G5']['max_abs_reject_diff']:.3e}, flips "
         f"{gates['G5']['reject_verdict_changes_at_threshold']} | FRR <= "
         f"0.33830 AND max|Δ| <= 1e-6 AND 0 flips AND frozen file identical "
         f"| {yn(gates['G5']['pass'])} |",
         f"| G6 browser/parity | {gates['G6'].get('browser_evidence', gates['G6']).get('status', 'pending')} | "
         f"export + atol<=2e-5 parity + UI click-through | "
         f"{'PENDING' if gates['G6'].get('browser_evidence', gates['G6']).get('pass') is None else yn(gates['G6'].get('browser_evidence', gates['G6'])['pass'])} |",
         "| G7 two-seed | decided in --summary | both seeds pass | PENDING |",
         "", "### G3/G4 fresh class coverage", "",
         "| Bin | Fresh recall | Bar | In-run baseline | Pass |",
         "|---|---|---|---|---|"]
    cov_key = "G4" if tag == "head12" else "G3"
    for b, d in gates[cov_key]["per_bin"].items():
        L.append(f"| {b} | {d['fresh_recall']:.4f} | {d['bar']:.4f} | "
                 f"{d['in_run_baseline_recall']:.4f} | {yn(d['pass'])} |")
    L += ["", f"**{cov_key} overall: {yn(gates[cov_key]['pass'])}**; G1-G5 composite: "
          f"**{yn(gates['all_g1_g5_pass'])}**", "",
          "## 2. LOOP-12 rejection isolation", "",
          f"- fresh 131 one-at-a-time: max |Δreject| = "
          f"{equiv['fresh']['reject_max_abs_diff']:.3e}, bit-exact = "
          f"{equiv['fresh']['reject_bit_exact']}, reject verdict flips = "
          f"{equiv['fresh']['reject_verdict_changes_at_threshold']}",
          "- original pools: " + "; ".join(
              f"{n}: max|d| {d['reject_max_abs_diff']:.3e}, flips "
              f"{d['reject_verdict_changes_at_threshold']}"
              for n, d in equiv["original_pools"].items())
          + f" (overall incl. fresh: max|d| "
          f"{equiv['overall']['reject_max_abs_diff']:.3e}, flips "
          f"{equiv['overall']['reject_verdict_changes_at_threshold']})",
          f"- file-level frozen digests identical: "
          f"{equiv['file_level_freeze_check']['pass']} ({ftensors} tensors)",
          f"- bins argmax changes on fresh (expected - this is the trained "
          f"head): {equiv['fresh']['bins_argmax_changes']}; three-way verdict "
          f"changes: {equiv['fresh']['three_way_verdict_changes']}",
          f"- **pass: {equiv['pass']}**", "",
          "## 3. Existing + fresh benchmark numbers", "",
          "| Metric | Candidate | In-run shipped baseline | Locked |",
          "|---|---|---|---|",
          f"| orig accuracy | {o['classification']['accuracy']:.4f} | "
          f"{ob['classification']['accuracy']:.4f} | "
          f"{ea.LOCKED['orig_accuracy']:.4f} |",
          f"| orig macro-F1 | {o['classification']['macro_f1']:.4f} | "
          f"{ob['classification']['macro_f1']:.4f} | "
          f"{ea.LOCKED['orig_macro_f1']:.4f} |",
          f"| orig AUROC | {o['rejection']['auroc']:.5f} | "
          f"{ob['rejection']['auroc']:.5f} | {ea.LOCKED['orig_auroc']:.5f} |",
          f"| orig OOD detect @0.0702 | "
          f"{o['rejection']['ood_detect_at_production_threshold']:.5f} | "
          f"{ob['rejection']['ood_detect_at_production_threshold']:.5f} | - |",
          f"| fresh supported acc (63) | {f['classification']['accuracy']:.4f} "
          f"| {fb['classification']['accuracy']:.4f} | "
          f"{ea.LOCKED['fresh_accuracy']:.4f} |",
          f"| fresh macro-F1 | {f['classification']['macro_f1']:.4f} | "
          f"{fb['classification']['macro_f1']:.4f} | "
          f"{ea.LOCKED['fresh_macro_f1']:.4f} |",
          f"| fresh AUROC | {f['rejection']['auroc']:.5f} | "
          f"{fb['rejection']['auroc']:.5f} | - |",
          f"| fresh FRR @0.0702 | {f['rejection']['false_rejection_rate']:.5f} "
          f"| {fb['rejection']['false_rejection_rate']:.5f} | "
          f"{ea.LOCKED['fresh_frr']:.5f} |",
          f"| fresh OOD detection | "
          f"{f['rejection']['overall_ood_detection']:.5f} | "
          f"{fb['rejection']['overall_ood_detection']:.5f} | "
          f"{ea.LOCKED['fresh_ood_detect']:.5f} |",
          "",
          "Fresh accuracy deltas vs Iteration 10 (seed: delta): "
          + ", ".join(f"{s}: {d:+.4f}" for s, d in
                      gates["fresh_accuracy_delta_vs_iteration10"].items())
          + " (Iteration 10 scored 0.5238 / 0.5397; outcome B bar 0.4600).",
          "",
          "## 4. Notes", "",
          "- The reject head is frozen, so rejection metrics are production's "
          "by construction; the in-run baseline column re-measures the shipped "
          "model in the same process to prove the harness did not move.",
          "- Threshold 0.0702 is untouched; no checkpoint selection: the "
          "candidate is the FINAL epoch.",
          ""]
    return "\n".join(L) + "\n"


# --- Per-seed evaluation (G1-G5 + G6 evidence) ------------------------------

def run_one(args) -> None:
    seed = int(args.seed)
    tag = getattr(args, "tag", None) or "head11"
    gate = resolve_gate_table(getattr(args, "gates", None))
    cand_path = Path(args.model)
    if not cand_path.exists():
        raise SystemExit(f"candidate not found: {cand_path} - train the seed "
                         f"first (src/train_head_adaptation.py --seed {seed})")
    audit = None if args.skip_audit else audit_iter11(tag)[0]   # dict (§3)
    base = ea.candidate_metrics(Path(args.baseline))
    cand = ea.candidate_metrics(cand_path)
    # LOOP-12 isolation (part of G5): candidate vs shipped reject probability.
    equiv = rejection_equivalence(cand_path, seed, tag=tag, gate=gate)
    gates = gates_for(cand, base, equiv, gate=gate, tag=tag)
    (OUT_DIR / f"{tag}s{seed}_eval.json").write_text(
        json.dumps({"seed": seed, "audit": audit, "baseline_in_run": base,
                    "candidate": cand}, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / f"{tag}s{seed}_gates.json").write_text(
        json.dumps({"seed": seed, "model": cand["model"],
                    "model_md5": cand["md5"],
                    "gate_table": getattr(args, "gates", None)
                    or "head11_gates.json",
                    "equivalence_file": str(OUT_DIR /
                                            f"{tag}_rejection_equivalence.json"),
                    **gates}, indent=2) + "\n", encoding="utf-8")
    if audit:
        (OUT_DIR / f"{tag}s{seed}_report.md").write_text(
            report_md_iter11(seed, cand, base, gates, equiv, audit,
                             tag=tag,
                             protocol=getattr(args, "protocol", None)
                             or "iteration11_head_adaptation_protocol.md"),
            encoding="utf-8")
    g = gates
    if tag == "head12":
        print(f"[seed {seed}] G1={g['G1']['pass']} G2={g['G2']['pass']} "
              f"G3={g['G3']['pass']} G4={g['G4']['pass']} G5={g['G5']['pass']} "
              f"G6={g['G6']['pass']} G8={g['G6'].get('browser_evidence', {}).get('status')} "
              f"-> G1-G6 all={g['all_g1_g5_pass']}")
        print(f"[seed {seed}] orig acc {g['G1']['orig_accuracy']:.4f}, orig F1 "
              f"{g['G2']['orig_macro_f1']:.4f}; fresh acc "
              f"{g['G3']['fresh_accuracy']:.4f} (delta vs Iter-10: "
              f"{json.dumps(g['fresh_accuracy_delta_vs_iteration10'])}); "
              f"fresh OOD {g['G6']['fresh_ood_detect']:.5f}; fresh FRR "
              f"{g['G5']['fresh_frr']:.5f}; isolation "
              f"max|d|={g['G5']['max_abs_reject_diff']:.3e}")
    else:
        print(f"[seed {seed}] G1={g['G1']['pass']} G2={g['G2']['pass']} "
              f"G3={g['G3']['pass']} G4={g['G4']['pass']} G5={g['G5']['pass']} "
              f"G6={g['G6'].get('status')} -> G1-G5 all={g['all_g1_g5_pass']}")
        print(f"[seed {seed}] fresh acc {g['G2']['fresh_accuracy']:.4f} "
              f"(delta vs Iter-10: "
              f"{json.dumps(g['fresh_accuracy_delta_vs_iteration10'])}); "
              f"fresh FRR {g['G5']['fresh_frr']:.5f}; isolation "
              f"max|d|={g['G5']['max_abs_reject_diff']:.3e}")


# --- Two-seed summary, G7, decision (protocol §7) ---------------------------

GATE_KEYS = ("G1", "G2", "G3", "G4", "G5", "G6")


def spread(values: dict) -> dict:
    v = list(values.values())
    return {"mean": float(np.mean(v)),
            "spread": float(np.max(v) - np.min(v)), "values": values}


def load_seed_gates(tag: str = "head11") -> dict[int, dict]:
    out = {}
    for s in ADAPT10_SEEDS:
        p = OUT_DIR / f"{tag}s{s}_gates.json"
        if p.exists():
            g = json.loads(p.read_text(encoding="utf-8"))
            if tag == "head12":
                # Head12: G6 is the measured fresh-OOD gate; G8 browser
                # evidence rides inside G6.browser_evidence (do NOT overwrite
                # the measured gate with the evidence file).
                ev = browser_gate(tag)
                if isinstance(g.get("G6"), dict):
                    g["G6"] = {**g["G6"], "browser_evidence": ev}
            else:
                g["G6"] = browser_gate(tag)   # re-read: evidence may be newer
            out[s] = g
    return out


def run_summary(args) -> None:
    tag = getattr(args, "tag", None) or "head11"
    protocol = getattr(args, "protocol", None) or \
        "iteration11_head_adaptation_protocol.md"
    is_head12 = tag == "head12"
    gates = load_seed_gates(tag)
    if not gates:
        raise SystemExit(f"no {tag}s{{42,43}}_gates.json - run --seed first")
    missing = [s for s in ADAPT10_SEEDS if s not in gates]
    browser = browser_gate(tag)
    per_seed = {}
    for s, g in sorted(gates.items()):
        if is_head12:
            g8 = (g.get("G6", {}).get("browser_evidence", {})
                  if isinstance(g.get("G6"), dict) else {})
            passes = {k: g[k]["pass"] for k in
                      ("G1", "G2", "G3", "G4", "G5", "G6")}
            passes["G8"] = g8.get("pass")
        else:
            passes = {k: (g[k]["pass"] if g[k].get("pass") is not None
                          else None) for k in GATE_KEYS}
        scored = [k for k, v in passes.items() if v is not None]
        entry = {
            "model": g["model"], "model_md5": g["model_md5"],
            **{f"{k}_pass": passes[k] for k in passes},
            "gates_scored_pass": bool(all(passes[k] for k in scored)),
        }
        if is_head12:
            entry.update({
                "fresh_accuracy": g["G3"]["fresh_accuracy"],
                "fresh_macro_f1": g["G3"]["fresh_macro_f1"],
                "orig_accuracy": g["G1"]["orig_accuracy"],
                "orig_macro_f1": g["G2"]["orig_macro_f1"],
                "orig_auroc": g["G5"]["orig_auroc"],
                "fresh_ood_detect": g["G6"]["fresh_ood_detect"],
                "fresh_frr": g["G5"]["fresh_frr"],
                "fresh_frr_isolation_pass": g["G5"]["isolation_pass"],
                "fresh_accuracy_delta_vs_iteration10":
                    g["fresh_accuracy_delta_vs_iteration10"],
            })
        else:
            entry.update({
                "fresh_accuracy": g["G2"]["fresh_accuracy"],
                "fresh_macro_f1": g["G2"]["fresh_macro_f1"],
                "orig_accuracy": g["G1"]["orig_accuracy"],
                "orig_macro_f1": g["G1"]["orig_macro_f1"],
                "orig_auroc": g["G4"]["orig_auroc"],
                "fresh_frr": g["G5"]["fresh_frr"],
                "fresh_frr_isolation_pass": g["G5"]["isolation_pass"],
                "fresh_accuracy_delta_vs_iteration10":
                    g["fresh_accuracy_delta_vs_iteration10"],
            })
        per_seed[str(s)] = entry
    # G7 (head12): BOTH seeds satisfy G1-G4 numerically. Outcome A further
    # needs G5+G6 measured pass AND G8 browser evidence scored pass.
    # Head11 path unchanged: G1-G5 contention, G6 evidence, G7 = all.
    g8_pending = (browser.get("pass") is None) if is_head12 else None
    g6_pending = browser.get("pass") is None
    both = not missing

    def s15(s: int) -> bool:
        if is_head12:
            return all(per_seed[str(s)][f"{k}_pass"]
                       for k in ("G1", "G2", "G3", "G4"))
        return all(per_seed[str(s)][f"{k}_pass"]
                   for k in ("G1", "G2", "G3", "G4", "G5"))

    # Outcome A can only be decided once the browser evidence is scored; if
    # the measured gates already fail anywhere, A is excluded regardless of
    # G8, so B/C stay determinable.
    a_contender = bool(both and all(s15(s) for s in ADAPT10_SEEDS))
    if is_head12:
        g7_pass = bool(a_contender)
        a_full = bool(
            a_contender and not g8_pending
            and all(per_seed[str(s)]["G5_pass"] for s in ADAPT10_SEEDS)
            and all(per_seed[str(s)]["G6_pass"] for s in ADAPT10_SEEDS)
            and all(per_seed[str(s)]["G8_pass"] for s in ADAPT10_SEEDS))
    else:
        g7_pass = bool(a_contender and not g6_pending
                       and all(per_seed[str(s)]["G6_pass"]
                               for s in ADAPT10_SEEDS))
        a_full = g7_pass
    accs = {str(s): per_seed[str(s)]["fresh_accuracy"]
            for s in per_seed}
    if not both:
        decision, reason = (
            "Pending",
            f"seed(s) {missing} not evaluated yet - G7 needs both seeds "
            f"before the decision vocabulary applies.")
    elif is_head12 and a_contender and g8_pending:
        decision, reason = (
            "Pending",
            "both seeds satisfy G1-G4 and G7 passes, but G8 "
            "(browser export/parity/UI evidence) is unscored - run that step, "
            f"write {tag}_browser.json, re-run --summary.")
    elif a_contender and g6_pending and not is_head12:
        decision, reason = (
            "Pending",
            "both seeds are in contention for Outcome A (G1-G5 pass), so G6 "
            "(browser export/parity/UI evidence) is required - run that step, "
            f"write {tag}_browser.json, re-run --summary.")
    elif a_full:
        decision, reason = (
            "A - Ship-eligible",
            "both seeds pass G1-G6 (=> G7 satisfied): the seed-42 candidate "
            "may be promoted in the promotion step §8 (threshold 0.0702 "
            "unchanged).") if not is_head12 else (
            "A - Ship-eligible",
            "both seeds pass G1-G6 and G8 browser evidence (=> G7 "
            "satisfied): the seed-42 candidate may be promoted "
            "(threshold 0.0702 unchanged).")
    elif all(a >= MEANINGFUL_IMPROVEMENT_MIN for a in accs.values()):
        decision, reason = (
            "B - Improvement-but-fail",
            f"fresh supported accuracy >= {MEANINGFUL_IMPROVEMENT_MIN} for "
            "both seeds (meaningful improvement) but at least one gate fails "
            "-> production unchanged, improvement documented; the protocol is "
            "not renegotiated after the measurement.")
    else:
        decision, reason = (
            "C - No meaningful improvement",
            f"fresh supported accuracy < {MEANINGFUL_IMPROVEMENT_MIN} for at "
            "least one seed -> the classification-head-only approach is "
            "rejected; Variant C stays production.")
    metrics = {k: spread({str(s): per_seed[str(s)][k] for s in per_seed})
               for k in ("orig_accuracy", "orig_macro_f1", "orig_auroc",
                         "fresh_accuracy", "fresh_macro_f1", "fresh_frr")}
    g7_block = {
            "rule": "both seeds must satisfy G1-G4 "
                    "(head12; disagreement on G3 = Outcome B)"
                    if is_head12 else
                    "both seeds must satisfy G2 and every other gate "
                    "(numerically); disagreement on G2 = Outcome B",
            "both_seeds_present": bool(both),
            "g6_status": browser.get("status", "pending"),
            "g2_pass_disagreement": bool(
                both and per_seed["42"]["G3_pass" if is_head12 else "G2_pass"]
                != per_seed["43"]["G3_pass" if is_head12 else "G2_pass"]),
            "outcome_a_contention_g1_g5": a_contender,
            "both_seeds_pass_g1_g6": g7_pass,
            "metric_mean_and_spread": metrics,
            "iteration10_reference": {"fresh_accuracy": ITER10_FRESH_ACC,
                                      "fresh_frr": ITER10_FRR_OBSERVED,
                                      "note": "reported as observations only"},
        }
    if is_head12:
        g7_block["g7_pass"] = g7_pass
        g7_block["outcome_a_full_g1_g6_g8"] = a_full
        g7_block["g8_status"] = browser.get("status", "pending")
    out = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": f"docs/rejection_experiment/{protocol} §7"
        if is_head12 else "docs/rejection_experiment/"
        "iteration11_head_adaptation_protocol.md §6 G7 / §7",
        "iteration": "head12-distillation" if is_head12 else "head11",
        "per_seed": per_seed,
        "g7_seed_consistency": g7_block,
        "decision": decision,
        "decision_reason": reason,
        "g6_browser": browser,
        "promotion_target": "models/checkpoints/"
                            "wastelens_rej_shipped_best.keras <- seed-42 "
                            "candidate if and only if the decision is A",
        "production_state": {
            "shipped_checkpoint": str(SHIPPED),
            "shipped_checkpoint_md5": tr.md5_file(SHIPPED),
            "production_threshold": SHIPPED_THRESHOLD,
            "web_model_dir_rewritten": False,
            "note": "no promotion unless Outcome A triggers §8; production "
                    "artifacts and the 0.0702 threshold are read-only "
                    f"throughout {'Iteration 12' if is_head12 else 'Iteration 11'}",
        },
    }
    decision_file = ("iteration12_decision.md" if is_head12
                     else "iteration11_decision.md")
    (OUT_DIR / f"{tag}_seed_consistency.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / decision_file).write_text(
        decision_md(out, tag=tag, protocol=protocol), encoding="utf-8")
    print(f"[summary] seeds scored: {sorted(per_seed)}; G6 "
          f"{browser.get('status')}; G7 pass={g7_pass}")
    print(f"[summary] decision: {decision}")
    return out


def yn3(v) -> str:
    return "PENDING" if v is None else ("PASS" if v else "FAIL")


def decision_md(out: dict, tag: str = "head11",
                protocol: str = "iteration11_head_adaptation_protocol.md") -> str:
    ps, g7 = out["per_seed"], out["g7_seed_consistency"]
    title = ("# Iteration 12 decision - distillation-constrained adaptation"
             if tag == "head12"
             else "# Iteration 11 decision - classification-head-only adaptation")
    L = [title, "",
         f"- decision: **{out['decision']}**",
         f"- reason: {out['decision_reason']}",
         f"- protocol: `docs/rejection_experiment/{protocol}` §7 vocabulary,"
         " decided by the numbers", "",
         "| Seed | G1 | G2 | G3 | G4 | G5 | G6 | fresh acc | fresh macro-F1 "
         "| fresh FRR |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(ps):
        d = ps[s]
        g8 = d.get("G8_pass", d["G6_pass"] if tag != "head12" else None)
        L.append(f"| {s} | {yn3(d['G1_pass'])} | {yn3(d['G2_pass'])} | "
                 f"{yn3(d['G3_pass'])} | {yn3(d['G4_pass'])} | "
                 f"{yn3(d['G5_pass'])} | {yn3(d['G6_pass'])} | "
                 f"{yn3(g8) if tag == 'head12' else d['fresh_accuracy']:.4f}"
                 + (f" | {d['fresh_accuracy']:.4f} | {d['fresh_macro_f1']:.4f} | "
                    f"{d['fresh_frr']:.5f} |" if tag == "head12" else
                    f" | {d['fresh_macro_f1']:.4f} | "
                    f"{d['fresh_frr']:.5f} |"))
    if tag == "head12":
        L[5] = ("| Seed | G1 | G2 | G3 | G4 | G5 | G6 | G8 | fresh acc | "
                "fresh macro-F1 | fresh FRR |")
        L[6] = ("|---|---|---|---|---|---|---|---|---|---|---|")
    L += ["", "## Seed consistency (G7)", ""]
    for k, v in g7["metric_mean_and_spread"].items():
        L.append(f"- {k}: mean {v['mean']:.4f} ± {v['spread']:.4f} "
                 f"({json.dumps(v['values'])})")
    L += ["", f"- both seeds pass G1-G6: **{g7['both_seeds_pass_g1_g6']}**; "
          f"G6 status: {g7['g6_status']}; G2 disagreement: "
          f"{g7['g2_pass_disagreement']}",
          "",
          "## Iteration-10 / production reference (observations)", "",
          f"- Iteration-10 fresh accuracy: {json.dumps(ITER10_FRESH_ACC)} "
          f"(bar for Outcome B: {MEANINGFUL_IMPROVEMENT_MIN})",
          f"- Iteration-10 fresh FRR: {json.dumps(ITER10_FRR_OBSERVED)} "
          "- lower than production because Iteration 10 re-calibrated the "
          "reject head; Iteration 11 freezes it by design",
          f"- production fresh FRR 0.3333 (bar {GATE['G5']['fresh_frr_max']}), "
          f"threshold {SHIPPED_THRESHOLD} unchanged",
          "",
          "## Production state", "",
          f"- shipped checkpoint md5: `{out['production_state']['shipped_checkpoint_md5']}`",
          f"- web/model rewritten: {out['production_state']['web_model_dir_rewritten']}",
          f"- {out['production_state']['note']}",
          ""]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="WasteLens Iteration-11/12 evaluator: pre-registered gates "
                    "for a classification-head-only candidate (head11 §6; "
                    "head12 via --tag/--gates/--protocol) plus the "
                    "rejection-isolation test; --summary computes G7/G8 "
                    "and the decision.")
    ap.add_argument("--seed", type=int, choices=ADAPT10_SEEDS,
                    help="evaluate one trained seed")
    ap.add_argument("--model", default=None,
                    help="candidate .keras (default: models/checkpoints/"
                         "wastelens_rej_{tag}s{SEED}_final.keras)")
    ap.add_argument("--tag", default="head11",
                    help="artifact prefix: head11 (default) | head12")
    ap.add_argument("--protocol",
                    default="iteration11_head_adaptation_protocol.md",
                    help="protocol doc filename for evidence strings")
    ap.add_argument("--gates", default="head11_gates.json",
                    help="gate table: head11_gates.json (default) | "
                         "head12_gates.json")
    ap.add_argument("--baseline", default=str(SHIPPED),
                    help="in-run shipped baseline for re-measurement (§6)")
    ap.add_argument("--skip-audit", action="store_true",
                    help="skip the §3 dataset-drift audit (debug only)")
    ap.add_argument("--summary", action="store_true",
                    help="G7 two-seed summary -> {tag}_seed_consistency.json "
                         "+ iteration{nn}_decision.md")
    args = ap.parse_args()
    if args.summary:
        run_summary(args)
        return
    if args.seed is None:
        ap.error("--seed (42|43) is required unless --summary is given")
    if args.model is None:
        args.model = str(Path("models/checkpoints") /
                         f"wastelens_rej_{args.tag}s{args.seed}_final.keras")
    run_one(args)


if __name__ == "__main__":
    main()







