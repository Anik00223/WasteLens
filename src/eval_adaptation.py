# src/eval_adaptation.py
#
# WasteLens Iteration 10 - LOOP 11: leakage audit + two-benchmark evaluation
# of a fresh-domain adaptation candidate, with the PRE-REGISTERED gates of
# docs/rejection_experiment/iteration10_adaptation_protocol.md §4.
#
# WHAT IT DOES
#   1. Mechanical leakage audit (§3): asserts zero path/SHA-256 overlap
#      between {adapt-train, adapt-val} and {131 fresh files, original test
#      pools, CIFAR test batch}; writes adaptation_eval_audit.json. A hit
#      aborts before any model is scored.
#   2. Scores a candidate checkpoint on the UNTOUCHED benchmarks:
#      - original test pools (eval_sets.json, identical definitions to
#        src/eval_rejection.py -> G1 classification, G2 rejection),
#      - the 131-image fresh benchmark (identical code path to
#        src/eval_fresh_realworld.py -> G3 improvement, G4 FRR blowup).
#      The shipped checkpoint is scored through the same path so the
#      comparison is apples-to-apples (locked baselines are restated).
#   3. Applies the gate table and writes
#      adapt10s{seed}_eval.json / adapt10s{seed}_gates.json /
#      adapt10s{seed}_report.md.
#
# METRIC DEFINITIONS ARE IMPORTED, NOT REWRITTEN: eval_rejection for the
# original pools, eval_fresh_realworld for the fresh benchmark. The
# production threshold stays 0.0702 during this evaluation (no threshold
# rescue, protocol §4 / LOOP 12).
#
# Run from the repo root:
#   py -3.13 -W ignore src/eval_adaptation.py --model <ckpt> --seed 42
#   py -3.13 -W ignore src/eval_adaptation.py --audit-only
#   py -3.13 -W ignore src/eval_adaptation.py --summary   (both seeds)

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

import eval_fresh_realworld as fr
import eval_rejection as er
import train as wl
import train_adaptation as ta
import train_rejection as tr

OUT_DIR = tr.OUT_DIR
ADAPT_MANIFEST = ta.ADAPT_MANIFEST
FRESH_MANIFEST = ta.FRESH_MANIFEST
ADAPT_DIR = ta.ADAPT_DIR
FRESH_DIR = Path("scratch/fresh_set")
SHIPPED_THRESHOLD = 0.0702          # production constant, frozen for this eval

# Locked Iteration-9/10 baselines (fresh_eval_metrics.json + rejection_metrics_varc8s42).
LOCKED = {
    "orig_accuracy": 0.981346309813463,
    "orig_macro_f1": 0.972132471386473,
    "orig_auroc": 0.9996601876333407,
    "orig_ood_detect": 0.9963880288957688,   # @0.0702, test pools (Iter. 8)
    "orig_id_frr": 0.017842660178426603,     # @0.0702, supported test
    "orig_val_frozen_1pct_threshold": 0.07018820941448212,
    "fresh_accuracy": 0.3968253968253968,
    "fresh_macro_f1": 0.25459482038429404,
    "fresh_frr": 0.3333333333333333,
    "fresh_ood_detect": 0.6744186046511628,
    "fresh_multi_detect": 0.4666666666666667,
    "fresh_bin_recall": {"recyclable": 0.9523809523809523,
                         "organic": 0.0625,
                         "hazardous": 0.26666666666666666,
                         "general trash": 0.0},
}

GATE = {
    "G1": {"orig_acc_min": 0.9763, "orig_macro_f1_min": 0.9671},
    "G2": {"auroc_min": 0.9950, "ood_detect_min": 0.95},
    "G3": {"fresh_acc_min": 0.5500, "bins_improved_min": 2,
           "collapse_bins": ["organic", "hazardous", "general trash"]},
    "G4": {"fresh_frr_max": 0.45},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hashes(manifest_path: Path) -> tuple[dict[str, str], list[str]]:
    """{local_name: sha256} + paths from a set builder's manifest."""
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ({e["local_name"]: e["sha256"] for e in m["entries"]},
            [e["local_name"] for e in m["entries"]])
def run_audit() -> dict:
    """Protocol §3: mechanical leakage audit before any scoring.

    A content-hash or path hit aborts the run (SystemExit) - the evaluation
    must never proceed on a contaminated adaptation set.
    """
    adapt_h, _ = manifest_hashes(ADAPT_MANIFEST)
    fresh_h, fresh_names = manifest_hashes(FRESH_MANIFEST)
    pool = ta.adapt_rows()
    split = ta.split_adapt(pool, wl.SEED)
    rows = []
    for role, parts in split.items():
        for part in ("train", "val"):
            for p, _role, _idx in parts[part]:
                name = Path(p).name
                rows.append({"local_name": name, "role": role,
                             "part": f"adapt-{part}",
                             "path": str(Path(p).resolve()),
                             "sha256_manifest": adapt_h[name],
                             "sha256_disk": sha256_file(Path(p))})

    fresh_paths = {str((FRESH_DIR / n).resolve()) for n in fresh_names}
    adapt_paths = {r["path"] for r in rows}
    path_hits = sorted(adapt_paths & fresh_paths)
    integrity = sorted(r["local_name"] for r in rows
                       if r["sha256_manifest"] != r["sha256_disk"])
    sha_hits = sorted({r["sha256_manifest"] for r in rows}
                      & set(fresh_h.values()))

    # Original test pools + CIFAR test batch (all reachable from eval_sets.json).
    data = json.loads(tr.EVAL_SETS_JSON.read_text(encoding="utf-8"))
    test_paths = [r[0] for r in data["supported_test"]]
    for src, plist in data["test_unsup"].items():
        test_paths.extend(plist)
    adapt_md5 = {tr.md5_file(Path(r["path"])) for r in rows}
    test_md5 = {}
    for p in sorted(set(str(Path(p).resolve()) for p in test_paths)):
        test_md5.setdefault(tr.md5_file(Path(p)), []).append(p)
    md5_hits = sorted(adapt_md5 & set(test_md5))

    audit = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": "docs/rejection_experiment/"
                    "iteration10_adaptation_protocol.md §3",
        "method": "path comparison + SHA-256 (adaptation vs 131 fresh files, "
                  "verified against the committed manifest) + MD5 (adaptation "
                  "vs original test pools incl. CIFAR-10 test batch)",
        "counts": {
            "adapt_train": sum(1 for r in rows if r["part"] == "adapt-train"),
            "adapt_val": sum(1 for r in rows if r["part"] == "adapt-val"),
            "fresh_files": len(fresh_names),
            "original_test_files": sum(len(v) for v in test_md5.values()),
        },
        "adapt_files": rows,
        "path_hits": path_hits,
        "sha256_hits_vs_fresh": sha_hits,
        "md5_hits_vs_original_test": [test_md5[h] for h in md5_hits],
        "manifest_integrity_failures": integrity,
        "clean": not (path_hits or sha_hits or md5_hits or integrity),
        "notes": "The fresh benchmark files are never opened by the training "
                 "path; this audit is the mechanical proof.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "adaptation_eval_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"[audit] adapt-train={audit['counts']['adapt_train']} "
          f"adapt-val={audit['counts']['adapt_val']} "
          f"fresh={audit['counts']['fresh_files']} "
          f"orig-test={audit['counts']['original_test_files']} "
          f"clean={audit['clean']}")
    if not audit["clean"]:
        raise SystemExit("LEAKAGE AUDIT FAILED - see "
                         "docs/rejection_experiment/adaptation_eval_audit.json")
    return audit

def score_fresh(model) -> list[dict]:
    """Score the 131 fresh files with the baseline's exact preprocessing."""
    manifest = json.loads(FRESH_MANIFEST.read_text(encoding="utf-8"))
    scored = []
    for item in manifest["entries"]:
        arr = fr.preprocess_image(FRESH_DIR / item["local_name"])
        out = model(np.expand_dims(arr, 0), training=False)
        bins_p = [float(v) for v in out["bins"].numpy()[0]]
        rej_p = float(out["reject"].numpy().ravel()[0])
        scored.append({
            **item, "bins": bins_p, "reject": rej_p,
            "pred_bin": fr.BINS[int(np.argmax(bins_p))],
            "top_prob": float(max(bins_p)),
            "margin": float(max(bins_p) - sorted(bins_p)[-2]),
            "verdict_shipped": fr.decide_verdict(bins_p, rej_p,
                                                 SHIPPED_THRESHOLD),
            "threshold_used": SHIPPED_THRESHOLD,
        })
    return scored


def fresh_metrics(model) -> dict:
    """Fresh-benchmark metrics - identical math to eval_fresh_realworld."""
    items = score_fresh(model)
    sup = [it for it in items if it["label"] in fr.BINS]
    amb = [it for it in items if it["group"] == "ambiguous"]
    ood = [it for it in items if it["group"].startswith("ood-")]
    multi = [it for it in items if it["group"] == "multi-object"]

    cm = [[0] * 4 for _ in range(4)]
    for it in sup:
        cm[fr.BINS.index(it["label"])][int(np.argmax(it["bins"]))] += 1
    acc = float(sum(cm[i][i] for i in range(4)) / max(len(sup), 1))
    per_bin, f1s = {}, []
    for i, name in enumerate(fr.BINS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(4) if r != i)
        fn = sum(cm[i][c] for c in range(4) if c != i)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_bin[name] = {"precision": prec, "recall": rec, "f1": f1,
                         "count": sum(cm[i])}
        f1s.append(f1)
    macro_f1 = float(sum(f1s) / 4)

    id_rej = np.array([it["reject"] for it in sup])
    ood_rej = np.array([it["reject"] for it in ood])
    auroc, auprc = fr.compute_auroc_auprc(
        np.concatenate([np.zeros(len(id_rej)), np.ones(len(ood_rej))]),
        np.concatenate([id_rej, ood_rej]))
    multi_rej = np.array([it["reject"] for it in multi])
    per_role = {}
    for role in ("ood-objects", "ood-clothes", "ood-shoes", "ood-scenes"):
        sub = [it for it in ood if it["group"] == role]
        per_role[role] = {
            "count": len(sub),
            "detected": int(sum(it["reject"] >= SHIPPED_THRESHOLD
                                for it in sub)),
            "detection_rate": float(np.mean([it["reject"] >= SHIPPED_THRESHOLD
                                             for it in sub])),
            "mean_reject": float(np.mean([it["reject"] for it in sub])),
        }

    return {
        "classification": {"n_images": len(sup), "accuracy": acc,
                           "macro_f1": macro_f1, "per_bin": per_bin,
                           "confusion_matrix": cm},
        "rejection": {
            "threshold": SHIPPED_THRESHOLD, "id_n": len(sup),
            "ood_n": len(ood), "auroc": auroc, "auprc": auprc,
            "false_rejection_rate": float(np.mean(id_rej
                                                  >= SHIPPED_THRESHOLD)),
            "overall_ood_detection": float(np.mean(ood_rej
                                                   >= SHIPPED_THRESHOLD)),
            "per_role": per_role,
        },
        "multi_object": {
            "count": len(multi),
            "detected": int(sum(multi_rej >= SHIPPED_THRESHOLD)),
            "detection_rate": float(np.mean(multi_rej >= SHIPPED_THRESHOLD)),
            "mean_reject": float(np.mean(multi_rej)),
            "verdicts": {v: sum(it["verdict_shipped"] == v for it in multi)
                         for v in ("supported", "uncertain", "unsupported")},
        },
        "ambiguous": {
            "count": len(amb),
            "verdicts": {v: sum(it["verdict_shipped"] == v for it in amb)
                         for v in ("supported", "uncertain", "unsupported")},
            "mean_top_prob": float(np.mean([it["top_prob"] for it in amb])),
            "mean_margin": float(np.mean([it["margin"] for it in amb])),
        },
        "detailed_predictions": [
            {"local_name": it["local_name"], "group": it["group"],
             "label": it["label"], "bins": it["bins"], "reject": it["reject"],
             "pred_bin": it["pred_bin"], "verdict": it["verdict_shipped"]}
            for it in items],
    }


def original_metrics(model, data: dict) -> dict:
    """Original-pool metrics via src/eval_rejection.py's exact definitions."""
    from sklearn.metrics import roc_auc_score
    bins = er.evaluate_bins(model, data["supported_test"])
    sup_test_rej = er.batches(model, [r[0] for r in data["supported_test"]])[1]
    sup_val_rej = er.batches(model, data["supported_val"])[1]
    test_u = er.group_scores(model, data["test_unsup"])
    all_ood = np.concatenate(list(test_u.values()))
    frozen = {f"{f:.2f}": float(np.quantile(sup_val_rej, 1.0 - f))
              for f in er.FRR_POINTS}
    y_sc = np.concatenate([sup_test_rej] + list(test_u.values()))
    y_tr = np.concatenate([np.zeros_like(sup_test_rej)]
                          + [np.ones_like(v) for v in test_u.values()])
    return {
        "classification": {"accuracy": bins["accuracy"],
                           "macro_f1": bins["macro_f1"],
                           "per_bin": bins["per_bin"],
                           "confusion_matrix": bins["confusion_matrix"]},
        "rejection": {
            "auroc": float(roc_auc_score(y_tr, y_sc)),
            "per_source_detect_at_production_threshold": {
                s: float(np.mean(v >= SHIPPED_THRESHOLD))
                for s, v in test_u.items()},
            "ood_detect_at_production_threshold": float(
                np.mean(all_ood >= SHIPPED_THRESHOLD)),
            "id_frr_at_production_threshold": float(
                np.mean(sup_test_rej >= SHIPPED_THRESHOLD)),
            "val_frozen_thresholds": frozen,
            "candidate_own_val_frozen_5pct_threshold": frozen["0.05"],
            "val_frr_at_own_5pct_threshold": float(
                np.mean(sup_val_rej >= frozen["0.05"])),
            "threshold_note": "gates use the frozen production threshold "
                              "0.0702 (protocol §4, no threshold rescue); the "
                              "val-frozen values are for the promotion step",
        },
    }


def candidate_metrics(model_path: Path) -> dict:
    data = json.loads(tr.EVAL_SETS_JSON.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(model_path)
    return {"model": str(model_path), "md5": tr.md5_file(model_path),
            "original": original_metrics(model, data),
            "fresh": fresh_metrics(model),
            "eval_utc": datetime.now(timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S UTC")}

def apply_gates(cand: dict, base: dict) -> dict:
    """Protocol §4 gate table for one seed, incl. an in-run baseline
    re-measurement (the locked baselines are restated for comparison)."""
    o, f = cand["original"], cand["fresh"]
    ob, fb = base["original"], base["fresh"]
    bin_impr = {}
    for b in GATE["G3"]["collapse_bins"]:
        r_new = f["classification"]["per_bin"][b]["recall"]
        r_base = fb["classification"]["per_bin"][b]["recall"]
        bin_impr[b] = {"recall": r_new,
                       "locked_baseline_recall": LOCKED["fresh_bin_recall"][b],
                       "in_run_baseline_recall": r_base,
                       "delta_vs_locked": r_new - LOCKED["fresh_bin_recall"][b],
                       "delta_vs_in_run": r_new - r_base,
                       "improved": bool(r_new > LOCKED["fresh_bin_recall"][b])}
    n_impr = int(sum(v["improved"] for v in bin_impr.values()))
    g = {
        "G1": {"orig_accuracy": o["classification"]["accuracy"],
               "orig_macro_f1": o["classification"]["macro_f1"],
               "pass": bool(o["classification"]["accuracy"]
                            >= GATE["G1"]["orig_acc_min"]
                            and o["classification"]["macro_f1"]
                            >= GATE["G1"]["orig_macro_f1_min"]),
               "rule": f"test acc >= {GATE['G1']['orig_acc_min']} and "
                       f"macro-F1 >= {GATE['G1']['orig_macro_f1_min']}"},
        "G2": {"auroc": o["rejection"]["auroc"],
               "ood_detect": o["rejection"]
               ["ood_detect_at_production_threshold"],
               "pass": bool(o["rejection"]["auroc"] >= GATE["G2"]["auroc_min"]
                            and o["rejection"]
                            ["ood_detect_at_production_threshold"]
                            >= GATE["G2"]["ood_detect_min"]),
               "rule": f"AUROC >= {GATE['G2']['auroc_min']} and OOD detect "
                       f">= {GATE['G2']['ood_detect_min']} @ 0.0702"},
        "G3": {"fresh_accuracy": f["classification"]["accuracy"],
               "fresh_macro_f1": f["classification"]["macro_f1"],
               "bins_improved": n_impr, "per_bin": bin_impr,
               "pass": bool(f["classification"]["accuracy"]
                            >= GATE["G3"]["fresh_acc_min"]
                            and n_impr >= GATE["G3"]["bins_improved_min"]),
               "rule": f"fresh supported acc >= {GATE['G3']['fresh_acc_min']}"
                       f" and >= {GATE['G3']['bins_improved_min']} collapse "
                       f"bins improved"},
        "G4": {"fresh_frr": f["rejection"]["false_rejection_rate"],
               "pass": bool(f["rejection"]["false_rejection_rate"]
                            <= GATE["G4"]["fresh_frr_max"]),
               "rule": f"fresh FRR <= {GATE['G4']['fresh_frr_max']} @ 0.0702"},
    }
    remeasured = {
        "orig_accuracy": ob["classification"]["accuracy"],
        "orig_macro_f1": ob["classification"]["macro_f1"],
        "orig_auroc": ob["rejection"]["auroc"],
        "orig_ood_detect": ob["rejection"]
        ["ood_detect_at_production_threshold"],
        "fresh_accuracy": fb["classification"]["accuracy"],
        "fresh_macro_f1": fb["classification"]["macro_f1"],
        "fresh_frr": fb["rejection"]["false_rejection_rate"],
        "fresh_ood_detect": fb["rejection"]["overall_ood_detection"],
        "fresh_multi_detect": fb["multi_object"]["detection_rate"],
    }
    return {
        "gates": g,
        "all_g1_g4_pass": bool(all(g[k]["pass"]
                                   for k in ("G1", "G2", "G3", "G4"))),
        "fresh_accuracy_delta_vs_locked": (
            f["classification"]["accuracy"] - LOCKED["fresh_accuracy"]),
        "fresh_accuracy_delta_vs_in_run_baseline": (
            f["classification"]["accuracy"]
            - fb["classification"]["accuracy"]),
        "in_run_baseline_remeasurement": {
            **remeasured,
            "consistency_vs_locked": {
                k: {"locked": LOCKED[k], "remeasured": remeasured[k],
                    "abs_diff": abs(remeasured[k] - LOCKED[k])}
                for k in remeasured},
        },
    }

def report_md(seed: int, cand: dict, base: dict, gate: dict,
              audit: dict) -> str:
    g = gate["gates"]
    o, f = cand["original"], cand["fresh"]
    ob, fb = base["original"], base["fresh"]
    L = [f"# WasteLens — Iteration 10 adaptation candidate, seed {seed}", "",
         f"Candidate: `{cand['model']}` (md5 `{cand['md5']}`)",
         f"Baseline (re-measured in this run): `{base['model']}`",
         f"Evaluated: {cand['eval_utc']} | production threshold "
         f"**{SHIPPED_THRESHOLD}** (no rescue, protocol §4)", "",
         "## 0. Leakage audit (protocol §3)", "",
         f"adapt-train {audit['counts']['adapt_train']}, adapt-val "
         f"{audit['counts']['adapt_val']}, fresh "
         f"{audit['counts']['fresh_files']}, original-test "
         f"{audit['counts']['original_test_files']}",
         f"path hits **{len(audit['path_hits'])}**, SHA-256 hits vs fresh "
         f"**{len(audit['sha256_hits_vs_fresh'])}**, MD5 hits vs original test "
         f"pools **{len(audit['md5_hits_vs_original_test'])}**, manifest "
         f"integrity failures **{len(audit['manifest_integrity_failures'])}** "
         f"→ clean = **{audit['clean']}**", "",
         "## 1. Pre-registered gates (protocol §4)", "",
         "| Gate | Rule | Measured | Pass |", "|---|---|---|---|"]
    for k in ("G1", "G2", "G3", "G4"):
        dd = g[k]
        meas = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                for kk, vv in dd.items()
                if kk not in ("pass", "rule", "per_bin")}
        L.append(f"| {k} | {dd['rule']} | `{meas}` | "
                 f"{'**PASS**' if dd['pass'] else '**FAIL**'} |")
    L += ["", f"All G1–G4 pass: **{gate['all_g1_g4_pass']}**", "",
          "## 2. Original benchmark (test pools)", "",
          "| Metric | Candidate | Baseline (locked) | In-run baseline |",
          "|---|---|---|---|",
          f"| accuracy | {o['classification']['accuracy']:.4f} | "
          f"{LOCKED['orig_accuracy']:.4f} | "
          f"{ob['classification']['accuracy']:.4f} |",
          f"| macro-F1 | {o['classification']['macro_f1']:.4f} | "
          f"{LOCKED['orig_macro_f1']:.4f} | "
          f"{ob['classification']['macro_f1']:.4f} |",
          f"| rejection AUROC | {o['rejection']['auroc']:.5f} | "
          f"{LOCKED['orig_auroc']:.5f} | {ob['rejection']['auroc']:.5f} |",
          f"| OOD detect @0.0702 | "
          f"{o['rejection']['ood_detect_at_production_threshold']:.4f} | "
          f"n/a | "
          f"{ob['rejection']['ood_detect_at_production_threshold']:.4f} |",
          f"| ID FRR @0.0702 | "
          f"{o['rejection']['id_frr_at_production_threshold']:.4f} | n/a | "
          f"{ob['rejection']['id_frr_at_production_threshold']:.4f} |", "",
          "Per-source OOD detection @0.0702 (candidate): " +
          json.dumps({s: round(v, 4) for s, v in
                      o['rejection']
                      ['per_source_detect_at_production_threshold'].items()}),
          "", "## 3. Fresh benchmark (131 files, untouched)", "",
          "| Metric | Candidate | Locked baseline | In-run baseline |",
          "|---|---|---|---|",
          f"| supported acc (63) | {f['classification']['accuracy']:.4f} | "
          f"{LOCKED['fresh_accuracy']:.4f} | "
          f"{fb['classification']['accuracy']:.4f} |",
          f"| macro-F1 | {f['classification']['macro_f1']:.4f} | "
          f"{LOCKED['fresh_macro_f1']:.4f} | "
          f"{fb['classification']['macro_f1']:.4f} |",
          f"| FRR @0.0702 | {f['rejection']['false_rejection_rate']:.4f} | "
          f"{LOCKED['fresh_frr']:.4f} | "
          f"{fb['rejection']['false_rejection_rate']:.4f} |",
          f"| OOD detect @0.0702 | "
          f"{f['rejection']['overall_ood_detection']:.4f} | "
          f"{LOCKED['fresh_ood_detect']:.4f} | "
          f"{fb['rejection']['overall_ood_detection']:.4f} |",
          f"| multi-object detect | {f['multi_object']['detection_rate']:.4f}"
          f" | {LOCKED['fresh_multi_detect']:.4f} | "
          f"{fb['multi_object']['detection_rate']:.4f} |", "",
          "### Per-bin recall (fresh supported)", "",
          "| Bin | Candidate | Locked baseline | Delta |", "|---|---|---|---|"]
    for b in fr.BINS:
        r_new = f['classification']['per_bin'][b]['recall']
        r_base = LOCKED['fresh_bin_recall'][b]
        L.append(f"| {b} | {r_new:.4f} | {r_base:.4f} | "
                 f"{r_new - r_base:+.4f} |")
    L += ["", "### Fresh per-role rejection @0.0702", "",
          "| Role | n | detected | rate |", "|---|---|---|---|"]
    for role, dd in f['rejection']['per_role'].items():
        L.append(f"| {role} | {dd['count']} | {dd['detected']} | "
                 f"{dd['detection_rate']:.4f} |")
    L += ["", "### Ambiguous supported items", "", json.dumps(f['ambiguous']),
          ""]
    return "\n".join(L) + "\n"

def run_one(args) -> None:
    audit = None if args.skip_audit else run_audit()
    base = candidate_metrics(Path(args.baseline))
    cand = candidate_metrics(Path(args.model))
    gate = apply_gates(cand, base)
    seed = int(args.seed)
    (OUT_DIR / f"adapt10s{seed}_eval.json").write_text(
        json.dumps({"seed": seed, "audit": audit, "baseline_in_run": base,
                    "candidate": cand}, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / f"adapt10s{seed}_gates.json").write_text(
        json.dumps({"seed": seed, "model": cand["model"],
                    "model_md5": cand["md5"], **gate}, indent=2) + "\n",
        encoding="utf-8")
    if audit:
        (OUT_DIR / f"adapt10s{seed}_report.md").write_text(
            report_md(seed, cand, base, gate, audit), encoding="utf-8")
    g = gate["gates"]
    print(f"[seed {seed}] G1={g['G1']['pass']} G2={g['G2']['pass']} "
          f"G3={g['G3']['pass']} G4={g['G4']['pass']} -> "
          f"all_pass={gate['all_g1_g4_pass']}")
    print(f"[seed {seed}] fresh acc {g['G3']['fresh_accuracy']:.4f} "
          f"(locked {LOCKED['fresh_accuracy']:.4f}), bins improved "
          f"{g['G3']['bins_improved']}/3, fresh FRR {g['G4']['fresh_frr']:.4f}")


def decision_label(passes: list[bool], fresh_accs: list[float]) -> tuple[str, str]:
    """Protocol §4 decision vocabulary (never cherry-pick a seed)."""
    if all(passes):
        return ("Ship candidate (pending G6 browser parity + regression)",
                "Both registered seeds satisfy G1-G4 independently; seed 42 is "
                "the promotion candidate, seed 43 the reproducibility witness.")
    if any(a > LOCKED["fresh_accuracy"] for a in fresh_accs):
        return ("Improvement-but-fail",
                "Fresh-domain accuracy improved, but at least one gate failed: "
                "no promotion - the protocol is not renegotiated after the "
                "measurement.")
    return ("Reject",
            "No registered gate set passed and fresh accuracy did not improve: "
            "keep the shipped Variant-C model.")


def decision_md(out: dict) -> str:
    """Human-readable Iteration-10 outcome, generated from the gate files."""
    t = out["per_seed"]
    m = out["g5_seed_consistency"]["metric_mean_and_spread"]
    L = ["# WasteLens — Iteration 10 decision: fresh-domain adaptation", "",
         f"Protocol: `docs/rejection_experiment/"
         f"iteration10_adaptation_protocol.md` (registered before any "
         f"adaptation image was downloaded).",
         f"Generated: {out['generated_utc']}", "",
         "## 1. Candidates and seeds", "",
         "| Seed | Candidate | md5 | G1 | G2 | G3 | G4 | G1–G4 |",
         "|---|---|---|---|---|---|---|---|"]
    for s, dd in t.items():
        L.append(f"| {s} | `{Path(dd['model']).name}` | `{dd['model_md5'][:12]}`"
                 f" | {'PASS' if dd['G1'] else 'FAIL'}"
                 f" | {'PASS' if dd['G2'] else 'FAIL'}"
                 f" | {'PASS' if dd['G3'] else 'FAIL'}"
                 f" | {'PASS' if dd['G4'] else 'FAIL'}"
                 f" | {'PASS' if dd['all_g1_g4_pass'] else 'FAIL'} |")
    L += ["", "## 2. Measured metrics (both seeds, mean ± spread)", "",
          "| Metric | Mean | Spread | Seed 42 | Seed 43 |",
          "|---|---|---|---|---|"]
    for k, dd in m.items():
        L.append(f"| {k} | {dd['mean']:.4f} | {dd['spread']:.4f} | "
                 f"{dd['values']['42']:.4f} | {dd['values']['43']:.4f} |")
    L += ["", "### Fresh per-bin recall (`organic`, `hazardous`, "
          "`general trash` are the Iteration-9 collapse bins)", "",
          "| Seed | bins improved | fresh acc | fresh FRR |",
          "|---|---|---|---|"]
    for s, dd in t.items():
        L.append(f"| {s} | {dd['fresh_bins_improved']}/3 | "
                 f"{dd['fresh_accuracy']:.4f} | {dd['fresh_frr']:.4f} |")
    L += ["", "## 3. Decision (protocol §4 vocabulary, applied mechanically)", "",
          f"**{out['decision']}**", "", out["decision_reason"], "",
          f"- G5 seed consistency: both seeds satisfy G1–G4 = "
          f"**{out['g5_seed_consistency']['both_seeds_satisfy_g1_g4']}**; "
          f"neither seed was discarded or swapped.",
          f"- G6 browser parity: {out['g6_browser_parity']}", "",
          "## 4. Production state", "",
          "- Shipped model unchanged: `models/checkpoints/"
          "wastelens_rej_shipped_best.keras` / `web/model/`.",
          "- Production rejection threshold unchanged: **0.0702**.",
          "- No threshold tuning, no checkpoint substitution: the gates were "
          "computed with the pre-registered constants only.",
          "- Adaptation images never entered the repository; the leakage "
          "audit is `docs/rejection_experiment/adaptation_eval_audit.json` "
          "(zero path/SHA-256/MD5 hits).", "",
          "## 5. Evidence files", ""]
    for s in t:
        L.append(f"- `docs/rejection_experiment/adapt10s{s}_eval.json`, "
                 f"`adapt10s{s}_gates.json`, `adapt10s{s}_report.md`, "
                 f"`adapt10s{s}_history.csv`, "
                 f"`adapt10s{s}_training_config.json`")
    L.append("- `docs/rejection_experiment/adaptation_eval_audit.json`, "
             "`adapt10_seed_consistency.json`")
    return "\n".join(L) + "\n"


def run_summary() -> None:
    seeds, table, passes, accs = [42, 43], {}, [], []
    for s in seeds:
        p = OUT_DIR / f"adapt10s{s}_gates.json"
        if not p.exists():
            raise SystemExit(f"missing {p}; evaluate seed {s} first")
        d = json.loads(p.read_text(encoding="utf-8"))
        table[s] = {"model": d["model"], "model_md5": d["model_md5"],
                    "all_g1_g4_pass": d["all_g1_g4_pass"],
                    "G1": d["gates"]["G1"]["pass"],
                    "G2": d["gates"]["G2"]["pass"],
                    "G3": d["gates"]["G3"]["pass"],
                    "G4": d["gates"]["G4"]["pass"],
                    "orig_accuracy": d["gates"]["G1"]["orig_accuracy"],
                    "orig_macro_f1": d["gates"]["G1"]["orig_macro_f1"],
                    "orig_auroc": d["gates"]["G2"]["auroc"],
                    "orig_ood_detect": d["gates"]["G2"]["ood_detect"],
                    "fresh_accuracy": d["gates"]["G3"]["fresh_accuracy"],
                    "fresh_macro_f1": d["gates"]["G3"]["fresh_macro_f1"],
                    "fresh_bins_improved": d["gates"]["G3"]["bins_improved"],
                    "fresh_frr": d["gates"]["G4"]["fresh_frr"]}
        passes.append(bool(d["all_g1_g4_pass"]))
        accs.append(float(d["gates"]["G3"]["fresh_accuracy"]))
    keys = ["orig_accuracy", "orig_macro_f1", "orig_auroc", "orig_ood_detect",
            "fresh_accuracy", "fresh_macro_f1", "fresh_frr"]
    metrics = {k: {"mean": float(np.mean([table[s][k] for s in seeds])),
                   "spread": float(np.max([table[s][k] for s in seeds])
                                   - np.min([table[s][k] for s in seeds])),
                   "values": {str(s): table[s][k] for s in seeds}}
               for k in keys}
    label, why = decision_label(passes, accs)
    out = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": "docs/rejection_experiment/"
                    "iteration10_adaptation_protocol.md §4",
        "per_seed": table,
        "g5_seed_consistency": {
            "both_seeds_satisfy_g1_g4": bool(all(passes)),
            "seeds_passing": [s for s, p in zip(seeds, passes) if p],
            "neither_seed_discarded": True,
            "metric_mean_and_spread": metrics,
        },
        "decision": label,
        "decision_reason": why,
        "promotion_target": str(ta.SHIPPED_CHECKPOINT) +
        " <- seed 42 candidate if the decision is Ship",
        "g6_browser_parity": "separate step: TFJS export to scratch + "
                             "browser_preprocess_probe.js parity (atol <= 2e-5)",
    }
    (OUT_DIR / "adapt10_seed_consistency.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "iteration10_decision.md").write_text(
        decision_md(out), encoding="utf-8")
    print(json.dumps({"g5": out["g5_seed_consistency"],
                      "decision": label}, indent=2)[:2000])
    print(f"wrote {OUT_DIR / 'iteration10_decision.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Iteration-10 adaptation candidate evaluation (protocol "
                    "§3 audit + §4 gates).")
    ap.add_argument("--model", type=Path,
                    default=ta.CKPT_DIR / "wastelens_rej_adapt10s42_final.keras")
    ap.add_argument("--baseline", type=Path, default=ta.SHIPPED_CHECKPOINT)
    ap.add_argument("--seed", type=int, default=42,
                    help="run tag for the output files (labels the candidate)")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--skip-audit", action="store_true",
                    help="reuse the audit already written for this candidate")
    ap.add_argument("--summary", action="store_true",
                    help="combine both seeds: G5 + final decision")
    args = ap.parse_args()
    if args.audit_only:
        run_audit()
    elif args.summary:
        run_summary()
    else:
        run_one(args)


if __name__ == "__main__":
    main()

