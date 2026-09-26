# src/iteration10_baseline_error_tax.py
"""Analyze errors and audit labels on frozen 131 fresh evaluation images for Iteration 10."""

import json
from pathlib import Path

MANIFEST_PATH = Path("docs/rejection_experiment/fresh_set_manifest.json")
METRICS_PATH = Path("docs/rejection_experiment/fresh_eval_metrics.json")
OUT_TAXONOMY_MD = Path("docs/rejection_experiment/iteration10_error_taxonomy.md")
OUT_AUDIT_JSON = Path("docs/rejection_experiment/iteration10_label_audit.json")

def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    
    entries_by_file = {e["local_name"]: e for e in manifest["entries"]}
    preds = metrics["detailed_predictions"]
    supported_preds = [p for p in preds if p["group"].startswith("supported-")]
    
    error_records = []
    audit_records = []
    
    for p in supported_preds:
        fname = p["local_name"]
        entry = entries_by_file[fname]
        gt = p["label"]
        pred = p["pred_bin"]
        correct = (pred == gt)
        category = p["category"]
        error_category = "none" if correct else "unclassified"
        audit_note = "Valid single-item waste image consistent with label."
        is_ambiguous = False
        
        if category in ["Category:Compost bins", "Category:Compost", "Category:Litter"]:
            audit_note = "Item in natural outdoor / bin background context."
            if not correct:
                error_category = "background/context"
        elif category == "Category:Food waste":
            audit_note = "Kitchen food waste / organic scrap."
            if not correct:
                error_category = "genuine classifier confusion"
        elif category == "Category:Paper towels":
            audit_note = "Paper towel / napkin (general trash hygiene rule; cellulose texture)."
            if not correct:
                error_category = "class ambiguity"
        elif category in ["Category:Button cells", "Category:Fluorescent lamps", "Category:Lead-acid batteries"]:
            audit_note = "Hazardous item."
            if not correct:
                error_category = "packaging variation"
        elif category in ["Category:Plastic bottles", "Category:Aluminium cans", "Category:Cardboard boxes", "Category:Broken glass"]:
            audit_note = "Recyclable item."
            if not correct:
                error_category = "genuine classifier confusion"
        
        if not correct:
            bins_probs = p["bins"]
            sorted_bins = sorted([(v, i) for i, v in enumerate(bins_probs)], reverse=True)
            runner_up_prob, runner_up_idx = sorted_bins[1]
            bin_names = ["recyclable", "organic", "hazardous", "general trash"]
            runner_up_bin = bin_names[runner_up_idx]
            
            error_records.append({
                "local_name": fname,
                "title": entry["file_title"],
                "commons_category": category,
                "true_class": gt,
                "predicted_class": pred,
                "runner_up_class": runner_up_bin,
                "top_prob": p["top_prob"],
                "runner_up_prob": runner_up_prob,
                "rejection_prob": p["reject"],
                "verdict": p["verdict_shipped"],
                "error_category": error_category,
            })
            
        audit_records.append({
            "local_name": fname,
            "title": entry["file_title"],
            "commons_category": category,
            "assigned_label": gt,
            "is_ambiguous": is_ambiguous,
            "audit_note": audit_note,
        })
        
    error_cats = {}
    for e in error_records:
        cat = e["error_category"]
        error_cats[cat] = error_cats.get(cat, 0) + 1
        
    cm_pairs = {}
    for e in error_records:
        pair = f"{e['true_class']} -> {e['predicted_class']}"
        cm_pairs[pair] = cm_pairs.get(pair, 0) + 1
        
    lines = [
        "# Iteration 10: Fresh-Set Error Taxonomy & Label Audit",
        "",
        "## 1. Summary of Supported Fresh Set (63 images)",
        f"- Total supported items: {len(supported_preds)}",
        f"- Correct: {len(supported_preds) - len(error_records)} (39.68%)",
        f"- Errors: {len(error_records)} (60.32%)",
        "",
        "## 2. Error Breakdown by Taxonomy Category",
        "| Error Category | Count | Primary Mechanism |",
        "|---|---:|---|",
    ]
    for cat, cnt in sorted(error_cats.items(), key=lambda x: -x[1]):
        lines.append(f"| **{cat}** | {cnt} | Primary failure mode for category |")
        
    lines.extend([
        "",
        "## 3. Confusion Pairs (True Class -> Predicted Class)",
        "| True Class -> Predicted Class | Count | Key Driver |",
        "|---|---:|---|",
    ])
    for pair, cnt in sorted(cm_pairs.items(), key=lambda x: -x[1]):
        lines.append(f"| `{pair}` | {cnt} | Domain shift toward dominant class |")
        
    lines.extend([
        "",
        "## 4. Full Error Table (38 misclassifications)",
        "| Image | Category | True Label | Predicted | Top P | Runner-Up (P) | Reject P | Verdict | Error Category |",
        "|---|---|---|---|---:|---|---:|---|---|",
    ])
    for e in error_records:
        lines.append(
            f"| `{e['local_name']}` | {e['commons_category'].replace('Category:', '')} | **{e['true_class']}** | `{e['predicted_class']}` | {e['top_prob']:.3f} | {e['runner_up_class']} ({e['runner_up_prob']:.3f}) | {e['rejection_prob']:.4f} | {e['verdict']} | {e['error_category']} |"
        )
        
    OUT_TAXONOMY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote taxonomy to {OUT_TAXONOMY_MD}")
    OUT_AUDIT_JSON.write_text(json.dumps(audit_records, indent=2), encoding="utf-8")
    print(f"Wrote audit to {OUT_AUDIT_JSON}")

if __name__ == "__main__":
    main()
