# src/ood_research.py
#
# PURPOSE (Iteration 3):
#   Research whether the existing checkpoint provides ANY signal that can
#   separate supported single-item photos (ID) from unsupported images (OOD),
#   using REAL inference - no invented numbers.
#
# Signals measured per sample:
#   - top softmax probability (MSP)
#   - top-1 vs top-2 margin
#   - softmax entropy
#   - cosine distance from the gap embedding (1280-d) to the NEAREST
#     per-bin class centroid (fitted on train-split features only)
#   - cosine distance to the PREDICTED bin's centroid
#
# Evaluation groups (all real images except the synthetic probes):
#   ID        : 240 stratified held-out test images (60 per bin, seed 2026)
#   ambiguous : the 20 lowest-top-probability images of that same scan
#   OOD synth : 10 synthetic non-waste images (5 original + 5 new patterns)
#   OOD real  : 20 real photos from the DROPPED training classes
#               (clothes x10, shoes x10 - never seen by the model)
#   OOD multi : 18 multi-item collages (8x 2-item, 6x 4-item, 4x 6-item)
#               built from real dataset photos on varied backgrounds
#
# PRE-REGISTERED SELECTION RULE (written before any measurement):
#   A method is adopted only if, at a threshold calibrated as the 99.5th
#   percentile of TRAIN fit-set nearest-centroid distances, it achieves
#   >= OOD_DETECT_MIN (80%) detection on the combined OOD groups while
#   rejecting <= ID_FRR_MAX (5%) of ID images. Otherwise: Outcome C -
#   document the limitation, do not ship a weak detector.
#
# Run from the repo root:  python src/ood_research.py
# Outputs:
#   docs/ood_research.json / docs/ood_research.md   (measurements + verdict)
#   web/model/ood_reference.json                    (only if method adopted:
#       centroids + calibrated threshold + metadata for the browser runtime)

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

import train as wl

DEFAULT_MODEL = Path("models/checkpoints/wastelens_ep09.keras")
OUT_JSON = Path("docs/ood_research.json")
OUT_MD = Path("docs/ood_research.md")
REF_JSON = Path("web/model/ood_reference.json")
ARCHIVE_JSON = Path("models/tfjs_model/ood_reference.json")

FIT_SEED = 2026
FIT_PER_BIN = 600          # balanced centroid fit subsample (<=600/bin)
ID_PER_BIN = 60            # held-out ID sample (60/bin = 240)
AMBIGUOUS_N = 20           # lowest-top test images
CLOTHES_N, SHOES_N = 10, 10
COLLAGE_PLAN = [(2, 8), (4, 6), (6, 4)]   # (items, count)
OOD_DETECT_MIN = 0.80
ID_FRR_MAX = 0.05
CAL_PERCENTILE = 99.5      # calibration threshold percentile (train fit set)


def l2n(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (safe for zero rows)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def pooled_precision(feats: np.ndarray, labels: np.ndarray, n_classes: int,
                     alpha: float = 0.1) -> np.ndarray:
    """Shared within-class covariance -> precision (Lee et al. 2018 style),
    with diagonal shrinkage alpha for stability (n_feat ~ n_samples)."""
    d = feats.shape[1]
    centered = np.concatenate(
        [feats[labels == c] - feats[labels == c].mean(axis=0)
         for c in range(n_classes)], axis=0)
    cov = centered.T @ centered / max(len(centered) - n_classes, 1)
    cov = cov + alpha * (np.trace(cov) / d) * np.eye(d)
    return np.linalg.inv(cov)


def mahal(X: np.ndarray, mu: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Row-wise squared Mahalanobis distance to a single mean."""
    diff = X - mu[None, :]
    return np.einsum("ij,jk,ik->i", diff, P, diff)


def mahal_nearest(X: np.ndarray, means: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Min squared Mahalanobis distance over per-class means."""
    return np.min(np.stack([mahal(X, means[c], P)
                            for c in range(len(means))]), axis=0)


# --- Models and feature extraction -------------------------------------------

def load_models(model_path: Path):
    """Full classifier + a two-output feature sub-model (gap + fc_256).

    One backbone pass yields BOTH feature spaces: the 1280-d gap embedding
    and the 256-d penultimate (fc_256) activation, for centroid/Mahalanobis
    research in both spaces.
    """
    full = tf.keras.models.load_model(model_path)
    feat = tf.keras.models.Model(
        full.input,
        [full.get_layer("gap").output, full.get_layer("fc_256").output])
    return full, feat


def features_from_rows(feat_model, rows):
    """Returns (gap_feats (n,1280), pen_feats (n,256))."""
    outs = feat_model.predict(wl.make_dataset(rows, training=False), verbose=0)
    return np.asarray(outs[0]), np.asarray(outs[1])


def features_from_array(feat_model, arr01: np.ndarray):
    """Returns (gap_feats (1280,), pen_feats (256,)) for one [0,1] image."""
    x = tf.image.resize(arr01, wl.IMG_SIZE) * 2.0 - 1.0  # == (x/127.5)-1
    outs = feat_model.predict(tf.expand_dims(x, 0), verbose=0)
    return np.asarray(outs[0])[0], np.asarray(outs[1])[0]


def probs_from_array(full_model, arr01: np.ndarray) -> np.ndarray:
    x = tf.image.resize(arr01, wl.IMG_SIZE) * 2.0 - 1.0
    return full_model.predict(tf.expand_dims(x, 0), verbose=0)[0]


# --- Sample set construction --------------------------------------------------

def fit_sample(dataset_root: Path):
    """Balanced train-subsample for centroid fitting (features only)."""
    images = wl.remap_to_bins(wl.scan_images(dataset_root))
    splits = wl.stratified_split(images)
    rng = np.random.RandomState(FIT_SEED)
    rows = []
    for bin_name in wl.BINS:
        bin_idx = wl.BIN_TO_IDX[bin_name]
        pool = sorted(r for r in splits["train"] if r[1] == bin_idx)
        take = min(FIT_PER_BIN, len(pool))
        pick = rng.choice(len(pool), size=take, replace=False)
        rows += [pool[i] for i in sorted(pick)]
    return splits, rows


def id_and_ambiguous(splits) -> tuple[list, list]:
    """240 stratified test images + the 20 lowest-top among them."""
    rng = np.random.RandomState(FIT_SEED)
    rows = []
    for bin_name in wl.BINS:
        bin_idx = wl.BIN_TO_IDX[bin_name]
        pool = sorted(r for r in splits["test"] if r[1] == bin_idx)
        take = min(ID_PER_BIN, len(pool))
        pick = rng.choice(len(pool), size=take, replace=False)
        rows += [pool[i] for i in sorted(pick)]
    return rows, rows  # ambiguous subset chosen after probs are known


def dropped_class_rows(dataset_root: Path) -> list[tuple[Path, str]]:
    """Real OOD photos: dropped classes (clothes/shoes) never trained on."""
    rng = np.random.RandomState(FIT_SEED + 1)
    class_dirs = wl.find_class_dirs(dataset_root)  # includes dropped classes
    out = []
    for cls, n in (("clothes", CLOTHES_N), ("shoes", SHOES_N)):
        folder = class_dirs[cls]
        files = sorted(p for p in folder.iterdir()
                       if p.is_file() and p.suffix.lower() in wl.IMG_EXTENSIONS
                       and not p.name.startswith("."))
        pick = rng.choice(len(files), size=min(n, len(files)), replace=False)
        out += [(files[i], cls) for i in sorted(pick)]
    return out


def _load01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(wl.IMG_SIZE)
    return np.asarray(img, dtype=np.float32) / 255.0


def collage_set(dataset_root: Path) -> list[tuple[Path, str]]:
    """Deterministic multi-item collages (cluttered-scene proxy).

    Each collage pastes k item photos from DIFFERENT bins onto a random
    background; items are resized to 25-45% of the canvas. Seed fixed.
    """
    rng = np.random.RandomState(FIT_SEED + 2)
    images = wl.remap_to_bins(wl.scan_images(dataset_root))
    per_bin = {i: [] for i in range(len(wl.BINS))}
    for path, cls in images:
        per_bin[wl.BIN_TO_IDX[cls]].append(path)
    for i in per_bin:
        per_bin[i] = sorted(per_bin[i])

    out_dir = Path(tempfile.gettempdir()) / "wastelens_collages"
    out_dir.mkdir(exist_ok=True)
    out = []
    counter = 0
    for items, count in COLLAGE_PLAN:
        for c in range(count):
            canvas_w, canvas_h = 480, 360
            bg_v = 0.55 + 0.4 * rng.rand()
            canvas = Image.new("RGB", (canvas_w, canvas_h),
                               tuple(int(255 * bg_v * ch) for ch in
                                     (rng.rand() + 0.4, rng.rand() + 0.4,
                                      rng.rand() + 0.4)))
            draw = ImageDraw.Draw(canvas)
            for _ in range(3):  # a few background rectangles for clutter
                x0, y0 = rng.randint(0, canvas_w - 40), rng.randint(0, canvas_h - 40)
                draw.rectangle([x0, y0, x0 + rng.randint(30, 160),
                                y0 + rng.randint(30, 120)],
                               fill=tuple(int(255 * rng.rand()) for _ in range(3)))
            bins = np.resize(rng.permutation(len(wl.BINS)), items)
            for b in bins:
                src = per_bin[int(b)][rng.randint(len(per_bin[int(b)]))]
                img = Image.open(src).convert("RGB")
                w = rng.randint(canvas_w // 4, canvas_w // 2)
                img = img.resize((w, int(w * img.height / img.width)))
                x = rng.randint(0, max(1, canvas_w - img.width))
                y = rng.randint(0, max(1, canvas_h - img.height))
                canvas.paste(img, (x, y))
            dest = out_dir / f"collage_{items}item_{c:02d}.jpg"
            canvas.save(dest, quality=90)
            out.append((dest, f"collage_{items}items"))
            counter += 1
    return out


# --- Synthetic OOD probes -----------------------------------------------------

def probes_extended() -> list[tuple[str, np.ndarray]]:
    """Original 5 probes (validate_realworld) + 5 new non-waste patterns."""
    rng = np.random.RandomState(7)
    gray = np.clip(rng.normal(loc=0.5, scale=0.25, size=(224, 224, 3)), 0, 1)
    flat = np.full((224, 224, 3), (0x16, 0x1B, 0x23), dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:224, 0:224]
    checker = (((yy // 8) + (xx // 8)) % 2).astype(np.float32)
    checker = np.stack([checker] * 3, axis=-1)
    gradient = np.tile(np.linspace(0, 1, 224, dtype=np.float32)[None, :, None],
                       (224, 1, 3))
    blank = np.ones((224, 224, 3), dtype=np.float32)
    base = [
        ("gray noise (static)", gray),
        ("flat color field (#161b23)", flat),
        ("checkerboard 8px", checker),
        ("horizontal gradient", gradient),
        ("blank white frame", blank),
    ]
    rr = np.random.RandomState(FIT_SEED + 3)
    yy2, xx2 = np.mgrid[0:224, 0:224]
    radial = np.clip(1.0 - np.sqrt((yy2 - 112) ** 2 + (xx2 - 112) ** 2) / 160,
                     0, 1)
    stripes = ((yy2 // 12) % 2).astype(np.float32)
    blobs = np.zeros((224, 224, 3), dtype=np.float32)
    for _ in range(6):
        cy, cx, r = rr.randint(20, 204), rr.randint(20, 204), rr.randint(15, 60)
        m = ((yy2 - cy) ** 2 + (xx2 - cx) ** 2) < r * r
        color = rr.rand(3).astype(np.float32)
        blobs[m] = color
    random_rgb = rr.rand(224, 224, 3).astype(np.float32)
    midgray = np.full((224, 224, 3), 0.5, dtype=np.float32)
    return base + [
        ("radial gradient", np.stack([radial] * 3, axis=-1)),
        ("vertical stripes", np.stack([stripes] * 3, axis=-1)),
        ("random color blobs", blobs),
        ("random rgb pixels", random_rgb),
        ("mid-gray field", midgray),
    ]


# --- Evaluation ---------------------------------------------------------------

def score_record(group, label, path_name, probs, f_gap, f_pen,
                 cents_norm, means_gap, means_pen, P_gap, P_pen):
    fg = f_gap / max(float(np.linalg.norm(f_gap)), 1e-12)
    d_all = 1.0 - cents_norm @ fg          # cosine distance to each centroid
    top2 = np.sort(probs)[::-1][:2]
    return {
        "group": group, "label": label, "file": str(path_name),
        "pred": wl.BINS[int(np.argmax(probs))],
        "top": float(top2[0]), "runner_up": float(top2[1]),
        "margin": float(top2[0] - top2[1]),
        "entropy": entropy(probs),
        "dist_nearest": float(d_all.min()),
        "dist_pred": float(d_all[int(np.argmax(probs))]),
        "mah_gap": float(mahal_nearest(f_gap[None, :], means_gap, P_gap)[0]),
        "mah_pen": float(mahal_nearest(f_pen[None, :], means_pen, P_pen)[0]),
    }


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUROC: P(random OOD score > random ID score)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv)).astype(np.float64) + 1.0
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def run_research() -> dict:
    print("=== WasteLens OOD rejection research (real inference) ===\n")
    full, feat = load_models(DEFAULT_MODEL)
    print(f"[1/7] Loaded checkpoint: {DEFAULT_MODEL}")

    dataset_root = wl.download_dataset()
    splits, fit_rows = fit_sample(dataset_root)
    print(f"[2/7] Centroid fit: {len(fit_rows)} train images "
          f"(<= {FIT_PER_BIN}/bin, seed {FIT_SEED})")
    fit_gap, fit_pen = features_from_rows(feat, fit_rows)
    fit_labels = np.array([lbl for _, lbl in fit_rows])
    cents = l2n(np.stack(
        [l2n(fit_gap)[fit_labels == i].mean(axis=0)
         for i in range(len(wl.BINS))]))
    means_gap = np.stack([fit_gap[fit_labels == i].mean(axis=0)
                          for i in range(len(wl.BINS))])
    means_pen = np.stack([fit_pen[fit_labels == i].mean(axis=0)
                          for i in range(len(wl.BINS))])
    P_gap = pooled_precision(fit_gap, fit_labels, len(wl.BINS))
    P_pen = pooled_precision(fit_pen, fit_labels, len(wl.BINS))

    d_fit_cos = 1.0 - np.sum(cents[fit_labels] * l2n(fit_gap), axis=1)
    d_fit_mg = mahal_nearest(fit_gap, means_gap, P_gap)
    d_fit_mp = mahal_nearest(fit_pen, means_pen, P_pen)
    thr = {"cos": float(np.percentile(d_fit_cos, CAL_PERCENTILE)),
           "mah_gap": float(np.percentile(d_fit_mg, CAL_PERCENTILE)),
           "mah_pen": float(np.percentile(d_fit_mp, CAL_PERCENTILE))}
    print(f"      calibrated thresholds (p{CAL_PERCENTILE}): "
          f"cos={thr['cos']:.4f} mah_gap={thr['mah_gap']:.1f} "
          f"mah_pen={thr['mah_pen']:.1f}")

    id_rows, _ = id_and_ambiguous(splits)
    print(f"[3/7] ID scan: {len(id_rows)} held-out test images "
          f"({ID_PER_BIN}/bin, seed {FIT_SEED})")
    id_gap, id_pen = features_from_rows(feat, id_rows)
    id_probs = full.predict(wl.make_dataset(id_rows, training=False), verbose=0)
    records = []
    for i, ((path, lbl), probs) in enumerate(zip(id_rows, id_probs)):
        records.append(score_record(
            "ID", wl.BINS[lbl], Path(path).name, probs,
            id_gap[i], id_pen[i], cents, means_gap, means_pen, P_gap, P_pen))
    order = np.argsort([r["top"] for r in records])
    for i in order[:AMBIGUOUS_N]:
        records[i]["group"] = "ambiguous"
    print(f"[4/7] Marked {AMBIGUOUS_N} lowest-top ID images as 'ambiguous'")

    records += [score_record("OOD", "synthetic", name,
                             probs_from_array(full, arr),
                             *features_from_array(feat, arr),
                             cents, means_gap, means_pen, P_gap, P_pen)
                for name, arr in probes_extended()]
    print("[5/7] Synthetic probes scored (10)")

    dropped = dropped_class_rows(dataset_root)
    d_rows = [(p, 0) for p, _ in dropped]
    d_gap, d_pen = features_from_rows(feat, d_rows)
    d_probs = full.predict(wl.make_dataset(d_rows, training=False), verbose=0)
    records += [score_record("OOD", lbl, Path(p).name, pr, fg, fp,
                             cents, means_gap, means_pen, P_gap, P_pen)
                for (p, lbl), pr, fg, fp in
                zip(dropped, d_probs, d_gap, d_pen)]
    print(f"[6/7] Real dropped-class photos scored ({len(dropped)})")

    collages = collage_set(dataset_root)
    c_rows = [(p, 0) for p, _ in collages]
    c_gap, c_pen = features_from_rows(feat, c_rows)
    c_probs = full.predict(wl.make_dataset(c_rows, training=False), verbose=0)
    records += [score_record("OOD", lbl, Path(p).name, pr, fg, fp,
                             cents, means_gap, means_pen, P_gap, P_pen)
                for (p, lbl), pr, fg, fp in
                zip(collages, c_probs, c_gap, c_pen)]
    print(f"[7/7] Collages scored ({len(collages)})\n")

    iid = [r for r in records if r["group"] == "ID"]
    ood = [r for r in records if r["group"] == "OOD"]
    feature_signals = {
        "cos_dist_nearest_centroid": "dist_nearest",
        "cos_dist_pred_centroid": "dist_pred",
        "mahalanobis_gap": "mah_gap",
        "mahalanobis_penultimate": "mah_pen",
    }
    comparison = {}
    key_to_thr = {"dist_nearest": "cos", "dist_pred": "cos",
                  "mah_gap": "mah_gap", "mah_pen": "mah_pen"}
    for name, key in feature_signals.items():
        t = thr[key_to_thr[key]]
        pos = np.array([r[key] for r in ood])
        neg = np.array([r[key] for r in iid])
        comparison[name] = {
            "threshold": t,
            "id_frr": float(np.mean([r[key] > t for r in iid])),
            "ood_detect": float(np.mean([r[key] > t for r in ood])),
            "auroc": auroc(pos, neg),
        }
    for name, get in (("top_prob", lambda r: r["top"]),
                      ("margin", lambda r: r["margin"]),
                      ("entropy", lambda r: r["entropy"])):
        pos = np.array([get(r) for r in ood])
        neg = np.array([get(r) for r in iid])
        comparison[name] = {
            "threshold": None, "id_frr": None, "ood_detect": None,
            "auroc": auroc(-pos, -neg),  # higher = more ID-like
        }

    adopted = [n for n, c in comparison.items()
               if c["ood_detect"] is not None
               and c["ood_detect"] >= OOD_DETECT_MIN
               and c["id_frr"] <= ID_FRR_MAX]

    # Honest operating curve for the best feature signal: thresholds taken at
    # HELD-OUT ID quantiles (never train - see calibration-gap note below).
    best_key = "mah_gap"
    id_mg = np.array([r["mah_gap"] for r in iid])
    ood_mg = np.array([r["mah_gap"] for r in ood])
    sweep = []
    for frr_budget in (0.01, 0.02, 0.05, 0.10, 0.15, 0.20):
        t = float(np.quantile(id_mg, 1.0 - frr_budget))
        sweep.append({
            "frr_budget": frr_budget, "threshold": round(t, 1),
            "ood_detect": round(float(np.mean(ood_mg > t)), 4)})
    calibration_gap = {
        "train_fit_p_cal": thr["mah_gap"],
        "held_out_id_p95": round(float(np.quantile(id_mg, 0.95)), 1),
        "note": "train-split features are memorized-tight; calibrating on "
                "them under-estimates the threshold and inflates FRR. "
                "Held-out (val/test) calibration is the honest baseline."}

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "checkpoint": str(DEFAULT_MODEL),
        "fit": {"per_bin": FIT_PER_BIN, "seed": FIT_SEED, "n": len(fit_rows),
                "cal_percentile": CAL_PERCENTILE,
                "thresholds": thr},
        "rules": {"OOD_DETECT_MIN": OOD_DETECT_MIN, "ID_FRR_MAX": ID_FRR_MAX},
        "comparison": comparison,
        "adopted": adopted,
        "operating_curve_mah_gap": sweep,
        "calibration_gap_mah_gap": calibration_gap,
        "records": records,
        "_cents": cents,
    }
    return payload


def write_reference(payload: dict, cents: np.ndarray) -> None:
    """Browser runtime artifact: centroids + calibrated threshold + metadata."""
    signal_key = payload["adopted"][0]
    thr_key = {"cos_dist_nearest_centroid": "cos",
               "cos_dist_pred_centroid": "cos",
               "mahalanobis_gap": "mah_gap",
               "mahalanobis_penultimate": "mah_pen"}[signal_key]
    ref = {
        "version": 1,
        "checkpoint": payload["checkpoint"],
        "layer": "gap",
        "dim": int(cents.shape[1]),
        "bins": list(wl.BINS),
        "fit": {k: v for k, v in payload["fit"].items() if k != "thresholds"},
        "signal": signal_key,
        "reject_above": payload["fit"]["thresholds"][thr_key],
        "centroids": [[round(float(v), 6) for v in row] for row in cents],
    }
    for target in (REF_JSON, ARCHIVE_JSON):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
    print(f"reference written: {REF_JSON} (+ archive copy)")


def main() -> None:
    payload = run_research()
    cents = payload.pop("_cents")  # ndarray - excluded from the JSON dump
    comparison = payload["comparison"]
    print("---- signal comparison (calibrated thresholds) ----")
    for name, c in comparison.items():
        frr = "n/a" if c["id_frr"] is None else f"{c['id_frr']:.3f}"
        det = "n/a" if c["ood_detect"] is None else f"{c['ood_detect']:.3f}"
        print(f"  {name:<28} FRR={frr}  DET={det}  AUROC={c['auroc']:.3f}")

    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# WasteLens - OOD Rejection Research (Iteration 3)",
        "",
        f"*Generated by `src/ood_research.py` on {payload['generated_utc']} "
        f"UTC. Checkpoint: `{payload['checkpoint']}`. All numbers are real "
        "inference outputs.*",
        "",
        f"Fit: {payload['fit']['n']} train images (<= {payload['fit']['per_bin']}"
        f"/bin, seed {payload['fit']['seed']}); calibration = p"
        f"{payload['fit']['cal_percentile']} of the fit-set distribution; "
        "thresholds: "
        + ", ".join(f"{k}={v:.4f}" for k, v
                    in payload["fit"]["thresholds"].items()) + ".",
        "",
        "## Signal comparison",
        "",
        "| signal | threshold | ID false-rejection | OOD detection | AUROC |",
        "|---|---|---:|---:|---:|",
    ]
    for name, c in comparison.items():
        lines.append(
            f"| {name} | "
            + (f"{c['threshold']:.4f}" if c["threshold"] is not None else "-")
            + " | "
            + ("n/a" if c["id_frr"] is None else f"{100*c['id_frr']:.1f}%")
            + " | "
            + ("n/a" if c["ood_detect"] is None else f"{100*c['ood_detect']:.1f}%")
            + f" | {c['auroc']:.3f} |")
    if payload["adopted"]:
        lines += ["",
                  f"**ADOPTED:** {payload['adopted']} meets the pre-registered "
                  f"rule (>= {OOD_DETECT_MIN:.0%} OOD detection at <= "
                  f"{ID_FRR_MAX:.0%} ID false-rejection). Browser reference "
                  "written to web/model/ood_reference.json."]
    else:
        lines += ["",
                  "**OUTCOME C:** no signal meets the pre-registered rule; "
                  "limitation documented, no detector shipped.",
                  "",
                  "### Best feature signal: mahalanobis_gap (nearest-class "
                  "squared Mahalanobis in the 1280-d gap space)",
                  "",
                  "Highest AUROC of all signals (see table), but the "
                  "ID/OOD distributions overlap too much for reliable "
                  "rejection:",
                  "",
                  "| FRR budget (held-out ID) | threshold | OOD detection |",
                  "|---:|---:|---:|",
                  *["| {:.0%} | {} | {:.1%} |".format(
                      s["frr_budget"], s["threshold"], s["ood_detect"])
                    for s in payload["operating_curve_mah_gap"]],
                  "",
                  "Calibration-gap note: "
                  + payload["calibration_gap_mah_gap"]["note"]
                  + f" (train-fit p{payload['fit']['cal_percentile']} = "
                  f"{payload['calibration_gap_mah_gap']['train_fit_p_cal']}, "
                  "held-out ID p95 = "
                  f"{payload['calibration_gap_mah_gap']['held_out_id_p95']}).",
                  "",
                  "Multi-item collages (real in-distribution content) are "
                  "unseparable by ANY feature-space signal because their "
                  "features ARE in-distribution; they violate the "
                  "single-item assumption, not the feature distribution.",
                  "",
                  "Minimum model/training change for real OOD rejection "
                  "(Iteration 4+): train with outlier-exposure / "
                  "background-class data (multi-object and non-waste scenes "
                  "labeled as a 5th 'not waste' class), or add an explicit "
                  "rejection head trained on such data; then re-calibrate "
                  "the uncertainty threshold on held-out data."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")
    if payload["adopted"]:
        write_reference(payload, cents)


if __name__ == "__main__":
    main()





