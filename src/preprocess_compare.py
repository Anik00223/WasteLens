# src/preprocess_compare.py
#
# WasteLens Iteration 9 - LOOP 2/3/4: quantify the Keras-vs-browser image
# preprocessing discrepancy and test "browser-equivalent" Python variants.
#
# The two production paths are:
#   KERAS  (src/train.py make_dataset / _load, used for EVERY shipped metric):
#       tf.io.decode_image(channels=3) -> tf.image.resize(224,224)
#          (bilinear, half-pixel centres, antialias=False)
#       -> mobilenet_v2.preprocess_input  == (x/127.5)-1
#   BROWSER (web/index.html runInference, the RUNTIME path):
#       tf.browser.fromPixels(img)       (canvas decode, RGBA)
#       -> tf.image.resizeBilinear(224,224) -> div(127.5).sub(1)
#
# Normalisation math is identical and the model stage is already proven
# (parity <= 1e-5 on identical inputs), so any difference must come from
# DECODE (TF decoder vs Chrome/Skia decoder + colour management) or RESIZE.
#
# This tool:
#   1. dump-inputs : writes scratch/prep_probe/<name>.py.rgb (raw uint8 RGB as
#      TF decodes it) + manifest.json for the JS probe, and records the
#      Keras-path tensor + predictions.
#   2. compare     : reads the probe output (decode diff stats, browser tensor,
#      browser predictions) and measures, per Python variant: full-image decode
#      pixel diff, max normalised-input diff, max prediction diff, verdict
#      agreement. Writes docs/rejection_experiment/preprocess_alignment.json.
#
# Variants (browser-equivalent candidates, ALL experiments - production stays
# untouched unless the registered gate in LOOP 5 passes):
#   tf      : current production Python path
#   pil     : Pillow RGB decode + Pillow BILINEAR resize
#   pil_icm : Pillow decode + embedded-ICC -> sRGB (Chrome-like colour
#             management) + Pillow BILINEAR resize
#   pil_tf  : Pillow decode + ICC->sRGB + TF bilinear resize
#   tf_aa   : TF decode + TF resize antialias=True (control: tfjs has no AA)
#
# Run from repo root:
#   py -3.13 -W ignore src/preprocess_compare.py dump-inputs --images <json>
#   py -3.13 -W ignore src/preprocess_compare.py compare --probe <json>

from __future__ import annotations

import argparse
import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageCms

import train as wl

THRESHOLD = 0.0702          # shipped (unchanged this iteration)
MIN_TOP, MIN_MARGIN = 0.60, 0.50
PROBE_DIR = Path("scratch/prep_probe")
OUT_JSON = Path("docs/rejection_experiment/preprocess_alignment.json")


def tf_decode(path: str) -> np.ndarray:
    """Production decode: uint8 HWC RGB exactly as train.py sees it."""
    data = tf.io.read_file(path)
    img = tf.io.decode_image(data, channels=3, expand_animations=False)
    return img.numpy().astype(np.uint8)


def resize_tf(arr) -> np.ndarray:
    """TF bilinear resize + MobileNet normalisation (the shipped formula)."""
    x = tf.image.resize(tf.cast(arr, tf.float32), wl.IMG_SIZE).numpy()
    return (x / 127.5) - 1.0


def resize_conv(arr, mode: str) -> np.ndarray:
    """Bilinear resize with an EXPLICIT source-coordinate convention.

    Provenance (tfjs-backend-webgl@4.22.0 ResizeBilinearProgram, the shader the
    browser actually runs for web/index.html's tf.image.resizeBilinear):

        source = halfPixelCenters
                 ? (y + 0.5) * (in/out) - 0.5   <- Python tf.image.resize
                 :         y * (in/out)          <- tfjs DEFAULT, production

    Modes: 'tfjs'  = y * in/out                 (browser production path)
           'half'  = (y+0.5)*in/out - 0.5       (Python production path)
           'align' = y*(in-1)/(out-1)           (alignCorners=true control)
    Coordinates are clamped to [0, n-1] exactly like both implementations.
    """
    a = np.asarray(arr, dtype=np.float32)
    H, W = a.shape[:2]
    ho, wo = wl.IMG_SIZE

    def coord(ni: int, no: int) -> np.ndarray:
        o = np.arange(no, dtype=np.float32)
        s = ni / no
        if mode == "tfjs":
            return o * s
        if mode == "half":
            return (o + 0.5) * s - 0.5
        if mode == "align":
            return o * (ni - 1) / (no - 1)
        raise ValueError(f"unknown mode {mode}")

    ys = np.clip(coord(H, ho), 0, H - 1)
    xs = np.clip(coord(W, wo), 0, W - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = (ys - y0).astype(np.float32)[:, None, None]
    fx = (xs - x0).astype(np.float32)[None, :, None]
    Ia, Ib = a[y0][:, x0], a[y0][:, x1]
    Ic, Id = a[y1][:, x0], a[y1][:, x1]
    top = Ia * (1 - fx) + Ib * fx
    bot = Ic * (1 - fx) + Id * fx
    out = top * (1 - fy) + bot * fy
    return (out / 127.5) - 1.0


def icm_srgb(path: str):
    """Pillow decode with embedded-ICC -> sRGB conversion (Chrome-like
    colour management). Returns None if the image carries no usable profile."""
    try:
        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
            if not icc:
                return None
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            dst = ImageCms.createProfile("sRGB")
            return ImageCms.profileToProfile(im.convert("RGB"), src, dst,
                                             outputMode="RGB")
    except Exception:
        return None


def variant_tensors(path: str) -> dict[str, np.ndarray]:
    """Candidate Python preprocessings -> 224x224x3 float32 in [-1,1]."""
    out: dict[str, np.ndarray] = {}
    out["tf"] = resize_tf(tf_decode(path))                  # production

    with Image.open(path) as im:
        pil_plain = im.convert("RGB")
    out["pil"] = (np.asarray(
        pil_plain.resize(tuple(wl.IMG_SIZE), Image.BILINEAR),
        dtype=np.float32) / 127.5) - 1.0

    icm = icm_srgb(path)
    if icm is None:                                          # no ICC profile
        out["pil_icm"] = out["pil"]
        out["pil_tf"] = resize_tf(np.asarray(pil_plain, dtype=np.uint8))
    else:
        out["pil_icm"] = (np.asarray(
            icm.resize(tuple(wl.IMG_SIZE), Image.BILINEAR),
            dtype=np.float32) / 127.5) - 1.0
        out["pil_tf"] = resize_tf(np.asarray(icm, dtype=np.uint8))

    x = tf.image.resize(tf.cast(tf_decode(path), tf.float32), wl.IMG_SIZE,
                        antialias=True).numpy()              # control
    out["tf_aa"] = (x / 127.5) - 1.0

    # controls via the explicit-convention resampler (TF 2.x has no
    # align_corners kwarg on tf.image.resize)
    out["tf_ac"] = resize_conv(tf_decode(path), "align")     # alignCorners=true
    out["tf_half"] = resize_conv(tf_decode(path), "half")    # TF's own conv

    # convention-matched to the browser (see resize_conv docstring)
    out["tfjs_conv"] = resize_conv(tf_decode(path), "tfjs")
    out["tfjs_conv_pil"] = resize_conv(
        np.asarray(pil_plain, dtype=np.uint8), "tfjs")
    return out


def verdict(bins, reject, threshold=THRESHOLD) -> str:
    b = [float(v) for v in bins]
    top = max(b)
    margin = top - sorted(b)[-2]
    if float(reject) >= threshold:
        return "unsupported"
    if top < MIN_TOP - 1e-9 or margin + 1e-9 < MIN_MARGIN:
        return "uncertain"
    return "supported"


def predict(model, x: np.ndarray) -> tuple[np.ndarray, float]:
    out = model(np.expand_dims(x.astype(np.float32), 0), training=False)
    return (np.asarray(out["bins"])[0],
            float(np.asarray(out["reject"]).ravel()[0]))


def cmd_dump(args) -> None:
    import shutil
    images = json.loads(Path(args.images).read_text())["images"]
    model = tf.keras.models.load_model(args.model)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    manifest, keras_records = [], []
    for path in images:
        name = Path(path).stem
        raw = tf_decode(path)                       # production decode
        (PROBE_DIR / f"{name}.py.rgb").write_bytes(raw.tobytes())
        shutil.copy2(path, PROBE_DIR / Path(path).name)   # ORIGINAL bytes
        tensor = variant_tensors(path)["tf"]        # production tensor
        bins, reject = predict(model, tensor)
        top = float(np.max(bins))
        margin = float(top - np.sort(bins)[-2])
        keras_records.append({
            "name": name, "path": path,
            "dims": [int(raw.shape[0]), int(raw.shape[1])],
            "bins": [float(v) for v in bins], "reject": reject,
            "top": top, "margin": margin,
            "verdict": verdict(bins, reject),
        })
        manifest.append({"name": name,
                         "image": f"/scratch/prep_probe/{Path(path).name}",
                         "pyrgb": f"/scratch/prep_probe/{name}.py.rgb",
                         "pyw": int(raw.shape[1]), "pyh": int(raw.shape[0]),
                         "source_path": path})
        print(f"dumped {name}: {raw.shape[1]}x{raw.shape[0]} top={top:.4f} "
              f"margin={margin:.4f} reject={reject:.4f} -> "
              f"{keras_records[-1]['verdict']}")
    (PROBE_DIR / "manifest.json").write_text(json.dumps(
        {"images": manifest,
         "generated_utc": datetime.now(timezone.utc).strftime(
             "%Y-%m-%d %H:%M:%S UTC")}, indent=1))
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "keras_path": keras_records}, indent=1) + "\n")
    print(f"wrote {PROBE_DIR / 'manifest.json'} and {args.out}")


def cmd_compare(args) -> None:
    keras = json.loads(Path(args.keras_json).read_text())
    probe = json.loads(Path(args.probe).read_text())
    model = tf.keras.models.load_model(args.model)
    by_name = {r["name"]: r for r in keras["keras_path"]}
    browser = {r["name"]: r for r in probe["results"]}

    per_image = []
    for name, br in browser.items():
        k = by_name[name]
        raw = np.frombuffer((PROBE_DIR / f"{name}.py.rgb").read_bytes(),
                            dtype=np.uint8).reshape(
            k["dims"][0], k["dims"][1], 3)
        b_tens = np.frombuffer(base64.b64decode(br["tensor_b64"]),
                               dtype=np.float32).reshape(224, 224, 3)
        row = {"name": name,
               "dims": k["dims"],
               "decode_diff": br["decode_diff"],   # full-image, computed in-probe
               "keras": {kk: k[kk] for kk in
                         ("top", "margin", "bins", "reject", "verdict")},
               "browser": {kk: br[kk] for kk in
                           ("top", "margin", "bins", "reject", "verdict")},
               "verdict_agrees": k["verdict"] == br["verdict"],
               "max_abs_bins_diff": float(np.max(
                   np.abs(np.asarray(k["bins"]) - np.asarray(br["bins"])))),
               "abs_reject_diff": float(abs(k["reject"] - br["reject"])),
               "variants": {}}
        for vname, vt in variant_tensors(k["path"]).items():
            v_bins, v_rej = predict(model, vt)
            row["variants"][vname] = {
                "max_input_diff_vs_browser":
                    float(np.max(np.abs(vt - b_tens))),
                "mean_input_diff_vs_browser":
                    float(np.mean(np.abs(vt - b_tens))),
                "max_bins_diff_vs_browser":
                    float(np.max(np.abs(v_bins - np.asarray(br["bins"])))),
                "abs_reject_diff_vs_browser":
                    float(abs(v_rej - br["reject"])),
                "verdict_matches_browser":
                    verdict(v_bins, v_rej) == br["verdict"],
                "_bins": [float(v) for v in v_bins],
                "_reject": v_rej,
            }
        per_image.append(row)

    summary = {}
    for v in sorted(per_image[0]["variants"]):
        summary[v] = {
            "mean_max_input_diff": float(np.mean(
                [r["variants"][v]["max_input_diff_vs_browser"]
                 for r in per_image])),
            "worst_max_input_diff": float(np.max(
                [r["variants"][v]["max_input_diff_vs_browser"]
                 for r in per_image])),
            "mean_max_bins_diff": float(np.mean(
                [r["variants"][v]["max_bins_diff_vs_browser"]
                 for r in per_image])),
            "verdict_matches": f"{sum(1 for r in per_image if r['variants'][v]['verdict_matches_browser'])}/{len(per_image)}",
        }
    for r in per_image:            # drop internal fields from the artifact
        for v in r["variants"].values():
            v.pop("_bins", None)
            v.pop("_reject", None)

    out = {"generated_utc": datetime.now(timezone.utc)
           .strftime("%Y-%m-%d %H:%M:%S UTC"),
           "probe": str(args.probe), "threshold": THRESHOLD,
           "images": per_image, "variant_summary": summary,
           "keras_vs_browser_verdicts_agree":
               f"{sum(1 for r in per_image if r['verdict_agrees'])}/{len(per_image)}",
           "n_images": len(per_image)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"variant_summary": summary,
                      "keras_vs_browser_verdicts_agree":
                          out["keras_vs_browser_verdicts_agree"]}, indent=1))
    print(f"wrote {OUT_JSON}")


def tensor_for(path: str, mode: str) -> np.ndarray:
    """Single-variant fast path (used by the LOOP-5 gate over thousands of
    images): 'tf' = shipped Python path; 'browser' = browser-equivalent
    (Pillow decode + tfjs y*in/out bilinear)."""
    if mode == "tf":
        return resize_tf(tf_decode(path))
    with Image.open(path) as im:
        pil_rgb = im.convert("RGB")
    return resize_conv(np.asarray(pil_rgb, dtype=np.uint8), "tfjs")


def cmd_gate(args) -> None:
    """LOOP 5 gate: does adopting browser-equivalent preprocessing change
    normal predictions? Run BOTH paths over the full shipped test sets."""
    import train_rejection as tr
    from sklearn.metrics import accuracy_score, roc_auc_score, \
        precision_recall_fscore_support

    data = json.loads(tr.EVAL_SETS_JSON.read_text())
    model = tf.keras.models.load_model(args.model)
    supported = data["supported_test"]            # (path, label) x 1147
    ood = {s: ps for s, ps in data["test_unsup"].items()}

    def run(paths, mode):
        out_b, out_r = [], []
        for i in range(0, len(paths), args.batch):
            chunk = paths[i:i + args.batch]
            x = np.stack([tensor_for(p, mode) for p in chunk])
            out = model(x, training=False)
            out_b.append(np.asarray(out["bins"]))
            out_r.append(np.asarray(out["reject"]).ravel())
        return np.concatenate(out_b), np.concatenate(out_r)

    paths = [r[0] for r in supported] + [p for ps in ood.values() for p in ps]
    labels = np.array([r[1] for r in supported] + [-1] * (len(paths) - len(supported)))
    n_sup = len(supported)

    res = {}
    for mode in ("tf", "browser"):
        bins, rej = run(paths, mode)
        verdicts = [verdict(b, r) for b, r in zip(bins, rej)]
        sup_b, sup_r = bins[:n_sup], rej[:n_sup]
        y_pred = sup_b.argmax(axis=1)
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels[:n_sup], y_pred, labels=[0, 1, 2, 3], zero_division=0)
        ood_scores = rej[n_sup:]
        y_tr = np.concatenate([np.zeros(n_sup, np.int32),
                               np.ones(len(ood_scores), np.int32)])
        res[mode] = {
            "accuracy": float(accuracy_score(labels[:n_sup], y_pred)),
            "macro_f1": float(np.mean(f1)),
            "per_bin_f1": {wl.BINS[i]: float(f1[i]) for i in range(4)},
            "frr_at_0.0702": float(np.mean(sup_r >= THRESHOLD)),
            "ood_detect_at_0.0702": float(np.mean(ood_scores >= THRESHOLD)),
            "auroc": float(roc_auc_score(y_tr, rej)),
            "reject_mean_supported": float(np.mean(sup_r)),
            "n_correct_supported": int((y_pred == labels[:n_sup]).sum()),
            "pred_indices_supported": [int(v) for v in y_pred],
            "reject_supported": [float(v) for v in sup_r],
            "verdicts": verdicts,
        }

    # agreement between the two paths
    tf_v, br_v = res["tf"]["verdicts"], res["browser"]["verdicts"]
    flips = [{"path": paths[i], "tf": tf_v[i], "browser": br_v[i]}
             for i in range(len(paths)) if tf_v[i] != br_v[i]]
    sup_flips = [f for f in flips if f["path"] in
                 {r[0] for r in supported}]
    tf_pred = res["tf"]["pred_indices_supported"]
    br_pred = res["browser"]["pred_indices_supported"]
    bin_flips = [{"path": supported[i][0],
                  "tf": wl.BINS[tf_pred[i]], "browser": wl.BINS[br_pred[i]]}
                 for i in range(n_sup) if tf_pred[i] != br_pred[i]]
    n_correct_tf = res["tf"]["n_correct_supported"]
    n_correct_br = res["browser"]["n_correct_supported"]

    # gate criteria (registered here, evaluated immediately)
    acc_delta = abs(res["tf"]["accuracy"] - res["browser"]["accuracy"])
    f1_delta = abs(res["tf"]["macro_f1"] - res["browser"]["macro_f1"])
    frr_delta = abs(res["tf"]["frr_at_0.0702"] - res["browser"]["frr_at_0.0702"])
    ood_delta = abs(res["tf"]["ood_detect_at_0.0702"]
                    - res["browser"]["ood_detect_at_0.0702"])
    flip_rate = len(flips) / len(paths)
    gate = {
        "criteria": {
            "verdict_flip_rate_max": 0.01,
            "accuracy_delta_max": 0.001,      # 0.1pt
            "macro_f1_delta_max": 0.001,      # 0.1pt
            "frr_delta_max": 0.002,           # 0.2pt
            "ood_delta_max": 0.002},          # 0.2pt
        "measured": {
            "verdict_flip_rate": flip_rate,
            "accuracy_delta": acc_delta,
            "macro_f1_delta": f1_delta,
            "frr_delta": frr_delta,
            "ood_delta": ood_delta},
        "pass": bool(flip_rate <= 0.01 and acc_delta <= 0.001
                     and f1_delta <= 0.001 and frr_delta <= 0.002
                     and ood_delta <= 0.002),
    }
    out = {"generated_utc": datetime.now(timezone.utc)
           .strftime("%Y-%m-%d %H:%M:%S UTC"),
           "model": args.model, "threshold": THRESHOLD,
           "n_images": len(paths), "n_supported": n_sup,
           "tf_path": {k: v for k, v in res["tf"].items() if k != "verdicts"},
           "browser_equiv_path": {k: v for k, v in res["browser"].items()
                                  if k != "verdicts"},
           "verdict_flips_total": len(flips),
           "verdict_flips_supported": len(sup_flips),
           "flips_sample": flips[:40],
           "bin_prediction_flips": len(bin_flips),
           "bin_flip_sample": bin_flips[:40],
           "n_correct_supported": {"tf": n_correct_tf,
                                   "browser": n_correct_br},
           "gate": gate}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"tf": out["tf_path"],
                      "browser_equiv": out["browser_equiv_path"],
                      "flips": f"{len(flips)}/{len(paths)} "
                               f"(supported {len(sup_flips)})",
                      "gate": gate}, indent=1))
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump-inputs")
    d.add_argument("--images", required=True, help="json {images:[paths]}")
    d.add_argument("--model",
                   default="models/checkpoints/wastelens_rej_shipped_best.keras")
    d.add_argument("--out", default="scratch/prep_probe/keras_inputs.json")
    c = sub.add_parser("compare")
    c.add_argument("--probe", default="scratch/prep_probe/browser_probe.json")
    c.add_argument("--keras-json", default="scratch/prep_probe/keras_inputs.json")
    c.add_argument("--model",
                   default="models/checkpoints/wastelens_rej_shipped_best.keras")
    g = sub.add_parser("gate")
    g.add_argument("--model",
                   default="models/checkpoints/wastelens_rej_shipped_best.keras")
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--out",
                   default="docs/rejection_experiment/preprocess_gate.json")
    args = ap.parse_args()
    ({"dump-inputs": cmd_dump, "compare": cmd_compare, "gate": cmd_gate}
     [args.cmd])(args)


if __name__ == "__main__":
    main()
