# src/train_adaptation.py
#
# WasteLens Iteration 10 - LOOP 8/9: controlled fresh-domain adaptation.
#
# Implements docs/rejection_experiment/iteration10_adaptation_protocol.md
# §2 exactly: warm start from the shipped Variant C checkpoint, backbone
# frozen, fc_256 + bins + reject trainable, replayed 3:1 original:adapt
# mix, Adam(1e-3), batch 32, 6 epochs, deterministic, fixed-budget final
# epoch (= epoch 6) is the candidate. Checkpoints go to scratch paths
# (models/checkpoints/wastelens_rej_adapt10s{SEED}_*.keras) - NEVER to
# web/model/ or the shipped alias.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

# LOOP: determinism switch active before TF import (Iteration-8 pattern).
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow as tf

import train as wl
import train_rejection as tr

OP_DETERMINISM_ERROR = None
try:
    tf.config.experimental.enable_op_determinism()
except Exception as exc:  # pragma: no cover - recorded in config
    OP_DETERMINISM_ERROR = f"{type(exc).__name__}: {exc}"

SHIPPED_CHECKPOINT = Path("models/checkpoints/wastelens_rej_shipped_best.keras")
ADAPT_MANIFEST = Path("docs/rejection_experiment/adaptation_set_manifest.json")
ADAPT_DIR = Path("scratch/adaptation_set")
FRESH_MANIFEST = Path("docs/rejection_experiment/fresh_set_manifest.json")
CKPT_DIR = Path("models/checkpoints")
OUT_DIR = Path("docs/rejection_experiment")

EPOCHS = 6
DEFAULT_BINS_WEIGHT = 2.0
DEFAULT_REJECT_WEIGHT = 1.0
ORIG_ADAPT_RATIO = 3.0  # original : adaptation rows (≈3:1 per protocol §2)
ADAPT_VAL_FRACTION = 0.20
BACKBONE_NAME = "mobilenetv2_1.00_224"
BIN_NAMES = list(wl.BINS)
BIN_TO_IDX = {b: i for i, b in enumerate(BIN_NAMES)}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def adapt_rows() -> dict[str, list[tuple[Path, str, int]]]:
    """Adaptation pool: (path, role, bin_idx); bin_idx=-1 if unsupported."""
    m = json.loads(ADAPT_MANIFEST.read_text(encoding="utf-8"))
    by_role: dict[str, list] = {"supported": [], "ood": [], "multi": []}
    for e in m["entries"]:
        p = ADAPT_DIR / e["local_name"]
        if e["group"] == "adapt-supported":
            by_role["supported"].append((p, "supported",
                                         BIN_TO_IDX[e["label"]]))
        elif e["group"] == "adapt-ood":
            by_role["ood"].append((p, "ood", -1))
        else:
            by_role["multi"].append((p, "multi", -1))
    return by_role


def split_adapt(pool: dict, seed: int):
    """Deterministic 80/20 split of the adaptation pool (seed 42)."""
    rng = np.random.RandomState(seed)
    out = {}
    for role, rows in pool.items():
        rows = sorted(rows, key=lambda r: str(r[0]))
        order = rng.permutation(len(rows))
        n_va = max(1, int(round(len(rows) * ADAPT_VAL_FRACTION)))
        va_idx = set(order[:n_va].tolist())
        out[role] = {"train": [r for i, r in enumerate(rows)
                               if i not in va_idx],
                     "val": [r for i, r in enumerate(rows)
                             if i in va_idx]}
    return out


def upsample(rows: list, target: int, seed: int) -> list:
    """Tile rows deterministically to reach target length.

    Each repeat is drawn in a fresh permutation from a seed-fixed RNG, so the
    same inputs always produce the same output list (and the local file order
    of a repeat is not simply the list order again).
    """
    rows = list(rows)
    if not rows or len(rows) >= target:
        return rows[:target]
    rng = np.random.RandomState(seed)
    idx = np.arange(len(rows))
    out = list(rows)
    while len(out) < target:
        out.extend(rows[int(i)] for i in rng.permutation(idx))
    return out[:target]

# --- tf.data + model --------------------------------------------------------

def build_mixed_arrays(orig: dict, adapt: dict):
    """Replayed 3:1 original:adapt mix -> training arrays + weights.

    Deterministic and seed-independent: both registered seeds train on the
    byte-identical mix (protocol §2 "same files"), so any difference between
    seed 42 and seed 43 is model init + shuffling only.
    """
    orig_s = sorted(orig["splits"]["train"])
    orig_u_rows = []
    for src in ("clothes", "shoes", "collage", "nonwaste"):
        orig_u_rows.extend(orig["unsup"][src]["train"])
    orig_u_rows = sorted(orig_u_rows, key=lambda r: str(r[0]))
    n_os, n_ou = len(orig_s), len(orig_u_rows)

    ad_s = adapt["supported"]["train"]
    ad_u = (adapt["ood"]["train"] + adapt["multi"]["train"])
    # ≈3:1 original:adapt per head-role (protocol §2).
    tgt_as = max(len(ad_s), int(round(n_os / ORIG_ADAPT_RATIO)))
    tgt_au = max(len(ad_u), int(round(n_ou / ORIG_ADAPT_RATIO)))
    ad_s_up = upsample(sorted(ad_s, key=lambda r: str(r[0])), tgt_as,
                       wl.SEED + 11)
    ad_u_up = upsample(sorted(ad_u, key=lambda r: str(r[0])), tgt_au,
                       wl.SEED + 12)

    paths_s = ([str(p) for p, _ in orig_s]
               + [str(p) for p, _, _ in ad_s_up])
    y_s = ([int(l) for _, l in orig_s]
           + [int(b) for _, _, b in ad_s_up])
    paths_u = ([str(p) for p, _ in orig_u_rows]
               + [str(p) for p, _, _ in ad_u_up])
    n_s, n_u = len(paths_s), len(paths_u)
    n = n_s + n_u
    # bins-head inverse-frequency weights on the MIXED supported pool.
    counts = np.bincount(np.asarray(y_s), minlength=len(BIN_NAMES))
    total = int(counts.sum())
    cw = {i: float(total / (len(BIN_NAMES) * counts[i])) for i in range(4)}
    w_s = n / (2.0 * max(n_s, 1))
    w_u = n / (2.0 * max(n_u, 1))
    paths = np.array(paths_s + paths_u)
    y_bins = np.array(y_s + [0] * n_u, np.int32)
    y_rej = np.array([0] * n_s + [1] * n_u, np.float32)
    sw_bins = np.array([cw[l] for l in y_s] + [0.0] * n_u, np.float32)
    sw_rej = np.array([w_s] * n_s + [w_u] * n_u, np.float32)
    info = {"n_orig_supported": n_os, "n_orig_unsupported": n_ou,
            "n_adapt_supported_raw": len(adapt["supported"]["train"]),
            "n_adapt_unsupported_raw": len(adapt["ood"]["train"])
            + len(adapt["multi"]["train"]),
            "n_adapt_supported_used": len(ad_s_up),
            "n_adapt_unsupported_used": len(ad_u_up),
            "n_supported": n_s, "n_unsupported": n_u,
            "bins_class_weights": cw,
            "rej_weights": {"supported": w_s, "unsupported": w_u}}
    return paths, y_bins, y_rej, sw_bins, sw_rej, info
def pack_rows(rows_s: list, rows_u: list, cw: dict) -> tuple:
    """Package (supported-triples, unsupported-triples) into training arrays.

    Same contract as train_rejection.assemble: unsupported rows carry bins
    weight 0; rejection weights are inverse-frequency on the packed pool.
    `rows_*` entries are (path, role, bin_idx) triples from `adapt_rows()`.
    """
    n_s, n_u = len(rows_s), len(rows_u)
    n = n_s + n_u
    w_s = n / (2.0 * max(n_s, 1))
    w_u = n / (2.0 * max(n_u, 1))
    paths = np.array([str(r[0]) for r in rows_s]
                     + [str(r[0]) for r in rows_u])
    y_bins = np.array([int(r[2]) for r in rows_s] + [0] * n_u, np.int32)
    y_rej = np.array([0] * n_s + [1] * n_u, np.float32)
    sw_bins = np.array([cw[int(r[2])] for r in rows_s] + [0.0] * n_u,
                       np.float32)
    sw_rej = np.array([w_s] * n_s + [w_u] * n_u, np.float32)
    return paths, y_bins, y_rej, sw_bins, sw_rej, n_s, n_u


def make_ds(paths, y_bins, y_rej, sw_bins, sw_rej, training: bool,
            seed: int) -> tf.data.Dataset:
    """Byte-identical to train_rejection.make_weighted_ds except that the
    shuffle stream is the RUN seed, so seed 43 sees the same files in a new
    order (protocol §2)."""
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, y_bins, y_rej, sw_bins, sw_rej))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(p, yb, yr, sb, sr):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, wl.IMG_SIZE)
        img = wl.PREPROCESS_INPUT(img)
        return img, {"bins": yb, "reject": yr}, {"bins": sb, "reject": sr}

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(wl.BATCH_SIZE)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def img_only_ds(paths, batch_size: int = 8) -> tf.data.Dataset:
    """Inputs-only dataset for the warm-start proof (no labels, no weights)."""
    ds = tf.data.Dataset.from_tensor_slices(list(paths))

    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, wl.IMG_SIZE)
        return wl.PREPROCESS_INPUT(img)

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size)


def build_adapt_model(checkpoint: Path = SHIPPED_CHECKPOINT,
                      bins_weight: float = DEFAULT_BINS_WEIGHT,
                      reject_weight: float = DEFAULT_REJECT_WEIGHT):
    """Warm start from the shipped dual-head checkpoint, architecture frozen.

    The checkpoint already carries the Iteration-8 dual-head contract
    (frozen MobileNetV2 + trainable fc_256 / predictions / reject); loading it
    keeps the trained reject head, which rebuilding from the base model would
    silently reset. The assertions below make the contract explicit.
    """
    model = tf.keras.models.load_model(checkpoint)
    trainable = sorted(w.path for w in model.trainable_weights)
    expected = sorted(["fc_256/kernel", "fc_256/bias", "predictions/kernel",
                       "predictions/bias", "reject/kernel", "reject/bias"])
    if trainable != expected:
        raise RuntimeError(f"unexpected trainable tensors: {trainable}")
    backbone = model.get_layer(BACKBONE_NAME)
    for w in backbone.weights:
        if w.trainable:
            raise RuntimeError(f"backbone weight is trainable: {w.path}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"bins": "sparse_categorical_crossentropy",
              "reject": "binary_crossentropy"},
        loss_weights={"bins": float(bins_weight),
                      "reject": float(reject_weight)},
        metrics={"bins": ["sparse_categorical_accuracy"],
                 "reject": ["binary_accuracy"]},
    )
    return model


def warm_start_proof(model, data, n_images: int = 16,
                     warm_ckpt: Path = SHIPPED_CHECKPOINT) -> dict:
    """Prove the candidate starts EXACTLY at the shipped model.

    Both heads (bins + reject) are compared on real training images *before*
    any weight update, so every later change is attributable to training.
    """
    base = tf.keras.models.load_model(warm_ckpt)
    rows = sorted(data["splits"]["train"])
    sample = [str(p) for p, _ in rows[:n_images]]
    base_out = base.predict(img_only_ds(sample), verbose=0)
    cand_out = model.predict(img_only_ds(sample), verbose=0)
    diffs = {h: float(np.max(np.abs(np.asarray(base_out[h])
                                    - np.asarray(cand_out[h]))))
             for h in ("bins", "reject")}
    exact = {h: bool(np.array_equal(np.asarray(base_out[h]),
                                    np.asarray(cand_out[h]))) for h in diffs}
    if any(d > 1e-6 for d in diffs.values()):
        raise RuntimeError(f"warm-start proof FAILED: {diffs}")
    return {"method": f"{n_images}-image {sorted(diffs)} output comparison vs "
                      f"{warm_ckpt} at initialization (before training)",
            "max_abs_diff": diffs, "bit_exact": exact}


class AdaptValLogger(tf.keras.callbacks.Callback):
    """Monitors the adaptation-val pool once per epoch.

    Monitoring ONLY: the protocol fixes the candidate to the final epoch, so
    these numbers can never change which checkpoint is proposed. Records are
    kept in `self.records` (one dict per finished epoch) and merged into the
    history CSV by the caller.
    """

    def __init__(self, ds, steps: int):
        super().__init__()
        self.ds = ds
        self.steps = steps
        self.records: list[dict] = []

    def on_epoch_end(self, epoch, logs=None):
        res = self.model.evaluate(self.ds, steps=self.steps, verbose=0,
                                  return_dict=True)
        self.records.append({f"adapt_val_{k}": float(v)
                             for k, v in res.items()})
        head = "  ".join(f"{k}={v:.4f}" for k, v in self.records[-1].items())
        print(f"      [adapt-val] {head}")
# --- Artifacts --------------------------------------------------------------

def write_artifacts(args, tag: str, seed: int, epochs: int, info: dict,
                    proof: dict, hist, adapt_records: list[dict],
                    wall_sec: float, ckpt_pattern: str) -> None:
    """Per-seed history CSV + training config; copies the final epoch to the
    `_final`/`_best` aliases (copied, never selected)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(hist.history.keys())
    adapt_keys: list[str] = []
    for rec in adapt_records:
        for k in rec:
            if k not in adapt_keys:
                adapt_keys.append(k)
    hist_csv = OUT_DIR / f"adapt10s{seed}{tag}_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys + adapt_keys)
        for i in range(len(hist.history[keys[0]])):
            rec = adapt_records[i] if i < len(adapt_records) else {}
            w.writerow([float(hist.history[k][i]) for k in keys]
                       + [float(rec.get(k, np.nan)) for k in adapt_keys])

    final_ckpt = Path(ckpt_pattern.format(epoch=epochs))
    cand_ckpt = CKPT_DIR / f"wastelens_rej_adapt10s{seed}{tag}_final.keras"
    alias_ckpt = CKPT_DIR / f"wastelens_rej_adapt10s{seed}{tag}_best.keras"
    shutil.copyfile(final_ckpt, cand_ckpt)
    shutil.copyfile(final_ckpt, alias_ckpt)
    hashes = {p.name: md5_file(p) for p in (final_ckpt, cand_ckpt,
                                            alias_ckpt)}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"final-epoch copies differ: {hashes}")

    last_acc_key = [k for k in keys if k.startswith("val_bins")
                    and "accuracy" in k][0]
    cfg = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": "docs/rejection_experiment/"
                    "iteration10_adaptation_protocol.md",
        "rule": (f"warm start {args.model} (backbone frozen; fc_256 + "
                 f"predictions + reject trainable), replayed 3:1 "
                 f"original:adapt mix, Adam(1e-3), batch {wl.BATCH_SIZE}, "
                 f"{epochs} epochs, candidate = FINAL epoch (no selection)"),
        "seed": seed, "epochs": epochs, "candidate_epoch": epochs,
        "smoke": bool(args.smoke),
        "optimizer": "Adam", "learning_rate": 1e-3,
        "loss_weights": {"bins": float(args.bins_weight),
                         "reject": float(args.reject_weight)},
        "batch_size": int(wl.BATCH_SIZE), "img_size": list(wl.IMG_SIZE),
        "dataset_seed": int(wl.SEED),
        "determinism": {
            "TF_DETERMINISTIC_OPS_env": os.environ.get("TF_DETERMINISTIC_OPS"),
            "enable_op_determinism_error": OP_DETERMINISM_ERROR,
            "keras_set_random_seed": seed,
        },
        "warm_start": {"checkpoint": str(args.model),
                       "proof": proof},
        "mix": info,
        "adapt_manifest": {
            "path": str(ADAPT_MANIFEST),
            "md5": md5_file(ADAPT_MANIFEST),
        },
        "fresh_benchmark_reference": {"path": str(FRESH_MANIFEST),
                                      "note": "read for provenance only; no "
                                              "fresh file is ever opened by "
                                              "this script"},
        "adapt_val_records": adapt_records,
        "final_epoch": {
            "val_loss": float(hist.history["loss"][-1]) if "loss" in
            hist.history else None,
            "val_bins_acc": float(hist.history[last_acc_key][-1]),
        },
        "wall_time_sec": round(wall_sec, 1),
        "checkpoints": {"epoch_pattern": ckpt_pattern,
                        "final_epoch_file": str(final_ckpt),
                        "candidate_copy": str(cand_ckpt),
                        "best_alias_copy": str(alias_ckpt),
                        "md5": hashes},
        "history_csv": str(hist_csv),
        "promotion": "scratch artifact only - promotion is a separate, "
                     "post-gate decision (protocol §4)",
    }
    cfg_path = OUT_DIR / f"adapt10s{seed}{tag}_training_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"      candidate = {final_ckpt.name} "
          f"(md5 {hashes[final_ckpt.name]})")
    print(f"wrote {hist_csv}")
    print(f"wrote {cfg_path}")


# --- Run --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Iteration-10 fresh-domain adaptation fine-tune "
                    "(registered protocol; the final epoch is the candidate).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--bins-weight", type=float, default=DEFAULT_BINS_WEIGHT)
    ap.add_argument("--reject-weight", type=float,
                    default=DEFAULT_REJECT_WEIGHT)
    ap.add_argument("--model", type=Path, default=SHIPPED_CHECKPOINT,
                    help="warm-start checkpoint (registered: shipped varc)")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline check only: 1 epoch, truncated pools")
    args = ap.parse_args()
    if not args.smoke:
        if args.epochs != EPOCHS:
            raise SystemExit(
                f"protocol fixes the budget at {EPOCHS} epochs; "
                f"got {args.epochs}")
        if Path(args.model) != SHIPPED_CHECKPOINT:
            raise SystemExit(f"protocol fixes the warm start at "
                             f"{SHIPPED_CHECKPOINT}; got {args.model}")
    tag = "_smoke" if args.smoke else ""
    epochs = 1 if args.smoke else args.epochs
    seed = int(args.seed)

    print("REGISTERED RULE: docs/rejection_experiment/"
          "iteration10_adaptation_protocol.md §2 - warm start "
          f"{args.model}, backbone frozen, Adam(1e-3), batch {wl.BATCH_SIZE}, "
          f"{epochs} epochs, candidate = FINAL epoch (no selection).")
    data = tr.build_datasets()
    pool = adapt_rows()
    split = split_adapt(pool, wl.SEED)
    print(f"adaptation pool: supported {len(pool['supported'])}, ood "
          f"{len(pool['ood'])}, multi {len(pool['multi'])} -> deterministic "
          f"80/20 split (seed {wl.SEED})")
    paths, y_bins, y_rej, sw_bins, sw_rej, info = build_mixed_arrays(
        data, split)
    cw = info["bins_class_weights"]
    av_pack = pack_rows(sorted(split["supported"]["val"],
                               key=lambda r: str(r[0])),
                        sorted(split["ood"]["val"] + split["multi"]["val"],
                               key=lambda r: str(r[0])), cw)
    va_pack = tr.assemble(data, "val", smoke=False)
    if args.smoke:
        paths, y_bins = paths[:160], y_bins[:160]
        y_rej, sw_bins, sw_rej = y_rej[:160], sw_bins[:160], sw_rej[:160]
    for label, pack in (("original-val", va_pack), ("adapt-val", av_pack)):
        print(f"      {label}: {pack[5]} supported + {pack[6]} unsupported"
              f" = {len(pack[0])} images")
    print(f"      train mix: {info['n_supported']} supported + "
          f"{info['n_unsupported']} unsupported = {len(paths)} images "
          f"({math.ceil(len(paths) / wl.BATCH_SIZE)} steps/epoch)")
    print(f"      mix detail: {json.dumps(info)}")
    tr_ds = make_ds(paths, y_bins, y_rej, sw_bins, sw_rej, True, seed)
    va_ds = make_ds(*va_pack[:5], False, seed)
    av_ds = make_ds(*av_pack[:5], False, seed)

    tf.keras.utils.set_random_seed(seed)
    model = build_adapt_model(Path(args.model), args.bins_weight,
                              args.reject_weight)
    proof = warm_start_proof(model, data, warm_ckpt=Path(args.model))
    print(f"      warm-start proof: bit_exact={proof['bit_exact']} "
          f"max_abs_diff={proof['max_abs_diff']}")
    tf.keras.utils.set_random_seed(seed)   # Iteration-8 reseed-before-fit

    ckpt_pattern = str(CKPT_DIR /
                       f"wastelens_rej_adapt10s{seed}{tag}_{{epoch:02d}}.keras")
    ckpt_every = tf.keras.callbacks.ModelCheckpoint(
        ckpt_pattern, monitor="val_loss", save_best_only=False, verbose=0)
    adapt_logger = AdaptValLogger(av_ds,
                                  steps=math.ceil(len(av_pack[0])
                                                  / wl.BATCH_SIZE))
    callbacks = [ckpt_every, adapt_logger]
    if hasattr(tr, "maybe_tensorboard"):
        callbacks += list(tr.maybe_tensorboard(
            OUT_DIR / f"adapt10s{seed}{tag}_tensorboard"))
    t0 = time.time()
    hist = model.fit(tr_ds, steps_per_epoch=math.ceil(len(paths) / wl.BATCH_SIZE),
                     validation_data=va_ds,
                     validation_steps=math.ceil(len(va_pack[0]) / wl.BATCH_SIZE),
                     epochs=epochs, callbacks=callbacks, verbose=1)
    wall = time.time() - t0
    print(f"      trained {epochs} epochs in {wall / 60:.2f} min")
    write_artifacts(args, tag, seed, epochs, info, proof, hist,
                    adapt_logger.records, wall, ckpt_pattern)



if __name__ == "__main__":
    main()

