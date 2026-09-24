# src/train_rejection_variantc.py
#
# WasteLens Iteration 6 - VARIANT C: partially unfrozen rejection head.
#
# WHY THIS EXISTS
#   Iteration 4 measured two extremes:
#     Variant A (src/train_rejection.py)   - backbone frozen, WHOLE head trainable
#         at Adam(1e-3), loss weights 1.0/1.0. Rejection was near-perfect
#         (AUROC 0.9997, collage 97.8% at its shipped threshold) but the shared
#         fc_256 representation moved and four-bin accuracy regressed 1.06pt
#         (0.9716 vs the 0.9822 baseline) - outside the pre-registered 0.5pt
#         ACC_TOLERANCE.
#     Variant B (src/train_rejection_frozen.py) - EVERYTHING frozen, reject head
#         only. Bins output bit-identical (0.9822) but collage rejection weak
#         (21.1% at its shipped threshold).
#
#   Variant C is the middle point prescribed by the Iteration-6 brief:
#     - MobileNetV2 backbone stays FROZEN (identical to A and B).
#     - fc_256 (shared representation), predictions (4-bin head) and the
#       reject head are TRAINABLE.
#     - The ONLY changed hyperparameter vs Variant A is the loss weight:
#       bins 2.0 / reject 1.0 (vs A's 1.0/1.0) so rejection training cannot
#       destroy the classifier. Everything else is inherited unchanged:
#       labels, input size, preprocessing, four-bin ordering, optimizer
#       Adam(1e-3), batch size 32, epochs 10, seed, datasets, splits.
#
# DATA: identical to Variants A/B (train_rejection.build_datasets()).
#   Supported = train.py's UNCHANGED seed-42 split; unsupported = md5-deduped
#   clothes+shoes, CIFAR-10 non-waste, collages from DISJOINT train pools.
#   Test splits and synthetic probes are never trained on.
#
# PRE-TRAINING PROOF (this variant's analogue of Variant B's freeze proof):
#   1. every backbone weight is bit-identical to the base checkpoint, and
#   2. at initialization (before any training) the dual model's bins output is
#      bit-identical to the base checkpoint (fc_256 starts at shipped weights),
#   so any classification change after training is attributable to training,
#   not to construction.
#
# Run from the repo root:
#   python src/train_rejection_variantc.py [--smoke] [--epochs N]
#                                          [--bins-weight 2.0]
# Outputs:
#   models/checkpoints/wastelens_rej_varc_best.keras  (best val_loss epoch)
#   models/checkpoints/wastelens_rej_varc_ep{epoch:02d}.keras
#   docs/rejection_experiment/variantc_history.csv     (per-epoch metrics)
#   docs/rejection_experiment/variantc_training_config.json

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Iteration 8 - LOOP 3 (reproducibility): must be set BEFORE TensorFlow is
# imported so the deterministic-kernel switch is active for the whole process.
# setdefault() keeps any explicit override from the caller's environment.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow as tf

import train as wl
import train_rejection as tr

# Iteration 8: op-level determinism, enabled once at import (TF requires this
# before ops are created). Recorded in the training config; kept best-effort so
# a platform without a deterministic kernel fails loudly at fit time instead of
# silently degrading.
OP_DETERMINISM_ERROR = None
try:
    tf.config.experimental.enable_op_determinism()
except Exception as exc:  # pragma: no cover - defensive, recorded in config
    OP_DETERMINISM_ERROR = f"{type(exc).__name__}: {exc}"

BASE_CHECKPOINT = Path("models/checkpoints/wastelens_ep09.keras")
# Iteration 6's output path (kept as documentation of that run's artifacts;
# Iteration 7+ pass --out-tag / --out and derive names inside run_training()).
OUT_CKPT = Path("models/checkpoints/wastelens_rej_varc_best.keras")
OUT_DIR = Path("docs/rejection_experiment")
# Iteration 7: per-run artifact names are derived from --out-tag inside
# run_training() (empty tag reproduces the Iteration-6 names above).

BACKBONE_NAME = "mobilenetv2_1.00_224"
EPOCHS = tr.EPOCHS          # 10 - unchanged
SEED = tr.SEED              # 42 - unchanged
MONITOR = "val_loss"
DEFAULT_BINS_WEIGHT = 2.0   # THE changed knob (Variant A used 1.0)


def build_varc_dual(base_checkpoint: Path = BASE_CHECKPOINT,
                    bins_weight: float = DEFAULT_BINS_WEIGHT):
    """Backbone-frozen dual head with fc_256 + predictions + reject trainable.

    Inherits the baseline's saved trainable flags (backbone frozen when the
    checkpoint was trained), then re-asserts them explicitly and compiles with
    the higher bins loss weight.
    """
    base = tf.keras.models.load_model(base_checkpoint)
    backbone = base.get_layer(BACKBONE_NAME)
    backbone.trainable = False          # re-assert, do not touch anything else
    for w in backbone.weights:
        assert not w.trainable, f"backbone weight must stay frozen: {w.path}"
    rej = tf.keras.layers.Dense(1, activation="sigmoid", name="reject")(
        base.get_layer("dropout_2").output
    )
    model = tf.keras.Model(
        base.input,
        {"bins": base.output, "reject": rej},
        name="wastelens_reject_varc",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "bins": "sparse_categorical_crossentropy",
            "reject": "binary_crossentropy",
        },
        loss_weights={
            "bins": float(bins_weight),
            "reject": 1.0,
        },
        metrics={
            "bins": ["sparse_categorical_accuracy"],
            "reject": ["binary_accuracy"],
        },
    )
    return model


# --- Pre-training proof ----------------------------------------------------


def _img_dataset(paths, batch_size=8):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, wl.IMG_SIZE)
        return wl.PREPROCESS_INPUT(img)

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)


def unfreeze_proof(data, base_ckpt=BASE_CHECKPOINT):
    """Assert (a) backbone weights are bit-identical to the base checkpoint and
    (b) the UNTRAINED dual model's bins output is bit-identical to baseline."""
    base = tf.keras.models.load_model(base_ckpt)
    varc = build_varc_dual(base_ckpt)
    backbone = base.get_layer(BACKBONE_NAME)
    varc_backbone = varc.get_layer(BACKBONE_NAME)
    w_base = backbone.get_weights()
    w_varc = varc_backbone.get_weights()
    assert len(w_base) == len(w_varc), "backbone weight tensors mismatch"
    for a, b in zip(w_base, w_varc):
        if not np.array_equal(a, b):
            raise RuntimeError("unfreeze proof FAILED: backbone weights differ "
                               "from the base checkpoint at construction")
    rows = sorted(data["splits"]["train"])
    sample = [str(p) for p, _ in rows[:16]]
    base_probs = base.predict(_img_dataset(sample), verbose=0)
    varc_probs = varc.predict(_img_dataset(sample), verbose=0)["bins"]
    max_abs = float(np.max(np.abs(base_probs - varc_probs)))
    identical = bool(np.array_equal(base_probs, varc_probs))
    if not np.allclose(base_probs, varc_probs, atol=1e-5, rtol=1e-5):
        raise RuntimeError(
            f"unfreeze proof FAILED: initial bins output differs "
            f"(max_abs_diff={max_abs:.2e})")
    trainable = int(sum(np.prod(w.shape) for w in varc.trainable_weights))
    frozen = int(sum(np.prod(w.shape) for w in varc.non_trainable_weights))
    return {"method": "backbone weights bit-compared + 16-image bins-output "
                      "comparison at initialization (before training)",
            "backbone_weights_bit_identical": True,
            "initial_bins_bit_exact": identical,
            "initial_max_abs_diff": max_abs,
            "trainable_params": trainable,
            "frozen_params": frozen,
            "trainable_tensors": [w.path for w in varc.trainable_weights]}


# --- Training --------------------------------------------------------------


def run_training(data, smoke=False, epochs=None,
                 bins_weight: float = DEFAULT_BINS_WEIGHT,
                 out_tag: str = "", base_ckpt=None, out_ckpt_path=None,
                 train_seed=None, deterministic: bool = True):
    # Iteration 7: out_tag isolates a new run's artifacts (checkpoints,
    # history, config) from previous iterations' evidence. Empty tag keeps
    # the original Iteration-6 names.
    # Iteration 7 fix: --model/--out are now honored (they were accepted but
    # ignored in Iteration 6); defaults reproduce the original behaviour.
    # Iteration 8: train_seed controls ONLY model initialization, dropout and
    # training-time shuffling; the DATASET seed stays tr.SEED (42) so the
    # frozen dataset is byte-identical across seeds. deterministic=True turns
    # on op-level determinism (already enabled at import) and seeds EVERYTHING
    # BEFORE the model is constructed - the Iteration-6/7 code seeded after
    # construction, which left weight init unseeded and explains the observed
    # run-to-run spread.
    tag = out_tag
    seed = int(train_seed) if train_seed is not None else SEED
    base_ckpt = Path(base_ckpt) if base_ckpt else BASE_CHECKPOINT
    out_ckpt = (Path(out_ckpt_path) if out_ckpt_path else
                Path(f"models/checkpoints/wastelens_rej_varc{tag}_best.keras"))
    history_csv = OUT_DIR / f"variantc{tag}_history.csv"
    config_json = OUT_DIR / f"variantc{tag}_training_config.json"
    if deterministic:
        tf.keras.utils.set_random_seed(seed)   # python/random, numpy, tf global
    tr_pack = tr.assemble(data, "train", smoke)
    va_pack = tr.assemble(data, "val", smoke)
    tr_ds = tr.make_weighted_ds(*tr_pack[:5], training=True)
    va_ds = tr.make_weighted_ds(*va_pack[:5], training=False)
    n_tr = len(tr_pack[0])
    n_va = len(va_pack[0])
    print(f"      train: {tr_pack[5]} supported + {tr_pack[6]} unsupported = "
          f"{n_tr} images ({math.ceil(n_tr / wl.BATCH_SIZE)} steps/epoch)")
    print(f"      val:   {va_pack[5]} supported + {va_pack[6]} unsupported = "
          f"{n_va} images ({math.ceil(n_va / wl.BATCH_SIZE)} val steps)")
    n_epochs = int(epochs) if epochs else EPOCHS
    print(f"      train seed: {seed} (data seed stays {tr.SEED})  "
          f"deterministic: {deterministic}")
    model = build_varc_dual(base_ckpt, bins_weight=bins_weight)
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    frozen = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    print(f"      trainable params: {trainable:,}  (fc_256 + predictions + "
          f"reject; backbone frozen)")
    print(f"      frozen params:    {frozen:,}")
    proof = unfreeze_proof(data, base_ckpt=base_ckpt)
    print(f"      unfreeze proof: backbone_bit_identical="
          f"{proof['backbone_weights_bit_identical']}  "
          f"initial_bins_bit_exact={proof['initial_bins_bit_exact']}  "
          f"max_abs_diff={proof['initial_max_abs_diff']:.2e}")
    if deterministic:
        # Re-seed before fit so training-time dropout/shuffling start from the
        # same stream regardless of how much RNG the construction steps used.
        tf.keras.utils.set_random_seed(seed)

    ckpt_dir = Path("models/checkpoints")
    ckpt_every = tf.keras.callbacks.ModelCheckpoint(
        str(ckpt_dir / f"wastelens_rej_varc{tag}_{{epoch:02d}}.keras"),
        monitor=MONITOR, save_best_only=False, verbose=0)
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        str(out_ckpt), monitor=MONITOR, save_best_only=True, verbose=1)
    tb = tr.maybe_tensorboard(OUT_DIR / f"variantc{tag}_tensorboard") \
        if hasattr(tr, "maybe_tensorboard") else []
    t0 = time.time()
    hist = model.fit(tr_ds,
                     steps_per_epoch=math.ceil(n_tr / wl.BATCH_SIZE),
                     validation_data=va_ds,
                     validation_steps=math.ceil(n_va / wl.BATCH_SIZE),
                     epochs=n_epochs,
                     callbacks=[ckpt_every, ckpt_best] + list(tb),
                     verbose=1)
    dt = time.time() - t0
    keys = list(hist.history.keys())
    with history_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(hist.history[keys[0]])):
            w.writerow([float(hist.history[k][i]) for k in keys])
    best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)

    def _acc_key(prefix):
        for k in hist.history:
            if k.startswith(prefix) and "accuracy" in k:
                return k
        raise KeyError(prefix)

    val_bins_acc = float(hist.history[_acc_key("val_bins")][best_epoch - 1])
    val_rej_acc = float(hist.history[_acc_key("val_reject")][best_epoch - 1])
    cfg = {
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"),
        "variant": "C (backbone frozen; fc_256 + predictions + reject "
                   f"trainable; bins loss weight {float(bins_weight)})"
                   + (f" [run tag: {tag}]" if tag else ""),
        "base_checkpoint": str(base_ckpt),
        "out_checkpoint": str(out_ckpt),
        "epochs": n_epochs, "seed": seed, "monitor": MONITOR,
        # Iteration 8 reproducibility block (LOOP 3).
        "data_seed": int(tr.SEED),
        "train_seed": int(seed),
        "seed_note": "train_seed controls init/dropout/shuffle only; the "
                     "dataset seed stays 42 so all runs see the same frozen "
                     "splits, collages, probes and class weights",
        "deterministic": bool(deterministic),
        "op_determinism_requested": True,
        "tf_deterministic_ops_env": os.environ.get("TF_DETERMINISTIC_OPS"),
        "tf_enable_onednn_opts_env": os.environ.get("TF_ENABLE_ONEDNN_OPTS"),
        "op_determinism_error": OP_DETERMINISM_ERROR,
        "seed_before_construction": bool(deterministic),
        "optimizer": "Adam(1e-3) [unchanged from A/B]",
        "loss_weights": {"bins": float(bins_weight), "reject": 1.0},
        "loss_weights_note": "bins weight is THE changed knob vs Variant A "
                             "(A used 1.0/1.0); all other hyperparameters "
                             "inherited unchanged",
        "bins_class_weights": data["class_weights"],
        "unfreeze_proof": proof,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "train_images": n_tr, "val_images": n_va,
        "elapsed_seconds": round(dt, 2), "best_epoch": best_epoch,
        "best_val_loss": float(np.min(hist.history["val_loss"])),
        "best_val_bins_acc": val_bins_acc,
        "best_val_reject_acc": val_rej_acc,
    }
    config_json.write_text(json.dumps(cfg, indent=2) + "\n",
                           encoding="utf-8")
    print(f"      best epoch: {best_epoch}  "
          f"val_loss: {np.min(hist.history['val_loss']):.4f}  "
          f"val_bins_acc: {val_bins_acc:.4f}  "
          f"val_rej_acc: {val_rej_acc:.4f}  elapsed {dt:.0f}s  "
          f"config->{config_json}")
    print("      done.")


# --- CLI --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Iteration 6 Variant C: backbone frozen, fc_256 + "
                    "predictions + reject trainable, bins loss weight 2.0.")
    ap.add_argument("--model", default=str(BASE_CHECKPOINT),
                    help="baseline checkpoint (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="best-checkpoint output path (default: derived from "
                         "--out-tag, e.g. wastelens_rej_varc7_best.keras)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="training epochs (default: %(default)s)")
    ap.add_argument("--bins-weight", type=float, default=DEFAULT_BINS_WEIGHT,
                    help="bins loss weight (default: %(default)s)")
    ap.add_argument("--smoke", action="store_true",
                    help="96-image smoke run for a fast syntax/pipeline check")
    ap.add_argument("--out-tag", default="",
                    help="artifact tag isolating this run's outputs "
                         "(e.g. '7' -> wastelens_rej_varc7_*.keras)")
    ap.add_argument("--train-seed", type=int, default=None,
                    help="seed for init/dropout/shuffle (default: 42 = SEED); "
                         "the dataset seed stays 42 either way")
    ap.add_argument("--no-deterministic", dest="deterministic",
                    action="store_false",
                    help="seed nothing up front (legacy Iteration-6/7 "
                         "behaviour); off by default")
    ap.set_defaults(deterministic=True)
    args = ap.parse_args()

    print("WasteLens - Iteration 6 Variant C "
          "(backbone frozen; fc_256 + predictions + reject trainable)")
    print(f"      base checkpoint : {args.model}")
    print(f"      out checkpoint  : {args.out}")
    print(f"      epochs          : {args.epochs}")
    print(f"      bins loss weight: {args.bins_weight}")
    print(f"      train seed      : "
          f"{args.train_seed if args.train_seed is not None else SEED}")
    print(f"      deterministic   : {args.deterministic}")
    print(f"      smoke           : {args.smoke}")
    print("      data ...")
    data = tr.build_datasets()
    print(f"      done. supported={len(data['splits']['train'])} "
          f"train / {len(data['splits']['val'])} val / "
          f"{len(data['splits']['test'])} test")
    run_training(data, smoke=args.smoke, epochs=args.epochs,
                 bins_weight=args.bins_weight, out_tag=args.out_tag,
                 base_ckpt=args.model, out_ckpt_path=args.out,
                 train_seed=args.train_seed, deterministic=args.deterministic)


if __name__ == "__main__":
    main()

