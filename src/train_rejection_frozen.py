# src/train_rejection_frozen.py
#
# WasteLens Iteration 4 - VARIANT B: rejection head on the FROZEN baseline
# representation.
#
# WHY THIS EXISTS
#   Variant A (src/train_rejection.py) trains the whole head jointly at
#   Adam(1e-3): the shared fc_256 representation moves while the rejection head
#   learns, which cost the four-bin head 1.06 points of held-out accuracy
#   (0.9716 vs the 0.9822 baseline) - outside this experiment's pre-registered
#   ACC_TOLERANCE of 0.5pt, even though the rejection head itself separated
#   unsupported inputs almost perfectly (AUROC 0.9997).
#
#   Variant B freezes the ENTIRE baseline subgraph (backbone + gap + dropout_1 +
#   fc_256 + dropout_2 + predictions) at the shipped checkpoint's weights and
#   trains ONLY the new Dense(1, sigmoid) rejection head on the frozen
#   dropout_2 (fc_256) representation. Because no weight on the bins path can
#   move, the bins output is numerically IDENTICAL to
#   models/checkpoints/wastelens_ep09.keras - four-bin accuracy cannot degrade.
#   That isolates exactly one question, per the iteration brief (section 9):
#
#     can a rejection head trained on the FROZEN baseline features still
#     separate supported single-item photos from unsupported images?
#
#   Differences vs Variant A: ONLY the frozen flags (Variant A trains the whole
#   head). Same data, same splits, same seeds, same optimizer, same loss
#   weights, same epochs, same batch size.
#
# DATA (identical to Variant A, leakage controls unchanged)
#   train_rejection.build_datasets() -> assemble() -> make_weighted_ds()
#   Supported images: the untouched seed-42 train/val split from train.py.
#   Unsupported (train/val only): md5-deduped clothes+shoes, CIFAR-10 non-waste
#   objects, deterministic multi-item collages built from DISJOINT train-split
#   sources. Test parts and the synthetic probes are never trained on.
#
# Run from the repo root:  python src/train_rejection_frozen.py
# Outputs:
#   models/checkpoints/wastelens_rej_frozen_best.keras   (best val_loss epoch)
#   docs/rejection_experiment/frozen_history.csv         (per-epoch metrics)
#   docs/rejection_experiment/frozen_training_config.json (config + freeze proof)

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

import train as wl
import train_rejection as tr

BASE_CHECKPOINT = Path("models/checkpoints/wastelens_ep09.keras")
OUT_CKPT = Path("models/checkpoints/wastelens_rej_frozen_best.keras")
OUT_DIR = Path("docs/rejection_experiment")
HISTORY_CSV = OUT_DIR / "frozen_history.csv"
CONFIG_JSON = OUT_DIR / "frozen_training_config.json"

EPOCHS = tr.EPOCHS
SEED = tr.SEED
MONITOR = "val_loss"


def build_frozen_dual(base_checkpoint: Path = BASE_CHECKPOINT):
    """Baseline model with EVERY weight frozen + a trainable rejection head.

    The bins path keeps the exact shipped layer structure
    (gap -> dropout_1 -> fc_256 -> dropout_2 -> predictions); the rejection head
    hangs off the same dropout_2 tensor. Freezing every baseline layer means the
    bins output equals the baseline checkpoint's output exactly.
    """
    base = tf.keras.models.load_model(base_checkpoint)
    for layer in base.layers:
        layer.trainable = False
    rej = tf.keras.layers.Dense(1, activation="sigmoid", name="reject")(
        base.get_layer("dropout_2").output
    )
    model = tf.keras.Model(
        base.input,
        {"bins": base.output, "reject": rej},
        name="wastelens_reject_frozen",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "bins": "sparse_categorical_crossentropy",
            "reject": "binary_crossentropy",
        },
        loss_weights={
            "bins": 1.0,
            "reject": 1.0,
        },
        metrics={
            "bins": ["sparse_categorical_accuracy"],
            "reject": ["binary_accuracy"],
        },
    )
    return model


# --- Freeze proof ---------------------------------------------------------


def _img_dataset(paths, batch_size=8):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    def _load(p):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, wl.IMG_SIZE)
        return wl.PREPROCESS_INPUT(img)
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)


def freeze_proof(data, base_ckpt=BASE_CHECKPOINT):
    """Assert the frozen model's bins output is numerically identical to the
    baseline checkpoint on a representative sample, BEFORE any training."""
    base = tf.keras.models.load_model(base_ckpt)
    frozen = build_frozen_dual(base_ckpt)
    rows = sorted(data["splits"]["train"])
    sample = [str(p) for p, _ in rows[:16]]
    base_probs = base.predict(_img_dataset(sample), verbose=0)
    froz_probs = frozen.predict(_img_dataset(sample), verbose=0)["bins"]
    max_abs = float(np.max(np.abs(base_probs - froz_probs)))
    identical = bool(np.array_equal(base_probs, froz_probs))
    if not np.allclose(base_probs, froz_probs, atol=1e-5, rtol=1e-5):
        raise RuntimeError(
            f"freeze proof FAILED: bins differ (max_abs_diff={max_abs:.2e})")
    return {"method": "predict 16 supported train images (baseline vs frozen)",
            "max_abs_diff": max_abs, "identical_bit_exact": identical}


# --- Training -------------------------------------------------------------
def maybe_tensorboard(tb_log: Path) -> list:
    """Return [TensorBoard] when the `tensorboard` package is installed, else [].

    Keras can be installed without the standalone `tensorboard` package; in that
    case the callback constructs but `model.fit` raises
    TBNotInstalledError. This keeps both Iteration-4 trainers running in minimal
    environments (e.g. a CI/dev machine) without the optional dependency.
    """
    try:
        import tensorboard  # noqa: F401  (optional dependency)
    except Exception:
        return []
    tb_log.mkdir(parents=True, exist_ok=True)
    return [tf.keras.callbacks.TensorBoard(
        str(tb_log), histogram_freq=0, update_freq="epoch")]


def run_training(data, smoke=False, epochs=None):
    tr_pack = tr.assemble(data, "train", smoke)
    va_pack = tr.assemble(data, "val", smoke)
    tr_ds = tr.make_weighted_ds(*tr_pack[:5], training=True)
    va_ds = tr.make_weighted_ds(*va_pack[:5], training=False)
    n_tr = len(tr_pack[0]); n_va = len(va_pack[0])
    print(f"      train: {tr_pack[5]} supported + {tr_pack[6]} unsupported = "
          f"{n_tr} images ({math.ceil(n_tr / wl.BATCH_SIZE)} steps/epoch)")
    print(f"      val:   {va_pack[5]} supported + {va_pack[6]} unsupported = "
          f"{n_va} images ({math.ceil(n_va / wl.BATCH_SIZE)} val steps)")
    n_epochs = int(epochs) if epochs else EPOCHS
    model = build_frozen_dual()
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    print(f"      trainable params: {trainable:,}  (reject head only)")
    print(f"      frozen params:    {non_trainable:,}")
    fp = freeze_proof(data)
    print(f"      freeze proof: bit_exact={fp['identical_bit_exact']}  "
          f"max_abs_diff={fp['max_abs_diff']:.2e}")

    ckpt_dir = Path("models/checkpoints")
    ckpt_every = tf.keras.callbacks.ModelCheckpoint(
        str(ckpt_dir / "wastelens_rej_frozen_{epoch:02d}.keras"),
        monitor=MONITOR, save_best_only=False, verbose=0)
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        str(OUT_CKPT), monitor=MONITOR, save_best_only=True, verbose=1)
    tb = maybe_tensorboard(OUT_DIR / "frozen_tensorboard")
    t0 = time.time()
    hist = model.fit(tr_ds,
        steps_per_epoch=math.ceil(n_tr / wl.BATCH_SIZE),
        validation_data=va_ds,
        validation_steps=math.ceil(n_va / wl.BATCH_SIZE),
        epochs=n_epochs,
        callbacks=[ckpt_every, ckpt_best] + list(tb), verbose=1)
    dt = time.time() - t0
    keys = list(hist.history.keys())
    with HISTORY_CSV.open("w", newline="") as f:
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
    cfg = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
           "variant": "B (rejection head on FROZEN baseline representation)",
           "base_checkpoint": str(BASE_CHECKPOINT), "out_checkpoint": str(OUT_CKPT),
           "epochs": n_epochs, "seed": SEED, "monitor": MONITOR,
           "optimizer": "Adam(1e-3)",
           "loss_weights": {"bins": 1.0,
                            "reject": 1.0},
           "freeze_proof": fp, "trainable_params": trainable,
           "frozen_params": non_trainable,
           "train_images": n_tr, "val_images": n_va,
           "elapsed_seconds": round(dt, 2), "best_epoch": best_epoch,
           "best_val_loss": float(np.min(hist.history["val_loss"])),
           "best_val_bins_acc": val_bins_acc,
           "best_val_reject_acc": val_rej_acc}
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"      best epoch: {best_epoch}  val_loss: {np.min(hist.history['val_loss']):.4f}  "
          f"val_bins_acc: {val_bins_acc:.4f}  val_rej_acc: {val_rej_acc:.4f}  "
          f"elapsed {dt:.0f}s  config->{CONFIG_JSON}")
    print("      done.")


# --- CLI ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Iteration 4 Variant B: rejection head on the frozen "
                    "baseline representation.")
    ap.add_argument("--model", default=str(BASE_CHECKPOINT),
                    help="baseline checkpoint to freeze (default: %(default)s)")
    ap.add_argument("--out", default=str(OUT_CKPT),
                    help="best-checkpoint output path (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="training epochs (default: %(default)s)")
    ap.add_argument("--smoke", action="store_true",
                    help="96-image smoke run for a fast syntax/pipeline check")
    ap.add_argument("--no-freeze-proof", action="store_true",
                    help="skip the pre-training freeze proof (debug only)")
    args = ap.parse_args()

    print("WasteLens - Iteration 4 Variant B "
          "(rejection head on FROZEN baseline representation)")
    print(f"      base checkpoint : {args.model}")
    print(f"      out checkpoint  : {args.out}")
    print(f"      epochs          : {args.epochs}")
    print(f"      smoke           : {args.smoke}")
    print("      data ...")
    data = tr.build_datasets()
    print(f"      done. supported={len(data['splits']['train'])} "
          f"train / {len(data['splits']['val'])} val / "
          f"{len(data['splits']['test'])} test")
    if not args.no_freeze_proof:
        print("      freeze proof ...")
        fp = freeze_proof(data)
        print(f"      freeze proof: bit_exact={fp['identical_bit_exact']}  "
              f"max_abs_diff={fp['max_abs_diff']:.2e}")
    run_training(data, smoke=args.smoke, epochs=args.epochs)


if __name__ == "__main__":
    main()
