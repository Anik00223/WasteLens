# src/train_head_adaptation.py
#
# WasteLens Iteration 11 - LOOP 5: classification-head-only adaptation.
#
# Implements docs/rejection_experiment/iteration11_head_adaptation_protocol.md
# exactly: warm start from the shipped Variant C checkpoint, and the ONLY
# trainable tensors are `predictions/kernel` + `predictions/bias` (the
# Dense(4, softmax) bin head, 1,028 params). Backbone (incl. BatchNorm moving
# statistics), gap, both dropouts, fc_256 AND the rejection head are frozen.
# The dataset mix / split / weights code is IMPORTED from train_adaptation.py
# (same files, same 3:1 replay, same Adam(1e-3)/batch 32/6 epochs), so the
# trainable set is the single changed variable vs Iteration 10.
#
# The run records a freeze proof before and after training and refuses to
# continue if any frozen tensor (or the reject head's predictions) changed.
# Checkpoints go to scratch paths
# (models/checkpoints/wastelens_rej_head11s{SEED}_*.keras) - NEVER to
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
import train_adaptation as wl10
import train_rejection as tr

OP_DETERMINISM_ERROR = None
try:
    tf.config.experimental.enable_op_determinism()
except Exception as exc:  # pragma: no cover - recorded in config
    OP_DETERMINISM_ERROR = f"{type(exc).__name__}: {exc}"

SHIPPED_CHECKPOINT = wl10.SHIPPED_CHECKPOINT
ADAPT_MANIFEST = wl10.ADAPT_MANIFEST
ADAPT_DIR = wl10.ADAPT_DIR
FRESH_MANIFEST = wl10.FRESH_MANIFEST
CKPT_DIR = wl10.CKPT_DIR
OUT_DIR = wl10.OUT_DIR
BACKBONE_NAME = wl10.BACKBONE_NAME

# Registered dataset locks (protocol §3) - asserted before the model loads.
ADAPT_MANIFEST_MD5 = "acae0933aca123c835c1e55396af182a"
ADAPT_DIR_FILE_COUNT = 94
FRESH_MANIFEST_MD5 = "99540f56efb95e2c26b2b6cc9414ff5d"
SHIPPED_CHECKPOINT_MD5 = "8735019226412c4d67b0a17269dc38ad"

EPOCHS = wl10.EPOCHS                       # 6, protocol §4
TRAINABLE_PATHS = ["predictions/bias", "predictions/kernel"]
TRAINABLE_PARAMS = 1028
FROZEN_WEIGHT_COUNT = 264                  # 266 total - 2 trainable


def sha256_weights(tensors) -> dict[str, str]:
    """Path -> sha256 of the raw tensor bytes (all dtypes preserved)."""
    return {w.path: hashlib.sha256(np.asarray(w).tobytes()).hexdigest()
            for w in tensors}


def digest_aggregate(per_path: dict[str, str]) -> str:
    blob = "\n".join(f"{p}:{h}" for p, h in sorted(per_path.items()))
    return hashlib.sha256(blob.encode()).hexdigest()


def lock_dataset() -> dict:
    """Assert the protocol §3 dataset locks before anything else happens."""
    got_adapt = wl10.md5_file(ADAPT_MANIFEST)
    got_fresh = wl10.md5_file(FRESH_MANIFEST)
    n_files = len([p for p in ADAPT_DIR.iterdir() if p.is_file()])
    checks = {
        "adapt_manifest_md5": (got_adapt, ADAPT_MANIFEST_MD5),
        "adapt_dir_file_count": (n_files, ADAPT_DIR_FILE_COUNT),
        "fresh_manifest_md5": (got_fresh, FRESH_MANIFEST_MD5),
    }
    for name, (got, want) in checks.items():
        if got != want:
            raise SystemExit(f"dataset lock FAILED for {name}: got {got}, "
                             f"protocol §3 registers {want}")
    return {k: v[0] for k, v in checks.items()}


def build_head_only_model(checkpoint: Path = SHIPPED_CHECKPOINT,
                          bins_weight: float = 2.0,
                          reject_weight: float = 1.0):
    """Warm start from the shipped checkpoint; ONLY the bin head trains.

    Loading (not rebuilding) keeps every frozen tensor bit-identical to
    production; the assertions below make the trainable-set contract explicit.
    """
    model = tf.keras.models.load_model(checkpoint)
    for layer in model.layers:
        layer.trainable = layer.name == "predictions"
    model.get_layer(BACKBONE_NAME).trainable = False   # nested, explicit
    trainable = sorted(w.path for w in model.trainable_weights)
    if trainable != TRAINABLE_PATHS:
        raise RuntimeError(f"trainable set is not head-only: {trainable}")
    n_params = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    if n_params != TRAINABLE_PARAMS:
        raise RuntimeError(f"trainable params {n_params} != {TRAINABLE_PARAMS}")
    for w in model.weights:
        should_be_trainable = w.path in TRAINABLE_PATHS
        if bool(w.trainable) != should_be_trainable:
            raise RuntimeError(f"weight {w.path} trainable={w.trainable}, "
                               f"expected {should_be_trainable}")
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


def freeze_snapshot(model, data, warm_ckpt: Path) -> dict:
    """Before training: digests of every weight + warm-start output check."""
    per_path = sha256_weights(model.weights)
    frozen = {p: h for p, h in per_path.items() if p not in TRAINABLE_PATHS}
    if len(frozen) != FROZEN_WEIGHT_COUNT:
        raise RuntimeError(f"frozen tensor count {len(frozen)} != "
                           f"{FROZEN_WEIGHT_COUNT}")
    proof = wl10.warm_start_proof(model, data, warm_ckpt=warm_ckpt,
                                  n_images=16)
    return {
        "checkpoint_md5": wl10.md5_file(warm_ckpt),
        "checkpoint_md5_expected": SHIPPED_CHECKPOINT_MD5,
        "trainable_paths": TRAINABLE_PATHS,
        "trainable_params": TRAINABLE_PARAMS,
        "frozen_tensor_count": len(frozen),
        "frozen_aggregate_sha256": digest_aggregate(frozen),
        "all_aggregate_sha256": digest_aggregate(per_path),
        "bins_kernel_sha256_before": per_path["predictions/kernel"],
        "bins_bias_sha256_before": per_path["predictions/bias"],
        "reject_kernel_sha256": per_path["reject/kernel"],
        "reject_bias_sha256": per_path["reject/bias"],
        "frozen_digests": frozen,
        "warm_start_output_proof": proof,
    }


def verify_freeze_after(model, before: dict, data) -> dict:
    """After training: frozen tensors byte-identical; reject output unmoved."""
    per_path = sha256_weights(model.weights)
    frozen = {p: h for p, h in per_path.items() if p not in TRAINABLE_PATHS}
    changed = sorted(p for p in frozen
                     if frozen[p] != before["frozen_digests"].get(p))
    if changed:
        raise RuntimeError(f"FREEZE VIOLATION: frozen tensors changed: "
                           f"{changed}")
    before_bins = {"predictions/kernel": before["bins_kernel_sha256_before"],
                   "predictions/bias": before["bins_bias_sha256_before"]}
    bins_moved = sorted(p for p, h in before_bins.items() if per_path[p] != h)
    rows = sorted(data["splits"]["train"])
    sample = [str(p) for p, _ in rows[:16]]
    shipped = tf.keras.models.load_model(SHIPPED_CHECKPOINT)
    got = model.predict(wl10.img_only_ds(sample), verbose=0)
    ref = shipped.predict(wl10.img_only_ds(sample), verbose=0)
    diffs = {h: float(np.max(np.abs(np.asarray(got[h])
                                    - np.asarray(ref[h]))))
             for h in ("bins", "reject")}
    if diffs["reject"] > 1e-6:
        raise RuntimeError(f"reject outputs moved during head-only training: "
                           f"{diffs['reject']}")
    if not bins_moved:
        raise RuntimeError("bin head did not move - training had no effect")
    return {
        "frozen_tensors_verified": len(frozen),
        "frozen_aggregate_sha256_after": digest_aggregate(frozen),
        "frozen_identical": True,
        "bins_head_changed": bins_moved,
        "bins_kernel_sha256_after": per_path["predictions/kernel"],
        "bins_bias_sha256_after": per_path["predictions/bias"],
        "post_training_output_diff_vs_shipped": diffs,
        "note": "16-image check; the full-scale equivalence test is LOOP 12 "
                "(src/eval_head_adaptation.py)",
    }


# --- Artifacts --------------------------------------------------------------

def write_artifacts(seed: int, tag: str, epochs: int, info: dict,
                    snapshot: dict, after: dict, hist, adapt_records: list,
                    wall_sec: float, ckpt_pattern: str, locks: dict,
                    smoke: bool) -> None:
    """Per-seed history CSV, freeze proof and training config; copies the
    final epoch to the `_final`/`_best` aliases (copied, never selected)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(hist.history.keys())
    adapt_keys: list[str] = []
    for rec in adapt_records:
        for k in rec:
            if k not in adapt_keys:
                adapt_keys.append(k)
    hist_csv = OUT_DIR / f"head11s{seed}{tag}_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys + adapt_keys)
        for i in range(len(hist.history[keys[0]])):
            rec = adapt_records[i] if i < len(adapt_records) else {}
            w.writerow([float(hist.history[k][i]) for k in keys]
                       + [float(rec.get(k, np.nan)) for k in adapt_keys])

    final_ckpt = Path(ckpt_pattern.format(epoch=epochs))
    cand_ckpt = CKPT_DIR / f"wastelens_rej_head11s{seed}{tag}_final.keras"
    alias_ckpt = CKPT_DIR / f"wastelens_rej_head11s{seed}{tag}_best.keras"
    shutil.copyfile(final_ckpt, cand_ckpt)
    shutil.copyfile(final_ckpt, alias_ckpt)
    hashes = {p.name: wl10.md5_file(p)
              for p in (final_ckpt, cand_ckpt, alias_ckpt)}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(f"final-epoch copies differ: {hashes}")

    freeze_path = OUT_DIR / f"head11s{seed}{tag}_freeze_proof.json"
    freeze_path.write_text(json.dumps(
        {"generated_utc": datetime.now(timezone.utc)
         .strftime("%Y-%m-%d %H:%M:%S UTC"),
         "protocol": "docs/rejection_experiment/"
                     "iteration11_head_adaptation_protocol.md",
         "method": "sha256 over every weight's raw bytes; frozen set = all "
                   "tensors except predictions/{bias,kernel}",
         "before": snapshot, "after": after}, indent=2) + "\n",
        encoding="utf-8")

    last_acc_key = [k for k in keys if k.startswith("val_bins")
                    and "accuracy" in k][0]
    cfg = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "protocol": "docs/rejection_experiment/"
                    "iteration11_head_adaptation_protocol.md",
        "rule": (f"warm start {SHIPPED_CHECKPOINT}; CLASSIFICATION-HEAD-ONLY: "
                 f"trainable = predictions/kernel + predictions/bias "
                 f"({TRAINABLE_PARAMS} params); frozen = backbone (incl. BN "
                 f"stats), gap, dropouts, fc_256, reject; replayed 3:1 "
                 f"original:adapt mix, Adam(1e-3), batch {wl.BATCH_SIZE}, "
                 f"{epochs} epochs, candidate = FINAL epoch (no selection)"),
        "seed": seed, "epochs": epochs, "candidate_epoch": epochs,
        "smoke": bool(smoke),
        "optimizer": "Adam", "learning_rate": 1e-3,
        "loss_weights": {"bins": 2.0, "reject": 1.0},
        "batch_size": int(wl.BATCH_SIZE), "img_size": list(wl.IMG_SIZE),
        "dataset_seed": int(wl.SEED),
        "determinism": {
            "TF_DETERMINISTIC_OPS_env": os.environ.get("TF_DETERMINISTIC_OPS"),
            "enable_op_determinism_error": OP_DETERMINISM_ERROR,
            "keras_set_random_seed": seed,
        },
        "trainable_set": {"paths": snapshot["trainable_paths"],
                          "params": snapshot["trainable_params"],
                          "frozen_tensor_count":
                              snapshot["frozen_tensor_count"]},
        "warm_start": {
            "checkpoint": str(SHIPPED_CHECKPOINT),
            "checkpoint_md5": snapshot["checkpoint_md5"],
            "checkpoint_md5_expected": snapshot["checkpoint_md5_expected"],
            "output_proof": snapshot["warm_start_output_proof"],
            "frozen_aggregate_sha256": snapshot[
                "frozen_aggregate_sha256"],
        },
        "freeze_proof_after": {
            "frozen_identical": after["frozen_identical"],
            "frozen_tensors_verified": after["frozen_tensors_verified"],
            "frozen_aggregate_sha256": after[
                "frozen_aggregate_sha256_after"],
            "bins_head_changed": after["bins_head_changed"],
            "post_training_output_diff_vs_shipped": after[
                "post_training_output_diff_vs_shipped"],
            "detail_file": str(freeze_path),
        },
        "mix": info,
        "dataset_locks": locks,
        "fresh_benchmark_reference": {
            "path": str(FRESH_MANIFEST), "md5": locks["fresh_manifest_md5"],
            "note": "provenance only; this script never opens a fresh file"},
        "adapt_val_records": adapt_records,
        "final_epoch": {
            "train_loss": float(hist.history["loss"][-1]),
            "val_loss": float(hist.history["val_loss"][-1]),
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
                     "post-gate decision (protocol §8)",
    }
    cfg_path = OUT_DIR / f"head11s{seed}{tag}_training_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"      candidate = {final_ckpt.name} (md5 {hashes[final_ckpt.name]})")
    print(f"wrote {hist_csv}")
    print(f"wrote {freeze_path}")
    print(f"wrote {cfg_path}")


# --- Run --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Iteration-11 classification-head-only adaptation "
                    "(registered protocol §4; the final epoch is the "
                    "candidate).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--bins-weight", type=float, default=2.0)
    ap.add_argument("--reject-weight", type=float, default=1.0)
    ap.add_argument("--model", type=Path, default=SHIPPED_CHECKPOINT,
                    help="warm-start checkpoint (registered: shipped varc)")
    ap.add_argument("--smoke", action="store_true",
                    help="pipeline check only: 1 epoch, truncated pools")
    args = ap.parse_args()
    if not args.smoke:
        if args.epochs != EPOCHS:
            raise SystemExit(f"protocol fixes the budget at {EPOCHS} "
                             f"epochs; got {args.epochs}")
        if Path(args.model) != SHIPPED_CHECKPOINT:
            raise SystemExit(f"protocol fixes the warm start at "
                             f"{SHIPPED_CHECKPOINT}; got {args.model}")
    tag = "_smoke" if args.smoke else ""
    epochs = 1 if args.smoke else args.epochs
    seed = int(args.seed)

    print("REGISTERED RULE: docs/rejection_experiment/"
          "iteration11_head_adaptation_protocol.md §2/§4 - warm start "
          f"{args.model}, CLASSIFICATION-HEAD-ONLY (trainable: "
          "predictions/kernel + predictions/bias), everything else frozen; "
          f"Adam(1e-3), batch {wl.BATCH_SIZE}, {epochs} epochs, candidate = "
          "FINAL epoch (no selection).")
    locks = lock_dataset()
    got_md5 = wl10.md5_file(Path(args.model))
    if got_md5 != SHIPPED_CHECKPOINT_MD5:
        raise SystemExit(f"warm-start checkpoint md5 {got_md5} != registered "
                         f"{SHIPPED_CHECKPOINT_MD5}")
    print(f"      dataset locks OK: {json.dumps(locks)}")
    print(f"      warm-start checkpoint md5 OK: {got_md5}")

    data = tr.build_datasets()
    pool = wl10.adapt_rows()
    split = wl10.split_adapt(pool, wl.SEED)
    print(f"adaptation pool: supported {len(pool['supported'])}, ood "
          f"{len(pool['ood'])}, multi {len(pool['multi'])} -> deterministic "
          f"80/20 split (seed {wl.SEED})")
    paths, y_bins, y_rej, sw_bins, sw_rej, info = wl10.build_mixed_arrays(
        data, split)
    cw = info["bins_class_weights"]
    av_pack = wl10.pack_rows(sorted(split["supported"]["val"],
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
    tr_ds = wl10.make_ds(paths, y_bins, y_rej, sw_bins, sw_rej, True, seed)
    va_ds = wl10.make_ds(*va_pack[:5], False, seed)
    av_ds = wl10.make_ds(*av_pack[:5], False, seed)

    tf.keras.utils.set_random_seed(seed)
    model = build_head_only_model(Path(args.model), args.bins_weight,
                                  args.reject_weight)
    snapshot = freeze_snapshot(model, data, Path(args.model))
    print(f"      trainable: {snapshot['trainable_paths']} "
          f"({snapshot['trainable_params']} params); frozen tensors "
          f"{snapshot['frozen_tensor_count']} (aggregate "
          f"{snapshot['frozen_aggregate_sha256'][:16]}...)")
    wproof = snapshot["warm_start_output_proof"]
    print(f"      warm-start proof: max_abs_diff="
          f"{json.dumps(wproof['max_abs_diff'])} bit_exact="
          f"{json.dumps(wproof['bit_exact'])}")
    tf.keras.utils.set_random_seed(seed)   # Iteration-8 reseed-before-fit

    ckpt_pattern = str(CKPT_DIR /
                       f"wastelens_rej_head11s{seed}{tag}_{{epoch:02d}}.keras")
    ckpt_every = tf.keras.callbacks.ModelCheckpoint(
        ckpt_pattern, monitor="val_loss", save_best_only=False, verbose=0)
    adapt_logger = wl10.AdaptValLogger(
        av_ds, steps=math.ceil(len(av_pack[0]) / wl.BATCH_SIZE))
    callbacks = [ckpt_every, adapt_logger]
    if hasattr(tr, "maybe_tensorboard"):
        callbacks += list(tr.maybe_tensorboard(
            OUT_DIR / f"head11s{seed}{tag}_tensorboard"))
    t0 = time.time()
    hist = model.fit(tr_ds,
                     steps_per_epoch=math.ceil(len(paths) / wl.BATCH_SIZE),
                     validation_data=va_ds,
                     validation_steps=math.ceil(len(va_pack[0])
                                                / wl.BATCH_SIZE),
                     epochs=epochs, callbacks=callbacks, verbose=1)
    wall = time.time() - t0
    print(f"      trained {epochs} epochs in {wall / 60:.2f} min")
    after = verify_freeze_after(model, snapshot, data)
    diffs = after["post_training_output_diff_vs_shipped"]
    print(f"      freeze proof: frozen_identical={after['frozen_identical']}"
          f" ({after['frozen_tensors_verified']} tensors, aggregate "
          f"{after['frozen_aggregate_sha256_after'][:16]}...); reject drift "
          f"{diffs['reject']:.2e}; bins drift {diffs['bins']:.4f}")
    write_artifacts(seed, tag, epochs, info, snapshot, after, hist,
                    adapt_logger.records, wall, ckpt_pattern, locks,
                    args.smoke)


if __name__ == "__main__":
    main()
