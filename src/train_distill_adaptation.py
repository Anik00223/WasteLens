# Chunk 1/3: header, imports, teacher, KD loss, student builder.
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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import tensorflow as tf

import train as wl
import train_adaptation as wl10
import train_head_adaptation as th
import train_rejection as tr

OP_DETERMINISM_ERROR = None
try:
    tf.config.experimental.enable_op_determinism()
except Exception as exc:
    OP_DETERMINISM_ERROR = f"{type(exc).__name__}: {exc}"

SHIPPED_CHECKPOINT = th.SHIPPED_CHECKPOINT
CKPT_DIR = th.CKPT_DIR
OUT_DIR = th.OUT_DIR
BACKBONE_NAME = th.BACKBONE_NAME
SHIPPED_CHECKPOINT_MD5 = th.SHIPPED_CHECKPOINT_MD5

PROTOCOL = "iteration12_distillation_protocol.md"

LAMBDA_KD = 0.5
KD_TEMP = 2.0
KD_EPS = 1e-12

EPOCHS = wl10.EPOCHS
TRAINABLE_PATHS = th.TRAINABLE_PATHS
TRAINABLE_PARAMS = th.TRAINABLE_PARAMS


def build_teacher(path: Path = SHIPPED_CHECKPOINT) -> tf.keras.Model:
    """Frozen shipped Variant C as the distillation teacher."""
    if wl10.md5_file(path) != SHIPPED_CHECKPOINT_MD5:
        raise RuntimeError(f"teacher source md5 mismatch for {path}")
    teacher = tf.keras.models.load_model(path)
    for layer in teacher.layers:
        layer.trainable = False
    teacher.get_layer(BACKBONE_NAME).trainable = False
    if teacher.trainable_weights:
        raise RuntimeError(
            "teacher is not frozen: "
            f"{[w.path for w in teacher.trainable_weights]}")
    return teacher


def teacher_proof(teacher, student_snapshot: dict) -> dict:
    """LOOP 5: teacher byte-identical to production, zero trainable tensors."""
    stu_frozen = student_snapshot["frozen_digests"]
    n_match = 0
    total_frozen = 0
    for w in teacher.weights:
        if w.path in stu_frozen:
            total_frozen += 1
            digest = hashlib.sha256(np.asarray(w).tobytes()).hexdigest()
            if digest == stu_frozen[w.path]:
                n_match += 1
            else:
                raise RuntimeError(f"teacher/student divergence: {w.path}")
    return {
        "source_checkpoint": str(SHIPPED_CHECKPOINT),
        "source_md5": SHIPPED_CHECKPOINT_MD5,
        "teacher_trainable_params": int(sum(
            int(np.size(np.asarray(w)))
            for w in teacher.trainable_weights)),
        "teacher_total_weights": int(len(teacher.weights)),
        "frozen_tensors_byte_identical": n_match,
        "frozen_tensors_compared": total_frozen,
    }


def kd_loss_fn(t_prob, s_prob):
    """T^2-scaled batch-mean KD loss over masked original rows."""
    t_prob = tf.cast(t_prob, tf.float32)
    s_prob = tf.cast(s_prob, tf.float32)
    t_s = tf.pow(t_prob + KD_EPS, 1.0 / KD_TEMP)
    s_s = tf.pow(s_prob + KD_EPS, 1.0 / KD_TEMP)
    q_t = t_s / (tf.reduce_sum(t_s, axis=-1, keepdims=True) + KD_EPS)
    q_s = s_s / (tf.reduce_sum(s_s, axis=-1, keepdims=True) + KD_EPS)
    q_t = tf.clip_by_value(q_t, KD_EPS, 1.0)
    q_s = tf.clip_by_value(q_s, KD_EPS, 1.0)
    kl = tf.reduce_sum(q_t * (tf.math.log(q_t) - tf.math.log(q_s)), axis=-1)
    return (KD_TEMP ** 2) * kl


def build_distill_student(path: Path = SHIPPED_CHECKPOINT,
                          bins_weight: float = 2.0,
                          reject_weight: float = 1.0):
    """Warm-started student with ONLY the bin head trainable (Iter-11 set)."""
    if wl10.md5_file(path) != SHIPPED_CHECKPOINT_MD5:
        raise RuntimeError(f"student warm-start md5 mismatch for {path}")
    model = tf.keras.models.load_model(path)
    for w in model.weights:
        w._trainable = (w.path in TRAINABLE_PATHS)
    trainable = sorted(w.path for w in model.trainable_weights)
    if trainable != sorted(TRAINABLE_PATHS):
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


def kd_flag_ds(paths, y_bins, y_rej, sw_bins, sw_rej, is_orig,
               training: bool, seed: int):
    """Training ds = wl10.make_ds + extra per-row KD (original-domain) flag."""
    is_orig = np.asarray(is_orig, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, y_bins, y_rej, sw_bins, sw_rej, is_orig))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(p, yb, yr, sb, sr, ko):
        img = tf.io.read_file(p)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, wl.IMG_SIZE)
        img = wl.PREPROCESS_INPUT(img)
        return (img, {"bins": yb, "reject": yr},
                {"bins": sb, "reject": sr}, ko)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(wl.BATCH_SIZE)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def kd_mask_stats(paths, is_orig) -> dict:
    n = len(paths)
    no = int(np.sum(np.asarray(is_orig) > 0.5))
    return {"n_train_rows": int(n), "n_orig_rows": no,
            "n_adapt_rows": int(n - no),
            "orig_fraction": float(no / max(n, 1))}


def run_manual_epoch(student, teacher, kd_ds, steps: int):
    """One reshuffled manual KD epoch; returns loss-component means."""
    opt = student.optimizer
    bce = tf.keras.losses.BinaryCrossentropy(reduction="none")
    tot = np.zeros(5, dtype=np.float64)
    n_batches = 0
    for bx, (img_b, y_b, sw_b, ko_b) in enumerate(kd_ds):
        if bx >= steps:
            break
        yb = tf.cast(y_b["bins"], tf.int32)
        yr = tf.cast(y_b["reject"], tf.float32)
        swb = tf.cast(sw_b["bins"], tf.float32)
        swr = tf.cast(sw_b["reject"], tf.float32)
        ko = tf.cast(ko_b, tf.float32)
        with tf.GradientTape() as tape:
            out = student(img_b, training=True)
            sb = tf.cast(out["bins"], tf.float32)
            sr = tf.cast(out["reject"], tf.float32)
            row_ce = tf.keras.losses.sparse_categorical_crossentropy(yb, sb)
            row_re = bce(tf.reshape(yr, [-1]), tf.reshape(sr, [-1]))
            bins_ce = tf.reduce_sum(row_ce * swb) / (
                tf.reduce_sum(swb) + KD_EPS)
            rej_ce = tf.reduce_sum(row_re * swr) / (
                tf.reduce_sum(swr) + KD_EPS)
            base = 2.0 * bins_ce + 1.0 * rej_ce
            kd_rows = kd_loss_fn(
                tf.stop_gradient(teacher(img_b, training=False)["bins"]), sb)
            kd_m = tf.reduce_sum(kd_rows * ko) / (
                tf.reduce_sum(ko) + KD_EPS)
            combined = base + LAMBDA_KD * kd_m
        grads = tape.gradient(combined, student.trainable_weights)
        opt.apply_gradients(zip(grads, student.trainable_weights))
        tot += np.array([float(base), float(bins_ce), float(rej_ce),
                         float(kd_m), float(combined)])
        n_batches += 1
    return (tot / max(n_batches, 1)).tolist()

def verify_freeze_after(model, snapshot, data) -> dict:
    """Post-training freeze proof: frozen tensors + reject drift (Iter-11)."""
    cur = th.sha256_weights([w for w in model.weights
                             if w.path not in TRAINABLE_PATHS])
    before = snapshot["frozen_digests"]
    if set(cur) != set(before):
        raise RuntimeError("frozen tensor set changed during training")
    bad = [p for p in cur if cur[p] != before[p]]
    if bad:
        raise RuntimeError(f"frozen tensors changed: {bad[:5]}")
    agg_after = th.digest_aggregate(cur)
    shipped = tf.keras.models.load_model(SHIPPED_CHECKPOINT)
    rows = sorted(data["splits"]["train"])
    sample = [str(p) for p, _ in rows[:16]]
    got = model.predict(wl10.img_only_ds(sample), verbose=0)
    ref = shipped.predict(wl10.img_only_ds(sample), verbose=0)
    diffs = {h: float(np.max(np.abs(np.asarray(got[h])
                                    - np.asarray(ref[h]))))
             for h in ("bins", "reject")}
    if diffs["reject"] > 1e-6:
        raise RuntimeError(f"reject outputs moved during training: "
                           f"{diffs['reject']}")
    return {"frozen_identical": True,
            "frozen_tensors_verified": len(cur),
            "frozen_aggregate_sha256_after": agg_after,
            "aggregate_matches_before":
                agg_after == snapshot["frozen_aggregate_sha256"],
            "post_training_output_diff_vs_shipped": diffs,
            "post_training_output_bit_exact":
                {h: bool(d == 0.0) for h, d in diffs.items()}}


def git_head() -> str:
    """Short git HEAD for provenance (best-effort; 'unknown' if unavailable)."""
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_artifacts(seed, tag, epochs, info, kd_stats, snapshot, after,
                    teacher_p, hist_rows, adapt_records, wall_sec,
                    ckpt_pattern, locks, smoke) -> None:
    """Per-seed history CSV + training config + freeze proof (Iter-11 set)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adapt_keys: list = []
    for rec in adapt_records:
        for k in rec:
            if k not in adapt_keys:
                adapt_keys.append(k)
    keys = ["epoch", "train_loss", "train_bins_ce", "train_reject_ce",
            "train_kd", "train_combined",
            "val_loss", "val_bins_loss",
            "val_bins_sparse_categorical_accuracy",
            "val_reject_loss", "val_reject_binary_accuracy"] + adapt_keys
    hist_csv = OUT_DIR / f"head12s{seed}{tag}_history.csv"
    with hist_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(epochs):
            base = list(hist_rows[i]) if i < len(hist_rows) else []
            rec = adapt_records[i] if i < len(adapt_records) else {}
            w.writerow(base + [float(rec.get(k, float("nan")))
                               for k in adapt_keys])
    final_ckpt = Path(ckpt_pattern.format(epoch=epochs))
    for alias in (f"wastelens_rej_head12s{seed}{tag}_final.keras",
                  f"wastelens_rej_head12s{seed}{tag}_best.keras"):
        shutil.copyfile(final_ckpt, CKPT_DIR / alias)
    (OUT_DIR / f"head12s{seed}{tag}_freeze_proof.json").write_text(
        json.dumps(
            {"generated_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"),
             "protocol": f"docs/rejection_experiment/{PROTOCOL}",
             "method": "sha256 over every weight's raw bytes; frozen set = "
                       "all tensors except predictions/{bias,kernel}",
             "before": snapshot, "after": after,
             "teacher": teacher_p}, indent=2) + "\n", encoding="utf-8")



def write_config(seed, tag, epochs, info, kd_stats, snapshot, after,
                 teacher_p, wall_sec, ckpt_pattern, locks, smoke) -> None:
    """Training config JSON (split out so no single edit exceeds limits)."""
    final_ckpt = Path(ckpt_pattern.format(epoch=epochs))
    cfg = {
        "generated_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"),
        "protocol": f"docs/rejection_experiment/{PROTOCOL}",
        "rule": (f"warm start {SHIPPED_CHECKPOINT}; HEAD-ONLY + DISTILLATION "
                 f"(lambda={LAMBDA_KD}, T={KD_TEMP}); trainable = "
                 f"predictions/kernel + predictions/bias ({TRAINABLE_PARAMS} "
                 f"params); replayed 3:1 mix, Adam(1e-3), batch "
                 f"{wl.BATCH_SIZE}, {epochs} epochs, candidate = FINAL"),
        "seed": seed, "epochs": epochs, "candidate_epoch": epochs,
        "smoke": bool(smoke),
        "distillation": {
            "lambda": LAMBDA_KD, "temperature": KD_TEMP,
            "t_squared_scaling": KD_TEMP ** 2,
            "formulation": "L = L_ce + lambda * T^2 * mean_{orig rows} "
                           "KL(q(teacher,T) || q(student,T))",
            "kd_applies_to": "original-domain training rows only",
            "rejection_head_distilled": False,
            "teacher": teacher_p,
            "kd_orig_rows_per_epoch": kd_stats["n_orig_rows"],
            "kd_orig_fraction": kd_stats["orig_fraction"],
            "immutable": "lambda/T registered before training",
        },
        "optimizer": "Adam", "learning_rate": 1e-3,
        "loss_weights": {"bins": 2.0, "reject": 1.0},
        "batch_size": int(wl.BATCH_SIZE), "img_size": list(wl.IMG_SIZE),
        "dataset_seed": int(wl.SEED),
        "determinism": {
            "TF_DETERMINISTIC_OPS_env":
                os.environ.get("TF_DETERMINISTIC_OPS"),
            "enable_op_determinism_error": OP_DETERMINISM_ERROR,
            "keras_set_random_seed": seed},
        "trainable_set": {"paths": snapshot["trainable_paths"],
                          "params": snapshot["trainable_params"],
                          "frozen_tensor_count":
                              snapshot["frozen_tensor_count"]},
        "warm_start": {
            "checkpoint": str(SHIPPED_CHECKPOINT),
            "checkpoint_md5": snapshot["checkpoint_md5"],
            "checkpoint_md5_expected": snapshot["checkpoint_md5_expected"],
            "output_proof": snapshot["warm_start_output_proof"],
            "frozen_aggregate_sha256": snapshot["frozen_aggregate_sha256"]},
        "dataset": {
            "adapt_manifest_md5": locks["adapt_manifest_md5"],
            "adapt_dir_file_count": locks["adapt_dir_file_count"],
            "fresh_manifest_md5": locks["fresh_manifest_md5"],
            "mix": info,
            "train_rows": int(kd_stats["n_train_rows"])},
        "checkpoints": {
            "pattern": ckpt_pattern,
            "final_epoch_file": str(final_ckpt),
            "final_md5": wl10.md5_file(final_ckpt),
            "aliases": [f"wastelens_rej_head12s{seed}{tag}_final.keras",
                        f"wastelens_rej_head12s{seed}{tag}_best.keras"]},
        "freeze": {
            "frozen_identical": after["frozen_identical"],
            "frozen_tensors_verified": after["frozen_tensors_verified"],
            "aggregate_before": snapshot["frozen_aggregate_sha256"],
            "aggregate_after": after["frozen_aggregate_sha256_after"],
            "bins_drift_vs_shipped": after[
                "post_training_output_diff_vs_shipped"]["bins"],
            "reject_drift_vs_shipped": after[
                "post_training_output_diff_vs_shipped"]["reject"]},
        "wall_time": {"seconds": wall_sec, "minutes": wall_sec / 60.0},
        "git": {"head": git_head()},
    }
    (OUT_DIR / f"head12s{seed}{tag}_training_config.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bins-weight", type=float, default=2.0)
    ap.add_argument("--reject-weight", type=float, default=1.0)
    ap.add_argument("--model", type=str, default=str(SHIPPED_CHECKPOINT))
    args = ap.parse_args()

    seed, epochs = int(args.seed), int(args.epochs)
    tag = "_smoke" if args.smoke else ""
    print(f"[head12] seed={seed} epochs={epochs}{tag} "
          f"lambda={LAMBDA_KD} T={KD_TEMP}")

    locks = th.lock_dataset()
    print(f"      dataset locks OK: {json.dumps(locks)}")
    data = tr.build_datasets()
    pool = wl10.adapt_rows()
    split = wl10.split_adapt(pool, wl.SEED)
    paths, y_bins, y_rej, sw_bins, sw_rej, info = wl10.build_mixed_arrays(
        data, split)
    orig_rows = set(map(str, [p for p, _ in
                              sorted(data["splits"]["train"])]))
    for src in ("clothes", "shoes", "collage", "nonwaste"):
        orig_rows |= set(map(str, [r[0] for r in data["unsup"][src]["train"]]))
    is_orig = np.array([1.0 if str(p) in orig_rows else 0.0
                        for p in paths], dtype=np.float32)
    kd_stats = kd_mask_stats(paths, is_orig)
    print(f"      KD mask: {json.dumps(kd_stats)}")
    cw = info["bins_class_weights"]
    av_pack = wl10.pack_rows(sorted(split["supported"]["val"],
                                    key=lambda r: str(r[0])),
                             sorted(split["ood"]["val"]
                                    + split["multi"]["val"],
                                    key=lambda r: str(r[0])), cw)
    va_pack = tr.assemble(data, "val", smoke=False)
    if args.smoke:
        paths, y_bins = paths[:160], y_bins[:160]
        y_rej = y_rej[:160]
        sw_bins, sw_rej = sw_bins[:160], sw_rej[:160]
        is_orig = is_orig[:160]
        kd_stats = kd_mask_stats(paths, is_orig)
    print(f"      train mix: {info['n_supported']} supported + "
          f"{info['n_unsupported']} unsupported = {len(paths)} images "
          f"({math.ceil(len(paths) / wl.BATCH_SIZE)} steps/epoch)")
    kd_ds = kd_flag_ds(paths, y_bins, y_rej, sw_bins, sw_rej, is_orig,
                       True, seed)
    va_ds = wl10.make_ds(*va_pack[:5], False, seed)
    av_ds = wl10.make_ds(*av_pack[:5], False, seed)


    tf.keras.utils.set_random_seed(seed)
    student = build_distill_student(Path(args.model), args.bins_weight,
                                    args.reject_weight)
    teacher = build_teacher(Path(args.model))
    snapshot = th.freeze_snapshot(student, data, Path(args.model))
    teacher_p = teacher_proof(teacher, snapshot)
    print(f"      trainable: {snapshot['trainable_paths']} "
          f"({snapshot['trainable_params']} params); frozen "
          f"{snapshot['frozen_tensor_count']} (aggregate "
          f"{snapshot['frozen_aggregate_sha256'][:16]}...)")
    wproof = snapshot["warm_start_output_proof"]
    print(f"      teacher: {teacher_p['frozen_tensors_byte_identical']}/"
          f"{teacher_p['frozen_tensors_compared']} frozen tensors "
          f"byte-identical; trainable params "
          f"{teacher_p['teacher_trainable_params']}")
    print(f"      warm-start proof: max_abs_diff="
          f"{json.dumps(wproof['max_abs_diff'])} bit_exact="
          f"{json.dumps(wproof['bit_exact'])}")
    tf.keras.utils.set_random_seed(seed)

    ckpt_pattern = str(CKPT_DIR /
                       f"wastelens_rej_head12s{seed}{tag}_{{epoch:02d}}.keras")
    steps = math.ceil(len(paths) / wl.BATCH_SIZE)
    hist_rows: list = []
    adapt_records: list = []
    t0 = time.time()
    for ep in range(epochs):
        means = run_manual_epoch(student, teacher, kd_ds, steps)
        base, bce, rce, kdm, combined = means
        row = [ep + 1, base, bce, rce, kdm, combined]
        res_v = student.evaluate(va_ds, steps=math.ceil(
            len(va_pack[0]) / wl.BATCH_SIZE), verbose=0, return_dict=True)
        row += [float(res_v.get("loss", float("nan"))),
                float(res_v.get("bins_loss", float("nan"))),
                float(res_v.get("bins_sparse_categorical_accuracy",
                                float("nan"))),
                float(res_v.get("reject_loss", float("nan"))),
                float(res_v.get("reject_binary_accuracy", float("nan")))]
        hist_rows.append(row)
        res_a = student.evaluate(av_ds, steps=math.ceil(
            len(av_pack[0]) / wl.BATCH_SIZE), verbose=0, return_dict=True)
        rec = {f"adapt_val_{k}": float(v) for k, v in res_a.items()}
        adapt_records.append(rec)
        student.save(ckpt_pattern.format(epoch=ep + 1))
        head = "  ".join(f"{k}={v:.4f}" for k, v in rec.items())
        print(f"      [epoch {ep + 1}/{epochs}] base={base:.4f} "
              f"bins_ce={bce:.4f} reject_ce={rce:.4f} kd={kdm:.4f} "
              f"combined={combined:.4f} val_loss={row[6]:.4f} | "
              f"[adapt-val] {head}")
    wall = time.time() - t0
    print(f"      trained {epochs} epochs in {wall / 60:.2f} min")
    after = verify_freeze_after(student, snapshot, data)
    after["adapt_val_records"] = adapt_records
    diffs = after["post_training_output_diff_vs_shipped"]
    print(f"      freeze proof: frozen_identical={after['frozen_identical']}"
          f" ({after['frozen_tensors_verified']} tensors, aggregate "
          f"{after['frozen_aggregate_sha256_after'][:16]}...); reject drift "
          f"{diffs['reject']:.2e}; bins drift {diffs['bins']:.4f}")
    write_artifacts(seed, tag, epochs, info, kd_stats, snapshot, after,
                    teacher_p, hist_rows, adapt_records, wall,
                    ckpt_pattern, locks, args.smoke)
    write_config(seed, tag, epochs, info, kd_stats, snapshot, after,
                 teacher_p, wall, ckpt_pattern, locks, args.smoke)


if __name__ == "__main__":
    main()

