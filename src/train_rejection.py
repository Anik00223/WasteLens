# src/train_rejection.py
#
# PURPOSE (Iteration 4):
#   Build the leakage-controlled rejection dataset, then train the dual-head
#   model: unchanged 4-bin head ("which bin?") + new sigmoid rejection head
#   ("is this image within WasteLens's supported scope?").
#
# Head architecture (smallest change):
#   image -> MobileNetV2 (frozen) -> gap -> dropout_1 -> fc_256 -> dropout_2
#     -> predictions: Dense(4, softmax)          [unchanged layer names/shape]
#     -> reject:      Dense(1, sigmoid)          [new, taps dropout_2 output]
#   Backbone frozen exactly as the baseline; optimizer Adam(1e-3) and batch
#   size 32 unchanged; sparse-CE for bins + binary-CE for reject, both weight
#   1.0. Keras class_weight cannot be used with two outputs (weights would be
#   mis-shared), so per-output sample_weight arrays implement the same
#   inverse-frequency weighting for the bins head (0.0 weight for unsupported
#   rows, which have no valid bin target).
#
# Data (all splits deterministic, seeds fixed in constants below):
#   supported   : train.py's UNCHANGED seed-42 stratified split (no rebuild).
#   clothes/shoes: per-class 70/15/15 by file; md5-deduped against the whole
#                  supported set and within-source before splitting.
#   collages    : built ONLY from train-split waste photos (train and val
#                 collages from DISJOINT source pools); test collages from
#                 TEST-split photos only -> zero image leakage across splits.
#   non-waste   : CIFAR-10 photos (train batches 1-4 -> train+val; official
#                 test batch -> test; official disjointness, no overlap).
#   synthetic probes: EVALUATION ONLY (never trained on).
#
# Threshold calibration data is the VAL split; the TEST split is untouched
# by both training and calibration (enforced by construction in this script).
#
# Run from the repo root:  python src/train_rejection.py [--smoke]
#   --smoke: 1-epoch / tiny-subset plumbing check (no artifacts kept).

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

import train as wl  # single source of truth: bins, split (seed=42), pipeline

BASE_CHECKPOINT = Path("models/checkpoints/wastelens_ep09.keras")
OUT_DIR = Path("docs/rejection_experiment")
CKPT_DIR = Path("models/checkpoints")
EVAL_SETS_JSON = OUT_DIR / "eval_sets.json"
MANIFEST_JSON = OUT_DIR / "dataset_manifest.json"
CONFIG_JSON = OUT_DIR / "training_config.json"
CSV_LOG = OUT_DIR / "training_log.csv"
DATA_TMP = Path(tempfile.gettempdir()) / "wastelens_rejection_data"

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TRAIN_BATCHES = ["data_batch_1", "data_batch_2", "data_batch_3",
                       "data_batch_4"]          # official train batches
CIFAR_TEST_BATCHES = ["test_batch"]               # official test batch

SEED = 42                    # dataset construction (splits stay seed=42)
EPOCHS = 10                  # baseline-faithful
COLLAGE_TRAIN = [(2, 160), (4, 120), (6, 80)]    # 360
COLLAGE_VAL = [(2, 40), (4, 30), (6, 20)]        # 90
COLLAGE_TEST = [(2, 40), (4, 30), (6, 20)]       # 90
CIFAR_TRAIN_PER_CLASS = 300  # x10 classes = 3,000
CIFAR_VAL_PER_CLASS = 75     # x10 = 750
CIFAR_TEST_PER_CLASS = 75    # x10 = 750


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in wl.IMG_EXTENSIONS
                  and not p.name.startswith("."))


def split_70_15_15(items: list, seed: int) -> tuple[list, list, list]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_tr = int(0.70 * len(items))
    n_va = int(0.15 * len(items))
    tr = [items[i] for i in order[:n_tr]]
    va = [items[i] for i in order[n_tr:n_tr + n_va]]
    te = [items[i] for i in order[n_tr + n_va:]]
    return tr, va, te


def dedupe(paths, seen: set, dropped: list) -> list:
    """Drop files whose md5 is already in `seen`; add survivors to it."""
    kept = []
    for p in paths:
        h = md5_file(p)
        if h in seen:
            dropped.append(str(p))
            continue
        seen.add(h)
        kept.append(p)
    return kept


# --- CIFAR-10 (source B: non-waste objects) ----------------------------------

def ensure_cifar(cache_root: Path) -> Path:
    target = cache_root / "cifar-10-batches-py"
    if (target / CIFAR_TEST_BATCHES[0]).exists():
        return target
    cache_root.mkdir(parents=True, exist_ok=True)
    tar_path = cache_root / "cifar-10-python.tar.gz"
    expected_bytes = 170_498_071
    if not (tar_path.exists() and tar_path.stat().st_size >= expected_bytes):
        if tar_path.exists():
            tar_path.unlink()  # partial archive - never trust it
        print(f"      downloading CIFAR-10 (~{expected_bytes // 1_000_000} MB)...")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        actual = tar_path.stat().st_size
        if actual < expected_bytes:
            raise RuntimeError(
                f"CIFAR-10 archive incomplete: {actual} < {expected_bytes} bytes")
    with tarfile.open(tar_path) as tar:
        tar.extractall(cache_root)
    return target


def load_cifar_batch(data_dir: Path, name: str):
    with open(data_dir / name, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return data, np.asarray(d[b"labels"])


def cifar_to_pngs(data_dir: Path, batches: list[str], n_per_class: int,
                  seed: int, out_dir: Path, tag: str) -> list[Path]:
    """Deterministically select n_per_class/class and save as 32x32 PNGs.

    Resize to 224x224 happens inside the standard tf.data load path, exactly
    like any other non-224 image (same preprocessing contract).
    """
    rng = np.random.RandomState(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for batch in batches:
        data, labels = load_cifar_batch(data_dir, batch)
        for c in range(10):
            idx = np.where(labels == c)[0]
            idx = idx[rng.permutation(len(idx))][:n_per_class]
            for n, i in enumerate(idx):
                # Tag includes the source batch so train/val/test selections
                # can never collide on filenames (batch-disjoint=file-disjoint).
                p = out_dir / f"cifar_{tag}_{batch}_{c}_{n:04d}.png"
                if not p.exists():
                    Image.fromarray(data[i]).save(p)
                paths.append(p)
    return paths


# --- Collages (source C: multi-object scenes) --------------------------------

def build_collages(pool_rows: list, plan: list, seed: int, out_dir: Path,
                   tag: str) -> list:
    """Deterministic multi-item collages; pools MUST be pre-split disjoint."""
    rng = np.random.RandomState(seed)
    per_bin: dict[int, list[Path]] = {i: [] for i in range(len(wl.BINS))}
    for path, lbl in pool_rows:
        per_bin[lbl].append(path)
    for k in per_bin:
        per_bin[k].sort()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for items, count in plan:
        for c in range(count):
            w, h = 480, 360
            canvas = Image.new("RGB", (w, h), tuple(
                int(255 * v) for v in (rng.rand() * .6 + .2,
                                       rng.rand() * .6 + .2,
                                       rng.rand() * .6 + .2)))
            draw = ImageDraw.Draw(canvas)
            for _ in range(3):
                x0, y0 = rng.randint(0, w - 40), rng.randint(0, h - 40)
                draw.rectangle([x0, y0, x0 + rng.randint(30, 160),
                                y0 + rng.randint(30, 120)],
                               fill=tuple(int(255 * rng.rand()) for _ in range(3)))
            bins = np.resize(rng.permutation(len(wl.BINS)), items)
            for b in bins:
                src = per_bin[int(b)][rng.randint(len(per_bin[int(b)]))]
                img = Image.open(src).convert("RGB")
                iw = rng.randint(w // 4, w // 2)
                img = img.resize((iw, int(iw * img.height / img.width)))
                canvas.paste(img, (rng.randint(0, max(1, w - img.width)),
                                   rng.randint(0, max(1, h - img.height))))
            dest = out_dir / f"collage_{tag}_{items}item_{c:03d}.jpg"
            canvas.save(dest, quality=90)
            rows.append((dest, f"collage{items}"))
    return rows


# --- Dataset assembly --------------------------------------------------------

def build_datasets() -> dict:
    """Leakage-controlled dataset + manifest. See module docstring."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_TMP.mkdir(parents=True, exist_ok=True)
    dataset_root = wl.download_dataset()
    images = wl.remap_to_bins(wl.scan_images(dataset_root))
    splits = wl.stratified_split(images)          # UNCHANGED seed-42 split
    class_weights = wl.compute_class_weights(splits["train"])
    class_dirs = wl.find_class_dirs(dataset_root)

    # md5 set of ALL supported images (dedupe guard for every new source)
    print("      hashing supported images for the dedupe guard ...")
    seen = {md5_file(p) for p, _ in images}
    dropped: list[str] = []

    # --- source A: dropped classes (clothes/shoes) ---------------------------
    unsup = {}
    manifest = [{
        "source": "kaggle mostafaabla/garbage-classification (12-class)",
        "class": "4-bin taxonomy (10 classes mapped, clothes/shoes dropped)",
        "mapped_label": "bins 0-3 (supported)",
        "supported": True,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "seed": 42,
        "note": "UNCHANGED train.py pipeline; identical membership to the "
                "baseline model's data",
    }]
    for cls in ("clothes", "shoes"):
        files = dedupe(source_files(class_dirs[cls]), seen, dropped)
        tr, va, te = split_70_15_15(files, SEED)
        unsup[cls] = {"train": [(p, cls) for p in tr],
                      "val": [(p, cls) for p in va],
                      "test": [(p, cls) for p in te]}
        manifest.append({
            "source": "kaggle mostafaabla/garbage-classification (12-class)",
            "class": cls, "mapped_label": "unsupported",
            "supported": False,
            "split_counts": {"train": len(tr), "val": len(va),
                             "test": len(te)},
            "seed": SEED,
            "note": "md5-deduped against ALL supported images + within-source "
                    f"({len(dropped)} duplicates dropped so far)",
        })

    # --- source C: collages from DISJOINT source pools -----------------------
    rng = np.random.RandomState(SEED)
    tr_rows = sorted(splits["train"])
    rng.shuffle(tr_rows)
    pool_ctr = tr_rows[:1200]                 # train-collage sources
    pool_ctv = tr_rows[1200:1800]             # val-collage sources (disjoint)
    te_rows = sorted(splits["test"])
    rng.shuffle(te_rows)
    pool_cte = te_rows[:500]                  # test-collage sources (test split)
    c_tr = build_collages(pool_ctr, COLLAGE_TRAIN, SEED, DATA_TMP / "coll",
                          "tr")
    c_va = build_collages(pool_ctv, COLLAGE_VAL, SEED + 1, DATA_TMP / "coll",
                          "va")
    c_te = build_collages(pool_cte, COLLAGE_TEST, SEED + 2, DATA_TMP / "coll",
                          "te")
    unsup["collage"] = {"train": c_tr, "val": c_va, "test": c_te}
    manifest.append({
        "source": "synthetic composition of kaggle waste photos",
        "class": "collages (2/4/6 items on cluttered background)",
        "mapped_label": "unsupported", "supported": False,
        "split_counts": {"train": len(c_tr), "val": len(c_va),
                         "test": len(c_te)},
        "seed": SEED,
        "note": "train/val collages built from DISJOINT train-split source "
                "pools; test collages from TEST-split sources only",
    })

    # --- source B: non-waste objects (CIFAR-10) ------------------------------
    # Train/val/test use DISJOINT official batches: batches 1-3 -> train
    # (100/class/batch = 300/class), batch 4 -> val (75/class), official
    # test_batch -> test (75/class). One call per pool keeps the per-class
    # selections batch-pure; filenames carry the batch tag so file-level
    # disjointness is auditable (and asserted below by content hash).
    cifar_dir = ensure_cifar(DATA_TMP / "cifar")
    nw_tr = [(p, "nonwaste") for p in cifar_to_pngs(
        cifar_dir, ["data_batch_1", "data_batch_2", "data_batch_3"],
        100, SEED, DATA_TMP / "cifar_pngs", "tr")]
    nw_va = [(p, "nonwaste") for p in cifar_to_pngs(
        cifar_dir, ["data_batch_4"], CIFAR_VAL_PER_CLASS,
        SEED + 1, DATA_TMP / "cifar_pngs", "va")]
    nw_te = cifar_to_pngs(cifar_dir, CIFAR_TEST_BATCHES, CIFAR_TEST_PER_CLASS,
                          SEED + 3, DATA_TMP / "cifar_pngs", "te")
    # Content-hash leak audit: no identical file may appear in two splits
    # (catches accidental batch/tag reuse even if filenames differ).
    _audit = {}
    for _tag, _rows in (("train", nw_tr), ("val", nw_va)):
        for _p, _ in _rows:
            _h = md5_file(_p)
            assert _h not in _audit, f"CIFAR split leak: {_p} duplicates train/val content"
            _audit[_h] = _tag
    for _p in nw_te:
        assert md5_file(_p) not in _audit, f"CIFAR test leaks into train/val: {_p}"
    unsup["nonwaste"] = {"train": nw_tr,
                         "val": nw_va,
                         "test": [(p, "nonwaste") for p in nw_te]}
    manifest.append({
        "source": "CIFAR-10 (cs.toronto.edu)",
        "class": "airplane/automobile/bird/cat/deer/dog/frog/horse/"
                 "ship/truck",
        "mapped_label": "unsupported", "supported": False,
        "split_counts": {"train": len(nw_tr), "val": len(nw_va),
                         "test": len(nw_te)},
        "seed": SEED,
        "note": "official disjoint batches (train 1-4 vs test); train+val "
                "selected from train batches in one pass then sliced 300/75 "
                "per class",
    })
    manifest.append({
        "source": "deterministic synthetic patterns (validate_realworld.py)",
        "class": "10 probes", "mapped_label": "unsupported (EVAL ONLY)",
        "supported": False, "split_counts": {"train": 0, "val": 0, "test": 10},
        "seed": "n/a", "note": "never trained on; regression set only",
    })
    return {"splits": splits, "unsup": unsup, "class_weights": class_weights,
            "manifest": manifest, "dropped_dupes": dropped}


# --- Training arrays and tf.data ---------------------------------------------

def assemble(data: dict, split_name: str, smoke: bool = False):
    """Pack one split into (paths, y_bins, y_rej, sw_bins, sw_rej, n_s, n_u)."""
    sp, unsup, cw = data["splits"], data["unsup"], data["class_weights"]
    rows_s = sorted(sp[split_name])
    rows_u = []
    for src in ("clothes", "shoes", "collage", "nonwaste"):
        rows_u.extend(unsup[src][split_name])
    if smoke:
        rows_s = rows_s[:96]
        rows_u = rows_u[:96]
    n_s, n_u = len(rows_s), len(rows_u)
    n = n_s + n_u
    w_s = n / (2.0 * max(n_s, 1))   # inverse-frequency weights for the
    w_u = n / (2.0 * max(n_u, 1))   # rejection head (bins head keeps its own)
    paths = [str(p) for p, _ in rows_s] + [str(p) for p, _ in rows_u]
    y_bins = [lbl for _, lbl in rows_s] + [0] * n_u      # weight-0 rows
    y_rej = [0] * n_s + [1] * n_u
    sw_bins = [cw[lbl] for _, lbl in rows_s] + [0.0] * n_u
    sw_rej = [w_s] * n_s + [w_u] * n_u
    return (np.array(paths), np.array(y_bins, np.int32),
            np.array(y_rej, np.float32), np.array(sw_bins, np.float32),
            np.array(sw_rej, np.float32), n_s, n_u)


def make_weighted_ds(paths, y_bins, y_rej, sw_bins, sw_rej,
                     training: bool) -> tf.data.Dataset:
    """Same image pipeline as wl.make_dataset + per-output sample weights."""
    ds = tf.data.Dataset.from_tensor_slices(
        (paths, y_bins, y_rej, sw_bins, sw_rej))
    if training:
        ds = ds.shuffle(len(paths), seed=wl.SEED, reshuffle_each_iteration=True)

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


# --- Dual-head model ----------------------------------------------------------

def build_dual_model(base_checkpoint: Path = BASE_CHECKPOINT):
    """Baseline model + rejection head on the shared fc_256 representation."""
    base = tf.keras.models.load_model(base_checkpoint)
    drop2 = base.get_layer("dropout_2").output
    rej = tf.keras.layers.Dense(1, activation="sigmoid", name="reject")(drop2)
    model = tf.keras.models.Model(base.input,
                                  {"bins": base.output, "reject": rej},
                                  name="wastelens_reject")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"bins": "sparse_categorical_crossentropy",
              "reject": "binary_crossentropy"},
        loss_weights={"bins": 1.0, "reject": 1.0},
        metrics={"bins": ["sparse_categorical_accuracy"],
                 "reject": ["binary_accuracy"]})
    return model, base


# --- Orchestration ------------------------------------------------------------

def write_manifest(data: dict) -> None:
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "supported_split_seed": 42,
        "unsupported_split_seed": SEED,
        "dedupe": {"method": "md5 file hash",
                   "checked_against": "all 8,213 supported images + within-source",
                   "duplicates_dropped": len(data["dropped_dupes"]),
                   "examples": data["dropped_dupes"][:20]},
        "sources": data["manifest"],
        "rejection_head": {
            "input": "dropout_2 output (shared fc_256 representation)",
            "output": "sigmoid P(unsupported)",
            "bins_head": "unchanged Dense(4, softmax), label order = train.BINS",
        },
    }
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8")
    print(f"      manifest -> {MANIFEST_JSON}")


def write_eval_sets(data: dict) -> None:
    eval_sets = {
        "supported_test": [[str(p), int(l)]
                           for p, l in sorted(data["splits"]["test"])],
        "supported_val": [str(p) for p, _ in sorted(data["splits"]["val"])],
        "val_unsup": {src: [str(p) for p, _ in sorted(
            data["unsup"][src]["val"], key=lambda r: str(r[0]))]
            for src in data["unsup"]},
        "test_unsup": {src: [str(p) for p, _ in sorted(
            data["unsup"][src]["test"], key=lambda r: str(r[0]))]
            for src in data["unsup"]},
    }
    EVAL_SETS_JSON.write_text(json.dumps(eval_sets, indent=1) + "\n",
                              encoding="utf-8")
    print(f"      eval sets -> {EVAL_SETS_JSON}")


def train_model(data: dict, smoke: bool = False):
    tr = assemble(data, "train", smoke)
    va = assemble(data, "val", smoke)
    print(f"      train: {tr[5]} supported + {tr[6]} unsupported = "
          f"{len(tr[0])} images ({math.ceil(len(tr[0]) / wl.BATCH_SIZE)} steps)")
    print(f"      val:   {va[5]} supported + {va[6]} unsupported = "
          f"{len(va[0])} images")
    train_ds = make_weighted_ds(*tr[:5], training=True)
    val_ds = make_weighted_ds(*va[:5], training=False)

    model, base = build_dual_model()
    backbone = base.get_layer("mobilenetv2_1.00_224")
    assert all(not w.trainable for w in backbone.weights), \
        "backbone must stay frozen (baseline contract)"
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"      trainable params: {trainable:,} (heads only)")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(CKPT_DIR / "wastelens_rej_ep{epoch:02d}.keras"),
            save_freq="epoch", verbose=0),
        # Best-by-val_loss, mirroring the baseline convention (ep09 was best
        # by val_loss). Also the default --model of src/eval_rejection.py.
        tf.keras.callbacks.ModelCheckpoint(
            str(CKPT_DIR / "wastelens_rej_best.keras"),
            monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(CSV_LOG), append=False),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                         patience=3),
    ]
    config = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "base_checkpoint": str(BASE_CHECKPOINT),
        "epochs": 1 if smoke else EPOCHS, "batch_size": wl.BATCH_SIZE,
        "optimizer": "Adam(1e-3) [unchanged from baseline]",
        "losses": {"bins": "sparse_categorical_crossentropy",
                   "reject": "binary_crossentropy"},
        "loss_weights": {"bins": 1.0, "reject": 1.0},
        "bins_class_weights": data["class_weights"],
        "weighting_note": "per-output sample_weight (class_weight cannot be "
                          "shared across outputs); bins head gets the same "
                          "inverse-frequency weights as the baseline, 0.0 on "
                          "unsupported rows",
        "trainable_params": trainable,
    }
    CONFIG_JSON.write_text(json.dumps(config, indent=2) + "\n",
                           encoding="utf-8")

    t0 = time.time()
    history = model.fit(train_ds, steps_per_epoch=math.ceil(len(tr[0])
                                                           / wl.BATCH_SIZE),
                        validation_data=val_ds,
                        epochs=1 if smoke else EPOCHS,
                        callbacks=callbacks, verbose=2)
    print(f"      fit wall time: {time.time() - t0:.0f}s")
    (OUT_DIR / "history.json").write_text(
        json.dumps({k: [float(x) for x in v]
                    for k, v in history.history.items()}, indent=2) + "\n",
        encoding="utf-8")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="1-epoch tiny-subset plumbing check")
    args = parser.parse_args()
    print("=== WasteLens rejection-head experiment: dataset + training ===\n")
    data = build_datasets()
    write_manifest(data)
    write_eval_sets(data)
    print(f"[train] building dual model from {BASE_CHECKPOINT} ...")
    train_model(data, smoke=args.smoke)
    print("\nDone. Evaluate with: python src/eval_rejection.py")


if __name__ == "__main__":
    main()




