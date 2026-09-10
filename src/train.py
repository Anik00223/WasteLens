# src/train.py
#
# PURPOSE:
#   Train the WasteLens classifier using transfer learning.
#
#   - Loads a pretrained backbone (MobileNetV2, ImageNet weights).
#   - Freezes the backbone and trains a new classification head on the
#     waste categories (recyclable / organic / hazardous / general trash).
#   - Optionally fine-tunes the top layers of the backbone.
#   - Saves checkpoints to models/checkpoints/ and the final Keras model
#     for conversion to TensorFlow.js.
#
# STATUS: Implemented up to the pre-training summary (data load, remap,
#   stratified split, model build, class weights). `model.fit` is NOT
#   called yet - training starts only after explicit user confirmation
#   of the printed split counts.

"""WasteLens training script.

Run `python src/train.py` to:
  1. download the 12-class Kaggle dataset (kagglehub, cached),
  2. remap its classes to the 4-bin taxonomy,
  3. build a stratified 70/15/15 train/val/test split,
  4. build the frozen MobileNetV2 + head model,
  5. print per-bin counts and class weights,
then STOP before any training (model.fit is intentionally not called).
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# --- Configuration -----------------------------------------------------------

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SPLIT = (0.70, 0.15, 0.15)  # train / val / test fractions
KAGGLE_HANDLE = "mostafaabla/garbage-classification"
# NOTE: 12-class Kaggle set ("Garbage Classification (12 classes)"). The
# originally researched handle asdasdasasdas/garbage-classification replaced
# its files with a 6-class TrashNet variant (versions/2) and versioned
# anonymous downloads 404, so this verified 12-class mirror is used instead.
# Counts are re-calibrated per-class in EXPECTED_PER_CLASS on first scan.
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Fixed bin order - this is the model's output-label order and must match
# the web demo's label order later (see export_tfjs.py).
BINS = ["recyclable", "organic", "hazardous", "general trash"]
BIN_TO_IDX = {name: idx for idx, name in enumerate(BINS)}

# 12 Kaggle classes -> 4-bin taxonomy (docs/dataset_research.md, section 6).
CLASS_TO_BIN = {
    "cardboard": "recyclable",
    "paper": "recyclable",
    "plastic": "recyclable",
    "metal": "recyclable",
    "brown-glass": "recyclable",
    "green-glass": "recyclable",
    "white-glass": "recyclable",
    "biological": "organic",
    "battery": "hazardous",
    "trash": "general trash",
}

# Classes intentionally excluded (arguable bin labels; 7,302 images).
DROPPED_CLASSES = {"clothes", "shoes"}

# Expected image counts per kept class (verified in docs/dataset_research.md).
EXPECTED_PER_CLASS = {
    "battery": 945,
    "biological": 985,
    "brown-glass": 607,
    "cardboard": 891,
    "green-glass": 629,
    "metal": 769,
    "paper": 1050,
    "plastic": 865,
    "trash": 697,
    "white-glass": 775,
}
EXPECTED_TOTAL = sum(EXPECTED_PER_CLASS.values())  # 8,213

PREPROCESS_INPUT = tf.keras.applications.mobilenet_v2.preprocess_input


# --- Data loading ------------------------------------------------------------

def download_dataset() -> Path:
    """Download (or fetch from cache) the 12-class dataset; return its root."""
    print(f"[1/6] Downloading dataset via kagglehub: {KAGGLE_HANDLE}")
    root = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    print(f"      cached at: {root}")
    return root


def find_class_dirs(dataset_root: Path) -> dict[str, Path]:
    """Locate the class subdirectories anywhere below the dataset root."""
    wanted = set(CLASS_TO_BIN) | DROPPED_CLASSES
    found: dict[str, Path] = {}
    for path in sorted(dataset_root.rglob("*")):
        if path.is_dir() and path.name in wanted and path.name not in found:
            found[path.name] = path
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(
            f"Dataset layout changed: class folders not found: {missing}. "
            f"First directories under root: "
            f"{sorted(p.name for p in dataset_root.iterdir() if p.is_dir())[:30]}"
        )
    return found


def scan_images(dataset_root: Path) -> list[tuple[Path, str]]:
    """Collect (image_path, class_name) for the 10 kept classes."""
    print("[2/6] Scanning image files (dropping classes: clothes, shoes)")
    class_dirs = find_class_dirs(dataset_root)
    images: list[tuple[Path, str]] = []
    per_class: dict[str, int] = {}
    for class_name in sorted(CLASS_TO_BIN):
        class_dir = class_dirs[class_name]
        files = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMG_EXTENSIONS
            and not p.name.startswith(".")
        )
        per_class[class_name] = len(files)
        images.extend((p, class_name) for p in files)

    print("      per-class scan:")
    for class_name in sorted(per_class):
        expected = EXPECTED_PER_CLASS[class_name]
        marker = "ok" if per_class[class_name] == expected else "MISMATCH"
        print(
            f"        {class_name:<12} {per_class[class_name]:>5}  "
            f"(expected {expected:>5})  {marker}"
        )
    drift = abs(len(images) - EXPECTED_TOTAL) / EXPECTED_TOTAL
    if drift > 0.01:
        raise RuntimeError(
            f"Scanned {len(images)} images; expected ~{EXPECTED_TOTAL} (+/-1%). "
            "Dataset may have changed - investigate before training."
        )
    print(f"      scanned total: {len(images)} (expected {EXPECTED_TOTAL})  ok")
    return images


def remap_to_bins(images: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    """Replace the 12-class names with the 4-bin labels."""
    print("[3/6] Remapping 12 classes -> 4 bins")
    remapped: list[tuple[Path, str]] = []
    for path, class_name in images:
        bin_name = CLASS_TO_BIN.get(class_name)
        if bin_name is None:
            raise ValueError(f"Unknown class '{class_name}' - not in CLASS_TO_BIN")
        remapped.append((path, bin_name))
    counts = {b: 0 for b in BINS}
    for _, bin_name in remapped:
        counts[bin_name] += 1
    for bin_name in BINS:
        print(f"        {bin_name:<14} {counts[bin_name]:>5}")
    return remapped

# --- Splitting ---------------------------------------------------------------

def stratified_split(
    images: list[tuple[Path, str]],
) -> dict[str, list[tuple[Path, int]]]:
    """Stratified 70/15/15 split -> (path, bin_index) tuples per split."""
    print("[4/6] Stratified split (70/15/15, stratified by bin, seed=42)")
    train_ratio, val_ratio, _ = SPLIT
    paths = [str(p) for p, _ in images]
    labels = [b for _, b in images]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=1.0 - train_ratio,
        stratify=labels,
        random_state=SEED,
    )
    val_fraction = val_ratio / (1.0 - train_ratio)  # 0.5 of the temp pool
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=1.0 - val_fraction,
        stratify=temp_labels,
        random_state=SEED,
    )

    splits = {
        "train": [(Path(p), BIN_TO_IDX[b]) for p, b in zip(train_paths, train_labels)],
        "val": [(Path(p), BIN_TO_IDX[b]) for p, b in zip(val_paths, val_labels)],
        "test": [(Path(p), BIN_TO_IDX[b]) for p, b in zip(test_paths, test_labels)],
    }

    assert sum(len(v) for v in splits.values()) == len(images), "split lost images"
    all_bins = set(range(len(BINS)))
    for split_name, rows in splits.items():
        present = {bin_idx for _, bin_idx in rows}
        assert present == all_bins, f"{split_name} missing bins: {all_bins - present}"
    return splits


def report_split_sizes(splits: dict[str, list[tuple[Path, int]]]) -> None:
    """Print per-bin image counts for train/val/test."""
    print("\n  Per-bin split sizes (images):")
    header = f"    {'bin':<15}{'train':>8}{'val':>8}{'test':>8}{'total':>9}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for bin_idx, bin_name in enumerate(BINS):
        row = [
            sum(1 for _, lbl in splits[name] if lbl == bin_idx)
            for name in ("train", "val", "test")
        ]
        print(f"    {bin_name:<15}{row[0]:>8}{row[1]:>8}{row[2]:>8}{sum(row):>9}")
    totals = [len(splits[name]) for name in ("train", "val", "test")]
    print(f"    {'TOTAL':<15}{totals[0]:>8}{totals[1]:>8}{totals[2]:>8}{sum(totals):>9}")


def compute_class_weights(train_rows: list[tuple[Path, int]]) -> dict[int, float]:
    """Inverse-frequency class weights, computed from the TRAIN split only."""
    labels = [lbl for _, lbl in train_rows]
    counts = np.bincount(np.asarray(labels), minlength=len(BINS))
    total = int(counts.sum())
    weights = {
        idx: float(total / (len(BINS) * counts[idx])) for idx in range(len(BINS))
    }
    print("\n  Class weights (inverse frequency, train split only):")
    for idx, bin_name in enumerate(BINS):
        print(f"    {bin_name:<15} count={counts[idx]:>5}  weight={weights[idx]:.3f}")
    return weights

# --- Model -------------------------------------------------------------------

def build_model(num_bins: int = 4) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Frozen MobileNetV2 (ImageNet) + trainable classification head."""
    print("[5/6] Building model: MobileNetV2 (frozen) + 4-bin head")
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # frozen backbone; BatchNorm stays in inference mode

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc_256")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = tf.keras.layers.Dense(
        num_bins, activation="softmax", name="predictions"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="wastelens_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model, base


def make_dataset(rows: list[tuple[Path, int]], training: bool) -> tf.data.Dataset:
    """tf.data pipeline: decode -> resize(224x224) -> MobileNetV2 preprocess."""

    def load(path: tf.Tensor, label: tf.Tensor):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = PREPROCESS_INPUT(img)
        return img, label

    paths = [str(p) for p, _ in rows]
    labels = [lbl for _, lbl in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


# --- Orchestration -----------------------------------------------------------

def main() -> None:
    print("=== WasteLens training - pre-training stage (no model.fit) ===\n")
    dataset_root = download_dataset()
    images = scan_images(dataset_root)
    binned = remap_to_bins(images)
    splits = stratified_split(binned)
    report_split_sizes(splits)

    model, _base = build_model(num_bins=len(BINS))

    # Build the train pipeline and decode one batch as a shape/preprocess check.
    train_ds = make_dataset(splits["train"], training=True)
    for batch_images, batch_labels in train_ds.take(1):
        lo = float(batch_images.numpy().min())
        hi = float(batch_images.numpy().max())
        print(
            f"      pipeline check - batch: {batch_images.shape}, "
            f"labels: {batch_labels.shape}, value range [{lo:.2f}, {hi:.2f}] "
            f"(MobileNetV2 expects [-1, 1])"
        )

    class_weights = compute_class_weights(splits["train"])

    try:
        model.summary(line_length=100, show_trainable=True)
    except TypeError:
        model.summary(line_length=100)

    # [6/6] Pre-training summary, then STOP before model.fit.
    train_counts = {
        bin_name: sum(
            1 for _, lbl in splits["train"] if lbl == BIN_TO_IDX[bin_name]
        )
        for bin_name in BINS
    }
    print("[6/6] Pre-training summary")
    print("  =======================================")
    print(f"  total usable images : {len(binned)} "
          "(12 classes -> 4 bins, clothes/shoes dropped)")
    print(f"  split               : 70/15/15 stratified (seed={SEED})")
    print(f"  train images        : {len(splits['train'])}  "
          + ", ".join(f"{b}={train_counts[b]}" for b in BINS))
    print("  class weights       : "
          + ", ".join(f"{BINS[i]}={class_weights[i]:.2f}" for i in range(len(BINS))))
    print("  model               : MobileNetV2 (frozen, ImageNet) + "
          "GAP -> Dense(256) -> Dense(4, softmax)")
    print(f"  pipeline            : 224x224, batch={BATCH_SIZE}, "
          "preprocess=[-1, 1]")
    print("  =======================================")
    print("\nStopped BEFORE model.fit as planned - nothing has been trained.")
    print("Confirm the counts and split above; the training step will be "
          "added after your go-ahead.")


if __name__ == "__main__":
    main()



