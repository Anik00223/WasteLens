# Implementation Plan — src/train.py (WasteLens)

[Overview]
Implement `src/train.py`: a transfer-learning training script that loads the 12-class Kaggle "Garbage Classification" dataset (`asdasdasasdas/garbage-classification`) via kagglehub, remaps it to the 4-bin taxonomy (recyclable / organic / hazardous / general trash), performs a stratified 70/15/15 train/val/test split, builds a MobileNetV2 (frozen) + trainable classification-head model, computes class weights for the ~8:1 imbalance, and prints a summary of per-bin training image counts before stopping — awaiting user confirmation before any training begins.

The scope of THIS task is strictly up to the "print counts and stop" point. Checkpoint saving, fine-tuning of deeper layers, evaluation, and TF.js export are out of scope for this iteration (they live in `src/evaluate.py` / `src/export_tfjs.py` skeletons already present). The script targets CPU-only execution (no GPU detected), so all preprocessing must be lightweight.

[Types]
No formal type system (plain Python + Keras 3). Data structures used:
- `CLASS_TO_BIN: dict[str, str]` — the explicit 12-class → 4-bin remap (source of truth from `docs/dataset_research.md` §6):
  - `cardboard, paper, plastic, metal, brown-glass, green-glass, white-glass` → `recyclable`
  - `biological` → `organic`
  - `battery` → `hazardous`
  - `trash` → `general trash`
  - `clothes, shoes` → dropped entirely (not used in training)
- `BINS: list[str]` — `["recyclable", "organic", "hazardous", "general trash"]` (fixed index order; this exact order becomes the model's output-label order and must match `web/` later)
- `split_paths: dict[str, list[tuple[str, int]]]` — mapping `"train"/"val"/"test"` to a list of `(absolute_image_path, bin_index)` tuples; used to build a `tf.data.Dataset`
- Class weights: `dict[int, float]` — bin index → weight (inverse-frequency, computed from the train split only)

[Files]
New files:
- `C:\Users\dasa8\OneDrive\Desktop\WasteLens\src\train.py` — replaces the current skeleton placeholder comments with the implemented training entry point (up to the pre-training summary)
- `C:\Users\dasa8\OneDrive\Desktop\WasteLens\requirements.txt` — pinned dependencies for reproducibility (TF, kagglehub, scikit-learn, Pillow, numpy)

Existing files to modify: none.

Files to delete/move: none.

Configuration: none (all constants live at the top of `train.py`: `SEED=42`, `IMG_SIZE=(224,224)`, `BATCH_SIZE=32`, `SPLIT=(0.70, 0.15, 0.15)`).

[Functions]
New functions in `C:\Users\dasa8\OneDrive\Desktop\WasteLens\src\train.py`:

1. `download_dataset() -> Path` — uses `kagglehub.dataset_download("asdasdasasdas/garbage-classification")` to fetch/cache the 12-class dataset; returns the local `Path` to the extracted root (contains 12 subfolders: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass).
2. `scan_images(dataset_root: Path) -> list[tuple[Path, str]]` — walks the 12 class subfolders, collects `(image_path, class_name)` for every supported image file (.jpg/.jpeg/.png), filters out `clothes` and `shoes`, and validates the total count (expect ~8,213 across 10 classes).
3. `remap_to_bins(images: list[tuple[Path, str]]) -> list[tuple[Path, str]]` — replaces `class_name` with the 4-bin label via `CLASS_TO_BIN`; raises `ValueError` if any class name is missing from `CLASS_TO_BIN` (guards against dataset layout changes).
4. `stratified_split(images: list[tuple[Path, str]]) -> dict[str, list[tuple[Path, int]]]` — uses `sklearn.model_selection.train_test_split` twice (first 70/30, then 15/15 within the 30) with `stratify=labels` and `random_state=SEED`; converts bin labels to integer indices per `BINS`; returns `{"train": [...], "val": [...], "test": [...]}` of `(path, bin_index)` tuples.
5. `report_split_sizes(splits: dict[str, list[tuple[Path, int]]]) -> None` — prints a table of per-bin image counts for each of train/val/test (expected ~5,586 recyclable / ~985 organic / ~945 hazardous / ~697 general-trash totals).
6. `compute_class_weights(train_labels: list[int]) -> dict[int, float]` — inverse-frequency weighting from the train split only, so val/test remain untouched; prints the weights.
7. `build_model(num_bins: int = 4) -> tuple[keras.Model, keras.Model]` — constructs the frozen-transfer-learning model:
   - Base: `tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")` with `base.trainable = False` (all BatchNorm layers stay in inference mode)
   - Head (the "fine-tuned classification head"): `GlobalAveragePooling2D → Dropout(0.2) → Dense(256, ReLU) → Dropout(0.2) → Dense(4, softmax)`
   - Preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input` applied inside the pipeline (MobileNetV2 expects `[-1, 1]`)
   - Loss: `sparse_categorical_crossentropy`; optimizer: `Adam(1e-3)`; metric: `sparse_categorical_accuracy`
   - Returns `(full_model, base_model)` so a later iteration can unfreeze `base_model` layers for fine-tuning
8. `make_dataset(split: list[tuple[Path, int]], training: bool) -> tf.data.Dataset` — builds a `tf.data.Dataset` from `(path, label)` pairs: decode JPEG/PNG via `tf.io.decode_image`, resize to (224,224), apply `preprocess_input`, shuffle+repeat+prefetch when `training=True`.
9. `main() -> None` — orchestrates: download → scan → remap → split → report split sizes → build model → compute class weights → print per-bin TRAIN image count summary → **stop** (prints "Counts confirmed — run `train.py train` to begin training" and exits; a `train` CLI subcommand, added in a later iteration, is what will actually kick off `model.fit`).

No existing functions to modify (this file was a skeleton). No functions to remove.

[Classes]
No classes — plain functions only, per the existing codebase convention (skeleton `src/*.py` are plain scripts; no OOP patterns anywhere in the repo yet).

[Dependencies]
New packages to install (pinned in `requirements.txt`):
- `tensorflow==2.21.0` — for py3.13 (verified available); brings in Keras 3 (needed for `tf.keras.applications.MobileNetV2`, which requires TF ≥ 2.19 on Keras 3)
- `kagglehub` — for `dataset_download` (no Kaggle credentials needed for public datasets; caches under `%USERPROFILE%\.cache\kagglehub`)
- `scikit-learn==1.8.0` — already installed; pin it
- `pillow==12.1.0`, `numpy==2.4.1` — already installed; pin them

Integration: no changes to `web/` (TF.js conversion is a separate later step via `src/export_tfjs.py`).

[Testing]
- No formal unit-test framework exists in this repo; use lightweight inline assertions and print-based validation inside `train.py` (e.g., assert total scanned image count == 8,213 ± 1%; assert `len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(remapped_images)`; assert all 4 bins are present in train/val/test).
- Validation strategy for THIS task: run `python src/train.py` once — it must complete up to the pre-training summary and exit without invoking `model.fit`; visually confirm the printed per-bin split table matches the counts derived in `docs/dataset_research.md` §5.
- Later (out of scope for this iteration): `src/evaluate.py` produces the real per-class precision/recall/F1 table; no `tests/` folder planned yet.

[Implementation Order]
1. Confirm environment: install `tensorflow==2.21.0` and `kagglehub` into Python 3.13 (`py -3.13 -m pip install ...`), verify `import tensorflow` succeeds.
2. Write `requirements.txt` pinning all deps.
3. Replace `src/train.py` skeleton with the implemented script (constants → CLASS_TO_BIN → functions 1–9 → `if __name__ == "__main__": main()`).
4. Run `python src/train.py` end-to-end once — confirm kagglehub download, scan/remap validation, split, model build, class weights, and pre-training summary all print correctly, and that it **stops without training**.
5. Commit and push (`git add -A; git commit -m "Implement train.py: data loading, remap, stratified split, model build (pre-training)"; git push origin main`).
6. Present the printed per-bin counts/split table to the user and await explicit confirmation before adding the `model.fit` training step.

