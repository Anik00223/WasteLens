# src/make_parity_inputs.py
#
# PURPOSE:
#   Build a deterministic input set for the Python(Keras) <-> TensorFlow.js
#   parity check of the shipped dual-head model, and record the Keras outputs
#   for exactly those inputs.
#
#   Inputs are stored ALREADY PREPROCESSED to the model's input space
#   ([-1, 1], i.e. the result of resize 224x224 + (x/127.5)-1) so the TF.js
#   harness consumes byte-identical tensors - parity then measures only
#   inference-numerics differences, not preprocessing differences.
#
#   Input groups:
#   - 5 synthetic images built in code (zeros, checkerboard, gradient,
#     seeded gaussian noise, flat mid-gray) - always available.
#   - up to 2*N real images drawn with a fixed stride from the Iteration-4
#     eval sets (supported_test + each val_unsup group in
#     docs/rejection_experiment/eval_sets.json), captured from the SAME
#     tf.data preprocessing used by src/eval_rejection.py.
#
# Run from the repo root:
#   py -3.13 -W ignore src/make_parity_inputs.py --model <ckpt> --out <dir>
#
# Outputs (in --out):
#   inputs.f32   raw little-endian float32, C-order, N x 224 x 224 x 3
#   parity.json  {model, created_utc, n, input_shape, input_names, keras:{bins, reject}}

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

import train_rejection as tr

EVAL_SETS_JSON = Path("docs/rejection_experiment/eval_sets.json")


def build_synthetic_inputs() -> tuple[list[str], np.ndarray]:
    """5 deterministic images built directly in the model's [-1,1] input space."""
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    zeros = np.full((224, 224, 3), -1.0, np.float32)
    checker = ((((yy // 16) + (xx // 16)) % 2).astype(np.float32) * 2.0) - 1.0
    checker = np.stack([checker] * 3, axis=-1)
    grad = np.stack([(xx / 223.0) * 2.0 - 1.0] * 3, axis=-1).astype(np.float32)
    rng = np.random.default_rng(42)
    noise = np.clip(rng.normal(0.0, 0.5, (224, 224, 3)), -1.0, 1.0).astype(np.float32)
    flat = np.full((224, 224, 3), 0.1, np.float32)
    names = ["synth_zeros", "synth_checkerboard16", "synth_h_gradient",
             "synth_gauss_noise_seed42", "synth_flat_0.1"]
    return names, np.stack([zeros, checker, grad, noise, flat]).astype(np.float32)


def select_real_inputs(data: dict, per_group: int) -> list[dict]:
    """Deterministic stride sample from supported_test and each val_unsup group."""
    picked: list[dict] = []
    sup = sorted(data["supported_test"], key=lambda r: r[0])
    stride = max(1, len(sup) // per_group)
    for r in sup[::stride][:per_group]:
        picked.append({"name": f"real_supported_{Path(r[0]).name}", "path": r[0]})
    groups = sorted(data["val_unsup"])
    per_src = max(1, per_group // max(1, len(groups)))
    for src in groups:
        paths = sorted(data["val_unsup"][src])
        stride = max(1, len(paths) // per_src)
        for p in paths[::stride][:per_src]:
            picked.append({"name": f"real_{src}_{Path(p).name}", "path": p})
    return [item for item in picked if Path(item["path"]).exists()]


def preprocess_paths(paths: list[str]) -> np.ndarray:
    """The exact eval-time image pipeline (same as src/eval_rejection.py)."""
    n = len(paths)
    ds = tr.make_weighted_ds(
        np.array(paths), np.zeros(n, np.int32), np.zeros(n, np.float32),
        np.ones(n, np.float32), np.ones(n, np.float32), training=False)
    return np.concatenate(
        [x.numpy() for x in ds.map(lambda x, y, w: x)]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build parity inputs + Keras outputs")
    ap.add_argument("--model", type=Path,
                    default=Path("models/checkpoints/wastelens_rej_frozen_best.keras"))
    ap.add_argument("--out", type=Path, default=Path("scratch/parity_frozen"))
    ap.add_argument("--real-per-group", type=int, default=8)
    args = ap.parse_args()

    print(f"[1/4] Loading {args.model}")
    model = tf.keras.models.load_model(args.model)

    print("[2/4] Building synthetic inputs")
    names, synth = build_synthetic_inputs()
    x_parts = [synth]

    print("[3/4] Selecting + preprocessing real eval inputs")
    real = select_real_inputs(json.loads(EVAL_SETS_JSON.read_text()),
                              args.real_per_group)
    if real:
        x_parts.append(preprocess_paths([r["path"] for r in real]))
        names += [r["name"] for r in real]
    else:
        print("      (no real inputs available on disk - synthetic only)")

    x = np.concatenate(x_parts).astype(np.float32)
    print(f"      {len(x)} inputs total")
    out = model.predict(x, verbose=0)

    print(f"[4/4] Writing {args.out}/inputs.f32 + parity.json")
    args.out.mkdir(parents=True, exist_ok=True)
    x.tofile(args.out / "inputs.f32")
    payload = {
        "model": str(args.model),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "n": int(len(x)),
        "input_shape": [224, 224, 3],
        "inputs_note": ("raw float32 C-order N x 224 x 224 x 3, already in model "
                        "input space [-1,1]; names in input_names[] order"),
        "input_names": names,
        "keras": {"bins": out["bins"].tolist(),
                  "reject": out["reject"].ravel().tolist()},
    }
    (args.out / "parity.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print("done.")


if __name__ == "__main__":
    main()
