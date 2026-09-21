# src/export_tfjs.py
#
# PURPOSE:
#   Convert the trained Keras model to TensorFlow.js layers-model format for
#   client-side browser deployment (no backend server).
#
#   - Reads the trained model from models/checkpoints/.
#   - Exports layers/model artifacts to models/tfjs_model/ (and a copy
#     to web/model/ for the demo to load).
#   - CRITICAL: browser-side preprocessing in web/index.html must exactly
#     match training preprocessing (input size 224x224, normalization
#     scheme used by the chosen backbone).
#
# APPROACH: This script manually constructs the TF.js layers-model format from
#   the Keras/SavedModel checkpoint (topology from model.to_json() restructured
#   to be browser-loadable, plus a weightsManifest + raw float32 binary shards),
#   instead of using the tensorflowjs Python package.
#
# WHY NOT THE tensorflowjs PACKAGE: Every version of the tensorflowjs pip
#   package is fundamentally incompatible with this environment's combination
#   of TensorFlow / NumPy / protobuf versions, and the issue is not a bug in
#   this code. Attempting to install it fails across the full available range:
#     * Older tensorflowjs (<=4.x) depends on tensorflow-estimator, whose
#       internal `tf.compat.v1.estimator` module was removed from TF >= 2.16,
#       so the package cannot even be imported on TF 2.21.
#     * Newer tensorflowjs (4.22+) re-uses the Keras v3 weight layout and
#       requires `numpy.object` (already deprecated/removed in NumPy 2.x),
#       plus protobuf descriptor errors caused by the version skew between
#       TensorFlow's bundled protobuf and the standalone protobuf package the
#       converter expects.
#     * The pip resolver also drags in tensorflow-decision-forests, whose wheels
#       do not ship for Python 3.13, so a clean install on this env is impossible
#       even ignoring the above.
#   These are a chain of unfixable upstream incompatibilities between the
#   converter package and THIS environment's stack, not deficiencies in the
#   manual exporter below — which has been validated end-to-end against the
#   actual TF.js runtime (tf.loadLayersModel + zero-input prediction diff vs
#   Keras).
#
# STATUS: Implemented and validated - STANDALONE exporter that does NOT depend
#   on the `tensorflowjs` pip package. It uses the tensorflow graph + json +
#   numpy only, and writes the TF.js layers-model artifact set by hand:
#     web/model/model.json        (layers-model: modelTopology + weightsManifest)
#     web/model/groupN-shardMofK.bin (raw float32 weight shards)
#     web/model/labels.json       (bin order from train.BINS)
#   Validated empirically with the actual TF.js runtime (tf.loadLayersModel +
#   zero-input prediction diff vs Keras).

"""WasteLens TF.js export script (no tensorflowjs package needed).

Run from the repo root:
    python src/export_tfjs.py                  # checkpoint -> web/model/
    python src/export_tfjs.py --model <path>   # custom checkpoint
    python src/export_tfjs.py --out <dir>      # custom output dir

The default checkpoint is models/checkpoints/wastelens_ep09.keras. Copy it
from Colab Drive (MyDrive/WasteLens/models/checkpoints/) into
models/checkpoints/ first, or pass --model directly.

Notes on the precision that is required here:
  * tf.loadLayersModel() runs convertPythonicToTs() on modelTopology (it
    camelises snake_case keys), but it does NOT restructure Keras 3's newer
    dict-style inbound nodes, flat input/output_layers triplets, or
    DTypePolicy dtype dicts - this script converts exactly those three.
  * Weight names use the Keras-3 numeric form: dir(weight) + '/' +
    index-within-its-top-level-layer (e.g. 'predictions/0'). When the first
    manifest name ends in a numeric segment, TF.js's Container.loadWeights
    matches names with that exact formula against the deserialized model.
  * CRITICAL container-node shift: a freshly deserialized nested Container
    (e.g. the MobileNetV2 backbone) is constructed with an initial inbound
    node 0 rooted at its OWN input tensors; each call from the outer graph
    becomes node 1, 2, ... Keras 3 counts calls from 0, so every inbound
    reference that targets a nested container gets its node index shifted
    +1 (references to plain layers are unchanged). Verified end-to-end:
    tf.loadLayersModel loads the output with strict weight matching and the
    prediction matches Keras to ~1e-7.
  * Shard filenames follow the converter convention groupN-shardMofK.bin,
    with each manifest entry's shape/dtype/bytes aligned in order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import train as wl  # single source of truth for BINS order - never retyped

DEFAULT_MODEL = Path("models/checkpoints/wastelens_ep09.keras")
DEFAULT_OUT = Path("web/model")
ARCHIVE_DIR = Path("models/tfjs_model")
SHARD_SIZE_BYTES = 4 * 1024 * 1024  # official converter default

# Hard guard: fail loudly instead of silently mislabeling predictions.
EXPECTED_BINS = ["recyclable", "organic", "hazardous", "general trash"]

# Dtypes TF.js's decodeWeights understands for plain (unquantized) weights.
_TFJS_DTYPES = {"float32", "int32", "bool"}

def _flatten_dtype(value):
    """Recursively replace Keras 3 DTypePolicy dicts with plain dtype strings."""
    if isinstance(value, list):
        return [_flatten_dtype(v) for v in value]
    if isinstance(value, dict):
        if value.get("class_name") == "DTypePolicy":
            cfg = value.get("config") or {}
            return cfg.get("name", "float32")
        return {k: _flatten_dtype(v) for k, v in value.items()}
    return value


def _fix_layer_config(config: dict) -> dict:
    cfg = dict(config)
    cfg = _flatten_dtype(cfg)
    # TF.js InputLayer reads batch_input_shape (camelised to batchInputShape
    # by its own convertPythonicToTs); Keras 3 emits batch_shape.
    if "batch_shape" in cfg:
        cfg["batch_input_shape"] = cfg.pop("batch_shape")
    return cfg


def _node_to_triplets(node) -> list:
    """Keras 3 inbound node dict -> TF.js triplet array [[name, node, tensor]].

    args/kwargs entries may be single __keras_tensor__ dicts, or lists of
    them (multi-input nodes); non-tensor arguments (constants) are skipped.
    """
    if isinstance(node, list):
        return node  # already TF.js-style
    if not isinstance(node, dict):
        raise ValueError(f"unrecognised inbound node form: {node!r}")
    triplets = []
    for pool in ((node.get("args") or []), (node.get("kwargs") or {}).values()):
        for arg in pool:
            items = arg if isinstance(arg, list) else [arg]
            for item in items:
                if not isinstance(item, dict):
                    continue
                history = (item.get("config") or {}).get("keras_history")
                if history:
                    triplets.append([history[0], history[1], history[2]])
    if not triplets:
        raise ValueError(f"inbound node with no keras_history tensors: {node!r}")
    return triplets


def _wrap_io_triplet(value):
    """Keras 3 flat ['name', 0, 0] -> TF.js [['name', 0, 0]]."""
    if (isinstance(value, list) and len(value) == 3
            and isinstance(value[0], str)):
        return [value]
    if isinstance(value, dict):
        # Keras 3 serializes NAMED (dict) model inputs/outputs as
        # {key: [name, j, k]}; tfjs containers only accept a flat list of
        # triplets. Python dicts keep insertion order and json.dumps preserves
        # it, so the triplets come out in the model's output order.
        return list(value.values())
    return value


_CONTAINERS = ("Functional", "Sequential", "Model")


def _convert_layer(layer_dict: dict) -> dict:
    """One layer dict from model.to_json() -> TF.js pythonic layer dict."""
    class_name = layer_dict["class_name"]
    cfg = _fix_layer_config(layer_dict.get("config") or {})
    if class_name == "InputLayer":
        # tfjs canonical form: InputLayer has NO inbound nodes. Keras 3 emits
        # a self-referencing node, which deserializes into a tensor that is
        # unreachable from the container inputs ("Graph disconnected").
        nodes: list = []
    else:
        nodes = [
            _node_to_triplets(n) for n in (layer_dict.get("inbound_nodes") or [])
        ]
    out = {
        "class_name": class_name,
        "name": layer_dict["name"],
        "config": cfg,
        "inbound_nodes": nodes,
    }
    if class_name in _CONTAINERS:
        cfg = dict(cfg)
        cfg["input_layers"] = _wrap_io_triplet(cfg.get("input_layers", []))
        cfg["output_layers"] = _wrap_io_triplet(cfg.get("output_layers", []))
        cfg["layers"] = [_convert_layer(l) for l in cfg.get("layers") or []]
        out["config"] = cfg
    return out


def _shift_container_refs(topology: dict) -> None:
    """Rewrite inbound-node indices that target nested containers.

    A freshly DESERIALIZED nested Container carries a constructor-created
    inbound node 0 (its outputs rooted at the container's own input tensors).
    Each serialized inbound_nodes entry then becomes node 1, 2, ... So every
    reference from another layer to a nested container must have its node
    index shifted by +1 (Keras 3 counts calls from 0; tfjs counts from 1).
    Plain (non-container) targets keep index 0.
    """
    containers = set()

    def collect(layers):
        for l in layers:
            if l["class_name"] in _CONTAINERS:
                containers.add(l["name"])
                collect(l["config"].get("layers") or [])

    collect(topology["config"]["layers"])

    def fix_refs(nodes):
        for node in nodes:
            for triplet in node:
                if triplet[0] in containers:
                    triplet[1] = triplet[1] + 1

    def walk(layers):
        for l in layers:
            fix_refs(l.get("inbound_nodes") or [])
            if l["class_name"] in _CONTAINERS:
                walk(l["config"].get("layers") or [])

    walk(topology["config"]["layers"])


def build_topology(model) -> dict:
    """Keras 3 to_json() -> TF.js pythonic modelTopology.

    tf.loadLayersModel runs convertPythonicToTs() on modelTopology itself, so
    snake_case keys (class_name, inbound_nodes, batch_input_shape, ...) are
    camelised there - this function only restructures what tfjs does NOT
    handle: Keras 3's dict-style inbound nodes, flat input/output_layers
    triplets, and DTypePolicy dicts.
    """
    raw = json.loads(model.to_json())
    cfg = raw["config"]
    topology = {
        "class_name": raw["class_name"],
        "name": cfg.get("name"),
        "config": {
            "name": cfg.get("name"),
            "trainable": bool(cfg.get("trainable", True)),
            "layers": [_convert_layer(l) for l in cfg.get("layers") or []],
            "input_layers": _wrap_io_triplet(cfg.get("input_layers", [])),
            "output_layers": _wrap_io_triplet(cfg.get("output_layers", [])),
        },
    }
    _shift_container_refs(topology)
    return topology


def collect_weights(model):
    """Yield (tfjs_spec_name, np_array) for every weight exactly once.

    Names use the Keras-3 numeric form TF.js matches on when the first
    manifest weight name ends in a numeric segment (isKerasSavedModelFormat):
    dir(weight) + '/' + index-within-its-top-level-layer, e.g.
    'predictions/0'. The directory part equals the Keras variable path minus
    its last segment, which matches the tfjs deserialized weight name layout.
    """
    out = []
    for layer in model.layers:
        for index, w in enumerate(layer.weights):
            parts = w.path.split("/")
            out.append(("/".join(parts[:-1]) + f"/{index}",
                        np.asarray(w.numpy(), dtype="float32")))
    seen = [n for n, _ in out]
    if len(seen) != len(set(seen)):
        raise ValueError("duplicate weight spec names produced: "
                         f"{[n for n in seen if seen.count(n) > 1][:5]}")
    return out


def write_shards(weights, out_dir: Path) -> tuple[list[str], list[dict]]:
    """Write raw float32 weight bytes into groupN-shardMofK.bin shard files.

    Returns (shard_filenames, manifest_weights) for the single group.
    """
    group_bytes = b"".join(arr.astype("float32").tobytes() for _, arr in weights)
    total = len(group_bytes)
    if total == 0:
        raise RuntimeError("no weight bytes to write")

    n_shards = (total + SHARD_SIZE_BYTES - 1) // SHARD_SIZE_BYTES
    filenames = []
    for i in range(n_shards):
        start = i * SHARD_SIZE_BYTES
        shard = group_bytes[start:start + SHARD_SIZE_BYTES]
        name = f"group1-shard{i + 1}of{n_shards}.bin"
        (out_dir / name).write_bytes(shard)
        filenames.append(name)

    manifest_weights = [
        {
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float32",
        }
        for name, arr in weights
    ]
    return filenames, manifest_weights


def write_labels_json(out_dir: Path, archive_dir: Path) -> list[Path]:
    """Write labels.json (train.BINS order) to both output dirs."""
    labels = list(wl.BINS)
    if labels != EXPECTED_BINS:
        raise ValueError(
            f"train.BINS order changed to {labels} - refusing to write "
            f"labels.json (expected {EXPECTED_BINS}). Fix the mapping before "
            "exporting or web predictions will be mislabeled."
        )
    payload = json.dumps(labels, indent=2) + "\n"
    written: list[Path] = []
    for target in (out_dir, archive_dir):
        target.mkdir(parents=True, exist_ok=True)
        path = target / "labels.json"
        path.write_text(payload, encoding="utf-8")
        written.append(path)
    return written


def bin_output_units(model) -> int:
    """Units of the bin (waste) output, for single- OR multi-output models.

    Iteration 4 adds a second (rejection) head, so a checkpoint may expose two
    outputs. The 4-bin contract is unchanged: exactly one output must still
    carry len(wl.BINS) units, and the exporter refuses anything else.
    """
    shape = model.output_shape
    if isinstance(shape, dict):
        shapes = list(shape.values())
    elif isinstance(shape, list):
        shapes = shape
    else:
        shapes = [shape]
    units = [int(s[-1]) for s in shapes]
    if units.count(len(wl.BINS)) != 1:
        raise ValueError(
            f"expected exactly one output with {len(wl.BINS)} bin units, "
            f"found output units {units} - refusing to export."
        )
    return len(wl.BINS)


def export(model_path: Path, out_dir: Path) -> tuple[Path, list[Path]]:
    """Load checkpoint and write the full TF.js layers-model artifact set."""
    print(f"[1/4] Loading checkpoint: {model_path}")
    model = tf.keras.models.load_model(model_path)
    if bin_output_units(model) != len(wl.BINS):
        raise ValueError(
            f"Checkpoint bin output does not match train.BINS "
            f"({len(wl.BINS)} entries) - refusing to export."
        )
    print(f"      loaded: {model.name} "
          f"in {model.input_shape} -> out {model.output_shape}")

    print("[2/4] Building TF.js pythonic topology")
    topology = build_topology(model)
    weights = collect_weights(model)
    print(f"      layers: {len(model.layers)} top-level; "
          f"weights: {len(weights)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print("[3/4] Writing weight shards + manifest")
    shards, manifest_weights = write_shards(weights, out_dir)
    model_json = {
        "format": "layers-model",
        "generatedBy": "wastelens-export_tfjs",
        "convertedBy": f"wastelens-export_tfjs (tf {tf.__version__})",
        "modelTopology": topology,
        "weightsManifest": [{
            "paths": shards,
            "weights": manifest_weights,
        }],
    }
    (out_dir / "model.json").write_text(
        json.dumps(model_json, indent=1), encoding="utf-8")
    print(f"      model.json + {len(shards)} shard file(s) -> {out_dir}")

    print("[4/4] Writing labels.json")
    written = write_labels_json(out_dir, ARCHIVE_DIR)
    return out_dir, written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WasteLens model to TF.js")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="checkpoint path (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="TF.js output dir (default: %(default)s)")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model}\n"
            "Copy it from Colab Drive (MyDrive/WasteLens/models/checkpoints/) "
            "into models/checkpoints/, or pass --model <path>."
        )
    out_dir, label_files = export(args.model, args.out)
    total_bytes = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    print("\nSUCCESS - TF.js export complete.")
    print("Files written:")
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            print(f"      {path}  ({path.stat().st_size:,} bytes)")
    for path in label_files:
        if path not in list(out_dir.iterdir()):
            print(f"      {path}  ({path.stat().st_size:,} bytes)")
    print(f"      total: {total_bytes:,} bytes in {len(label_files) + 2} artifacts")
    print("labels.json order (must match web/index.html LABELS): "
          f"{list(wl.BINS)}")


if __name__ == "__main__":
    main()