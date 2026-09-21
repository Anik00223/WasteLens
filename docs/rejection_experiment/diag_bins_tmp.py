"""Temporary diagnostic: compare baseline vs dual-head bins accuracy
on the first 256 supported test images. DELETE BEFORE COMMIT."""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import tensorflow as tf

import train as wl

m = json.loads(Path("docs/rejection_experiment/eval_sets.json").read_text())
base = tf.keras.models.load_model("models/checkpoints/wastelens_ep09.keras")
dual = tf.keras.models.load_model("models/checkpoints/wastelens_rej_best.keras")

for name, model in [("baseline", base), ("dual", dual)]:
    rows = m["supported_test"][:256]
    y = np.array([r[1] for r in rows])
    ds = wl.make_dataset([(Path(p), int(l)) for p, l in rows], training=False)
    P = model.predict(ds, verbose=0)
    if isinstance(P, dict):
        P = P["bins"]
    print(name, "acc256=", float((P.argmax(1) == y).mean()), flush=True)
