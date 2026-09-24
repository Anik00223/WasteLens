# src/make_scratch_ui.py
#
# WasteLens - LOOP-18 helper: build a THROWAWAY copy of the production page in
# scratch/ that loads a CANDIDATE model with the CANDIDATE's own calibrated
# threshold, so the real UI click-through can be run against the candidate
# without touching web/ or web/model/.
#
# The copy is the production page with exactly two substitutions:
#   * MODEL_URL stays 'model/model.json' (the model files are copied next to it)
#   * REJECT_THRESHOLD <old> -> <candidate threshold>
# Everything else (markup, decision block, labels, uncertainty rule) is the
# shipped file, byte-for-byte, so the click-through exercises production logic.
#
# Usage:
#   py -3.13 src/make_scratch_ui.py --model-json scratch/tfjs_varc8s42/model.json \
#       --threshold 0.1234 --out scratch/ui_varc8s42

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

WEB_INDEX = Path("web/index.html")
THRESH_RE = re.compile(r"(var REJECT_THRESHOLD = )([0-9.]+)(;)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-json", type=Path, required=True,
                    help="candidate TF.js model.json (shards must sit beside it)")
    ap.add_argument("--threshold", type=float, required=True,
                    help="candidate's calibrated rejection threshold")
    ap.add_argument("--out", type=Path, required=True,
                    help="scratch directory for the throwaway page")
    args = ap.parse_args()

    html = WEB_INDEX.read_text(encoding="utf-8")
    if not THRESH_RE.search(html):
        raise SystemExit("REJECT_THRESHOLD not found in web/index.html - "
                         "production page changed shape, refusing to guess")
    old = THRESH_RE.search(html).group(2)
    new_html = THRESH_RE.sub(lambda m: f"{m.group(1)}{args.threshold:.6f}"
                                      f"{m.group(3)}", html, count=1)

    out = args.out
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.model_json.parent.iterdir()):
        if src.is_file():
            shutil.copy2(src, model_dir / src.name)
    (out / "index.html").write_text(new_html, encoding="utf-8")

    print(f"built {out / 'index.html'} (REJECT_THRESHOLD {old} -> "
          f"{args.threshold:.6f})")
    print(f"model files: {len(list(model_dir.iterdir()))} copied from "
          f"{args.model_json.parent}")
    shutil.copy2(WEB_INDEX, out / "index.production.html")
    print(f"reference copy of the untouched production page: "
          f"{out / 'index.production.html'}")


if __name__ == "__main__":
    main()
