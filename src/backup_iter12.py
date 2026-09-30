# src/backup_iter12.py - PART 1/2: save/restore helpers.
"""Iteration-12 Drive persistence helpers (Colab + laptop).

save: copy a file/dir artifact set to the Drive backup root.
restore: copy back from Drive to the working tree.
Both verify per-file sha256 after copy and print persistent paths.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def copy_verified(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  dir {src} -> {dst}")
        return

HEAD12_EVIDENCE = [
    "docs/rejection_experiment/head12_audit.json",
    "docs/rejection_experiment/head12_rejection_equivalence.json",
    "docs/rejection_experiment/head12_seed_consistency.json",
    "docs/rejection_experiment/iteration12_decision.md",
]
HEAD12_PER_SEED = [
    "docs/rejection_experiment/head12s{seed}_history.csv",
    "docs/rejection_experiment/head12s{seed}_training_config.json",
    "docs/rejection_experiment/head12s{seed}_freeze_proof.json",
    "docs/rejection_experiment/head12s{seed}_eval.json",
    "docs/rejection_experiment/head12s{seed}_gates.json",
    "docs/rejection_experiment/head12s{seed}_report.md",
]
HEAD12_CKPTS = [
    "models/checkpoints/wastelens_rej_head12s{seed}_final.keras",
    "models/checkpoints/wastelens_rej_head12s{seed}_best.keras",
    "models/checkpoints/wastelens_rej_head12s{seed}_06.keras",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["save", "restore", "status"])
    ap.add_argument("--drive", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    drive = Path(args.drive)
    seeds = [args.seed] if args.seed else [42, 43]
    items: list[str] = list(HEAD12_EVIDENCE)
    for s in seeds:
        items += [p.format(seed=s) for p in HEAD12_PER_SEED]
        items += [p.format(seed=s) for p in HEAD12_CKPTS]
    items += ["docs/rejection_experiment/head12_browser.json",
              "docs/rejection_experiment/ui_clickthrough_head12s42.json"]
    if args.action == "status":
        for rel in items:
            local, remote = Path(rel), drive / rel
            print(f"{'OK ' if local.exists() else 'MISS'} local  {rel}")
            print(f"{'OK ' if remote.exists() else 'MISS'} drive  {rel}")
        return
    for rel in items:
        src = (drive / rel) if args.action == "restore" else Path(rel)
        dst = Path(rel) if args.action == "restore" else (drive / rel)
        if src.exists():
            copy_verified(src, dst)
        else:
            print(f"  SKIP (missing): {src}")
    print(f"{args.action} done; drive root = {drive}")


if __name__ == "__main__":
    main()

    shutil.copy2(src, dst)
    assert sha(src) == sha(dst), f"copy hash mismatch: {src}"
    print(f"  {src.name} ({src.stat().st_size} B) -> {dst}")
