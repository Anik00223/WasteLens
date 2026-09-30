# src/backup_iter12.py
"""Iteration-12 Drive persistence helpers (Colab + laptop).

save    : copy the registered artifact set repo -> Drive backup root.
restore : copy it back Drive -> repo working tree.
status  : report which artifacts exist locally and on Drive.

Every copy is sha256-verified after the fact (directories are compared
file by file). Missing items are listed per item; any copy failure, or
an operation that copies nothing, exits non-zero - a cell can never
claim "saved" when the backup did not actually happen.
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


def dir_hashes(root: Path) -> dict[str, str]:
    """Relative path -> sha256 for every file under `root`."""
    return {str(p.relative_to(root)): sha(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def copy_verified(src: Path, dst: Path) -> None:
    """Copy file/dir, then prove every copied file is byte-identical."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        before = dir_hashes(src)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        after = dir_hashes(dst)
        missing = [rel for rel in before if rel not in after]
        bad = [rel for rel, h in before.items()
               if after.get(rel) not in (None, h)]
        assert not missing and not bad, (
            f"dir copy mismatch in {src}: missing={missing[:3]} "
            f"bad={bad[:3]}")
        print(f"  dir {src} -> {dst} ({len(before)} files verified)")
        return
    shutil.copy2(src, dst)
    assert sha(src) == sha(dst), f"copy hash mismatch: {src}"
    print(f"  {src.name} ({src.stat().st_size} B) -> {dst}")

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


def artifact_list(seeds: list[int]) -> list[str]:
    items: list[str] = list(HEAD12_EVIDENCE)
    for s in seeds:
        items += [p.format(seed=s) for p in HEAD12_PER_SEED]
        items += [p.format(seed=s) for p in HEAD12_CKPTS]
    items += ["docs/rejection_experiment/head12_browser.json",
              "docs/rejection_experiment/ui_clickthrough_head12s42.json"]
    return items


def status(drive: Path, items: list[str], repo_root: Path) -> int:
    n_local = n_drive = 0
    for rel in items:
        local, remote = repo_root / rel, drive / rel
        local_ok, drive_ok = local.exists(), remote.exists()
        n_local += local_ok
        n_drive += drive_ok
        print(f"{'OK ' if local_ok else 'MISS'} local  {rel}")
        print(f"{'OK ' if drive_ok else 'MISS'} drive  {rel}")
    print(f"status: {n_local}/{len(items)} local, "
          f"{n_drive}/{len(items)} on drive")
    return 0


def run_action(action: str, drive: Path, items: list[str],
               repo_root: Path) -> int:
    """Copy every item; return non-zero unless the action really happened."""
    failures: list[str] = []
    skipped: list[str] = []
    done = 0
    for rel in items:
        src = (drive / rel) if action == "restore" else (repo_root / rel)
        dst = (repo_root / rel) if action == "restore" else (drive / rel)
        if not src.exists():
            skipped.append(rel)
            print(f"  SKIP (missing): {src}")
            continue
        try:
            copy_verified(src, dst)
            done += 1
        except Exception as exc:  # noqa: BLE001 - must fail loudly
            print(f"  FAILED {src} -> {dst}: {exc}")
            failures.append(rel)
    if failures:
        print(f"{action} FAILED: {len(failures)} errors, {done} copied, "
              f"{len(skipped)} missing; drive root = {drive}")
        return 1
    if done == 0 and skipped:
        print(f"{action} FAILED: nothing copied ({len(skipped)} items "
              f"missing) - is the producing step / Drive backup in place? "
              f"drive root = {drive}")
        return 1
    print(f"{action} done: {done} copied, {len(skipped)} missing (skipped), "
          f"0 errors; drive root = {drive}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["save", "restore", "status"])
    ap.add_argument("--drive", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    drive = Path(args.drive)
    repo_root = Path(".")
    seeds = [args.seed] if args.seed else [42, 43]
    items = artifact_list(seeds)
    if args.action == "status":
        raise SystemExit(status(drive, items, repo_root))
    raise SystemExit(run_action(args.action, drive, items, repo_root))


if __name__ == "__main__":
    main()
