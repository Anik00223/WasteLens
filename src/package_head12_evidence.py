# src/package_head12_evidence.py - strict evidence bundler (PART 1/2).
"""Iteration-12 evidence packager: fail loudly on any missing artifact.

Collects the registered evidence set, writes a manifest index
(files + sha256 + bytes), and creates head12_evidence.tar.gz in BOTH
the repo root and the Drive backup dir. Refuses to build an incomplete
archive (exit 1 listing what is missing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


REQUIRED = [
    "docs/rejection_experiment/iteration12_distillation_protocol.md",
    "docs/rejection_experiment/head12_audit.json",
    "docs/rejection_experiment/head12_rejection_equivalence.json",
    "docs/rejection_experiment/head12_seed_consistency.json",
    "docs/rejection_experiment/iteration12_decision.md",
    "docs/rejection_experiment/head12_browser.json",
    "docs/rejection_experiment/ui_clickthrough_head12s42.json",
]
PER_SEED = [
    "docs/rejection_experiment/head12s{seed}_history.csv",
    "docs/rejection_experiment/head12s{seed}_training_config.json",
    "docs/rejection_experiment/head12s{seed}_freeze_proof.json",
    "docs/rejection_experiment/head12s{seed}_eval.json",
    "docs/rejection_experiment/head12s{seed}_gates.json",
    "docs/rejection_experiment/head12s{seed}_report.md",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default=None)
    ap.add_argument("--out", default="head12_evidence.tar.gz")
    args = ap.parse_args()
    wanted: list[str] = list(REQUIRED)
    for s in (42, 43):
        wanted += [p.format(seed=s) for p in PER_SEED]
    missing = [r for r in wanted if not Path(r).exists()]
    if missing:
        print("REFUSING to build an incomplete archive; missing:")
        for r in missing:
            print(f"  MISSING {r}")
        raise SystemExit(f"incomplete evidence: {len(missing)} files missing")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
    except Exception:
        head = "unknown"
    index = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "git_commit": head,
        "files": [{"path": r, "bytes": Path(r).stat().st_size,
                   "sha256": sha(Path(r))} for r in wanted],
    }
    idx_path = Path("docs/rejection_experiment/head12_evidence_index.json")
    idx_path.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    out = Path(args.out)
    with tarfile.open(out, "w:gz") as tar:
        tar.add(str(idx_path), arcname=str(idx_path))
        for r in wanted:
            tar.add(r, arcname=r)
    print(f"packed {len(wanted)} files + index -> {out} "
          f"({out.stat().st_size} B)")
    if args.drive:
        dest = Path(args.drive) / out.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(out.read_bytes())
        print(f"mirrored -> {dest}")


if __name__ == "__main__":
    main()
