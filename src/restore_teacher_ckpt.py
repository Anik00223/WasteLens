# src/restore_teacher_ckpt.py
"""Restore + verify the registered shipped teacher checkpoint (pre-S3/S4).

The 13.5 MB shipped Variant-C checkpoint is gitignored
(`models/checkpoints/*.keras`), so a fresh Colab runtime does not have
it. This script restores it from the persistent Drive backup
(<drive>/checkpoints/wastelens_rej_shipped_best.keras), verifies the
registered md5 BEFORE any training may start, mirrors a verified local
copy back to Drive, and otherwise exits non-zero with the exact upload
instructions. No other checkpoint is ever accepted as the teacher.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

REGISTERED_MD5 = "8735019226412c4d67b0a17269dc38ad"
REGISTERED_BYTES = 13583639
REGISTERED_SHA256 = ("65fc18bb7b6ab966e0cbfb93eec01f4634fc0c92f35c93e"
                     "1535f52522e00426e")
DEFAULT_PATH = Path("models/checkpoints/wastelens_rej_shipped_best.keras")
DRIVE_SUBDIR = "checkpoints"


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def md5_or_none(path: Path) -> str | None:
    return md5_file(path) if path.exists() else None


def instructions(drive_ckpt: Path, local: Path) -> str:
    return (
        "STOP: the registered teacher checkpoint is not available.\n"
        f"  required md5    : {REGISTERED_MD5}\n"
        f"  required bytes  : {REGISTERED_BYTES}\n"
        f"  required sha256 : {REGISTERED_SHA256}\n"
        "Upload the checkpoint that ships with the local repo checkout\n"
        f"  {local}\n"
        "to Google Drive, ONCE, from a machine that has it:\n"
        f"  {drive_ckpt}\n"
        "Then re-run this cell. Do NOT train from any other checkpoint;\n"
        "S3/S4 must not run until this restores and verifies.")


def ensure(drive: Path, path: Path) -> int:
    """Restore/verify the teacher; returns a process exit code."""
    drive_ckpt = drive / DRIVE_SUBDIR / path.name
    local_md5 = md5_or_none(path)
    drive_md5 = md5_or_none(drive_ckpt)
    print(f"teacher target: {path}")
    print(f"  local md5   : {local_md5 or 'MISSING'} "
          f" (want {REGISTERED_MD5})")
    print(f"  drive md5   : {drive_md5 or 'MISSING'}  ({drive_ckpt})")
    if local_md5 == REGISTERED_MD5:
        print("teacher: working tree already holds the registered checkpoint")
        if drive_md5 != REGISTERED_MD5:
            drive_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, drive_ckpt)
            if md5_or_none(drive_ckpt) != REGISTERED_MD5:
                print("STOP: mirror to Drive failed verification")
                return 1
            print(f"teacher: mirrored verified copy -> {drive_ckpt}")
        return 0
    if drive_md5 == REGISTERED_MD5:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(drive_ckpt, path)
        if md5_or_none(path) != REGISTERED_MD5:
            print("STOP: restore to the working tree failed verification")
            return 1
        print(f"teacher: restored {drive_ckpt} -> {path} (md5 verified)")
        return 0
    print(instructions(drive_ckpt, path))
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True,
                    help="Drive backup root (.../WasteLens/iteration12)")
    ap.add_argument("--path", default=str(DEFAULT_PATH),
                    help="working-tree checkpoint path")
    args = ap.parse_args()
    raise SystemExit(ensure(Path(args.drive), Path(args.path)))


if __name__ == "__main__":
    main()
