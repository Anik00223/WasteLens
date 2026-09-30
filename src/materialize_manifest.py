# src/materialize_manifest.py
"""Iteration-12 dataset materializer with byte-level SHA locks.

Every entry is (re)built from its manifest `thumb_url` ONLY - the
registered source; `original_url` drifted and is never used. Verified
files in --out / --drive are reused (resume) and never re-downloaded.

Transient SHA mismatches are retried against the same thumb_url with
bounded exponential backoff (--sha-retries / --sha-backoff); a file is
reported as a mismatch only after every attempt failed. Any incomplete
run exits non-zero: the pipeline must stop, never train on a partial or
drifted dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = {"User-Agent":
      "WasteLensResearch/1.0 (contact: wastelens-dev@example.org)"}


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
    except Exception:
        return "unknown"


def fetch(url: str, retries: int) -> bytes:
    """GET `url` with bounded retries on any transient failure."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                status = getattr(r, "status", None) or 0
                if status == 429:
                    raise IOError("HTTP 429 rate limited")
                if status >= 500:
                    raise IOError(f"HTTP {status}")
                return r.read()
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 60) + 0.5 * attempt)
    raise RuntimeError(f"fetch failed after {retries} tries: {last}")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_locked(raw: bytes, want_bytes: int, want_sha: str) -> bool:
    return len(raw) == want_bytes and sha256_bytes(raw) == want_sha


def locked_read(path: Path, want_bytes: int, want_sha: str) -> bytes | None:
    """Return file bytes iff the file exists and matches the lock."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return raw if is_locked(raw, want_bytes, want_sha) else None


def write_locked(dest: Path, raw: bytes) -> None:
    """Atomically (re)write `dest` with verified bytes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(raw)
    tmp.replace(dest)


def mirror_to_drive(drive: Path | None, name: str, raw: bytes,
                    want_bytes: int, want_sha: str) -> None:
    """Persist a verified file on Drive unless a verified copy is there."""
    if drive is None:
        return
    dfile = drive / name
    if locked_read(dfile, want_bytes, want_sha) is not None:
        return
    write_locked(dfile, raw)


def materialize(entries, out: Path, drive: Path | None, retries: int,
                sha_retries: int, sha_backoff: float) -> dict:
    """Verify every entry into `out`; downloads only ever use thumb_url."""
    verified: list[str] = []
    missing: list[str] = []
    mismatch: list[dict] = []
    errors: list[dict] = []
    stats = {"resumed_local": 0, "restored_from_drive": 0, "downloaded": 0}
    for e in entries:
        name = e["local_name"]
        want_sha, want_bytes = e["sha256"], e["bytes"]
        dest = out / name
        raw = locked_read(dest, want_bytes, want_sha)
        if raw is not None:
            stats["resumed_local"] += 1
        elif drive is not None:
            raw = locked_read(drive / name, want_bytes, want_sha)
            if raw is not None:
                write_locked(dest, raw)
                stats["restored_from_drive"] += 1
        if raw is not None:
            mirror_to_drive(drive, name, raw, want_bytes, want_sha)
            verified.append(name)
            continue
        outcome, last = None, None
        for attempt in range(1, sha_retries + 1):
            try:
                raw = fetch(e["thumb_url"], retries)
            except Exception as exc:
                if last is None:
                    outcome = "missing"
                    errors.append({"file": name, "error": str(exc)})
                else:
                    outcome = "mismatch"
                    last["error_after_mismatch"] = str(exc)
                break
            if is_locked(raw, want_bytes, want_sha):
                write_locked(dest, raw)
                mirror_to_drive(drive, name, raw, want_bytes, want_sha)
                stats["downloaded"] += 1
                verified.append(name)
                outcome = "verified"
                break
            last = {"file": name, "attempts": attempt,
                    "got_bytes": len(raw), "want_bytes": want_bytes,
                    "got_sha256": sha256_bytes(raw),
                    "want_sha256": want_sha}
            if attempt < sha_retries:
                time.sleep(min(sha_backoff * 2 ** (attempt - 1), 30)
                           + 0.25 * attempt)
        if outcome == "missing":
            missing.append(name)
        elif last is not None and outcome != "verified":
            mismatch.append(last)
    return {"verified": verified, "missing": missing,
            "sha_mismatch": mismatch, "fetch_errors": errors,
            "stats": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drive", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--retries", type=int, default=6,
                    help="network tries per download attempt")
    ap.add_argument("--sha-retries", type=int, default=4,
                    help="thumb_url re-downloads on a bytes/sha mismatch")
    ap.add_argument("--sha-backoff", type=float, default=2.0,
                    help="base seconds for bounded exponential backoff")
    args = ap.parse_args()
    man_path = Path(args.manifest)
    out = Path(args.out)
    drive = Path(args.drive) if args.drive else None
    out.mkdir(parents=True, exist_ok=True)
    if drive:
        drive.mkdir(parents=True, exist_ok=True)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    entries = man["entries"]
    res = materialize(entries, out, drive, args.retries,
                      args.sha_retries, args.sha_backoff)
    verified, missing = res["verified"], res["missing"]
    mismatch, errors = res["sha_mismatch"], res["fetch_errors"]
    report = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": str(man_path),
        "manifest_entries": len(entries),
        "expected_counts": man.get("counts_by_group"),
        "out_dir": str(out),
        "drive_dir": str(drive) if drive else None,
        "verified": len(verified),
        "resumed_local": res["stats"]["resumed_local"],
        "restored_from_drive": res["stats"]["restored_from_drive"],
        "downloaded": res["stats"]["downloaded"],
        "missing": missing,
        "sha_mismatch": mismatch,
        "fetch_errors": errors,
        "retry_policy": {"network_tries": args.retries,
                         "sha_tries": args.sha_retries,
                         "sha_backoff_base_seconds": args.sha_backoff},
        "complete": (len(verified) == len(entries) and not missing
                     and not mismatch and not errors),
        "git_commit": git_head(),
        "source_policy": "thumb_url only (original_url drifted); "
                         "byte+sha256 locked; verified files reused",
    }
    rep = Path(args.report) if args.report else Path(
        str(out).rstrip("/") + "_lock_report.json")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"materialize {man_path.name}: {len(verified)}/{len(entries)} "
          f"verified, {len(missing)} missing, {len(mismatch)} mismatched; "
          f"report -> {rep}")
    if not report["complete"]:
        raise SystemExit(f"INCOMPLETE: {len(missing)} missing, "
                         f"{len(mismatch)} mismatched - see {rep}")


if __name__ == "__main__":
    main()
