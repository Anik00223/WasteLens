# src/materialize_manifest.py - PART 1/2: header + helpers.
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
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 429:
                    raise IOError("HTTP 429 rate limited")
                if r.status >= 500:
                    raise IOError(f"HTTP {r.status}")
                return r.read()
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 60) + 0.5 * attempt)
    raise RuntimeError(f"fetch failed after {retries} tries: {last}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drive", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--retries", type=int, default=6)
    args = ap.parse_args()
    man_path = Path(args.manifest)
    out = Path(args.out)
    drive = Path(args.drive) if args.drive else None
    out.mkdir(parents=True, exist_ok=True)
    if drive:
        drive.mkdir(parents=True, exist_ok=True)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    entries = man["entries"]
    verified, missing, mismatch, errors = [], [], [], []
    for e in entries:
        name = e["local_name"]
        want_sha, want_bytes = e["sha256"], e["bytes"]
        dest = out / name
        if dest.exists():
            raw = dest.read_bytes()
            if len(raw) == want_bytes and \
                    hashlib.sha256(raw).hexdigest() == want_sha:
                verified.append(name)
                if drive and not (drive / name).exists():
                    (drive / name).write_bytes(raw)
                continue
        if drive and (drive / name).exists():
            raw = (drive / name).read_bytes()
            if len(raw) == want_bytes and \
                    hashlib.sha256(raw).hexdigest() == want_sha:
                dest.write_bytes(raw)
                verified.append(name)
                continue
        try:
            raw = fetch(e["thumb_url"], args.retries)
        except Exception as exc:
            errors.append({"file": name, "error": str(exc)})
            missing.append(name)
            continue
        got_sha = hashlib.sha256(raw).hexdigest()
        if len(raw) != want_bytes or got_sha != want_sha:
            mismatch.append({"file": name, "got_bytes": len(raw),
                             "want_bytes": want_bytes,
                             "got_sha256": got_sha,
                             "want_sha256": want_sha})
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(raw)
        tmp.replace(dest)
        if drive:
            (drive / name).write_bytes(raw)
        verified.append(name)
    report = {
        "generated_utc": datetime.now(timezone.utc)
        .strftime("%Y-%m-%d %H:%M:%S UTC"),
        "manifest": str(man_path),
        "manifest_entries": len(entries),
        "expected_counts": man.get("counts_by_group"),
        "out_dir": str(out),
        "drive_dir": str(drive) if drive else None,
        "verified": len(verified),
        "missing": missing,
        "sha_mismatch": mismatch,
        "fetch_errors": errors,
        "complete": len(verified) == len(entries) and not mismatch,
        "git_commit": git_head(),
        "source_policy": "thumb_url only (original_url drifted)",
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
