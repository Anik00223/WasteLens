# src/iter12_preflight.py
"""Iteration-12 strict preflight gate - runs BEFORE any training (S4).

All checks use the registered protocol values. The constants mirror
train_head_adaptation.py (protocol §3); tests assert the files stay in
sync.

  1. adaptation manifest md5 == acae0933aca123c835c1e55396af182a and
     the output dir holds exactly 94 files, every manifest entry present
     with the locked bytes + sha256;
  2. fresh manifest md5 == 99540f56efb95e2c26b2b6cc9414ff5d and the
     output dir holds all 131 entries with locked bytes + sha256;
  3. shipped teacher checkpoint md5 == 8735019226412c4d67b0a17269dc38ad.

Exit 0 only if every check passes; otherwise prints FAIL lines, writes
an optional JSON report, and exits 1. The trainer repeats the protocol
§3 lock check itself - this gate exists so the notebook STOPS before
training and no GPU time is spent on a broken runtime.

NOTE: the manifests are checked out with CRLF (see .gitattributes)
because the registered md5s are MD5s of their CRLF form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REGISTRY = {
    "adapt_manifest_md5": "acae0933aca123c835c1e55396af182a",
    "fresh_manifest_md5": "99540f56efb95e2c26b2b6cc9414ff5d",
    "teacher_md5": "8735019226412c4d67b0a17269dc38ad",
    "adapt_file_count": 94,
    "fresh_file_count": 131,
}
DEFAULTS = {
    "adapt-manifest":
        "docs/rejection_experiment/adaptation_set_manifest.json",
    "fresh-manifest": "docs/rejection_experiment/fresh_set_manifest.json",
    "adapt-dir": "scratch/adaptation_set",
    "fresh-dir": "scratch/fresh_set",
    "teacher": "models/checkpoints/wastelens_rej_shipped_best.keras",
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
    except Exception:
        return "unknown"


def add(results: list[dict], check: str, ok: bool, got, want,
        extra: dict | None = None) -> None:
    results.append({"check": check, "ok": bool(ok), "got": got,
                    "want": want, "extra": extra or {}})


def verify_manifest(results: list[dict], tag: str, manifest: Path,
                    data_dir: Path, expect_count: int,
                    expect_md5: str) -> None:
    got_md5 = md5_file(manifest) if manifest.exists() else None
    add(results, f"{tag}_manifest_md5", got_md5 == expect_md5,
        got_md5, expect_md5, {"path": str(manifest)})
    if got_md5 is None:
        add(results, f"{tag}_manifest_present", False, "missing",
            str(manifest))
        add(results, f"{tag}_files_byte_locked", False, "manifest missing",
            expect_count)
        return
    entries = json.loads(manifest.read_text(encoding="utf-8"))["entries"]
    add(results, f"{tag}_manifest_entries", len(entries) == expect_count,
        len(entries), expect_count)
    n_files = (len([p for p in data_dir.iterdir() if p.is_file()])
               if data_dir.is_dir() else 0)
    add(results, f"{tag}_dir_file_count", n_files == expect_count,
        n_files, expect_count, {"dir": str(data_dir)})
    bad: list[dict] = []
    for e in entries:
        p = data_dir / e["local_name"]
        if not p.exists():
            bad.append({"file": e["local_name"], "why": "missing"})
        elif p.stat().st_size != e["bytes"] or \
                sha256_file(p) != e["sha256"]:
            bad.append({"file": e["local_name"],
                        "why": "bytes/sha256 drift",
                        "got_bytes": p.stat().st_size,
                        "want_bytes": e["bytes"]})
    add(results, f"{tag}_files_byte_locked", not bad,
        {"verified": len(entries) - len(bad), "bad_total": len(bad)},
        {"verified": len(entries)}, {"bad_first10": bad[:10]})


def verify_teacher(results: list[dict], teacher: Path,
                   expect_md5: str) -> None:
    got = md5_file(teacher) if teacher.exists() else None
    add(results, "teacher_md5", got == expect_md5, got, expect_md5,
        {"path": str(teacher)})


def run_checks(adapt_manifest: Path, fresh_manifest: Path, adapt_dir: Path,
               fresh_dir: Path, teacher: Path | None,
               registry: dict | None = None) -> list[dict]:
    reg = dict(REGISTRY if registry is None else registry)
    results: list[dict] = []
    verify_manifest(results, "adapt", adapt_manifest, adapt_dir,
                    int(reg["adapt_file_count"]), reg["adapt_manifest_md5"])
    verify_manifest(results, "fresh", fresh_manifest, fresh_dir,
                    int(reg["fresh_file_count"]), reg["fresh_manifest_md5"])
    if teacher is not None:
        verify_teacher(results, teacher, reg["teacher_md5"])
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    for flag, default in DEFAULTS.items():
        ap.add_argument("--" + flag, default=default)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    results = run_checks(
        Path(args.adapt_manifest), Path(args.fresh_manifest),
        Path(args.adapt_dir), Path(args.fresh_dir), Path(args.teacher))
    ok = all(r["ok"] for r in results)
    for r in results:
        print(f"{'PASS' if r['ok'] else 'FAIL'} {r['check']}: "
              f"got {r['got']} want {r['want']}")
    report = {"generated_utc": datetime.now(timezone.utc)
              .strftime("%Y-%m-%d %H:%M:%S UTC"),
              "git_commit": git_head(), "registry": REGISTRY,
              "checks": results, "all_pass": ok}
    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(json.dumps(report, indent=1) + "\n",
                       encoding="utf-8")
        print("report ->", rep)
    if ok:
        print("preflight: ALL CHECKS PASSED")
        raise SystemExit(0)
    failed = [r["check"] for r in results if not r["ok"]]
    print(f"preflight FAILED: {len(failed)} check(s) - {failed}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
