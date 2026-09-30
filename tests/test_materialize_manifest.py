"""Tests for src/materialize_manifest.py: thumb_url-only, retries, resume."""
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import materialize_manifest as mm  # noqa: E402


def entry(name, raw, url="thumb://x"):
    return {"local_name": name, "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "thumb_url": url}


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "out"
        self.drive = self.root / "drive"

    def tearDown(self):
        self.tmp.cleanup()

    def test_downloads_and_mirrors_to_drive_on_success(self):
        raw = b"locked-bytes"
        e = entry("f1.bin", raw, "thumb://f1")
        calls = []

        def fake_fetch(url, retries):
            calls.append(url)
            return raw

        with mock.patch.object(mm, "fetch", fake_fetch):
            res = mm.materialize([e], self.out, self.drive, 2, 3, 0.01)
        self.assertEqual(res["verified"], ["f1.bin"])
        self.assertEqual(res["stats"]["downloaded"], 1)
        self.assertEqual(calls, ["thumb://f1"])     # exact thumb_url only
        self.assertEqual((self.out / "f1.bin").read_bytes(), raw)
        self.assertEqual((self.drive / "f1.bin").read_bytes(), raw)

    def test_verified_local_file_is_preserved_without_download(self):
        raw = b"already-here"
        e = entry("f1.bin", raw, "thumb://never-called")
        self.out.mkdir()
        (self.out / "f1.bin").write_bytes(raw)

        def fake_fetch(url, retries):               # pragma: no cover
            raise AssertionError("must not download a verified file")

        with mock.patch.object(mm, "fetch", fake_fetch):
            res = mm.materialize([e], self.out, None, 2, 3, 0.01)
        self.assertEqual(res["verified"], ["f1.bin"])
        self.assertEqual(res["stats"]["resumed_local"], 1)

    def test_drive_copy_is_reused_when_local_is_missing(self):
        raw = b"from-drive"
        e = entry("f1.bin", raw, "thumb://never-called")
        self.drive.mkdir()
        (self.drive / "f1.bin").write_bytes(raw)

        def fake_fetch(url, retries):               # pragma: no cover
            raise AssertionError("must not download when Drive has it")

        with mock.patch.object(mm, "fetch", fake_fetch):
            res = mm.materialize([e], self.out, self.drive, 2, 3, 0.01)
        self.assertEqual(res["verified"], ["f1.bin"])
        self.assertEqual(res["stats"]["restored_from_drive"], 1)

    def test_transient_mismatch_recovers_via_retry(self):
        raw = b"good"
        e = entry("f1.bin", raw, "thumb://f1")
        answers = [b"bad-1", b"bad-2", raw]
        seen = []

        def fake_fetch(url, retries):
            seen.append(url)
            return answers.pop(0)

        with mock.patch.object(mm, "fetch", fake_fetch):
            res = mm.materialize([e], self.out, None, 2, 3, 0.01)
        self.assertEqual(res["verified"], ["f1.bin"])
        self.assertEqual(res["sha_mismatch"], [])
        self.assertEqual(len(seen), 3)
        self.assertEqual(set(seen), {"thumb://f1"})  # same thumb_url only

    def test_persistent_mismatch_reported_after_bounded_retries(self):
        e = entry("f1.bin", b"good", "thumb://f1")
        calls = []

        def fake_fetch(url, retries):
            calls.append(url)
            return b"always-wrong"

        with mock.patch.object(mm, "fetch", fake_fetch):
            res = mm.materialize([e], self.out, None, 2, 2, 0.01)
        self.assertEqual(res["verified"], [])
        self.assertEqual(len(res["sha_mismatch"]), 1)
        self.assertEqual(res["sha_mismatch"][0]["attempts"], 2)
        self.assertEqual(len(calls), 2)
        self.assertFalse((self.out / "f1.bin").exists())

    def test_fetch_error_is_missing(self):
        e = entry("f1.bin", b"good", "thumb://f1")

        def fake_fetch(url, retries):
            raise RuntimeError("boom")

        with mock.patch.object(mm, "fetch", fake_fetch):
            res = mm.materialize([e], self.out, None, 1, 2, 0.01)
        self.assertEqual(res["missing"], ["f1.bin"])
        self.assertEqual(len(res["fetch_errors"]), 1)

    def test_main_exits_nonzero_and_writes_report_when_incomplete(self):
        man_path = self.root / "man.json"
        man_path.write_text(json.dumps(
            {"counts_by_group": {}, "entries": [entry("f1.bin", b"good")]}),
            encoding="utf-8")

        def fake_fetch(url, retries):
            return b"wrong"

        argv = ["materialize_manifest.py", "--manifest", str(man_path),
                "--out", str(self.out), "--report",
                str(self.root / "rep.json"), "--sha-retries", "1",
                "--sha-backoff", "0.01"]
        with mock.patch.object(mm, "fetch", fake_fetch):
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        mm.main()
        self.assertNotEqual(ctx.exception.code, 0)
        rep = json.loads((self.root / "rep.json").read_text(encoding="utf-8"))
        self.assertFalse(rep["complete"])
        self.assertEqual(rep["sha_mismatch"][0]["file"], "f1.bin")
        self.assertEqual(rep["retry_policy"]["sha_tries"], 1)
        self.assertEqual(rep["source_policy"].split(";")[0],
                         "thumb_url only (original_url drifted)")


if __name__ == "__main__":
    unittest.main()
