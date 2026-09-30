"""Tests for src/iter12_preflight.py: locks, byte locks, constants sync."""
import hashlib
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import iter12_preflight as pf  # noqa: E402


def entry(name, raw):
    return {"local_name": name, "group": "g",
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
            "thumb_url": "thumb://x"}


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = b"image-bytes"
        self.adir = self.root / "adapt"
        self.adir.mkdir()
        (self.adir / "a.jpg").write_bytes(self.raw)
        self.man = self.root / "man.json"
        self.man.write_text(
            json.dumps({"entries": [entry("a.jpg", self.raw)]}) + "\n",
            encoding="utf-8")
        self.md5 = hashlib.md5(self.man.read_bytes()).hexdigest()
        self.teacher = self.root / "teacher.keras"
        self.teacher.write_bytes(b"model-bytes")
        self.tmd5 = hashlib.md5(b"model-bytes").hexdigest()
        self.reg = {"adapt_manifest_md5": self.md5,
                    "fresh_manifest_md5": self.md5,
                    "teacher_md5": self.tmd5,
                    "adapt_file_count": 1, "fresh_file_count": 1}

    def tearDown(self):
        self.tmp.cleanup()

    def run_checks(self, **kw):
        args = dict(adapt_manifest=self.man, fresh_manifest=self.man,
                    adapt_dir=self.adir, fresh_dir=self.adir,
                    teacher=self.teacher, registry=self.reg)
        args.update(kw)
        return pf.run_checks(**args)

    def test_clean_fixture_passes(self):
        res = self.run_checks()
        self.assertTrue(all(r["ok"] for r in res))

    def test_tampered_dataset_file_fails_byte_lock(self):
        (self.adir / "a.jpg").write_bytes(b"drifted")
        res = self.run_checks()
        self.assertFalse(all(r["ok"] for r in res))
        bad = [r for r in res if r["check"] == "adapt_files_byte_locked"][0]
        self.assertEqual(bad["extra"]["bad_first10"][0]["why"],
                         "bytes/sha256 drift")

    def test_missing_file_fails_completeness(self):
        (self.adir / "a.jpg").unlink()
        res = self.run_checks()
        self.assertFalse(all(r["ok"] for r in res))

    def test_wrong_file_count_fails(self):
        (self.adir / "extra.jpg").write_bytes(b"x")
        res = self.run_checks()
        counts = [r for r in res
                  if r["check"] == "adapt_dir_file_count"][0]
        self.assertFalse(counts["ok"])

    def test_manifest_md5_drift_fails(self):
        res = self.run_checks(
            registry=dict(self.reg, adapt_manifest_md5="0" * 32))
        chk = [r for r in res if r["check"] == "adapt_manifest_md5"][0]
        self.assertFalse(chk["ok"])

    def test_teacher_md5_drift_fails(self):
        res = self.run_checks(registry=dict(self.reg, teacher_md5="0" * 32))
        chk = [r for r in res if r["check"] == "teacher_md5"][0]
        self.assertFalse(chk["ok"])

    def test_main_exit_code_and_report(self):
        (self.adir / "a.jpg").write_bytes(b"drifted")
        argv = ["iter12_preflight.py", "--adapt-manifest", str(self.man),
                "--fresh-manifest", str(self.man),
                "--adapt-dir", str(self.adir), "--fresh-dir", str(self.adir),
                "--teacher", str(self.teacher),
                "--report", str(self.root / "rep.json")]
        with mock.patch.object(pf, "REGISTRY", self.reg):
            with mock.patch.object(sys, "argv", argv):
                with redirect_stdout(StringIO()):
                    with self.assertRaises(SystemExit) as ctx:
                        pf.main()
        self.assertEqual(ctx.exception.code, 1)
        rep = json.loads((self.root / "rep.json").read_text(encoding="utf-8"))
        self.assertFalse(rep["all_pass"])

    def test_registry_mirrors_trainer_constants(self):
        text = (ROOT / "src" / "train_head_adaptation.py").read_text(
            encoding="utf-8")
        pats = {
            "adapt_manifest_md5":
                r'ADAPT_MANIFEST_MD5 = "([0-9a-f]{32})"',
            "fresh_manifest_md5":
                r'FRESH_MANIFEST_MD5 = "([0-9a-f]{32})"',
            "teacher_md5":
                r'SHIPPED_CHECKPOINT_MD5 = "([0-9a-f]{32})"',
        }
        for key, pat in pats.items():
            m = re.search(pat, text)
            self.assertTrue(m, key)
            self.assertEqual(m.group(1), pf.REGISTRY[key])
        m = re.search(r"ADAPT_DIR_FILE_COUNT = (\d+)", text)
        self.assertEqual(int(m.group(1)), pf.REGISTRY["adapt_file_count"])

    def test_gitattributes_pins_manifest_eol(self):
        text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        for rel in ("docs/rejection_experiment/adaptation_set_manifest.json",
                    "docs/rejection_experiment/fresh_set_manifest.json"):
            lines = [ln for ln in text.splitlines() if ln.startswith(rel)]
            self.assertTrue(lines, rel)
            self.assertIn("eol=crlf", lines[0])


if __name__ == "__main__":
    unittest.main()
