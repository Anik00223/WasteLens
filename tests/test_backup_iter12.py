"""Tests for src/backup_iter12.py: save/restore, sha verification, missing."""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import backup_iter12 as b  # noqa: E402


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.drive = self.root / "drive"
        self.items = ["models/checkpoints/x.keras", "docs/a.json"]
        (self.repo / "models/checkpoints").mkdir(parents=True)
        (self.repo / "docs").mkdir()
        (self.repo / self.items[0]).write_bytes(b"ckpt-bytes")
        (self.repo / self.items[1]).write_text("{}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_then_restore_roundtrip(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = b.run_action("save", self.drive, self.items, self.repo)
        self.assertEqual(rc, 0, out.getvalue())
        for rel in self.items:
            self.assertTrue((self.drive / rel).exists())
        for rel in self.items:                     # wipe repo copies
            (self.repo / rel).unlink()
        with redirect_stdout(out):
            rc = b.run_action("restore", self.drive, self.items, self.repo)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertEqual((self.repo / self.items[0]).read_bytes(),
                         b"ckpt-bytes")
        self.assertEqual(b.sha(self.repo / self.items[0]),
                         b.sha(self.drive / self.items[0]))

    def test_sha_mismatch_after_copy_is_a_failure(self):
        real_copy2 = b.shutil.copy2

        def tampered(src, dst, *a, **k):
            real_copy2(src, dst, *a, **k)
            Path(dst).write_bytes(b"corrupted")

        out = io.StringIO()
        with mock.patch.object(b.shutil, "copy2", tampered):
            with redirect_stdout(out):
                rc = b.run_action("save", self.drive, self.items, self.repo)
        self.assertEqual(rc, 1)
        self.assertIn("FAILED", out.getvalue())

    def test_missing_sources_are_reported_and_nothing_copied_fails(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = b.run_action("restore", self.drive, self.items, self.repo)
        text = out.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn("SKIP (missing)", text)
        self.assertIn("nothing copied", text)

    def test_partial_restore_is_ok_when_something_was_copied(self):
        (self.drive / self.items[1]).parent.mkdir(parents=True)
        (self.drive / self.items[1]).write_text("{}", encoding="utf-8")
        for rel in self.items:
            (self.repo / rel).unlink()
        with redirect_stdout(io.StringIO()):
            rc = b.run_action("restore", self.drive, self.items, self.repo)
        self.assertEqual(rc, 0)

    def test_status_counts(self):
        (self.drive / self.items[0]).parent.mkdir(parents=True)
        (self.drive / self.items[0]).write_bytes(b"x")
        out = io.StringIO()
        with redirect_stdout(out):
            rc = b.status(self.drive, self.items, self.repo)
        self.assertEqual(rc, 0)
        self.assertIn("2/2 local", out.getvalue())
        self.assertIn("1/2 on drive", out.getvalue())

    def test_dir_copy_verifies_every_file(self):
        srcdir = self.root / "srcdir"
        srcdir.mkdir()
        (srcdir / "f1").write_bytes(b"1")
        (srcdir / "f2").write_bytes(b"2")
        dst = self.root / "dstdir"
        b.copy_verified(srcdir, dst)
        self.assertTrue((dst / "f1").exists())
        self.assertEqual(b.dir_hashes(srcdir), b.dir_hashes(dst))


if __name__ == "__main__":
    unittest.main()
