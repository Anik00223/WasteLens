"""Tests for src/restore_teacher_ckpt.py: Drive restore + md5 gate."""
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import restore_teacher_ckpt as rc  # noqa: E402

REAL_MD5 = "8735019226412c4d67b0a17269dc38ad"
REAL_BYTES = 13583639


class RestoreTeacherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.drive = self.root / "drive"
        self.path = self.root / "models" / "ckpt.keras"
        self.bytes = b"synthetic-checkpoint"
        self.old = (rc.REGISTERED_MD5, rc.REGISTERED_BYTES,
                    rc.REGISTERED_SHA256)
        rc.REGISTERED_MD5 = hashlib.md5(self.bytes).hexdigest()
        rc.REGISTERED_BYTES = len(self.bytes)
        rc.REGISTERED_SHA256 = hashlib.sha256(self.bytes).hexdigest()

    def tearDown(self):
        (rc.REGISTERED_MD5, rc.REGISTERED_BYTES,
         rc.REGISTERED_SHA256) = self.old
        self.tmp.cleanup()

    def _drive_ckpt(self):
        p = self.drive / rc.DRIVE_SUBDIR / self.path.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.bytes)
        return p

    def test_restores_from_drive_and_verifies(self):
        self._drive_ckpt()
        with redirect_stdout(io.StringIO()):
            code = rc.ensure(self.drive, self.path)
        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes(), self.bytes)

    def test_mirrors_verified_local_copy_to_drive(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.bytes)
        with redirect_stdout(io.StringIO()):
            code = rc.ensure(self.drive, self.path)
        self.assertEqual(code, 0)
        self.assertEqual(
            (self.drive / rc.DRIVE_SUBDIR / self.path.name).read_bytes(),
            self.bytes)

    def test_missing_everywhere_fails_with_actionable_instructions(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = rc.ensure(self.drive, self.path)
        text = out.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("STOP", text)
        self.assertIn("Upload", text)
        self.assertIn(rc.REGISTERED_MD5, text)
        self.assertIn("Do NOT train from any other checkpoint", text)

    def test_wrong_local_copy_is_replaced_by_verified_drive_copy(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"corrupt")
        self._drive_ckpt()
        with redirect_stdout(io.StringIO()):
            code = rc.ensure(self.drive, self.path)
        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes(), self.bytes)

    def test_wrong_local_and_wrong_drive_fails(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"corrupt")
        p = self.drive / rc.DRIVE_SUBDIR / self.path.name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"also-corrupt")
        with redirect_stdout(io.StringIO()):
            code = rc.ensure(self.drive, self.path)
        self.assertEqual(code, 1)

    def test_shipped_checkpoint_matches_registered_values(self):
        real = ROOT / "models/checkpoints/wastelens_rej_shipped_best.keras"
        if not real.exists():
            self.skipTest("shipped checkpoint not present locally")
        rc.REGISTERED_MD5 = REAL_MD5
        rc.REGISTERED_BYTES = REAL_BYTES
        self.assertEqual(rc.md5_file(real), REAL_MD5)
        self.assertEqual(real.stat().st_size, REAL_BYTES)


if __name__ == "__main__":
    unittest.main()
