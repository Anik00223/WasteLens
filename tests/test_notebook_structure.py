"""Structural checks for the built Iteration-12 Colab notebook."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "src" / "build_iter12_notebook.py"
FAKE_PIN = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


class NotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        target = Path(cls._tmp.name) / "nb.ipynb"
        subprocess.run([sys.executable, str(BUILDER), "--pin", FAKE_PIN,
                        "--out", str(target)], check=True, cwd=str(ROOT))
        cls.nb = json.loads(target.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def code_cells(self):
        return [c for c in self.nb["cells"] if c["cell_type"] == "code"]

    def all_code_text(self):
        return "\n".join("".join(c["source"]) for c in self.code_cells())

    def test_nbformat_and_pin_checkout(self):
        self.assertEqual(self.nb["nbformat"], 4)
        text = self.all_code_text()
        self.assertIn("PINNED = '" + FAKE_PIN + "'", text)
        self.assertIn("'checkout', '--force', PINNED", text)
        self.assertIn("assert head == PINNED", text)

    def test_every_code_cell_compiles_when_magics_are_stripped(self):
        for i, c in enumerate(self.code_cells()):
            lines = []
            for ln in "".join(c["source"]).splitlines():
                if ln.lstrip().startswith(("!", "%")):
                    lines.append("pass  # magic")
                else:
                    lines.append(ln)
            src = "\n".join(lines) + "\n"
            try:
                ast.parse(src)
            except SyntaxError as exc:
                self.fail(f"code cell {i} does not compile: {exc}\n{src}")

    def test_status_cell_is_pure_python(self):
        status = [c for c in self.code_cells()
                  if "RESUME RULE" in "".join(c["source"])]
        self.assertEqual(len(status), 1)
        self.assertNotIn("!", "".join(status[0]["source"]))

    def test_no_invalid_ls_magic_anywhere(self):
        self.assertNotIn("!ls", self.all_code_text())

    def test_required_commands_and_locks_present(self):
        text = self.all_code_text()
        for needle in ("restore_teacher_ckpt.py", "iter12_preflight.py",
                       "materialize_manifest.py", "backup_iter12.py",
                       "package_head12_evidence.py", "head12_gates.json",
                       "8735019226412c4d67b0a17269dc38ad", "0.0702"):
            self.assertIn(needle, text, needle)

    def test_gpu_cell_uses_valid_magic(self):
        self.assertIn("!nvidia-smi", self.all_code_text())


if __name__ == "__main__":
    unittest.main()
