# src/build_iter12_notebook_v2.py
"""Build notebooks/iteration12_distillation_colab.ipynb cell by cell.

PowerShell-safe: invoked as `py src/build_iter12_notebook.py`, no heredoc.
Cells use plain python - Colab magics (!cmd / %cd) live only in string
literals, never in this script's own syntax.

The notebook checks out the pinned commit itself (git checkout --force)
and asserts HEAD == PINNED, so a moved main tip can never make the pin a
lie (the old stale-pin failure mode). Default pin = git HEAD at build
time (the commit that carries the fixed pipeline); --pin overrides.

Locks this notebook enforces before training may start:
  * commit pin checkout + assert (fails loudly on any mismatch),
  * CRLF checkout of both dataset manifests (registered MD5 bytes),
  * teacher checkpoint md5 8735019226412c4d67b0a17269dc38ad restored
    from Drive (fails loudly with upload instructions if absent),
  * materializer 94/94 + 131/131 byte+sha locked (thumb_url only),
  * iter12_preflight gate (manifest locks + completeness + teacher).
"""
import argparse
import json
import subprocess
from pathlib import Path

OUT = Path("notebooks/iteration12_distillation_colab.ipynb")


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines}


def build(pin: str) -> list:
    cells = []
    cells.append(md("# WasteLens - Iteration 12 (Colab GPU)\n"))
    cells.append(md(
        "Protocol: docs/rejection_experiment/"
        "iteration12_distillation_protocol.md (registered pre-training,"
        " commit 3b88e70). Run every code cell top to bottom; every heavy"
        " cell STOPS the run on any non-zero exit - nothing may be"
        " skipped, reordered, or re-run after completion.\n"))
    cells.append(md("## S0 - environment\n"))
    cells.append(code([
        "!nvidia-smi\n",
    ]))
    cells.append(code([
        "import sys, platform\n",
        "import tensorflow as tf\n",
        "print('Python:', sys.version.split()[0], '|',"
        " platform.platform())\n",
        "print('TensorFlow:', tf.__version__)\n",
        "gpus = tf.config.list_physical_devices('GPU')\n",
        "print('GPUs:', gpus if gpus else 'NONE - enable a GPU runtime')\n",
        "assert gpus, 'STOP: no GPU visible'\n",
        "import numpy, sklearn, PIL\n",
        "print('numpy:', numpy.__version__,\n",
        "      '| sklearn:', sklearn.__version__,\n",
        "      '| pillow:', PIL.__version__)\n",
    ]))
    cells.append(md("## S1 - pin the repo: checkout the registered commit\n"))
    cells.append(code([
        "import os, subprocess, sys\n",
        "from pathlib import Path\n",
        "\n",
        "PINNED = '" + pin + "'\n",
        "REPO = '/content/WasteLens'\n",
        "\n",
        "if not Path(REPO, 'src', 'train_distill_adaptation.py').exists():\n",
        "    subprocess.run(['git', 'clone',\n",
        "                    'https://github.com/Anik00223/WasteLens.git',\n",
        "                    REPO], check=True)\n",
        "os.chdir(REPO)\n",
        "subprocess.run(['git', 'fetch', 'origin'], check=True)\n",
        "subprocess.run(['git', 'checkout', '--force', PINNED], check=True)\n",
        "\n",
        "head = subprocess.run(['git', 'rev-parse', 'HEAD'],\n",
        "                      capture_output=True, text=True,\n",
        "                      check=True).stdout.strip()\n",
        "origin = subprocess.run(['git', 'rev-parse', 'origin/main'],\n",
        "                        capture_output=True, text=True,\n",
        "                        check=True).stdout.strip()\n",
        "print('HEAD        :', head)\n",
        "print('PINNED      :', PINNED)\n",
        "print('origin/main :', origin, '(informational - the pin wins)')\n",
        "assert head == PINNED, ('STOP: HEAD ' + head + ' != PINNED '\n",
        "                        + PINNED)\n",
        "MANIFESTS = ['docs/rejection_experiment/"
        "adaptation_set_manifest.json',\n",
        "             'docs/rejection_experiment/fresh_set_manifest.json']\n",
        "print('repo pinned OK')\n",
    ]))
    cells.append(code([
        "for rel in MANIFESTS:\n",
        "    attrs = subprocess.run(['git', 'check-attr', 'text', 'eol',"
        " '--', rel],\n",
        "                           capture_output=True, text=True,\n",
        "                           check=True).stdout\n",
        "    print(attrs.strip())\n",
        "    assert 'eol: crlf' in attrs, (\n",
        "        'STOP: ' + rel + ' is not checked out with the registered"
        " CRLF bytes (the dataset locks are MD5s of the CRLF form)')\n",
        "print('manifest EOL pin OK')\n",
        "\n",
        "def run(cmd, log=None):\n",
        "    \"\"\"Run a repo command; any non-zero exit stops the cell.\"\"\"\n",
        "    cmd = [str(c) for c in cmd]\n",
        "    print('+', ' '.join(cmd), flush=True)\n",
        "    if log:\n",
        "        with open(log, 'a', encoding='utf-8') as fh:\n",
        "            p = subprocess.Popen(cmd, cwd=REPO,"
        " stdout=subprocess.PIPE,\n",
        "                                 stderr=subprocess.STDOUT,"
        " text=True, bufsize=1)\n",
        "            for line in p.stdout:\n",
        "                print(line, end='', flush=True)\n",
        "                fh.write(line)\n",
        "            rc = p.wait()\n",
        "        if rc:\n",
        "            raise SystemExit('STOP rc=' + str(rc) + ': '"
        " + ' '.join(cmd) + ' (log: ' + log + ')')\n",
        "    else:\n",
        "        rc = subprocess.run(cmd, cwd=REPO).returncode\n",
        "        if rc:\n",
        "            raise SystemExit('STOP rc=' + str(rc) + ': '"
        " + ' '.join(cmd))\n",
        "\n",
        "PY = sys.executable\n",
        "print('runner ready; PY =', PY)\n",
    ]))
    cells.append(md(
        "## S2 - Drive mount + teacher restore (md5-gated)\n\n"
        "One-time manual step: the shipped checkpoint is gitignored, so a"
        " fresh runtime does not have it. Upload\n"
        "`models/checkpoints/wastelens_rej_shipped_best.keras` (md5"
        " `8735019226412c4d67b0a17269dc38ad`, 13,583,639 B) from the"
        " laptop repo to `MyDrive/WasteLens/iteration12/checkpoints/`"
        " before S5/S6. The cell fails loudly with the exact instructions"
        " if it is absent - training never starts from another model.\n"))
    cells.append(code([
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "BK = '/content/drive/MyDrive/WasteLens/iteration12'\n",
        "for sub in ('datasets/adaptation_set', 'datasets/fresh_set',\n",
        "            'checkpoints', 'evidence', 'logs'):\n",
        "    os.makedirs(BK + '/' + sub, exist_ok=True)\n",
        "print('backup root:', BK)\n",
        "run([PY, 'src/restore_teacher_ckpt.py', '--drive', BK])\n",
    ]))
    cells.append(md("## S3 - materialize datasets (thumb_url only,"
                    " byte+sha locked, bounded retries)\n"))
    cells.append(code([
        "for manifest, out, drive_sub, report in (\n",
        "        (MANIFESTS[0], 'scratch/adaptation_set',\n",
        "         'datasets/adaptation_set', 'adaptation_lock_report.json'),\n",
        "        (MANIFESTS[1], 'scratch/fresh_set',\n",
        "         'datasets/fresh_set', 'fresh_lock_report.json')):\n",
        "    run([PY, '-u', 'src/materialize_manifest.py',\n",
        "         '--manifest', manifest, '--out', out,\n",
        "         '--drive', BK + '/' + drive_sub,\n",
        "         '--report', BK + '/datasets/' + report])\n",
        "print('datasets materialized: 94/94 + 131/131 or this cell stopped')\n",
    ]))
    cells.append(md("## S4 - strict preflight (manifest locks + 94/94 +"
                    " 131/131 + teacher md5)\n"))
    cells.append(code([
        "run([PY, '-u', 'src/iter12_preflight.py',\n",
        "     '--report', BK + '/datasets/head12_preflight.json'])\n",
    ]))
    cells.append(md("## S5 - train seed 42 (6 epochs, FINAL epoch ="
                    " candidate) + back up immediately\n"))
    cells.append(code([
        "run([PY, '-u', 'src/train_distill_adaptation.py', '--seed', '42'],\n",
        "    log=BK + '/logs/head12s42_train.log')\n",
        "run([PY, 'src/backup_iter12.py', 'save', '--drive', BK, '--seed',"
        " '42'])\n",
        "import glob\n",
        "print('seed-42 checkpoints:', sorted(glob.glob(\n",
        "    'models/checkpoints/wastelens_rej_head12s42*')))\n",
        "print('seed-42 evidence docs:', sorted(glob.glob(\n",
        "    'docs/rejection_experiment/head12s42*')))\n",
    ]))
    cells.append(md("## S6 - train seed 43 (identical recipe) + back up"
                    " immediately\n"))
    cells.append(code([
        "run([PY, '-u', 'src/train_distill_adaptation.py', '--seed', '43'],\n",
        "    log=BK + '/logs/head12s43_train.log')\n",
        "run([PY, 'src/backup_iter12.py', 'save', '--drive', BK, '--seed',"
        " '43'])\n",
        "print('seed-43 checkpoints:', sorted(glob.glob(\n",
        "    'models/checkpoints/wastelens_rej_head12s43*')))\n",
        "print('seed-43 evidence docs:', sorted(glob.glob(\n",
        "    'docs/rejection_experiment/head12s43*')))\n",
    ]))
    cells.append(md("## S7 - evaluate both seeds (frozen threshold 0.0702,"
                    " head12 gate table) + summary\n"))
    cells.append(code([
        "for seed in ('42', '43'):\n",
        "    run([PY, '-u', 'src/eval_head_adaptation.py', '--seed', seed,\n",
        "         '--tag', 'head12', '--protocol',\n",
        "         'iteration12_distillation_protocol.md',\n",
        "         '--gates', 'head12_gates.json'],\n",
        "        log=BK + '/logs/head12s' + seed + '_eval.log')\n",
        "run([PY, '-u', 'src/eval_head_adaptation.py', '--summary',\n",
        "     '--tag', 'head12', '--protocol',\n",
        "     'iteration12_distillation_protocol.md'],\n",
        "    log=BK + '/logs/head12_summary.log')\n",
        "run([PY, 'src/backup_iter12.py', 'save', '--drive', BK])\n",
        "print('eval + decision mirrored; production model/threshold"
        " untouched')\n",
    ]))
    cells.append(md("## S8 - strict evidence bundle (refuses to build if"
                    " anything is missing)\n"))
    cells.append(code([
        "run([PY, 'src/package_head12_evidence.py', '--drive',\n",
        "     BK + '/evidence', '--out', 'head12_evidence.tar.gz'])\n",
        "import os\n",
        "for p in ('head12_evidence.tar.gz',\n",
        "          BK + '/evidence/head12_evidence.tar.gz'):\n",
        "    print(p, os.path.getsize(p), 'bytes')\n",
    ]))
    cells.append(md("## S9 - SAVE NOW (run any time before the runtime"
                    " times out)\n"))
    cells.append(code([
        "run([PY, 'src/backup_iter12.py', 'save', '--drive', BK])\n",
        "print('full artifact set re-mirrored to', BK)\n",
    ]))
    cells.append(md(
        "## S10 - RESTORE from Drive (fresh runtime) + disaster recovery\n\n"
        "1. run S0, S1 (pin), S2 (Drive + teacher restore);\n"
        "2. run this cell - it restores every finished artifact from Drive"
        " and re-materializes the datasets (verified files are reused,"
        " never re-downloaded);\n"
        "3. run the S11 status cell - it lists exactly what exists;\n"
        "4. resume at the FIRST missing step only (S5/S6 for a seed"
        " without _final.keras, then S7, S8). Never re-run a completed"
        " training cell;\n"
        "5. nothing on Drive yet? this cell stops with a clear message -"
        " start from S3 instead.\n"))
    cells.append(code([
        "run([PY, 'src/backup_iter12.py', 'restore', '--drive', BK])\n",
        "for manifest, out, drive_sub, report in (\n",
        "        (MANIFESTS[0], 'scratch/adaptation_set',\n",
        "         'datasets/adaptation_set', 'adaptation_lock_report.json'),\n",
        "        (MANIFESTS[1], 'scratch/fresh_set',\n",
        "         'datasets/fresh_set', 'fresh_lock_report.json')):\n",
        "    run([PY, '-u', 'src/materialize_manifest.py',\n",
        "         '--manifest', manifest, '--out', out,\n",
        "         '--drive', BK + '/' + drive_sub,\n",
        "         '--report', BK + '/datasets/' + report])\n",
        "print('restore complete - run the S11 status cell for the resume"
        " point')\n",
    ]))
    cells.append(md("## S11 - DISASTER RECOVERY STATUS (pure python)\n"))
    cells.append(code([
        "import glob, hashlib, os, subprocess\n",
        "\n",
        "def md5(p):\n",
        "    h = hashlib.md5()\n",
        "    with open(p, 'rb') as f:\n",
        "        for chunk in iter(lambda: f.read(1 << 20), b''):\n",
        "            h.update(chunk)\n",
        "    return h.hexdigest()\n",
        "\n",
        "def n_files(d):\n",
        "    if not os.path.isdir(d):\n",
        "        return 0\n",
        "    return len([f for f in os.listdir(d)\n",
        "                if os.path.isfile(os.path.join(d, f))])\n",
        "\n",
        "head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,\n",
        "                      capture_output=True, text=True).stdout.strip()\n",
        "print('HEAD               :', head)\n",
        "print('pinned as expected :', head == PINNED)\n",
        "print('git status --short :')\n",
        "print(subprocess.run(['git', 'status', '--short'], cwd=REPO,\n",
        "                     capture_output=True, text=True).stdout.strip()\n",
        "      or '  (clean)')\n",
        "print('adaptation_set     :', n_files('scratch/adaptation_set'),"
        " '/ 94')\n",
        "print('fresh_set          :', n_files('scratch/fresh_set'),"
        " '/ 131')\n",
        "for rel in MANIFESTS:\n",
        "    print('manifest md5       :', md5(rel), '<-', rel)\n",
        "teacher = 'models/checkpoints/wastelens_rej_shipped_best.keras'\n",
        "print('teacher ckpt md5   :',\n",
        "      md5(teacher) if os.path.exists(teacher) else 'MISSING',\n",
        "      '(want 8735019226412c4d67b0a17269dc38ad)')\n",
        "print('threshold          : 0.0702 (frozen; production untouched)')\n",
        "print('seed-42 ckpts      :', sorted(os.path.basename(p) for p in\n",
        "      glob.glob('models/checkpoints/wastelens_rej_head12s42*')))\n",
        "print('seed-43 ckpts      :', sorted(os.path.basename(p) for p in\n",
        "      glob.glob('models/checkpoints/wastelens_rej_head12s43*')))\n",
        "print('seed-42 docs       :', sorted(os.path.basename(p) for p in\n",
        "      glob.glob('docs/rejection_experiment/head12s42*')))\n",
        "print('seed-43 docs       :', sorted(os.path.basename(p) for p in\n",
        "      glob.glob('docs/rejection_experiment/head12s43*')))\n",
        "print('decision/evidence  :', [p for p in (\n",
        "    'docs/rejection_experiment/iteration12_decision.md',\n",
        "    'docs/rejection_experiment/head12_seed_consistency.json',\n",
        "    'head12_evidence.tar.gz',\n",
        "    BK + '/evidence/head12_evidence.tar.gz') if os.path.exists(p)])\n",
        "print('drive root listing :',\n",
        "      sorted(os.listdir(BK)) if os.path.isdir(BK) else 'MISSING')\n",
        "print()\n",
        "print('RESUME RULE: start at S5/S6 for any seed without a"
        " _final.keras, then S7 and S8; never re-run a completed training"
        " cell.')\n",
    ]))
    cells.append(md(
        "## Locks and safety rails (quick reference)\n\n"
        "* commit pin: this notebook runs `git checkout --force <PINNED>`"
        " and asserts HEAD == PINNED; PINNED is the commit that carries the"
        " fixed pipeline.\n"
        "* dataset manifests: md5 `acae0933aca123c835c1e55396af182a`"
        " (94 files) and `99540f56efb95e2c26b2b6cc9414ff5d` (131 images);"
        " CRLF checkout pinned by `.gitattributes` so Linux reproduces the"
        " registered bytes; the trainer re-checks the lock itself.\n"
        "* teacher: md5 `8735019226412c4d67b0a17269dc38ad`, restored from"
        " Drive and verified before S5/S6 may run.\n"
        "* threshold: 0.0702 (frozen); the production model is untouched by"
        " every cell here.\n"
        "* every heavy cell stops the run on a non-zero exit; nothing is"
        " silently skipped.\n"))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", default=None,
                    help="commit the notebook must check out (default: git"
                         " HEAD at build time)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    pin = args.pin or git_head()
    cells = build(pin)
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"accelerator": "GPU",
                       "kernelspec": {"name": "python3",
                                      "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", out, "cells:", len(cells), "pin:", pin)


if __name__ == "__main__":
    main()
