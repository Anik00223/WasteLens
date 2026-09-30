"""Build notebooks/iteration12_distillation_colab.ipynb cell by cell.

PowerShell-safe: invoked as `py src/build_iter12_notebook.py`, no heredoc.
Cells use plain python - Colab magics (!cmd / %cd) live only in string
literals, never in this script's own syntax.
"""
import json
from pathlib import Path

OUT = Path("notebooks/iteration12_distillation_colab.ipynb")
PINNED = "d930d3c07ec183e1fa8fbf3c686488bac8922612"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(lines):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines}


cells = []
cells.append(md("# WasteLens - Iteration 12 (Colab GPU)\n"))
cells.append(md("Compute env is Colab GPU; protocol unchanged from "
                "docs/rejection_experiment/iteration12_distillation_protocol.md "
                "(registered pre-training, commit 3b88e70). Run every code "
                "cell top to bottom; do not skip or reorder."))
cells.append(md("## S0 - environment"))
cells.append(code(["!nvidia-smi\n"]))
cells.append(code([
    "import sys, platform\n",
    "import tensorflow as tf\n",
    "print('Python:', sys.version.split()[0], '|', platform.platform())\n",
    "print('TensorFlow:', tf.__version__)\n",
    "gpus = tf.config.list_physical_devices('GPU')\n",
    "print('GPUs:', gpus if gpus else 'NONE - enable a GPU runtime')\n",
    "assert gpus, 'STOP: no GPU visible'\n",
    "import numpy, sklearn, PIL\n",
    "print('numpy:', numpy.__version__,\n",
    "      '| sklearn:', sklearn.__version__,\n",
    "      '| pillow:', PIL.__version__)\n",
]))
cells.append(md("## S1 - repo sync (start from main, pin registered commit)"))
cells.append(code([
    "import subprocess\n",
    "from pathlib import Path\n",
    "REPO = Path('/content/WasteLens')\n",
    "PINNED = '" + PINNED + "'\n",
    "if not (REPO / 'src' / 'train_distill_adaptation.py').exists():\n",
    "    subprocess.run(['git', 'clone',\n",
    "                    'https://github.com/Anik00223/WasteLens.git',\n",
    "                    str(REPO)], check=True)\n",
]))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!git fetch origin main\n",
    "!git checkout main\n",
    "!git rev-parse HEAD\n",
    "!git rev-parse origin/main\n",
    "!git status --short\n",
]))
cells.append(code([
    "import subprocess\n",
    "head = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,\n",
    "                      text=True, cwd='/content/WasteLens').stdout.strip()\n",
    "print('HEAD:', head)\n",
    "assert head == '" + PINNED + "', 'STOP: HEAD != pinned commit'\n",
    "print('repo pinned OK')\n",
]))
cells.append(md("## S2 - Drive setup + resume-aware dataset (thumb_url + SHA locks)"))
cells.append(code([
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "BK = '/content/drive/MyDrive/WasteLens/iteration12'\n",
    "!mkdir -p \"$BK/datasets/adaptation_set\" \"$BK/datasets/fresh_set\" "
    "\"$BK/checkpoints\" \"$BK/evidence\" \"$BK/logs\"\n",
    "print('backup root:', BK)\n",
]))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python src/materialize_manifest.py "
    "--manifest docs/rejection_experiment/adaptation_set_manifest.json "
    "--out scratch/adaptation_set "
    "--drive \"$BK/datasets/adaptation_set\" "
    "--report \"$BK/datasets/adaptation_lock_report.json\"\n",
    "!python src/materialize_manifest.py "
    "--manifest docs/rejection_experiment/fresh_set_manifest.json "
    "--out scratch/fresh_set "
    "--drive \"$BK/datasets/fresh_set\" "
    "--report \"$BK/datasets/fresh_lock_report.json\"\n",
]))
cells.append(code([
    "import sys\n",
    "sys.path.insert(0, 'src')\n",
    "import train_head_adaptation as th\n",
    "print(th.lock_dataset())\n",
    "import train_adaptation as wl10\n",
    "print('shipped md5:', wl10.md5_file(th.SHIPPED_CHECKPOINT))\n",
    "print('backup root BK =', BK)\n",
]))
cells.append(md("## S3 - train seed 42 (6 epochs, FINAL epoch = candidate)"))
cells.append(code(["%cd /content/WasteLens\n",
                   "!python -u src/train_distill_adaptation.py --seed 42 "
                   "2>&1 | tee \"$BK/logs/head12s42_train.log\"\n"]))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python src/backup_iter12.py save --drive \"$BK\" --seed 42\n",
    "!ls -la models/checkpoints/*head12s42* docs/rejection_experiment/head12s42*\n",
    "print('seed-42 artifacts mirrored to $BK')\n",
]))
cells.append(md("## S4 - train seed 43 (identical recipe)"))
cells.append(code(["%cd /content/WasteLens\n",
                   "!python -u src/train_distill_adaptation.py --seed 43 "
                   "2>&1 | tee \"$BK/logs/head12s43_train.log\"\n"]))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python src/backup_iter12.py save --drive \"$BK\" --seed 43\n",
    "!ls -la models/checkpoints/*head12s43* docs/rejection_experiment/head12s43*\n",
    "print('seed-43 artifacts mirrored to $BK')\n",
]))
cells.append(md("## S5 - evaluate both seeds (threshold 0.0702, gates per protocol)"))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python -u src/eval_head_adaptation.py --seed 42 "
    "--tag head12 --protocol iteration12_distillation_protocol.md "
    "--gates head12_gates.json\n",
    "!python -u src/eval_head_adaptation.py --seed 43 "
    "--tag head12 --protocol iteration12_distillation_protocol.md "
    "--gates head12_gates.json\n",
]))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python -u src/eval_head_adaptation.py --summary "
    "--tag head12 --protocol iteration12_distillation_protocol.md\n",
]))
cells.append(md("## S6 - strict evidence bundle (repo + Drive, fail loudly if incomplete)"))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python src/package_head12_evidence.py --drive \"$BK/evidence\" "
    "--out head12_evidence.tar.gz\n",
    "!cp head12_evidence.tar.gz \"$BK/evidence/head12_evidence.tar.gz\"\n",
    "!ls -la head12_evidence.tar.gz \"$BK/evidence/head12_evidence.tar.gz\"\n",
    "print('evidence in repo + $BK/evidence')\n",
]))
cells.append(md("## S7 - SAVE CHECKPOINT NOW (manual: run any time before timeout)"))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python src/backup_iter12.py save --drive \"$BK\"\n",
    "print('checkpoint saved to $BK')\n",
]))
cells.append(md("## S8 - RESTORE FROM DRIVE (fresh runtime: datasets + artifacts)"))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!python src/backup_iter12.py restore --drive \"$BK\"\n",
    "!python src/materialize_manifest.py "
    "--manifest docs/rejection_experiment/adaptation_set_manifest.json "
    "--out scratch/adaptation_set "
    "--drive \"$BK/datasets/adaptation_set\" --retries 2\n",
    "!python src/materialize_manifest.py "
    "--manifest docs/rejection_experiment/fresh_set_manifest.json "
    "--out scratch/fresh_set "
    "--drive \"$BK/datasets/fresh_set\" --retries 2\n",
    "print('restore done; resume at the first missing S3/S4/S5 cell')\n",
]))
cells.append(md("## S9 - DISASTER RECOVERY STATUS"))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!echo \"HEAD: $(git rev-parse HEAD)\"; echo \"origin: $(git rev-parse origin/main)\"; git status --short\n",
    "!ls scratch/adaptation_set | wc -l; !ls scratch/fresh_set | wc -l\n",
    "!ls -la models/checkpoints/*head12s42*final.keras models/checkpoints/*head12s43*final.keras 2>/dev/null || echo 'NO head12 final checkpoints yet'\n",
    "!ls -la docs/rejection_experiment/head12s42_gates.json docs/rejection_experiment/head12s43_gates.json docs/rejection_experiment/iteration12_decision.md 2>/dev/null || echo 'NO head12 eval/decision yet'\n",
    "!ls -la head12_evidence.tar.gz \"$BK/evidence/head12_evidence.tar.gz\" 2>/dev/null || echo 'NO evidence archive yet'\n",
    "!ls \"$BK\" 2>/dev/null || echo 'Drive backup root missing'\n",
    "!python -c \"import sys; sys.path.insert(0,'src'); import train_adaptation as w; print('shipped md5:', w.md5_file(w.SHIPPED_CHECKPOINT), '| threshold: 0.0702 (frozen)')\"\n",
]))

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {
          "accelerator": "GPU",
          "kernelspec": {"name": "python3", "display_name": "Python 3"},
          "language_info": {"name": "python"}},
      "cells": cells}
OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("wrote", OUT, "cells:", len(cells))
