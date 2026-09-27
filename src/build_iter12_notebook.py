"""Build notebooks/iteration12_distillation_colab.ipynb cell by cell.

PowerShell-safe: invoked as `py src/build_iter12_notebook.py`, no heredoc.
Cells use plain python - Colab magics (!cmd / %cd) live only in string
literals, never in this script's own syntax.
"""
import json
from pathlib import Path

OUT = Path("notebooks/iteration12_distillation_colab.ipynb")
PINNED = "e017f34dbb487ec21a5633bd965543fbd8de7e9d"


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
cells.append(md("## S2 - dataset + teacher locks"))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!ls scratch/adaptation_set | wc -l\n",
    "!ls scratch/fresh_set | wc -l\n",
]))
cells.append(code([
    "import sys\n",
    "sys.path.insert(0, 'src')\n",
    "import train_head_adaptation as th\n",
    "print(th.lock_dataset())\n",
    "import train_adaptation as wl10\n",
    "print('shipped md5:', wl10.md5_file(th.SHIPPED_CHECKPOINT))\n",
]))
cells.append(md("## S3 - train seed 42 (6 epochs, FINAL epoch = candidate)"))
cells.append(code(["%cd /content/WasteLens\n",
                   "!python -u src/train_distill_adaptation.py --seed 42\n"]))
cells.append(md("## S4 - train seed 43 (identical recipe)"))
cells.append(code(["%cd /content/WasteLens\n",
                   "!python -u src/train_distill_adaptation.py --seed 43\n"]))
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
cells.append(md("## S6 - evidence bundle (download to laptop)"))
cells.append(code([
    "%cd /content/WasteLens\n",
    "!git status --short\n",
    "!ls -la docs/rejection_experiment/head12*\n",
    "!tar -czf /content/head12_evidence.tar.gz docs/rejection_experiment/head12* docs/rejection_experiment/iteration12_decision.md\n",
    "!ls -la /content/head12_evidence.tar.gz\n",
]))
cells.append(code([
    "from google.colab import files\n",
    "files.download('/content/head12_evidence.tar.gz')\n",
]))

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {
          "accelerator": "GPU",
          "kernelspec": {"name": "python3", "display_name": "Python 3"},
          "language_info": {"name": "python"}},
      "cells": cells}
OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("wrote", OUT, "cells:", len(cells))
