#!/usr/bin/env bash
set -euo pipefail
source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
conda activate llm_ist
echo "[env] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "[env] python=$(command -v python)"
python -c "import sys, fire; print(\"[env] executable=\" + sys.executable); print(\"[env] fire-ok\")"
bash experiments/server_command_runs/active_command_2026-05-17_diagnose_invalid_blocks.sh
