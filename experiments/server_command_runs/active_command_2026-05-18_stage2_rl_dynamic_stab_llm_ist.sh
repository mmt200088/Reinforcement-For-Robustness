#!/usr/bin/env bash
set -euo pipefail
source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
conda activate llm_ist
export GLUE_LOCAL_DATASET_DIR=/var/tmp/root-home/datasets/glue_local
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
echo "[env] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "[env] python=$(command -v python)"
echo "[env] GLUE_LOCAL_DATASET_DIR=${GLUE_LOCAL_DATASET_DIR}"
echo "[env] HF_ENDPOINT=${HF_ENDPOINT}"
python -c "import sys, fire; print(\"[env] executable=\" + sys.executable); print(\"[env] fire-ok\")"
bash experiments/server_command_runs/active_command_2026-05-18_stage2_rl_dynamic_stab.sh
