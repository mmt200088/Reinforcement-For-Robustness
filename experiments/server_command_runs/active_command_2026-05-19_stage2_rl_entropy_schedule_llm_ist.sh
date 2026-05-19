#!/usr/bin/env bash
set -euo pipefail
source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
conda activate llm_ist
export GLUE_LOCAL_DATASET_DIR=/var/tmp/root-home/datasets/glue_local
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
echo CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV
which python
python -V
bash experiments/server_command_runs/active_command_2026-05-19_stage2_rl_entropy_schedule.sh
