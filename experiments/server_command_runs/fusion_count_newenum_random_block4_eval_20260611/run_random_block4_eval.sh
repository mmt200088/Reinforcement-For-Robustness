#!/usr/bin/env bash
set -euo pipefail
cd /hy-tmp/fusion_count_newenum_513a1ff_20260611_005952/src
export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
python3 scripts/run_fusion_count_action_eval.py \
  --action-dir experiments/server_command_runs/fusion_count_newenum_random_block4_eval_20260611/action_configs \
  --map-report experiments/server_command_runs/fusion_count_newenum_random_block4_eval_20260611/fusion_count_map_report.json \
  --output-root experiments/server_command_runs/fusion_count_newenum_random_block4_eval_20260611/paean_outputs \
  --output-json experiments/server_command_runs/fusion_count_newenum_random_block4_eval_20260611/fusion_count_action_eval_results.json \
  --output-html reports/html_reports/20260611_mrpc_fusion_count_action_eval_newenum_random_block4_results.html \
  --repeat 5 \
  --batch-size 64 \
  --rescale-optimizer-root Rescale_optimizer \
  --force
