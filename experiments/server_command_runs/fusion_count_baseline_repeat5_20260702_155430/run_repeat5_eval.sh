#!/usr/bin/env bash
set -euo pipefail
cd "/hy-tmp/fusion_count_baseline_repeat5_68176e0_20260702_155408/src"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/hy-tmp/hf_cache
export TRANSFORMERS_CACHE=/hy-tmp/hf_cache/hub
export HF_DATASETS_CACHE=/hy-tmp/hf_cache/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python3 scripts/run_fusion_count_action_eval.py \
  --action-dir "/hy-tmp/fusion_count_baseline_repeat5_68176e0_20260702_155408/src/experiments/server_command_runs/fusion_count_baseline_repeat5_20260702_155430/map_report/action_configs" \
  --map-report "/hy-tmp/fusion_count_baseline_repeat5_68176e0_20260702_155408/src/experiments/server_command_runs/fusion_count_baseline_repeat5_20260702_155430/fusion_count_map_report.json" \
  --output-root "/hy-tmp/fusion_count_baseline_repeat5_68176e0_20260702_155408/src/experiments/server_command_runs/fusion_count_baseline_repeat5_20260702_155430/paean_outputs" \
  --output-json "/hy-tmp/fusion_count_baseline_repeat5_68176e0_20260702_155408/src/experiments/server_command_runs/fusion_count_baseline_repeat5_20260702_155430/fusion_count_action_eval_results.json" \
  --output-html "/hy-tmp/fusion_count_baseline_repeat5_68176e0_20260702_155408/src/experiments/server_command_runs/fusion_count_baseline_repeat5_20260702_155430/fusion_count_action_eval_report.html" \
  --repeat 5 \
  --batch-size 64 \
  --rescale-optimizer-root Rescale_optimizer \
  --force
