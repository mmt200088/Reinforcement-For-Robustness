#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
source /var/tmp/root-home/miniconda3/etc/profile.d/conda.sh
conda activate llm_ist

# Run all 8 GLUE tasks (uses all training samples by default)
mkdir -p gelu_analysis

nohup python -u analyze_gelu_distribution.py \
  --output_dir gelu_analysis \
  --batch_size 32 \
  --max_length 128 \
  --device cuda \
  > gelu_analysis/run.log 2>&1 &
echo $! > gelu_analysis/pid.txt
disown

# ---- Alternative usages ----
# Run specific tasks only:
#   python analyze_gelu_distribution.py --tasks sst2 mrpc cola --output_dir gelu_analysis

# Limit training samples (faster, good for quick testing):
#   python analyze_gelu_distribution.py --max_samples 5000 --output_dir gelu_analysis

# Run on CPU:
#   python analyze_gelu_distribution.py --device cpu --batch_size 16 --output_dir gelu_analysis
