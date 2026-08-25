# Reinforcement for Robustness

This repository searches efficient approximation and MPC configurations for
BERT sequence-classification models.

## Supported Profiles

- Models: `bert-base`, `bert-large`
- GLUE tasks: `mrpc`, `rte`, `sst2`

All searches use a fixed, stratified 256-example subset of the GLUE training
split. The full validation split is reserved for final evaluation.

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- Hugging Face Transformers and Datasets
- scikit-learn, NumPy, and the dependencies in `requirements.txt`
- A generated train-probe fixture and fusion maps for the selected profile

The Stage-2 path uses the real in-process Rescale optimizer. GPU assignment is
elastic: every healthy visible GPU is used for deterministic reward trials.

## Stage 1

Start a new run:

```bash
bash llama_7B_LayerImportance.sh run rl \
  --preset bert-base-mrpc-stage1-rl --fresh
```

Resume the same run by omitting `--fresh`:

```bash
bash llama_7B_LayerImportance.sh run rl \
  --preset bert-base-mrpc-stage1-rl
```

Replace `bert-base-mrpc` with any supported model/task pair.

## Stage 2

Start a new layerwise PPO run:

```bash
bash llama_7B_LayerImportance.sh run rl \
  --preset bert-base-mrpc-stage2-rl --fresh
```

Resume by running the same command without `--fresh`. The production preset
uses 150,000 maximum episodes, PPO rollout 120, learning rate `5e-5`, precision
tolerance `0.001`, stability multiplier `2.0`, three online trials, baseline
`5x3`, and three independent 15-trial search-gate banks.

## Comparators

The formal comparators use BERT-base MRPC and run their own Stage 1, bind that
result into Stage 2, and perform strict top-5 selection.

```bash
bash llama_7B_LayerImportance.sh run bo_rf --fresh
bash llama_7B_LayerImportance.sh run greedy --fresh
bash llama_7B_LayerImportance.sh run coinn_ga --fresh
```

Use `--comparator-stage1-only` to stop after Stage 1. Omit `--fresh` to resume.

## Final Evaluation

```bash
bash llama_7B_LayerImportance.sh eval \
  --preset mrpc-final-eval-only \
  --resume-from "Parting Chapter/persistent/rl/bert-base/mrpc/<run>"
```

Paean reuses the selected action, the calibrated fusion map, and the in-process
Rescale materialization path.

## Outputs

- Stage 1: `Parting Chapter/stage1/<model task>/`
- Stage 2: `Parting Chapter/persistent/rl/<model>/<task>/<constraints>/`
- Comparators: `Parting Chapter/persistent/<algorithm>/...`
- Structured records: `rl_training_data_points/`
- Paean final evaluation: `Paean/outputs/`

Generated outputs are intentionally ignored by Git. Historical training
artifacts are stored on `codex/archive-training-artifacts-20260825`; unrelated
experiments and generated reports are on
`codex/experiment-unrelated-artifacts-20260825`. The repository keeps the final
configuration JSON files and one compact example under
`examples/representative_rl_log/`.

Place externally supplied model weights in `local_assets/models/` and datasets
in `local_assets/datasets/`. Their contents remain local and are never tracked.

Use `bash llama_7B_LayerImportance.sh --list-presets` to list available
production presets.
