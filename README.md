# Reinforcement for Robustness

This repository searches efficient approximation and MPC configurations for
BERT sequence-classification models.

## Supported Profiles

- Models: `bert-base`, `bert-large`
- GLUE tasks: `mrpc`, `rte`, `sst2`

Every search uses the same fixed, stratified 256-example subset of the GLUE
training split. Full GLUE validation is reserved for final evaluation.

## Layout

```text
src/rfr/preparation/   Probe data, fusion maps, and Rescale baselines
src/rfr/search/        RL, BO-RF, Greedy, COINN-GA, and GPU runtime
src/rfr/evaluation/    Full-validation evaluation
src/rfr/cli/           Command-line entrypoints
configs/               Presets and immutable production configuration
outputs/               Generated artifacts grouped by algorithm
```

## Requirements

- Python 3.10+
- CUDA-enabled PyTorch
- Dependencies from `requirements.txt`
- Model weights and datasets available through Hugging Face or `local_assets/`

## Search Preparation

The checked-in probe fixture, fusion maps, and Rescale configs are ready for
the supported profiles. Regenerate them only when their source inputs change.

```bash
PYTHONPATH=src python -m rfr.preparation.data.build_probe_fixture

PYTHONPATH=src python -m rfr.preparation.fusion.build_map \
  --profile mrpc \
  --out-dir configs/preparation/fusion/maps/mrpc
```

## Stage 1

Start a fresh PPO search:

```bash
bash run_search.sh run rl \
  --preset bert-base-mrpc-stage1-rl \
  --fresh
```

Run the same command without `--fresh` to resume. Replace the preset with any
supported model/task pair.

## Stage 2

```bash
bash run_search.sh run rl \
  --preset bert-base-mrpc-stage2-rl \
  --fresh
```

The production preset uses the fixed Stage-1 prerequisite, layerwise fusion
plus H/M/L precision actions, robust constraints, strict A/B/C validation
banks, and elastic assignment across every healthy visible GPU.

## Comparators

The formal comparators run their own Stage 1, bind that result into Stage 2,
and perform strict top-5 validation.

```bash
bash run_search.sh run bo_rf --fresh
bash run_search.sh run greedy --fresh
bash run_search.sh run coinn_ga --fresh
```

Use `--comparator-stage1-only` to stop after Stage 1. Omit `--fresh` to resume.

## Validation Evaluation

```bash
bash run_search.sh eval \
  --preset mrpc-final-eval-only \
  --resume-from outputs/rl/bert-base/mrpc/stage2/<run>
```

The evaluator consumes the selected materialized action and runs on the full
GLUE validation split. It does not use the GLUE test split or create a GLUE
submission archive.

## Run Control and Outputs

Send `SIGINT` to the recorded PID for a graceful stop. Durable checkpoints and
observation journals are written before exit and are reused on resume.

Generated files are organized under:

```text
outputs/<algorithm>/<model>/<dataset>/
```

RL uses `stage1/<run>` and `stage2/<run>`. Comparators use
`two_stage/<run>`. Each run owns its logs, checkpoints, structured records,
stage artifacts, and validation results. Generated outputs, model weights, and
datasets are ignored by Git.

List available presets with:

```bash
bash run_search.sh --list-presets
```
