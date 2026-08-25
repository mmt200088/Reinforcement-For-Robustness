# Reinforcement for Robustness

This repository searches approximation and MPC configurations for BERT
sequence-classification models. It contains the two-stage PPO method, three
search comparators, preparation tools, and full-validation evaluation.

## Supported Profiles

- Models: `bert-base` and `bert-large`
- GLUE tasks: `mrpc`, `rte`, and `sst2`
- Formal comparators: BERT-base MRPC

Preparation, Stage 1, and Stage 2 use the same deterministic, stratified
256-example probe from the GLUE training split. Final evaluation uses the full
GLUE validation split. The GLUE test split is not used.

## Setup

Python 3.9-3.12, a CUDA-enabled PyTorch build, and Linux `flock` are required
for production runs. Install the PyTorch wheel that matches the server CUDA
driver, then install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Models and GLUE data may come from the Hugging Face cache or `local_assets/`.
Generated weights, datasets, and run outputs are not tracked by Git.

## Preparation

The repository includes the probe fixture, fusion maps, and Rescale optimizer
configurations used by the supported profiles. Rebuild these files only when
their source data or graph definitions change:

```bash
PYTHONPATH=src python -m rfr.preparation.data.build_probe_fixture

PYTHONPATH=src python -m rfr.preparation.fusion.build_map \
  --profile mrpc \
  --out-dir configs/preparation/fusion/maps/mrpc
```

## RL Search

List the available model/task presets:

```bash
bash run_search.sh --list-presets
```

Start Stage 1 with a fresh output directory:

```bash
bash run_search.sh run rl \
  --preset bert-base-mrpc-stage1-rl \
  --fresh
```

The Stage 1 presets use batch size 512, learning rate `2e-5`, PPO updates every
120 episodes, seed 42, and an unbounded episode limit that stops when policy
entropy reaches 0.1.

Start Stage 2:

```bash
bash run_search.sh run rl \
  --preset bert-base-mrpc-stage2-rl \
  --fresh
```

The Stage 2 presets run 150,000 episodes with batch size 64, learning rate
`5e-5`, rollout size 120, precision tolerance `0.001`, stability multiplier
`2.0`, and three online trials. Baseline calibration uses 5 groups of 3 trials.
The default `all4` prerequisite fixes GELU degree 4 and Softmax degree 6 in
every layer. Pass `--stage2-fixed-config-source stage1_result` to bind the
latest completed Stage 1 record instead. Reward trials are distributed across
all healthy visible GPUs when elastic GPU mode is enabled.

## Comparator Search

Each comparator runs its own Stage 1, binds that result into Stage 2, and
strictly validates its top five Stage 2 candidates:

```bash
bash run_search.sh run bo_rf --fresh
bash run_search.sh run greedy --fresh
bash run_search.sh run coinn_ga --fresh
```

Formal comparator settings are fixed by the launcher:

| Method | Stage 1 | Stage 2 |
| --- | --- | --- |
| BO-RF | 10,000 evaluations; stop after 1,000 without improvement | 50,000 evaluations; stop after 2,000 without improvement |
| Greedy | Exhaustive best-improvement 1-opt and 2-opt | Exhaustive best-improvement 1-opt and 2-opt |
| COINN-GA | Population 64, 7 elites, 200 update generations | Population 64, 7 elites, 200 update generations |

COINN-GA performs exactly 11,464 inference-reaching evaluations per stage and
does not stop early. Comparator Stage 1 uses batch size 16; Stage 2 uses batch
size 64. Online evaluation uses three trials, the baseline uses 5 x 3 trials,
and strict validation uses three 15-trial banks. Run only Stage 1 with:

```bash
bash run_search.sh run bo_rf --comparator-stage1-only --fresh
```

## Resume and Outputs

Send `SIGINT` to the recorded PID for a graceful stop. The current candidate,
observation journal, checkpoint, and metadata are flushed before exit. Run the
same command without `--fresh` to resume. Using `--fresh` again discards the
existing output for that exact run contract.

Outputs are grouped by algorithm:

```text
outputs/rl/<model>/<dataset>/stage1/<run>/
outputs/rl/<model>/<dataset>/stage2/<run>/
outputs/<bo_rf|greedy|coinn_ga>/bert-base/mrpc/two_stage/<run>/
```

Each run directory contains logs, durable search state, selected
configurations, structured observations, and validation artifacts.

## Final Evaluation

Evaluate a completed search result on the full validation split:

```bash
bash run_search.sh eval \
  --preset mrpc-final-eval-only \
  --source search \
  --resume-from outputs/rl/bert-base/mrpc/stage2/<run> \
  --foreground
```

Use `--dry-run` with any search or evaluation command to inspect the resolved
configuration without starting model inference.
