# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a research codebase for searching approximation and simulated CKKS/MPC noise configurations for privacy-preserving inference, primarily on BERT GLUE classifiers. The Hugging Face backbone is frozen; RL/GA/greedy controllers search configurations, while model forwards provide reward and evaluation measurements.

The active two-stage RL model is:

- **Stage 1:** choose one GELU polynomial degree per transformer layer. Softmax is fixed at degree 6; do not describe Stage 1 as a GELU+Softmax action space.
- **Stage 2 (`blb_v3`):** choose BLB noise/fusion/truncation configurations bound to a fixed Stage-1 result. The active implementation is under `blb_stage2_rl/`; `noise_rl_module_v2.py` is legacy compatibility code.

The PyTorch model simulates the effect of cryptographic operations by installing approximation modules, Gaussian noise, and fixed-point truncation. It does not execute real ciphertext inference.

## Source-of-truth order

When prose disagrees, use this order:

1. Current launcher validation and current code paths.
2. Accepted decisions in `docs/adr/` and the index in `docs/adr/README.md`.
3. Stable constraints in this file and the detailed notes in `AGENTS.md`.
4. `README.md`, `docs/ARCHITECTURE.md`, `docs/GLOBALS.md`, and historical handoff documents.

Some older documents and log messages still describe automatic final evaluation, a flat Stage-2 output layout, a 59-step Stage-2 schedule, or Stage 1 searching Softmax. Verify these claims in the current launcher and action-space code before relying on them. `AGENTS.md` is detailed but also contains historical sections; do not copy a dated statement without checking the implementation.

There are currently no repository Cursor rules or Copilot instruction file. `.claude/settings*.json` contains tool permissions, not architectural guidance or approval to run experiments.

## Common commands

### Environment

If the machine already has a working CUDA-enabled PyTorch installation, preserve it and install the remaining dependencies:

```bash
pip install -r requirements.txt
```

CUDA 12.4 / PyTorch 2.5.1 fallback environment:

```bash
bash scripts/setup_cuda124_env.sh
```

Initialize external submodules only when the task needs them:

```bash
git submodule update --init --recursive
```

`Rescale_optimizer/` is checked-in source, not a submodule. Python support is `>=3.9,<3.13`; `transformers` is pinned in the requirements.

### Tests

The canonical test runner is standard-library `unittest`:

```bash
make test              # alias of test-fast; CI-style test_blb_*.py suite
make test-smoke        # sequential end-to-end smoke test
make test-all          # best-effort full discovery; inspect skipped tests
```

Run one test file:

```bash
python3 tests/test_blb_continuous_reward.py
```

Run one test method:

```bash
PYTHONPATH="$PWD" python3 -m unittest \
  tests.test_blb_continuous_reward.DeterminismTest.test_same_inputs_same_reward -v
```

`pyproject.toml` contains pytest discovery settings, but CI and project documentation use `unittest`; pytest may not be installed in a minimal environment. Torch, CUDA, multiple GPUs, GLUE data, or the real optimizer can cause tests to skip. `make test-all` passing with skips is not evidence that GPU/model-forward behavior was exercised.

### Lint, format, and repository checks

```bash
make lint
make lint-fix
make format
make preset-check
```

`make docs-check` is intended to validate the HTML guide and ADR coverage, but the current Makefile target exits before validation because its first inline Python command is syntactically invalid. Do not report it as a passing gate unless the target itself has been fixed and rerun.

CI-equivalent Ruff checks:

```bash
ruff check --no-cache .
ruff format --no-cache --check .
```

There is no supported mypy/pyright gate and no Python package-release build. The repository intentionally does not ship a wheel. The supported image build is:

```bash
make docker
```

The Dockerfile is the authority for its CUDA/PyTorch versions; the image is a fallback and must not be used to downgrade a working server environment without a concrete reason.

### Launching experiments

Use the launcher for normal work; it applies presets, validates modes, creates the correct output layout, writes metadata, and manages background jobs:

```bash
bash llama_7B_LayerImportance.sh run rl --preset <preset> --fresh
```

Remove `--fresh` to resume. Current RL runs must explicitly select `stage1-only` or `stage2-only` through their preset/flags. Do not launch production training by calling `rl_tune.py` directly. Heavy final evaluation is separate and is orchestrated through Paean rather than being assumed to run automatically after training.

Use `make help` for the maintained Makefile target list.

## High-level architecture

```text
llama_7B_LayerImportance.sh
  -> preset expansion, validation, output paths, process management
  -> rl_tune.py / rl_tune_genetic.py / rl_tune_general.py /
     rl_ga_compare_runner.py
  -> LayerImportanceEvaluator
  -> Stage-1 RL, BLB Stage-2 RL, GA, greedy, or general-policy controller
  -> ReversibleLayerHandler installs approximations/noise into the frozen model
  -> rescale_optimizer_bridge.py calls the real Rescale_optimizer
  -> checkpoints + structured diagnostics + reports
  -> Paean performs standalone or embedded final evaluation
```

### Control plane and model evaluation

- `llama_7B_LayerImportance.sh` is the canonical human entry point. Command-line arguments are applied after preset values and therefore override them.
- `rl_tune.py` loads/tokenizes GLUE data, loads the sequence-classification model, freezes the backbone, constructs `LayerImportanceEvaluator`, and triggers the search through a Transformers evaluation callback.
- `layer_importance_evaluator.py` is the main orchestration and evaluation facade. It owns Stage-1 setup, fixed Stage-1 configuration resolution for Stage 2, metric evaluation/caching, constraints, and routing to the active search backend.
- `rl_tune_genetic.py` selects GA or greedy backends while reusing the same evaluator semantics. `rl_tune_general.py` is the separate multi-task/general-policy path; its Stage-2 representation is not interchangeable with canonical BLB Stage-2 checkpoints.

### Model mutation boundary

`function_handler.py` contains `ReversibleLayerHandler`, the layer that actually replaces GELU/Softmax modules and installs/removes legacy or BLB noise. `blb_rl_bridge.py` decodes and routes BLB block configurations to the handler; it does not independently implement model mutation. Legacy and BLB noise installations are mutually exclusive.

### Stage 1

Stage 1 uses a recurrent GTrXL actor-critic with PPO to choose GELU degree layer by layer. The central learner owns the policy, optimizer, rollout buffer, and update order.

`stage1_rl/parallel_runner.py` parallelizes complete episode collection while preserving central PPO updates. GPU-count-independent episode seeding and ordered replay apply only when an explicit `--stage1-rl-devices` list selects that path; merely setting `CUDA_VISIBLE_DEVICES` does not prove it is active.

Stage 2 must record the exact Stage-1 binding it used: source type (`record`, JSON, manual, or same-process result), source identity, and final per-layer degree vector. A directory name alone is not sufficient evidence of the binding.

### Stage 2 BLB

The active path is the sequential GTrXL+PPO implementation in `blb_stage2_rl/`:

- `runner.py` chooses the active runner and establishes progress paths.
- `action_space.py` defines the compatibility action vector and the active `(layer, block)` schedule.
- `sequential_policy.py`, `sequential_env.py`, and `sequential_runner.py` implement policy, per-step optimizer evaluation, terminal model probes, PPO, checkpointing, and output generation.
- Block 3 and first-input are not active installation decisions. Compatibility slots may remain in a full action vector; do not infer the active horizon from vector width. The 12-layer BERT-base schedule is currently 47 steps, not the stale 59-step value found in old prose/logs. Derive other horizons from `action_space.py`.
- Fusion-count mode compresses each block decision to `(fusion_option, K)`. The option expands to block field values; `K` remains the truncation decision.

`fusion_maps/<profile>/*.json` files are generated caches, not the semantic source of truth. Their generators, real replan behavior, and validation gates define correctness. Before a fusion run, verify that every graph key needed by the selected model/dataset profile has a compatible map. Changing the action grid or skeleton invalidates maps that store action indices.

Stage-2 episode-parallel execution is for fusion-count mode. Workers execute complete episodes with model/handler/environment replicas; one central learner reorders results by global episode and performs PPO updates. The noise trial count `K` is an algorithm parameter, not the number of GPUs. Do not emulate parallelism by launching independent RL jobs with separate learners or run directories.

### Rescale optimizer boundary

`Rescale_optimizer` is the source of truth for modulus-chain feasibility, fusion count, and total-bit cost. The main contract is `rescale_optimizer_bridge.py`:

1. Decode a candidate block cfg into skeleton-ordered `t_new` and delta overrides.
2. Invoke a real `ReplanSession` (in-process is the production path; subprocess is for isolation/debugging).
3. Apply optimizer output back to the cfg, including fused-away rescale removal, surviving scale updates, propagation deltas, and effective rotations.
4. Install the resulting executable cfg for model evaluation.

Training and publishable final evaluation must not use heuristic/stub cost paths. The static skeleton archives under `Rescale_optimizer/configs/<profile>/` and the current `skeleton_stage_map.py` mapping jointly define active topology and ordering; missing profiles/graph keys must fail rather than silently falling back to an all-max estimate.

### Final evaluation

Paean is the final-evaluation orchestration layer:

- `Paean/run_final_eval.py` builds standalone evaluation commands and run directories.
- `Paean/embedded.py` selects the appropriate evaluation protocol.
- `final_evaluation_module.py` handles the legacy/config-oriented comparison protocol.
- `Paean/blb_action_eval.py` is the aligned path for a concrete BLB action: it restores the Stage-1 binding, expands fusion/boosted fields, runs real replan, applies optimizer output, installs noise, and evaluates the model.

Do not treat legacy-compatible `best_noise_config` fields as the complete BLB action when fusion metadata or `best_action_vec` is available.

## Persistence and generated artifacts

The launcher currently uses different canonical layouts:

- Stage 1: `Parting Chapter/stage1/<combo>/`, with record snapshots and completion metadata managed by `config/run_layout.py`.
- Stage 2: `Parting Chapter/persistent/rl/<model>/<dataset>/<constraint_slug>/stage2_noise/progress/`.

For Stage 2:

- Resume training from the live checkpoint.
- Rebuild plots/reports from append-only `diagnostics/episodes.jsonl` and `diagnostics/ppo_updates.jsonl` using `scripts/blb_regen_stage2_outputs.py`.
- A checkpoint can lag diagnostics when interruption occurs between save intervals; do not use it as the sole curve source.

Every new Stage-1 and Stage-2 RL run must preserve structured raw data sufficient to reproduce paper figures: manifest/config, baselines and constraints, per-step/per-episode records, PPO updates, throughput/parallelism, best-so-far state, and final summary. PNG/NPZ files are derived inspection artifacts, not the only evidence.

User-facing Stage-2 reports must decode actions into per-layer/per-block values, including fusion count, truncation `K`, and relevant SF values. Do not publish action indices alone. Final HTML deliverables belong under `reports/html_reports/`; intermediate assets should remain with their run or generator output.

Generated fusion maps, precision-boosted maps, reports, and other canonical artifacts must retain their source identity and validation logs. Run their builder/golden/runtime-install gates before promoting or committing them.

## Validation levels

Keep these evidence levels distinct:

- **F0:** real `Rescale_optimizer` feasibility/cost checks; no model-forward quality claim.
- **F1:** training-time small probe through the installed candidate.
- **F4:** full or near-full final evaluation with the real BLB action installed and repeated noise trials.

Only F4-quality evidence supports a final “best configuration” claim. CPU/torch-free CI, a short smoke run, a single noise trial, or an optimizer-only pass cannot substitute for GPU/model-forward validation. Report skipped tests and unrun validation levels explicitly.

## Repository-specific workflow constraints

- Preserve the existing dirty working tree. Inspect `git status`/diff before editing and do not include or clean unrelated user changes.
- Canonical source edits happen locally. The server is for pulling committed source, running jobs, producing logs/checkpoints/reports, and returning generated artifacts. A server diagnostic patch is not a canonical fix.
- Normal synchronization is local edit -> commit/push -> server pull/run -> generated artifact return. Do not commit or push unless explicitly requested.
- `SERVER_COMMAND.md` is the regular server command bridge; the server agent executes its first Bash fenced block. Keep time-sensitive server addresses, hardware snapshots, PIDs, and active-run state out of this long-lived file.
- Use explicit device-list flags and logs/diagnostics to prove multi-GPU paths. `CUDA_VISIBLE_DEVICES` alone is insufficient.
- Do not change reward semantics, action schemas, persistence layouts, checkpoint compatibility, optimizer contracts, or parallel determinism merely for speed. Measure the real critical path and preserve result semantics.
- Significant cross-file, difficult-to-reverse, or experiment-driven architecture changes require an ADR. Change an Accepted decision through a new/superseding ADR; retain Rejected ADRs to avoid repeating failed approaches.
- The three Git submodules may be absent in a normal checkout and are not required by torch-free CI. Check sparse-checkout and submodule state before assuming a historical artifact or vendored tree is missing.

## Investigation map

Start from the narrowest authoritative boundary:

- Commands, presets, paths, and process behavior: `llama_7B_LayerImportance.sh`
- RL CLI and model/data setup: `rl_tune.py`
- Search/evaluation orchestration: `layer_importance_evaluator.py`
- Stage-1 parallelism/determinism: `stage1_rl/`
- Stage-2 actions and schedule: `blb_stage2_rl/action_space.py`
- Stage-2 training flow: `blb_stage2_rl/runner.py`, `sequential_runner.py`, `sequential_env.py`, `sequential_policy.py`
- Fusion generation/runtime decode: `fusion_enum*.py`, `fusion_count_map.py`, `scripts/blb_build_fusion_count_map.py`
- Model installation: `function_handler.py`, `blb_rl_bridge.py`
- Optimizer truth and write-back: `rescale_optimizer_bridge.py`, `Rescale_optimizer/`
- Stage-2 diagnostics/regeneration: `blb_stage2_rl/diagnostics.py`, `scripts/blb_regen_stage2_outputs.py`
- Final evaluation: `Paean/`, `final_evaluation_module.py`
- Major design rationale: `docs/adr/README.md` and the relevant ADR
