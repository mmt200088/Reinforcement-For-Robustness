# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This research repository searches deployment configurations for simulated CKKS + MPC privacy-preserving Transformer inference, primarily BERT on GLUE tasks. The Hugging Face backbone is frozen. Search policies choose approximation and cryptographic configuration decisions; PyTorch plaintext forwards simulate approximation, Gaussian-noise, and truncation effects for reward and validation.

The active RL stages are separate jobs:

- **Stage 1:** one GELU degree decision per Transformer layer. Choices are `{4, 2, 1}`; degree 0/ReLU remains a compatibility value but is masked. Softmax is fixed to degree 6.
- **Stage 2 (`blb_v3`):** a layerwise robust policy bound to a fixed Stage-1 configuration. For BERT-base it makes 12 layer decisions; each decision contains Block4 fusion and one H/M/L truncation-precision preset.

The current canonical Stage-2 representation is not the old node-wise SF space, the retired 47-step blockwise path, or the legacy noise arrays in `noise_rl_module_v2.py`.

## Authority and branch freshness

This repository changes rapidly and is edited by multiple agents. Before treating the checkout as current, run:

```bash
git status -sb
git rev-list --left-right --count HEAD...origin/jk_standard_rl
git log -5 --oneline --decorate
```

Do not analyze an old divergent checkout as the current architecture. If the main worktree is dirty, preserve it on a recovery branch or use a separate worktree; do not pull, reset, or rebase over unknown changes.

When documentation disagrees, use this order:

1. Current implementation and focused tests.
2. Launcher behavior and the active preset.
3. Current checkpoint/run schema and archived runtime evidence.
4. Later dated, non-superseded specs and plans.
5. `CLAUDE.md`, `AGENTS.md`, `README.md`, and older ADRs as navigation/history only.

`docs/adr/` records important historical decisions through ADR-019, but several later layerwise/HML/robust changes are documented in dated specs, plans, tests, presets, and code rather than that ADR index.

There are no repository Cursor rules or Copilot instruction file. `.claude/settings*.json` contains permissions, not architecture or authorization to launch experiments.

## Common commands

### Environment

Preserve an already working CUDA-enabled PyTorch installation and install the remaining dependencies:

```bash
pip install -r requirements.txt
```

CUDA 12.4 / PyTorch 2.5.1 fallback environment:

```bash
bash scripts/setup_cuda124_env.sh
```

Initialize external submodules only when a task needs them:

```bash
git submodule update --init --recursive
```

`Rescale_optimizer/` is checked-in source, not a submodule. The repository is a research workspace and intentionally does not publish a Python wheel.

### Tests

The Makefile/CI core gate uses standard-library `unittest`:

```bash
make test              # torch-free test_blb_*.py discovery
make test-smoke        # sequential smoke path
make test-all          # best-effort full discovery; inspect skips
```

Run one current contract file:

```bash
python3 -m unittest tests.test_blb_layerwise_action -v
```

Run one test method:

```bash
PYTHONPATH="$PWD" python3 -m unittest \
  tests.test_blb_layerwise_action.LayerwiseScheduleTest.test_has_canonical_twelve_step_geometry -v
```

`pyproject.toml` also configures pytest, but minimal CI and the Makefile use `unittest`. Despite the `test-fast` label, current discovery includes modules that import `torch` or `pytest`; a bare system Python without project dependencies fails during collection rather than skipping every such test. Run the gate in the project environment. Torch, CUDA, multiple GPUs, GLUE cache, or real optimizer requirements can also cause skips; report failures and skip counts instead of treating them as executed coverage.

### Lint, formatting, and checks

```bash
make lint
make lint-fix
make format
```

`make preset-check` currently validates the main `presets/*.conf` files but false-fails Paean presets because `tools/validate_preset.py` does not extract Paean's actual flags from the wrapper/config parser correctly. Do not report the all-preset gate as passing until that validator is fixed.

CI-equivalent Ruff checks:

```bash
ruff check --no-cache .
ruff format --no-cache --check .
```

`make docs-check` is intended to validate the HTML guide, but its first inline Python command is currently syntactically invalid (`Makefile:112-115`). Do not report that gate as passing unless the target is fixed and rerun.

There is no supported mypy/pyright gate. The supported image build is:

```bash
make docker
```

The Dockerfile, not the stale Makefile description, is the authority for image CUDA/PyTorch versions. Do not downgrade a working server environment merely to match the fallback image.

### Launching current RL jobs

Stage 1:

```bash
bash llama_7B_LayerImportance.sh run rl \
  --preset bert-base-mrpc-stage1-rl --fresh
```

Stage 2:

```bash
bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl --fresh
```

Remove `--fresh` to use the launcher's resume behavior. Normal training must go through `llama_7B_LayerImportance.sh`; it expands presets, validates incompatible modes, selects persistence paths, writes metadata, and configures elastic GPU execution.

The launcher only permits explicit `stage1-only` or `stage2-only` RL jobs and forces heavy external final evaluation off. Final evaluation is a separate Paean job.

A BLB final-eval preset must use the real optimizer. A safe baseline example is:

```bash
bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed
```

Do not use a preset that leaves `rescale_invoker_kind=heuristic` for BLB action claims. The BLB evaluator rejects heuristic/stub evidence.

Use `make help` for maintained Makefile targets.

## High-level architecture

```text
llama_7B_LayerImportance.sh
  -> preset expansion, mode validation, output paths, elastic GPU settings
  -> rl_tune.py / rl_tune_genetic.py / rl_tune_general.py /
     rl_ga_compare_runner.py
  -> LayerImportanceEvaluator loads the frozen model/data and orchestrates search
  -> Stage-1 PPO or BLB Stage-2 PPO/search baseline
  -> ReversibleLayerHandler installs approximation/noise/truncation behavior
  -> rescale_optimizer_bridge.py calls the real Rescale_optimizer
  -> checkpoints + JSON/JSONL diagnostics + reports
  -> standalone Paean final evaluation
```

### Control plane

- `llama_7B_LayerImportance.sh` is the canonical user entry point. Command-line flags appear after preset values and therefore override them.
- `rl_tune.py` loads/tokenizes GLUE data, freezes the sequence-classification model, constructs `LayerImportanceEvaluator`, and triggers the configured search through a Transformers evaluation callback.
- `layer_importance_evaluator.py` is the main model/evaluation facade. It owns Stage-1 evaluation, Stage-2 fixed-configuration resolution, constraints, metric caching, and BLB runner dispatch.
- `function_handler.py` contains `ReversibleLayerHandler`, which actually installs GELU/Softmax approximation and legacy/BLB noise. Action codecs and bridges do not independently mutate the model.

### Stage 1 current contract

For BERT-base, one episode has 12 steps; for BERT-large, 24. Step `t` selects GELU degree for encoder layer `t`:

```text
action index 0 -> degree 4
action index 1 -> degree 2
action index 2 -> degree 1
action index 3 -> degree 0/ReLU, compatibility-only and permanently masked
```

Softmax is fixed at degree 6 and has no active policy head. Stage-1 evaluation is deterministic plaintext-only on `validation_full`; no BLB noise is installed.

Distinguish the two baselines:

- exact metric baseline: original GELU/Softmax (`-1` sentinels);
- cost reference: GELU 4 / Softmax 6.

The active MRPC Stage-1 preset has no fixed episode cap (`0` means unbounded) and stops after a PPO update when its entropy criterion is satisfied. Check the active preset before changing this contract.

### Stage 2 current layerwise contract

The active MRPC preset uses the canonical `stage2_layerwise_<layers>x2_hml_v3` action space. BERT-base has 12 policy steps—one per layer. Every step has exactly two categorical slots:

```text
slot 0: Block4 fusion       in {0, 1}
slot 1: precision preset    in {high, medium, low}
```

The precision preset materializes all five block truncation values for that layer:

```text
high   = [B1=11, B2=10, B3=10, B4=12, B5=11]
medium = [B1= 9, B2= 8, B3= 8, B4=10, B5= 9]
low    = [B1= 7, B2= 6, B3= 6, B4= 8, B5= 7]
```

Fusion/SF ownership is:

- Block1: baseline SF; K comes from the precision preset.
- Block2: fusion fixed to 1; K comes from the preset.
- Block3: baseline SF; K comes from the preset.
- Block4: fusion 0/1 selected by the policy; K comes from the preset.
- Block5: fusion fixed to 1; K comes from the preset.

Thus BERT-base has 24 compact policy coordinates, not 47 policy steps and not 60 independent K actions. One episode materializes 60 K values. The environment performs 59 immediate block replan requests because layer-0 Block1 has no SF/replan graph, but its truncation K is still installed and executed.

The bottom-level K codec remains the ordered compatibility domain:

```text
K_LEVELS = (8, 9, 11, 13, 10, 12, 6, 7)
```

Do not confuse this codec with the current 3-way policy preset or with `--stage2-k-trials` (the number of Monte Carlo noise trials).

The compatibility full vector remains 877 integers (`73 * 12 + 1`), but it is not the policy action. A complete current action identity includes:

```text
layerwise action matrix
+ materialized full vector
+ boosted field overrides
```

Boosted fusion options can contain SF values that the legacy integer vector cannot represent by itself.

### Fixed Stage-1 binding for Stage 2

Stage-2 does not automatically consume the latest Stage-1 result. If no fixed source is specified, the launcher currently selects `all4`:

```text
GELU    = [4] * num_layers
Softmax = [6] * num_layers
```

Any experiment claiming method-specific Stage-1 -> Stage-2 binding must explicitly pass and persist the source type, source identity, and resolved per-layer degrees. Do not infer the binding from a directory name or a stale preset comment.

### Stage-2 reward and validation

The active layerwise objective combines normalized compute and communication savings. Compute is the learned Block4 fusion axis; communication is the H/M/L preset axis. The MRPC preset currently gives them equal weight.

The active v12 statistical protocol is:

- online terminal probe: fixed stratified 256-example validation subset, 3 trials;
- baseline Bank A: 5 groups x 3 = 15 trials;
- promotion adds Bank B: 15 independent trials;
- final certification adds Bank C: 15 independent trials;
- final A+B+C evidence: 45 trials on `validation_full`.

The six robust channels are loss/accuracy/weighted-F1 means and their three standard deviations. Current formal limits use 0.1% mean tolerance and a 2.0x baseline-std multiplier, with bootstrap probability gates for online, promotion, and final certification. Compute-only and communication-only counterfactuals split the accuracy budget when the active network weighting requires it.

Only the final layer step runs the full model probe. Earlier layer steps materialize and replan the current layer and return zero intermediate reward. Invalid terminal materialization skips model forward.

### Canonical Stage-2 non-PPO baselines

Current HEAD already includes canonical Stage-2-only search backends in `blb_stage2_rl/search_baselines.py`:

- `bo_rf`: constrained SMAC-style random-forest surrogate optimization;
- `greedy`: canonical layerwise greedy search;
- `coinn_ga`: COINN-inspired mutation-based population search.

They use the same layerwise H/M/L action codec, real model-forward path, replan install checks, structured observations, and strict-validation handoff as PPO. They are selected through `rl_tune.py` / `--blb-v3-search-backend`; they do not run through the legacy `rl_tune_genetic.py` path.

Important limitations:

- these backends implement Stage 2 only;
- there is no canonical GELU-only Stage-1 BO/Greedy/GA yet;
- current canonical Greedy is not a verified 2-opt implementation;
- current `coinn_ga` has mutation and elite-neighborhood seeding but no crossover or traditional exact elite carry-over;
- canonical search baselines persist per-evaluation evidence but do not support resume;
- archived smoke evidence proves wiring/runtime behavior, not a scientific full-search result.

### Legacy GA/Greedy paths

`rl_tune_genetic.py`, `genetic_search_module.py`, and `greedy_search_module.py` remain legacy:

- Stage 1 still searches GELU and Softmax jointly;
- Stage 2 searches seven arrays of legacy noise scaling factors;
- they do not use the canonical layerwise BLB action;
- `rl_tune_genetic.py` does not provide the canonical BO-RF backend.

Do not use their presets for a current canonical BLB comparison without an explicit migration decision.

### Rescale optimizer boundary

`Rescale_optimizer` is the source of truth for modulus-chain feasibility, fusion behavior, and bit cost. `rescale_optimizer_bridge.py` converts cfgs to replan input, invokes a real `ReplanSession`, and applies optimizer output back to the executable cfg—including fused-away rescale removal, surviving scale updates, propagation deltas, and effective rotations.

Production training and scientific final evaluation must use the real in-process optimizer. Missing profiles/graph keys or failed imports must fail closed; do not silently replace them with heuristic cost estimates.

### Final evaluation

Paean is the standalone final-evaluation orchestration layer:

- `Paean/run_final_eval.py` builds commands and run directories;
- `Paean/embedded.py` selects the evaluator;
- `Paean/blb_action_eval.py` handles concrete BLB actions with real replan/install semantics;
- `final_evaluation_module.py` handles legacy/config-oriented comparisons.

A canonical BLB search result must carry reloadable layerwise action metadata, the full vector, and boosted overrides so Paean evaluates the same installed configuration used during search. Do not treat an unexecuted `final_eval_repeat_n` default or a skipped Paean stage as final-eval evidence.

## Persistence and artifacts

Current launcher layouts differ by stage:

- Stage 1: `Parting Chapter/stage1/<combo>/`, with record archives.
- Stage 2: `Parting Chapter/persistent/rl/<model>/<dataset>/<constraint_slug>/stage2_noise/progress/`.
- Paean: standalone output under its configured output root.

Stage-2 is not currently archived through the old flat `Parting Chapter/stage2/.../record` path because the launcher disables decoupled layout for formal Stage-2 jobs.

Every new Stage-1/Stage-2 RL run must also mirror structured raw data under:

```text
rl_training_data_points/<stage>/<model>/<dataset>/<run_id>/
```

Preserve enough JSON/JSONL data to recreate paper figures: manifest, source identity, baselines, constraints, actions and decoded values, per-step/per-episode metrics, PPO updates, throughput/parallelism, best-so-far state, and termination summary. PNG/NPZ files are derived inspection artifacts.

For current layerwise Stage-2 reports, show per-layer Block4 fusion, H/M/L preset, and all five materialized K values. Do not publish only compact indices, aggregate fusion count, or a nested action blob.

Canonical Stage-2 search baselines write append-only `observations.jsonl` plus manifest/history/summary and strict-validation evidence. Their fresh-output guard is not resume support.

## Evidence levels

Keep these evidence categories distinct:

- **F0:** real optimizer feasibility/cost; no model-forward quality claim.
- **F1:** online installed-candidate probe during search.
- **F4:** full validation with the real action installed and independent validation banks.
- **Paean final eval:** a separate executed job, not implied by F4 or a preset default.

CPU/torch-free CI, a short smoke run, a single bank, or optimizer-only validation cannot support a final best-configuration claim. Report unexecuted gates and skipped tests explicitly.

## Local/Git/server workflow

- Canonical source changes happen locally and are aggregated into an exact commit before server use.
- The server pulls or receives that exact aggregate source, runs jobs, and returns generated artifacts. It is not a canonical source-editing workspace.
- With multiple agents, do not deploy a branch containing only one agent's partial change. Verify aggregate commit ID and source tree identity.
- Preserve unknown dirty changes. Do not broaden `.gitignore`, reset files, or include unrelated artifacts without inspection.
- Do not commit or push unless the user explicitly requests it.
- Elastic GPU mode defaults to automatic placement. Stage-2 reward probes use persistent spawned processes by default; there is still one learner and one action stream.
- `CUDA_VISIBLE_DEVICES` alone is not proof that the intended multi-GPU path ran. Use launcher logs, worker-device records, telemetry, and hardware evidence.

`SERVER_COMMAND.md` is a legacy/manual command transport, not automatically the current experiment command. Before using its first fenced Bash block, verify that it targets the intended aggregate commit, does not edit/commit source on the server, and matches the current task. Never store server credentials in repository files or persistent project memory.

Graceful Stage-2 stop supports Ctrl+C or `STOP_RL` and exits at a complete PPO update/checkpoint boundary. Treat an episode-cap termination as `max_episodes_reached`, not natural convergence.

## Investigation map

- Launcher, presets, paths, GPU placement: `llama_7B_LayerImportance.sh`
- Model/data setup and shared orchestration: `rl_tune.py`, `layer_importance_evaluator.py`
- Stage-1 policy/environment: `layer_importance_evaluator.py`, `stage1_rl/`
- Current Stage-2 action: `blb_stage2_rl/layerwise_action.py`, `precision_presets.py`, `truncation_levels.py`
- Current Stage-2 environment/training: `layerwise_env.py`, `layerwise_runner.py`, `sequential_runner.py`
- Canonical Stage-2 non-PPO baselines: `search_baselines.py`, `search_baseline_runner.py`
- Compatibility/full-vector decode: `blb_stage2_rl/action_space.py`
- Model installation: `function_handler.py`, `blb_rl_bridge.py`
- Optimizer truth/write-back: `rescale_optimizer_bridge.py`, `Rescale_optimizer/`
- Persistence and diagnostics: `blb_stage2_rl/persistence.py`, `diagnostics.py`, `rl_data_points.py`
- Final evaluation: `Paean/`, `final_evaluation_module.py`
- Historical rationale: `docs/adr/`, dated `docs/superpowers/specs/` and `plans/`
