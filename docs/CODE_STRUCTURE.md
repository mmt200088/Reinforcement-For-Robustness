# Code Structure and Reading Guide

This repository searches configurations for an already-trained BERT model.
The learned PPO policies choose approximation and execution parameters; they
are not a second BERT fine-tuning pipeline. The runtime simulates noise and
truncation on tensors. Rescale plans come from the real in-process optimizer,
not from an estimated substitute or a complete encrypted-inference backend.

## Start with the execution path

```text
run_search.sh
  -> cli/run.py                 model, tokenizer and data setup
  -> search/workflow.py         shared state and stage dispatch
     -> rl/stage1/runner.py     Stage-1 baseline/search/JSON handoff
     -> rl/stage2/entrypoint.py Stage-2 public configuration adapter
        -> orchestration.py    setup, resume and backend dispatch
           -> ppo_runner.py    PPO loop and candidate promotion
           OR comparators/common/stage2_runner.py

run_search.sh eval
  -> cli/evaluate.py            separate final-evaluation launcher
  -> cli/run.py / search/workflow.py in final-eval-only mode
  -> evaluation/action_eval.py  selected configuration, full validation
```

The workflow is a Transformers callback, `SearchWorkflow`. It owns the model,
data loaders, model-evaluation helpers and persistent state shared between
stages. Stage-1 policy, environment, PPO update and search-loop implementations
live under `search/rl/stage1/`, not in `search/common/`. Its state is passed to
Stage-1 functions explicitly; it is not copied into a second trainer.

The public shell commands and result filenames remain unchanged. Internal
Python modules have one canonical location; there are no old-path shims.

## Source ownership

All paths below are relative to `src/rfr/`.

### Command-line entrypoints: `cli/`

- `run.py`: the `run` callable validates settings, loads BERT/GLUE, binds the
  fixed data protocol and constructs the workflow. It does not fine-tune BERT
  weights.
- `evaluate.py`: launch one selected-config final evaluation, including
  foreground/background and dry-run handling.
- `evaluation_config.py`: final-evaluation argument and output-path contract.
- `validate_preset.py`: check preset flags, duplicates and value types.

### Stage 1: `search/rl/stage1/`

- `config.py`: action constants, PPO defaults and mutable rollout settings.
  Read mutable settings through the module so launcher updates are visible.
- `policy.py`: GTrXL actor-critic and return normalization.
- `env.py`: sequential GELU-choice environment; production search fixes the
  Softmax approximation parameter to 6.
- `rollout.py`: recurrent transition storage, masks and tensor packing.
- `training.py`: GAE, PPO updates and worker rollout collection. These
  functions operate on the workflow's state without taking over Stage 2.
- `runner.py`: baseline measurement, PPO/comparator dispatch, episode loop,
  normal-completion checks and Stage-1 JSON handoff.
- `parallel_runner.py`: worker replicas and deterministic rollout collection.
- `eval_cache.py`: shared deterministic plaintext-evaluation cache.
- `checkpoint.py`: atomic checkpoints and random/log-state recovery.
- `seed_utils.py`: episode seeds and worker-independent episode assignment.

### Stage 2: `search/rl/stage2/`

- `entrypoint.py`: `BLBStage2RLRunner`, public train settings and probe-batch
  setup. The name denotes an entrypoint, not the PPO update implementation.
- `orchestration.py`: construct baselines and environments, validate resume
  evidence, dispatch PPO/comparators and finalize strict selection.
- `ppo_runner.py`: layerwise PPO episodes, validation banks, candidate
  promotion, checkpoint evidence and strict ranking. The horizon is the model
  layer count: 12 for BERT-base and 24 for BERT-large.
- `policy_env.py`: policy-facing environment, with one compact two-choice
  action per BERT layer.
- `execution_env.py`: execute a fully materialized action, run repeated noisy
  probes and compute terminal metrics/reward. It is not another policy space.
- `policy.py`: shared small GTrXL, actor heads, critic, rollout buffer and PPO
  update mathematics.
- `policy_network.py`: fixed architecture/checkpoint identity contract.
- `reward.py`: reward construction and diagnostic decomposition.
- `seed_utils.py`: disjoint policy, probe, promotion and update seed domains.

### Shared search semantics: `search/common/`

- `layerwise_action.py`: compact layerwise fusion/precision action codec.
- `full_action.py`: full runtime slot vector and block-config decoding. This
  is an execution interchange format, not an alternative learned policy.
- `action_io.py`: labeled-slot serialization and reconstruction.
- `precision_presets.py`: H/M/L precision presets and resource trade-offs.
- `truncation_levels.py`: supported simulation K values and checkpoint domain.
- `skeleton_stage_map.py`: optimizer-node to runtime-field bindings.
- `noise_tables.py`: AST-read access to the constants in `model_handler.py`,
  without importing Torch or reading a local CSV.
- `precision_boost.py`: precision refinement of prepared fusion options.
- `statistical_constraints.py`: deterministic bootstrap constraint evidence.
- `eval_metrics.py`: shared loss/accuracy/F1 aggregation rules.
- `candidate_store.py`: candidate identity, trial evidence and promotion store.
- `best_config.py`: validated Stage-1 and final-search JSON contracts.
- `diagnostics.py`: durable episode and PPO-update diagnostics.
- `data_points.py`: structured raw training records and protocol metadata.
- `persistence.py`: Stage-2 status boards, summaries and curve persistence.
- `run_artifacts.py`: shared output layout and stage-status metadata.
- `training_curves.py`: shared training-curve rendering, not final evaluation.

### Comparator algorithms: `search/comparators/`

Each of `bo_rf/`, `greedy/` and `coinn_ga/` has `stage1.py` and `stage2.py`.
They propose candidates using random-forest Bayesian optimization, complete
greedy neighborhoods or genetic operators respectively.

`common/stage1_core.py` and `common/stage2_core.py` define search spaces,
constraints, ordering and dispatch. The corresponding `stage1_runner.py` and
`stage2_runner.py` adapt those algorithms to real evaluation, persistence and
strict selection. Their production profile remains BERT-base MRPC.

### Execution: `search/runtime/`

- `model_handler.py`: reversible BERT replacements, function approximations,
  block noise configurations and tensor-level noise/truncation simulation.
- `model_installation.py`: install or clear already-materialized block configs.
  It does **not** call the Rescale planner; that belongs to `optimizer_bridge`.
- `inference_eval.py`: forward passes and metrics after configuration install.
- `probe_runner.py`: deterministic parallel repeated-noise trials.
- `device_utils.py`: logical device parsing.
- `elastic_gpu.py`: typed infrastructure failures and restart requests.
- `supervisor.py`: CUDA health checks, device quarantine and process recovery.
- `control.py`: graceful stopping, signals and progress formatting.
- `cuda/block3_fused_cuda.py`: exact-FP32 Block-3 kernel acceleration.
- `cuda/block5_fused_cuda.py`: exact-FP32 Block-5 kernel acceleration.
- `cuda/truncation_fused_cuda.py`: exact-FP32 binary truncation kernel.

### Preparation: `preparation/`

Data:

- `data/protocol.py`: supported profiles, stratified train-probe identity and
  full-validation separation.
- `data/build_probe_fixture.py`: generate the no-text probe identity fixture.
- `data/mrpc_contract.py`: fixed BERT-base MRPC comparator runtime contract.

Fusion maps:

- `fusion/build_map.py`: offline map-building entrypoint.
- `fusion/enumeration.py`: reference enumeration, grouping and ranking.
- `fusion/enumeration_fast.py`: equivalent direct-replan enumeration fast path.
- `fusion/count_map.py`: runtime map objects and option lookup.
- `fusion/fixed_action.py`: reconstruct exact mapped options from saved actions.

Rescale integration:

- `rescale/baseline_bootstrap.py`: bind Stage-1 vectors to static skeletons.
- `rescale/block_materialization.py`: decode/replan/materialize one block.
- `rescale/optimizer_bridge.py`: translate runtime configs to optimizer inputs
  and translate the real planner outputs back to runtime configs.
- `rescale/optimizer_cost.py`: full-action materialization, fingerprints and
  optimizer-derived costs.

Rescale algorithm (`rescale/optimizer/`):

- `graph.py`: node, edge, scale, noise and cost data structures.
- `config_loader.py`: graph/config construction from JSON.
- `feasibility.py`: feasible stage and tail edges.
- `reachability.py`: forward/backward feasible Rescale-step counts.
- `backward_level_dp.py`: minimum-cost skeleton dynamic programming.
- `modulus_chain.py`: modulus construction, repair, fallback and validation.
- `optimizer.py`: orchestrate those optimization stages.
- `replan.py`: replan an existing skeleton with supplied scale choices.
- `replan_interface.py`: reusable in-process sessions and graph selection.

### Final evaluation: `evaluation/`

- `action_eval.py`: rebuild and measure one selected configuration, then write
  the final JSON/Markdown reports.
- `protocol.py`: require the matching full-validation data protocol.

This package contains no search-time training-curve utility or new searcher.

### Generic utilities: `common/`

`cli_parse_utils.py`, `json_utils.py`, `jsonl_utils.py`,
`report_format_utils.py`, `runtime_error_reporter.py` and `stats_utils.py`
provide parsing, serialization/recovery, formatting, error summaries and
simple statistics. Search-specific statistics belong to `search/common/`.

Most `__init__.py` files are package markers. Rescale exports its config root
and optimizer API; the Stage-1 and Stage-2 initializers use lazy exports.

## Configuration and artifact boundaries

- `configs/presets/`: 2 models x 3 datasets x 2 search stages = 12 launch presets.
- `configs/preparation/fusion/maps/<profile>/`: five prepared graph maps per
  profile (Block 2, Block 4 and three Block-5 GELU variants).
- `configs/preparation/fusion/max_sfs/`: default and MRPC scaling-factor tables.
- `configs/preparation/rescale/<profile>/`: 11 graph definitions plus a static
  skeleton archive per profile.
- `fixtures/reproducibility/`: fixed train-probe identities, not test results.
- `local_assets/`: untracked downloaded model/data assets.
- `outputs/`: untracked run logs, checkpoints, candidate evidence and reports.
- `agent_handoffs/`: historical task/integration evidence, not runtime input.
- `scripts/`, `.githooks/`, `.github/workflows/`: repository/verification tools.

Preserved configuration filenames and identifiers are scientific inputs, not
an invitation to activate extra search dimensions. Older comments inside a
pinned JSON or historical handoff are not current Python import paths.

## Terms that must not be conflated

1. **BERT layer vs Block 1-5:** blocks are computation fragments, not the first
   five Transformer layers. Layerwise decisions span all 12 or 24 layers.
2. **Stage-1 output vs search dimensions:** JSON carries both GELU and Softmax
   vectors; the current search chooses GELU with fixed Softmax parameter 6.
3. **Compact vs full action:** the policy chooses `[Block4 fusion, H/M/L]`;
   `full_action` describes the many installation slots derived from it.
4. **Rescale fusion vs CUDA fusion:** the former changes the selected execution
   plan; the latter accelerates equivalent FP32 tensor arithmetic.
5. **Ciphertext K vs simulation K:** paper metadata and installed simulation
   values are separate fields in the precision presets.
6. **Search validation vs GLUE validation:** A/B/C gates and strict top-5
   selection still use the fixed 256-example training probe. Only the separate
   final-evaluation command uses complete GLUE validation.
7. **`stage1_best_config.json` vs `search_best_config.json`:** the first passes
   approximation choices to Stage 2; the second identifies a strictly selected
   configuration eligible for final evaluation. Neither is a measurement report.

## Reading order

1. README, launcher, `cli/run.py`, `search/workflow.py`.
2. Data protocol and best-config JSON contract.
3. Model replacements in `runtime/model_handler.py`.
4. Stage-1 config, environment, policy, rollout, training and runner.
5. Layerwise/full actions, precision presets, fusion maps and Rescale bridge.
6. Stage-2 policy environment, execution environment, constraints and reward.
7. Stage-2 policy/PPO loop, orchestration, resume and strict selection.
8. Comparators, final evaluation and hardware supervision.

## Checking the structure

Run `python scripts/production_surface_guard.py --json` after staging changes.
In addition to retired-path checks, it checks local Python module/symbol imports,
sole ownership of Stage-1 core definitions and dynamic access to mutable PPO
settings. It does not substitute for CUDA or real-model smoke tests.

This layout change preserves launch flags, scientific configs/fixtures, policy
identities, checkpoint payloads and result schemas. Module moves are not
supported as an in-place update to an already-running Python process; finish or
checkpoint that process before starting another checkout.
