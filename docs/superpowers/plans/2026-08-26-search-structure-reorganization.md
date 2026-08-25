# Search Repository Structure Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the production repository by search workflow, give RL and each comparator a clear code and output home, and remove `Model_analysis/` and `tests/` without changing scientific behavior.

**Architecture:** Move production Python into a single `src/rfr` package with preparation, search, runtime, evaluation, and CLI modules. Keep shared scientific logic centralized while placing BO-RF, Greedy, and COINN-GA operators in separate algorithm packages. Change generated paths only after the module migration passes exact characterization gates.

**Tech Stack:** Python 3.10+, Bash, PyTorch/CUDA, Hugging Face Transformers/Datasets, NumPy, scikit-learn, Git worktrees, repository sync guard.

---

## File Map

### Shared package

- Create: `src/rfr/__init__.py`
- Move: `json_utils.py` -> `src/rfr/common/json_utils.py`
- Move: `jsonl_utils.py` -> `src/rfr/common/jsonl_utils.py`
- Move: `csv_field_utils.py` -> `src/rfr/common/csv_field_utils.py`
- Move: `stats_utils.py` -> `src/rfr/common/stats_utils.py`
- Move: `numeric_parse_utils.py` -> `src/rfr/common/numeric_parse_utils.py`
- Move: `cli_parse_utils.py` -> `src/rfr/common/cli_parse_utils.py`
- Move: `report_format_utils.py` -> `src/rfr/common/report_format_utils.py`
- Move: `runtime_error_reporter.py` -> `src/rfr/common/runtime_error_reporter.py`
- Move: `config/constants.py` -> `src/rfr/common/config/constants.py`
- Move: `config/paths.py` -> `src/rfr/common/config/paths.py`
- Move: `config/run_layout.py` -> `src/rfr/common/config/run_layout.py`

### Preparation package

- Move: `glue_data_protocol.py` -> `src/rfr/preparation/data/protocol.py`
- Move: `mrpc_reproducibility.py` -> `src/rfr/preparation/data/mrpc_reproducibility.py`
- Move: `scripts/build_glue_train_probe_fixture.py` -> `src/rfr/preparation/data/build_probe_fixture.py`
- Move: `blb_stage2_rl/fusion_count_map.py` -> `src/rfr/preparation/fusion/count_map.py`
- Move: `blb_stage2_rl/fusion_enum.py` -> `src/rfr/preparation/fusion/enumeration.py`
- Move: `blb_stage2_rl/fusion_enum_fast.py` -> `src/rfr/preparation/fusion/enumeration_fast.py`
- Move: `blb_stage2_rl/fusion_fixed_action.py` -> `src/rfr/preparation/fusion/fixed_action.py`
- Move: `blb_stage2_rl/fusion_cost.py` -> `src/rfr/preparation/fusion/cost.py`
- Move: `scripts/blb_build_fusion_count_map.py` -> `src/rfr/preparation/fusion/build_map.py`
- Move: `scripts/audit_fusion_count_maps.py` -> `src/rfr/preparation/fusion/audit_maps.py`
- Move: `scripts/blb_export_action_registry.py` -> `src/rfr/preparation/fusion/export_action_registry.py`
- Move: `Rescale_optimizer/rescale_optimizer/*.py` -> `src/rfr/preparation/rescale/optimizer/`
- Move: `Rescale_optimizer/scripts/batch_run_configs.py` -> `src/rfr/preparation/rescale/batch_run_configs.py`
- Move: `Rescale_optimizer/scripts/gen_replan_actions.py` -> `src/rfr/preparation/rescale/gen_replan_actions.py`
- Move: `rescale_optimizer_bridge.py` -> `src/rfr/preparation/rescale/bridge.py`
- Move: `blb_stage2_rl/baseline_bootstrap.py` -> `src/rfr/preparation/rescale/baseline_bootstrap.py`
- Move: `blb_stage2_rl/block_materialization.py` -> `src/rfr/preparation/rescale/block_materialization.py`
- Move: `blb_stage2_rl/optimizer_cost.py` -> `src/rfr/preparation/rescale/optimizer_cost.py`
- Move: `blb_stage2_rl/optimizer_output_introspect.py` -> `src/rfr/preparation/rescale/output_introspection.py`

### Search shared and runtime packages

- Move action, constraint, candidate, persistence, and metric modules from
  `blb_stage2_rl/` to `src/rfr/search/common/`:
  `action_io.py`, `action_space.py`, `candidate_store.py`, `diagnostics.py`,
  `eval_metrics.py`, `feasibility.py`, `layerwise_action.py`,
  `logging_helpers.py`, `persistence.py`, `precision_boost.py`,
  `precision_presets.py`, `skeleton_stage_map.py`,
  `statistical_constraints.py`, `strict.py`, `truncation_levels.py`.
- Move: `noise_tables.py` -> `src/rfr/search/common/noise_tables.py`
- Move: `noise_targets_registry.py` -> `src/rfr/search/common/noise_targets_registry.py`
- Move: `rl_data_points.py` -> `src/rfr/search/common/data_points.py`
- Move: `rl_local_optimum.py` -> `src/rfr/search/common/local_optimum.py`
- Move: `device_utils.py` -> `src/rfr/search/runtime/device_utils.py`
- Move: `elastic_gpu.py` -> `src/rfr/search/runtime/elastic_gpu.py`
- Move: `function_handler.py` -> `src/rfr/search/runtime/model_handler.py`
- Move: `blb_rl_bridge.py` -> `src/rfr/search/runtime/blb_bridge.py`
- Move: `blb_stage2_rl/inference_eval.py` -> `src/rfr/search/runtime/inference_eval.py`
- Move: `blb_stage2_rl/probe_runner.py` -> `src/rfr/search/runtime/probe_runner.py`
- Move: `blb_stage2_rl/runtime_control.py` -> `src/rfr/search/runtime/control.py`
- Move: `blb_stage2_rl/schedule_geometry.py` -> `src/rfr/search/runtime/schedule_geometry.py`
- Move CUDA files from `blb_stage2_rl/` to `src/rfr/search/runtime/cuda/`:
  `block3_fused_cuda.py`, `block5_fused_cuda.py`,
  `truncation_fused_cuda.py`.
- Move: `scripts/elastic_gpu_supervisor.py` -> `src/rfr/search/runtime/supervisor.py`

### RL packages

- Move `stage1_rl/checkpoint.py`, `eval_cache.py`, `parallel_runner.py`, and
  `seed_utils.py` to `src/rfr/search/rl/stage1/`.
- Move Stage-2 RL files to `src/rfr/search/rl/stage2/`:
  `env.py`, `layerwise_env.py`, `layerwise_runner.py`, `policy_network.py`,
  `reward.py`, `seed_utils.py`, `sequential_policy.py`,
  `sequential_runner.py`, and `training.py`.

### Comparator packages

- Split the shared Stage-1 types, space, ranking, cache, initial design, and
  dispatcher from `stage1_rl/search_baselines.py` into
  `src/rfr/search/comparators/common/stage1_core.py`.
- Move `stage1_rl/search_runner.py` to
  `src/rfr/search/comparators/common/stage1_runner.py`.
- Split the shared Stage-2 types, space, ranking, cache, initial design, and
  dispatcher from `blb_stage2_rl/search_baselines.py` into
  `src/rfr/search/comparators/common/stage2_core.py`.
- Move `blb_stage2_rl/search_baseline_runner.py` to
  `src/rfr/search/comparators/common/stage2_runner.py`.
- Extract BO-RF operator bodies into:
  `src/rfr/search/comparators/bo_rf/stage1.py` and `stage2.py`.
- Extract Greedy operator bodies into:
  `src/rfr/search/comparators/greedy/stage1.py` and `stage2.py`.
- Extract COINN-GA operator bodies into:
  `src/rfr/search/comparators/coinn_ga/stage1.py` and `stage2.py`.

### Evaluation and CLI packages

- Move `Paean/action_grid.py`, `blb_action_eval.py`, `embedded.py`, and
  `final_eval_layout.py` to `src/rfr/evaluation/`.
- Move: `final_evaluation_module.py` -> `src/rfr/evaluation/final_evaluation.py`
- Move: `training_curve_plot.py` -> `src/rfr/evaluation/training_curve_plot.py`
- Move: `Paean/config.py` -> `src/rfr/cli/evaluation_config.py`
- Move: `Paean/run_final_eval.py` -> `src/rfr/cli/evaluate.py`
- Move: `rl_tune.py` -> `src/rfr/cli/run.py`
- Move: `tools/validate_preset.py` -> `src/rfr/cli/validate_preset.py`
- Move: `scripts/blb_make_run_manifest.py` -> `src/rfr/cli/make_run_manifest.py`
- Move: `layer_importance_evaluator.py` ->
  `src/rfr/search/common/evaluator.py`
- Rename: `llama_7B_LayerImportance.sh` -> `run_search.sh`

### Configuration and generated paths

- Move `presets/*.conf` -> `configs/presets/`.
- Move `Paean/presets/*.conf` -> `configs/evaluation/presets/`.
- Move `Paean/action_configs/*.json` -> `configs/evaluation/actions/`.
- Move `Rescale_optimizer/configs/**` -> `configs/preparation/rescale/**`.
- Move `blb_stage2_rl/fusion_maps/**` -> `configs/preparation/fusion/maps/**`.
- Move `blb_stage2_rl/max_sfs/**` -> `configs/preparation/fusion/max_sfs/**`.
- Move `glue_configs.json` -> `configs/models/glue.json`.
- Move `glue_final_configs_best_ppo.json` -> `configs/reference/rl.json`.
- Move `glue_final_configs_best_genetic.json` ->
  `configs/reference/coinn_ga.json`.
- Create ignored generated roots `outputs/rl/`, `outputs/bo_rf/`,
  `outputs/greedy/`, and `outputs/coinn_ga/` with one tracked
  `outputs/README.md` describing the layout.

## Task 1: Capture the Current Characterization Baseline

**Files:**
- Create on result branch: `structure_reorganization/baseline/`
- Read: all current production files

- [ ] **Step 1: Record exact source identity**

Run on the server checkout of `c799a2780a84ffc7c63e08bf834ec8cc4360ec42`:

```bash
git rev-parse HEAD^{commit} HEAD^{tree}
git status --porcelain=v1
```

Expected: commit `c799a2780a84ffc7c63e08bf834ec8cc4360ec42`,
tree `f6e6aafea6068bfc432a02dc27b848425d1b2240`, and empty status.

- [ ] **Step 2: Capture launcher behavior**

Run every file under `presets/*.conf` with `--dry-run`, plus comparator
Stage-1-only and complete dry-runs. Normalize only absolute repository paths
and output-root strings. Save raw and normalized output under the result
branch.

- [ ] **Step 3: Capture scientific snapshots**

Use the existing tests and a small standalone snapshot driver to serialize:

```python
snapshot = {
    "precision_presets": precision_payload,
    "action_decoding": decoded_actions,
    "candidate_hashes": candidate_hashes,
    "fusion_maps": fusion_map_hashes,
    "rescale_materialization": materialized_cfg_hashes,
    "stage1_rank_order": stage1_rank_order,
    "stage2_rank_order": stage2_rank_order,
    "formal_contracts": formal_contracts,
}
```

Write canonical sorted JSON and its SHA-256. No random trial or model inference
is needed for this snapshot.

- [ ] **Step 4: Run baseline server suite**

Run the complete current `pytest` and `unittest` suites on the server. Record
pass, skip, and failure counts before editing source.

- [ ] **Step 5: Commit result evidence**

Commit and push baseline evidence only to
`codex/result-search-structure-reorganization-20260826`; do not merge that
branch into source.

## Task 2: Create the `src/rfr` Package and Move Shared Utilities

**Files:** shared package paths listed in the File Map; temporary test import
updates under `tests/`.

- [ ] **Step 1: Add package initializers**

Create `__init__.py` files under every package directory. Each initializer is
empty except where the old package exposed an intentional public symbol.

- [ ] **Step 2: Move shared utilities with Git renames**

Use `git mv` for every shared utility and config module in the File Map.

- [ ] **Step 3: Update imports mechanically**

Replace imports with explicit package paths, for example:

```python
from rfr.common.json_utils import read_json_file, write_json_file
from rfr.common.config import run_layout
from rfr.common.config.constants import GELU_COST, SOFTMAX_COST
```

Do not add re-export wrappers at the old paths.

- [ ] **Step 4: Update temporary tests**

Change only import paths and source-inspection paths in `tests/`; do not change
asserted values.

- [ ] **Step 5: Verify and commit**

Run compilation and the utility/config-focused server tests. Commit as:

```text
refactor: move shared code into rfr package
```

Push immediately.

## Task 3: Move Search Preparation Code and Configuration

**Files:** all preparation package and preparation config paths in the File
Map; tests for data protocol, fusion maps, and Rescale.

- [ ] **Step 1: Move data-protocol modules and fixture builder**

Update callers to import:

```python
from rfr.preparation.data.protocol import TRAIN_PROBE_SPLIT
```

The fixture JSON bytes remain unchanged.

- [ ] **Step 2: Move fusion code and map JSON**

Define one package constant in `rfr.preparation.fusion.count_map`:

```python
FUSION_CONFIG_ROOT = REPO_ROOT / "configs" / "preparation" / "fusion"
```

Use it for maps and max-SF config discovery. Do not alter map contents.

- [ ] **Step 3: Move Rescale implementation and config JSON**

Define one package constant in `rfr.preparation.rescale`:

```python
RESCALE_CONFIG_ROOT = REPO_ROOT / "configs" / "preparation" / "rescale"
```

Update in-process optimizer imports and remove the old `sys.path` insertion.

- [ ] **Step 4: Compare configuration hashes**

Hash every moved JSON file before and after. Expected: identical SHA-256 for
every file.

- [ ] **Step 5: Verify and commit**

Run data-protocol, fusion, action-materialization, optimizer, and replan tests
on the server. Commit as:

```text
refactor: organize search preparation modules
```

Push immediately.

## Task 4: Move Search Common and Runtime Code

**Files:** search common and runtime paths in the File Map; their focused
tests.

- [ ] **Step 1: Move shared scientific modules**

Update relative imports to `rfr.search.common.*` and preparation imports to
their new package paths. Function bodies and default values remain unchanged.

- [ ] **Step 2: Move runtime and CUDA modules**

Update lazy CUDA imports in `model_handler.py` to:

```python
from rfr.search.runtime.cuda.truncation_fused_cuda import binary_truncate
from rfr.search.runtime.cuda.block3_fused_cuda import install_block3_fused
from rfr.search.runtime.cuda.block5_fused_cuda import install_block5_fused
```

Use the actual exported symbol names from the moved files; do not rename
runtime symbols in this task.

- [ ] **Step 3: Preserve elastic GPU semantics**

Keep health-probe interval, device ordering, fallback behavior, process
backend, trial assignment, and seed binding byte-for-byte equivalent.

- [ ] **Step 4: Verify and commit**

Run model-hook, truncation, fused CUDA, probe-runner, inference, device, and
elastic GPU tests on the server. Commit as:

```text
refactor: organize shared search runtime
```

Push immediately.

## Task 5: Move Stage-1 and Stage-2 RL Code

**Files:** RL package paths in the File Map; RL-focused tests.

- [ ] **Step 1: Move Stage-1 PPO modules**

Update `layer_importance_evaluator` imports to
`rfr.search.rl.stage1.*`. Preserve checkpoint filename, entropy-stop behavior,
parallel episode order, and seed derivation.

- [ ] **Step 2: Move Stage-2 PPO modules**

Move the production small GTrXL policy and layerwise runner without changing
class definitions, tensor operations, or defaults. Rebuild the public exports
in `rfr.search.rl.stage2.__init__` for the exact symbols consumed by the
orchestrator.

- [ ] **Step 3: Verify policy identity**

Compare policy architecture ID, parameter names, state-dict keys, action
dimensions, and deterministic fixed-input forward output before and after.

- [ ] **Step 4: Verify and commit**

Run all Stage-1 and Stage-2 RL tests, including deterministic probe, PPO,
checkpoint, persistence, and production-config gates. Commit as:

```text
refactor: separate stage1 and stage2 rl modules
```

Push immediately.

## Task 6: Separate BO-RF, Greedy, and COINN-GA Implementations

**Files:** comparator package paths in the File Map; comparator tests.

- [ ] **Step 1: Extract common Stage-1 and Stage-2 types**

Move spaces, evaluations, configs, ranking keys, caches, structured initial
design, and dispatch into `comparators/common`. Preserve declaration order and
constant values.

- [ ] **Step 2: Extract Greedy operators**

Move the existing `_scan_neighborhood`/`_run_greedy` Stage-1 bodies and
`_evaluate_full_neighborhood`/`_run_greedy` Stage-2 bodies into Greedy modules.
The public adapter is:

```python
def run(evaluator, cache, config, **state):
    return _run_greedy(evaluator, cache, config, **state)
```

- [ ] **Step 3: Extract BO-RF operators**

Move surrogate, tree-prediction, candidate-pool, acquisition, and `_run_bo_rf`
bodies into BO-RF modules. Do not alter floating-point expressions or stable
tie-breaking.

- [ ] **Step 4: Extract COINN-GA operators**

Move parent weighting, tournament, diversity, crossover, mutation, child
generation, and `_run_coinn_ga` bodies into COINN-GA modules. Do not alter RNG
draw order.

- [ ] **Step 5: Dispatch by exact backend name**

The common dispatcher imports one module only after normalized backend
selection:

```python
if backend == "bo_rf":
    from rfr.search.comparators.bo_rf import stage1 as implementation
elif backend == "greedy":
    from rfr.search.comparators.greedy import stage1 as implementation
elif backend == "coinn_ga":
    from rfr.search.comparators.coinn_ga import stage1 as implementation
else:
    raise ValueError(f"unsupported comparator backend: {backend}")
```

Use the corresponding Stage-2 module in the Stage-2 dispatcher.

- [ ] **Step 6: Prove ordered equivalence**

For each algorithm and stage, compare candidate action sequence, evaluation
count, incumbent history, termination proof, and final ranking on deterministic
fake evaluators. Expected: exact equality.

- [ ] **Step 7: Verify and commit**

Run every Stage-1 and Stage-2 comparator test on the server. Commit as:

```text
refactor: separate comparator algorithm packages
```

Push immediately.

## Task 7: Move Validation Evaluation and the Main Orchestrator

**Files:** evaluation and CLI Python paths in the File Map; evaluation tests.

- [ ] **Step 1: Move Paean and final evaluation modules**

Update imports to the new search, preparation, runtime, and common packages.
Keep full GLUE validation as the only final-evaluation split.

- [ ] **Step 2: Move `layer_importance_evaluator.py`**

Place it at `src/rfr/search/common/evaluator.py`; update all lazy imports to
new package paths without changing control flow.

- [ ] **Step 3: Move `rl_tune.py` to the CLI package**

Preserve the existing Fire parsing wrapper and callable:

```python
if __name__ == "__main__":
    run_fire_entrypoint(
        fire,
        train,
        program_name="run_search",
    )
```

Only the displayed program name changes with the launcher. Do not change the
Fire argument interface.

- [ ] **Step 4: Verify and commit**

Run final-evaluation, Paean, dataset isolation, orchestrator, and configuration
tests on the server. Commit as:

```text
refactor: organize validation evaluation and cli
```

Push immediately.

## Task 8: Replace the Launcher and Centralize Configuration

**Files:** `run_search.sh`, `configs/**`, `src/rfr/cli/**`, `Makefile`,
`Dockerfile`, `pyproject.toml`, temporary launcher tests.

- [ ] **Step 1: Rename the launcher**

Rename to `run_search.sh`, set:

```bash
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
```

Call `python3 -m rfr.cli.run` and `python3 -m rfr.cli.evaluate`. Remove the old
launcher path and `Paean/run_final_eval.sh` entirely.

- [ ] **Step 2: Move presets and static configs**

Update preset discovery to `configs/presets/*.conf` and evaluation preset
discovery to `configs/evaluation/presets/*.conf`. Keep each preset file's
argument lines byte-identical.

- [ ] **Step 3: Configure setuptools src layout**

Set:

```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["rfr*"]
```

Update Ruff first-party package names to `rfr`.

- [ ] **Step 4: Compare normalized launcher output**

Run all old/new dry-run pairs. Ignore only executable module path, preset file
path, and generated output root. Every scientific option and value must match.

- [ ] **Step 5: Verify and commit**

Run shell syntax, preset validation, launcher tests, and all dry-runs on the
server. Commit as:

```text
refactor: replace launcher and centralize configs
```

Push immediately.

## Task 9: Unify Algorithm Output Roots

**Files:** `src/rfr/common/config/paths.py`, `run_layout.py`,
`src/rfr/cli/run.py`, `run_search.sh`, `src/rfr/evaluation/**`, `.gitignore`,
`outputs/README.md`.

- [ ] **Step 1: Define output root constants**

Use:

```python
OUTPUT_ROOT = REPO_ROOT / "outputs"
ALGORITHM_NAMES = ("rl", "bo_rf", "greedy", "coinn_ga")
```

- [ ] **Step 2: Build deterministic run paths**

Generate:

```text
outputs/<algorithm>/<model>/<dataset>/<stage-and-constraint-slug>/
```

Preserve the current constraint slug values, optional run-tag normalization,
same-command resume, completed marker, lock, latest-run pointer, and graceful
stop behavior.

- [ ] **Step 3: Separate run artifacts by stage**

Route Stage-1 journal/checkpoint/result under `stage1/`, Stage-2 PPO/candidate
state under `stage2/`, validation output under `evaluation/`, durable model and
optimizer states under `checkpoints/`, and text streams under `logs/`.

- [ ] **Step 4: Assert algorithm-root confinement**

Use dry-run and smoke fixtures to assert every generated path resolves below
`outputs/<algorithm>/`; reject `..` traversal and cross-algorithm resume roots.

- [ ] **Step 5: Verify resume equivalence**

Run fresh then resume smokes for RL Stage 1, RL Stage 2, BO-RF, Greedy, and
COINN-GA. Compare scientific state before and after path normalization.

- [ ] **Step 6: Verify and commit**

Commit as:

```text
refactor: isolate generated outputs by algorithm
```

Push immediately.

## Task 10: Update Documentation and Production Surface

**Files:** `README.md`, `AGENTS.md`, `CLAUDE.md`, `Dockerfile`, `Makefile`,
`scripts/production_surface_guard.py`, `.gitignore`, `.github/workflows/ci.yml`,
`local_assets/README.md`, `examples/representative_rl_log/README.md`.

- [ ] **Step 1: Rewrite the concise English README**

Document preparation, Stage-1 RL, Stage-2 RL, comparator, resume, graceful
stop, and validation commands using `run_search.sh`. Include the four output
roots and supported six model/task profiles.

- [ ] **Step 2: Update repository instructions**

Replace old module/path references with the new package and output paths.
Preserve all scientific and Git protocol requirements.

- [ ] **Step 3: Update production-surface guard**

Require the new package and output skeleton, forbid old package roots and the
old launcher, and retain all existing retired-feature tokens.

- [ ] **Step 4: Remove obsolete CI test job**

Keep Ruff and dependency audit jobs. Do not leave a workflow that references a
deleted `tests/` directory.

- [ ] **Step 5: Verify and commit**

Run the production-surface guard, Ruff, shell syntax, Python compilation, and
`git diff --check`. Commit as:

```text
docs: document reorganized search workflow
```

Push immediately.

## Task 11: Run Complete Migration Verification Before Test Removal

**Files:** current reorganized source and temporarily updated `tests/`.

- [ ] **Step 1: Run complete server test suites**

Run `pytest` and `unittest` from the exact pushed task commit. Record all pass
and skip counts and explain every skip.

- [ ] **Step 2: Run Torch/CUDA production gates**

Run real fusion-map load, Rescale materialization, action installation,
terminal probe, one PPO update smoke, each comparator's deterministic smoke,
checkpoint resume, and full-validation loader smoke.

- [ ] **Step 3: Compare exact scientific snapshots**

Compare the baseline and reorganized sorted JSON. Expected: exact equality.
Compare dry-run options after permitted path normalization. Expected: exact
equality.

- [ ] **Step 4: Commit verification-only test updates**

Commit the temporarily updated tests so the exact tested tree is reproducible:

```text
test: verify reorganized production imports
```

Push immediately and record this commit as the verification source.

## Task 12: Remove Development Tests and Obsolete Analysis

**Files:** delete `tests/**`, `Model_analysis/**`; modify `pyproject.toml`,
`.github/workflows/ci.yml`, and `scripts/production_surface_guard.py`.

- [ ] **Step 1: Delete requested directories**

Use Git-aware deletion for `tests/` and `Model_analysis/`. Do not delete
fixtures, production guards, repository sync tools, or examples.

Also remove empty retired package roots and initializers left by the moves:
`Paean/`, `Rescale_optimizer/`, `blb_stage2_rl/`, `config/`, `stage1_rl/`, and
`tools/`. Keep `scripts/` because it owns repository and environment workflow.

- [ ] **Step 2: Remove test-only project configuration**

Remove pytest test-path settings and pytest-only optional dependencies. Keep
Ruff configuration for production code.

- [ ] **Step 3: Prove source identity since verification commit**

Run:

```bash
git diff --name-status <verification-commit>..HEAD
```

Expected: only `tests/**`, `Model_analysis/**`, and test-only CI/project config
changes. No production Python, shell, JSON, or preset file changes.

- [ ] **Step 4: Run final exact-source smoke**

From the final source commit run compilation, shell syntax, all preset
validations, all dry-runs, production-surface guard, import smoke, real action
materialization, one terminal probe, and one smoke per algorithm.

- [ ] **Step 5: Commit and push**

Commit as:

```text
chore: remove development tests and retired analysis
```

Push immediately.

## Task 13: Complete Handoff, Aggregate, and Three-Way Parity

**Files:**
- Create: `agent_handoffs/tasks/search-structure-reorganization-20260826.json`
- Create: `agent_handoffs/aggregates/search-structure-reorganization-20260826.json`

- [ ] **Step 1: Publish the task handoff**

Record source commit/tree, base commit/tree, complete changed paths, test and
GPU evidence, scientific snapshot SHA-256, normalized dry-run comparison, and
`aggregate_eligible=true`, `deployment_eligible=false`.

- [ ] **Step 2: Run task-finish guard**

Run `repo_sync_guard.py agent-finish` against the handoff-only tip. Fix any
protocol error before aggregation.

- [ ] **Step 3: Refresh all remote heads and run aggregate preflight**

Review every completed non-superseded handoff and record each remote head as
integrated, superseded, rejected, or unrelated. Do not merge archive, result,
or unfinished branches.

- [ ] **Step 4: Build and verify the aggregate**

Create a clean `codex/aggregate-*` branch from current canonical, integrate the
task source and handoff, run local static gates, push, and synchronize the
server only through Git to the exact aggregate commit/tree.

- [ ] **Step 5: Run final server gates**

Repeat final source smoke and scientific snapshot comparison on the exact
aggregate. Return evidence through a result branch.

- [ ] **Step 6: Finalize canonical**

Run `aggregate-finalize`, fast-forward `jk_standard_rl`, and update the clean
canonical checkout at `/Users/pengjunkai/Documents/USENIX Security CODE`.

- [ ] **Step 7: Verify parity**

Require exact equality of local canonical, remote canonical, and server
canonical commit IDs and tree IDs, with tracked-clean status in both local and
server checkouts.
