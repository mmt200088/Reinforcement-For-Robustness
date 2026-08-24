# Production Cleanup Phase 2 Runtime Slice Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Remove rollback, ablation, and legacy runtime paths while preserving the verified six-profile Stage-1, Stage-2, comparator, final-evaluation, Rescale, and elastic-GPU behavior.

**Architecture:** Treat the Phase-1 integration commit as the golden behavior boundary. Add an executable production-surface guard first, then collapse one configurable axis at a time. Extract current helpers before deleting legacy modules, and run owner tests after every deletion group.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face, Bash, unittest, pytest, Ruff, Git.

---

### Task 1: Add the production-surface guard

**Files:**
- Create: scripts/production_surface_guard.py
- Create: tests/test_production_surface.py
- Modify: pyproject.toml

- [ ] **Step 1: Write the failing path-contract test**

Create tests that load tracked paths with git ls-files and reject:

~~~python
FORBIDDEN_PATHS = {
    "noise_rl_module_v2.py",
    "general_policy_module.py",
    "rl_tune_general.py",
    "rl_ga_compare_runner.py",
    "genetic_search_module.py",
    "greedy_search_module.py",
    "grpo_common.py",
    "blb_stage2_rl/network_variants.py",
    "blb_stage2_rl/action_mask.py",
    "blb_stage2_rl/policy.py",
    "blb_stage2_rl/parallel_runner.py",
    "blb_stage2_rl/sequential_env.py",
    "blb_stage2_rl/substage_env.py",
    "blb_stage2_rl/substage_runner.py",
    "blb_stage2_rl/osr.py",
    "blb_stage2_rl/fusion_curriculum.py",
    "blb_stage2_rl/protected_k1.py",
    "blb_stage2_rl/same_action_parity.py",
    "approximation.py",
    "approximation_exp.py",
    "bert-test.py",
    "commonsense_evaluate.py",
    "moe_sample.py",
}
~~~

Also reject active-source filenames containing .bak, legacy_results, or
ablation. Exclude preserved data/result roots from the design specification.

- [ ] **Step 2: Write forbidden-reference checks**

Scan retained Python, shell, JSON, TOML, and preset files for:

~~~python
FORBIDDEN_REFERENCES = (
    "legacy_v2",
    "shared_gtrxl_v1",
    "separate_critic_gtrxl_v1",
    "separate_critic_mlp_v1",
    "stage2_rl_devices",
    "substage_mode",
    "osr_scan_only",
    "rescale_invoker_kind",
    "HeuristicStubInvoker",
    "SubprocessInvoker",
)
~~~

- [ ] **Step 3: Run RED on the server**

~~~bash
python -m unittest -v tests.test_production_surface
~~~

Expected: failure lists current obsolete paths and references.

- [ ] **Step 4: Implement the guard CLI**

The script must obtain tracked paths through git ls-files -z, exclude preserved
result roots, report forbidden paths and references separately, return zero
only when both lists are empty, and support --json.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/production_surface_guard.py tests/test_production_surface.py pyproject.toml
git commit -m "test: define the production source surface"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 2: Collapse Stage-2 to one policy network

**Files:**
- Create: blb_stage2_rl/policy_network.py
- Modify: blb_stage2_rl/sequential_policy.py
- Modify: blb_stage2_rl/sequential_runner.py
- Modify: blb_stage2_rl/runner.py
- Modify: layer_importance_evaluator.py
- Modify: rl_tune.py
- Modify: llama_7B_LayerImportance.sh
- Create: tests/test_blb_policy_network.py
- Delete: blb_stage2_rl/network_variants.py
- Delete: tests/test_blb_policy_network_variants.py

- [ ] **Step 1: Freeze the current network contract**

~~~python
POLICY_NETWORK_ID = "shared_gtrxl_small_v1"
POLICY_RL_VARIANT = "blb_v3_layerwise_robust_shared_gtrxl_small_v1"
POLICY_ARCHITECTURE = {
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 256,
}
~~~

Tests assert this exact identity and architecture, a shared actor/critic trunk,
stable state-dict keys, and an explicit checkpoint identity.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v tests.test_blb_policy_network
~~~

Expected: blb_stage2_rl.policy_network is missing.

- [ ] **Step 3: Implement the fixed contract**

policy_network.py binds the fixed architecture into the algorithm contract and
rejects any policy shape mismatch. Checkpoint validation accepts only the
explicit current identity and never infers an old variant. Keep the serialized
key policy_network_variant with the single value shared_gtrxl_small_v1 so
current small-network checkpoints remain readable.

- [ ] **Step 4: Remove alternate critic construction**

SequentialPolicyConfig has no network selector or custom-architecture flag.
BLBStage2SequentialPolicy retains the current shared GTrXL trunk and value head,
and removes independent GTrXL/MLP critic builders and RNG branches.

- [ ] **Step 5: Remove the CLI selector**

Delete --blb-v3-policy-network-variant parsing, forwarding, aliases, and
metadata compatibility. Persist POLICY_NETWORK_ID automatically under the
existing policy_network_variant checkpoint field.

- [ ] **Step 6: Run focused tests**

~~~bash
python -m unittest -v \
  tests.test_blb_policy_network \
  tests.test_blb_layerwise_policy \
  tests.test_blb_layerwise_runner \
  tests.test_stage2_persistent_launcher
~~~

Expected: all tests pass.

- [ ] **Step 7: Commit**

~~~bash
git add blb_stage2_rl layer_importance_evaluator.py rl_tune.py \
  llama_7B_LayerImportance.sh tests
git commit -m "refactor: keep the production Stage-2 policy network"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 3: Extract current checkpoint and stop handling

**Files:**
- Create: stage1_rl/checkpoint.py
- Create: blb_stage2_rl/runtime_control.py
- Modify: layer_importance_evaluator.py
- Modify: blb_stage2_rl/sequential_runner.py
- Modify: blb_stage2_rl/persistence.py
- Test: tests/test_stage1_elastic_checkpoint.py
- Test: tests/test_blb_layerwise_runner.py

- [ ] **Step 1: Add direct helper tests**

Tests import current Stage-1 atomic checkpoint save/load from
stage1_rl.checkpoint and graceful-stop state from
blb_stage2_rl.runtime_control. A missing or old dataset protocol hash raises
RuntimeError.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v \
  tests.test_stage1_elastic_checkpoint \
  tests.test_blb_layerwise_runner
~~~

Expected: the new modules are missing.

- [ ] **Step 3: Move active implementations**

Move Stage-1 checkpoint save/load, detail-file recovery, and size tracking into
stage1_rl/checkpoint.py. Move signal state and STOP_RL file handling into
blb_stage2_rl/runtime_control.py. Keep observable filenames and atomic write
semantics unchanged.

- [ ] **Step 4: Replace legacy imports**

~~~bash
rg -n "from noise_rl_module_v2 import" \
  layer_importance_evaluator.py blb_stage2_rl stage1_rl
~~~

Expected after replacement: no output.

- [ ] **Step 5: Run focused tests and commit**

~~~bash
python -m unittest -v \
  tests.test_stage1_elastic_checkpoint \
  tests.test_stage1_search_baselines \
  tests.test_blb_layerwise_runner \
  tests.test_stage2_ga_extension_preflight
git add stage1_rl blb_stage2_rl layer_importance_evaluator.py tests
git commit -m "refactor: own current checkpoint and stop handling"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 4: Remove legacy Stage-2 and General-RL

**Files:**
- Delete: noise_rl_module_v2.py
- Delete: general_policy_module.py
- Delete: rl_tune_general.py
- Delete: grpo_common.py
- Modify: rl_tune.py
- Modify: layer_importance_evaluator.py
- Modify: final_evaluation_module.py
- Modify: Paean/config.py
- Modify: Paean/run_final_eval.py
- Modify: llama_7B_LayerImportance.sh
- Delete: tests/test_grpo_common.py
- Delete: tests/test_grpo_wiring.py

- [ ] **Step 1: Add launcher rejection tests**

The launcher rejects general, --stage2-rl-variant, legacy_v2, --rl-algo, and
grpo. Retained Stage-1 and Stage-2 commands still produce valid dry-run
commands.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v tests.test_stage2_persistent_launcher
~~~

Expected: legacy/general options remain accepted.

- [ ] **Step 3: Remove dispatch and imports**

rl_tune.py always constructs the BLB layerwise Stage-2 path. Paean accepts only
the current BLB result schema. The launcher removes the general command and
Stage-2 implementation selector.

- [ ] **Step 4: Verify no retained reference and delete**

~~~bash
rg -n "noise_rl_module_v2|general_policy_module|rl_tune_general|grpo_common" \
  --glob '*.py' --glob '*.sh' --glob '*.json' --glob '*.conf' \
  --glob '!tests/test_production_surface.py'
~~~

Expected after deletion: no output.

- [ ] **Step 5: Run tests and commit**

~~~bash
python -m unittest -v \
  tests.test_stage2_persistent_launcher \
  tests.test_blb_layerwise_runner \
  tests.test_blb_paean_handoff_ordinary \
  tests.test_final_evaluation_config_cache
git add -A
git commit -m "refactor: remove legacy and general RL paths"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 5: Make layerwise PPO the only Stage-2 runner

**Files:**
- Create: blb_stage2_rl/training.py
- Modify: blb_stage2_rl/sequential_runner.py
- Modify: blb_stage2_rl/sequential_policy.py
- Modify: blb_stage2_rl/layerwise_runner.py
- Modify: layer_importance_evaluator.py
- Modify: rl_tune.py
- Delete: blb_stage2_rl/runner.py
- Delete: blb_stage2_rl/action_mask.py
- Delete: blb_stage2_rl/policy.py
- Delete: blb_stage2_rl/parallel_runner.py
- Delete: blb_stage2_rl/sequential_env.py
- Delete: blb_stage2_rl/substage_env.py
- Delete: blb_stage2_rl/substage_runner.py
- Delete: blb_stage2_rl/osr.py
- Delete: blb_stage2_rl/fusion_curriculum.py
- Delete: blb_stage2_rl/protected_k1.py
- Delete: blb_stage2_rl/same_action_parity.py

- [ ] **Step 1: Add the fixed training-config test**

Create tests/test_stage2_production_config.py. The production dataclass must not
contain these removed fields:

~~~python
REMOVED_FIELDS = {
    "sequential_rl",
    "stage2_rl_devices",
    "stage2_workers_per_device",
    "substage_mode",
    "osr_results_path",
    "fusion_neighbor_curriculum_enabled",
    "protected_k1_enabled",
    "action_mask_enabled",
    "warmstart_neighbor_sampling",
    "guarded_radius2_enabled",
    "reward_design",
    "decision_granularity",
}
~~~

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v tests.test_stage2_production_config
~~~

Expected: training.py is missing or removed fields remain.

- [ ] **Step 3: Build the focused facade**

training.py exposes BLBStage2TrainConfig and one
run_stage2_search(evaluator, fixed_stage1, config) entrypoint. Configuration
retains current scientific limits, PPO budget, comparator budget, batch size,
in-process Rescale paths, reward devices, A/B/C trials, and convergence
settings. Layerwise fusion-count robust PPO is fixed internally.

- [ ] **Step 4: Delete alternate branches**

Remove single-shot, per-block, substage, OSR, curriculum, protected-K1,
warmstart, action-mask, fast-reward batching, and episode-parallel branches
from sequential_runner.py. Preserve layerwise actions, robust baseline,
search-gate banks, candidate store, strict top-5, PPO, process reward probes,
and current checkpoint diagnostics.

Move any comparison helper still needed by acceptance tests from
same_action_parity.py into tests/support before deleting the runtime module.

- [ ] **Step 5: Delete retired tests**

~~~text
tests/test_blb_action_mask.py
tests/test_blb_fusion_curriculum.py
tests/test_blb_osr.py
tests/test_blb_protected_k1.py
tests/test_blb_substage_assembly.py
tests/test_blb_warmstart_resume.py
tests/test_stage2_parallel_runner.py
~~~

- [ ] **Step 6: Run Stage-2 suites**

~~~bash
python -m unittest -v \
  tests.test_stage2_production_config \
  tests.test_blb_layerwise_action \
  tests.test_blb_layerwise_env \
  tests.test_blb_layerwise_policy \
  tests.test_blb_layerwise_runner \
  tests.test_blb_search_baseline_runner \
  tests.test_probe_runner_process_backend
~~~

Expected: all tests pass, with CUDA-only skips allowed when unavailable.

- [ ] **Step 7: Commit**

~~~bash
git add -A
git commit -m "refactor: keep only layerwise Stage-2 training"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 6: Keep only real in-process Rescale execution

**Files:**
- Modify: rescale_optimizer_bridge.py
- Modify: blb_stage2_rl/optimizer_cost.py
- Modify: Paean/blb_action_eval.py
- Modify: Paean/config.py
- Modify: Paean/run_final_eval.py
- Modify: rl_tune.py
- Modify: llama_7B_LayerImportance.sh
- Test: tests/test_blb_optimizer_cost_consistency.py
- Test: tests/test_blb_paean_handoff_ordinary.py

- [ ] **Step 1: Add no-fallback tests**

Tests require one in-process invoker factory, fail on missing profile/config,
and reject --rescale-invoker-kind.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v \
  tests.test_blb_optimizer_cost_consistency \
  tests.test_blb_paean_handoff_ordinary
~~~

Expected: alternate invokers remain reachable.

- [ ] **Step 3: Remove alternate invokers**

Retain one factory:

~~~python
def build_rescale_invoker(*, root, profile):
    return InProcessInvoker.from_profile(root=root, profile=profile)
~~~

Initialization and replan failures propagate without fallback.

- [ ] **Step 4: Run tests and commit**

~~~bash
python -m unittest -v \
  tests.test_blb_optimizer_cost_consistency \
  tests.test_blb_action_materialization \
  tests.test_blb_stage2_eval_single_path_static \
  tests.test_blb_paean_handoff_ordinary
git add rescale_optimizer_bridge.py blb_stage2_rl Paean rl_tune.py \
  llama_7B_LayerImportance.sh tests
git commit -m "refactor: require in-process Rescale execution"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 7: Consolidate comparator entrypoints

**Files:**
- Modify: rl_tune_genetic.py
- Modify: layer_importance_evaluator.py
- Modify: llama_7B_LayerImportance.sh
- Modify: stage1_rl/search_baselines.py
- Modify: stage1_rl/search_runner.py
- Modify: blb_stage2_rl/search_baselines.py
- Modify: blb_stage2_rl/search_baseline_runner.py
- Modify: Paean/run_final_eval.py
- Delete: genetic_search_module.py
- Delete: greedy_search_module.py
- Delete: rl_ga_compare_runner.py

- [ ] **Step 1: Add the comparator CLI matrix test**

The launcher exposes exactly BO-RF, Greedy, and COINN-GA. run ga and compare
are rejected. Every comparator uses the shared Stage-1 and Stage-2 runners.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v \
  tests.test_blb_search_backend_wiring \
  tests.test_stage2_persistent_launcher
~~~

Expected: old GA/compare paths remain visible.

- [ ] **Step 3: Make rl_tune_genetic.py a thin shared-flow entrypoint**

Remove causal-LM, LoRA, old GA class, and old final-eval wrapper imports. Load
one supported BERT/GLUE profile, construct the shared evaluator, and forward
the selected comparator backend.

- [ ] **Step 4: Delete old modules after reference scan**

~~~bash
rg -n "genetic_search_module|greedy_search_module|rl_ga_compare_runner" \
  --glob '*.py' --glob '*.sh' --glob '*.json' --glob '*.conf'
~~~

Expected after deletion: no output.

- [ ] **Step 5: Run tests and commit**

~~~bash
python -m unittest -v \
  tests.test_stage1_search_baselines \
  tests.test_blb_search_baselines \
  tests.test_blb_search_baseline_runner \
  tests.test_blb_search_backend_wiring \
  tests.test_stage2_persistent_launcher
git add -A
git commit -m "refactor: keep the three formal comparators"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 8: Reduce launchers, presets, and preparation tools

**Files:**
- Rewrite: llama_7B_LayerImportance.sh
- Modify: Paean/run_final_eval.sh
- Create: six presets/bert-*-*-stage2-rl.conf files
- Delete: run_all_experiments.sh
- Delete: run_noise_scaling_sweep.sh
- Delete: run_glue_submission.sh
- Delete: run_gelu_analysis.sh
- Delete: generate_glue_submission.py
- Delete: experiment/core/
- Delete: experiment/scripts/
- Delete: root-level Python scripts under experiments/ while preserving every
  result subdirectory recorded by the Phase-1 tree receipt
- Delete: tools/aggregate_seeds.py
- Delete: tools/experiments_log.py
- Delete: tools/paper_figures.py
- Delete: tools/run_multi_seed.sh

- [ ] **Step 1: Add exact command-contract tests**

Exercise these commands with a fake Python executable or --dry-run:

~~~text
run rl --preset bert-base-mrpc-stage1-rl --fresh
run rl --preset bert-base-mrpc-stage2-rl --fresh
run bo_rf --dataset mrpc --model-type bert-base --comparator-stage1-only --fresh
run greedy --dataset mrpc --model-type bert-base --fresh
run coinn_ga --dataset mrpc --model-type bert-base --fresh
eval --preset bert-base-mrpc-final-eval
~~~

- [ ] **Step 2: Rewrite the launcher**

Accept only run rl, run bo_rf, run greedy, run coinn_ga, eval, and
--list-presets. Keep fresh/resume, supported profile selection, scientific
limits, search budgets, reward devices, output root, and final-eval config.

- [ ] **Step 3: Add six Stage-2 presets**

Each preset fixes mode stage2-only, batch 64, 150000 episodes, rollout 120,
learning rate 5e-5, 0.001 precision limits, stability multiplier 2.0, online
K=3, probe size 256, baseline 5x3, and promotion/final banks of 15 trials.
Only model, dataset, and logfile differ.

- [ ] **Step 4: Keep this preparation-tool allowlist**

~~~text
scripts/audit_fusion_count_maps.py
scripts/blb_build_fusion_count_map.py
scripts/blb_export_action_registry.py
scripts/blb_make_run_manifest.py
scripts/blb_phase0_preflight.py
scripts/build_glue_train_probe_fixture.py
scripts/elastic_gpu_supervisor.py
scripts/install_git_protocol_hooks.sh
scripts/repo_sync_guard.py
scripts/setup_cuda124_env.sh
scripts/production_surface_guard.py
tools/validate_preset.py
~~~

Delete other one-off scripts after confirming no retained reference. Preserve
experiment/outputs and all result subdirectories under experiments byte-for-byte.

- [ ] **Step 5: Run tests and commit**

~~~bash
python -m unittest -v \
  tests.test_stage2_persistent_launcher \
  tests.test_supported_profile_matrix \
  tests.test_glue_data_protocol \
  tests.test_audit_fusion_count_maps \
  tests.test_blb_build_fusion_count_map \
  tests.test_repo_sync_guard
bash -n llama_7B_LayerImportance.sh Paean/run_final_eval.sh
git add -A
git commit -m "refactor: expose only production launch paths"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 9: Remove obsolete configuration and backup source

**Files:**
- Delete: Model_analysis/analyze_all_distribution_new.py.bak_ln_norm_20260601_2152
- Delete: Rescale_optimizer/configs/mrpc/static_skeletons_mrpc_old.json
- Delete: Rescale_optimizer/rescale_optimizer/*.bak_*
- Delete: Rescale_optimizer/scripts/*.bak_*
- Delete: approximation.py
- Delete: approximation_exp.py
- Delete: bert-test.py
- Delete: commonsense_evaluate.py
- Delete: moe_sample.py
- Delete: text_utils.py after its one-off script callers are removed
- Modify: Model_analysis/configs/approx_per_dataset.json
- Modify: glue_configs.json
- Modify: config/run_layout.py
- Modify: pyproject.toml

- [ ] **Step 1: Add a six-profile configuration test**

Require exactly mrpc, mrpc_large, rte, rte_large, sst2, and sst2_large config
families. Reject unsupported labels and backup filenames outside preserved
result roots.

- [ ] **Step 2: Run RED**

~~~bash
python -m unittest -v \
  tests.test_supported_profile_matrix \
  tests.test_run_layout \
  tests.test_production_surface
~~~

Expected: backup or unsupported paths fail.

- [ ] **Step 3: Delete and normalize**

Remove dated backups and old skeletons. Keep six production Rescale config
directories, six fusion-map directories, and every file referenced by current
manifests.

- [ ] **Step 4: Run tests and commit**

~~~bash
python -m unittest -v \
  tests.test_supported_profile_matrix \
  tests.test_run_layout \
  tests.test_blb_fusion_count_map \
  tests.test_blb_optimizer_cost_consistency \
  tests.test_production_surface
git add -A
git commit -m "chore: remove obsolete source and configuration"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 10: Rewrite comments in retained runtime code

**Files:**
- Modify: retained production Python, shell, and preset files
- Test: tests/test_production_surface.py

- [ ] **Step 1: Add stale-comment scans**

Reject active comments/docstrings containing prompt language, user-request
narration, dated debugging chronology, rollback instructions, ablation-arm
notes, and AI-agent names. Exclude generic agent protocol documents from the
AI-name rule.

- [ ] **Step 2: Remove stale and commented-out code**

Delete historical narratives, abandoned commands, commented imports, dated
hotfix explanations, and obvious syntax narration. Preserve shebangs,
encoding declarations, noqa/type/format pragmas, licenses, and non-obvious
invariants.

- [ ] **Step 3: Add concise invariant comments**

Keep short English comments for train-probe isolation, action decode, MPC
ciphertext versus simulation K, hard-priority reward, deterministic seeds,
Rescale ownership, checkpoint atomicity, and elastic GPU assignment.

- [ ] **Step 4: Format retained code**

~~~bash
ruff format \
  glue_data_protocol.py rl_tune.py rl_tune_genetic.py \
  layer_importance_evaluator.py final_evaluation_module.py \
  blb_stage2_rl stage1_rl Paean config scripts
ruff check \
  glue_data_protocol.py rl_tune.py rl_tune_genetic.py \
  blb_stage2_rl stage1_rl Paean config scripts
~~~

Expected: both commands exit zero.

- [ ] **Step 5: Run tests and commit**

~~~bash
python -m unittest -v \
  tests.test_production_surface \
  tests.test_glue_data_protocol \
  tests.test_blb_layerwise_runner \
  tests.test_stage2_persistent_launcher
git diff --check
git add -A
git commit -m "style: document only production invariants"
git push origin codex/task-production-code-cleanup-20260825
~~~

### Task 11: Verify the runtime slice

**Files:**
- Test: tests/
- Validate: all retained production source

- [ ] **Step 1: Run the production guard**

~~~bash
python scripts/production_surface_guard.py --json
~~~

Expected: zero forbidden paths and references.

- [ ] **Step 2: Compile and validate shell**

~~~bash
python -m compileall -q \
  blb_stage2_rl stage1_rl Paean Model_analysis \
  Rescale_optimizer/rescale_optimizer \
  glue_data_protocol.py rl_tune.py rl_tune_genetic.py \
  layer_importance_evaluator.py final_evaluation_module.py
bash -n llama_7B_LayerImportance.sh Paean/run_final_eval.sh
~~~

- [ ] **Step 3: Run full server tests**

~~~bash
python -m unittest discover -v
python -m pytest -q
~~~

Expected: all retained tests pass or have explicit environment-only skips.

- [ ] **Step 4: Repeat scientific-state parity**

Compare Phase 1 and Phase 2 for six probe identities, policy state keys,
Stage-1 action/reward/checkpoint state, Stage-2 action/materialization/reward/
cost/candidate state, comparator fixed observations, final-eval structured
state, and MPC execution snapshot hash. Scientific fields must compare exactly.

- [ ] **Step 5: Recheck preserved result trees**

Compare current path tree IDs with
/tmp/rfr-production-cleanup-preserved-trees.before.

Expected: every preserved data/result tree is unchanged.

- [ ] **Step 6: Publish Phase-2 evidence**

Create a result branch from the exact Phase-2 source with compact evidence
under experiments/server_command_runs/production_cleanup_phase2_<short_sha>_20260825/
and validate result-check with remote parity.

Phase 3 starts only after this gate passes.
