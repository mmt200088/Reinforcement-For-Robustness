# Production Cleanup Phase 1 Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the latest six-profile train-probe data protocol with the canonical MPC truncation paper semantics before any source deletion begins.

**Architecture:** Start from canonical `f86359bc`, which already contains the MPC metadata/execution split, and merge train-probe source `d67100a7`. Preserve both feature lines, resolve the sole overlapping file deliberately, and establish an exact tested integration commit that Phase 2 uses as its deletion baseline.

**Tech Stack:** Git worktrees and guard scripts, Python 3.10, PyTorch, Hugging Face datasets/transformers, unittest, pytest, Bash.

---

### Task 1: Freeze the two source boundaries

**Files:**
- Inspect: `agent_handoffs/aggregates/20260825-mpc-truncation-paper-semantics.json`
- Inspect: `docs/superpowers/specs/2026-08-25-production-code-cleanup-design.md`

- [ ] **Step 1: Refresh all remote heads**

```bash
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*'
```

Expected: command exits zero and `origin/jk_standard_rl` remains
`f86359bce6ac432cd7f431c21627efd5930f9e02`.

- [ ] **Step 2: Verify the cleanup branch base**

```bash
git merge-base --is-ancestor \
  f86359bce6ac432cd7f431c21627efd5930f9e02 HEAD
git status --porcelain=v2 --branch
```

Expected: the ancestry check exits zero and the worktree is tracked-clean.

- [ ] **Step 3: Verify train-probe source provenance**

```bash
test "$(git rev-parse d67100a7cb0444275498b780c10b46631a7577c1^{tree})" = \
  "b00cd632871ccf0d19b1a6883b39673b3723862a"
test "$(git merge-base HEAD d67100a7cb0444275498b780c10b46631a7577c1)" = \
  "480e154053b1303e140077a05c46295cab95ef0a"
```

Expected: both checks exit zero.

- [ ] **Step 4: Record preserved result-tree identities outside the repository**

```bash
for path in \
  'server_backups' \
  'rl_training_data_points' \
  'Parting Chapter' \
  'Prelude Chapter' \
  'Previous Chapter' \
  'Previous Chapter Server Reserve' \
  'Paean/outputs' \
  'experiment/outputs' \
  'gelu_analysis' \
  'glue_submission' \
  'reports' \
  'Model_analysis/model_statistics/weight_hist_out' \
  'Rescale_optimizer/diagnose_certacc_output' \
  'glue_final_configs_best_genetic.json' \
  'glue_final_configs_best_ppo.json'
do
  printf '%s %s\n' "$path" "$(git rev-parse "HEAD:$path")"
done > /tmp/rfr-production-cleanup-preserved-trees.before
git ls-tree -d --name-only HEAD:experiments | while IFS= read -r directory
do
  path="experiments/$directory"
  printf '%s %s\n' "$path" "$(git rev-parse "HEAD:$path")"
done >> /tmp/rfr-production-cleanup-preserved-trees.before
```

Expected: every declared result path is recorded outside tracked source.

### Task 2: Merge the train-probe implementation

**Files:**
- Merge source: `d67100a7cb0444275498b780c10b46631a7577c1`
- Resolve: `AGENTS.md`
- Add through merge: `glue_data_protocol.py`
- Add through merge: `fixtures/reproducibility/glue_train_probe_v1.json`
- Add through merge: `scripts/build_glue_train_probe_fixture.py`

- [ ] **Step 1: Start a non-fast-forward merge without committing**

```bash
git merge --no-ff --no-commit d67100a7cb0444275498b780c10b46631a7577c1
```

Expected: Git reports only `AGENTS.md` as a content conflict. If another path
conflicts, stop and review it before proceeding.

- [ ] **Step 2: Verify the conflict set**

```bash
git diff --name-only --diff-filter=U
```

Expected output:

```text
AGENTS.md
```

- [ ] **Step 3: Resolve `AGENTS.md` semantically**

Retain these contracts in the resolved file:

```markdown
- Six supported profiles: BERT-base/BERT-large with MRPC/RTE/SST-2.
- Profile and all search gates use the fixed 256-example train probe.
- Full validation is final-evaluation-only.
- Paper ciphertext K is metadata; output_truncation_k is simulation K.
- Only the current small shared GTrXL checkpoint schema may resume.
```

Do not keep duplicate chronological entries; Phase 3 replaces this document
with the concise final version.

- [ ] **Step 4: Verify both feature lines before committing**

```bash
python3 - <<'PY'
from glue_data_protocol import supported_profiles
from blb_stage2_rl.precision_presets import PRECISION_PRESETS

assert len(supported_profiles()) == 6
assert [p.simulation_k_by_block for p in PRECISION_PRESETS] == [
    (11, 10, 10, 12, 11),
    (9, 8, 8, 10, 9),
    (7, 6, 6, 8, 7),
]
assert [p.ciphertext_k_by_block for p in PRECISION_PRESETS] == [
    (13, 13, 13, 13, 13),
    (12, 12, 12, 12, 12),
    (11, 11, 11, 12, 11),
]
PY
```

Expected: exits zero.

- [ ] **Step 5: Commit the integration boundary**

```bash
git add AGENTS.md
git commit -m "integrate: combine train probe and MPC semantics"
git push origin codex/task-production-code-cleanup-20260825
```

Expected: one merge commit is pushed; no cleanup deletion is included.

### Task 3: Run focused integration gates on the server

**Files:**
- Test: `tests/test_glue_data_protocol.py`
- Test: `tests/test_search_split_isolation.py`
- Test: `tests/test_supported_profile_matrix.py`
- Test: `tests/test_blb_layerwise_precision_presets.py`
- Test: `tests/test_blb_layerwise_action.py`
- Test: `tests/test_blb_layerwise_runner.py`
- Test: `tests/test_stage2_persistent_launcher.py`

- [ ] **Step 1: Synchronize an isolated server checkout through Git**

Use a blobless sparse checkout at the exact pushed integration commit. Verify:

```bash
git rev-parse HEAD
git rev-parse HEAD^{tree}
git status --porcelain=v1
```

Expected: exact intended commit/tree and no tracked changes.

- [ ] **Step 2: Run the data and split suite**

```bash
python -m unittest -v \
  tests.test_glue_data_protocol \
  tests.test_glue_dataset_loading \
  tests.test_search_split_isolation \
  tests.test_profile_train_probe_protocol \
  tests.test_supported_profile_matrix
```

Expected: all tests pass; only explicit environment-only skips are accepted.

- [ ] **Step 3: Run Stage-1 and Stage-2 integration suites**

```bash
python -m unittest -v \
  tests.test_stage1_search_baselines \
  tests.test_stage1_elastic_checkpoint \
  tests.test_blb_layerwise_precision_presets \
  tests.test_blb_layerwise_action \
  tests.test_blb_layerwise_runner \
  tests.test_blb_search_baselines \
  tests.test_blb_search_baseline_runner \
  tests.test_stage2_persistent_launcher
```

Expected: all tests pass; CUDA tests may skip only when CUDA is unavailable.

- [ ] **Step 4: Run final-evaluation and checkpoint gates**

```bash
python -m unittest -v \
  tests.test_final_evaluation_config_cache \
  tests.test_final_eval_normalize_arrays \
  tests.test_blb_paean_handoff_ordinary \
  tests.test_blb_two_stage_binding_ordinary \
  tests.test_stage2_ga_extension_preflight
```

Expected: all tests pass and old validation-probe checkpoints fail closed.

### Task 4: Prove MPC executable parity on the integrated source

**Files:**
- Runtime: `blb_stage2_rl/precision_presets.py`
- Runtime: `blb_stage2_rl/layerwise_action.py`
- Test helper: `tests/test_blb_layerwise_action.py`

- [ ] **Step 1: Generate the production execution snapshot**

Use the snapshot procedure recorded in
`experiments/server_command_runs/aggregate_mpc_truncation_paper_semantics_909f0589_20260825/`:

- uniform H/M/L with Block4 fusion 0 and 1;
- mixed 12-layer action;
- complete legacy vector;
- decoded simulation K;
- fusion option IDs;
- boosted overrides;
- all variable-cost fields.

- [ ] **Step 2: Compare against the frozen baseline**

```bash
cmp -s execution_before_480e154.json execution_integrated.json
sha256sum execution_before_480e154.json execution_integrated.json
```

Expected: `cmp` exits zero and both hashes equal:

```text
68a50ef270d894f3995bd01437b6febcb0bd2b3c757b42edb03485ad2ceb63e7
```

- [ ] **Step 3: Audit forbidden runtime changes**

```bash
git diff --name-only \
  f86359bce6ac432cd7f431c21627efd5930f9e02..HEAD | \
  grep -E '^(function_handler.py|blb_stage2_rl/(truncation_fused_cuda.py|block3_fused_cuda.py|block5_fused_cuda.py|action_space.py))$'
```

Expected: no output.

### Task 5: Run complete integration verification

**Files:**
- Test: `tests/`
- Validate: `llama_7B_LayerImportance.sh`
- Validate: `Paean/run_final_eval.sh`

- [ ] **Step 1: Compile integration modules**

```bash
python -m py_compile \
  glue_data_protocol.py \
  rl_tune.py \
  layer_importance_evaluator.py \
  final_evaluation_module.py \
  blb_stage2_rl/*.py \
  stage1_rl/*.py \
  Paean/*.py \
  Model_analysis/*.py
bash -n llama_7B_LayerImportance.sh Paean/run_final_eval.sh
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Run full unittest**

```bash
python -m unittest discover -v
```

Expected: all retained tests pass or have an explicit environment-only skip.

- [ ] **Step 3: Run full pytest**

```bash
python -m pytest -q
```

Expected: all retained tests pass or have an explicit environment-only skip.

- [ ] **Step 4: Recheck preserved data/result trees**

```bash
for path in \
  'server_backups' \
  'rl_training_data_points' \
  'Parting Chapter' \
  'Prelude Chapter' \
  'Previous Chapter' \
  'Previous Chapter Server Reserve' \
  'Paean/outputs' \
  'experiment/outputs' \
  'gelu_analysis' \
  'glue_submission' \
  'reports' \
  'Model_analysis/model_statistics/weight_hist_out' \
  'Rescale_optimizer/diagnose_certacc_output' \
  'glue_final_configs_best_genetic.json' \
  'glue_final_configs_best_ppo.json'
do
  printf '%s %s\n' "$path" "$(git rev-parse "HEAD:$path")"
done > /tmp/rfr-production-cleanup-preserved-trees.after-integration
git ls-tree -d --name-only HEAD:experiments | while IFS= read -r directory
do
  path="experiments/$directory"
  printf '%s %s\n' "$path" "$(git rev-parse "HEAD:$path")"
done >> /tmp/rfr-production-cleanup-preserved-trees.after-integration
cmp -s \
  /tmp/rfr-production-cleanup-preserved-trees.before \
  /tmp/rfr-production-cleanup-preserved-trees.after-integration
```

Expected: `cmp` exits zero.

- [ ] **Step 5: Publish compact Phase-1 evidence**

Create a result branch from the exact integration commit containing only:

```text
experiments/server_command_runs/production_cleanup_phase1_<short_sha>_20260825/
```

Include source commit/tree, focused/full test summaries, the two MPC snapshots,
their SHA-256 values, and the preserved-tree comparison. Validate with
`repo_sync_guard.py result-check --require-remote`.

Phase 2 may begin only after every step above passes.
