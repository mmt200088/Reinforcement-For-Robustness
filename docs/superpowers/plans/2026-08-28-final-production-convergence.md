# Final Production Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the repository to the production two-stage search and selected-config evaluation workflow with one JSON handoff at each boundary and one normal termination condition per algorithm.

**Architecture:** A new torch-free `best_config` module owns both JSON schemas and their validation. Search stages write those files atomically, Stage 2 and final evaluation reload them, and all derived BLB state is materialized through the existing fusion-map and Rescale path. Legacy selectors and dead modules are removed only after server-side parity evidence is captured.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers/Datasets, Bash, JSON, Git, CUDA.

---

### Task 1: Capture the pre-change production contract

**Files:**
- Server evidence only: `/hy-tmp/rfr_final_convergence_baseline_<commit>/`

- [ ] Record canonical commit/tree, all launcher dry-runs, six map audits, precision presets, representative action materializations, comparator trajectories, and current CLI rejection behavior.
- [ ] Run a real BERT-base MRPC baseline/probe canary and retain outputs only under `/hy-tmp`.
- [ ] Hash all evidence and confirm the server checkout remains tracked-clean.

### Task 2: Add the Stage-1 and complete-search JSON contracts

**Files:**
- Create: `src/rfr/search/common/best_config.py`
- Modify: `src/rfr/search/common/evaluator.py`
- Modify: `src/rfr/search/rl/stage2/runner.py`
- Modify: `src/rfr/search/comparators/common/stage1_runner.py`
- Modify: `src/rfr/search/comparators/common/stage2_runner.py`

- [ ] Define `stage1_best_config_v1` and `search_best_config_v1` constants.
- [ ] Implement strict loaders that validate algorithm, profile, layer count, GELU/Softmax domains, action-matrix shape, fusion `0/1`, precision `0/1/2`, protocol identity, and final-eval eligibility.
- [ ] Implement atomic writers whose editable payload is limited to Stage-1 vectors and the Stage-2 action matrix.
- [ ] Write `stage1_best_config.json` after every normally completed Stage-1 search.
- [ ] Make comparator Stage 2 reload the Stage-1 JSON and compare it with the in-memory selection before continuing.
- [ ] Write `search_best_config.json` after every normally completed two-stage search; do not write it for interruption, failure, or test-only state.
- [ ] Verify that reloaded actions produce the exact original full vector, option IDs, boost overrides, and final configuration fingerprint.
- [ ] Commit and push the coherent JSON-contract change.

### Task 3: Make JSON the only Stage-1 to Stage-2 path

**Files:**
- Modify: `run_search.sh`
- Modify: `configs/presets/*-stage1-rl.conf`
- Modify: `configs/presets/*-stage2-rl.conf`
- Modify: `src/rfr/cli/run.py`
- Modify: `src/rfr/search/common/evaluator.py`
- Modify: `src/rfr/search/rl/stage2/training.py`

- [ ] Add the single launcher option `--stage1-config PATH` for Stage 2.
- [ ] Remove `--stage2-fixed-config-source`, `--stage2-fixed-config`, manual vectors, Stage-1 record discovery, Stage-1 run IDs, all-max defaults, and in-memory resolution branches.
- [ ] Validate Stage-1 JSON before model loading and record its path and content hash in Stage-2 manifests/checkpoints.
- [ ] Update every Stage-2 preset so a user-supplied Stage-1 JSON is required.
- [ ] Confirm no old selector token remains in active source or README.
- [ ] Commit and push the coherent Stage-1/Stage-2 convergence change.

### Task 4: Make selected-search JSON the only final-eval path

**Files:**
- Modify: `run_search.sh`
- Modify: `src/rfr/cli/evaluate.py`
- Modify: `src/rfr/cli/evaluation_config.py`
- Modify: `src/rfr/cli/run.py`
- Modify: `src/rfr/search/common/evaluator.py`
- Modify: `src/rfr/evaluation/action_eval.py`
- Remove: `src/rfr/evaluation/action_grid.py`
- Remove: `src/rfr/evaluation/final_eval_layout.py`
- Remove or reduce: `src/rfr/evaluation/final_evaluation.py`
- Remove: `src/rfr/evaluation/embedded.py`
- Remove: `configs/reference/`
- Remove: `configs/evaluation/actions/`
- Remove: obsolete evaluation presets

- [ ] Reduce the command to `run_search.sh eval --config PATH` plus output and repeat controls.
- [ ] Derive algorithm, model, dataset, profile, and layer count from `search_best_config.json`.
- [ ] Rebuild the full action through the production fusion/Rescale materializer and evaluate exactly that one configuration.
- [ ] Keep repeated evaluation of the selected configuration only.
- [ ] Remove checkpoint scanning, resume-source, manual vectors, legacy reference JSONs, action templates, ranges, fixed overrides, random controls, and cost-matched controls.
- [ ] Reject incomplete, smoke, strict-ineligible, mismatched, or malformed search-best JSON.
- [ ] Commit and push the coherent final-eval convergence change.

### Task 5: Converge termination semantics

**Files:**
- Modify: `run_search.sh`
- Modify: `README.md`
- Modify: `configs/presets/*.conf`
- Modify: `src/rfr/cli/run.py`
- Modify: `src/rfr/search/common/evaluator.py`
- Modify: `src/rfr/search/rl/stage2/layerwise_runner.py`
- Modify: `src/rfr/search/rl/stage2/runner.py`
- Modify: `src/rfr/search/comparators/common/stage1_core.py`
- Modify: `src/rfr/search/comparators/common/stage1_runner.py`
- Modify: `src/rfr/search/comparators/common/stage2_core.py`
- Modify: `src/rfr/search/comparators/common/stage2_runner.py`
- Modify: algorithm-specific comparator modules as required

- [ ] Set Stage-1 PPO defaults to 51,000 maximum episodes and Stage-2 PPO defaults to 150,000 maximum episodes.
- [ ] Remove entropy, plateau, minimum-horizon, and patience completion paths from PPO while retaining graceful interruption and strict result certification.
- [ ] Expose GA update generations and remove GA patience/evaluation-cap termination.
- [ ] Expose BO-RF consecutive no-improvement evaluations and remove its evaluation cap.
- [ ] Expose Greedy no-improvement neighborhood rounds, default one, and remove evaluation-cap or alternate completion paths.
- [ ] Remove comparator smoke behavior from production source.
- [ ] Verify default termination reproduces the current default search semantics and configurable short runs complete through the same result writers.
- [ ] Commit and push the coherent termination-contract change.

### Task 6: Remove raw data and proven-dead production code

**Files:**
- Modify: `src/rfr/preparation/data/mrpc_reproducibility.py`
- Modify: `src/rfr/cli/run.py`
- Remove: `fixtures/reproducibility/mrpc_validation_v1.json`
- Remove: unreachable modules confirmed by the final static/dynamic audit
- Remove: `configs/preparation/fusion/maps/mrpc/block1_mrpc.json`
- Modify: `configs/preparation/fusion/maps/mrpc/_summary.json`
- Modify: `scripts/production_surface_guard.py`

- [ ] Rebuild MRPC validation order from the pinned dataset revision and seeds without storing raw sentences.
- [ ] Retain only the no-text train-probe selector/hash fixture.
- [ ] Remove modules and symbols absent from all production roots, dynamic imports, CUDA registration, and server coverage.
- [ ] Remove retired config/result templates and the unused Block-1 MRPC fusion map.
- [ ] Update the production-surface guard to enforce the new reduced surface, JSON-only handoffs, no raw dataset rows, and no retired selectors.
- [ ] Verify no personal data, PDF, weight, dataset archive, result log set, or stale runtime path remains.
- [ ] Commit and push the coherent dead-code/data cleanup.

### Task 7: Rewrite the concise production README

**Files:**
- Modify: `README.md`
- Modify: `local_assets/README.md`
- Modify: `outputs/README.md`
- Modify: code comments touched by the implementation

- [ ] Document environment setup, preparation, four algorithms, all termination settings, Stage-1 JSON, Stage-2 JSON input, search-best JSON, final evaluation, output paths, resume, and graceful interruption.
- [ ] Keep the README in natural English and remove descriptions of retired modes.
- [ ] Verify every documented command with the exact launcher parser.
- [ ] Scan tracked Python comments/docstrings for non-English or prompt-like text.
- [ ] Commit and push the documentation update.

### Task 8: Run server parity and real production gates

**Files:**
- Server evidence only: `/hy-tmp/rfr_final_convergence_<commit>/`

- [ ] Git-sync the exact task commit to a clean server worktree.
- [ ] Run compilation, shell syntax, imports, production-surface guard, and all documented dry-runs.
- [ ] Run old/new deterministic PPO, BO-RF, Greedy, and COINN-GA trajectory comparisons.
- [ ] Run Stage-1 JSON, Stage-2 JSON, edited JSON, and final-eval JSON round trips with malformed-input rejection.
- [ ] Run all six fusion-map audits and compare retained map bytes with the authoritative restore commit.
- [ ] Compare paper K metadata and executable simulation K for every preset, layer, and block.
- [ ] Run real BERT/CUDA canaries for PPO and all three comparators, including Stage-2 materialization and one selected-config final evaluation.
- [ ] Confirm search results, rewards, metrics, candidate semantics, and installed model configurations match pre-change evidence wherever the specification did not intentionally change.
- [ ] Store all test artifacts outside the local final checkout and hash the evidence.

### Task 9: Final audit, aggregation, and synchronization

**Files:**
- Create task handoff and aggregate manifest under `agent_handoffs/`

- [ ] Run the eleven-item final repository audit and record pass/fail evidence.
- [ ] Refresh every remote head and complete aggregate preflight.
- [ ] Push the exact aggregate, validate it on the server, and finalize the aggregate.
- [ ] Fast-forward `jk_standard_rl` under explicit aggregator authorization.
- [ ] Fast-forward `/Users/pengjunkai/Documents/USENIX Security CODE` and the server canonical checkout through Git.
- [ ] Verify local, remote, and server full commit/tree parity and tracked-clean state.
- [ ] Confirm no verification process remains on the server.
