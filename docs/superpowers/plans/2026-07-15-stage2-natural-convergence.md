# Stage-2 Natural Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make formal Stage-2 layerwise PPO optimize reward only and run without a fixed episode budget until its policy and robust-feasible frontier naturally converge.

**Architecture:** Keep the existing factorized PPO, reward, candidate store, and checkpoint boundaries. Change only the active layerwise entropy/termination contract: zero entropy coefficient at every update, `total_episodes=0` as an unbounded iterator, and exit after the existing dual-entropy plus feasible-frontier stall signal. Thread this contract through the parser, evaluator, runner, launcher, preset, manifests, checkpoints, and summaries while retaining positive budgets for bounded tests and smoke runs.

**Tech Stack:** Python 3, PyTorch PPO, NumPy, unittest/pytest, Bash launcher, JSON/JSONL persistence.

---

### Task 1: Lock The Natural-Convergence Core Contract

**Files:**
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`

- [ ] **Step 1: Write failing convergence and zero-entropy tests**

Replace the old fixed-horizon entropy tests with assertions that the tracker can
converge below 30k after 100 unchanged feasible-frontier updates, and add a real
`train_layerwise` test that captures `ent_coef_override` and expects `0.0`.

- [ ] **Step 2: Verify the tests fail for the old behavior**

Run:

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py -k 'convergence or entropy' -q
```

Expected: failures caused by the 30k gate and positive cosine entropy value.

- [ ] **Step 3: Implement reward-only PPO and episode-independent convergence**

Remove the active cosine coefficient call, pass `ent_coef_override=0.0`, remove
the 30k/60k conditions, and retain the dual entropy, feasible candidate, and
100-update frontier-stall gates.

- [ ] **Step 4: Verify focused tests pass**

Run the command from Step 2 and expect zero failures.

### Task 2: Add An Unbounded Layerwise Training Loop

**Files:**
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`

- [ ] **Step 1: Write a failing unbounded-loop test**

Use `total_episodes=0`, a one-update convergence setup, and assert that the loop
collects an episode, performs one PPO update, stops after convergence, and
reports the actual completed episode count. Retain a bounded-budget test that
stops without inventing convergence.

- [ ] **Step 2: Verify zero currently produces no training**

Run:

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py -k 'unbounded or bounded_budget' -q
```

Expected: the unbounded test fails because `range(0)` collects no episodes.

- [ ] **Step 3: Implement bounded/unbounded iteration**

Use a monotonic local episode counter. For positive budgets, stop after that
many new episodes; for zero, keep producing complete update windows. Break only
after callbacks and checkpoint persistence observe a converged update. Return
the actual completed count and remove extension-budget output.

- [ ] **Step 4: Verify loop and resume behavior**

Run:

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py -q
```

Expected: all layerwise runner tests pass, with torch-gated skips allowed.

### Task 3: Thread The Contract Through Runtime Persistence

**Files:**
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `tests/test_stage2_stage1_rl_alignment.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `blb_stage2_rl/runner.py`

- [ ] **Step 1: Write failing contract tests**

Assert that the active branch declares disabled entropy regularization,
`ppo.ent_coef=0.0`, natural-convergence termination metadata, a new algorithm
revision, unbounded resume arithmetic, and actual completed episodes in output.
Assert that zero is rejected for non-layerwise/non-robust legacy dispatch.

- [ ] **Step 2: Verify the old fixed-horizon contract fails**

Run:

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py tests/test_stage2_stage1_rl_alignment.py -q
```

Expected: failures reference cosine entropy metadata, v4 revision, or fixed
remaining-episode arithmetic.

- [ ] **Step 3: Implement the runtime contract**

Set the active PPO entropy coefficient to zero; replace entropy schedule
metadata with monitor-only metadata; add termination metadata; bump the
algorithm revision; preserve zero across resume; avoid rollout/warmup clamps
against zero; and report actual completed episodes.

- [ ] **Step 4: Verify contract tests pass**

Run the command from Step 2 and expect zero failures.

### Task 4: Make Zero The Formal CLI And Preset Default

**Files:**
- Modify: `tests/test_cli_parse_utils.py`
- Modify: `tests/test_stage2_persistent_launcher.py`
- Modify: `cli_parse_utils.py`
- Modify: `rl_tune.py`
- Modify: `layer_importance_evaluator.py`
- Modify: `llama_7B_LayerImportance.sh`
- Modify: `presets/mrpc-blb-stage2-rl.conf`
- Modify: `AGENTS.md`

- [ ] **Step 1: Write failing parser and launcher tests**

Add a Stage-2 episode-limit parser test accepting `0` and rejecting negatives,
and assert the formal MRPC preset uses `--stage2-search-episodes 0`.

- [ ] **Step 2: Verify current positive-only parsing fails**

Run:

```bash
python3 -m pytest tests/test_cli_parse_utils.py tests/test_stage2_persistent_launcher.py -q
```

Expected: zero parsing/preset assertions fail.

- [ ] **Step 3: Implement CLI, evaluator, shell, and preset semantics**

Add a nonnegative Stage-2 parser; allow evaluator value zero; skip PPO minimum
budget checks for zero; change shell validation/default/help text; set the MRPC
preset to zero; and record the standing natural-convergence rule in `AGENTS.md`.

- [ ] **Step 4: Verify CLI and launcher behavior**

Run the command from Step 2 plus:

```bash
bash -n llama_7B_LayerImportance.sh
```

Expected: tests pass and Bash syntax exits zero.

### Task 5: Run Regression And Algorithm Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run the focused Stage-2 suite**

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py tests/test_blb_layerwise_action.py tests/test_blb_sequential_policy.py tests/test_blb_robust_reward.py tests/test_stage2_stage1_rl_alignment.py tests/test_stage2_persistent_launcher.py tests/test_cli_parse_utils.py -q
```

Expected: zero failures; environment-dependent torch skips must be enumerated.

- [ ] **Step 2: Run syntax checks**

```bash
python3 -m py_compile blb_stage2_rl/layerwise_runner.py blb_stage2_rl/sequential_runner.py blb_stage2_rl/runner.py cli_parse_utils.py rl_tune.py layer_importance_evaluator.py
bash -n llama_7B_LayerImportance.sh
```

Expected: both commands exit zero.

- [ ] **Step 3: Re-run the synthetic factorized bandit**

Run the repository's existing production-K-order convergence test and verify it
still selects Block4 fusion `1`, real K `8`, and drives both normalized
entropies below `0.1` without entropy regularization.

- [ ] **Step 4: Audit and publish**

Review `git diff --check`, inspect the complete diff against the accepted
design, commit only scoped files, push `codex/stage2-k-convergence`, and report
the exact test evidence and remaining server-runtime validation requirement.
