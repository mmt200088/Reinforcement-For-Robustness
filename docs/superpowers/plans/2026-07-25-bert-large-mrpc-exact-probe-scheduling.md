# BERT-Large MRPC Exact Probe Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Balance four actions' exact K=5 terminal trials across four healthy
GPUs while preserving every scientific and PPO-visible result.

**Architecture:** Add an action/trial task API to the existing persistent probe
pool, an internal deferred-terminal handoff to the layerwise environment, and a
PPO-boundary-aware collector/finalizer in `train_layerwise()`. Reuse the
existing terminal-eval batch-size option for ON/OFF A/B and automatically fall
back when K is already balanced or exact deferral is unavailable.

**Tech Stack:** Python 3, PyTorch multiprocessing/CUDA, unittest, JSONL
diagnostics, Git-synchronized Linux GPU verification.

---

### Task 1: Specify Exact Task Assignment

**Files:**
- Modify: `tests/test_probe_runner_process_backend.py`
- Modify: `blb_stage2_rl/probe_runner.py`

- [ ] **Step 1: Write the failing assignment test**

Add a test for an action-major split of four actions, five trials, and four
workers. It must expect five tasks per worker and exactly the 20 identities
`(action_index, trial_index)` once each.

- [ ] **Step 2: Run the RED test on the Git-synced server**

Run:

```bash
python -m unittest \
  tests.test_probe_runner_process_backend.ProbeRunnerProcessBackendTest.test_exact_action_trial_tasks_balance_k5_across_four_workers
```

Expected: FAIL because the exact action/trial split helper does not exist.

- [ ] **Step 3: Implement the cached split helper**

Implement an immutable cached helper that round-robins the flattened
action-major sequence:

```python
for action_index in range(action_count):
    for trial_index in range(k):
        worker_index = (action_index * k + trial_index) % worker_count
        assignments[worker_index].append((action_index, trial_index))
```

- [ ] **Step 4: Run the focused test**

Expected: PASS with per-worker task counts `[5, 5, 5, 5]`.

### Task 2: Add Exact Process And Thread Worker APIs

**Files:**
- Modify: `tests/test_probe_runner_process_backend.py`
- Modify: `tests/test_blb_chain_integrity.py`
- Modify: `blb_stage2_rl/probe_runner.py`

- [ ] **Step 1: Write RED protocol tests**

Cover both process and thread backends. Tests must assert:

- each worker receives its assigned ordered task list
- a decoded action is installed once for consecutive tasks of that action
- base seeds remain attached to their original action
- returned results are grouped by action and trial index
- per-action diagnostic seeds equal `_trial_seed(base_seed, trial_index)`
- missing or duplicate task identities raise

- [ ] **Step 2: Run the RED tests on the server**

Expected: FAIL because `run_action_trial_groups()` and the child operation do
not exist.

- [ ] **Step 3: Implement the child operation and parent aggregation**

Add `run_action_trial_tasks` to `_probe_process_main()` and a public
`ProbeRunner.run_action_trial_groups()` method. Use preallocated
`action_count x K` result storage, submit replica work before running the local
worker's list, and receive replica results afterward.

- [ ] **Step 4: Preserve current APIs**

Keep `run_trials()` and `run_action_trials_once()` behavior unchanged. Expose
the grouped method through `ProbeRunnerView`.

- [ ] **Step 5: Run all probe-runner tests**

Run:

```bash
python -m unittest \
  tests.test_probe_runner_process_backend \
  tests.test_blb_chain_integrity.ProbeRunnerHelpersTest \
  tests.test_blb_chain_integrity.ProbeRunnerTwoGPUTest
```

Expected: PASS, apart from baseline environment skips already present before
the change.

### Task 3: Batch Prepared K-Trial Actions

**Files:**
- Modify: `tests/test_blb_stage2_rl_regressions.py`
- Modify: `blb_stage2_rl/env.py`

- [ ] **Step 1: Write RED environment tests**

Prepare two valid actions with distinct explicit base seeds and K=5. Assert one
grouped probe call, original per-action trial order, exact trial seeds, one
finalization per action in input order, and no reward cache.

- [ ] **Step 2: Run the RED tests on the server**

Expected: FAIL because K greater than one currently loops over actions
synchronously.

- [ ] **Step 3: Implement explicit seed capture**

Allow `prepare_action_for_terminal_probe()` to retain an optional
`probe_base_seed` in its prepared mapping without changing callers that omit
it.

- [ ] **Step 4: Implement the grouped K path**

In `evaluate_prepared_terminal_batch()`, use the grouped API only when at least
two forward-required actions have explicit seeds and the runner supports it.
Reconstruct `EpisodeMetrics` per action with the existing aggregation helper,
then call `_finish_prepared_terminal_probe()` in input order. Invalid prepared
actions remain per-action short circuits.

- [ ] **Step 5: Run environment regressions**

Run:

```bash
python -m unittest tests.test_blb_stage2_rl_regressions
```

Expected: PASS.

### Task 4: Add Layerwise Deferred Terminal Handoff

**Files:**
- Modify: `tests/test_blb_layerwise_env.py`
- Modify: `blb_stage2_rl/layerwise_env.py`

- [ ] **Step 1: Write the RED handoff test**

Assert that deferred mode completes all layer replans and resource fields,
calls `base.prepare_action_for_terminal_probe()` with the exact full vector,
resource objective, boosted overrides, and episode probe seed, does not call
`base.step()`, and exposes one prepared payload before reset.

- [ ] **Step 2: Run the RED test on the server**

Expected: FAIL because the layerwise environment has no deferred mode.

- [ ] **Step 3: Implement deferred mode**

Add an internal flag and a read-once prepared-terminal property. Leave the
default synchronous terminal branch byte-for-byte behavior-equivalent. Clear
the deferred payload on reset.

- [ ] **Step 4: Run layerwise environment tests**

Run:

```bash
python -m unittest tests.test_blb_layerwise_env
```

Expected: PASS.

### Task 5: Collect And Finalize Exact Layerwise Batches

**Files:**
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`

- [ ] **Step 1: Write RED order and boundary tests**

Use a fake deferred environment and grouped evaluator to assert:

- four actions are collected before one grouped K=5 evaluation
- records, candidate appends, callbacks, and rewards retain episode order
- rollout transitions retain step order
- a batch never crosses a PPO update boundary
- PPO receives the same 120 episodes and runs once at the same boundary
- batch size one uses the old synchronous path
- K divisible by worker count resolves to batch size one

- [ ] **Step 2: Run the RED tests on the server**

Expected: FAIL because `train_layerwise()` finalizes every episode
synchronously.

- [ ] **Step 3: Implement eligibility resolution**

Compute the minimum balancing action count as:

```python
worker_count // math.gcd(online_k, worker_count)
```

Cap it by the requested terminal-eval batch size. Return one when eligibility
conditions fail.

- [ ] **Step 4: Implement draft collection and ordered finalization**

Capture every variable needed by the existing finalization block in an
episode-owned draft. Collect only up to the next update/run boundary, evaluate
the prepared payloads once, then execute reward redistribution, candidate
promotion, callbacks, PPO, and checkpoint callbacks in original order.

- [ ] **Step 5: Propagate the existing runtime setting**

When constructing the layerwise `SequentialTrainConfig`, copy
`train_cfg.terminal_eval_batch_size`. Record requested/effective batch sizes in
runtime diagnostics only, not in the algorithm contract or candidate identity.

- [ ] **Step 6: Run layerwise tests**

Run:

```bash
python -m unittest tests.test_blb_layerwise_runner
```

Expected: PASS with no new failures.

### Task 6: Server Contract And A/B Gate

**Files:**
- Reuse: `scripts/stage2_ngpu_ab_compare.py`
- Create after the run:
  `experiments/server_command_runs/stage2_exact_probe_batch_<timestamp>/`

- [ ] **Step 1: Push the implementation and synchronize the server**

Verify local, `origin`, and server HEAD/tree IDs before any run.

- [ ] **Step 2: Run focused and broad Stage-2 tests**

Run the Task 1-5 suites followed by the repository's Stage-2 contract gate.
Compare any broad-suite failures against the `e1a1bba9` baseline.

- [ ] **Step 3: Run matched profile-off arms**

Run at least 240 fresh episodes per arm with identical BERT-large MRPC source,
seed, K=5, four reward devices, probe size 256, PPO interval 120, and all
algorithm settings. Set only:

```text
control:   --blb-v3-terminal-eval-batch-size 1
candidate: --blb-v3-terminal-eval-batch-size 4
```

- [ ] **Step 4: Compare exact outputs and wall throughput**

Run:

```bash
python scripts/stage2_ngpu_ab_compare.py \
  --one CONTROL_RUN \
  --many CANDIDATE_RUN \
  --one-ppo CONTROL_PPO \
  --many-ppo CANDIDATE_PPO \
  --one-wall CONTROL_WALL \
  --many-wall CANDIDATE_WALL \
  --atol 0 \
  --require-equal \
  --min-speedup 1.2 \
  --require-speedup
```

Expected: quality/effect equality PASS, PPO equality PASS, and speedup at least
`1.2x`. Timing/device diagnostic differences are expected and separately
reported.

- [ ] **Step 5: Accept or revert**

Keep batch size four as the current four-GPU production setting only if every
gate passes. Otherwise set the effective size to one and remove the rejected
production path.

### Task 7: Aggregate Deployment

**Files:**
- Modify after verified evidence: `AGENTS.md`
- Add: compact A/B evidence under `experiments/server_command_runs/`

- [ ] **Step 1: Refresh all agent heads**

Fetch/prune origin and inspect every recently updated agent branch. Integrate
completed, non-superseded changes without resetting unrelated work.

- [ ] **Step 2: Commit and push the aggregate**

Commit source, tests, design, plan, and compact evidence. Push the aggregate
branch and update the shared branch only after review.

- [ ] **Step 3: Synchronize through Git**

Have the server fetch and check out the aggregate commit. Do not patch server
source.

- [ ] **Step 4: Verify parity**

Require identical local, origin, and server commit IDs and source-tree IDs.
Confirm the stopped episode-10,920 checkpoint and all original run artifacts
remain intact before any production resume.
