# Stage-2 Calibrated Action Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make production RL, Paean final evaluation, action-grid decoding, and GLUE submission decode Stage-2 actions with one static-skeleton-calibrated SF context.

**Architecture:** `blb_stage2_rl.baseline_bootstrap` owns construction of a `Stage2CalibratedActionContext` from the Rescale Optimizer archive and Stage-1 degree vectors. Paean builds it once after resolving Stage-1 and passes its `max_sfs` explicitly through action-grid, candidate evaluation, slot export, and GLUE generation. Low-level action-grid utilities retain an explicit legacy fallback only when callers omit `max_sfs`.

**Tech Stack:** Python 3, dataclasses, NumPy, unittest, PyTorch server integration, Rescale Optimizer static-skeleton archive.

---

### Task 1: Shared calibrated context

**Files:**
- Modify: `blb_stage2_rl/baseline_bootstrap.py`
- Test: `tests/test_blb_baseline_bootstrap.py`

- [ ] **Step 1: Write the failing context test**

Add a test importing `load_calibrated_stage2_action_context`, constructing an
MRPC all-GELU4/all-Softmax6 context, and asserting its baseline action decodes
Block3 to `x_fresh=31`, `inv_2n=15`, and six square-rescale SFs of `35`.

- [ ] **Step 2: Run the red test**

Run:

```bash
python3 -m unittest tests.test_blb_baseline_bootstrap.BLBBaselineBootstrapTests.test_calibrated_action_context_matches_mrpc_block3_static_baseline -v
```

Expected: import/error failure because the shared context loader does not exist.

- [ ] **Step 3: Implement the context**

Add `Stage2CalibratedActionContext` and
`load_calibrated_stage2_action_context(...)`. Reuse the two existing baseline
functions with `snap_sf_to_noise_table=False`; include archive SHA-256 and the
degree vectors in provenance.

- [ ] **Step 4: Run the green test and baseline suite**

```bash
python3 -m unittest tests.test_blb_baseline_bootstrap -v
```

Expected: all tests pass.

### Task 2: Explicit action-grid context injection

**Files:**
- Modify: `Paean/action_grid.py`
- Test: `tests/test_paean_action_grid.py`

- [ ] **Step 1: Write the failing injection test**

Pass a sentinel `max_sfs` to `load_action_grid_config` and
`build_action_candidates`; assert slot decoding and selector application receive
the sentinel while the profile-only loader is never called.

- [ ] **Step 2: Run the red test**

```bash
python3 -m unittest tests.test_paean_action_grid -v
```

Expected: failure because those APIs do not accept `max_sfs`.

- [ ] **Step 3: Implement explicit propagation**

Add optional `max_sfs`, `gelu_degree`, and `attn_degree` inputs to candidate
builders. Propagate them through nested batch manifests and slot-form loading;
use the existing profile cache only when `max_sfs is None`.

- [ ] **Step 4: Run the green test**

```bash
python3 -m unittest tests.test_paean_action_grid -v
```

Expected: all tests pass, including legacy fallback coverage.

### Task 3: Paean and GLUE use the same context

**Files:**
- Modify: `Paean/blb_action_eval.py`
- Modify: `generate_glue_submission.py`
- Test: `tests/test_paean_blb_action_eval_static.py`
- Test: `tests/test_blb_stage2_rl_regressions.py`
- Test: `tests/test_blb_glue_boost_install.py`

- [ ] **Step 1: Write failing propagation guards**

Add guards requiring Paean to build the calibrated context after Stage-1
resolution, pass its table into every candidate and slot-export path, and pass
the context to GLUE. Update candidate regression tests to supply explicit
`max_sfs`. Update the GLUE boost test to build and use the shared context.

- [ ] **Step 2: Run the red tests**

```bash
python3 -m unittest tests.test_paean_blb_action_eval_static tests.test_blb_stage2_rl_regressions -v
```

Expected: propagation guards fail against the profile-only implementation.

- [ ] **Step 3: Implement Paean propagation**

Build one context in `BLBActionFinalEvaluationModule.run`, pass its `max_sfs`
to `build_action_candidates`, selected/random candidate evaluation, slot export,
and `_maybe_run_glue_submission`, and remove the profile-only Paean cache.

- [ ] **Step 4: Implement GLUE propagation**

Allow `generate_blb_glue_submission` to accept a validated context or build one
from the same degrees and repository Rescale Optimizer root. Pass its table into
action-grid loading and `_process_blb_task`; remove the internal generic reload.
Return context provenance in the generation summary.

- [ ] **Step 5: Run focused tests**

```bash
python3 -m unittest tests.test_blb_baseline_bootstrap tests.test_paean_action_grid tests.test_paean_blb_action_eval_static tests.test_blb_stage2_rl_regressions -v
```

Expected: all available local tests pass.

### Task 4: Server parity gate and delivery

**Files:**
- Modify: `tests/test_blb_glue_boost_install.py`
- Create: `experiments/server_command_runs/stage2_calibrated_action_context_verify_20260721/` compact evidence

- [ ] **Step 1: Commit and push the verified source snapshot**

Commit only the design, implementation, and tests, then push to
`origin/jk_standard_rl` after checking that the remote parent has not advanced.

- [ ] **Step 2: Run server tests from the exact commit**

Run the focused baseline, Paean, action-grid, GLUE boost, and Block3 runtime
tests on the five-GPU server. Record source commit and clean status.

- [ ] **Step 3: Compare configuration fingerprints**

Replay one MRPC candidate through production layerwise decode, Paean, and GLUE
decode. Assert every installed per-layer/block configuration matches, including
Block3 baseline SFs and selected K.

- [ ] **Step 4: Archive and push compact evidence**

Pull JSON/log summaries only, commit them to `jk_standard_rl`, and verify local
HEAD equals `origin/jk_standard_rl` with a clean worktree.
