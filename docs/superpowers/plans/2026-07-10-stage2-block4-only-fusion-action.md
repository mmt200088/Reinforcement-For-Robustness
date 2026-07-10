# Stage-2 Block4-Only Fusion Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Block4 the only policy-selectable fusion count while fixing every Block2/Block5 occurrence at `fusion_count=1`, then evaluate two MRPC BERT-base GELU ladders and complete/audit five remaining profile maps.

**Architecture:** Keep the existing two-slot `(fusion, K)` policy and sequential horizon. Add a policy-local fusion domain to `FusionStepSpec`, resolve it once to the real map option, and use that resolved option throughout expansion, precision boost, replan, install, reward, and logging. Fixed-action JSON continues to store real map option IDs and passes an explicit map-option override into the same sequential-environment body, allowing the fusion-zero control without reopening that action to RL.

**Tech Stack:** Python, NumPy, PyTorch server tests, Rescale Optimizer, existing Stage-2 RL-path evaluator, shell orchestration on the five-GPU server.

---

### Task 1: Lock the action-domain semantics

**Files:**
- Modify: `tests/test_blb_fusion_count_map.py`
- Modify: `blb_stage2_rl/action_space.py`

- [ ] **Step 1: Write failing schedule tests**

Add tests asserting that Block2/5 expose one policy-local option mapped to the
real map option with `fusion_count=1`, Block4 retains all real options, and K
still exposes `LEVELS_K` choices. Add malformed-map tests for missing and
duplicate `fusion_count=1` options.

```python
def test_block2_and_block5_are_fixed_to_fusion_one(self):
    for spec in self.sched:
        if spec.block_idx in (2, 5):
            self.assertEqual(spec.fusion_num_options, 1)
            map_id = _aspace.resolve_fusion_map_option_id(spec, 0)
            option = self.m.options(spec.graph_key_suffix)[map_id]
            self.assertEqual(option.fusion_count, 1)

def test_block4_keeps_full_fusion_domain(self):
    spec = next(s for s in self.sched if s.block_idx == 4)
    self.assertEqual(spec.fusion_num_options, self.m.num_options("block4"))
    self.assertEqual(spec.map_option_ids, tuple(range(self.m.num_options("block4"))))
```

- [ ] **Step 2: Run the tests on the server and verify RED**

Run:

```bash
python3 -m unittest -v tests.test_blb_fusion_count_map.FusionScheduleTest
```

Expected: FAIL because `FusionStepSpec.map_option_ids` and
`resolve_fusion_map_option_id` do not exist and Block2/5 still expose two
choices.

- [ ] **Step 3: Implement the minimal schedule mapping**

In `FusionStepSpec`, add `map_option_ids: Tuple[int, ...]`. Build it as:

```python
if s.block_idx in (2, 5):
    matching = tuple(
        int(option.option_id)
        for option in fusion_map.options(gk)
        if int(option.fusion_count) == 1
    )
    if len(matching) != 1:
        raise ValueError(
            f"{gk}: block{s.block_idx} requires exactly one fusion_count=1 "
            f"option, found {list(matching)}"
        )
    map_option_ids = matching
else:
    map_option_ids = tuple(int(option.option_id) for option in fusion_map.options(gk))
```

Add exact-range conversion helpers:

```python
def resolve_fusion_map_option_id(spec, policy_option_index):
    idx = int(policy_option_index)
    if idx < 0 or idx >= len(spec.map_option_ids):
        raise ValueError(f"policy fusion option {idx} out of range")
    return int(spec.map_option_ids[idx])

def resolve_fusion_policy_option_index(spec, map_option_id):
    option_id = int(map_option_id)
    try:
        return int(spec.map_option_ids.index(option_id))
    except ValueError as exc:
        raise ValueError(f"map fusion option {option_id} is not selectable") from exc
```

Make `expand_fusion_step_action` resolve the policy-local index before calling
`fusion_map.expand`.

- [ ] **Step 4: Run the schedule tests and verify GREEN**

Run the same unittest command. Expected: all `FusionScheduleTest` tests pass.

- [ ] **Step 5: Commit the action-domain change**

```bash
git add blb_stage2_rl/action_space.py tests/test_blb_fusion_count_map.py
git commit -m "Fix Block2 and Block5 fusion at one"
```

### Task 2: Use the resolved option throughout training and evaluation

**Files:**
- Modify: `blb_stage2_rl/sequential_env.py`
- Modify: `scripts/run_fusion_count_action_eval_rlpath.py`
- Modify: `tests/test_blb_stage2_rl_regressions.py`
- Modify: `tests/test_blb_fusion_single_path_guards.py`

- [ ] **Step 1: Write failing integration tests**

Tests must assert that a Block2 policy action `[0, k]` expands and logs the real
map option `1`, carries the boosted explicit field values, records
`fusion_count=1`, and leaves the selected K unchanged. Add a fixed-action test
showing that JSON `option_id=0` reaches the same `evaluate_step` body through an
explicit map-option override for the control group.

- [ ] **Step 2: Run focused tests on the server and verify RED**

```bash
python3 -m unittest -v \
  tests.test_blb_stage2_rl_regressions \
  tests.test_blb_fusion_single_path_guards
```

Expected: FAIL because current code indexes `_opts` directly with the
policy-local action and the fixed evaluator passes map IDs directly.

- [ ] **Step 3: Apply the canonical resolution once per step**

In `BLBStage2SequentialEnv.evaluate_step`, resolve `action[0]` to
`map_option_id` before expansion. Use `map_option_id` for option lookup,
fusion-count bookkeeping, precision boost, and persisted `option_id`; retain
the policy-local index separately as `policy_option_index` for PPO diagnostics.

In `_run_group`, pass the fixed config's real map option ID as
`map_option_id_override` while supplying a harmless local placeholder action.
Persist both IDs in the step records.

- [ ] **Step 4: Run focused tests and the existing fusion contract suite**

```bash
python3 -m unittest -v \
  tests.test_blb_fusion_count_map \
  tests.test_blb_stage2_rl_regressions \
  tests.test_blb_fusion_single_path_guards \
  tests.test_blb_fusion_fixed_action \
  tests.test_blb_final_eval_fusion_fixed_action
```

Expected: all tests pass with no skips caused by missing server dependencies.

- [ ] **Step 5: Commit the canonical-path integration**

```bash
git add blb_stage2_rl/sequential_env.py scripts/run_fusion_count_action_eval_rlpath.py \
  tests/test_blb_stage2_rl_regressions.py tests/test_blb_fusion_single_path_guards.py
git commit -m "Resolve fixed fusion options through one path"
```

### Task 3: Add a reusable fusion-count upper-bound audit

**Files:**
- Create: `scripts/audit_fusion_count_maps.py`
- Create: `tests/test_audit_fusion_count_maps.py`

- [ ] **Step 1: Write failing audit tests**

Cover clean `[0,1]`, anomalous `[0,1,2]`, missing graph files, and JSON output
that identifies profile, graph key, option ID, fusion count, and slots.

- [ ] **Step 2: Verify RED on the server**

```bash
python3 -m unittest -v tests.test_audit_fusion_count_maps
```

Expected: import failure because the audit module does not exist.

- [ ] **Step 3: Implement the torch-free auditor**

The CLI accepts repeated `--profile-dir`, `--max-allowed 1`, and
`--output-json`. It exits non-zero when required Block2/4/5 graphs are missing
or any option exceeds the limit. It reads only `block*.json` map files and does
not mutate or promote maps.

- [ ] **Step 4: Verify GREEN and commit**

```bash
python3 -m unittest -v tests.test_audit_fusion_count_maps
git add scripts/audit_fusion_count_maps.py tests/test_audit_fusion_count_maps.py
git commit -m "Audit fusion map count bounds"
```

### Task 4: Run the MRPC BERT-base two-pair experiment

**Files:**
- Server artifacts: `experiments/server_command_runs/stage2_b2b5_fixed1_mrpc_<timestamp>/`
- Report: `reports/html_reports/<date>_stage2_b2b5_fixed1_mrpc.html`

- [ ] **Step 1: Push the verified source branch and create an exact server snapshot**

Push the feature branch, fetch that exact commit on the server, and record its
full SHA in the run manifest. Do not use the shared server checkout.

- [ ] **Step 2: Generate pair-specific fixed-action configs**

Use `scripts/report_fusion_count_map.py` for each GELU ladder and retain only:

- `all_fusion0`;
- `block2_block5_all_layers_fusionmax`.

Verify both use `K=13`, Block4 fusion 0, and differ only in Block2/5 fusion.

- [ ] **Step 3: Run both GELU pairs concurrently on separate GPUs**

Invoke `scripts/run_fusion_count_action_eval_rlpath.py` with `--repeat 5`,
`--probe-size 408`, and the appropriate `--stage1-gelu`. Run the Stage-1-best
pair on one GPU and the GELU4 pair on another. Both must pass through
`SequentialEnv.evaluate_step -> commit_step -> BLBStage2Env.step`.

- [ ] **Step 4: Gate the results**

Require all steps valid, effective Block2/5 fusion count 1 in the treatment,
effective Block4 fusion 0 in both groups, K distribution `{13: 47}`, boosted
install evidence for every selected Block2/5 option, and five terminal trials.

- [ ] **Step 5: Render one comparison report**

Include per-pair means/stds, absolute and percentage deltas, per-layer
fusion/K tables, replan/install evidence, source SHA, seeds, and protocol.

### Task 5: Complete and audit the other five profile maps

**Files:**
- Server staging artifacts: `experiments/server_command_runs/stage2_other5_maps_<timestamp>/`
- Canonical map targets: `blb_stage2_rl/fusion_maps/{rte,sst2,mrpc_large,rte_large,sst2_large}/`

- [ ] **Step 1: Build into profile-local staging directories**

Run `scripts/blb_build_fusion_count_map.py` for each profile with the correct
12/24-layer setting and current Stage-1 degree profile. Apply both precision
boost phases and run existing golden/replan/install verification.

- [ ] **Step 2: Audit before promotion**

Run `scripts/audit_fusion_count_maps.py --max-allowed 1` against all five
staging directories. If any Block2/4/5 graph has `fusion_count > 1`, preserve
the staged map and diagnostics, do not promote it, and report the exact graph
and fused rescale set.

- [ ] **Step 3: Promote only clean profiles**

Copy clean staged maps to canonical profile directories, rerun
`FusionCountMap.load`, option0/baseline, K-independence, precision-boost, and
runtime install gates.

- [ ] **Step 4: Pull compact artifacts and commit locally**

Exclude checkpoints/caches. Commit maps, audit JSON, build summaries, and HTML
reports locally, then rebase/cherry-pick onto the latest remote branch without
touching unrelated changes and push.

### Task 6: Final verification and handoff

**Files:**
- Verify all files changed by Tasks 1-5.

- [ ] **Step 1: Run fresh server verification**

```bash
python3 -m unittest -v \
  tests.test_blb_fusion_count_map \
  tests.test_blb_stage2_rl_regressions \
  tests.test_blb_fusion_single_path_guards \
  tests.test_blb_fusion_fixed_action \
  tests.test_blb_final_eval_fusion_fixed_action \
  tests.test_audit_fusion_count_maps
```

- [ ] **Step 2: Perform local static verification only**

```bash
git diff --check
git status --short
```

- [ ] **Step 3: Report outcomes**

Provide the source commit, report paths, Pair A/B metrics and deltas, effective
fusion/K tables, and a profile-by-profile map audit. Explicitly flag every
missing graph or `fusion_count > 1` anomaly.
