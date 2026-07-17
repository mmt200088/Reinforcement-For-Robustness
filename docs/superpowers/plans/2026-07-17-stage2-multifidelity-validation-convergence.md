# Stage-2 Multi-Fidelity Validation And Practical Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage-2 PPO learn on the fixed 256-example probe while allowing only independent `validation_full` evidence into the authoritative frontier, and stop naturally when the authoritative optimum is stable and freshly revalidated.

**Architecture:** Keep the existing layerwise PPO and reward path unchanged for F1. Add fidelity-specific candidate contexts, a second probe environment whose batches cover all of `validation_full`, and an F4-only promotion/frontier path. Convergence uses a 100-finite-update frontier/action plateau followed by a fresh 25-trial F4 revalidation; episode count and entropy remain diagnostic.

**Tech Stack:** Python 3, PyTorch, existing BLB `ProbeRunner`, append-only JSONL `CandidateStore`, `unittest`/`pytest`, Paean final evaluation.

---

### Task 1: Separate Probe And Full-Validation Evidence

**Files:**
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Test: `tests/test_blb_layerwise_runner.py`

- [ ] **Step 1: Write failing tests for fidelity-specific identities**

Add tests which append five F1 trials and 25 F4 trials for the same action, then assert that the online assessment sees only five trials, promotion sees exactly 25 F4 trials, and the candidate keys differ because their identity contexts carry `fidelity="F1"` and `fidelity="F4"`.

- [ ] **Step 2: Run the focused tests and verify the mixed-evidence behavior fails**

Run: `python3 -m pytest -q tests/test_blb_layerwise_runner.py -k 'fidelity or promotion'`

Expected: FAIL because current promotion reads the base identity and pools the first five probe trials into its 25-trial prefix.

- [ ] **Step 3: Implement fidelity context derivation**

Add a small helper:

```python
def evidence_identity_context(identity_context, fidelity):
    context = dict(identity_context)
    context["fidelity"] = normalize_fidelity(fidelity)
    return context
```

Use the F1 context for online groups and pooled online assessments. Use the F4 context for promotion groups, promotion status records, restore, accepted-candidate keys, and strict-frontier identity.

- [ ] **Step 4: Run focused tests**

Run: `python3 -m pytest -q tests/test_blb_layerwise_runner.py -k 'fidelity or promotion or restore'`

Expected: PASS.

### Task 2: Build An Authoritative Full-Validation Probe Path

**Files:**
- Modify: `blb_stage2_rl/runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Test: `tests/test_blb_stage2_rl_regressions.py`
- Test: `tests/test_blb_layerwise_runner.py`
- Test: `tests/test_blb_robust_baseline.py`

- [ ] **Step 1: Write failing full-loader and promotion-environment tests**

Cover these contracts:

```python
assert len(full_batches) == math.ceil(len(validation_full) / batch_size)
assert full_example_count == len(validation_full)
assert promotion_base_env is not online_base_env
assert promotion_base_env.statistical_reference is full_reference
assert online_base_env.statistical_reference is probe_reference
```

The promotion fake must expose a distinct `evaluate_prepared_terminal_batch` and prove it receives `validation_required=True` and all 25 requested F4 trials.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest -q tests/test_blb_stage2_rl_regressions.py tests/test_blb_layerwise_runner.py tests/test_blb_robust_baseline.py -k 'validation_full or promotion_env or robust_baseline'`

Expected: FAIL because only `_build_probe_batches` exists and promotion calls `env.base`.

- [ ] **Step 3: Add the uncapped validation loader**

Implement `BLBStage2RLRunner._build_validation_full_batches(ev)` using the evaluator's exact `validation_full` dataset, existing collator, and batch size. Do not call `_get_stability_probe` and do not apply `_effective_probe_batch_count`. Raise a clear error if `validation_full` is absent or empty.

- [ ] **Step 4: Construct the dedicated promotion environment**

In the robust layerwise branch, create a shallow evaluation clone that reuses the canonical primary bridge but owns full-validation batches, an independent env config/cache, and a dedicated multi-GPU `ProbeRunner`. Give it copied baseline/reward objects. Collect a separate 5x5 full-validation baseline reference and install it only on this environment.

- [ ] **Step 5: Route promotion and restore through F4**

Add `promotion_base_env` and `promotion_statistical_reference` parameters to `train_layerwise`, `promote_candidate_if_eligible`, and `restore_promoted_candidates`. Clear the online persistent runner before and after F4 evaluation, and reset its installed-action hash so the next F1 action cannot be skipped incorrectly.

- [ ] **Step 6: Persist both evidence contracts**

Add probe/full split names, example counts, trial counts, and both baseline summaries to the algorithm contract, run context, manifest, and structured baseline payload. Bump the revision to `factorized_slot_credit_multifidelity_convergence_v8` so old pooled-evidence checkpoints fail closed.

- [ ] **Step 7: Run focused tests**

Run: `python3 -m pytest -q tests/test_blb_stage2_rl_regressions.py tests/test_blb_layerwise_runner.py tests/test_blb_robust_baseline.py`

Expected: PASS.

### Task 3: Implement Objective-Driven Plateau Convergence

**Files:**
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `blb_stage2_rl/runner.py`
- Test: `tests/test_blb_layerwise_runner.py`

- [ ] **Step 1: Write boundary tests first**

Test all cases:

```python
# 99 stable updates: not plateau-ready
# 100 stable updates: plateau-ready but not converged before strict revalidation
# a fresh 25-trial F4 revalidation pass confirms convergence
# a failed revalidation removes the selected candidate and resumes search
# a frontier improvement or selected identity change resets its counter
# non-finite PPO updates do not advance patience
# completed episode count never forces or prevents convergence
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest -q tests/test_blb_layerwise_runner.py -k convergence`

Expected: FAIL because the tracker still applies episode limits and does not
require fresh strict revalidation.

- [ ] **Step 3: Parameterize the tracker and loop**

Keep one active natural-convergence setting in the long-run train configs:

```python
convergence_patience_updates = 100
```

Expose plateau readiness separately from convergence. At the plateau boundary,
collect a new F4 trial group under a distinct evidence identity and assess it at
the final 0.95 gate. Stop the unbounded loop only after that revalidation passes;
bounded smoke runs continue to obey only their explicit episode limit.

- [ ] **Step 4: Persist honest termination state**

Write `running`, `plateau_revalidation_failed`, `converged`, or
`bounded_budget_exhausted` consistently to checkpoint, manifest, status, PPO
diagnostics, summary, and algorithm contract. Keep B4/K entropy monitor-only.

- [ ] **Step 5: Make strict selection deterministic and objective-aligned**

Rank feasible candidates by maximum cost, then worst-first lexicographic vectors
covering all six constraint probabilities and all six normalized safety margins,
then full-action-vector lexicographic order, and only then candidate identity.
Keep equal-cost candidates eligible for F4 and add focused equal-cost tests.

- [ ] **Step 6: Run convergence and runner tests**

Run: `python3 -m pytest -q tests/test_blb_layerwise_runner.py tests/test_sequential_smoke.py -k 'layerwise or convergence or termination'`

Expected: PASS.

### Task 4: Verify The Full Change Set

**Files:**
- Check: all modified production and test files

- [ ] **Step 1: Compile modified modules**

Run: `python3 -m py_compile blb_stage2_rl/runner.py blb_stage2_rl/sequential_runner.py blb_stage2_rl/layerwise_runner.py`

Expected: exit 0.

- [ ] **Step 2: Run all Stage-2 focused tests**

Run: `python3 -m pytest -q tests/test_blb_candidate_store_identity.py tests/test_blb_layerwise_runner.py tests/test_blb_robust_baseline.py tests/test_blb_stage2_rl_regressions.py tests/test_sequential_smoke.py`

Expected: all non-torch tests pass; environment-declared torch skips remain skips.

- [ ] **Step 3: Inspect the diff**

Run: `git diff --check` and `git diff --stat`

Expected: no whitespace errors and only scoped Stage-2/docs/tests changes.

### Task 5: Finish The Current Run With Scheme 3

**Files:**
- Read: server frozen best/checkpoint/baseline artifacts
- Generate: `reports/html_reports/20260717_stage2_current_run_validation_full_final.html`

- [ ] **Step 1: Evaluate baseline and frozen best on `validation_full`**

Use `Paean/run_final_eval.sh` with one `paean_action_batch_v1` manifest, Stage-1 GELU all 4 / Softmax all 6, exact fusion groups and boosted overrides, 25 repeats, in-process real Rescale optimizer, no random controls, and no GLUE submission.

- [ ] **Step 2: Validate result provenance**

Require the Paean artifact to state `validation_full`, 25 trials for both selected actions, baseline fusion count 0/K=13, and frozen-best action identity equal to the frozen training artifact.

- [ ] **Step 3: Generate and visually inspect the HTML**

Include training reward/entropy/cost curves, the probe-derived selection label, full-validation mean/std comparison for loss/accuracy/F1, percentage deltas, all 12 layer decisions for Block-4 fusion and Block-1/2/3/4/5 K, and the termination caveat for the pre-change run.

- [ ] **Step 4: Copy the report locally and report the path**

Copy the complete compact result bundle into the repository report directory and provide a clickable absolute path.
