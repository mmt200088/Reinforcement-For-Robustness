# Stage-2 Small GTrXL Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 128-dimensional, two-layer shared GTrXL as the default for fresh Stage-2 runs while preserving the existing large network and historical checkpoint behavior exactly.

**Architecture:** Extend the existing policy-network variant registry with an immutable architecture contract. The policy configuration resolves dimensions from the selected variant before module construction; legacy checkpoints remain bound to the explicit large variant. Launcher and persistence plumbing continue to use the existing variant field.

**Tech Stack:** Python 3, PyTorch, dataclasses, unittest, Bash launcher, Git tags.

---

### Task 1: Lock Variant And Architecture Contracts

**Files:**
- Modify: `tests/test_blb_policy_network_variants.py`
- Modify: `tests/test_blb_layerwise_policy.py`
- Modify: `tests/test_stage2_persistent_launcher.py`

- [ ] Add failing tests asserting that the fresh default is
  `shared_gtrxl_small_v1`, its architecture is `128/4/2/256`, and its
  production parameter count is 680,221.
- [ ] Add regression assertions that explicit `shared_gtrxl_v1` remains
  `256/8/4/512` with 5,330,461 parameters.
- [ ] Add checkpoint tests proving legacy metadata maps only to the explicit
  large variant and that small/large cross-resume is rejected.
- [ ] Run:
  `python -m unittest tests.test_blb_policy_network_variants tests.test_blb_layerwise_policy tests.test_stage2_persistent_launcher`
  and confirm the new assertions fail before implementation.

### Task 2: Implement The Small Variant

**Files:**
- Modify: `blb_stage2_rl/network_variants.py`
- Modify: `blb_stage2_rl/sequential_policy.py`

- [ ] Add an immutable architecture payload to each registered variant.
- [ ] Register `shared_gtrxl_small_v1` with `d_model=128`, `n_heads=4`,
  `n_layers=2`, and `d_ff=256`; make it the fresh-run default.
- [ ] Keep `shared_gtrxl_v1` and both separate-critic variants on the existing
  `256/8/4/512` actor architecture.
- [ ] Resolve architecture dimensions deterministically in
  `SequentialPolicyConfig.__post_init__` and include them in the small
  algorithm contract.
- [ ] Preserve legacy checkpoint inference as `shared_gtrxl_v1`, independent
  of the new fresh-run default.
- [ ] Run the focused tests and confirm all variant and parameter assertions
  pass.

### Task 3: Update Launcher And Persistence Expectations

**Files:**
- Modify: `llama_7B_LayerImportance.sh`
- Modify: `tests/test_stage2_persistent_launcher.py`
- Modify: `docs/stage2_policy_network_ablation_v10.md`

- [ ] Change the launcher's default variant to `shared_gtrxl_small_v1` while
  retaining explicit `shared_gtrxl_v1` support.
- [ ] Update help text and examples to distinguish fresh small runs from exact
  large-network rollback runs.
- [ ] Document both immutable Git tags and the CLI needed to select every
  retained arm.
- [ ] Run `bash -n llama_7B_LayerImportance.sh` and the launcher tests.

### Task 4: Verify PPO And Backward Compatibility

**Files:**
- Test: `tests/test_blb_layerwise_policy.py`
- Test: `tests/test_blb_layerwise_runner.py`
- Test: `tests/test_sequential_smoke.py`
- Test: `tests/test_blb_diagnostics_static.py`
- Test: `tests/test_rl_data_points.py`

- [ ] Run all policy variants through a real Torch factorized PPO update.
- [ ] Verify large-variant actor logits remain bit-identical under the fixed
  historical seed.
- [ ] Verify the small contract, manifest, diagnostics, and checkpoint all
  report the same architecture and variant.
- [ ] Run the combined Torch and static suites, `python -m py_compile` on all
  changed Python modules, `bash -n`, and `git diff --check`.

### Task 5: Commit And Publish

**Files:**
- Commit all files listed above.

- [ ] Confirm the worktree contains no unrelated modifications.
- [ ] Commit the small-network implementation on
  `codex/stage2-network-ablation-v10`.
- [ ] Push the branch and verify both rollback tags resolve to the intended
  commits on the remote.
- [ ] Report that convergence remains unproven until a controlled long server
  run is completed.
