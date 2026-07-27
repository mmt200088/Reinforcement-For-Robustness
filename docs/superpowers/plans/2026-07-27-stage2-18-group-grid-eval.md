# Stage-2 18-Group Grid Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the requested 3 fusion profiles by 6 truncation-K profiles on BERT-base MRPC validation_full with exact installed-configuration audits.

**Architecture:** Add one experiment driver that reuses the production Stage-2 layerwise environment, canonical optimizer materialization, installed-model inference, and shared repeat-metric helpers. Keep group construction and action rewriting torch-free so the 18-group contract can be tested locally. Run one experiment seed per GPU and aggregate the five paired seeds into JSON and HTML.

**Tech Stack:** Python, NumPy, unittest, PyTorch/Transformers on the GPU server, existing `blb_stage2_rl` runtime.

---

### Task 1: Lock the requested grid contract

**Files:**
- Create: `tests/test_stage2_precision_stability_grid_eval.py`
- Create: `scripts/run_stage2_precision_stability_grid_eval.py`

- [x] Add failing tests for the exact 3x6 group set, K6/K7 policy indices, all 60 effective K positions, and policy/control path separation.
- [x] Run `python3 -m unittest -v tests.test_stage2_precision_stability_grid_eval` and verify it fails because the driver contract is absent.
- [x] Implement only the torch-free group/action helpers.
- [x] Re-run the focused test and require all tests to pass.

### Task 2: Reuse the production runtime chain

**Files:**
- Modify: `scripts/run_stage2_precision_stability_grid_eval.py`

- [x] Build BERT-base MRPC validation_full runtime through the existing evaluator and `BLBStage2Env`.
- [x] Run `0/0/0` controls from the calibrated baseline vector with only K coordinates changed.
- [x] Run `1/0/1` and `1/1/1` through `BLBStage2LayerwiseEnv`, which supplies the committed boosted/fused options.
- [x] Require real forward, valid optimizer output, post-replan installation, exact fusion totals, exact per-layer B1-B5 K values, and non-empty installed-config fingerprints.
- [x] Package each five-trial result with `pack_repeat_evaluation`.

### Task 3: Aggregate paired evidence

**Files:**
- Modify: `scripts/run_stage2_precision_stability_grid_eval.py`

- [x] Aggregate all 25 trials per group with `pack_repeat_evaluation`.
- [x] Verify paired trial seeds across all 18 groups and stable installed-config fingerprints across the five experiment seeds.
- [x] Produce raw JSON plus an HTML report containing the 18-group metrics, per-seed mean/std, precision/stability gates, fusion totals, and exact K profiles.

### Task 4: Verify and deploy

**Files:**
- Test: `tests/test_stage2_precision_stability_grid_eval.py`
- Test: existing Stage-2 K/materialization suites

- [x] Run Python compilation and focused torch-free tests locally.
- [ ] Commit and push the experiment branch, then fast-forward the remote aggregate/main only after checks pass.
- [ ] Sync an exact Git commit to a separate server source path and run torch-backed K6 materialization tests.
- [ ] Launch five paired seed workers across five GPUs, aggregate results, pull artifacts locally, and place the final HTML on the Desktop.
