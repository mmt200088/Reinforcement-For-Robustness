# Stage-2 Equivalence-Aware Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop Stage-2 PPO on a stable robust-feasible optimum and exact selected configuration without forcing Block4/K policy entropy below an arbitrary threshold.

**Architecture:** Extend the existing `LayerwiseConvergenceTracker` with a persisted strict-best candidate identity and an independent action-stability counter. Keep entropy collection unchanged as diagnostics, and version the algorithm contract so entropy-gated checkpoints cannot silently resume under new semantics.

**Tech Stack:** Python 3, `unittest`, existing Stage-2 candidate store and layerwise PPO modules.

---

### Task 1: Specify Pure Convergence Rules

**Files:**
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`

- [x] Add failing tests showing high/absent entropy does not block convergence when cost and exact winner are stable.
- [x] Add a failing test showing a same-cost strict-winner change resets action stability.
- [x] Add failing state round-trip and deterministic tie-break tests.
- [x] Run `python3 -m unittest tests.test_blb_layerwise_runner.LayerwiseRunnerPureRulesTests -v` and confirm failures are caused by the missing identity contract.
- [x] Implement the minimal tracker and strict snapshot changes.
- [x] Re-run the pure-rule tests and require zero failures.

### Task 2: Integrate Training, Resume, And Persistence

**Files:**
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`

- [x] Add failing unbounded-run and resume tests for selected-action stability.
- [x] Pass strict-best identity into each tracker update and persist both counters.
- [x] Reconcile restored promoted candidates before honoring a converged checkpoint.
- [x] Bump the algorithm revision and replace entropy threshold contract fields with selected-action stability fields.
- [x] Run the focused layerwise runner suite and require zero failures.

### Task 3: Verify And Report

**Files:**
- Modify only if an existing assertion requires the new public contract.

- [x] Run `python3 -m py_compile blb_stage2_rl/layerwise_runner.py blb_stage2_rl/sequential_runner.py`.
- [x] Run the focused Stage-2 output and persistence tests.
- [x] Evaluate the live run post hoc against both 100-window criteria without stopping it.
- [x] Generate a self-contained HTML snapshot with curves, current/best configuration, constraints, entropy diagnostics, and the new convergence verdict.
