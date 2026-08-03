# Stage2 Stage1 Curve Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Stage-2 RL main training curve use the same layout and drawing convention as Stage-1 RL.

**Architecture:** Extract the Stage-1 PPO training curve drawing convention into a small reusable torch-free plotting module. Keep Stage-2-specific cost/fusion diagnostics out of the main training curve so `blb_stage2_training_curve.png` mirrors Stage-1's Reward / Loss / metric panels.

**Tech Stack:** Python, matplotlib Agg backend, existing torch-free Stage-2 output tests.

---

### Task 1: Lock Stage-2 Main Curve Contract

**Files:**
- Modify: `tests/test_blb_stage2_outputs.py`

- [x] Add a torch-free test that monkeypatches the Stage-1-style renderer and asserts Stage-2 passes exactly Reward, Loss, metric1, and metric2 to the main curve.
- [x] Run the focused test and confirm it fails before implementation.

### Task 2: Add Shared Stage-1-Style Renderer

**Files:**
- Create: `training_curve_plot.py`
- Modify: `blb_stage2_rl/persistence.py`

- [x] Move the Stage-1 2x2/1x3 plot convention into `save_stage1_style_training_curve`.
- [x] Change `write_training_curves` to call that renderer for `blb_stage2_training_curve.png`.
- [x] Keep entropy and paper reward plots unchanged.

### Task 3: Verify

**Files:**
- Test: `tests/test_blb_stage2_outputs.py`

- [x] Run the focused Stage-2 output tests.
- [x] Compile touched Python files.
