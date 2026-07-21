# Stage-2 Block3 Runtime Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Block3 K affect real model inference while keeping Block3 fusion/SF permanently fixed to the Rescale Optimizer baseline.

**Architecture:** Reuse the existing canonical action decode, optimizer request/write-back, and `BLBNoiseRLBridge` installation pipeline. Block3 SFs remain baseline-owned; only K is policy-owned and passes unchanged through replan into the installed per-layer config.

**Tech Stack:** Python, unittest/pytest, PyTorch, existing BLB noise bridge, Rescale Optimizer bridge.

---

### Task 1: Lock the missing Block3 request and installation behavior

**Files:**
- Modify: `tests/test_blb_block3_removed_schedule.py`
- Modify: `tests/test_blb_stage2_rl_regressions.py`

- [ ] **Step 1: Replace the obsolete removal assertions with failing wiring assertions**

Add tests that require `build_optimizer_requests()` to emit Block3 entries and
require `BLBNoiseRLBridge.apply()` to invoke `replace_layer_block3_noise` with
the per-layer configuration.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest tests/test_blb_block3_removed_schedule.py tests/test_blb_stage2_rl_regressions.py -q
```

Expected: failures showing Block3 is excluded from optimizer requests and
ignored by the bridge.

### Task 2: Connect Block3 through optimizer and bridge

**Files:**
- Modify: `blb_stage2_rl/action_space.py`
- Modify: `blb_rl_bridge.py`

- [ ] **Step 1: Include Block3 in optimizer requests**

Remove the Block3 skip in `build_optimizer_requests`; retain the existing
layer-0 Block1 exception only.

- [ ] **Step 2: Install Block3 per layer**

Call:

```python
self.handler.replace_layer_block3_noise(
    layer_indices=list(block3_cfgs.keys()),
    layer_name=self.layers_attribute,
    cfg_per_layer=dict(block3_cfgs),
)
```

Track `block3` in `_installed` so `clear()` restores it through the existing
reverse-order cleanup.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run the Task 1 command and require zero failures.

### Task 3: Prove baseline SF ownership and K passthrough

**Files:**
- Modify: `tests/test_blb_chain_integrity.py`
- Modify: `tests/test_blb_block3_removed_schedule.py`

- [ ] **Step 1: Add a failing baseline/K chain test**

Construct two layerwise actions that differ only in Block3 K. Assert their
Block3 SF fields are identical and baseline-derived, while
`output_truncation_k` differs.

- [ ] **Step 2: Add a failing replan preservation test**

Run the canonical optimizer write-back on Block3 and assert the selected K is
unchanged while the final SF fields match optimizer output.

- [ ] **Step 3: Run tests and verify RED or existing behavior**

If the decode/passthrough assertions already pass, retain them as regression
coverage; the model-install assertion from Task 1 supplies the required RED.

- [ ] **Step 4: Make only the minimal implementation adjustments needed**

Do not add a Block3 SF/fusion policy action. Do not add K to Rescale Optimizer
delta arithmetic.

### Task 4: Verify actual Block3 truncation execution

**Files:**
- Modify: `tests/test_blb_stage2_rl_regressions.py`

- [ ] **Step 1: Add a deterministic model-hook test**

Install two otherwise identical `Block3NoiseConfig` values with different K,
control the noise generator, and assert the post-polynomial output differs due
to `_apply_truncation`.

- [ ] **Step 2: Run focused torch-backed tests**

```bash
python -m pytest \
  tests/test_blb_block3_removed_schedule.py \
  tests/test_blb_stage2_rl_regressions.py \
  tests/test_blb_chain_integrity.py -q
```

Expected: all selected tests pass.

### Task 5: Documentation, full verification, and server gate

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Replace stale Block3-removed documentation**

Document baseline-owned Block3 SF, policy-owned K, RO SF-only replan, and real
model installation.

- [ ] **Step 2: Run local verification**

```bash
python -m py_compile blb_stage2_rl/action_space.py blb_rl_bridge.py
python -m pytest tests/test_blb_block3_removed_schedule.py tests/test_blb_stage2_rl_regressions.py tests/test_blb_chain_integrity.py -q
```

- [ ] **Step 3: Commit and push the isolated branch**

Commit only the scoped source, tests, and documentation.

- [ ] **Step 4: Run the verified commit on the GPU server**

Run the same focused tests with torch/model support, then execute a narrow
Block3 action replay that records baseline SFs, selected K, replan status,
installed Block3 config, and output difference between two K values.

