# Layer-0 Block1 Truncation-K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first layer's Block1 truncation K a real RL-selected and model-executed action.

**Architecture:** Reuse the canonical layerwise codec, legacy action materializer, bridge, and `NoisyBlock1LayerNorm`. Decouple Block1 Gaussian-noise enablement from Block1 truncation so layer 0 can use the exact same K executor without enabling a new SF/noise family.

**Tech Stack:** Python, NumPy, PyTorch, unittest, existing Stage-2 BLB action/materialization/bridge modules.

---

### Task 1: Lock The Action And Cost Contract

**Files:**
- Modify: `tests/test_blb_layerwise_action.py`
- Modify: `tests/test_blb_layerwise_policy.py`

- [x] Replace the old masked-layer-zero assertions with tests requiring all six
  layer-0 slots to be active.
- [x] Require `apply_layer_action` to decode Block1 K at layer 0.
- [x] Require one-coordinate neighbors to include `(layer=0, slot=block1_k)`.
- [x] Require communication denominators and removed-bit totals to use `5 * L`.
- [x] Run the tests and confirm they fail on the old mask/exclusion behavior.

### Task 2: Materialize And Install Layer-0 Block1 K

**Files:**
- Modify: `function_handler.py`
- Modify: `blb_rl_bridge.py`
- Modify: `blb_stage2_rl/action_space.py`
- Modify: `tests/test_blb_action_materialization.py`
- Modify: `tests/test_blb_stage2_rl_regressions.py`

- [x] Add a failing materialization test requiring `block1_cfgs[0]` with the
  selected K and Gaussian noise disabled.
- [x] Add a failing bridge regression proving a layer-0 Block1 cfg is installed.
- [x] Add `Block1NoiseConfig.noise_enabled=True` and a builder argument with the
  same default.
- [x] Build layer 0 through the common Block1 action/cfg builders with
  `noise_enabled=False`, while keeping later layers unchanged.
- [x] Remove the bridge's layer-0 Block1 filter; retain the RO request omission
  because K is independent of SF replan.

### Task 3: Execute K At The Real Block1 Boundary

**Files:**
- Modify: `function_handler.py`
- Modify: `tests/test_blb_truncation_backends.py`

- [x] Add a failing torch test that installs a noise-disabled Block1 config and
  proves K is applied to variance before `rsqrt`.
- [x] Make the Block1 FFN2 and LayerNorm noise branches clean when
  `noise_enabled=False`.
- [x] Keep `_apply_configured_truncation(var, cfg)` common to both layer 0 and
  later layers.
- [x] Verify no Gaussian sampler is called in the noise-disabled test.

### Task 4: Update Contracts And Verify Shared Paths

**Files:**
- Modify: `docs/BLB_stage2_rl_spec.md`
- Modify: `docs/blb_baseline_handover_protocol.md`
- Modify: affected tests containing the old `5 * L - 1` or masked-K contract.

- [x] Update documentation to distinguish “no layer-0 Block1 SF/RO entry” from
  “layer-0 Block1 K is active”.
- [x] Run focused unittest suites locally.
- [ ] Run torch/CUDA focused suites on the verified server checkout.
- [ ] Confirm training/final-eval/GLUE static shared-path gates remain green.
- [ ] Commit, push, and update the server checkout to the exact pushed commit.
