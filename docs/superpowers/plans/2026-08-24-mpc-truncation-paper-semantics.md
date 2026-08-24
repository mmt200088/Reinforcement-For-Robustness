# MPC Truncation Paper Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the paper's H/M/L MPC precision and ring-width metadata while preserving every executable cleartext truncation value, action vector, reward input, and model path.

**Architecture:** Extend the existing `PrecisionPreset` value object with an explicit ciphertext semantic plane and retain `k_by_block` as a compatibility alias for the unchanged simulation plane. Keep the legacy action/materialization path untouched and enrich only the shared action-description payload so every report consumer receives both planes and their reserve relationship.

**Tech Stack:** Python 3, dataclasses, NumPy, `unittest`, existing Stage-2 layerwise action codec.

---

### Task 1: Lock the dual-plane preset contract

**Files:**
- Modify: `tests/test_blb_layerwise_precision_presets.py`
- Modify: `blb_stage2_rl/precision_presets.py`

- [ ] **Step 1: Write the failing preset metadata test**

Add a test that requires exact paper tuples, existing simulation tuples, ring
widths, derived reserve values, and the compatibility alias:

```python
def test_preset_table_exposes_paper_semantics_without_changing_simulation_k(self):
    observed = [
        (
            preset.name,
            preset.ciphertext_k_by_block,
            preset.simulation_k_by_block,
            preset.reserve_bits_by_block,
            preset.ciphertext_ring_bits,
            preset.k_by_block,
            preset.communication_utility,
        )
        for preset in PRECISION_PRESETS
    ]
    self.assertEqual(observed, [
        ("high", (13, 13, 13, 13, 13), (11, 10, 10, 12, 11),
         (2, 3, 3, 1, 2), 40, (11, 10, 10, 12, 11), 0.0),
        ("medium", (12, 12, 12, 12, 12), (9, 8, 8, 10, 9),
         (3, 4, 4, 2, 3), 39, (9, 8, 8, 10, 9), 0.5),
        ("low", (11, 11, 11, 12, 11), (7, 6, 6, 8, 7),
         (4, 5, 5, 4, 4), 38, (7, 6, 6, 8, 7), 1.0),
    ])
```

- [ ] **Step 2: Run the test to verify RED**

Run on the approved server checkout:

```bash
python -m unittest -v \
  tests.test_blb_layerwise_precision_presets.LayerwisePrecisionPresetContractTest.test_preset_table_exposes_paper_semantics_without_changing_simulation_k
```

Expected: `ERROR` because the ciphertext/simulation/reserve/ring attributes do
not exist yet.

- [ ] **Step 3: Implement the minimal dual-plane value object**

Change `PrecisionPreset` to hold explicit ciphertext and simulation tuples,
derive reserve bits, and keep the old executable API:

```python
@dataclass(frozen=True)
class PrecisionPreset:
    name: str
    ciphertext_k_by_block: Tuple[int, int, int, int, int]
    simulation_k_by_block: Tuple[int, int, int, int, int]
    ciphertext_ring_bits: int
    communication_utility: float

    @property
    def reserve_bits_by_block(self) -> Tuple[int, int, int, int, int]:
        return tuple(
            int(ciphertext) - int(simulation)
            for ciphertext, simulation in zip(
                self.ciphertext_k_by_block,
                self.simulation_k_by_block,
            )
        )

    @property
    def k_by_block(self) -> Tuple[int, int, int, int, int]:
        return self.simulation_k_by_block
```

Define the three exact paper/simulation/ring tuples and add a separate metadata
version constant without changing `PRECISION_PRESET_VERSION`.

- [ ] **Step 4: Run the focused test to verify GREEN**

Run the same command. Expected: `OK`.

- [ ] **Step 5: Commit the preset contract**

```bash
git add blb_stage2_rl/precision_presets.py \
  tests/test_blb_layerwise_precision_presets.py
git commit -m "feat(stage2): expose paper MPC preset semantics"
```

### Task 2: Freeze executable action and reward parity

**Files:**
- Modify: `tests/test_blb_layerwise_precision_presets.py`
- Modify: `tests/test_blb_layerwise_action.py`

- [ ] **Step 1: Add pre-change executable golden tests**

For all three preset indices, assert that `apply_layer_action()` still emits
the original decoded tuples and K-level indices:

```python
goldens = {
    0: (11, 10, 10, 12, 11),
    1: (9, 8, 8, 10, 9),
    2: (7, 6, 6, 8, 7),
}
for preset_index, expected in goldens.items():
    application = apply_layer_action(
        baseline, (1, preset_index), self.schedule[0], self.fusion_map,
    )
    self.assertEqual(
        tuple(application.decoded.k_by_block.values()), expected,
    )
```

Freeze the existing reward-facing diagnostics:

```python
self.assertEqual(
    [
        compute_variable_cost_from_action_matrix([[0, index]]).removed_k_bits
        for index in range(3)
    ],
    [11, 21, 31],
)
```

- [ ] **Step 2: Verify executable parity tests pass without production changes**

Run on the server:

```bash
python -m unittest -v \
  tests.test_blb_layerwise_action \
  tests.test_blb_layerwise_precision_presets
```

Expected: all executable parity assertions pass. This is a characterization
test, not a RED test; it proves the new metadata did not alter execution.

- [ ] **Step 3: Commit the parity characterization**

```bash
git add tests/test_blb_layerwise_action.py \
  tests/test_blb_layerwise_precision_presets.py
git commit -m "test(stage2): freeze MPC preset execution parity"
```

### Task 3: Enrich the shared action description

**Files:**
- Modify: `tests/test_blb_layerwise_precision_presets.py`
- Modify: `blb_stage2_rl/layerwise_action.py`

- [ ] **Step 1: Expand the readable-description expectation**

Require each layer description to preserve the compatibility field and add the
paper-facing fields:

```python
{
    "precision_preset_name": "high",
    "truncation_k_by_block": {
        "block1": 11, "block2": 10, "block3": 10,
        "block4": 12, "block5": 11,
    },
    "cleartext_simulation_k_by_block": {
        "block1": 11, "block2": 10, "block3": 10,
        "block4": 12, "block5": 11,
    },
    "ciphertext_truncation_k_by_block": {
        "block1": 13, "block2": 13, "block3": 13,
        "block4": 13, "block5": 13,
    },
    "reserve_bits_by_block": {
        "block1": 2, "block2": 3, "block3": 3,
        "block4": 1, "block5": 2,
    },
    "ciphertext_ring_bits": 40,
}
```

- [ ] **Step 2: Run the description test to verify RED**

Run on the server:

```bash
python -m unittest -v \
  tests.test_blb_layerwise_precision_presets.LayerwisePrecisionPresetContractTest.test_readable_action_description_lists_every_layer_and_block_k
```

Expected: `FAIL` because the new paper-facing keys are absent.

- [ ] **Step 3: Add metadata only to `describe_layerwise_action_matrix()`**

Build all dictionaries from the selected `PrecisionPreset`. Continue to source
`truncation_k_by_block` from `simulation_k_by_block`, then add the explicit
ciphertext, simulation, reserve, and ring fields. Do not modify
`apply_layer_action()`, `function_handler.py`, or CUDA code.

- [ ] **Step 4: Run the focused and shared-path suites**

Run on the server:

```bash
python -m unittest -v \
  tests.test_blb_layerwise_precision_presets \
  tests.test_blb_layerwise_action \
  tests.test_blb_stage2_eval_single_path_static
```

Expected: `OK` and no action/materialization parity regression.

- [ ] **Step 5: Commit reporting metadata**

```bash
git add blb_stage2_rl/layerwise_action.py \
  tests/test_blb_layerwise_precision_presets.py
git commit -m "feat(stage2): report ciphertext MPC precision presets"
```

### Task 4: Document semantics and complete handoff verification

**Files:**
- Modify: `docs/BLB_stage2_rl_spec.md`
- Modify: `docs/BLB_stage2_rl_INTERNAL_FLOW.md`
- Modify: `AGENTS.md`
- Create: `agent_handoffs/tasks/mpc-truncation-paper-semantics-20260824.json`

- [ ] **Step 1: Document the two semantic planes**

State unambiguously that `output_truncation_k` remains the executable
cleartext value, while H/M/L public metadata uses the paper tuples and ring
widths. Include the reserve equation and exact tables.

- [ ] **Step 2: Run syntax and focused server tests**

From the pushed task commit in an isolated server checkout:

```bash
python -m py_compile \
  blb_stage2_rl/precision_presets.py \
  blb_stage2_rl/layerwise_action.py
python -m unittest -v \
  tests.test_blb_layerwise_precision_presets \
  tests.test_blb_layerwise_action \
  tests.test_blb_layerwise_env \
  tests.test_blb_stage2_eval_single_path_static \
  tests.test_blb_truncation_backends
```

Expected: all tests pass. CUDA-specific tests may skip only when CUDA is not
available; the final evidence must state the skip count.

- [ ] **Step 3: Audit forbidden execution changes**

Require the final diff to exclude:

```text
function_handler.py
blb_stage2_rl/truncation_fused_cuda.py
blb_stage2_rl/block3_fused_cuda.py
blb_stage2_rl/block5_fused_cuda.py
blb_stage2_rl/action_space.py
```

Review `git diff 480e154...HEAD --name-only` and fail the task if any listed
file changed without a newly approved design amendment.

- [ ] **Step 4: Push the source commit and create the handoff**

Push `codex/task-mpc-truncation-paper-semantics-20260824`, create the handoff
JSON with exact source commit/tree and server evidence, then make one final
handoff-only commit.

- [ ] **Step 5: Run `agent-finish`**

```bash
python3 scripts/repo_sync_guard.py agent-finish \
  --handoff agent_handoffs/tasks/mpc-truncation-paper-semantics-20260824.json \
  --remote origin
```

Expected: guard passes. Report the task branch, handoff path, source commit,
source tree, and server verification evidence to the sole aggregator. Do not
update `jk_standard_rl` or deploy this task branch as server canonical.
