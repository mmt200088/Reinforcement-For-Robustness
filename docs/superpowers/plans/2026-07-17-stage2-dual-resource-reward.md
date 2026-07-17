# Stage-2 Dual-Resource Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Stage-2 scalar fusion/K exchange rate with independent compute and communication axes, a max-min PPO surrogate, resource-local Shapley credit, two-dimensional strict selection, and compatible persistence.

**Architecture:** Keep the current 12-step factorized PPO and robust six-constraint gates. Make `layerwise_action.py` the pure source of truth for resource scoring and attribution, pass its bounded score through the existing terminal reward interface, and make `layerwise_runner.py` recompute the exact two-dimensional objective from every candidate action matrix for ranking, restore, Pareto tracking, and convergence. Break checkpoint compatibility explicitly through the existing algorithm contract.

**Tech Stack:** Python 3, NumPy, PyTorch PPO, `unittest`/`pytest`, append-only JSONL candidate persistence.

---

### Task 1: Implement the pure dual-resource objective

**Files:**
- Modify: `blb_stage2_rl/layerwise_action.py`
- Modify: `tests/test_blb_layerwise_action.py`

- [ ] **Step 1: Write failing tests for independent axes and the packed max-min score**

Add tests that build canonical 12x6 action matrices and assert:

```python
baseline = compute_variable_cost_from_action_matrix(all_k13_fusion0)
fusion_only = compute_variable_cost_from_action_matrix(one_fusion)
k_only = compute_variable_cost_from_action_matrix(one_k12)

self.assertEqual(baseline.compute_saving, 0.0)
self.assertEqual(baseline.communication_saving, 0.0)
self.assertEqual(fusion_only.compute_saving, 1.0 / 12.0)
self.assertEqual(fusion_only.communication_saving, 0.0)
self.assertEqual(k_only.compute_saving, 0.0)
self.assertEqual(k_only.communication_saving, 1.0 / 295.0)
self.assertEqual(fusion_only.robust_floor, 0.0)
self.assertEqual(k_only.robust_floor, 0.0)
```

Enumerate every realizable `(F, C)` pair and prove that a positive
`robust_floor` improvement always beats any secondary-only difference in
`ppo_resource_score`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_blb_layerwise_action.py -q
```

Expected: failures for absent dual-resource fields/constants and old scalar-unit expectations.

- [ ] **Step 3: Implement the pure resource model and Shapley attribution**

Replace the exchange-rate constants with:

```python
MAX_COMPUTE_SAVING_UNITS = 12.0
MAX_COMMUNICATION_SAVING_UNITS = 59.0 * (13.0 - 8.0)
RESOURCE_SECONDARY_EPSILON = 1.0e-4
LAYERWISE_COST_MODEL_REVISION = "dual_resource_maxmin_shapley_v1"
```

Extend the transport dataclass with explicit fields:

```python
@dataclass(frozen=True)
class VariableCost:
    compute_saving: float
    communication_saving: float
    robust_floor: float
    secondary_progress: float
    ppo_resource_score: float
    compute_shapley_credit: float
    communication_shapley_credit: float
    fusion_count: int
    removed_k_bits: int
    layer_resource_rewards: Tuple[float, ...]
    slot_resource_rewards: Tuple[Tuple[float, ...], ...]

    @property
    def normalized(self) -> float:
        return self.ppo_resource_score
```

Use pure helpers:

```python
def dual_resource_score(compute_saving: float, communication_saving: float) -> tuple[float, float, float]:
    robust_floor = min(compute_saving, communication_saving)
    secondary = 0.5 * (compute_saving + communication_saving)
    packed = (robust_floor + RESOURCE_SECONDARY_EPSILON * secondary) / (
        1.0 + RESOURCE_SECONDARY_EPSILON
    )
    return robust_floor, secondary, packed

def resource_shapley_credits(compute_saving: float, communication_saving: float) -> tuple[float, float]:
    value = lambda f, c: dual_resource_score(f, c)[2]
    compute_credit = 0.5 * value(compute_saving, 0.0) + 0.5 * (
        value(compute_saving, communication_saving) - value(0.0, communication_saving)
    )
    communication_credit = 0.5 * value(0.0, communication_saving) + 0.5 * (
        value(compute_saving, communication_saving) - value(compute_saving, 0.0)
    )
    return compute_credit, communication_credit
```

Distribute compute credit only over the 12 Block4 slots and communication credit
only over the 59 active K slots, proportional to each slot's normalized
within-family contribution. Layer-0 Block1 K remains zero.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_blb_layerwise_action.py -q
```

Expected: all layerwise action tests pass, including exact Shapley efficiency and per-family isolation.

- [ ] **Step 5: Commit the pure objective**

```bash
git add blb_stage2_rl/layerwise_action.py tests/test_blb_layerwise_action.py
git commit -m "feat(stage2): separate compute and communication reward"
```

### Task 2: Wire the dual-resource score through reward and environment

**Files:**
- Modify: `blb_stage2_rl/reward.py`
- Modify: `blb_stage2_rl/layerwise_env.py`
- Modify: `tests/test_blb_layerwise_env.py`
- Modify: `tests/test_blb_continuous_reward.py`

- [ ] **Step 1: Write failing terminal-handoff and P1/P2 isolation tests**

Assert that terminal info contains:

```python
{
    "compute_saving": objective.compute_saving,
    "communication_saving": objective.communication_saving,
    "robust_floor": objective.robust_floor,
    "secondary_progress": objective.secondary_progress,
    "ppo_resource_score": objective.ppo_resource_score,
    "compute_shapley_credit": objective.compute_shapley_credit,
    "communication_shapley_credit": objective.communication_shapley_credit,
    "layer_resource_rewards": [...],
    "slot_resource_rewards": [...],
}
```

For P1, P2, and invalid assessments, vary `ppo_resource_score` across `[0, 1]`
and assert the reward is unchanged. For P3, assert the resource score is added
once and remains bounded.

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m pytest tests/test_blb_layerwise_env.py tests/test_blb_continuous_reward.py -q
```

Expected: missing resource fields and legacy `variable_cost` payload assertions fail.

- [ ] **Step 3: Update the environment handoff without changing probe semantics**

Keep `external_cost_score` as the compatibility transport into `base.step`, but
pass only `objective.ppo_resource_score`. Never pass `F + C`, raw fusion count,
or removed K bits as the reward scalar. Emit the explicit resource-objective
mapping in `terminal_info` and in `RewardBreakdown` diagnostics.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python3 -m pytest tests/test_blb_layerwise_env.py tests/test_blb_continuous_reward.py -q
```

- [ ] **Step 5: Commit the reward/environment handoff**

```bash
git add blb_stage2_rl/reward.py blb_stage2_rl/layerwise_env.py \
  tests/test_blb_layerwise_env.py tests/test_blb_continuous_reward.py
git commit -m "feat(stage2): hand off dual-resource terminal reward"
```

### Task 3: Preserve factorized PPO return and isolate slot credit

**Files:**
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `tests/test_blb_layerwise_policy.py`

- [ ] **Step 1: Write failing reward-redistribution and actor-credit tests**

Use a P3 objective whose compute and communication Shapley credits differ.
Assert:

```python
self.assertAlmostEqual(sum(layer_rewards), ppo_resource_score)
self.assertTrue(all(row[0] == 0.0 for row in k_only_slot_rewards))
self.assertTrue(all(value == 0.0 for row in fusion_only_slot_rewards for value in row[1:]))
self.assertAlmostEqual(sum(map(sum, slot_rewards)), ppo_resource_score)
```

Assert P1/P2 actor slot credits are all zero and shared constraint return remains
the terminal reward. Assert P3 actor shared return is
`terminal_reward - ppo_resource_score` and each factor receives only its own
resource slot credit.

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py tests/test_blb_layerwise_policy.py -q
```

- [ ] **Step 3: Consume resource-local credits in the layerwise rollout**

Read `ppo_resource_score`, `layer_resource_rewards`, and
`slot_resource_rewards` from terminal info. Keep the existing generic rollout
buffer API; it already accepts one nonnegative credit per active factor. Validate
that layer and slot sums equal the packed score before mutating the buffer.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py tests/test_blb_layerwise_policy.py -q
```

- [ ] **Step 5: Commit factorized resource credit**

```bash
git add blb_stage2_rl/layerwise_runner.py \
  tests/test_blb_layerwise_runner.py tests/test_blb_layerwise_policy.py
git commit -m "feat(stage2): assign resource-local PPO credit"
```

### Task 4: Make strict selection, Pareto tracking, and convergence two-dimensional

**Files:**
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `tests/test_blb_layerwise_runner.py`

- [ ] **Step 1: Write failing strict-rank, Pareto, restore, and convergence tests**

Construct strict-feasible candidates where:

```text
A: F=0.2, C=0.9
B: F=0.3, C=0.3
C: F=0.3, C=0.5
```

Assert `C` ranks before `B`, `B` ranks before `A`, and the Pareto frontier keeps
`A` and `C` while dropping `B` because `C` dominates it. Assert confidence,
safety margin, and action lexicographic order are used only after equal `(B,S)`.

Persist a deliberately stale scalar and assert restore recomputes all resource
fields from the 12x6 action matrix. Assert convergence patience resets when
either `(B,S)` improves or the selected action changes.

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py -q
```

- [ ] **Step 3: Implement exact objective keys and a pure Pareto helper**

Use an ascending strict key shaped as:

```python
(
    -candidate["robust_floor"],
    -candidate["secondary_progress"],
    *confidence_order,
    *margin_order,
)
```

Add a pure helper that returns deterministic non-dominated candidates under
`(compute_saving, communication_saving)`. Recompute the resource objective from
`action_matrix` in promotion, restore, revalidation, and strict snapshot code;
never trust persisted scalar cost values.

Replace convergence scalar fields with an exact two-value objective and keep the
selected action identity as the second plateau gate. Final strict revalidation
remains mandatory.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py -q
```

- [ ] **Step 5: Commit two-dimensional selection**

```bash
git add blb_stage2_rl/layerwise_runner.py tests/test_blb_layerwise_runner.py
git commit -m "feat(stage2): select dual-resource robust frontier"
```

### Task 5: Version and persist the new scientific contract

**Files:**
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `blb_stage2_rl/persistence.py`
- Modify: `tests/test_blb_layerwise_runner.py`
- Modify: `tests/test_blb_stage2_outputs.py`
- Modify: `tests/test_rl_data_points.py`

- [ ] **Step 1: Write failing contract and structured-output tests**

Assert the algorithm contract includes:

```python
{
    "algorithm_revision": "dual_resource_maxmin_shapley_multifidelity_convergence_v9",
    "cost_model_revision": "dual_resource_maxmin_shapley_v1",
    "resource_secondary_epsilon": 1.0e-4,
    "compute_axis_denominator": 12,
    "communication_axis_denominator": 295,
    "resource_credit_mode": "two_family_shapley_per_slot_v1",
    "strict_resource_order": ["robust_floor", "secondary_progress"],
}
```

Assert a v8 checkpoint is rejected before manifest mutation. Assert episode,
PPO-update, status, final summary, and report fixtures contain both axes, both
family credits, the exact `(B,S)` key, Pareto rows, and the complete action table.

- [ ] **Step 2: Run tests and verify RED**

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py \
  tests/test_blb_stage2_outputs.py tests/test_rl_data_points.py -q
```

- [ ] **Step 3: Update contract, candidate identity, and structured writers**

Set the v9 algorithm revision and extend the existing hash input. Replace active
layerwise `best_variable_cost` output with explicit resource fields while keeping
a read-only compatibility alias `best_variable_cost = ppo_resource_score` only
where an older report reader requires it. The alias must never rank candidates
or drive convergence.

Persist the deferred scenario-weighted option only in the design/spec metadata;
do not add a runtime weight flag or inactive reward branch.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
python3 -m pytest tests/test_blb_layerwise_runner.py \
  tests/test_blb_stage2_outputs.py tests/test_rl_data_points.py -q
```

- [ ] **Step 5: Commit contract and persistence**

```bash
git add blb_stage2_rl/sequential_runner.py blb_stage2_rl/persistence.py \
  tests/test_blb_layerwise_runner.py tests/test_blb_stage2_outputs.py \
  tests/test_rl_data_points.py
git commit -m "feat(stage2): persist dual-resource reward contract"
```

### Task 6: Prove policy behavior and run regression gates

**Files:**
- Modify: `tests/test_blb_layerwise_policy.py`
- Modify: `tests/test_stage2_stage1_rl_alignment.py`

- [ ] **Step 1: Replace the old all-max scalar bandit with a constrained dual-resource bandit**

Create an all-feasible synthetic environment where reward uses the production
dual-resource helper. Train factorized PPO long enough to assert that the
deterministic policy improves both `F` and `C`; a policy with `F == 0` and low K
must fail even if its old scalar sum would be high. Keep entropy regularization
at zero so convergence is policy-driven.

- [ ] **Step 2: Run the synthetic test and verify it fails before final wiring**

```bash
python3 -m pytest \
  tests/test_blb_layerwise_policy.py::LayerwisePolicyTests::test_factorized_ppo_converges_both_resource_axes \
  -q
```

- [ ] **Step 3: Complete any minimal wiring exposed by the synthetic test**

Adjust only dual-resource actor-credit plumbing. Do not alter PPO clipping,
optimizer hyperparameters, action masks, K levels, or constraint probabilities.

- [ ] **Step 4: Run focused and broad regression suites**

```bash
python3 -m pytest \
  tests/test_blb_layerwise_action.py \
  tests/test_blb_layerwise_env.py \
  tests/test_blb_layerwise_policy.py \
  tests/test_blb_layerwise_runner.py \
  tests/test_blb_continuous_reward.py \
  tests/test_blb_stage2_outputs.py \
  tests/test_rl_data_points.py \
  tests/test_stage2_stage1_rl_alignment.py -q

python3 -m py_compile \
  blb_stage2_rl/layerwise_action.py \
  blb_stage2_rl/layerwise_env.py \
  blb_stage2_rl/layerwise_runner.py \
  blb_stage2_rl/reward.py \
  blb_stage2_rl/sequential_runner.py
```

Expected: all focused tests pass; torch-gated tests may run only where PyTorch is installed, but no non-torch failure is accepted.

- [ ] **Step 5: Audit the final diff against the approved scope**

```bash
git diff --check HEAD~5..HEAD
git status --short
rg -n "fusion1_khalf_per_bit_v1|MAX_VARIABLE_COST_UNITS" \
  blb_stage2_rl tests
```

Expected: no active layerwise objective references the retired exchange-rate
model; unrelated source and artifacts are untouched.

- [ ] **Step 6: Commit verification-only test adjustments**

```bash
git add tests/test_blb_layerwise_policy.py tests/test_stage2_stage1_rl_alignment.py
git commit -m "test(stage2): verify dual-resource PPO convergence"
```
