# Stage-2 Layerwise Robust Constrained PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active Stage-2 fusion-count search with a 12-step layerwise PPO that chooses per-layer Block4 fusion and Block1/2/3/4/5 truncation K, while enforcing 0.1% precision and 200% stability constraints with robust 5x5 baseline evidence.

**Architecture:** Keep the legacy 47-step per-block implementation as a rollback path. Add pure statistical and layerwise-action modules, a layerwise environment that reuses the existing Block runtime and terminal model probe, and a focused layerwise rollout loop that reuses the existing GTrXL PPO update. Wire the new mode through the current runner, persistence, and launcher without changing fusion-map or noise-install semantics.

**Tech Stack:** Python 3, NumPy, PyTorch, existing BLB Stage-2 fusion maps, Rescale Optimizer bridge, unittest/pytest, shell launcher, JSON/JSONL persistence.

---

## File Structure

New files:

- `blb_stage2_rl/statistical_constraints.py`: raw trial records, baseline reference, deterministic bootstrap, six-channel feasibility assessment.
- `blb_stage2_rl/layerwise_action.py`: 12-step six-slot schedule, Block3 K-only codec, variable cost, and neighbor generation.
- `blb_stage2_rl/layerwise_env.py`: one policy step per Transformer layer and shared terminal probe handoff.
- `blb_stage2_rl/layerwise_runner.py`: layerwise rollout collection, PPO updates, evidence accumulation, convergence tracking, promotion, and strict selection.
- `tests/test_blb_statistical_constraints.py`: torch-free statistical contract tests.
- `tests/test_blb_layerwise_action.py`: schedule, Block3 K, fixed fusion, cost, and neighbor tests.
- `tests/test_blb_layerwise_env.py`: layer aggregation and terminal handoff tests.
- `tests/test_blb_layerwise_policy.py`: schedule identities, initial probabilities, and undiscounted credit tests.
- `tests/test_blb_robust_constrained_reward.py`: strict reward ordering and six independent gates.
- `tests/test_blb_robust_baseline.py`: raw-trial propagation and 5x5 baseline orchestration tests.
- `tests/test_blb_layerwise_runner.py`: 25-trial baseline, promotion, convergence, and final selection tests.

Modified files:

- `blb_stage2_rl/reward.py`: carry raw trial values and add the robust constrained reward branch.
- `blb_stage2_rl/env.py`: preserve raw trial values and assess candidates against an installed baseline reference.
- `blb_stage2_rl/sequential_env.py`: expose a shared one-block runtime helper without changing legacy behavior.
- `blb_stage2_rl/sequential_policy.py`: accept exact schedule identities and initialize slot logits from explicit probabilities.
- `blb_stage2_rl/sequential_runner.py`: build the robust baseline reference and dispatch the active fusion path to the layerwise runner.
- `blb_stage2_rl/fusion_cost.py`: keep legacy cost and expose variable-only layerwise cost diagnostics.
- `blb_stage2_rl/seed_utils.py`: deterministic baseline group seeds.
- `blb_stage2_rl/diagnostics.py`, `blb_stage2_rl/persistence.py`, `rl_data_points.py`: raw trials, six probabilities, layer actions, per-type entropy, and robust-best fields.
- `blb_stage2_rl/runner.py`, `rl_tune.py`, `layer_importance_evaluator.py`, `llama_7B_LayerImportance.sh`, `presets/mrpc-blb-stage2-rl.conf`: configuration and launcher plumbing.
- `scripts/blb_regen_stage2_outputs.py`: compact per-layer Block4/K table in regenerated reports.

### Task 1: Build The Statistical Constraint Core

**Files:**
- Create: `blb_stage2_rl/statistical_constraints.py`
- Create: `tests/test_blb_statistical_constraints.py`

- [ ] **Step 1: Write failing tests for pooled baseline statistics**

Add tests that construct five `TrialSeries` groups of five values and assert pooled means, `ddof=1` standard deviations, 0.1% precision limits, and 2.0x stability limits:

```python
ref = build_baseline_reference(
    groups,
    precision_tolerance=0.001,
    stability_multiplier=2.0,
    bootstrap_samples=512,
    seed=17,
)
assert ref.trial_count == 25
assert ref.loss_limit == pytest.approx(np.mean(loss) * 1.001)
assert ref.metric1_limit == pytest.approx(np.mean(m1) * 0.999)
assert ref.metric2_limit == pytest.approx(np.mean(m2) * 0.999)
assert ref.loss_std_limit == pytest.approx(np.std(loss, ddof=1) * 2.0)
```

- [ ] **Step 2: Write failing tests for six independent probabilities**

Use candidates where exactly one channel fails at a time. Assert that no average can hide a loss, m1, m2, loss-std, m1-std, or m2-std failure. Assert deterministic equality from two calls with the same bootstrap seed.

- [ ] **Step 3: Write failing tests for degenerate baseline rejection**

Assert `DegenerateBaselineVariance.channels` names every zero/non-finite channel and that fewer than 25 trials raises `InsufficientBaselineTrials`.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
python3 -m pytest -q tests/test_blb_statistical_constraints.py
```

Expected: import failure because `statistical_constraints.py` does not exist.

- [ ] **Step 5: Implement the pure statistical API**

Implement these public types and functions:

```python
@dataclass(frozen=True)
class TrialSeries:
    loss: Sequence[float]
    metric1: Sequence[float]
    metric2: Sequence[float]
    seeds: Sequence[int] = field(default_factory=tuple)

@dataclass(frozen=True)
class BaselineReference:
    trials: TrialSeries
    trial_count: int
    precision_tolerance: float
    stability_multiplier: float
    bootstrap_seed: int
    bootstrap_samples: int
    loss_mean: float
    metric1_mean: float
    metric2_mean: float
    loss_std: float
    metric1_std: float
    metric2_std: float
    loss_limit: float
    metric1_limit: float
    metric2_limit: float
    loss_std_limit: float
    metric1_std_limit: float
    metric2_std_limit: float
    bootstrap_means: Mapping[str, np.ndarray]
    bootstrap_stds: Mapping[str, np.ndarray]

@dataclass(frozen=True)
class ConstraintAssessment:
    loss_precision_probability: float
    metric1_precision_probability: float
    metric2_precision_probability: float
    loss_stability_probability: float
    metric1_stability_probability: float
    metric2_stability_probability: float
    precision_probability: float
    stability_probability: float
    gate_probability: float
    online_precision_pass: bool
    online_stability_pass: bool
```

The public functions are `build_baseline_reference(groups, *, precision_tolerance,
stability_multiplier, bootstrap_samples, seed) -> BaselineReference` and
`assess_candidate(trials, reference, *, gate_probability, bootstrap_seed) ->
ConstraintAssessment`.

Use vectorized NumPy bootstrap index matrices. For each bootstrap row, independently resample both the pooled baseline trials and the candidate trials, derive the baseline-relative precision limit, and compare the resampled candidate mean. Compare the resampled candidate `ddof=1` standard deviation against `stability_multiplier * resampled_baseline_std`. Clip returned probabilities to `[0, 1]` and reject non-finite inputs.

- [ ] **Step 6: Run tests and verify GREEN**

Run the focused test command again. Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add blb_stage2_rl/statistical_constraints.py tests/test_blb_statistical_constraints.py
git commit -m "Add robust Stage-2 statistical constraints"
```

### Task 2: Preserve Raw Trials And Calibrate A 5x5 Baseline

**Files:**
- Modify: `blb_stage2_rl/reward.py`
- Modify: `blb_stage2_rl/env.py`
- Modify: `blb_stage2_rl/seed_utils.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Create: `tests/test_blb_robust_baseline.py`
- Test: `tests/test_blb_stage2_rl_regressions.py`

- [ ] **Step 1: Write failing tests for raw trial propagation**

Assert `_aggregate_probe_trials()` returns population summaries plus exact tuples:

```python
metrics = env._aggregate_probe_trials(
    [0.3, 0.4], [0.8, 0.9], [0.7, 0.8], trial_seeds=[11, 12],
)
assert metrics.loss_trials == (0.3, 0.4)
assert metrics.metric1_trials == (0.8, 0.9)
assert metrics.metric2_trials == (0.7, 0.8)
assert metrics.trial_seeds == (11, 12)
assert metrics.loss_std == pytest.approx(np.std([0.3, 0.4], ddof=1))
```

The test must expect `ddof=1`, replacing the current biased `ddof=0` summary.

- [ ] **Step 2: Write failing tests for baseline group orchestration**

Use a fake base env that records probe seeds and returns five raw trials per call. Assert `_collect_robust_baseline_reference()` calls exactly five disjoint groups for a healthy baseline, 25 trials total, and extends group-by-group to at most 50 when a channel is degenerate.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python3 -m pytest -q \
  tests/test_blb_stage2_rl_regressions.py \
  tests/test_blb_robust_baseline.py
```

Expected: missing raw fields and baseline collection helper.

- [ ] **Step 4: Extend `EpisodeMetrics` and probe aggregation**

Add backward-compatible tuple fields:

```python
loss_trials: Sequence[float] = field(default_factory=tuple)
metric1_trials: Sequence[float] = field(default_factory=tuple)
metric2_trials: Sequence[float] = field(default_factory=tuple)
trial_seeds: Sequence[int] = field(default_factory=tuple)
```

Change all three standard deviations in `_aggregate_probe_trials()` to `ddof=1` for `n > 1`. Populate trial seeds from deterministic and probe-runner paths.

- [ ] **Step 5: Add deterministic baseline group seeds**

Add:

```python
def derive_baseline_group_probe_seed(base_seed: int, group_idx: int) -> int:
    return derive_probe_seed(base_seed, PREFLIGHT_EPISODE - 1 - int(group_idx))
```

Test uniqueness for groups 0 through 9 and reproducibility across calls.

- [ ] **Step 6: Implement baseline collection**

In `sequential_runner.py`, add `_collect_robust_baseline_reference()` that evaluates the metric-baseline action with five trials per group, builds a reference after group 5, extends only on `DegenerateBaselineVariance`, and raises before PPO after group 10. Persist the raw groups and reference summary in `baseline_preflight_metrics`.

- [ ] **Step 7: Run focused tests and commit**

```bash
python3 -m pytest -q \
  tests/test_blb_stage2_rl_regressions.py \
  tests/test_blb_robust_baseline.py
git add blb_stage2_rl/reward.py blb_stage2_rl/env.py \
  blb_stage2_rl/seed_utils.py blb_stage2_rl/sequential_runner.py \
  tests/test_blb_stage2_rl_regressions.py tests/test_blb_robust_baseline.py
git commit -m "Calibrate Stage-2 constraints from raw 5x5 trials"
```

### Task 3: Define The Six-Slot Layerwise Action And Restore Block3 K

**Files:**
- Create: `blb_stage2_rl/layerwise_action.py`
- Create: `tests/test_blb_layerwise_action.py`
- Modify: `tests/test_blb_fusion_count_map.py`

- [ ] **Step 1: Write failing schedule tests**

Assert a 12-layer schedule has 12 steps and canonical slots:

```python
assert LAYERWISE_SLOT_NAMES == (
    "block4_fusion", "block1_k", "block2_k",
    "block3_k", "block4_k", "block5_k",
)
assert schedule[0].slot_mask == (True, False, True, True, True, True)
assert all(s.slot_mask == (True,) * 6 for s in schedule[1:])
assert schedule[-1].terminal
```

- [ ] **Step 2: Write failing codec tests with the real MRPC maps**

Start from `make_all_max_action_vector(12)`, apply one layer action, and assert:

- Block1 is fusion 0 when present.
- Block2 and Block5 resolve the unique map option whose declared fusion count is 1.
- Block4 resolves exactly the policy-selected fusion count.
- Block3 changes only `output_truncation_k`; every Block3 SF index remains baseline.
- GELU all 4 selects `block5_n4`.

- [ ] **Step 3: Write failing variable-cost and neighbor tests**

Assert 59 active K slots, monotonic lower-K saving, fixed Block2/5 invariance, equal fusion/K budget, and exactly `12 + 59 * 5 == 307` neighbors when six K levels are present.

- [ ] **Step 4: Run tests and verify RED**

```bash
python3 -m pytest -q tests/test_blb_layerwise_action.py
```

- [ ] **Step 5: Implement schedule, codec, cost, and neighbors**

Expose:

```python
@dataclass(frozen=True)
class LayerwiseStepSpec:
    step_idx: int
    layer_idx: int
    slot_dims: Sequence[int]
    slot_mask: Sequence[bool]
    terminal: bool

@dataclass(frozen=True)
class LayerwiseDecodedAction:
    block4_fusion: int
    k_by_block: Mapping[int, int]

@dataclass(frozen=True)
class LayerActionApplication:
    full_vector: np.ndarray
    decoded: LayerwiseDecodedAction
    fusion_option_ids: Mapping[int, int]
    boosted_field_values_by_block: Mapping[int, Mapping[str, int]]

@dataclass(frozen=True)
class VariableCost:
    fusion_saving: float
    truncation_saving: float
    normalized: float
```

Expose `layerwise_schedule(num_layers, fusion_map) -> list[LayerwiseStepSpec]`,
`apply_layer_action(full_vector, layer_action, step_spec, fusion_map) ->
LayerActionApplication`, `compute_variable_cost(actions) -> VariableCost`, and
`one_coordinate_neighbors(action) -> Iterator[list[list[int]]]`.

`compute_variable_cost()` must use decoded values and the exact normalization:

```python
fusion_saving = sum(a.block4_fusion for a in actions) / 12.0
k_values = [k for action in actions for k in action.k_by_block.values()]
truncation_saving = sum((13 - k) / 5.0 for k in k_values) / 59.0
variable_cost = 0.5 * fusion_saving + 0.5 * truncation_saving
```

Resolve map options by declared `fusion_count`, never by assuming `option_id == fusion_count`. Locate Block3 K through `_BLOCK_SPECS[3]` and the legacy full-vector offsets.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest -q tests/test_blb_layerwise_action.py tests/test_blb_fusion_count_map.py
git add blb_stage2_rl/layerwise_action.py tests/test_blb_layerwise_action.py \
  tests/test_blb_fusion_count_map.py
git commit -m "Add layerwise Block4 and truncation actions"
```

### Task 4: Add A Layerwise Environment Without Regressing The Legacy Path

**Files:**
- Create: `blb_stage2_rl/layerwise_env.py`
- Create: `tests/test_blb_layerwise_env.py`
- Modify: `blb_stage2_rl/sequential_env.py`
- Test: `tests/test_sequential_smoke.py`

- [ ] **Step 1: Write a legacy parity test before extraction**

With fake bridge outputs, record the current `BLBStage2SequentialEnv.evaluate_step()` result for one Block2 and one boosted Block5 action. Assert decoded cfg, validity, total bits, fusion count, boosted overrides, and optimizer overrides.

- [ ] **Step 2: Extract a shared Block runtime helper**

Move the existing action-vector-to-cfg, SF-direct boosted rebuild, bridge evaluate, and optimizer-output apply code behind an internal helper. Define its result explicitly:

```python
@dataclass(frozen=True)
class BlockRuntimeResult:
    cfg: Mapping[str, object]
    valid: bool
    optimizer_output: Mapping[str, object]
    total_bits: float
    fusion_count: int
    boosted_overrides: Mapping[str, int]
    optimizer_overrides: Mapping[str, object]
```

Use this interface:

```python
result = evaluate_block_from_full_vector(
    base_env=base_env,
    full_vec=full_vec,
    layer_idx=layer_idx,
    block_idx=block_idx,
    graph_key=graph_key,
    boosted_field_values=boosted_field_values,
)
assert isinstance(result, BlockRuntimeResult)
```

Make the legacy env call this helper and run the parity tests before adding new behavior.

- [ ] **Step 3: Write failing layerwise environment tests**

Assert one `step()` consumes six policy slots, replans the four active layer-0 blocks including Block3, advances the outer step once, and records one aggregate optimizer signal. For layers 1-11, assert five blocks are replanned. Assert only step 11 invokes the terminal model probe.

- [ ] **Step 4: Implement `BLBStage2LayerwiseEnv`**

The class must expose the same rollout-facing surface used by PPO:

```python
env = BLBStage2LayerwiseEnv(base_env=base_env, schedule=schedule)
assert env.horizon == 12
assert env.max_step_dim == 6
observation = env.reset(seed=17)
step_spec = env.current_spec()
next_observation, reward, done, info = env.step(policy_action)
```

Keep the pending full vector baseline-seeded. Apply and replan all active blocks in the current layer, aggregate validity/bits/fusion into one history row, preserve boosted overrides, and call `base.step()` exactly once at terminal with the variable cost and robust reference installed.

- [ ] **Step 5: Verify old and new paths**

```bash
python3 -m pytest -q tests/test_blb_layerwise_env.py tests/test_sequential_smoke.py
```

- [ ] **Step 6: Commit**

```bash
git add blb_stage2_rl/layerwise_env.py blb_stage2_rl/sequential_env.py \
  tests/test_blb_layerwise_env.py tests/test_sequential_smoke.py
git commit -m "Add 12-step Stage-2 layerwise environment"
```

### Task 5: Align Policy Identity, Initialization, And Credit Assignment

**Files:**
- Modify: `blb_stage2_rl/sequential_policy.py`
- Create: `tests/test_blb_layerwise_policy.py`
- Modify: `tests/test_stage2_stage1_rl_alignment.py`

- [ ] **Step 1: Write failing exact-identity tests**

Construct a policy config with `horizon=12`, `step_layer_indices=range(12)`, and layer-level block tokens. Assert `_step_layer_block_indices()` returns the provided arrays and no retired 59-step arithmetic is used.

- [ ] **Step 2: Write failing initial-probability tests**

Initialize slot 0 with `[0.60, 0.40]` and slots 1-5 with probabilities assigned by K value:

```python
{13: 0.50, 12: 0.20, 11: 0.12, 10: 0.08, 9: 0.06, 8: 0.04}
```

Assert softmaxed zero-state logits match these probabilities within tolerance and every legal action has nonzero support.

- [ ] **Step 3: Write failing undiscounted-credit test**

Build a 12-transition episode with only terminal reward 2.0 and zero critic values. Assert `gamma=1.0, lambda=1.0` produces return and advantage 2.0 at every layer.

- [ ] **Step 4: Implement exact schedule identity and probability initialization**

Extend `SequentialPolicyConfig` with validated optional tuples:

```python
step_layer_indices: Sequence[int] | None = None
step_block_indices: Sequence[int] | None = None
```

Add `set_initial_slot_probabilities()` that writes log-probability bias by slot and decoded K index. Keep legacy derivation only when explicit arrays are absent.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m pytest -q tests/test_blb_layerwise_policy.py tests/test_stage2_stage1_rl_alignment.py
git add blb_stage2_rl/sequential_policy.py tests/test_blb_layerwise_policy.py \
  tests/test_stage2_stage1_rl_alignment.py
git commit -m "Align Stage-2 policy with layerwise decisions"
```

### Task 6: Implement Strict Robust Reward And Variable-Only Cost

**Files:**
- Modify: `blb_stage2_rl/reward.py`
- Modify: `blb_stage2_rl/env.py`
- Modify: `blb_stage2_rl/fusion_cost.py`
- Create: `tests/test_blb_robust_constrained_reward.py`
- Modify: `tests/test_blb_fusion_reward.py`

- [ ] **Step 1: Write property tests for strict reward ordering**

Generate probabilities across `[0, 1]` and costs across `[0, 1]`. Assert every valid P3 scalar exceeds every P2 scalar, every P2 exceeds every P1, and invalid equals `-5`. Assert P1/P2 reward is unchanged when cost changes.

- [ ] **Step 2: Write channel-isolation tests**

For each of the six probability fields, lower only that field below 0.5 and assert the expected priority. In particular, independently fail loss-std, m1-std, and m2-std.

- [ ] **Step 3: Write variable-cost tests**

Assert changing fixed Block2/5 fusion does not change `C`; one Block4 0->1 and every one-level K reduction increase `C`; `C=0.5*F+0.5*T`; and actual K values drive the calculation despite legacy index order.

- [ ] **Step 4: Implement robust reward**

Add `reward_design="robust_constrained"` and a pure helper matching the accepted formula:

```python
def boundary_signal(probability: float, eps: float = 1e-8) -> float:
    return float(np.clip(np.log((probability + eps) / (0.5 + eps)), -1.0, 1.0))

q_precision = min(
    assessment.loss_precision_probability,
    assessment.metric1_precision_probability,
    assessment.metric2_precision_probability,
)
q_stability = min(
    assessment.loss_stability_probability,
    assessment.metric1_stability_probability,
    assessment.metric2_stability_probability,
)

reward, priority, precision_signal, stability_signal = robust_constrained_reward(
    assessment,
    invalid=invalid,
    variable_cost=variable_cost,
    eps=1e-8,
)
```

When invalid, the helper returns reward `-5.0`, priority `INVALID`, and the two
computed boundary signals. Otherwise it returns
`-3.0 + 0.5 * boundary_signal(q_precision)` for P1 when `q_precision < 0.5`,
`-1.5 + 0.5 * boundary_signal(q_stability)` for P2 when precision passes but
`q_stability < 0.5`, and
`1.0 + variable_cost + 0.0005 * (boundary_signal(q_precision) +
boundary_signal(q_stability))` for P3. Cost must not enter P1 or P2.

Extend `RewardBreakdown` with six probabilities, `variable_cost`, and `constraint_policy="bootstrap_5x5_v1"`. In `BLBStage2Env._finish_prepared_terminal_probe()`, assess raw trials against `self.statistical_reference` and pass the assessment into `compute_reward()`.

- [ ] **Step 5: Verify focused and legacy reward tests**

```bash
python3 -m pytest -q \
  tests/test_blb_robust_constrained_reward.py \
  tests/test_blb_fusion_reward.py \
  tests/test_blb_continuous_reward.py \
  tests/test_blb_log_barrier_reward.py
```

- [ ] **Step 6: Commit**

```bash
git add blb_stage2_rl/reward.py blb_stage2_rl/env.py \
  blb_stage2_rl/fusion_cost.py tests/test_blb_robust_constrained_reward.py \
  tests/test_blb_fusion_reward.py
git commit -m "Add robust constrained Stage-2 reward"
```

### Task 7: Add Layerwise PPO Rollout, Evidence Promotion, And Strict Selection

**Files:**
- Create: `blb_stage2_rl/layerwise_runner.py`
- Create: `tests/test_blb_layerwise_runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `blb_stage2_rl/candidate_store.py`

- [ ] **Step 1: Write failing 12-step rollout tests**

Use a fake layerwise env and policy. Assert one episode writes 12 transitions, each joint action log probability is the sum over active slots, layer 0 excludes Block1 K, and the PPO buffer uses `gamma=lambda=1.0`.

- [ ] **Step 2: Write failing evidence accumulation tests**

Add `CandidateStore.append_trial_group(action_indices, trials, metadata)` and
`CandidateStore.trial_evidence_for_action(action_indices, identity_context)`.
Record the same action twice with disjoint five-trial groups. Assert the store
returns ten raw trials and the runner recomputes assessment from all ten instead
of returning the first aggregate.

- [ ] **Step 3: Write failing promotion tests**

Assert a P3 frontier action with six probabilities >=0.80 is promoted to 25 total trials exactly once; P1/P2 and dominated P3 actions are not promoted.

- [ ] **Step 4: Write failing final-rank and convergence tests**

Assert strict-feasible candidates sort by variable cost before reward. Break ties by descending minimum feasibility confidence, then ascending loss and descending m1/m2. Assert fixed-slot entropy is excluded, convergence requires both Block4 and K normalized entropy <0.1 plus 100 update windows without cost improvement, and no run converges before 30k episodes.

- [ ] **Step 5: Implement the layerwise rollout loop**

Expose:

```python
summary = train_layerwise(
    env=env,
    policy=policy,
    train_cfg=train_cfg,
    candidate_store=candidate_store,
    on_episode_end=on_episode_end,
    on_ppo_update_end=on_ppo_update_end,
)
assert "best_action" in summary
```

The `candidate_store` parameter is the existing append-only `CandidateStore`,
extended by Step 2; do not introduce a second persistence format.

Reuse `SequentialRolloutBuffer` and `sequential_ppo_update`. Do not copy legacy warmstart, forced probes, neighbor curriculum, or epsilon exploration branches. Implement promotion with the existing deferred terminal evaluation mechanism and fresh trial seeds.

- [ ] **Step 6: Dispatch only the active fusion mode**

In `run_sequential_via_runner()`, choose `BLBStage2LayerwiseEnv` and `train_layerwise()` when `decision_granularity == "layer"` and fusion count actions are enabled. Preserve `BLBStage2SequentialEnv` and `train_sequential()` for explicit `"block"` rollback.

- [ ] **Step 7: Run tests and commit**

```bash
python3 -m pytest -q tests/test_blb_layerwise_runner.py tests/test_sequential_smoke.py
git add blb_stage2_rl/layerwise_runner.py blb_stage2_rl/sequential_runner.py \
  blb_stage2_rl/candidate_store.py tests/test_blb_layerwise_runner.py \
  tests/test_sequential_smoke.py
git commit -m "Train Stage-2 PPO with layerwise robust rollouts"
```

### Task 8: Persist Auditable Evidence And Wire The Launcher

**Files:**
- Modify: `blb_stage2_rl/runner.py`
- Modify: `blb_stage2_rl/diagnostics.py`
- Modify: `blb_stage2_rl/persistence.py`
- Modify: `rl_data_points.py`
- Modify: `rl_tune.py`
- Modify: `layer_importance_evaluator.py`
- Modify: `llama_7B_LayerImportance.sh`
- Modify: `presets/mrpc-blb-stage2-rl.conf`
- Modify: `scripts/blb_regen_stage2_outputs.py`
- Test: `tests/test_rl_data_points.py`
- Test: `tests/test_stage2_persistent_launcher.py`
- Test: `tests/test_blb_stage2_outputs.py`

- [ ] **Step 1: Write failing persistence tests**

Require manifest and episode JSONL fields for baseline groups, raw trials, six probabilities, thresholds, variable cost, per-layer action matrix, Block4/K entropy, promotion count, convergence state, and strict-best assessment. Require a compact 12-row table with Block4 fusion and K_B1/B2/B3/B4/B5.

- [ ] **Step 2: Write failing launcher tests**

Require active defaults:

```text
decision_granularity=layer
reward_design=robust_constrained
stage2_limit_tolerance=0.001
stage2_stability_multiplier=2.0
stage2_k_trials=5
baseline_groups=5
baseline_trials_per_group=5
constraint_bootstrap_samples=4096
online_constraint_probability=0.50
promotion_constraint_probability=0.80
final_constraint_probability=0.95
promotion_validation_trials=25
final_selection_validation_trials=25
rollout_size=120
stage2_lr=5e-5
gamma=1.0
gae_lambda=1.0
```

Assert substage mode and all retired scaffolds remain disabled.

- [ ] **Step 3: Implement config and CLI plumbing**

Add typed config fields through shell -> `rl_tune.py` -> evaluator -> `BLBStage2TrainConfig`. Validate decision granularity in `{"layer", "block"}`, probability thresholds in `(0,1]`, baseline counts as positive integers, and stability multiplier as positive. The robust layerwise path must use `stage2_stability_multiplier=2.0` directly as `candidate_std <= 2.0 * baseline_std`; it must never route through the legacy additive `1 + stage2_stability_tolerance` interpretation. Keep the old tolerance field only for the explicit legacy block rollback path.

- [ ] **Step 4: Extend diagnostics and structured writer**

Write raw evidence without converting non-finite values into invalid JSON. Mirror every run under:

```text
rl_training_data_points/stage2/bert-base/mrpc/<run-id>/
```

Keep PNG/NPZ optional inspection outputs; JSON/JSONL is mandatory.

- [ ] **Step 5: Update report regeneration**

Render baseline distributions, reward/entropy curves, six constraint probabilities, variable cost, promotion/final evidence, and the full 12-row layerwise action table including Block3 K.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m pytest -q \
  tests/test_rl_data_points.py \
  tests/test_stage2_persistent_launcher.py \
  tests/test_blb_stage2_outputs.py
git add blb_stage2_rl/runner.py blb_stage2_rl/diagnostics.py \
  blb_stage2_rl/persistence.py rl_data_points.py rl_tune.py \
  layer_importance_evaluator.py llama_7B_LayerImportance.sh \
  presets/mrpc-blb-stage2-rl.conf scripts/blb_regen_stage2_outputs.py \
  tests/test_rl_data_points.py tests/test_stage2_persistent_launcher.py \
  tests/test_blb_stage2_outputs.py
git commit -m "Persist and launch robust layerwise Stage-2 PPO"
```

### Task 9: Verify Integration, Run The Server Gate, Then Launch The Long Search

**Files:**
- Modify if needed: `SERVER_COMMAND.md`
- Create through runtime: `rl_training_data_points/stage2/bert-base/mrpc/<run-id>/`
- Create through runtime: Stage-2 persistent run directory and final HTML report.

- [ ] **Step 1: Run the complete focused local suite**

```bash
python3 -m py_compile \
  blb_stage2_rl/statistical_constraints.py \
  blb_stage2_rl/layerwise_action.py \
  blb_stage2_rl/layerwise_env.py \
  blb_stage2_rl/layerwise_runner.py
python3 -m pytest -q \
  tests/test_blb_statistical_constraints.py \
  tests/test_blb_layerwise_action.py \
  tests/test_blb_layerwise_env.py \
  tests/test_blb_layerwise_policy.py \
  tests/test_blb_robust_constrained_reward.py \
  tests/test_blb_layerwise_runner.py \
  tests/test_stage2_stage1_rl_alignment.py \
  tests/test_sequential_smoke.py
```

Expected: all locally runnable tests pass; torch-gated skips are listed explicitly.

- [ ] **Step 2: Commit the verified source snapshot**

```bash
git status --short
git log -1 --oneline
```

Expected: clean worktree and a commit containing all implementation tasks.

- [ ] **Step 3: Upload the exact Git snapshot to the new server**

Use `git archive HEAD`, transfer to `root@100.64.229.185:8722`, record source commit, archive SHA256, Python path, CUDA/PyTorch versions, GPU model, and command line in the run directory. Do not copy the dirty main worktree.

- [ ] **Step 4: Run server-only focused tests**

Run the Task 9 focused suite in the server environment with torch. Add the real map/install tests:

```bash
python3 -m pytest -q \
  tests/test_blb_fusion_count_map.py \
  tests/test_blb_verify_boosted_install.py \
  tests/test_blb_final_eval_fusion_fixed_action.py
```

Expected: layerwise Block3 K, fixed Block2/5, Block4 boost, and terminal install tests all pass.

- [ ] **Step 5: Run a 120-episode integrity smoke**

The smoke is accepted only if logs prove:

- GELU `[4]*12`, Softmax `[6]*12`;
- metric baseline fusion all 0 / K13;
- baseline calibration has 5 groups and 25 raw trials;
- action horizon 12 and max step dim 6;
- every episode records 12 Block4 choices and 59 K choices including Block3;
- reward exposes six probabilities and strict P1/P2/P3 ranges;
- raw JSONL mirrors to `rl_training_data_points/`;
- GPU terminal probes are active and no config-install regression appears.

Do not interpret reward quality or convergence from this smoke.

- [ ] **Step 6: Launch and monitor the 60k search**

Launch BERT-base MRPC with precision tolerance `0.001`, stability multiplier `2.0`, 60,000 initial episodes, five trials per online candidate, rollout size 120, and learning rate `5e-5`. Keep the single RTX 4090 occupied with approved probe work; overlap CPU replan and persistence where existing code permits.

Monitor reward, P1/P2/P3 rates, collapse sentinels, six-channel probabilities, variable cost, Block4/K entropy, PPO KL/clip/value diagnostics, throughput, and GPU utilization. Do not stop for isolated noise or judge before the long-run criteria.

- [ ] **Step 7: Continue if not converged**

After 60k, continue in 12k increments until both normalized entropies are below 0.1 and robust feasible cost has not improved for 100 PPO updates. Preserve one run identity and append to the same persistent data tree.

- [ ] **Step 8: Revalidate, audit neighbors, and report**

Revalidate the cost-ranked top 20 to 25 probe trials, run top five plus metric baseline on `validation_full` with 5x5 trials, require all six probabilities >=0.95, run the 307-neighbor local audit, and generate the final HTML report. The report must call the winner a tested local optimum, not a mathematically proven global optimum.

- [ ] **Step 9: Commit compact evidence only**

Commit the final HTML report, compact JSON summaries, manifest, configuration, and verification logs. Keep large checkpoints/raw model outputs in the persistent artifact store and link them from the report.
