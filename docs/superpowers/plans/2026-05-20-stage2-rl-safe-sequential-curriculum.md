# Stage2 RL Safe Sequential Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BLB Stage-2 sequential RL leave the baseline anchor through a safe near-baseline curriculum instead of immediately sampling unrestricted accuracy-catastrophic actions.

**Research-loop target:** This task is complete only when server evidence shows
RL can train after the anchor without collapse. The reward curve must remain a
normal RL curve, terminal metrics must avoid collapse sentinels such as
`loss_mean=100`, priority must not enter sustained P1(acc), and monitored
parameters such as loss, PPO entropy/clip fraction, mutation count, radius,
valid steps, and GPU reward-probe state must not show unexplained abrupt
pathologies. If a server run exposes a new abnormal point, do not patch the
server; design the next experiment, update local code, push, pull on server,
run again, and keep iterating.

**Researcher operating mode:** Expect this to take repeated experiment cycles,
possibly over many hours. If the evidence points to training dynamics,
hyperparameters, reward calibration, or exploration design instead of a simple
bug, create a focused experiment, run it, inspect the curve/logs, and update the
local fix or tuning choice based on that evidence. Do not declare completion
because a short unit test passes.

**Architecture:** Extend the sequential policy and PPO buffer with optional per-level masks, then have the sequential runner build one near-baseline mutation mask per episode. The runner keeps all non-selected slots baseline-only, stores the collection mask in the transition, and resolves anchor/entropy schedules using absolute episode indices for resume correctness.

**Tech Stack:** Python `unittest`, NumPy, PyTorch on the server, existing BLB sequential runner/policy modules.

---

### Task 1: Regression Tests

**Files:**
- Modify: `tests/test_sequential_smoke.py`

- [ ] **Step 1: Add torch-free tests**

Add tests that extract small helper functions from `blb_stage2_rl/sequential_runner.py` source and assert:

- `_resolve_sequential_force_baseline_episodes` honors explicit `force_baseline_episodes`.
- It falls back to `warmstart_anchor_episodes` before the old rollout-size fallback.
- `_near_baseline_level_indices` respects non-monotonic `K_LEVELS`.
- `_build_step_level_mask` makes non-selected slots baseline-only and selected slots near-baseline.
- Source wiring contains `action_level_mask` in `SequentialTransition`, `SequentialRolloutBuffer.add`, `to_tensors`, and `sequential_ppo_update`.

- [ ] **Step 2: Run red test**

Run:

```bash
python3 -m unittest tests.test_sequential_smoke.WarmstartFixedRegressionTest -v
```

Expected before implementation: failures for missing helpers / mask wiring.

### Task 2: Sequential Policy Mask Support

**Files:**
- Modify: `blb_stage2_rl/sequential_policy.py`

- [ ] **Step 1: Add optional level masks**

Add `action_level_mask: Optional[torch.Tensor] = None` to `sample_action` and
`evaluate_action`. Combine it with the existing slot/level validity mask before
building `Categorical`.

- [ ] **Step 2: Store masks in PPO buffer**

Add `action_level_mask` to `SequentialTransition`, `SequentialRolloutBuffer.add`,
`to_tensors`, and `sequential_ppo_update`, then pass it to `policy.evaluate_action`
during PPO updates.

- [ ] **Step 3: Verify targeted tests**

Run the same local test command from Task 1. Full torch behavior is verified on
the server.

### Task 3: Safe Sequential Curriculum

**Files:**
- Modify: `blb_stage2_rl/sequential_runner.py`

- [ ] **Step 1: Add helper functions**

Add helpers to resolve anchor count, compute near-baseline allowed indices, build
default/near-baseline step masks, sample episode-level mutation offsets, and
choose mutation budget/radius from absolute episode progress.

- [ ] **Step 2: Wire absolute episode schedules**

Set `SequentialTrainConfig.absolute_episode_start` from the runner's
`start_episode`. Use absolute episode indices for forced baseline and entropy
schedule.

- [ ] **Step 3: Replace post-anchor unrestricted sampling**

When `warmstart_neighbor_sampling` is enabled, generate a per-episode mutation
offset set after the anchor. During each step, pass a per-level mask to
`policy.sample_action`, retry optimizer-invalid samples through the existing
blacklist loop, and store the same mask in the PPO buffer.

- [ ] **Step 4: Log the safety mode**

Add startup log lines for safe neighbor sampling and per-episode max mutation /
radius settings.

### Task 4: Verification and Server Protocol

**Files:**
- Modify: `SERVER_COMMAND.md`
- Create: `experiments/server_command_runs/<run>/stage2_rl_safe_curriculum_report.html`

- [ ] **Step 1: Run local tests**

Run:

```bash
python3 -m unittest tests.test_sequential_smoke.WarmstartFixedRegressionTest -v
```

- [ ] **Step 2: Commit and push local code**

Commit only the plan/spec, tests, code, `SERVER_COMMAND.md`, and final report.
Do not stage unrelated untracked files.

- [ ] **Step 3: Server pull and smoke**

Use `SERVER_COMMAND.md` to run server contract tests and a fresh dual-GPU
sequential smoke of at least 600 episodes.

Monitor at least all windows after the anchor. The smoke is acceptable only if
there is no `loss_mean=100`, no sustained `priority=P1(acc)`, positive reward
windows after the anchor, stable PPO metrics, and visible safe-neighbor
diagnostics in `details/`.

- [ ] **Step 4: Report**

Generate an HTML report with root cause, failed-run evidence, changed files,
test results, server reward curve summary, and remaining long-run caveats.
