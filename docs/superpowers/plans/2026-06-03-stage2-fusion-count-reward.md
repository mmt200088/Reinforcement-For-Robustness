# Stage-2 Fusion-Count Reward Redesign — Implementation Plan

**Goal:** Per-block-type weighted fusion+truncation P3 cost (80:150:130:40:50),
`total_bits` removed from the reward scalar, K=4 cross-GPU stability probe enabled,
stronger baseline warmstart — fusion-mode only, hard priority preserved.

**Architecture:** New pure helper `fusion_cost.py` computes the per-block weighted
saving from accumulated `(option,K)` choices; `compute_reward` gains optional
`external_cost_score/rank` that the env threads in at the terminal step. Non-fusion
paths bit-for-bit unchanged.

**Tech:** torch-free helper + reward (locally unit-tested); env/runner glue (local
py_compile + ruff; real validation on server). Spec:
`docs/superpowers/specs/2026-06-03-stage2-fusion-count-reward-design.md`.

---

### Task R1: `fusion_cost.py` pure helper + tests

**Files:** Create `blb_stage2_rl/fusion_cost.py`; Test `tests/test_blb_fusion_reward.py`.

- [ ] `FusionCostResult` dataclass: `cost_norm: float`, `cost_rank: float`,
  `max_actual: float`, `per_block: list[dict]`.
- [ ] `BlockChoice` dataclass: `block_idx:int, graph_key:str, fusion_count:int,
  max_fusion:int, k_value:int`.
- [ ] `compute_fusion_cost_saving(choices, *, fusion_w, trunc_w, k_max=13, k_min=8,
  max_actual=None) -> FusionCostResult`:
  - per block: `fusion_saving = fusion_count/max_fusion if max_fusion>0 else 0.0`;
    `trunc_saving = (k_max-k)/(k_max-k_min)` clamped `[0,1]`.
  - `actual = Σ fusion_w[block_idx]*fusion_saving + Σ trunc_w*trunc_saving`.
  - `max_actual` defaults to `Σ fusion_w[block_idx]*(1 if max_fusion>0 else 0) +
    Σ trunc_w` (so `cost_norm = actual/max_actual ∈ [0,1]`).
  - return `cost_norm`, `cost_rank=actual`, `max_actual`, `per_block` breakdown.
- [ ] `max_actual_for_choices(choices, *, fusion_w, trunc_w)` (same denom, exposed so
  the env can precompute once).
- [ ] Tests: (a) baseline choices (all `fusion_count=0`, `k=13`) → `cost_norm==0`;
  (b) all `fusion_count==max_fusion>0` + `k=8` → `cost_norm≈1.0`;
  (c) block1/block4 (`max_fusion=0`) contribute 0 to actual AND 0 to denom;
  (d) `cost_rank` monotonic in saving.
- [ ] Run: `python tests/test_blb_fusion_reward.py` → PASS.
- [ ] `ruff check blb_stage2_rl/fusion_cost.py tests/test_blb_fusion_reward.py`.

### Task R2: `reward.py` constants + `external_cost` params

**Files:** Modify `blb_stage2_rl/reward.py`; Test `tests/test_blb_fusion_reward.py`.

- [ ] Add module constants: `FUSION_COST_W = {1:80.0, 2:150.0, 4:130.0, 5:40.0}`,
  `TRUNC_COST_W = 50.0`, `K_MAX_BITS = 13`, `K_MIN_BITS = 8`.
- [ ] `compute_reward(...)`: add kwargs `external_cost_score: Optional[float] = None`,
  `external_cost_rank: Optional[float] = None`.
- [ ] In the P3 branch (`metric_ok and stab_ok`): if `external_cost_score is not None`,
  set `effective_cost_score = clip(external_cost_score, 0.0, p3_cost_budget)` and
  `effective_cost_rank_score = external_cost_rank or 0.0`, bypassing
  `_adaptive_scalar_cost_score`. Leave `r_fusion/r_k/r_bits` at 0 (fusion path doesn't
  use the bits tiebreaker). `None` → unchanged old path.
- [ ] Guard: external cost is ignored for P1/P2 (only read inside the
  `metric_ok and stab_ok` block) — preserves item-7.
- [ ] Tests: P3 with `external_cost_score=3.0` → `cost_score==3.0`, reward in P3 band;
  same external on a P1 (acc fail) episode → cost contributes 0.
- [ ] Run tests → PASS; `ruff check blb_stage2_rl/reward.py`.

### Task R3: `env.py` — thread external cost into the terminal reward

**Files:** Modify `blb_stage2_rl/env.py`.

- [ ] `BLBStage2Env.__init__`: init `self._external_cost_score = None`,
  `self._external_cost_rank = None`.
- [ ] `BLBStage2Env.step(...)`: add kwargs `external_cost_score=None,
  external_cost_rank=None`; stash on `self` at entry; clear to `None` at the end (so a
  later non-fusion step can't inherit stale cost).
- [ ] At the 5 `compute_reward(...)` call sites, pass
  `external_cost_score=self._external_cost_score,
  external_cost_rank=self._external_cost_rank`.
- [ ] `py_compile blb_stage2_rl/env.py`; `ruff check blb_stage2_rl/env.py`.

### Task R4: `sequential_env.py` — accumulate choices, compute at terminal

**Files:** Modify `blb_stage2_rl/sequential_env.py`.

- [ ] `reset()`: `self._fusion_choices = []`.
- [ ] `evaluate_step` fusion branch: stash `option_idx`, `k_idx`, `k_value`,
  `max_fusion`, `graph_key`, `block_idx` into `eval_info` (alongside `fusion_count`).
- [ ] `commit_step` fusion branch: append a `BlockChoice` to `self._fusion_choices`.
- [ ] Terminal (before `self.base.step(...)`): if fusion mode, compute
  `res = compute_fusion_cost_saving(self._fusion_choices, fusion_w=FUSION_COST_W,
  trunc_w=TRUNC_COST_W)`; set `cost_score = res.cost_norm * p3_cost_budget`,
  `cost_rank = res.cost_rank`; call
  `self.base.step(self._pending_full_vec, external_cost_score=cost_score,
  external_cost_rank=cost_rank)`. (Read `p3_cost_budget` from
  `self.base.reward_weights`.)
- [ ] `py_compile`; `ruff check blb_stage2_rl/sequential_env.py`.

### Task R5: runner config — K=4 guard, warmstart bump, baseline cost

**Files:** Modify `blb_stage2_rl/sequential_runner.py`, `blb_stage2_rl/runner.py`.

- [ ] Fusion mode: if `num_trials_per_step < 2`, log a loud warning (std gate needs
  ≥2 trials); keep fast-reward `online-k=1` deferral disabled in fusion mode (it
  already defaults off — assert it stays off when `fusion_map is not None`).
- [ ] Bump `warmstart_bias_gain` to `2.5` for the fusion warmstart branch (where
  `preferred=[0, baseline_k_idx]` is applied).
- [ ] Baseline `compute_reward` calls (runner.py ~1350/1377): in fusion mode pass
  `external_cost_score=0.0, external_cost_rank=0.0` (baseline = no saving).
- [ ] `py_compile` both; `ruff check`.

### Task R6: docs + final validation

**Files:** `CLAUDE.md`, `docs/adr/ADR-008-fusion-count-action.md` (consequences note),
memory, `SEQ_RL_VARIANT`? (no — reward change is checkpoint-compatible; do NOT bump).

- [ ] CLAUDE.md "Fusion-count action" section: add the reward bullet (per-block
  weighted P3 cost 80:150:130:40:50, total_bits removed, K=4 std, warmstart 2.5).
- [ ] ADR-008 consequences: note the reward followed the action change.
- [ ] Memory `stage2-fusion-count-action.md`: append reward-redesign status.
- [ ] `make test` equivalent torch-free subset locally:
  `BLB_STRICT=0 python -m unittest tests.test_blb_fusion_reward -v` + the existing
  fusion-map tests.
- [ ] `ruff check` the full changeset.
- [ ] Commit (do NOT push unless asked; user runs server). Server validates via a
  fusion smoke once SERVER_COMMAND.md is free.
