# Stage-2 Fusion-Count Reward Redesign — Design

**Date:** 2026-06-03
**Status:** Approved (design Q&A 2026-06-03)
**Related:** ADR-008 (fusion-count action), ADR-002/007 (reward), Stage-1 reward
(`layer_importance_evaluator.py` `_compute_dense_step_reward` / `_compute_final_reward`)

## Goal

Redesign the Stage-2 **fusion-count** RL reward so the per-block cost signal is a
per-block-type weighted fusion + truncation saving (user ratio
`block1:block2:block4:block5:truncation = 80:150:130:40:50`), `total_bits` is removed
from the reward scalar, a 4-trial cross-GPU stability probe is (re)enabled per
episode, and the hard priority `accuracy > stability >> cost` is preserved. Borrow
Stage-1's "cost = saving ratio + per-block decomposition" flavor; keep Stage-2's
hard-tier backbone and add the stability gate Stage-1 lacks.

This applies **only** to the fusion-count action path (`--blb-v3-fusion-count-action 1`).
The per-slot legacy path and single-shot path are unchanged.

## Locked decisions (design Q&A)

- **Q1 — block1/block4 fusion-degeneracy:** implement weights as specified. With the
  current mrpc maps block1/block4 have only option 0 (fusion≡0), so their fusion
  weights (80/130) are **inert** — those blocks tune only K. block2(150)/block5(40)
  fusion is active. 80/130 are reserved for a future SF-reduction option set
  (ADR-008). No server rebuild.
- **Q2 — cost accounting:** terminal **P3-gated** cost. Cost is awarded only when
  accuracy AND stability both pass (priority 3). Per-block decomposition is preserved
  (each block's fusion/K contribution is explicit and weighted); GAE distributes
  credit across the 47 steps. Honors the CLAUDE.md item-7 contract (cost never offsets
  an accuracy/stability violation).

## Reward structure

### Backbone (kept from reward.py v3)

Hard priority via tier bonus:
- **P1** (invalid OR accuracy fail): shaping `margin_acc + invalid_term`, clipped
  `[-5,+5]`, tier `0` → roughly `[-5, 0]`. Cost contributes `0`.
- **P2** (accuracy ok, stability fail): shaping `margin_acc + stability_penalty`,
  tier `+20` → roughly `[+15, +25]`. Cost contributes `0`.
- **P3** (accuracy ok AND stability ok): shaping `p3_metric_margin + cost_score`,
  tier `+40` → roughly `[+40, +45]`. Cost lives here, bounded by `p3_cost_budget=4.5`,
  so it can never cross a tier (`stability >> cost`, `accuracy >> stability`).

### ① P3 cost — per-block weighted fusion + truncation saving (the redesign)

Weights (new constants in `reward.py`):
```
FUSION_COST_W = {1: 80.0, 2: 150.0, 4: 130.0, 5: 40.0}   # per block-TYPE, per block instance
TRUNC_COST_W  = 50.0                                       # per block instance
K_MAX = 13   K_MIN = 8                                     # K_LEVELS range
```

Per block `b` of block-type `t(b)`, with chosen `(fusion_option, K)`:
```
max_fusion(t)  = max fusion_count in that block-type's map   # block2/block5 -> 1; block1/block4 -> 0
fusion_saving_b = chosen_fusion_count / max_fusion(t)        # 0 when max_fusion==0 (block1/block4)
trunc_saving_b  = (K_MAX - K) / (K_MAX - K_MIN)              # in [0,1]

actual = Σ_b FUSION_COST_W[t(b)] * fusion_saving_b  +  Σ_b TRUNC_COST_W * trunc_saving_b
norm   = actual / MAX_ACTUAL                                  # in [0,1]
```
`MAX_ACTUAL` is the maximum achievable weighted saving for the active schedule (sum of
the per-block weights that can actually move). For the current mrpc 47-block schedule:
`12·150 (block2 fusion) + 12·40 (block5 fusion) + 47·50 (truncation) = 4630`. It is
computed from the schedule + map at setup, not hard-coded, so degree/skeleton changes
stay correct.

- **Bounded P3 cost (PPO scalar):** `cost_score = norm * p3_cost_budget`  (≤ 4.5).
- **Unbounded P3 rank (candidate/frontier selection only, never PPO):**
  `cost_rank = actual`.

`total_bits` is removed from the reward scalar entirely (still reported in diagnostics
as a real cost number). In fusion mode the `_adaptive_scalar_cost_score` /
bits-tiebreaker path is bypassed.

### ② Stability — 4-trial cross-GPU probe per episode

- Each terminal probe runs **K=4 trials** (one per GPU, probe subset size 256),
  yielding mean + std. `std` drives the existing P2 stability gate
  (`combined_stab_excess` over `m1_std/m2_std/loss_std`, weights 30:30:1). Unchanged
  gate math; the requirement is that K=4 actually runs so std is real.
- The **noisy baseline preflight** (already installs the all-max baseline and reads
  noisy std) uses the same K=4 probe — already wired; the only requirement is fusion
  mode must not silently degrade online K to 1 (fast-reward `online-k=1` deferral is
  disabled in fusion mode; `num_trials_per_step >= 2` is required, warn otherwise).

### ③ Accuracy gate — unchanged

Existing m1/m2 dual-metric hard threshold (`baseline·(1-tol)` + probe-size guard).
Stage-1's log-barrier shape is **not** adopted (YAGNI; the linear margin + tier already
give a hard gate). Noted as a future option.

### ④ Warmstart — stronger baseline prior

The `fusion=0 / K=max` choice (= option 0 + baseline-K = the baseline action, the
safest/highest-accuracy point) gets a stronger initial logit prior so cold-start sits
at baseline and explores outward. `warmstart_bias_gain` for fusion mode: `1.2 → 2.5`
(decay schedule unchanged). The action space is tiny (≤2 options × 6 K per block) so a
strong prior is safe.

## Implementation surface (surgical, fusion-mode only)

1. **`blb_stage2_rl/fusion_cost.py` (new, torch-free, unit-tested):**
   `compute_fusion_cost_saving(choices, fusion_map, *, weights) -> FusionCostResult`
   where `choices` is the list of per-block `(block_idx, graph_key, fusion_count,
   max_fusion, k_value)` accumulated during the episode. Returns `(cost_norm ∈ [0,1],
   cost_rank, per_block_breakdown, max_actual)`. Pure arithmetic.

2. **`reward.py`:** add `FUSION_COST_W`, `TRUNC_COST_W`, `K_MAX/K_MIN` constants and
   two optional `compute_reward` params `external_cost_score` / `external_cost_rank`.
   When `external_cost_score is not None` (fusion mode), P3 uses
   `clip(external_cost_score, 0, p3_cost_budget)` as `cost_score` and
   `external_cost_rank` as `cost_rank_score`, bypassing `_adaptive_scalar_cost_score`
   and the bits path. `None` → old behavior bit-for-bit.

3. **`blb_stage2_rl/env.py`:** `BLBStage2Env.step` accepts optional
   `external_cost_score` / `external_cost_rank`, stashed on `self` (reset per step),
   read by the compute_reward calls (5 terminal/branch sites) and passed through.

4. **`blb_stage2_rl/sequential_env.py`:** fusion branch accumulates each committed
   step's `(spec, fusion_option, k_value, fusion_count)` into `self._fusion_choices`
   (reset on `reset()`). At the terminal step, compute the cost via
   `compute_fusion_cost_saving(...)` and pass `external_cost_score/rank` into
   `self.base.step(...)`.

5. **`blb_stage2_rl/sequential_runner.py` + `runner.py`:** fusion-mode config —
   guard `num_trials_per_step >= 2` (warn/raise otherwise), disable fast-reward
   online-k=1 deferral in fusion mode, bump `warmstart_bias_gain` to 2.5; pass
   `external_cost_score=0.0/rank=0.0` for the baseline compute_reward in fusion mode.

6. **Tests (`tests/test_blb_fusion_reward.py`, torch-free):** baseline → 0 saving;
   all-fusion + min-K → norm≈1.0; block1/block4 fusion inert (weight×0); MAX_ACTUAL
   from schedule; `external_cost` threading in `compute_reward` (P3 uses it, P1/P2
   ignore it); K=4 std gate sanity.

## Non-goals / unchanged

- Per-slot legacy path, single-shot `BLBStage2Env`, F0 scan, candidate store schema.
- Tier structure, accuracy gate, stability gate math (only the K=4 enablement).
- `total_bits` still reported in diagnostics/HTML (removed only from the reward scalar).
- block3 (frozen, removed from baseline+cost since 2026-06-03).

## Reward ranges (sanity)

`P1 ≈ [-5,0] < P2 ≈ [15,25] < P3 ≈ [40,45]`, cost spans ≤4.5 inside P3. Confirms
`accuracy >> stability >> cost`.
