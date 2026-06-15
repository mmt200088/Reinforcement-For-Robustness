# ADR-015: Continuous bounded reward + strict stability brake (port of Stage-1 + original Stage-2)

Date: 2026-06-14 (stability-gate framing corrected 2026-06-15)
Status: Accepted

> **2026-06-15 correction.** An earlier draft called `--stage2-stability-tolerance
> 5.0` (500%) a "vacuous gate" and implemented the threshold as
> `baseline_std·(1+tol) + floor` (fractional slack). Per user spec the std gate is
> a **MULTIPLIER**: `thr = max(baseline.X_std × tol, stab_floor)`. So tol=5.0 → 5×
> baseline std — a deliberately LENIENT but **real** gate (a config with std > 5×
> baseline still fails it), tol=1.2 → the original Stage-2's 1.2×. 500% is not
> vacuous; the user's std requirement simply isn't strict, and >100% is meaningful.
> The principled anti-runaway is therefore primarily the continuous bounded reward
> (an accuracy violation is clipped to −5, no ±40 jumps) + the strict 0.5% accuracy
> gate + strict feasibility selection; the std multiplier is a tunable secondary
> constraint the user runs lenient (5.0×). Sections below are corrected accordingly.
Supersedes: the reward SHAPING of ADR-011/012/013/014 (split budget, near-miss
tier, log-barrier-on-margin tier add-on, fusion saturation) and the tier 0/+20/+40
structure inherited from `noise_rl_module_v2`. KEEPS: the fusion-count action +
offline maps (ADR-008), deterministic seeding + episode-parallel (ADR-009/010),
and the ADR-014 debug instrumentation (health log / diagnostics curve / persisted
mu / regenerator). The `reward_design="tiered"` toggle preserves the ADR-014 path
for A/B + rollback.

## Context

Four 60k fusion runs collapsed (COLD/COLD/HOT/HOT) despite ADR-011→014 patches.
After studying the curves the user gave fundamental feedback: the exploration
strategy, the initial policy, and the initial reward magnitude were all wrong; the
reward varied too wildly across the run; and — unlike Stage-1 — the search did NOT
produce a config that STRICTLY satisfies accuracy AND stability ("I don't even see
a stability constraint"). The directive: study how the ORIGINAL Stage-2
(`noise_rl_module_v2.py`) and Stage-1 (`layer_importance_evaluator.py`) were
designed and port them, instead of adding a 5th patch.

What the study found (the three designs, by function):

| | Stage-1 `_compute_final_reward` | Original Stage-2 `_compute_terminal_reward_mc` | current fusion Stage-2 |
|---|---|---|---|
| reward | continuous log-barrier(loss,m1,m2)+cost, **/20, clip[-5,5]** | margin perf/violation+cost+**std stability**, clip±5 **+tiers 0/20/40** | tiers + ADR patches; amplitude [-7,+45] |
| stability | none (N/A) | **std gate** (`stability_base_std_tolerance=1.2`, a 1.2× MULTIPLIER), only when metric_ok | gate present but threshold was mis-implemented as `(1+tol)+floor` fractional slack |
| selection | **strict `_candidate_meets_constraints`** (loss/m1/m2) + rank metric→loss→cost, else baseline | hard-priority | hard-priority rank without strict feasibility fallback |
| exploration | cosine entropy 0.05→0.001 (25% plateau, lower-bound 0.012, recovery 25×, KL-stop 0.02) | — | baseline anchor + decaying prior + curriculum |

Root-cause → complaint map: ① amplitude = the tier ±40 jumps at the feasibility
boundary (Stage-1 avoids them by being CONTINUOUS); ② "no stability" = the std
gate's threshold was mis-implemented as `(1+tol)+floor` fractional slack rather
than a `baseline_std × tol` multiplier, and there was no strict feasibility
selection wiring it into the reported best; ③ "doesn't strictly satisfy" = no
strict feasibility selection; ④ exploration/initial-policy = the baseline anchor
(which biases the policy toward fusion=0 and is opposite of Stage-1's high-entropy
start). Key insight: the principled anti-runaway is the continuous bounded reward
(an accuracy violation is clipped to −5; there is no longer a ±40 cliff a single
fusion can fall off) + the strict 0.5% accuracy gate + strict feasibility
selection. The std gate is a real `baseline_std × tol` multiplier the user runs
LENIENT (5.0× — their std requirement isn't strict); it still catches a
runaway-std config (std > 5× baseline → P2 → no cost reward) but it is not the
primary brake. This is cleaner than the ADR-014 saturation hack.

User decisions: **(Q1)** continuous bounded reward (no tiers); **(Q2)** a real
(principled) std-multiplier gate wired into priority + selection, retiring the
ADR-014 saturation + the accreted exploration patches. The user runs the gate
lenient (`--stage2-stability-tolerance 5.0` = 5×), so the reward bounding +
accuracy gate + strict selection carry the anti-runaway.

## Decision

### 1. Continuous bounded reward (`reward.py`, `reward_design="continuous"`, default)
`raw = W_acc·mean(acc_barrier) + W_stab·mean(stab_barrier)`; the PPO scalar is
`clip(raw/NORM + W_cost·cost_frac, -5, +5)`. `stage1_log_barrier(margin)` is a
faithful port of Stage-1's: satisfied `SAT·log(margin+eps)`, violated
`-VIO·exp(-margin·STEEP)` (exponent clamped, then the clip bounds it). The
accuracy (performance) barrier runs over the active metric margins — m1/m2 means
(higher-better, allowed to DROP ≤ `limit_tolerance`) AND `loss_mean` (lower-better,
allowed to RISE ≤ `limit_tolerance`; 2026-06-15 user spec "loss 也是", mirroring
Stage-1's loss/m1/m2 joint constraint and folded into `metric_ok`/priority). Each
quantity's margin is computed in ITS OWN direction, so a higher-better mean is
never multiplied by the std tolerance (the "1.2 ↔ 0.8" trap). The stability barrier
runs over the signed std margins `(thr-std)/|thr-baseline|` where
`thr = baseline.X_std × stab_tolerance` (a MULTIPLIER). `loss_mean` gating is
continuous-only; the tiered rollback keeps its m1/m2-only gate bit-identical.
`cost_frac` is the P3-gated
fusion saving in `[0,1]`. Constants mirror Stage-1 (VIO=10, STEEP=20, SAT=0.5,
NORM=20; W_acc=W_stab=1, W_cost=4). **No tiers** — hard priority / item 7 holds via
WEIGHTING (a violated barrier pins the scalar at CLIP_MIN while cost can only lift
a fully-feasible config by ≤ W_cost) plus strict selection. The result is bounded
to ~[-5,+5] and CONTINUOUS across the feasibility boundary (locked by
`tests/test_blb_continuous_reward.py`: boundary gap < 8 vs the tiered path's > 20).

### 2. Std-multiplier stability gate (real, tunable, runs lenient)
The std gate is `std ≤ max(baseline.X_std × tol, stab_floor)` — `tol` is a
**MULTIPLIER** on the noisy-baseline std (2026-06-15 user spec), not fractional
slack. It feeds both the continuous stab_barrier AND the priority. The run uses a
LENIENT `--stage2-stability-tolerance 5.0` (= 5× baseline std) because the user's
std requirement isn't strict; tol=1.2 reproduces the original Stage-2's 1.2× if a
tighter gate is wanted. Even at 5× the gate is real: a config with std > 5×
baseline → P2 → no cost reward, so a runaway-std solution still can't harvest cost.
The primary anti-runaway, though, is §1's bounded reward + the strict accuracy gate
+ §3's strict selection. The ADR-014 `fusion_saturation_tau` default is set to 0
(retired; saturate_fusion stays dormant for the tiered rollback).

### 3. Strict feasibility selection + baseline fallback (`sequential_runner.py`)
Port of Stage-1's `_select_stage1_reward_best_config`: the reported best must be
`terminal_priority == 3` (strict accuracy AND stability). If the search never
found a P3 candidate, the best falls back to the baseline (feasible by
construction). So the reported optimum STRICTLY satisfies both constraints — what
the user asked for.

### 4. Stage-1 exploration (`sequential_runner.py`)
`_resolve_cosine_ent_coef_schedule` (port of Stage-1's): start 0.05, hold for the
first 25%, cosine-decay to 0.001, floored at 0.012 — high→low, NO anchor. Under
`reward_design="continuous"` the ADR-011/012 patches are gated OFF: baseline
anchor (`force_baseline_episodes=0`), warmstart prior (`gain=0`), fusion probes
(`interval=0`), ε floor (`0`), and the safe-neighbor curriculum (off). The small
all-valid (option,K) space wants high-entropy exploration from episode 1.

## Invariants
- item 7 (cost never offsets an accuracy/stability violation): a violated barrier
  dwarfs the P3-gated cost in the scalar, and cost is P3-gated + selection is
  strict.
- 1==N: the reward is a pure deterministic function of metrics/action.
- Policy/critic shapes unchanged → checkpoint-compatible (no SEQ_RL_VARIANT bump);
  `reward_design` is a scalar/exploration switch, not a network change.
- Reward NOT comparable across ADR-015.

## Consequences
- Judge the next 60k by Stage-1's standard: a smooth small-amplitude (~[-5,5])
  reward curve; the reported best STRICTLY satisfies accuracy + stability (or is
  an explicit baseline fallback). With the lenient 5× std gate, fusion is mainly
  bounded by the 0.5% accuracy gate (not stability); it stabilizes at a
  strictly-feasible level, which may be modest — the honest result of strict
  feasibility selection, which the user prefers over maximal fusion. The ADR-014
  health log / diagnostics curve / persisted mu make all of this readable.
- The std gate is a tunable knob: tighten `--stage2-stability-tolerance` toward
  1.2× to make stability bite harder, or keep it lenient (5×) if accuracy is the
  binding constraint — a deliberate, visible trade, not a hidden one.
