# ADR-014: Structural anti-runaway fusion cost + collapse-debug instrumentation

Date: 2026-06-14
Status: Accepted
Amends: ADR-013 (the log-barrier is KEPT, but it is no longer the sole restoring
force — a structural cap on the fusion incentive is added, and MARGIN_REF is
re-tuned for the real fusion-regime probe noise). ADR-011/012 cost weights and
probe size/K are unchanged.

## Context

The 4th 60k fusion run (artifacts `stage2_grid_gate_60k_20260613_175503`,
ADR-013 log-barrier active, commit `4e3aec0`) STILL collapsed HOT and the
watchdog killed it at 40320/60000 (P3 < 2% for 12 consecutive windows). The
trajectory (in-repo only as a server-bash health log at the time):

```
ep      meanFus  meanM1   P1/600  P3/600   meanRew
1739    1.1      ~0.87    8       591      37.0
14219   10.2     ~0.86    19      581      37.4    <- still healthy at fusion ~10
15539   13.3     ~0.85    223     376      22.6    <- P1 surges as fusion crosses ~13
20759   23.7     ~0.80    592     8        -5.6    <- P3 ~ 0
31379   35.0     ~0.69    600     0        -8.6    <- saturated, all P1, frozen to 40320
```

Fusion climbed MONOTONICALLY 1 → 10 (healthy) → 35 (all P1), never stabilizing.
The best feasible point (ep23645, fusion 27, P3, reward 40.7) was a rare
low-variance survivor amid a P1 sea, NOT the rolling state. Entropy did not go to
zero → this is reward/constraint shaping (over-fusion), not dead exploration.

**Root cause.** The best point's `terminal_metric1_std` was ~0.0155, but the
feasibility margin (baseline 0.871 − threshold 0.858) is only ~0.013 — the probe
noise EXCEEDS the entire margin and is ~8.6× the 0.0018 baseline sigma ADR-013
calibrated MARGIN_REF (0.25) against. So the log-barrier's headroom is
noise-drowned and cannot form a measurable restoring attractor, while the LINEAR
fusion cost reward (`fusion_norm` × budget; block4 paid 130/fuse) is a
deterministic monotone incentive. Deterministic beats noise → fusion runs away
until accuracy is so deeply violated that even the noisy barrier is consistently
negative, by which point fusion ≈ 35 and the policy is frozen in a flat P1 basin.

**Debugging gap (the second, explicit ask).** The ADR-013 barrier/margin
quantities (`worst_signed_margin`=mu, `acc_barrier_sat/vio`, `near_miss`,
`margin_m1/m2`) were computed in `RewardBreakdown` but NEVER persisted to
`episodes.jsonl`; `fusion_count_b2/b4/b5` weren't surfaced as curves; and the
rolling-health collapse trajectory was computed only by a throwaway server bash
script. So the failing mechanism was a black box — the root cause above had to be
*inferred* from `terminal_metric1_std` rather than read from mu directly.

The user chose (signed off): a structural anti-runaway term + barrier re-tuning,
**keeping the cost weights AND the probe size/K** (no slower probe). So the fix
must work *with* the existing noise: the incentive must flatten before fusion
reaches the noisy boundary.

## Decision

### A. Structural anti-runaway (the fix)
Make the fusion cost reward CONCAVE/saturating instead of linear.
`fusion_cost.saturate_fusion(x, tau) = (1−exp(−x/tau))/(1−exp(−1/tau))` is applied
to `fusion_norm ∈ [0,1]`; the env scales `fusion_norm_saturated` (not the raw) by
the fusion budget. With `FUSION_SATURATION_TAU=0.15`, ~80% of the fusion reward is
harvested by `fusion_norm≈0.23` (≈fusion 8, safely below the noisy boundary
~10–13). The shape has a STEEP initial slope (still pulls UP to the knee → no cold
collapse) and a FLAT tail (no deterministic pull past the knee → no hot collapse).
`tau≤0` is identity (bit-for-bit ADR-013). `fusion_norm` stays RAW for diagnostics.

`DEFAULT_ACC_BARRIER_MARGIN_REF` is raised 0.25 → 0.5 (the 0.25 headroom was
sub-sigma in the fusion regime). Together, `cost(fusion) + barrier(margin(fusion))`
has an interior maximum at a moderate POSITIVE margin (unit-locked in
`tests/test_blb_fusion_saturation.py::InteriorPeakTest` — max fusion is never
optimal). The stable optimum will adopt FEWER fusions than the knife-edge 27 (it
keeps probe-resolvable headroom by design); that is a stable WIN over no-fusion,
not a regression.

Invariants: cost stays P3-gated and the violated barrier stays < the P3 tier floor
→ cost can never offset an accuracy violation (item 7). Saturation is a pure
deterministic function of the action → 1==N byte-identity is unaffected.
Priority / rank-key / selection are bit-identical (only the PPO scalar's cost
shape changes). Reward is NOT comparable across this ADR.

### B. Collapse-debug instrumentation (so the next collapse is read, not guessed)
1. **Persist the black box**: `worst_signed_margin`, `acc_barrier_sat/vio`,
   `near_miss`, `margin_m1/m2`, `fusion_norm_raw/saturated`, `fusion_count_b2/b4/b5`
   now reach `episodes.jsonl` (serial + episode-parallel paths).
2. **In-repo rolling-health log** (`blb_stage2_health.log`): rolling-600 P1/P2/P3 +
   reward + fusion + per-block + margin, written every `save_interval` — the
   server-bash trajectory is now a reproducible repo artifact.
3. **Collapse-diagnostics curve** (`blb_stage2_diagnostics_curve.png`): priority
   mix, fusion (total + per-block), accuracy margin mu, reward components, and
   probe-noise-vs-margin — the fusion↑→mu↓→P3→0 smoking gun at a glance.
4. **Collapse attribution** (`rl_local_optimum.attribute_collapse` →
   `blb_stage2_search_log.txt`): onset episode + fusion trend + HOT/COLD verdict.
5. **Offline regenerator** (`scripts/blb_regen_stage2_outputs.py`) rebuilds all of
   the above from any run's `episodes.jsonl` (torch-free), so past/future runs are
   replayable offline.

## Alternatives considered
- **Bigger probe / more K to make noise < margin** (ADR-013's "next lever"): the
  user chose to keep probe speed; the structural cap removes the need.
- **De-weight / cap block4 fusion** (the most accuracy-toxic, paid 130): the user
  kept the weights; the aggregate concave cap restrains all block types without a
  per-type weight change. Re-open if the curves show block4 specifically.
- **Re-tune the barrier alone (bigger MARGIN_REF / SAT)**: rejected as insufficient
  on its own — against a large noise + a deterministic monotone incentive, only
  removing the incentive past the knee reliably stops the runaway.

## Consequences
- Judge the next 60k by: a non-collapsing curve, fusion stabilizing at a MODERATE
  positive-margin level (not 0, not 35), P3 fraction > 0, best ≥ the no-fusion cap
  — and, crucially, the new `blb_stage2_health.log` + diagnostics curve + persisted
  mu let us SHOW the stabilization or pinpoint any new failure from data.
- Policy/critic shapes unchanged → checkpoint-compatible (no SEQ_RL_VARIANT bump).
- `FUSION_SATURATION_TAU` and `acc_barrier_margin_ref` are the tuning knobs if the
  stable point sits too low (more fusion wanted) or too high (still over-fusing).
