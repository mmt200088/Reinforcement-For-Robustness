# ADR-013: Stage-1-style log-barrier accuracy boundary (replaces near-miss tier + linear P3 margin)

Date: 2026-06-13
Status: Accepted
Supersedes: ADR-012's graded near-miss tier (P1 shaping) and the linear P3
metric-margin term from the 2026-06-03 fusion-count reward design.
Amends: ADR-007's tier shaping at the accuracy boundary (tiers / hard-priority
selection unchanged).

## Context

The 3rd 60k fusion run (artifacts `stage2_grid_gate_60k_20260612_191530`, ADR-012
active: graded near-miss + epsilon floor + policy-K probes, tolerances 5.0/0.005)
flipped the failure mode. The first two 60k runs collapsed COLD (fusion never
adopted / entropy froze at 0, fusion=0). This one collapsed HOT:

```
ep      meanFus  meanM1   P1     P3     meanRew
0       1.4      0.866    74     2919   37.7
12000   11.7     0.857    1240   1740   28.1
21000   19.8     0.838    2694   296     4.6
30000   31.8     0.746    3000   0      -6.9
33000+  35.1     0.690    3000   0      -6.95   (frozen to ep 60000)
```

Fusion marched MONOTONICALLY 1.4 → 35; metric1 fell 0.866 → 0.690; the back
half of the run (30k episodes, 8.5h) froze flat at -6.95 — every episode a
catastrophic P1, zero policy gradient, pure waste. Hard-priority selection still
rescued an excellent config from the climb (ep20880: fusion 22, P3,
reward 40.8, > the no-fusion cap 39.5), so the search CAN find good configs; it
just does not STABILIZE at the feasible frontier.

Five compounding root causes (all evidenced):

1. **Tiny accuracy budget (~1.3%)** consumed JOINTLY by fusion AND deep-K. The
   offline "b2+b5=24 fusion is free" finding was measured on LOSS at baseline-K;
   in the fusion+deep-K regime RL actually explores, metric1 declines steadily
   with fusion from the start. The 3rd-60k optimum sat at metric1 margin 0.0003
   — far below the 256-sample probe's sigma (0.0018), i.e. a coin flip.
2. **Monotone fusion incentive, no restoring force.** The P3 cost `fusion_norm`
   rewards fusing ALL blocks (incl. all 12 block4 layers @ 130); reward keeps
   rising with fusion until accuracy snaps.
3. **block4 cost/accuracy inversion.** block4 fusion is paid the 2nd-highest
   weight (130) but is the most accuracy-toxic (12-layer block4 → loss 0.624).
4. **ADR-012's graded near-miss tier removed the brake.** Softening a borderline
   P1 from the -7 cliff to 15–35 made overshooting the boundary nearly free, so
   the policy slid up the fusion axis past the optimum.
5. **No recovery.** Once fusion saturated, every episode scored the flat -6.95
   cliff floor → zero gradient → frozen. The epsilon floor (0.05) flips too few
   blocks to climb back.

The user pointed at Stage-1 RL (which trains well): its boundary is a smooth
log-barrier (`layer_importance_evaluator.py:log_barrier_reward`) and it never
overshoots or avoids the constraint. The user chose the log-barrier direction
and chose to KEEP the cost weights (80:150:130:40) unchanged — the barrier must
be the sole restoring force, strong enough to counter block4's 130 on its own.

## Decision

Port Stage-1's two-piece log-barrier onto the Stage-2 accuracy margin. It
REPLACES both the ADR-012 near-miss/cliff (P1 shaping) and the linear P3
metric-margin term. Operates on `mu` = the worst per-channel SIGNED margin in
`|baseline - threshold|` units (the same coordinate as the old near_miss_band).

`reward.accuracy_margin_barrier(mu)`:

* **mu ≥ MARGIN_REF** (comfortable headroom) → `0`. Cost reward alone decides
  among comfortable P3 configs.
* **0 ≤ mu < MARGIN_REF** → `SAT·(log(mu+eps) − log(MARGIN_REF+eps))`, a ≤0
  restoring penalty whose slope `SAT/mu → ∞` as the margin thins. Because the
  cost reward rises with fusion while this term falls steeply near the boundary,
  `cost(fusion) + barrier(margin(fusion))` has an interior maximum at a POSITIVE
  margin → the policy is pushed back instead of overshooting.
* **mu < 0** (violated, P1) → `b0 − VIO·(−mu)`, continuous at `mu=0`, LINEAR (not
  exp) so it never flattens over the realistic collapse depth: a collapsed policy
  always sees a gradient toward feasibility → recovery, which the flat -6.95
  cliff did not provide. Clamped to `acc_barrier_floor` (−10).

Defaults (RewardWeights, all tunable): `acc_barrier_enabled=True`,
`sat_scale=0.5`, `margin_ref=0.25` (≈1.8 probe-sigma headroom — THE
aggressiveness knob; server sweep {0.15, 0.25, 0.35}), `vio_scale=0.30`,
`floor=-10`, `eps=1e-3`. `acc_barrier_enabled=False` falls back to the ADR-012
near-miss path (kept for A/B and the NearMissGradedTierTest).

**Invariants preserved (the barrier rewrites only the PPO scalar):**
`terminal_priority`, the hard-priority rank key, candidate-store, and
best-selection are bit-identical. The violated barrier stays < the P3 tier floor
(40) and P1 never receives cost, so cost can never offset an accuracy violation
even in the scalar (mental-model item 7). Invalid episodes keep the legacy
invalid_term shaping (their metrics are unreliable). The barrier is a pure
deterministic function of the metrics → 1==N byte-identity is unaffected.

Also added: per-block-TYPE fusion split (`fusion_count_b2/b4/b5`) in
episodes.jsonl, and a server-side collapse watchdog (sustained P3≈0 → kill the
training PID; the best is already checkpointed periodically). ADR-012's
borderline-retest, epsilon floor, and policy-K probes are RETAINED (the retest
denoises the margin the barrier reads — more useful now).

## Alternatives considered

- **De-weight block4 fusion cost** to fix the inversion directly: the user chose
  to keep the 80:150:130:40 ratios and let the accuracy barrier do the work.
- **Re-steepen the ADR-012 near-miss tier** (4th attempt within that structure):
  rejected — three reward-shaping tweaks have now over/under-corrected; the
  log-barrier is the principled mechanism with a built-in restoring force and is
  the one Stage-2-adjacent thing (Stage-1) that demonstrably trains well.
- **Bigger probe to resolve the knife-edge 22-fusion regime stably**: deferred.
  The barrier deliberately settles at a probe-resolvable headroom (MARGIN_REF);
  reaching the sub-sigma high-fusion optimum stably is a probe-resolution
  problem, the documented next lever if more fusion is wanted.

## Consequences

- Reward scale near the accuracy boundary changes again; reward values are NOT
  comparable across this ADR. Judge runs by: a healthy (non-collapsing) curve,
  fusion stabilizing at a positive-margin level (not 0, not 35), P3 fraction
  staying > 0, and best ≥ the no-fusion cap.
- Policy/critic shapes unchanged → checkpoint-compatible (no SEQ_RL_VARIANT
  bump). Resume works across this change.
- The stable optimum will likely adopt FEWER fusions than the knife-edge 22
  (it leaves ~1.8σ headroom by design) — this is the safety/aggressiveness
  trade controlled by MARGIN_REF, not a regression.
