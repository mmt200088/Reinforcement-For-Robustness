# ADR-016: Headroom-coupled cost + linear recovery gradient (fix the ADR-015 fusion runaway)

Date: 2026-06-16
Status: Accepted
Refines: ADR-015 (continuous bounded reward). Does NOT change the action space, the
offline fusion maps, determinism/episode-parallelism, strict feasibility selection,
the cosine-entropy schedule, the loss-mean selection gate, or the ADR-014 debug
instrumentation. `reward_design="tiered"` (ADR-014 rollback) is untouched.

## Context

ADR-015's continuous reward was run as the 5th 60k (`77ffdc0`; the 1==N gate PASSED
— the determinism fix held). It STILL hot-collapsed: watchdog killed it at 42180 for
"12 consecutive windows P3<2%". Forensics on the real artifacts
(`stage2_grid_gate_60k_20260615_170844/long60k/run/`) found three compounding,
code-confirmed mechanisms:

1. **Cost lure dominates the feasible region.** `scalar = barrier_raw/NORM +
   W_cost·cost_frac`, with `CONT_W_COST=4.0` ≫ `CONT_W_ACC=1.0`. The satisfied
   barrier `0.5·log(margin)/20` is ≈ −0.02 to −0.17 — negligible vs the cost. Real
   ep4259 (P3): `cost_score=2.43` ≫ `acc_barrier/NORM≈−0.02`. So in P3 the reward is
   ≈ pure cost → the policy pushes fusion right up to the accuracy boundary.
2. **Knife-edge boundary.** Cost is P3-gated (`effective_cost_score=0` when not P3),
   so crossing into P1 cliffs the cost from ~2.4 to 0.
3. **Flat violated plateau, zero recovery gradient.** The violated barrier
   `−10·exp(−margin·20)` saturates for margin < ~−0.25 → the reward clip flattens it
   to −5. Real data: a MILD violation (ep8459, m1=0.84) and a CATASTROPHIC one
   (ep35000/41000, m1=0.63) earned the IDENTICAL terminal_reward=−5.0. With no
   gradient between "slightly over" and "collapsed", a single overshoot is permanent.

Net effect (health log): healthy until ep~7000 (P3≈92%, fusion≈18), then a one-way
ratchet — cost lure pulls fusion up, overshoots into the flat −5 region, can't climb
back → fusion ratcheted 18→24→30→36 and froze at max (reward=−6.941) until the
watchdog fired. The saved best (strict selection) was healthy (fusion≈19, P3,
loss≈0.33); only the training trajectory collapsed.

User decision (AskUserQuestion): **refine the continuous reward** so the landscape has
a stable interior optimum at a SAFE margin + a recovery gradient — not "only a
recovery gradient" (optimum still on the knife-edge) and not a structural fusion cap.

## Decision (`blb_stage2_rl/reward.py`)

### Fix A — headroom-coupled cost (kills the knife-edge, creates a stable interior optimum)
In `_continuous_reward`, scale the cost reward by the worst-margin headroom:
```
worst_overall = min(acc_margins ∪ std_margins)
headroom      = clip(worst_overall / CONT_COST_HEADROOM_MARGIN_REF, 0, 1)
scalar        = barrier_raw/NORM + W_cost · cost_frac · headroom
```
`headroom` → 1 at a safe margin, ramps smoothly to 0 as the worst margin → 0 (the
boundary), and is 0 for any violation (margin<0 ⇒ no cost ⇒ item 7 holds on top of
the upstream P3-gate). So pushing fusion toward the boundary now LOSES cost reward —
a restoring force — and the optimum sits at the highest fusion that keeps a safe
margin, with NO cliff (cost fades continuously through the boundary).
`CONT_COST_HEADROOM_MARGIN_REF=1.0`.

### Fix B — linear violation penalty (recovery gradient, kills the freeze)
In `stage1_log_barrier`, replace the violated branch with a LINEAR penalty:
```
violated (m<0):   CONT_BARRIER_VIOLATION_SLOPE · m       # m<0 → negative; deeper → strictly lower
satisfied (m≥0):  CONT_BARRIER_SATISFACTION_SCALE · log(m + 1e-5)   # unchanged
```
A constant gradient across the realistic violation range means a milder violation
always scores strictly higher than a deeper one → the policy has a gradient to climb
back out of P1. Bounded below only by the reward clip (engages only at extreme
depth). `CONT_BARRIER_VIOLATION_SLOPE=4.0` (deep margin ≈ −23 ⇒ ≈ clip_min after
/NORM). `CONT_BARRIER_STEEPNESS` is now unused by the continuous path.

### Calibration is data-driven, not guessed
Constants were calibrated by an **offline reward-landscape replay** over the 5th
run's real 42180 episodes (recorded `(margin, fusion, cost, priority)` → new reward).
The chosen defaults give: interior **peak at fusion≈18** (the observed healthy
region), monotone decline past it (f18 +0.73 → f24 −0.49 → f36 −2.81 — the OLD reward
went UP f18→f36), and a recovery gradient (P1 mild −0.36 ≫ P1 deep −2.88, no −5
plateau). This offline-validate-before-60k step is itself a process fix (the prior 5
attempts each shipped and only discovered the collapse after ~13h).

## Invariants
- item 7: violation ⇒ headroom=0 ⇒ cost=0; priority/rank/selection unchanged.
- 1==N: reward is still a pure deterministic function of the recorded metrics/margins.
- Bounded: clip[−5,+5] preserved; amplitude unchanged.
- Policy/critic shapes unchanged → checkpoint-compatible (no SEQ_RL_VARIANT bump);
  the new constants are scalar reward shaping. Reward NOT comparable across ADR-016.

## Consequences
- Expect the next 60k to STABILIZE at a feasible moderate fusion (≈ peak, ~92% P3),
  with a smooth small-amplitude reward curve and no watchdog collapse. Fusion harvest
  may be modest — the honest result of an interior, safe-margin optimum.
- Knobs: `CONT_COST_HEADROOM_MARGIN_REF` (raise → peak at a safer/lower fusion, fewer
  P1; lower → higher peak/more harvest, closer to the boundary) and
  `CONT_BARRIER_VIOLATION_SLOPE` (steepness of the recovery gradient). Tune via the
  offline replay, not blind.
- Locks: `tests/test_blb_continuous_reward.py::ADR016LandscapeTest`.
