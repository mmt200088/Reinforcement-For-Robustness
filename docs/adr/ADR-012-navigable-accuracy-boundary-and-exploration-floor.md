# ADR-012: Navigable accuracy boundary (graded near-miss + borderline retest) + exploration floor + policy-K probes

Date: 2026-06-12
Status: Accepted
Supersedes: ADR-011's probe design (baseline-K forcing, b2→b5→b4 rotation).
Amends: ADR-007's tier shaping at the P1 boundary (tiers/selection unchanged).

## Context

The second 60k fusion run (artifacts
`stage2_grid_gate_60k_20260612_004130`, with ADR-011's split budget / no
option prior / probes / relaxed tolerances all active) STILL converged to
fusion=0. The forensics overturned the previous diagnosis:

1. **The split budget works.** On-policy P3 fusion episodes beat their ±40
   no-fusion neighbors in EVERY phase: +0.081 (ep<2k), +0.130 (2k-6k),
   +0.194 (6k-10k), +0.108 (10k-17k); positive for single flips and
   multi-flips alike. The reward ranking was no longer the problem.
2. **The killer is the P1 cliff tax.** Of 1327 P1 episodes, 1226 were
   on-policy fusion episodes and ALL of them were borderline
   (m1 ∈ [0.833, 0.858], ZERO catastrophic); no-fusion episodes produced
   exactly ONE P1 in 60k. Fusion+deep-K pushes true m1 to the tolerance
   boundary, and the 256-sample probe's quantization noise (σ≈0.0018,
   16 distinct m1 values) stochastically drops it below the threshold —
   each touch costing the full −46 gulf. Expected fusion advantage:
   +0.117 − 8.4%×46 ≈ −3.8. P1 rate scaled with fusion count (fc2 4.0%,
   fc3-4 6.8%, fc5-7 19.6%, fc8+ 38.1%). The policy killed fusion
   RATIONALLY; on-policy attempts (14606 of them!) stopped at ep 16495.
3. **All three ADR-011 probes were useless or harmful.** Forcing baseline K
   cancelled the fusion gain against the policy's learned deep-K savings:
   b2 probe netted +0.07 (≈ noise), b5 probe netted **−0.86** (it actively
   taught that fusion is bad), and the 12-layer b4 probe was a guaranteed
   catastrophic P1 (−7.23, 100/100) hammering anti-fusion generalization.
4. **The policy froze.** From PPO update ≈700 (ep ≈42k) to the end:
   entropy 0.000, clip_fraction 0.000 — literally no learning for the last
   18k episodes (saturated softmax ⇒ vanishing entropy gradient; the
   entropy-recovery mechanism cannot resurrect a fully collapsed head).

The user pointed at Stage-1 RL (which trains well) as a reference: its
boundary is a smooth log-barrier and its exploration never dies. This ADR
imports exactly those two properties without replacing the tier backbone.

## Decision

1. **Graded near-miss tier** (`RewardWeights.near_miss_tier_cap=35,
   near_miss_tier_floor=15, near_miss_band=1.0`): a metric fail that is NOT
   invalid and whose worst per-channel deficit ≤ band (in units of
   |baseline − threshold|) earns `tier = 35 − 20×deficit_norm` instead of
   the cliff. Priority stays 1 — selection and candidate ranking are
   bit-identical (hard-priority rank keys untouched); near-miss cap 35 is
   strictly below the P3 floor 40, so cost can never offset an accuracy
   violation in the PPO scalar either (mental-model item 7 holds). Beyond
   the band (and for catastrophic fails like 12-layer b4 fusion) the old
   cliff is unchanged. The boundary region becomes a gradient ("back off a
   notch") instead of a minefield.
2. **Borderline retest** (`BLBStage2EnvConfig.borderline_retest_enabled`,
   multiplier 2): a metric fail within the near-miss band triggers ONE fresh
   re-measurement with 2×num_trials on a salted deterministic probe seed
   (golden-ratio XOR of the episode-keyed `probe_noise_seed` → disjoint
   trial stream); the retest verdict replaces the first. False-fail
   probability drops roughly quadratically (a true-within-tolerance config
   at 1.4σ above threshold: 8% → ~0.2%) while true violators keep failing.
   Only active on the deterministic probe path; pure function of the
   episode → 1==N preserved. Episodes log a `borderline_retest` info dict.
3. **Probe v2 — force the option, inherit the policy** (supersedes
   ADR-011's probe): a probe episode now runs the NORMAL sampling path
   (curriculum mask included) and only (a) injects the target block's
   option-1 level into the mask, (b) overrides the sampled option to 1 on
   target-block steps, re-evaluating log_prob/value for the modified action
   under the same mask. K and all other blocks follow the current policy —
   the probe measures "what if you ALSO fused block-type T on top of what
   you already do", worth ≈ +1.4 standing advantage instead of ≈ 0.
   Rotation is `(2, 5)`: block4 is dropped (its 12-layer probe is a
   guaranteed catastrophic fail teaching only anti-fusion; selective
   per-layer block4 fusion is left to ε exploration under the graded
   boundary).
4. **Exploration floor** (`fusion_exploration_epsilon=0.05` option slot,
   `0.02` K slot; `--blb-v3-fusion-exploration-epsilon`): the policy's
   per-slot categorical becomes the mixture π' = (1−ε)π + ε·Uniform(masked
   support) in BOTH sampling and evaluation (PPO replays the identical
   mixture; ε=0 reproduces the old distribution exactly, unit-locked).
   The fusion choice can never become deterministic again: ~0.9 option
   flips per episode keep flowing forever, and entropy/clip can never pin
   at exactly 0.

## Alternatives considered

- **Raise the fusion bonus** so it beats the cliff tax: needs ~×40; would
  dwarf every other signal and break the tier band. Rejected.
- **Probe-quantization fix via always-larger K trials**: halving σ costs 2×
  wall on every episode; the retest pays only on borderline fails. Rejected.
- **Entropy-coefficient floor instead of ε-mixture**: the entropy gradient
  itself vanishes on a saturated softmax (observed: entropy recovered
  0.26→1.2 mid-run then crashed to exactly 0.000) — a bonus cannot
  resurrect a collapsed head; the mixture floor is mechanical. Rejected.
- **Stage-1-style dense per-step cost credit**: reverses the 2026-06-03
  grill decision (terminal P3-gated cost); deferred — only reconsider (with
  user sign-off) if this round still fails to adopt fusion with healthy
  entropy and correct probes.

## Consequences

- PPO scalar near the accuracy boundary changes scale (borderline P1 −7 →
  ~26.6 typical); reward curves are not comparable across this ADR. P1
  counts in dashboards now include benign near-miss exploration — judge by
  `near_miss` flag / fusion adoption, not raw P1 rate.
- Retest adds ~2×probe cost on borderline episodes only (a few % early,
  rare once the policy settles inside tolerance).
- Watchdog note: with the graded band, sparse near-miss P1s are exploration
  cost, not failures; sustained CATASTROPHIC P1s (beyond band) remain hard
  signals.
- Expected end-state per the cost model: block2+block5 fused (24 fusions,
  +1.78) + as much K depth as fits inside the accuracy gate, reward ≈ 41+;
  the previous best (no-fusion deep-K) caps at ≈ 39.5.
