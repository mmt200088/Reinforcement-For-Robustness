# Stage-2 Dual-Resource Reward Design

## Status

Accepted by the user on 2026-07-17. This design supersedes the scalar
`fusion1_khalf_per_bit_v1` objective from the Stage-2 K-convergence design.
It changes only the RL resource objective, credit assignment, strict candidate
ranking, convergence identity, and their persisted diagnostics. It does not
change action installation, fusion maps, truncation levels, model evaluation,
precision or stability constraints, PPO architecture, or probe trial counts.

## Problem

The active layerwise action has 12 learnable Block4 fusion decisions and 59
learnable truncation-K decisions. The superseded objective assigns one raw cost
unit to Block4 fusion `0 -> 1` and one raw unit to K `13 -> 11`:

```text
U_fusion = sum(Block4_fusion[layer])
U_k      = 0.5 * sum(13 - K[layer, block])
C_old    = (U_fusion + U_k) / 159.5
```

This converts compute and communication savings into one exchange rate. It also
lets K contribute 147.5 of the 159.5 maximum units (92.48%) because there are
59 K coordinates but only 12 Block4 coordinates. A feasible policy can therefore
obtain most of its scalar cost reward by lowering K while leaving every
learnable Block4 fusion decision at zero. That behavior is consistent with the
old objective, but it is not the intended physical objective:

- Block4 fusion reduces homomorphic-compute work.
- Lower K reduces MPC communication volume.
- Neither resource has a justified fixed conversion into the other without a
  deployment compute/network model.

## Considered Objectives

### Selected: deployment-agnostic robust balance

Treat compute and communication as independent normalized resource axes. Select
the strict-feasible action that maximizes the weaker axis first, then the total
relative progress. This is the max-min solution for an unknown deployment mix:
when the relative value of compute and communication is not known, a solution
cannot hide an unimproved resource behind arbitrarily large savings in the other.

### Deferred alternative: scenario-weighted physical latency

Retain this as an explicit future option, but do not implement it now. Once
representative measurements exist, rank a strict-feasible candidate under a
deployment scenario `theta` using a physical latency/cost estimate such as:

```text
L(action; theta)
  = compute_time(action; hardware)
  + communication_bytes(action) / bandwidth(theta)
  + communication_rounds(action) * RTT(theta)
```

An equivalent normalized scalar may use measured, not guessed, scenario weights:

```text
C_theta = w_compute(theta) * F + w_communication(theta) * C
```

WAN scenarios will generally assign more value to communication savings; LAN
scenarios will generally assign less. The strict-feasible two-dimensional Pareto
archive from the selected design can later be rescored under these scenarios.
If the archive covers the relevant frontier, this does not require retraining.
Conditioned or weight-swept Pareto training remains a possible follow-up only if
the observed archive lacks the required extremes.

### Rejected for the current objective: compute-first lexicographic order

Maximizing fusion before considering K would correct the present K preference,
but it would silently assume compute is always more valuable than communication.
That assumption is no more defensible than the old fixed exchange rate.

## Independent Resource Axes

For the current 12-layer BERT-base action space, define:

```text
F = sum(Block4_fusion[layer]) / 12
C = sum(13 - K[layer, block]) / (59 * (13 - 8))
```

Both values are in `[0, 1]`, but they retain different meanings:

- `F` is relative learnable compute saving.
- `C` is relative learnable communication saving.

The normalization describes progress within each resource; it is not an
exchange rate between the resources. Fixed Block2/Block5 fusion is reported in
the effective configuration but excluded from `F` because it is constant across
all policy actions and cannot rank learnable candidates.

Define the exact selection objectives:

```text
B = min(F, C)        # robust floor: improve both resources
S = (F + C) / 2      # secondary progress among equal robust floors
```

## PPO Cost Surrogate

PPO still requires one bounded scalar return. Use a scalar packing that follows
the exact selection order closely without restoring a meaningful exchange rate:

```text
eta   = 1e-4
C_ppo = (B + eta * S) / (1 + eta)
```

`C_ppo` is in `[0, 1]`. The realizable `B` values come from multiples of `1/12`
and `1/295`; because 12 and 295 are coprime, the smallest nonzero gap between
distinct realizable values is at least `1/3540`, which is greater than `eta`.
Therefore no full-range change in `S` can outweigh a positive realizable
improvement in `B` inside the cost surrogate.

The existing constraint order remains unchanged:

```text
invalid: -5
P1:      precision constraint failure; no resource credit
P2:      precision passes, stability fails; no resource credit
P3:      1 + C_ppo + 0.0005 * (precision_signal + stability_signal)
```

The six loss/metric1/metric2 precision and stability gates remain authoritative.
The small boundary signal remains PPO safety shaping inside P3. Final candidate
selection uses the exact tuple `(B, S)` rather than the packed reward, so safety
shaping cannot redefine the reported optimum.

## Resource-Aware Credit Assignment

The current factorized PPO correctly gives each action slot its own behavior
ratio and actor advantage, but its cost credits come from the old additive unit
model. Replace those credits with a two-family Shapley decomposition of
`V(F, C) = C_ppo`:

```text
phi_F = 0.5 * (V(F, 0) - V(0, 0))
      + 0.5 * (V(F, C) - V(0, C))

phi_C = 0.5 * (V(0, C) - V(0, 0))
      + 0.5 * (V(F, C) - V(F, 0))
```

Shapley efficiency gives `phi_F + phi_C = C_ppo`. Distribute each family credit
only within its own family:

```text
fusion_slot_credit_i = phi_F * fusion_contribution_i / F, when F > 0
k_slot_credit_j      = phi_C * communication_contribution_j / C, when C > 0
```

Zero family saving produces zero family credit. Layer-0 Block1 K remains masked
and receives zero credit. Slot credits sum to `C_ppo`; their per-layer sums are
used for critic reward redistribution, preserving the exact episode return.

For P3, each factorized actor advantage is the shared terminal constraint return
plus that slot's resource credit. For P1, P2, and invalid episodes, every resource
credit is zero and all factors see only the shared constraint outcome. Thus:

- K slots cannot claim deterministic Block4 compute credit.
- Block4 slots cannot claim deterministic K communication credit.
- Once one resource is substantially ahead, its additional marginal credit is
  limited by the max-min objective instead of continuing to substitute for the
  weaker resource.

## Strict Candidate Selection And Pareto Archive

Only candidates that pass strict repeated-evaluation precision and stability
gates enter the robust frontier. Select one deterministic best candidate in this
order:

1. maximize `B = min(F, C)`;
2. maximize `S = (F + C) / 2`;
3. maximize the ordered six constraint probabilities;
4. maximize the ordered six normalized safety margins;
5. use the full action-vector lexicographic order and candidate identity as
   deterministic final tie-breakers.

In parallel, retain every strict-feasible non-dominated `(F, C)` candidate. A
candidate dominates another when it is no worse on both axes and strictly better
on at least one. Reports must show the selected robust best and the full
compute/communication Pareto frontier. This archive is the handoff point for the
deferred scenario-weighted physical-latency option.

## Convergence

Convergence remains evidence-based and does not force policy entropy downward.
Replace the scalar-cost plateau identity with the exact resource objective:

- a strict-feasible selected candidate exists;
- its exact `(B, S)` objective has not improved for 100 PPO updates;
- its deterministic selected action has not changed for 100 PPO updates;
- final strict repeated evaluation still passes all six constraints.

Block4 and K entropy remain diagnostics, not termination gates. The final action
is stable because selection is deterministic even when several policy modes are
equivalent.

## Persistence And Compatibility

Persist enough information to audit and redraw the objective without inference:

- `compute_saving`, `communication_saving`, `robust_floor`, and
  `secondary_progress`;
- packed `ppo_resource_score` and both family Shapley credits;
- per-layer and per-slot resource credits;
- the strict `(B, S)` rank key and two-dimensional Pareto records;
- the complete 12-layer Block4/K configuration and all six constraint results.

Use a new stable cost-model revision, `dual_resource_maxmin_shapley_v1`, and a
new layerwise algorithm/checkpoint revision. Include `eta`, axis denominators,
credit mode, and exact selection order in the algorithm contract hash and
candidate identity.

Old checkpoints must fail closed before policy, optimizer, or append-only state
is changed. In particular, the gracefully stopped v5 run remains resumable only
under its original source and reward contract. Training with this design starts
a fresh run; it must not relabel the v5 optimizer state as the new objective.

## Verification

Implementation is accepted only when focused tests prove:

1. changing K cannot change `F`, and changing Block4 fusion cannot change `C`;
2. all 12 Block4 and 59 active K coordinates contribute to the correct axis;
3. any realizable `B` improvement outranks every possible `S`-only improvement
   in `C_ppo`;
4. family Shapley credits are finite, nonnegative, and sum exactly to `C_ppo`;
5. slot credits remain within their resource family, exclude inactive slots,
   and sum through layers to the scalar episode resource reward;
6. P1, P2, and invalid episodes receive no compute or communication credit;
7. strict ranking follows `(B, S, confidence, margin, deterministic action)` and
   the Pareto archive removes only truly dominated candidates;
8. convergence resets on either objective improvement or selected-action change
   and requires final strict revalidation;
9. the checkpoint contract rejects every pre-design revision without mutating
   persistent state;
10. a synthetic constrained bandit learns a policy that improves both resource
    axes instead of exhausting only the larger K action family;
11. structured JSONL and HTML/report fixtures expose both physical resource axes
    and the complete selected configuration;
12. existing precision, stability, action-installation, factorized PPO, and
    multifidelity-validation tests remain green.

Server validation starts with a controlled smoke that checks resource-axis and
credit telemetry, then uses a new long run for convergence evidence. A short
smoke can verify plumbing but cannot establish that the stochastic search found
the final optimum.
