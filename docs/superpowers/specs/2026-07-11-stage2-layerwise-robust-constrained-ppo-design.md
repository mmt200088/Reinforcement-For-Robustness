# Stage-2 Layerwise Robust Constrained PPO Design

## Status

Accepted in conversation on 2026-07-11.

## Goal

Find the lowest-compute BERT-base MRPC Stage-2 configuration that remains
statistically equivalent to the all-fusion-zero baseline within the accepted
precision and stability constraints.

The Stage-1 prerequisite is fixed to GELU degree 4 and Softmax degree 6 in every
layer. Stage-2 fixes Block2 and Block5 to fusion count 1, lets PPO choose Block4
fusion count 0 or 1 per layer, and lets PPO choose every active truncation K.

The optimizer remains PPO. The design changes the decision granularity,
statistical evidence, reward ordering, credit assignment, cost normalization,
candidate promotion, and final selection. It does not change the fusion maps,
noise installation semantics, Rescale Optimizer semantics, or model metrics.

## Two Distinct Baselines

The implementation must not conflate metric calibration with the lowest-cost
point that is reachable by the policy.

### Metric baseline

- GELU: `[4] * 12`
- Softmax: `[6] * 12`
- Block2, Block4, Block5 fusion count: all 0
- Every active truncation K: 13

This configuration defines all precision and stability thresholds.

### Search cost origin

- GELU: `[4] * 12`
- Softmax: `[6] * 12`
- Block2 and Block5 fusion count: fixed to 1
- Block4 fusion count: 0
- Every active truncation K: 13

Block2 and Block5 fusion savings are constant throughout the policy domain.
They remain visible in total-cost diagnostics and reports, but are excluded
from the learnable cost normalization and PPO cost reward.

## Layerwise Action Architecture

The existing runtime is per-block sequential with 47 policy steps. The new
runtime is per-layer sequential with exactly 12 policy steps, matching the
earlier Stage-1 layerwise decision pattern.

Every layer uses the same canonical five-slot layout:

| Slot | Decision | Domain |
|---|---|---|
| 0 | Block4 fusion count | `{0, 1}` |
| 1 | Block1 truncation K | configured `K_LEVELS` |
| 2 | Block2 truncation K | configured `K_LEVELS` |
| 3 | Block4 truncation K | configured `K_LEVELS` |
| 4 | Block5 truncation K | configured `K_LEVELS` |

Layer 0 has no Block1, so slot 1 is masked and contributes neither log
probability nor entropy. Block3 remains outside the Stage-2 action space and
stays at its baseline configuration.

At one policy step, all active decisions for that layer are sampled jointly
from factorized categorical heads. The joint log probability is the sum of the
active slot log probabilities. The environment then resolves the fixed
Block2/5 fusion options, resolves the selected Block4 option, installs all K
choices for the layer, and replans the active blocks. Optimizer signals are
aggregated into one layer-step observation before the next layer is sampled.

The policy receives exact layer and block-slot identities from the layerwise
schedule. It must not derive identities from the retired 59-step formula. This
removes the current mismatch where 46 of 47 effective steps receive the wrong
layer or block embedding.

## Baseline Statistical Reference

Baseline calibration runs five independent seed groups with five noise trials
per group, for 25 raw trials total. For each trial the system persists loss,
metric1, metric2, group index, trial index, and noise seed.

For each channel:

- the baseline location is the pooled 25-trial mean;
- the baseline scale is the pooled unbiased standard deviation (`ddof=1`);
- deterministic bootstrap samples represent uncertainty in both quantities.

The precision tolerance remains 0.1 percent relative:

```text
candidate_loss_mean <= baseline_loss_mean * 1.001
candidate_m1_mean   >= baseline_m1_mean   * 0.999
candidate_m2_mean   >= baseline_m2_mean   * 0.999
```

The stability tolerance is 200 percent of the baseline scale:

```text
candidate_loss_std <= 2.0 * baseline_loss_std
candidate_m1_std   <= 2.0 * baseline_m1_std
candidate_m2_std   <= 2.0 * baseline_m2_std
```

The historical fixed `0.01` stability floor is removed. If any baseline channel
has zero or non-finite variance after 25 trials, calibration adds groups of five
up to 50 total trials. If the channel remains statistically degenerate, training
fails before PPO starts and reports the raw evidence instead of silently
weakening or tightening the constraint.

## Candidate Statistical Assessment

An online candidate initially runs five trials. A pure NumPy statistical module
uses deterministic, precomputed bootstrap index banks to propagate baseline and
candidate sampling uncertainty. It returns six independent feasibility
probabilities:

```text
P(loss precision passes)
P(m1 precision passes)
P(m2 precision passes)
P(loss stability passes)
P(m1 stability passes)
P(m2 stability passes)
```

The gates are channelwise. Averaging cannot allow one metric to compensate for
another.

- Online PPO gate: every applicable probability must be at least 0.50.
- Promotion gate: every applicable probability must be at least 0.80.
- Final strict gate: every applicable probability must be at least 0.95.

The candidate store keeps raw trials by effective action hash. A repeated
action appends fresh evidence instead of permanently reusing its first
five-trial aggregate.

## Strictly Ordered Reward

Define:

```text
P_precision = min(P_loss, P_m1, P_m2)
P_stability = min(P_loss_std, P_m1_std, P_m2_std)
tau = 0.5
B(p) = clip(log((p + eps) / (tau + eps)), -1, 1)
```

The online priority order is precision first, stability second, cost third:

```text
invalid: -5
P1:      -3.0 + 0.5 * B(P_precision)
P2:      -1.5 + 0.5 * B(P_stability)
P3:      +1.0 + C + 0.0005 * (B(P_precision) + B(P_stability))
```

This creates disjoint scalar ranges, so every P3 reward is greater than every
P2 reward, and every P2 reward is greater than every P1 reward. P1 and P2 never
receive cost reward. The bounded log-barrier preserves a smooth restoring
signal near each boundary without allowing safety headroom to outweigh even
the smallest discrete cost improvement inside P3.

## Learnable Cost Objective

Only policy-controlled cost variation enters the PPO objective:

```text
F = mean(Block4 fusion_count over 12 layers)
T = mean((13 - K) / (13 - 8) over all 47 active K slots)
C = 0.5 * F + 0.5 * T
```

Thus Block4 fusion and truncation receive equal total budget. Actual K values,
not categorical indices, determine truncation saving. Higher Block4 fusion and
lower K always increase `C`.

The existing per-block weights remain available for total-cost reporting, but
fixed Block2/5 fusion contributions cannot consume the learnable fusion reward
budget. Within the strict feasible set, final selection orders candidates by
`C` first, not by the mixed training reward.

## PPO And Credit Assignment

The layerwise episode has terminal metric reward and no unconditional dense
cost shaping. To avoid position-dependent credit attenuation, PPO uses
undiscounted Monte Carlo GAE:

```text
gamma = 1.0
gae_lambda = 1.0
```

All 12 layer actions therefore receive the same terminal outcome before their
state-value baselines are subtracted.

Initial policy probabilities keep full action support while preferring the
known-safe area without forcing baseline episodes:

```text
P(Block4 fusion=0/1) = 0.60 / 0.40
P(K=13/12/11/10/9/8) = 0.50 / 0.20 / 0.12 / 0.08 / 0.06 / 0.04
```

The K prior is assigned by decoded K value, independent of the legacy
`K_LEVELS` index order. No forced-baseline phase, neighbor curriculum, fusion
probe schedule, or epsilon exploration scaffold is enabled.

Initial training settings are:

```text
learning rate       = 5e-5
PPO clip range      = 0.2
PPO epochs          = 4
rollout/update      = 120 episodes
gamma               = 1.0
GAE lambda          = 1.0
```

KL early stop, gradient clipping, advantage normalization, and the existing
cosine entropy schedule remain enabled.

## Promotion, Convergence, And Final Selection

Training runs for at least 60,000 episodes. Runs of a few hundred episodes are
startup checks only and are never treated as convergence evidence.

A P3 candidate on the current cost frontier is promoted from five to 25 trials.
The training data writer persists the action, all raw trials, six feasibility
probabilities, thresholds, priority, reward components, Block4 choices, every K
choice, entropy, PPO diagnostics, and throughput fields.

Convergence excludes fixed or one-option action slots. It requires:

1. at least 30,000 completed episodes;
2. normalized Block4 entropy below 0.1;
3. normalized K entropy below 0.1;
4. no improvement in the robust feasible best cost for 100 PPO updates.

If the criteria are not met at episode 60,000, training continues in 12,000
episode increments rather than declaring convergence by episode count alone.

After training:

1. Rank candidates by learnable cost and take the top 20.
2. Accumulate each to 25 probe trials and require all six probabilities >= 0.95.
3. Evaluate the top five strict candidates and the metric baseline on
   `validation_full`, using five seed groups times five trials.
4. Select the highest-cost-saving candidate that passes all six full-validation
   constraints. Break equal-cost ties by minimum feasibility confidence, then
   loss, metric1, and metric2.
5. Audit every one-coordinate neighbor of the winner: 12 Block4 flips and up
   to five alternate K values for each of 47 K slots, at most 247 neighbors.
   Screen at five trials, promote plausible lower-cost candidates to 25 trials,
   and repeat until no strict-feasible one-coordinate cost improvement exists.

This final audit establishes a tested local optimum. PPO plus the audit cannot
mathematically prove the global optimum of the full combinatorial space; the
report must state that limitation rather than overclaiming.

## Verification Contract

Focused tests must prove:

1. the episode has 12 layer steps and the canonical five-slot layout;
2. layer 0 masks Block1 K and all other layers expose all four active K slots;
3. Block2/5 fusion is fixed to 1 and Block4 remains binary;
4. baseline calibration consumes 5 x 5 raw trials and uses `ddof=1`;
5. zero-variance calibration extends to 50 then fails loudly if unresolved;
6. loss, metric1, and metric2 independently gate precision and stability;
7. P3 rewards are always greater than P2, and P2 greater than P1;
8. fixed Block2/5 savings do not change learnable cost reward;
9. lower K and higher Block4 fusion monotonically increase cost reward;
10. terminal credit is undiscounted across all 12 layer actions;
11. final selection prefers lower compute among strict-feasible candidates;
12. raw trials, probabilities, layer actions, PPO diagnostics, and best state are
    mirrored under the project-root `rl_training_data_points/` tree.

Server verification consists of focused tests, a short integrity smoke that is
not interpreted as a quality result, and then the full convergence run. The
final HTML report must show baseline and winner distributions, reward and
entropy curves, all six constraint results, and a compact per-layer table of
Block4 fusion count plus Block1/2/4/5 truncation K.
