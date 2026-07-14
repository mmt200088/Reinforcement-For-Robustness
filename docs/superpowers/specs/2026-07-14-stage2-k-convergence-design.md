# Stage-2 K Convergence Design

## Status

Accepted by the user on 2026-07-14. This design replaces the equal-family
cost budget from the 2026-07-11 layerwise design with an explicit
per-coordinate equivalence: lowering one active truncation K by two is worth
exactly the same cost reward as changing one layer's Block4 fusion count from
zero to one.

## Evidence And Problem

The completed 60,000-episode BERT-base MRPC run was healthy but did not
converge. Block4 normalized entropy reached `0.2070`, while K entropy remained
`0.9207`. The best robust-feasible cost stopped improving at episode 22,680.
The implementation and run artifacts show four algorithmic causes:

1. the old cost formula made one Block4 toggle worth about 24.6 one-bit K
   reductions;
2. all cost reward arrived at the terminal transition, so the five K choices
   made at each layer shared a noisy twelve-step outcome;
3. PPO clipped the product of all active slot probability ratios, allowing one
   slot to clip the update for the other slots in the same layer;
4. the entropy schedule had a permanent `0.012` floor, so the six-way K heads
   were still explicitly rewarded for remaining stochastic at episode 60,000.

The run had no invalid-action resurgence, no collapse sentinel, and normal PPO
KL/clip diagnostics. The fix therefore targets objective scaling, credit
assignment, and convergence pressure rather than model/config installation.

## Cost Contract

For the 12 Block4 decisions and 59 active K decisions, define raw cost units:

```text
U_fusion = sum(Block4_fusion[layer])
U_k      = 0.5 * sum(13 - K[layer, block])
U        = U_fusion + U_k
U_max    = 12 + 0.5 * 59 * (13 - 8) = 159.5
C        = U / U_max
```

`C` remains in `[0, 1]`. A Block4 `0 -> 1` change and any K decrease of two
both change `U` by exactly one and `C` by exactly `1 / 159.5`. This literal
per-coordinate equivalence means K accounts for 147.5 of 159.5 maximum raw
units (92.48%); that is an intentional consequence of the requested cost
model, not a separate family weight.

Block2 and Block5 fusion remain fixed and do not enter learnable cost.

## Reward Redistribution

The precision-first, stability-second, cost-third terminal ordering remains:

```text
invalid: -5
P1:      -3.0 + 0.5 * precision_boundary
P2:      -1.5 + 0.5 * stability_boundary
P3 base: +1.0 + 0.0005 * (precision_boundary + stability_boundary)
```

For P3 only, decompose `C` into twelve layer-local terms and add each term to
the transition that selected that layer's actions. P1, P2, and invalid
episodes receive no cost term. The terminal transition receives the P3 base
without another copy of `C`.

With `gamma = lambda = 1`, the sum of rewards in a P3 episode remains exactly
`P3 base + C`, so candidate ordering and reported episode reward do not change.
The redistribution only lowers policy-gradient variance and mirrors Stage-1's
dense per-layer cost credit.

## Factorized PPO

The environment remains a 12-step layerwise environment. Each step still
samples one six-slot factorized MultiDiscrete action. The PPO surrogate changes
from one joint ratio to one ratio per active slot:

```text
ratio[layer, slot] = exp(new_logp[layer, slot] - old_logp[layer, slot])
```

The shared constrained return advantage is broadcast to active slots and PPO
clipping is applied independently per slot before averaging over active slots.
The critic remains scalar and continues to predict the total step return.
This matches the Stage-1 credit shape more closely: each categorical decision
has its own clipped actor update, while all decisions still learn from the same
precision/stability outcome.

Old per-slot log probabilities are captured during rollout. Masked Layer-0
Block1 K contributes no ratio, entropy, or loss. Existing non-layerwise callers
retain summed-log-prob PPO behavior.

## Entropy And Convergence

Layerwise PPO keeps the initial safe prior and the cosine exploration phase,
but its entropy coefficient has no positive lower bound. It decays to zero by
the planned episode horizon and remains zero for convergence extensions.
Per-slot entropy recovery stays disabled.

Convergence remains evidence-based:

1. at least 30,000 episodes;
2. Block4 normalized entropy `< 0.1`;
3. K normalized entropy `< 0.1`;
4. a strict robust-feasible candidate exists;
5. its cost frontier has not improved for 100 PPO updates.

A 60,000-episode limit alone is not convergence. If either action family is
still diffuse, the run requests an extension with zero entropy bonus.

## Persistence And Reporting

The structured writer and checkpoint add:

- fusion, K, and total raw/normalized cost units;
- per-layer redistributed cost rewards;
- joint and factorized PPO diagnostics;
- Block4 and K entropy coefficients;
- the existing complete 12-layer fusion/K best-action table.

Checkpoint resume must preserve behavior-policy per-slot log probabilities only
inside the active rollout buffer; completed update windows remain unchanged.

## Verification

Focused tests must prove:

1. K `13 -> 11` and Block4 `0 -> 1` have identical normalized increments;
2. all 59 K slots participate and Layer-0 Block1 remains masked;
3. local P3 cost rewards sum exactly to `C`, while P1/P2 receive zero;
4. reward redistribution preserves total episode return;
5. per-slot ratio/clipping ignores masked slots and cannot cross-clip sibling
   actions;
6. the layerwise entropy schedule reaches and remains exactly zero;
7. a synthetic all-feasible layerwise bandit drives both Block4 and K entropy
   below `0.1` and selects the known cost optimum;
8. existing Stage-2 reward, action, persistence, and non-layerwise PPO tests
   remain green.

Server validation uses a short controlled smoke first. A new formal 60k run is
started only after the focused convergence test and runtime smoke pass.
