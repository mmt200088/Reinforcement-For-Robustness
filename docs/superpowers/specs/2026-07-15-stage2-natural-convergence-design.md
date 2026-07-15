# Stage-2 Natural Convergence Design

## Status

Accepted by the user on 2026-07-15. This document amends the entropy and
termination section of the 2026-07-14 Stage-2 K convergence design. The cost,
credit assignment, factorized PPO, robust constraints, candidate promotion,
and final revalidation protocols remain unchanged.

## Objective

Formal Stage-2 layerwise PPO must optimize only the constrained reward. It must
not be pushed toward or away from entropy by a schedule, floor, recovery term,
or fixed training horizon. Training continues until the learned policy itself
provides convergence evidence.

This is an operational convergence contract, not a mathematical guarantee of
the global optimum. A completed search may claim only the best robust-feasible
configuration found under the tested action space, followed by the existing
post-search local-neighborhood revalidation protocol.

## Entropy Contract

The active layerwise PPO path uses:

```text
ent_coef = 0.0 for every PPO update
```

Entropy remains available for diagnostics and termination only. The following
mechanisms must not affect action sampling or gradients:

- entropy cosine decay;
- entropy coefficient floors;
- per-slot entropy recovery;
- negative entropy penalties;
- forced low-entropy sampling or deterministic annealing.

The existing full-support initial Block4 and K distributions remain. Policy
concentration must therefore come from reward and PPO updates.

## Termination Contract

`stage2-search-episodes=0` means an unbounded formal search. There is no
30,000-episode minimum, 60,000-episode stop, or automatic extension budget.
The loop checks convergence after every complete PPO update window and exits
only when all conditions hold:

1. normalized Block4 entropy is below `0.1`;
2. normalized truncation-K entropy is below `0.1`;
3. at least one promotion-qualified robust-feasible candidate exists;
4. the best robust-feasible cost frontier has not improved for 100 complete
   PPO update windows.

The 100-window criterion is convergence patience, not a training budget. A
frontier improvement, retraction, or loss of the current feasible frontier
resets the patience exactly as in the existing tracker. PPO attempts skipped or
interrupted by non-finite minibatches do not advance this patience.

Positive episode counts remain supported only as bounded smoke/test budgets.
Exhausting such a budget is not convergence and must not be reported as a
completed formal search unless the same convergence conditions happened to be
met.

## Resume And Persistence

An unbounded run resumes from the exact checkpoint episode without converting
the run into a remaining-episode calculation. Policy, optimizer, PPO auxiliary
state, RNG state, candidate evidence, frontier state, update count, and entropy
diagnostics remain checkpointed. A checkpoint already marked converged may
return immediately after its candidate frontier is revalidated.

The algorithm contract records:

- `termination.mode = natural_convergence`;
- `episode_limit = null` for formal unbounded runs;
- the two entropy thresholds and 100-window frontier patience;
- `entropy_regularization.kind = disabled` and coefficient `0.0`.

The algorithm revision is bumped so checkpoints produced under the old cosine
entropy/fixed-horizon contract cannot be resumed silently.

## Launcher Contract

The MRPC Stage-2 preset and default Stage-2 CLI value use
`--stage2-search-episodes 0`. CLI parsing accepts zero and rejects negative
values. The active layerwise robust path is the only formal path allowed to
interpret zero as natural convergence; legacy paths must receive a positive
debug budget.

## Verification

Focused tests must prove:

1. every layerwise PPO update receives `ent_coef_override=0.0`;
2. convergence no longer depends on 30k/60k episode thresholds;
3. an unbounded loop continues past update windows and stops on convergence;
4. bounded smoke runs still stop at their explicit budget without a false
   convergence claim;
5. resume preserves natural-convergence state and a valid converged checkpoint
   can return without another episode;
6. CLI, evaluator, runner, preset, manifest, checkpoint, and summary agree that
   zero means natural convergence;
7. legacy unsupported zero-budget paths fail clearly;
8. the focused Stage-2 regression suite and synthetic factorized bandit remain
   green.
