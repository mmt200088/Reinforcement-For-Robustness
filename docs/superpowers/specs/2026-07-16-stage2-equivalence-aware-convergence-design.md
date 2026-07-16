# Stage-2 Equivalence-Aware Convergence Design

## Status

Accepted by the user on 2026-07-16. This document replaces the entropy gates
in `2026-07-15-stage2-natural-convergence-design.md`. Reward, PPO updates,
candidate promotion, and entropy regularization remain unchanged.

## Objective

Training must stop when the constrained optimization result is stable, not
when every factorized categorical head arbitrarily selects one member of a
cost-equivalent action class. The selected output must still be one exact,
deterministic 12-layer Block4/K configuration.

## Selection

Promotion-qualified candidates retain the existing strict ordering:

1. highest variable cost saving;
2. highest worst constraint probability;
3. lowest loss mean;
4. highest metric1 mean;
5. highest metric2 mean;
6. candidate key as a deterministic final tie-break.

The selected candidate key is the stable identity used by convergence and
resume validation.

## Termination

After each finite PPO update, an unbounded run converges only when:

1. a promotion-qualified robust-feasible candidate exists;
2. its robust-feasible cost frontier has not improved for 100 counted updates;
3. the strictly selected exact action has not changed for 100 counted updates.

Frontier loss/retraction and selected-action changes reset their respective
counters. Non-finite or skipped PPO updates may reconcile changed state but do
not advance either counter. Block4 and K entropy remain persisted diagnostics
and never affect gradients, sampling, ranking, or termination.

## Resume

The checkpoint persists the current selected candidate key and its stability
counter. Restored promoted candidates are revalidated before training resumes.
A missing or different selected key resets action stability and clears a stale
converged flag. Old entropy-gated checkpoints are incompatible through an
algorithm revision bump.

## Verification

Tests prove high or unavailable diagnostic entropy cannot block objective
convergence, a same-cost winner change prevents convergence, state round-trips,
frontier retraction remains safe, strict ties are deterministic, and bounded
smoke semantics remain unchanged.
