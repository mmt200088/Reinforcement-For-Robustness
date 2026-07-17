# Stage-2 Multi-Fidelity Validation And Practical Convergence Design

## Goal

Keep PPO interaction affordable while making every authoritative Stage-2 claim
on the complete validation set. The online policy continues to learn from a
fixed stratified 256-example probe with five noise trials. Candidate promotion,
the robust frontier, convergence, and final selection use `validation_full`
with 25 independent trials and a separately calibrated baseline.

The search stops only when its authoritative objective has stabilized. Episode
count and policy entropy are diagnostics, not stopping gates or budgets.

## Evidence Tiers

### F1: online reward evidence

- Dataset: fixed stratified 256-example subset sampled from the configured
  validation split.
- Trial group: five deterministic, independent noise trials per episode.
- Purpose: PPO reward, advantage estimation, and inexpensive candidate
  prefiltering.
- Constraint reference: a 25-trial baseline measured on the same probe.
- Authority: diagnostic and optimization signal only. F1 cannot promote a
  candidate, enter the strict frontier, trigger convergence, or support a final
  scientific claim.

### F4: authoritative validation evidence

- Dataset: every example in `validation_full` (408 examples for MRPC).
- Trial target: 25 deterministic, independent noise trials per candidate.
- Purpose: promotion at probability 0.80, strict-frontier ranking,
  convergence, and final reporting.
- Constraint reference: a separate 25-trial baseline measured on
  `validation_full`.
- Authority: F4 is the only evidence accepted by the strict frontier.

F1 and F4 identity contexts are distinct. Raw trials, seeds, bootstrap
assessments, means, and standard deviations must never be pooled across tiers.

## Promotion Flow

1. PPO evaluates every sampled action on F1 and receives the existing robust
   constrained reward.
2. A candidate becomes eligible for F4 only when it is P3 on F1, passes the
   online probability gate, and has strictly greater variable-cost score than
   the current F4 frontier.
3. The promotion evaluator installs the exact same action, fusion group, K
   values, and boosted overrides on a dedicated `validation_full` probe runner.
4. It collects 25 F4 trials from scratch. No F1 trials count toward this total.
5. The candidate enters the strict frontier only when all six precision and
   stability probabilities pass the 0.80 gate against the F4 baseline.

The online and promotion evaluators may share the primary model serially, but
each owns its own batch set and multi-GPU replicas. Before and after an F4
evaluation, the online persistent-install cache is cleared so a shared primary
model cannot incorrectly skip the next install.

## Objective-Driven Convergence

Natural convergence has no minimum episode count and no maximum episode cap.
The future stopping contract is:

- at least one candidate has passed all six precision and stability constraints
  on F4;
- the maximum robust-feasible variable-cost score has not improved for 100
  finite PPO updates;
- the exact deterministically selected F4 action has not changed for the same
  100 finite PPO updates;
- entropy remains diagnostic only and cannot force or prevent convergence;
- once the plateau conditions hold, the selected action is evaluated again on
  `validation_full` with 25 fresh trials that do not reuse its promotion trials;
- termination is `converged` only if that fresh assessment passes all six
  constraints at the final 0.95 probability gate.

If the fresh revalidation fails, the candidate is removed from the current
strict frontier, the deterministic winner and both stability counters are
reconciled, and PPO continues. A bounded smoke run still stops at its explicit
episode count and is reported as `bounded_budget_exhausted`, never as natural
convergence.

The strict winner is unique. Candidates are ordered by: all constraints
passing, maximum variable-cost score, lexicographically strongest sorted vector
of all six constraint probabilities (worst constraint first), lexicographically
largest sorted vector of all six normalized safety margins (worst constraint
first), lexicographically smallest full action vector, then candidate identity
only as an unreachable duplicate fallback. Equal-cost candidates remain
eligible for F4 promotion so discovery order cannot bypass these tie-breakers.

## Persistence And Resume

- The algorithm revision and run-context hash change because old checkpoints
  pooled F1 and pseudo-F4 probe evidence.
- New checkpoints persist the 100-update patience, termination reason, both
  stability counters, and final-revalidation evidence/status.
- F1 and F4 records carry fidelity-specific identity contexts.
- A resume rebuilds the strict frontier only from promoted F4 evidence and the
  F4 baseline reference.
- Final revalidation writes a durable pass/fail verdict onto the base F4
  identity. A pass restores from the fresh final-evaluation context at the 0.95
  gate; a failure remains excluded after restart and cannot be silently
  re-admitted from the earlier 0.80 promotion record.
- Old revision checkpoints fail closed rather than being silently resumed.

## Current Run Handling

The already-running pre-change job is not retrofitted or restarted. Its frozen
probe-selected strict best and fusion-zero baseline are independently evaluated
with the existing Paean merged final-evaluation module on `validation_full`, 25
trials each. The report labels the training selection as probe-derived and the
new full-validation comparison as the authoritative final result.

## Verification

- Unit tests prove F1 and F4 trial groups remain disjoint even for the same
  action and seed values.
- Promotion tests prove the first five F1 trials are not counted toward the 25
  F4 trials and that F4 uses a distinct evaluator/reference.
- Restore tests prove only promoted F4 evidence reconstructs the frontier.
- Convergence tests cover the 99/100-update boundary, frontier/action resets,
  non-finite updates, strict revalidation pass/fail, and absence of episode
  count gates.
- Runner tests prove the complete validation loader has no probe-size cap and
  that the algorithm contract records both evidence tiers.
- The current run report is generated only after Paean records 25
  `validation_full` trials for both baseline and frozen best.
