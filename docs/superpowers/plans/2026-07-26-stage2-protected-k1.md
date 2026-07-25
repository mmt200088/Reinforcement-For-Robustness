# Stage-2 Protected K=1 Implementation Plan

## Goal

Reduce terminal-probe work without allowing a one-trial stochastic estimate to
select, promote, or certify a Stage-2 candidate.

## Contract

- K=1 may reject only an extreme precision failure.
- Stability is never inferred from one trial.
- A candidate that is near the boundary, is not strictly dominated on the
  compute/communication resource frontier, or is selected for deterministic
  audit receives the original exact K=5 evaluation.
- The one-trial reject probability never exceeds the reward's P1 boundary
  (`0.5`), even if an experiment configures a stricter online probability gate.
- Only exact K=5 evidence may enter the candidate store, promotion path, best
  selection, convergence state, or final certification.
- Follow-up trials use the original trial indices `1..4`, so a protected K=5
  result has the same trial seeds and metrics as the existing K=5 path.
- The feature is opt-in and emits enough structured telemetry to measure reject
  rate, audit false negatives, trial savings, and realized wall-time impact.
- An audited precision false negative permanently fails open to exact K=5; the
  latch and cumulative counters are checkpointed and survive resume.

## Steps

1. Add pure protected-K1 decision and audit helpers with unit tests.
2. Add exact trial-index subset execution to the thread and process probe
   runners, preserving existing dense K behavior.
3. Add a two-stage prepared-terminal evaluator that runs trial 0, then trials
   1..4 for all protected candidates.
4. Keep K1-only rejects out of candidate evidence and record explicit episode
   telemetry.
5. Wire opt-in configuration through the evaluator and Stage-2 runner.
6. Run focused tests, compile checks, historical replay, and Git/server parity
   verification.

## Historical Calibration

The replay source is
`server_backups/20260720_stage2_mrpc_ep114240_full_recovery/archives/diagnostics_episodes.jsonl.gz.part*`.
It compares each episode's first raw trial with its recorded precision limits,
normalizing by the pooled baseline standard deviation (the recorded standard
deviation limit divided by the run's `2.0` stability multiplier).

On all 114,240 episodes, a `4 sigma` one-trial precision guard had zero observed
K=5 P3 false negatives but screened only 20 episodes. The idealized
terminal-forward speedup is approximately `1.00014x` before two-stage scheduling
overhead, so this implementation prioritizes scientific safety and measurement
rather than claiming a material speedup.

| Guard | Screened episodes | K=5 P3 false negatives | Ideal terminal speedup with 2% audit |
|---:|---:|---:|---:|
| 4 sigma | 20 / 114,240 | 0 | 1.00014x |
| 3 sigma | 340 / 114,240 | 5 | 1.00234x |
| 2 sigma | 3,089 / 114,240 | 949 | 1.0217x |
| 1 sigma | 17,120 / 114,240 | 8,598 | 1.133x |

The frontier protection implemented here can only reduce the screen rate from
these upper bounds. With the safe 4-sigma default, actual end-to-end throughput
is therefore expected to be unchanged or slightly lower due to the extra
scheduling stage.
