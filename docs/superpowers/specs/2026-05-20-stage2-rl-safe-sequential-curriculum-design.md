# Stage2 RL Safe Sequential Curriculum Design

## Goal

Stop the BLB Stage-2 sequential RL run from falling off the all-max baseline into
optimizer-valid but accuracy-catastrophic full actions immediately after the
forced-baseline anchor.

Treat this as a research/debugging goal, not a one-shot code change. The final
target is a Stage-2 RL training run that keeps running normally after the
anchor: the reward curve should look like a normal RL reward curve, terminal
metrics should not contain collapse sentinels such as `loss_mean=100`, priority
should not jump into sustained P1(acc), and monitored training parameters
should not show sudden pathological jumps. If one attempted fix still produces
abnormal server curves, design the next experiment, inspect the evidence, and
continue the local-edit -> git -> server-run loop until the target is met.

## Diagnosis

The bad run fails exactly at the anchor boundary. Episodes 1-120 execute the
baseline action and stay in P3(cost). Episode 121 is the first sampled-policy
episode; it remains `any_invalid=False`, so Rescale_optimizer accepts every
block, but the terminal model probe returns capped loss (`loss_mean=100`) and
MRPC accuracy around `0.3164`, so reward becomes P1(acc). Later windows remain
near `terminal_reward=-5`.

This means the invalid-action blacklist is working only for optimizer-invalid
tuples. It cannot protect the model-forward accuracy gate because that signal
appears only after the full 59-step action vector is assembled.

The first 600-episode smoke after the safe-neighbor change removed the sustained
collapse but exposed a second issue: the all-max baseline itself can
occasionally get P1(acc) with normal loss (`loss_mean≈0.34`) because the noisy
accuracy threshold is tighter than the online probe granularity. With
`stage2_probe_size=256`, one probe example is about 0.0039 accuracy; the nominal
threshold `noisy_baseline_metric1 - 0.005` was only about 0.0005 above observed
baseline jitter points. The sequential threshold therefore needs a one-sample
probe guard, not a broad reward relaxation.

## Approach

Keep the sequential formulation, but make the first policy-driven phase safe:

1. Honor the configured anchor length in the sequential path, using absolute
   episode indices so resume does not restart the anchor.
2. After the anchor, sample from a near-baseline episode curriculum instead of
   unrestricted per-step categorical actions.
3. Pick a small number of full-action-vector offsets per episode. Only those
   offsets may move to near-baseline indices; all other slots are baseline-only
   for that episode.
4. Store the exact per-level action mask used for each transition in the PPO
   buffer, so PPO ratios are computed under the same action support used during
   collection.

The curriculum uses existing BLB semantics:

- SF-like slots may move downward from the baseline index by the configured
  radius.
- K slots follow `K_LEVELS` order and keep the top local K values, rather than
  assuming categorical indices are monotonic.
- Accuracy threshold calibration subtracts a one-sample guard
  (`1 / stage2_probe_size`) from `noisy_baseline_metric1 - allowed_acc_drop` so
  baseline probe quantization does not become false P1(acc). This still leaves
  true collapses such as `m1≈0.31` far below the gate.

## Success Criteria

- Local torch-free tests prove the anchor resolver, per-step mask construction,
  and PPO buffer/mask wiring exist.
- Server contract tests pass.
- A fresh dual-GPU Stage-2 sequential smoke run reaches at least 600 episodes
  without `loss_mean=100`, without P1 collapse windows, and with a normal
  positive reward curve after the anchor.
- The server-side monitor checks reward windows, loss caps, priority labels,
  safe-neighbor diagnostics, PPO metrics, and GPU probe evidence. Any abnormal
  point triggers another diagnosis iteration instead of being treated as
  acceptable noise.
- Results and the investigation report are written as HTML and pushed through
  git from the local workspace.
