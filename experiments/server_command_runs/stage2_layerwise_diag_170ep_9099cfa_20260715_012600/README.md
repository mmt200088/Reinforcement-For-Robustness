# Stage-2 layerwise timing diagnosis

This is a real 170-episode 1-GPU versus 5-GPU A/B run on five RTX 5090s.
Both cases used source `9099cfa38453b264a50e5d99fb7aaa67cc38acd8`, seed 42,
five online reward trials, probe size 256, rollout/PPO interval 120, and the
production layerwise robust-constrained command.

## Correctness

- All 170 episode research outputs are exactly equal after excluding timing,
  device, and execution-path bookkeeping fields.
- Both PPO updates are exactly equal.
- `corrected_equality_check.txt` records the zero-tolerance comparison.
- The original wrapper returned 2 because the comparator had not yet
  classified `terminal_probe_install_skipped` and
  `terminal_probe_clear_skipped` as diagnostic fields. Those values are
  expected to differ: the 1-GPU path clears each episode, while the 5-GPU path
  intentionally keeps persistent probe wrappers installed.

## Performance

| Measure | 1 GPU | 5 GPUs |
|---|---:|---:|
| End-to-end wall time | 582 s | 331 s |
| Throughput | 1,051.5 ep/h | 1,848.9 ep/h |
| Probe mean per episode | 2.6311 s | 1.3149 s |
| Install mean per episode | 0.0049 s | 0.0243 s |
| Cost-eval mean per episode | 0.0101 s | 0.0148 s |

End-to-end speedup is only `1.758x`. Splitting five trials over five GPUs cuts
probe wall time by only `2.001x`, not close to the ideal `5x`: one trial takes
about 0.526 seconds in the sequential case but about 1.315 seconds when five
model replicas are driven concurrently by Python threads. The 5-GPU install
path accounts for only 4.12 seconds over the whole run, so install/clear is not
the primary bottleneck. Sampled mean utilization was about 29.2%-31.8% on all
five cards, consistent with host-side concurrent dispatch contention.

The next optimization should target the same-process threaded model-forward
fanout, with persistent per-GPU processes as the leading hypothesis. Any
replacement must retain exact episode and PPO equality.

## Contents

- `one_episodes.jsonl`, `many_episodes.jsonl`: compact episode evidence.
- `one_ppo_updates.jsonl`, `many_ppo_updates.jsonl`: exact PPO evidence.
- `*_nvidia_smi.csv`, `*_gpu_utilization.*`: physical GPU samples and reports.
- `timing_summary.json`: full-run layerwise timing aggregation.
- `stage2_layerwise_diag_*log`: RED/GREEN focused test records.
- `launcher_minimum_rejection/`: the rejected 64-episode attempt, which used no
  GPU because the production launcher correctly enforces a 170-episode minimum.

Temporary checkpoints were deliberately excluded. The project-root
`rl_training_data_points/` mirrors contain the complete structured data needed
for downstream plots and audits.
