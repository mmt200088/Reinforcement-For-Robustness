# RL End-to-End Runtime Optimization Evidence

Date: 2026-07-24

## Scope

This artifact records runtime-only optimizations applied on top of aggregate
commit `accc27d6`. The retained changes preserve actions, rewards, trial seeds,
metrics, candidate decisions, PPO inputs/updates, and public mutation-isolation
contracts.

The benchmarked optimized Stage-2 implementation is `6ed9eb10`. Final source
commit `ede0896a` retains that Stage-2 implementation; the intervening commits
only revert the rejected Stage-1 device-batch experiment and its tests.

## Retained Optimizations

1. Read-only internal Rescale optimizer cache access avoids repeated deep
   copies while public callers keep isolated mutable results.
2. Fixed Stage-2 metadata, observations, and already-computed assessment
   probabilities are reused instead of being rebuilt or recomputed.
3. In inference mode only, BLB noise addition reuses the freshly sampled noise
   tensor as the `torch.add` output and caches immutable noise standard
   deviations. Autograd/training behavior is unchanged.

## Stage-2 End-to-End A/B

The production MRPC path ran 180 episodes with five terminal-probe trials split
one per GPU across `cuda:0..4`.

| Measurement | Aggregate `accc27d6` | Optimized | Improvement |
| --- | ---: | ---: | ---: |
| Wall time | 412 s | 391 s | 1.0537x |
| Throughput | 1572.816 ep/h | 1657.289 ep/h | 1.0537x |
| Terminal probe mean | 0.819 s/ep | 0.785 s/ep | 1.043x |

The gate's historical labels are generic: `one_*` is the optimized run and
`many_*` is the aggregate baseline. The gate reports:

- quality/effect equality: PASS, 180/180 episodes
- strict diagnostic equality: PASS
- PPO update equality: PASS, 2/2 updates

All five GPUs reached 100% sampled peak utilization. Mean sampled utilization
was 78.74%, 80.84%, 81.48%, 71.87%, and 71.44% for GPUs 0 through 4. GPU 3
executed all 180 probe episodes and its assigned 180 trials.

## Optimization Decomposition

On the same optimized commit, disabling only the inference noise fast path
changed wall time from 391 s to 401 s while preserving 180/180 episode and 2/2
PPO equality. Therefore:

- read-only cache plus fixed-data reuse: `412 / 401 = 1.0274x`
- inference noise fast path: `401 / 391 = 1.0256x`
- combined retained improvement: `412 / 391 = 1.0537x`

In `noise_fastpath/`, `one_*` is fast path ON and `many_*` is fast path OFF.

## Rejected Changes

- A reusable noisy-weight workspace produced no wall-time benefit
  (391 s versus 391 s) and changed a trial/PPO trajectory, so it was reverted.
- Keeping Stage-1 validation batches resident on each GPU preserved rollout
  signatures and final policy, but regressed 170-episode wall time from
  50.538933 s to 55.504209 s. It was reverted.
- Larger terminal batches, GTrXL KV caching, and lockstep rollout batching were
  already measured as ineffective or slower and remain disabled.

Stage-1 device-batch parity evidence includes identical rollout signatures
`8ace808d83126260` and `84d2e0cb7f19175d`, and identical best GELU
`[2,2,2,1,1,1,2,1,4,1,1,1]` with Softmax fixed to 6.

## Artifact Layout

- `final_vs_aggregate/`: end-to-end retained code versus aggregate baseline.
- `noise_fastpath/`: same-commit fast-path ON/OFF isolation.
- `stage1_rejected/`: extracted parity/timing summary and wall files for the
  reverted Stage-1 experiment.
- `raw_artifact_sha256.txt`: hashes of full server-side episode/PPO records.
- `raw_artifact_line_counts.txt`: record counts for those full artifacts.
- `git_provenance.txt`: benchmark source commits and trees.
- `verification.md`: focused and broad regression results.

Large raw episode files remain in server runtime artifacts and are represented
here by hashes and line counts rather than duplicated in Git.
