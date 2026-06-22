# ADR-018: Revert the rollout-speedup experiments (KV-cache + batched)

- **Status**: Accepted
- **Date**: 2026-06-22
- **Supersedes**: ADR-017 (batched episode rollout) — Rejected/Reverted. Also retires
  the KV-cache rollout (2026-06-19, no standalone ADR).

## Context

Two attempts were made to speed up the Stage-2 episode-parallel **fusion rollout**
(`collect_fusion_episode`), which a profile had put at ~42% of the per-episode wall
(the K=5 terminal probe is the other ~58%):

1. **KV-cache incremental forward** (`--blb-v3-kv-cache-rollout`, 2026-06-19):
   `SequentialGTrXLBlock.forward_incremental` + `IncrementalRolloutCache` +
   `commit_kv_cache`, turning the per-step O(H²) full-prefix rebuild into O(H).
2. **Batched lockstep rollout** (`--blb-v3-batched-rollout`, 2026-06-21, ADR-017):
   advance a worker's episodes in lockstep and stack their current observations into
   ONE `[B, state_dim]` GTrXL forward per step (launch-bound cost amortized ~B×), with
   a batch-invariant seeded sampler replacing `torch.manual_seed` + `multinomial`.

Both deliberately **dropped bit-exact 1==N determinism** (KV-cache: a hand-written
attention reimpl ~1e-6; batched: batched GEMM ~1e-6 + the sampler RNG swap, which
also changed reward comparability on the **default serial path**).

## Decision

**Remove both.** Server measurements show neither delivers an end-to-end speedup, so
the determinism cost buys nothing:

- **KV-cache: 0.60× (SLOWER).** At H≤59 / d_model=256 / 4 layers the forward is
  **launch-bound, not FLOP-bound**, so O(H²)→O(H) buys nothing while the hand-written
  attention adds small kernels.
- **Batched: 1.0000× end-to-end (NOT EFFECTIVE).** The isolated forward profiler
  (`scripts/blb_rollout_profile.py`) confirmed the forward *does* batch — 3.96× at B=4 —
  but the end-to-end A/B over 600 episodes was throughput-flat. The decomposition shows
  why: the **K=5 terminal probe (~5.2 s/ep) is the critical path**, and the rollout
  (~3.7 s/ep) already **overlaps** with the sibling worker's probe under
  `--stage2-workers-per-device 2`. Shrinking rollout wall-time therefore does not shrink
  episode wall-time. (Quality was MATCHED, reward 1.2027 ON==OFF — the change wasn't
  *wrong*, just useless.)

Evidence: `experiments/server_command_runs/stage2_5gpu_speed_60k_20260622_131933/`
(`rollout_profile.txt`, `batched_ab_verdict.txt`).

## What was removed

- `blb_stage2_rl/parallel_runner.py` restored to its pre-speedup state (commit `a866d5f`):
  original inline-forward serial rollout; no `collect_fusion_episodes_batched` /
  `_fusion_episode_generator`.
- `blb_stage2_rl/sequential_policy.py` restored to its pre-speedup state (commit `21fd371`):
  no `forward_incremental` / `IncrementalRolloutCache` / `commit_kv_cache` /
  `forward_and_mask` / `sample_from_logits`; the rollout samples via
  `torch.manual_seed` + `multinomial` again. **Bit-exact 1==N restored.** (Parameter set
  unchanged → existing checkpoints resume fine.)
- Flags deleted end to end: `--blb-v3-kv-cache-rollout`, `--blb-v3-batched-rollout`,
  `--blb-v3-rollout-profile` (launcher init/case/cmd, `rl_tune.py`,
  `layer_importance_evaluator.py`, `BLBStage2TrainConfig` fields + wiring).
- Deleted scaffolding: `scripts/blb_rollout_profile.py`, `scripts/blb_kvcache_ab_compare.py`,
  `scripts/blb_kvcache_benchmark.py`, `tests/test_blb_batched_rollout.py`,
  `tests/test_blb_rollout_ab_compare.py`, `tests/test_blb_kvcache_rollout.py`.

`run_window`'s signature and the `sequential_runner.py` call site are unchanged
(`sequential_runner.py` was never touched by the speedup commits), so the restore is a
drop-in revert.

## Consequences

- The episode-parallel fusion path is **fully deterministic again** (bit-exact 1==N), so
  the device-/worker-count invariance gate applies as before, and reward values are again
  comparable to pre-speedup runs.
- The in-flight ADR-016 reward 60k was already running with batched OFF, so production was
  never affected; future runs use the restored deterministic sampler.
- **Lesson for future throughput work:** the per-episode bottleneck is the **terminal
  probe**, which is already K-split across GPUs. The rollout is not on the end-to-end
  critical path under `workers_per_device≥2` (it overlaps the sibling's probe), so
  rollout-only optimizations cannot move the wall clock. Target the probe instead. And:
  *profile the target in isolation under symmetric `cuda.synchronize` AND confirm it is on
  the end-to-end critical path before building a speedup* — both attempts here optimized a
  component that wasn't the bottleneck.
