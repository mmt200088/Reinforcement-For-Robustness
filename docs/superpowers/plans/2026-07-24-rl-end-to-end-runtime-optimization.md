# RL End-to-End Runtime Optimization Plan

> **Scope:** Runtime and hardware efficiency only. Preserve actions, rewards,
> probe trials, RNG state, metrics, PPO inputs/updates, candidate decisions,
> checkpoints, and scientific conclusions.

## Baseline

- Aggregate source: `accc27d65b6e99435913c3f1bed8629e755742e8`
- Stage-1 hot path: one full `validation_full` BERT evaluation per episode.
- Stage-2 hot path: 12 layer decisions, 59 per-block replans, one five-trial
  terminal probe, statistical assessment, candidate persistence, and PPO every
  120 episodes.
- Already accepted: candidate promotion index, bounded histories, buffered
  structured writers, process-backed shared F1/F4 probe pool, small GTrXL,
  Block3/Block5 fused CUDA paths, and `batch_size=64`.
- Already rejected: KV cache, lockstep rollout batching, and terminal probe
  batch sizes 128/256.

## Success Contract

1. Same source inputs and seed produce identical Stage-1 action/reward streams,
   final configurations, metrics, PPO update summaries, and post-run RNG state.
2. Same source inputs and seed produce identical Stage-2 action matrices,
   trial seeds/values, terminal metrics/rewards, promotion states, PPO update
   summaries, candidate records, and post-run RNG state.
3. Optimization is accepted only when server A/B shows lower end-to-end wall
   time or lower critical-path component time without a throughput regression.
4. Local, Git remote, and server source must resolve to the same commit and tree
   before the final A/B.

## Task 1: Stage-2 Read-Only Optimizer Cache Hits

**Files**

- Modify: `rescale_optimizer_bridge.py`
- Modify: `blb_stage2_rl/optimizer_cost.py`
- Modify: `blb_stage2_rl/env.py`
- Modify: `blb_stage2_rl/sequential_env.py`
- Test: `tests/test_rescale_optimizer_bridge_cache.py`
- Test: `tests/test_blb_action_materialization.py`

**Implementation**

- Add an internal read-only cache-hit API that returns a fresh top-level payload
  while borrowing immutable nested optimizer JSON.
- Keep the public mutation-isolated `evaluate()` behavior unchanged.
- Route only canonical in-process materialization consumers that never mutate
  optimizer output through the read-only API.
- Preserve LRU ordering, hit/miss counters, cache diagnostic fields, output
  parsing, and materialization order exactly.

**Verification**

- Red test: internal read-only hits do not recursively clone nested JSON.
- Public cache mutation-isolation test remains green.
- Materialized configuration fingerprint, signals, replan application, and
  serialized diagnostics match the public path exactly.
- Server microbenchmark covers 59 per-block hits plus terminal materialization.

## Task 2: Stage-1 Device-Resident Validation Batches

**Files**

- Modify: `layer_importance_evaluator.py`
- Modify: `stage1_rl/parallel_runner.py`
- Test: `tests/test_stage1_eval_accel.py`
- Test: `tests/test_stage1_parallel_runner.py`

**Implementation**

- Cache already-collated deterministic non-training batches once per CUDA
  device.
- Reuse the device-resident tensors for every Stage-1 episode on that device.
- Keep batch boundaries, order, dtype, model call arguments, loss aggregation,
  logits aggregation, and metrics unchanged.
- Do not preload the mutable/lazy training split and do not use this cache in
  any noisy Stage-2 path.

**Verification**

- Red test: repeated worker evaluations perform one host-to-device population,
  not one transfer per episode.
- Compare labels/logits/loss/metrics byte-for-byte against the host-batch path.
- Run current-source Stage-1 1-GPU and four-healthy-GPU A/B with identical
  rollout signatures and PPO summaries.

## Task 3: Layerwise Fixed Metadata and Assessment Reuse

**Files**

- Modify: `blb_stage2_rl/layerwise_env.py`
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/statistical_constraints.py`
- Test: `tests/test_blb_layerwise_env.py`
- Test: `tests/test_blb_layerwise_runner.py`
- Test: `tests/test_blb_statistical_constraints.py`

**Implementation**

- Reuse the immutable layer schedule and static observation prefix across
  resets while retaining fresh episode-owned state arrays.
- Remove redundant deep copies of newly-owned diagnostic payloads without
  exposing mutable internal state.
- When pooled evidence is exactly the just-computed online trial group, retarget
  the already-computed constraint assessment to the promotion gate instead of
  rerunning the same deterministic bootstrap.

**Verification**

- Snapshot all observations, info payloads, terminal handoff fields, assessment
  fields, and mutation-isolation behavior before/after.
- Server component benchmark reports reset/step/assessment time separately.

## Task 4: End-to-End Gates

- Run focused tests first, then the complete Stage-1 and Stage-2 suites.
- Replay known baseline failures against the unchanged aggregate commit.
- Run at least one full PPO window for Stage-1 and Stage-2 on the four healthy
  GPU UUIDs, recording GPU utilization, memory, CPU utilization, component
  timing, wall time, and structured-output equality.
- Re-run the Stage-2 probe test with five workers after GPU 3 is repaired.
- Reject any optimization that changes a semantic output or fails to improve
  the measured critical path.

## Deferred Pending Evidence

- Stage-1 process-isolated workers: potentially high impact, but requires a
  dedicated deterministic A/B because it changes process scheduling.
- Duplicate probe worker on one healthy GPU for `K=5` with four physical GPUs:
  benchmark only; do not make default without exact trial/RNG parity and a real
  wall-time win.
- CUDA fusion for remaining Block1/2/4 operations: profile first; do not add
  kernels without a demonstrated terminal-probe contribution and bitwise parity.
