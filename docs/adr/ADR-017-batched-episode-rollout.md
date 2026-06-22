# ADR-017: Batched episode rollout (replaces KV-cache rollout)

- **Status**: **Rejected / Reverted 2026-06-22 — see ADR-018.** Server validation
  (artifacts `stage2_5gpu_speed_60k_20260622_131933`) found the batched forward
  *does* amortize in isolation (3.96× at B=4) but end-to-end throughput was
  **1.0000× (NOT EFFECTIVE)**: the K=5 terminal probe is the critical path and the
  rollout already overlaps the sibling worker's probe under `workers_per_device=2`,
  so shrinking rollout wall-time does not shrink episode wall-time. The change was
  reverted (it cost bit-exact 1==N for no end-to-end gain). Kept as history.
- **Date**: 2026-06-21
- **Supersedes**: the KV-cache rollout (`--blb-v3-kv-cache-rollout`, 2026-06-19) — retired, kept default-OFF.

## Context

The Stage-2 episode-parallel rollout (`collect_fusion_episode`) is ~42% of the per-episode
wall (the K=5 terminal probe is the other ~58%). The 2026-06-19 KV-cache attempt tried to cut
the rollout's GTrXL forward from O(H²) to O(H) FLOPs, but the server measured it **0.60×
(slower)**. Forensics (code + benchmark):

1. The ON path ran **two** incremental forwards per step (current pass with K/V discarded +
   `commit_kv_cache` re-running every block to grow the cache), vs OFF's one fused
   `nn.MultiheadAttention` forward.
2. At H≤59 / d_model=256 / 4 layers the forward is **launch-bound, not FLOP-bound** (~5 ms/step
   for trivial FLOPs). Reducing FLOPs buys ~nothing; the hand-written attention's extra small
   kernels + the double forward cost *more*.
3. The forward is only ~8% of the rollout (benchmark: 298 ms forward vs 3.63 s rollout) — the
   mandatory profiling that would have shown this was skipped, and the per-step timers lacked
   `cuda.synchronize`, so the premise ("rollout is forward-bound") was never validated.

KV-cache was the wrong lever for a launch-bound regime.

## Decision

Batch the per-step GTrXL forward **across a worker's episodes**. All episodes traverse the
identical, episode-independent step schedule, so a worker's B episodes can advance in **lockstep**
and have their B current observations stacked into ONE `[B, state_dim]` forward per step — the
launch-bound cost amortized ~B×. This also amortizes the per-step host/tensor construction.

Implementation (`blb_stage2_rl/`):

- **`_fusion_episode_generator`** (`parallel_runner.py`): the per-episode logic as a generator
  that **yields** the per-step forward inputs and **receives** `(logits, safe_logits, value)`.
  Because GTrXL logits are **action-independent**, ONE yield per step suffices — sampling, the
  forced anchor, the fusion probe, rejection retries, and the fallback all reuse the same logits
  (no extra forwards). `collect_fusion_episode` (serial, B=1, forward inline) and
  `collect_fusion_episodes_batched` (B=W, batched forward) drive the SAME generator, so they are
  equivalent by construction.
- **Batch-invariant seeded sampler** (`sequential_policy.py`): `forward_and_mask` (forward +
  mask, no sampling) + `sample_from_logits` (per-row inverse-CDF on the masked probs using
  `numpy.default_rng(seed)` uniforms) + `logprob_from_logits`. The only randomness is the per-row
  numpy seed (`derive_policy_step_seed`), so a row's draw is identical for ANY batch size / GPU
  count. The distribution is unchanged (Categorical over masked logits incl. the ADR-012 ε
  mixture), so `evaluate_action` (PPO replay) log-probs stay consistent. This **replaces** the
  global `manual_seed` + `multinomial` rollout sampling — which also removes its device-lock
  fragility (the seeded sampler is lock-free; the terminal probe still uses `env.probe_device_lock`).
- **B seq_envs share one base_env** (model) per worker — no extra model copies. Per-step rollout
  touches only each env's own accumulator; the terminal probe (the only base mutation) runs
  serially per row within the worker's single thread. `probe_noise_seed` is set
  per-step-before-commit so the shared base carries the right per-episode seed.
- Terminal probe stays per-episode (distinct noise configs cannot share a forward).

Opt-in `--blb-v3-batched-rollout` (default OFF) + `--blb-v3-rollout-profile` (cuda-synced
per-step timing). Threaded rl_tune → evaluator → `BLBStage2TrainConfig.batched_rollout_enabled`
→ `run_window`.

## Determinism / quality gate (honest: float-equivalent, NOT bit-exact)

A batched GEMM differs from per-episode at ~1e-6, so this **drops bit-exact 1==N** (the user
dropped it for KV-cache 2026-06-19 and reaffirmed it here by choosing "go straight to batched
rollout"). The gate becomes:

1. **Batch-invariance self-test** (`tests/test_blb_batched_rollout.py`, torch): the seeded
   sampler gives a row identical actions/log-probs in a batch vs alone; `forward_and_mask`
   batched matches per-row within 1e-4; sampler reproduces the categorical distribution; masked
   levels are never drawn.
2. **Device-invariance** (SERVER_COMMAND): 1-GPU vs N-GPU with batched ON — reward / priority /
   fusion distributions match (float-equivalent).
3. **Distribution-equivalence**: the sampling distribution is unchanged (Categorical), so RL
   behavior is statistically identical; the specific seed→action mapping changes (like a
   re-seed), so **reward values are NOT comparable across this change**.

## Real-speedup validation (the point)

Short A/B (batched OFF vs ON) measuring per-episode wall + ep/h + the synced rollout-phase wall.
Must measure **EFFECTIVE** before the default flips ON. Honest expectation: W = interval /
num_workers (e.g. 60/10 = 6) → forward launch amortized ~6×; the probe (~58%) is untouched, so
the whole-episode gain is bounded (~15–30%, not 2×). W can be raised via interval/worker config
for more amortization.

## Consequences

- KV-cache rollout retired (the per-step forward is hoisted to the driver; the cache code path
  in `collect_fusion_episode` is removed; the flag stays default-OFF as inert).
- PPO update / replay path unchanged (each episode is still one trajectory). Item-7 / reward
  shaping unaffected (this is a throughput change). Checkpoints unaffected.

## Amendment 2026-06-21 (round-1 A/B was confounded; two real bugs fixed)

The first server A/B (artifacts `stage2_batched_rollout_validate_20260621_185832`, commit
`d9e4777`) reported **quality MATCHED but speedup 0.97× (NOT EFFECTIVE)**. That verdict was **not
trustworthy** — the experiment was confounded, and the batched path had a real perf bug:

1. **Asymmetric measurement.** The A/B ran OFF with `--blb-v3-rollout-profile 0` (the serial
   forward timer is async / undercounts) and ON with `1` (an explicit `cuda.synchronize()` around
   the batched forward). The two `policy_rollout_wall_seconds` numbers measured different things,
   so 0.97× could not detect a real speedup either way. **Fix:** `collect_fusion_episode` now
   takes a `profile` flag and applies the same `cuda.synchronize()`; the A/B runs **both** sides
   `profile=1`.
2. **Batched path disabled truncation.** `_parse_state` gated the causal-prefix truncation to
   `B == 1`, so the batched driver ran with `truncate_to_current=False` — processing all `H=59`
   tokens every step versus the serial path's ~`t+1`. That ~2.4× extra token work per step
   partially cancels the `B×` launch amortization. In lockstep all `B` rows share `current_step`,
   so truncation is exact (the causal mask zeroes tokens `> current_step`). **Fix:** `_parse_state`
   truncates to the batch-max `current_step` for any `B`; the batched driver uses
   `truncate_to_current=True`. Locked by `test_truncate_batched_equals_full_lockstep`
   (truncated == full forward within 1e-4 for `B>1`).

**Process fix (the meta-cause, shared with KV-cache):** neither KV-cache nor the first batched
A/B first *proved, in isolation under symmetric sync*, that the per-step forward is the dominant
rollout cost and that batching reduces it. ADR-017 (and the KV-cache plan) required that
profiling and it was skipped both times → wasted server rounds. Added
`scripts/blb_rollout_profile.py`: a torch-only micro-benchmark (no model / Rescale_optimizer /
GLUE) that builds the real production policy and measures per-episode forward wall **serial vs
batched(B) vs batched-no-truncate** under symmetric `cuda.synchronize()`, sweeping `B` and
printing the speedup **at the real operating point** `B ≈ rollout_size(32) / (NGPU ×
workers_per_device) ≈ 4`. SERVER_COMMAND now runs the profiler as the **decisive** step before the
A/B; if the forward is `~1.0×` at the real `B`, batching is the wrong lever (like KV-cache) and is
abandoned without spending a 60k.
