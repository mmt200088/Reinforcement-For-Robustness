# ADR-009: Stage-2 deterministic seeding + episode-parallel rollout + uniform SF grid

Date: 2026-06-10
Status: Accepted (user-approved in the 2026-06-10 grill session)
Supersedes: the K-split-only multi-GPU scheme of the "N-GPU / four-GPU reward-probe
parallelism" era for fusion mode (the K-split path itself stays available); the
2026-06-04 hybrid 2/1 SF sweep (part of ADR-008's decode notes).

## Context

Three independent findings forced this decision:

1. **Stage-2 probe noise was true-random by design.** The injected CKKS/MPC noise
   draws from dedicated per-device `torch.Generator`s seeded from `os.urandom`
   at first use and never reseeded (`function_handler._NOISE_GENERATORS`);
   `reseed_noise_rng` had zero production call sites. Every `torch.manual_seed`
   in the probe paths was inert for the noise, the recorded trial seeds were
   cosmetic, `_derive_probe_base_seed` mixed in `time.time_ns()`, and
   ProbeRunner workers called `torch.manual_seed` (which seeds ALL devices)
   concurrently — a seed race. Consequence: same command, same seed → different
   results every run; 1 GPU vs N GPUs could never match.
2. **K-split parallelism caps far below N×.** Measured on the 6000-episode 5-GPU
   A/B (2026-06-10): probe 0.548 s/ep (split 5-way, 4.97×) but GTrXL rollout
   0.239 s + install/bookkeeping ~0.23 s stay serial → ≈2.9× on 5 GPUs. Worse,
   the K=NGPU convention (4 GPUs → K=4, 5 GPUs → K=5) made results depend on
   the GPU count by construction.
3. **The SF level grid needed a semantics change** (user spec): uniform step-2
   spacing (the hybrid 2/1 sweep is out), no slot may go below SF=12, slots
   whose arithmetic sequence exits the floor simply have fewer levels — and it
   must work for any dataset/baseline.

## Decision

1. **Deterministic noise keyed by (run_seed, global_episode, trial).**
   `blb_stage2_rl/seed_utils.py` (torch-free) derives salted streams for policy
   sampling (per episode/step/attempt), probe noise (per episode; trial mix
   identical to `probe_runner._trial_seed`), and the PPO-update shuffle (per
   update index). `function_handler.reseed_noise_rng_for_device(device, seed)`
   reseeds ONE device's noise generator; `BLBStage2Env.probe_noise_seed` routes
   `_eval_on_probe` through `_eval_on_probe_deterministic` (serial K trials on
   the env's own device, no global RNG mutation, no save/restore). The noisy
   baseline preflight uses the reserved `PREFLIGHT_EPISODE=-1` stream. CUDA
   Philox streams are device-independent, so the same seed gives the same noise
   on any GPU. Statistics are unchanged (distinct (episode, trial) keys =
   independent Gaussian streams); only reproducibility changes. Final eval
   keeps true-random MC repeats.
2. **Episode-parallel rollout for fusion mode** (`--stage2-rl-devices`, mirrors
   `--stage1-rl-devices`): `blb_stage2_rl/parallel_runner.py` runs N workers,
   each owning a deep-copied model/handler/bridge/env + per-window policy
   replica, collecting COMPLETE episodes (47-step rollout + per-step replan +
   serial K-trial probe) assigned as balanced contiguous chunks of GLOBAL
   episode indices and reassembled in global order; one PPO learner on the
   primary device updates at window boundaries with a deterministic pre-update
   reseed. K is a fixed hyperparameter (5), decoupled from the GPU count.
   Per-window `rollout_sig` (sha1 of all episodes' actions/rewards/terminal
   metrics) supports the 1-vs-N byte-identity harness. Scope: fusion-count mode
   only — its offline map is all-valid and it has no per-slot masks / frontier
   seeds, so episodes within a PPO window share no cross-episode state; the
   per-slot path (stateful `ForbiddenActionMask` etc.) keeps the legacy serial
   loop and the K-split `--blb-v3-reward-devices` path unchanged. The duplicate
   `terminal_metric_cache` is not consulted in this mode (cached values are not
   reproducible under per-episode noise keys; cost of always re-evaluating
   ≈1.5%). Expected ≈4.5–4.8× on 5 GPUs (residual = the serial PPO update).
3. **Uniform SF grid with a hard floor** (`action_space.MIN_SF_FLOOR = 12`):
   `sf_from` decodes `baseline − 2·dist`, clamped defensively at
   `min(12, baseline)`; `distinct_sf_level_indices` defines the SELECTABLE
   levels as the strict arithmetic sequence members ≥ 12 (baseline always
   selectable; odd baselines stop above the floor — no pseudo-12 level; a
   baseline below the floor contributes exactly one frozen level), so
   `option0 == baseline` holds for any calibrated baseline. The fusion map
   builder enumerates only selectable levels, and `_eval_block` /
   `BLBStage2SequentialEnv.evaluate_step` decode only the consumed
   `(layer, block)` via `action_vector_to_cfgs(..., only=...)` — bit-identical
   per-(L,B) output, large enumeration/rollout speedup. All committed fusion
   maps built under the old decode are STALE and must be rebuilt.

## Consequences

* Same seed + same GPU arch → byte-identical runs at any GPU count (server gate:
  per-window `rollout_sig` diff + `episodes.jsonl` value diff between
  `--stage2-rl-devices 0` and `0,…,N-1`).
* Old runs (true-random noise, K=NGPU) are statistically comparable but not
  reproducible; the 2026-06-10 A/B remains valid evidence.
* Fusion maps must be rebuilt whenever `sf_from` / `MIN_SF_FLOOR` / the level
  count changes (the map stores action indices).
* If a fusion map ever yields an invalid action at runtime, the worker logs
  `[ANOMALY]` and falls back to baseline — that event voids the 1==N guarantee
  for the run and means the map is broken; the server gate fails on it.
