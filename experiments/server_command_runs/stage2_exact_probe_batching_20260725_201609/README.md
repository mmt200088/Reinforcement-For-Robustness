# BERT-large MRPC exact terminal-probe scheduling

## Scope

This run validates a runtime-only Stage-2 RL optimization for the observed
`K=5` terminal probe on four usable CUDA devices. The control assigned trials
as `(2,1,1,1)` for every episode, leaving three devices idle while the first
device ran its second trial.

The optimized path collects at most four episodes without crossing a PPO or
run boundary, preserves every action and trial seed, and schedules the
resulting 20 action/trial tasks round-robin. Each device receives exactly five
trials. Rewards, candidate promotion, PPO updates, callbacks, and checkpoint
state are committed in original episode order.

No trial was removed. Probe data batch shape, model arithmetic, precision,
metric aggregation, reward logic, and PPO configuration were not changed.

## Configuration

| Item | Value |
| --- | --- |
| Model / task | BERT-large MRPC |
| Start checkpoint | episode 10920, PPO update 91 |
| End checkpoint | episode 11280, PPO update 94 |
| Episodes per arm | 360 |
| PPO window | 120 episodes |
| BLB run seed | 20260725 |
| Dataset / validation seed | 42 |
| Online probe trials | 5 |
| Probe devices | CUDA 0,1,2,3 |
| Physical usable GPUs | NVML 0,1,2,4 |
| Control terminal batch | 1 |
| Optimized terminal batch | 4 |
| Equality tolerance | 0 |
| Required end-to-end speedup | 1.2x |

The final optimized arm ran source commit `201611d2` and tree `a390f5c2`.
The control was captured at `05b6863e`. The intervening source changes only
fix grouped-path mixed valid/invalid handling and grouped diagnostics; the
control's batch-1 path does not execute those branches.

## Results

| Metric | Control | Optimized | Ratio |
| --- | ---: | ---: | ---: |
| Training-loop wall time | 1692.807 s | 1147.595 s | 1.475x |
| Episodes/hour | 765.592 | 1129.319 | 1.475x |
| Terminal probe mean | 3.991 s | 2.518 s | 1.585x |
| Full process wall time | 1888.721 s | 1343.888 s | 1.405x |

The end-to-end gate passed with a 47.5% throughput increase. The optimized
trial assignment was exactly balanced: 450 trials per device across the 360
episodes, versus 720/360/360/360 in the control.

GPU sampling during the training interval showed the control's three
secondary cards at roughly 43-44% mean utilization with 0% median utilization.
The optimized arm raised all four usable cards to roughly 81-83% mean and 100%
median utilization.

## Exactness

- All 360 episode records passed quality/effect and strict diagnostic equality
  at absolute tolerance 0 after excluding timing/device fields.
- All three PPO update records passed equality at absolute tolerance 0 after
  excluding timestamps and elapsed time.
- Policy, optimizer, convergence state, best/frontier state, and
  Python/NumPy/Torch/CUDA RNG checkpoint state compared recursively with zero
  differences.
- The checkpoint comparison excluded only output-specific metadata:
  `structured_run_id`, physical candidate-store size, diagnostic file sizes,
  and file fingerprints.
- All 11,527 active candidate-store records were semantically equal after
  excluding only `created_at`.
- Both arms had zero invalid episodes and identical terminal-priority counts:
  priority 1 = 62, priority 2 = 5, priority 3 = 293.

## Edge Review

Independent review found that a batch containing exactly one model-ready
action could fall back to an ambient seed, and that grouped finalization could
roll back the base environment step counter. The accepted source:

- routes one or more explicitly seeded valid actions through the grouped API;
- finalizes mixed invalid/valid items in input order with a monotonic base step;
- separates per-action `k=5` diagnostics from batch-level `group_k`;
- adds a regression for invalid/valid/invalid/invalid ordering, exact seeds,
  and step progression from 17 to 21.

The old source failed both new regression assertions. The fixed source passed
them and the final focused suite passed 73/73 tests.

The broader 216-test CPU command retained three failures that were reproduced
on untouched base `e1a1bba9`: a missing Paean action-eval symbol and two stale
optimizer-baseline expectations. They are unrelated to this optimization.

## Evidence

The `final_evidence_201611d2/` directory contains:

- `final_verdict.txt`: strict episode/PPO and 1.2x speed gate;
- `checkpoint_compare.txt`: recursive checkpoint equality;
- `candidate_store_compare.txt`: active-record semantic equality;
- `benchmark_summary.json`: machine-readable timing and equality summary;
- `gpu_utilization_summary.txt`: sampled per-GPU utilization;
- compressed raw episode and PPO slices for both arms;
- exact argv/environment snapshots and focused test logs;
- `SHA256SUMS` for all evidence files.
