# Elastic RL GPU Scheduling Evidence

Date: 2026-07-27

## Scope

This artifact records runtime-only Stage-2 RL scheduling changes that discover
healthy GPUs once at launch, split terminal-probe trials across all healthy
devices, quarantine a failed replica, and resume at a committed PPO boundary.
The implementation preserves actions, seeds, trial order, rewards, metrics,
candidate state, PPO/checkpoint state, and structured training data.

Runtime evidence was produced at commit
`6e8d1cfc88e03d2e6fd20d5255685569c4821fc5`. Commit `2af7bd55` only updates a
static source-order test for the conditional restart call.

## Scaling

Matched 170-episode BERT-large MRPC Stage-2 runs used the same single-GPU
control and exact recursive equivalence gate.

| Healthy GPUs | Wall time | Throughput | Speedup | Efficiency |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7573 s | 80.813 ep/h | 1.000x | 100.0% |
| 2 | 4002 s | 152.924 ep/h | 1.892x | 94.6% |
| 4 | 2221 s | 275.552 ep/h | 3.410x | 85.2% |

Both multi-GPU runs passed quality/effect, strict diagnostic, and PPO equality.
Their full gate output is in `scaling_1v2_verdict.txt` and
`scaling_1v4_verdict.txt`.

## Fault Injection

A four-GPU run deliberately terminated the worker mapped to physical GPU 4
after 30 completed online groups. Episode 32 was the first three-device
episode. The remaining 138 episodes ran on three GPUs and preserved every
five-trial group exactly once:

- three-device partition shapes rotated through `2/2/1`, `2/1/2`, and `1/2/2`;
- cumulative post-fault trial loads were `231/230/229`;
- the supervisor restarted once at the PPO checkpoint and resumed the same
  structured run;
- the final process exited with return code 0 without a no-work recovery
  restart;
- recursive comparison found no differences across 170 diagnostic episodes,
  two PPO updates, the checkpoint, 485 candidate records, and all structured
  episode/PPO records.

The fault-tolerant run took 2852 seconds, or `2.655x` the single-GPU control.
Avoiding the terminal no-work restart reduced the prior fault-run wall time
from 3012 to 2852 seconds (`1.056x`, 5.31%).

Physical GPU 3 was excluded because the server reported it unhealthy. The
health check is outside the episode hot path; training uses the resulting
healthy-device set until a replica failure or low-frequency recovery request
requires a PPO-boundary restart.

## Verification

- Focused elastic and Stage-2 tests: 149 passed.
- Full server suite with `CUDA_VISIBLE_DEVICES=0,1,2`: 1836 passed, 3 skipped.
- `fault_tolerance_summary.json` is the compact machine-readable audit.
- `strict_fault_equivalence.json` contains the recursive zero-diff result.
- `fault_health_events.jsonl` records launch, quarantine restart, resume, and
  clean exit.
- `pre_terminal_fix_health_events.jsonl` records the eliminated second restart.
- `SHA256SUMS` authenticates the copied server evidence.

Full raw runtime artifacts remain on the server under
`/hy-tmp/elastic_rl_gpu_20260726/`; only compact evidence is duplicated in Git.
