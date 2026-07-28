# Stage-2 Five-GPU Near-Linear Runtime Design

## Objective

Optimize the current BERT-large MRPC Stage-2 RL training path on the new
five-RTX-4090 server. The accepted implementation must make the complete
170-episode run at least 4.5 times faster than a matched one-GPU control while
preserving the exact scientific state.

The optimization may change only execution efficiency and infrastructure
telemetry. It must not change actions, masks, random seeds, trial identities or
order, model inputs, reward, metrics, validation-bank decisions, candidate
semantics, PPO state, checkpoint state, structured training records, or
scientific conclusions.

## Accepted Starting Point

The clean local optimization branch starts from aggregate commit
`117f94590eb8bf22d40ff1f47abc518cebf53b74`, which contains the prior elastic
GPU implementation, the K6-K13 action-domain work, precision-preset changes,
and the latest Paean final-evaluation fix.

The target server has:

- five healthy NVIDIA GeForce RTX 4090 devices with 49,140 MiB each;
- 120 CPU cores and 311 GB RAM;
- an empty 500 GB `/hy-tmp` data volume;
- no active training process at design time.

Canonical source changes occur only in the clean local worktree. The server
receives committed source through Git fetch/pull or a SHA-256-verified Git
bundle and is used only for environment setup, tests, profiles, A/B runs, and
runtime artifacts.

## Existing Evidence And New Gap

The previous elastic implementation proved exact three-GPU execution and
four-to-three failure recovery. On the earlier server, matched 170-episode
BERT-large MRPC runs measured:

| GPUs | Wall time | Speedup | Parallel efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 7,573 s | 1.000x | 100.0% |
| 2 | 4,002 s | 1.892x | 94.6% |
| 3 | 2,792 s | 2.712x | 90.4% |
| 4 | 2,221 s | 3.410x | 85.2% |

Those results do not establish five-GPU scaling for the current aggregate or
the new server. They also show declining efficiency, so simply exposing the
fifth GPU is not sufficient. The new work must measure and reduce the serial
and per-worker overhead that prevents near-linear end-to-end scaling.

## Rejected Approaches

### Independent Episode Training Per GPU

Running five independent action/PPO streams would make aggregate episode
throughput easy to scale, but it would alter action order, policy state,
candidate order, PPO windows, and the final trajectory. This approach is
outside the efficiency-only contract.

### Reduced Validation Or Trial Counts

Changing the validation subset, K=5 trial count, validation-bank trial count,
promotion frequency, or report/checkpoint requirements changes the scientific
protocol. These are not runtime optimizations.

### Approximate Numeric Modes

AMP, TF32, changed reductions, approximate kernels, and batch-size changes are
disabled unless the strict server A/B proves tensor- and record-exact output.
Metric closeness or statistically similar reward is not sufficient.

### Rollout-Only Optimization

ADR-018 already established that KV-cache and batched lockstep rollout did not
improve end-to-end wall time and lost bit-exact determinism. They remain
rejected unless a fresh end-to-end profile proves the rollout has returned to
the critical path.

## Chosen Architecture

Retain one authoritative PPO learner and one canonical episode/action stream.
Use five persistent terminal-probe model replicas, one per healthy GPU. For
K=5, each action assigns exactly one immutable trial identity to each replica.
All returned results are restored to canonical trial order before metrics,
reward, candidate promotion, PPO, checkpoint, or structured-data mutation.

Optimization proceeds as a measured ladder. A candidate is retained only if it
passes exact equivalence and lowers profile-off end-to-end wall time.

## Baseline And Profile

Before changing runtime code, deploy the accepted starting commit and capture:

1. A short deterministic one-GPU and five-GPU screening run.
2. CPU and GPU utilization sampled at low frequency outside the hot path.
3. Per-episode wall time split into:
   - policy rollout and action materialization;
   - ordinary K=5 terminal probe;
   - validation-bank probes;
   - candidate lookup, append, and promotion;
   - PPO update;
   - checkpoint and structured-data transaction;
   - diagnostics/report generation;
   - worker dispatch, IPC wait, and replica straggler time.
4. Per-worker trial count, active time, install/clear time, forward time,
   batch-loop time, IPC time, and idle time.
5. The exact worker generation and physical/logical GPU mapping.

Profiling must use symmetric CUDA synchronization around measured GPU regions.
Profiling remains disabled in the final speed measurement.

## Five-Way Scientific Work Scheduling

Each probe task has identity:

```text
(absolute_episode, action_index, trial_index, trial_seed, batch_set_key)
```

The scheduler may change only the worker assigned to an identity. With five
healthy workers and K=5, assignment is:

```text
trial 0 -> worker 0
trial 1 -> worker 1
trial 2 -> worker 2
trial 3 -> worker 3
trial 4 -> worker 4
```

Validation-bank and final-certification probes use the same persistent pool and
identity contract. Grouped action evaluation may distribute multiple actions
across the five workers, but reductions and writes remain in canonical
action/trial order.

A worker failure quarantines only that worker. Missing identities retain their
original seeds and are retried on the remaining generation. A learner or
primary-device failure resumes only from a committed PPO transaction boundary.
Health resolution runs once at launch and after a real infrastructure failure;
it is not polled in the episode hot path.

## Per-GPU Efficiency Work

The profile determines which of these candidates are implemented:

1. **Persistent replicas and batch sets**
   - construct each BERT replica once;
   - register immutable validation batch sets once per worker;
   - keep tensors device-resident when the existing memory contract permits;
   - prove that no episode repeats CPU-to-GPU batch materialization.

2. **Action installation**
   - fingerprint the effective installed configuration;
   - skip only installs or clears already proven to be no-ops for that replica;
   - dispatch replica installation concurrently and preserve failure handling;
   - keep Block/SF/K/fusion materialization identical.

3. **Inference execution**
   - enforce eval and inference-only execution without autograd state;
   - remove avoidable Python object construction inside the per-batch loop;
   - reuse immutable kwargs and metric accumulators where values remain exact;
   - sweep evaluation batch size only behind a strict exact-output gate.

4. **Replica process scheduling**
   - send work to all secondary replicas before primary-device execution;
   - avoid CPU thread oversubscription across five BERT processes;
   - set measured Torch/OpenMP/MKL thread counts explicitly;
   - preserve one process and one CUDA context per replica.

5. **Straggler control**
   - report dispatch-to-result time per replica;
   - keep trial balance spread at zero for K=5;
   - investigate any worker whose repeated wall time exceeds the fastest worker
     by more than 10%;
   - do not speculatively duplicate scientific trials.

CUDA Graphs or `torch.compile` may be screened only after the ordinary path is
profiled. They are retained only when the full scientific comparator is exact
and end-to-end wall time improves.

## Serial-Path Efficiency Work

The remaining serial fraction is addressed without reordering scientific
state:

- cache deterministic action decoding, materialization, and cost identities
  using complete semantic keys;
- use the candidate-store active-status indexes rather than JSONL rescans;
- keep checkpoint-coupled diagnostic and structured writers append-only;
- move PNG/HTML/NPZ rendering outside the training hot path;
- prepare serialization payloads without copying large unchanged structures;
- allow infrastructure-only writes to finish asynchronously, but flush and
  verify them before PPO/checkpoint transaction commit;
- overlap CPU work with GPU probes only when it cannot observe or mutate the
  pending scientific result.

No optimization may move an episode across a PPO boundary or make candidate
promotion observe a different prefix.

## Measurement Ladder

### Screening

Use a short fixed-seed run to rank hotspots and reject regressions quickly.
Change one factor at a time. For each candidate:

1. run the focused correctness tests on the server;
2. run matched profile-off control and candidate measurements;
3. compare complete scientific artifacts;
4. retain only an end-to-end wall-time improvement;
5. commit and push before the next server deployment.

### Final A/B

Run fresh 170-episode controls from the same accepted starting state:

- one GPU: physical GPU 0;
- five GPUs: physical GPUs 0,1,2,3,4;
- BERT-large MRPC Stage-2;
- identical model, dataset, action domain, seed, K=5, PPO interval,
  validation banks, checkpoints, and structured writers;
- profiling disabled;
- no concurrent server workload.

The A/B harness records command lines, source commit/tree, environment,
wall-clock time, episodes/hour, sampled utilization, per-worker trial balance,
and the strict comparator output.

## Acceptance Gates

The optimization is complete only when all gates pass:

1. One-GPU and five-GPU runs each complete exactly 170 episodes.
2. End-to-end speedup is at least 4.5x and parallel efficiency at least 90%.
3. Throughput is measured from external process wall time, not summed component
   timers.
4. Every GPU receives exactly 170 ordinary K=5 trials, except explicitly
   recorded invalid-action short circuits shared by both arms.
5. Probe-worker straggler and utilization evidence explains any remaining gap
   to 5.0x.
6. Recursive comparison reports `equal=true` and `diffs=[]` for:
   - checkpoint tensors and optimizer/RNG state;
   - diagnostic episodes and PPO updates;
   - active candidate records and promotion state;
   - structured episodes, steps when present, and PPO updates.
7. Actions, rewards, priorities, metrics, trial seeds/order, and validation
   decisions are exact.
8. Focused and full server tests introduce no new failure.
9. Local branch, remote Git branch, and server source have identical full
   commit and tree IDs and tracked-clean status.

If speedup is below 4.5x, the run is evidence for the next profile cycle, not a
completion result. If exact equivalence fails, the candidate optimization is
reverted regardless of speed.

## Evidence And Audit

Compact evidence is returned through Git under:

```text
experiments/server_command_runs/stage2_five_gpu_runtime_20260728/
```

The archive contains:

- source and environment manifest;
- baseline and optimized commands;
- one-GPU and five-GPU wall-time verdict;
- utilization and per-worker balance summaries;
- strict recursive equivalence JSON;
- focused and full test logs;
- accepted/rejected optimization ledger;
- SHA-256 manifest.

Large checkpoints, model caches, raw nvidia-smi streams, and complete runtime
directories remain under `/hy-tmp` and are not added to Git.

## Completion Protocol

Before final deployment, refresh all remote agent heads and integrate only
completed, non-superseded changes into one clean aggregate. Re-run focused
tests and the final A/B if integration changes any runtime-relevant source.

The final report states the measured speedup, parallel efficiency, remaining
serial fraction, exact-equivalence result, source commit/tree, server parity,
and evidence paths. It does not claim scientific-quality conclusions from the
170-episode performance run.
