# Elastic RL GPU Scheduling Design

## Objective

Make Stage-1 and Stage-2 RL use every healthy server GPU while preserving the
exact scientific execution contract. A run must start on the healthy GPU set,
shrink after a device failure, and expand after a device recovery without
changing actions, random seeds, trial order or values, rewards, metrics,
candidate semantics, PPO inputs or updates, checkpoint state, structured
training data, or scientific conclusions.

The parallel portion should scale as closely as practical to the number of
healthy GPUs. End-to-end scaling is measured and reported separately because
serial policy sampling, PPO, checkpointing, and persistence remain bounded by
Amdahl's law.

## Current Failure Modes

The launcher accepts static device lists. It does not reject a GPU that is
visible to CUDA but requires reset. Stage-1 and Stage-2 capture their worker
count at startup, and neither path can change that count during training.

Stage-2 probe workers are persistent processes, but any worker timeout, process
exit, or CUDA error aborts the whole probe batch. The exact K-trial terminal
batch period is computed once, so a later worker-count change would leave the
scheduler imbalanced.

Stage-1 validation workers use threads in the learner process. A failed CUDA
context can therefore poison the whole process, and the remaining GPUs cannot
continue independently.

The current server exposes five RTX 4090 devices, but GPU 3 reports
`GPU requires reset`. A correct launch should use the other four devices
without repeatedly probing them in the RL hot path.

## Decision

Use a layered, event-driven elastic design:

1. Resolve the healthy device set once before the training process initializes
   CUDA.
2. Isolate model-evaluation workers in per-device child processes.
3. Assign stable scientific task identities through a deterministic work
   queue, and restore canonical result order before any reduction or write.
4. Quarantine a failed secondary worker and retry only unresolved identities.
5. Recompute exact scheduling periods whenever the live worker generation
   changes.
6. Recover a failed learner/primary GPU through a supervisor from the latest
   committed PPO transaction boundary.
7. Admit a recovered device only at a transaction boundary.

Explicit device lists remain available for controlled A/B runs. Auto mode is
the production server mode.

## Low-Overhead Health Resolution

Startup resolution performs one batched `nvidia-smi` query for physical index,
UUID, and `gpu_recovery_action`. A device whose recovery action is not `None`
is excluded immediately; on the current server this identifies GPU 3 as
`Reset` without initializing Torch or a CUDA context. A small isolated CUDA
allocation/kernel/synchronize canary is reserved for ambiguous query results,
an explicitly requested strict startup audit, and recovery admission. Multiple
required canaries run in parallel rather than serially. The resolved physical
identifiers, logical remap, status, reason, and elapsed time are written to the
run manifest.

There is no periodic health query in the episode, terminal-probe, validation,
PPO, candidate, checkpoint, or report hot paths. Healthy workers are considered
healthy until their existing process pipe, timeout, or CUDA execution reports a
failure.

Only quarantined devices are checked for recovery. A separate low-priority
monitor runs at a default 60-second interval, first performs a cheap
`nvidia-smi` status check, and runs the isolated CUDA canary only when that
status becomes eligible. The monitor never synchronizes a healthy training
device. Recovery is communicated to the supervisor and applied at the next PPO
transaction boundary.

The normal query-only startup check has a 0.5-second target budget on the
current five-GPU server. Optional isolated canary time is reported separately
and is never charged to episode throughput. Background monitoring must add less
than 0.5% to profile-off end-to-end wall time; otherwise the recovery monitor
remains disabled while startup filtering and event-driven shrink stay enabled.

## Stable Scientific Work

Every parallel unit has an immutable identity:

- Stage-1: `(absolute_episode, validation_partition)`
- Stage-2: `(absolute_episode, action_index, trial_index, trial_seed)`

The scheduler may change only the worker assigned to an identity. It may not
change the identity, seed, input, retry count visible to scientific reduction,
or canonical order.

Workers return identity-tagged results. The parent rejects duplicates with
different values, rejects unknown identities, and waits for every required
identity. Results are restored to original episode/action/trial order before
metrics, reward, candidate promotion, PPO, callbacks, or structured data are
updated.

If a worker fails, completed identities already received remain committed in
the in-memory batch. Only identities without an accepted result are put back on
the queue. A retried deterministic probe uses the original trial seed and
inputs. CUDA/process infrastructure failures are recoverable; model, shape,
validation, or scientific-contract exceptions remain fatal.

## Stage-2 Elastic Probe Pool

The existing persistent process pool becomes generation-aware. A generation
contains an ordered set of live workers and a monotonically increasing
generation number. Worker failure closes and quarantines only that process,
increments the generation, and redistributes unresolved tasks over the
remaining workers.

The layerwise loop reads the current live worker count at every terminal batch
collection boundary. For K trials and W workers, it chooses the smallest
positive action count that makes `K * actions` divisible by W, capped by the
configured terminal batch size, the PPO boundary, convergence boundary, and
remaining episode budget. If exact balancing is unavailable, it uses the
smallest deterministic batch with no scientific fallback.

For K=5 this gives:

| Healthy workers | Exact action period |
| ---: | ---: |
| 5 | 1 |
| 4 | 4 |
| 3 | 3 |
| 2 | 2 |
| 1 | 1 |

An invalid action remains an action-local short circuit and does not consume a
probe worker. Diagnostics may add infrastructure fields such as pool
generation, device assignment, quarantine reason, and retry count. Existing
scientific fields must remain byte-for-byte or numerically exact.

## Stage-1 Elastic Validation Pool

Stage-1 moves GPU validation from threads sharing the learner CUDA process to
persistent per-device worker processes. Absolute episode IDs, action sampling,
and PPO order remain owned by the parent. Complete episode evaluations are
dispatched by identity and returned in absolute episode order.

The same quarantine and unresolved-task retry rules apply. The active rollout
window is capped at the next PPO boundary so a worker-count change cannot move
an episode across an update. Worker creation, model loading, and validation
dataset materialization happen outside timed episode execution and are reused
across windows.

## Learner Failure And Transactional Restart

The learner process is supervised by a small launcher process that does not
initialize CUDA. At each PPO boundary, the learner commits one transaction
containing:

- policy, value network, optimizer, scheduler, and PPO counters
- Python, NumPy, CPU Torch, and CUDA RNG states
- next absolute episode and probe-seed derivation state
- candidate index/store state required for exact promotion behavior
- append offsets and hashes for structured JSON/JSONL and diagnostics
- best-so-far, convergence, checkpoint, and report state
- active and quarantined device metadata

Writes produced after that boundary are provisional until the next transaction
commit. If the learner GPU fails, the supervisor terminates the poisoned
process, restores the previous transaction, truncates only provisional
append-only output to recorded offsets, verifies hashes, resolves the new
healthy set, and deterministically replays the uncommitted window.

No committed episode is replayed. No provisional episode survives alongside
its replay. Candidate evidence and promotion decisions are restored with the
same boundary, so restart cannot duplicate or skip candidate effects.

The supervisor has a bounded restart policy and records every recovery. It
fails closed if a file hash, checkpoint component, RNG state, or candidate
state cannot be restored exactly.

## Expansion Semantics

A recovered quarantined device is canary-tested in isolation. The current
window continues on its existing generation. At the next committed PPO
boundary, the supervisor can create a new worker generation containing the
recovered device. This avoids changing scheduling inside an already sampled
window.

Expansion changes only wall time and infrastructure diagnostics. The canonical
task stream and reduction order remain unchanged.

## Telemetry

The run manifest and structured throughput records add:

- candidate, healthy, and quarantined physical device identifiers
- logical CUDA remap and pool generation
- startup health-check wall time
- live worker count per window
- task counts, retries, worker exits, quarantine and recovery events
- useful evaluation wall time, scheduler idle time, and serial wall time
- one-, two-, and N-GPU throughput plus parallel efficiency

These fields are diagnostic. They are excluded from scientific-equivalence
comparisons unless the comparison explicitly checks infrastructure behavior.

## Verification Gate

All executable verification runs on the GPU server from Git-synchronized
source. The change is enabled only after:

1. Focused startup resolver, task identity, canonical ordering, quarantine,
   retry, generation-change, transaction, and recovery tests pass.
2. Existing Stage-1 and Stage-2 contract suites have no new failures.
3. Fixed-source profile-off 1/2/4-GPU A/B runs use identical model, dataset,
   seed, K, action stream, PPO interval, and episode budget.
4. Every scientific episode, trial, reward, metric, candidate, PPO,
   checkpoint, and structured state field is exactly equal at `atol=0`.
5. Killing a secondary worker mid-probe yields the same final state as the
   no-failure control.
6. Killing the learner at a selected provisional episode produces an automatic
   restart and the same final state as the no-failure control.
7. GPU 3 is excluded automatically while it requires reset, with no attempted
   training allocation on that device.
8. Query-only startup checking meets its 0.5-second target and background
   monitoring adds less than 0.5% end-to-end wall time.
9. Parallel evaluation throughput improves monotonically from one to two to
   four healthy GPUs. End-to-end speedup and parallel efficiency are reported
   against the one-GPU control and compared with the measured Amdahl limit.

If a semantic field differs, a recovery cannot prove exact rollback, or an
optimization does not improve profile-off end-to-end wall time, it is not
enabled in the production aggregate.

## Source And Deployment Protocol

Canonical source changes are made only in a clean local isolated worktree.
Every completed source batch is committed and pushed before server execution.
The server receives source only through Git fetch and checkout/pull and is used
only for tests, profiles, A/B runs, and artifacts.

Before final deployment, refresh all remote agent heads and integrate only
completed, non-superseded work into one clean aggregate. Verify identical full
commit IDs, source-tree IDs, and tracked-clean status for the local aggregate,
Git remote, and server. Runtime artifacts remain untracked and intact.
