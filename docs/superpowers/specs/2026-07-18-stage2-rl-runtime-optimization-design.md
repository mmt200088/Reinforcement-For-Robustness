# Stage-2 RL Runtime Optimization Design

## Status

Accepted in conversation on 2026-07-18. The implementation baseline is
`48b03e869934aa8b3aa904a1fe8b611a1e2d618a`. This work optimizes only runtime,
memory, storage, and hardware utilization for the new layerwise Stage-2 RL
path. It does not change the research objective or acceptance criteria.

## Scope And Invariants

The optimized path must preserve all of the following:

- the 12-step, six-slot layerwise action space and exact K-level ordering;
- the dual-resource max-min objective and Shapley credit assignment;
- F1's fixed 256-example probe and five independent trials per episode;
- F4's complete `validation_full` split and 25 independent trials;
- F1/F4 seed derivation and evidence separation;
- promotion eligibility, probability gates, strict ranking, Pareto dominance,
  final revalidation, and natural-convergence patience;
- factorized PPO math, update windows, optimizer state, and checkpoint resume;
- complete structured evidence needed to reproduce reports and figures.

The implementation must not reduce the number of examples or trials, skip F4
assessments, approximate the statistical bootstrap, use mixed precision, alter
the reward, or continue F1 collection asynchronously while an F4 result is
pending. Those changes could alter the trajectory and are outside this task.

## Measured Baseline

The live five-RTX-5090 run at `48b03e8` provided the production baseline. At
4,224 completed episodes:

- median inter-episode time was `0.703377 s` and mean time was `1.515592 s`;
- online F1 probe time averaged `0.530650 s`, about 75% of the median episode;
- P95 inter-episode time was `4.900220 s` because F4 promotion is expensive;
- 783 F4 promotion assessments ran, 770 failed the probability gate, and 10
  were promoted;
- the process topology contained one F1 and one F4 five-GPU probe pool, for
  eight replica children and roughly one thousand process threads;
- GPUs 1-4 each retained two model replicas, while every replica process also
  created a CUDA context on GPU 0;
- `candidate_store.jsonl` averaged about `25.7 KiB` per row; its 5,049 trial
  rows occupied about `131.4 MB` and 783 status rows occupied `18.6 MB`;
- one typical F1 row repeated the full action three times, repeated a 1.6 KiB
  identity context, and carried about 11.9 KiB of derivable boosted overrides;
- the training stack retained complete episode objects in both
  `train_layerwise()` and its caller in addition to append-only diagnostics.

These measurements make F1/F4 model-forward throughput the wall-clock target,
and resident probe pools, history retention, and repeated JSON payloads the
resource-efficiency targets.

## Selected Design

### 1. One Shared GPU Probe Pool With Fidelity Views

F1 and F4 execute serially and use the same model weights, reversible handler,
BLB bridge, device list, and metric profile. Replace the second replica pool
with one owning `ProbeRunner` and two non-owning fidelity views:

- `F1` selects the existing stratified-probe batch set;
- `F4` selects a separately registered `validation_full` batch set;
- each child owns one model, handler, bridge, and a keyed batch-set registry;
- trial commands carry the batch-set key but retain the existing base seed and
  trial-index assignment;
- installs and clears remain serialized, and the existing pre/post-F4 cache
  invalidation remains authoritative;
- closing a view never closes the shared pool; the owner closes all workers
  exactly once.

The primary process keeps two lightweight `ProbeWorker` views over the same
model and bridge. Replica children receive F4 batches once, move them to their
own target GPU, and then reuse them. No model is duplicated for the second
fidelity.

At child startup, set the assigned CUDA device before any CUDA capability or
fast-math call. Limit child intra-op and inter-op CPU thread pools because probe
batches and metric tensors are already GPU-resident. This prevents every child
from creating a default-device context and avoids dormant CPU thread-pool
oversubscription.

### 2. Evidence-Gated Probe Batch Size

Batch size changes grouping only; they must not change samples, actions, noise
seeds, trial count, metric definitions, or trial order. Add explicit F1 and F4
probe batch-size plumbing while keeping the current evaluator batch size as the
compatibility fallback.

On the server, benchmark batch sizes `64`, `128`, and `256` with the same fixed
actions and exact seed sets. A larger size is eligible only if:

1. it does not OOM on any configured worker;
2. raw per-trial loss, metric1, and metric2 match the reference exactly, or the
   repository's existing deterministic parity gate proves they are equivalent;
3. all six constraint probabilities, promotion verdicts, rewards, priorities,
   and selected actions are identical;
4. measured F1 and F4 wall time improves over the current size.

Pin only the fastest passing size in the MRPC preset and record both effective
batch sizes in the run context and diagnostics. If no larger size passes, keep
`64`; the shared-pool and streaming changes remain valid optimizations.

There is no runtime auto-tuner in formal training. Selection is an offline,
reproducible server gate so a run never pays hidden calibration work or silently
changes its numerical contract.

### 3. Bounded In-Memory Training History

Append-only JSONL remains the source of complete episode and PPO history.
Production layerwise training keeps only data required by the next update and
live status:

- the current rollout window;
- a bounded recent-episode deque for update-window statistics;
- scalar counts, cumulative extrema, and the current strict best/frontier;
- bounded health windows and the existing top-K/diagnostic Pareto structures.

Do not retain complete `records`, `rewards`, and duplicate caller-side episode
lists for an unbounded production run. Direct unit-test callers may request
history retention explicitly for compatibility. Final curves and summaries are
rebuilt from the mandatory JSONL files, as the current completion path already
does.

Resume duplicate detection uses monotonic high-water marks for contiguous
episode and update IDs instead of sets containing every historical ID. Restore
must still fail closed on gaps, conflicting rows, or checkpoint fingerprint
mismatches.

### 4. Backward-Compatible Candidate Evidence Compaction

Introduce a compact trial-group record revision while continuing to read all
existing records:

- persist one canonical full action vector, not identical `action_indices`,
  `raw_action_indices`, and `effective_action_indices` arrays;
- intern each immutable identity context once by its existing SHA-256 hash and
  reference that hash from trial rows;
- retain the action matrix and all raw trial values/seeds;
- F1 rows retain the boosted-overrides hash and provenance but omit the verbose
  derivable override materialization;
- F4 promotion and final-revalidation evidence retain the full override payload
  required for authoritative replay;
- readers expose the same normalized aliases in memory so selection and resume
  code do not depend on the physical encoding revision.

The append-only recovery marker, candidate key, logical generation, committed
byte-size checkpoint, and incremental SHA-256 fingerprint contracts remain
unchanged. A mixed old/new store must restore identically.

### 5. Performance Telemetry

Persist enough telemetry to prove the optimization rather than infer it:

- shared pool ID, ownership, fidelity batch-set key, process count, and worker
  thread limits;
- effective F1/F4 probe batch sizes and batch counts;
- separate F1 probe, F4 promotion, install, clear, policy, PPO, checkpoint, and
  persistence wall times;
- candidate bytes written per episode and peak/current process RSS at update
  boundaries;
- GPU utilization and memory inventory in the server benchmark artifacts.

Telemetry must not add per-batch synchronization or synchronous rendering.

## Verification

### Focused Tests

Tests must prove:

1. F1 and F4 views route to different immutable batch sets on the same workers;
2. one shared pool starts four replica children for five GPUs and closes once;
3. trial seeds, assignment order, raw metrics, and diagnostics are unchanged;
4. F4 cache invalidation prevents a stale F1/F4 install from being reused;
5. child startup binds its target device before CUDA initialization and applies
   the intended CPU thread limits;
6. production history remains bounded across many synthetic update windows;
7. retained-history test mode preserves the existing direct-call API;
8. old and compact candidate stores restore the same evidence, promotions,
   strict winner, frontier, and checkpoint fingerprints;
9. F1 compact rows remain sufficient to reconstruct action/resource fields,
   while F4/final rows retain complete authoritative replay payloads;
10. existing layerwise action, reward, PPO, convergence, persistence, and report
    tests remain green.

### Server Gates

All execution occurs on the server from a Git-synchronized worktree. The active
`48b03e8` natural-convergence run must not be stopped or have its source tree
changed.

The server sequence is:

1. run focused CPU/static tests in a new worktree at the optimization SHA;
2. after the active GPU run releases the devices, run fixed-action F1/F4 batch
   parity and timing gates for `64/128/256`;
3. run a bounded seeded Stage-2 smoke comparing `48b03e8` with the optimized
   SHA, requiring equal action stream, trial seeds, raw trial metrics, rewards,
   priorities, promotion/revalidation decisions, and PPO diagnostics;
4. compare wall time, throughput, RSS, process/thread count, per-GPU memory, and
   utilization;
5. run the full repository test gate before promotion.

Short smokes establish plumbing, parity, and performance only. They do not make
RL quality or convergence claims.

## Git And Server Delivery

Implementation lives on `codex/stage2-rl-runtime-opt-48b03e8`, based exactly on
`48b03e8`. Local source changes are committed and pushed first. The server then
fetches that commit and creates a separate clean worktree; it never receives
source edits by copy, archive, or direct modification. Generated benchmark and
test artifacts return through Git before local synchronization. Final delivery
reports exact local, remote, and server SHAs and clean source status.
