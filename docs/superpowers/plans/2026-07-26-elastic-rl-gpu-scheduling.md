# Elastic RL GPU Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage-1 and Stage-2 RL automatically use all healthy GPUs, shrink and recover after device failures, and retain exact scientific state while adding negligible health-monitoring overhead.

**Architecture:** A CPU-only launcher supervisor resolves physical GPU health and remaps healthy devices before CUDA initialization. Stage-2 process workers and Stage-1 rollout workers keep stable scientific task identities, retain accepted results, and reschedule only missing work after a replica failure; primary or process-wide failures restart from the existing PPO transaction checkpoint. Every result is restored to canonical episode/action/trial order before reward, candidate, PPO, checkpoint, or structured-data updates.

**Tech Stack:** Bash launcher, Python 3 standard library, PyTorch/CUDA, multiprocessing pipes, existing Stage-1 and Stage-2 PPO/checkpoint code, unittest/pytest on the GPU server, Git worktrees.

---

## File Map

- Create `elastic_gpu.py`: torch-light exception, failure classification,
  logical/physical mapping, restart request, and atomic control-record helpers.
- Create `scripts/elastic_gpu_supervisor.py`: `nvidia-smi` health resolution,
  optional isolated canary, command remapping, child supervision, recovery
  polling, and restart telemetry.
- Modify `runtime_error_reporter.py`: convert only typed elastic restart
  exceptions to reserved exit code 75 and persist a machine-readable record.
- Modify `llama_7B_LayerImportance.sh`: expose elastic mode, select the correct
  Stage-1/Stage-2 auto-device flag, and launch the CPU-only supervisor.
- Modify `scripts/launcher_gpu_audit.py`: understand `auto` as all resolved
  logical devices rather than one literal token.
- Modify `blb_stage2_rl/probe_runner.py`: quarantine failed replica processes,
  retain accepted task results, retry missing identities, and publish pool
  generation diagnostics.
- Modify `blb_stage2_rl/layerwise_runner.py`: recompute the exact terminal
  action period when the probe-pool generation changes.
- Modify `blb_stage2_rl/sequential_runner.py`: capture/restore CUDA RNG state
  across a changed visibility map and honor checkpoint-boundary expansion
  requests.
- Modify `stage1_rl/parallel_runner.py`: quarantine a failed replica worker and
  retry only missing absolute episode IDs in the same PPO window.
- Modify `rl_data_points.py`: optionally checkpoint/recover `steps.jsonl` as
  well as episode and PPO mirrors.
- Modify `noise_rl_module_v2.py`: persist Stage-1 structured-data/detail-file
  boundaries, RNG state, and run identity.
- Modify `layer_importance_evaluator.py`: reuse the Stage-1 structured run ID,
  recover provisional output, classify worker GPU failures, and honor
  checkpoint-boundary expansion requests.
- Create `tests/test_elastic_gpu.py`.
- Create `tests/test_elastic_gpu_supervisor.py`.
- Modify `tests/test_launcher_gpu_audit.py`.
- Modify `tests/test_stage2_persistent_launcher.py`.
- Modify `tests/test_probe_runner_process_backend.py`.
- Modify `tests/test_blb_layerwise_runner.py`.
- Modify `tests/test_stage1_parallel_semantics.py`.
- Modify `tests/test_rl_data_points.py`.
- Create `tests/test_stage1_elastic_checkpoint.py`.
- Create `scripts/elastic_rl_scaling_ab.py`: server-only exact 1/2/4-GPU and
  fault-injection evidence comparator.

## Server Test Worktree

All executable verification happens on the server. Use the dedicated path
below so concurrent server jobs and untracked artifacts in the main checkout
are not touched:

```bash
SERVER_WT=/hy-tmp/rfr-elastic-rl-gpu-scheduling-20260726
git -C /hy-tmp/Reinforcement-For-Robustness fetch origin
git -C /hy-tmp/Reinforcement-For-Robustness worktree add --detach \
  "$SERVER_WT" origin/codex/elastic-rl-gpu-scheduling-20260726
```

For every later pushed commit:

```bash
git -C /hy-tmp/Reinforcement-For-Robustness fetch origin
git -C "$SERVER_WT" merge --ff-only \
  origin/codex/elastic-rl-gpu-scheduling-20260726
git -C "$SERVER_WT" status --short
```

Expected: fast-forward succeeds and tracked status is empty.

### Task 1: Elastic Failure And Control Primitives

**Files:**
- Create: `elastic_gpu.py`
- Create: `tests/test_elastic_gpu.py`
- Modify: `runtime_error_reporter.py`

- [ ] **Step 1: Write failing pure-Python contract tests**

Cover transport failures, known CUDA device-loss messages, fatal model errors,
logical-to-physical mapping under `CUDA_VISIBLE_DEVICES`, atomic failure
records, restart requests, and exit-code conversion:

```python
def test_transport_failure_is_recoverable(self):
    self.assertTrue(is_recoverable_gpu_failure(BrokenPipeError("pipe")))

def test_shape_error_is_not_recoverable(self):
    self.assertFalse(
        is_recoverable_gpu_failure(RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
    )

def test_logical_device_maps_through_visibility(self):
    self.assertEqual(
        physical_token_for_logical_device("cuda:2", "0,1,4,7"),
        "4",
    )
```

- [ ] **Step 2: Commit and push the red tests**

```bash
git add tests/test_elastic_gpu.py
git commit -m "test: define elastic GPU failure contracts"
git push
```

- [ ] **Step 3: Run the red test on the server**

```bash
python -m unittest -v tests.test_elastic_gpu
```

Expected: import failure for `elastic_gpu`.

- [ ] **Step 4: Implement the minimal control module**

Use a reserved restart status and fail-closed classifier:

```python
ELASTIC_GPU_RESTART_EXIT_CODE = 75

class ElasticGPUFailure(RuntimeError):
    def __init__(self, *, device, role, operation, cause):
        self.device = str(device)
        self.role = str(role)
        self.operation = str(operation)
        self.cause = cause
        super().__init__(
            f"{self.role} GPU {self.device} failed during "
            f"{self.operation}: {cause}"
        )

class ElasticGPURestartRequested(RuntimeError):
    pass
```

`is_recoverable_gpu_failure()` accepts process death, EOF/broken pipe,
one-hour worker timeout, `gpu requires reset`, `device is lost`,
`unspecified launch failure`, driver shutdown, and unavailable-device errors.
It explicitly rejects shape/assertion/model exceptions and CUDA OOM.

`runtime_error_reporter.run_fire_entrypoint()` catches only the two typed
elastic exceptions before the generic handler, writes
`logs/elastic_gpu_failure.json`, and exits 75.

- [ ] **Step 5: Commit, push, and run the green test**

```bash
git add elastic_gpu.py runtime_error_reporter.py
git commit -m "feat: add elastic GPU restart control"
git push
python -m unittest -v tests.test_elastic_gpu
```

Expected: all tests pass.

### Task 2: Fast Physical GPU Health Resolver And Supervisor

**Files:**
- Create: `scripts/elastic_gpu_supervisor.py`
- Create: `tests/test_elastic_gpu_supervisor.py`

- [ ] **Step 1: Write failing resolver and command-remap tests**

Use offline CSV fixtures so no local or test-process CUDA initialization is
required:

```python
GPU_CSV = """\
0, GPU-a, None
1, GPU-b, None
2, GPU-c, None
3, GPU-d, Reset
4, GPU-e, None
"""

def test_reset_device_is_removed_and_logical_ids_are_dense(self):
    resolved = resolve_health_snapshot(
        parse_nvidia_smi_csv(GPU_CSV),
        candidate_tokens=["0", "1", "2", "3", "4"],
    )
    self.assertEqual(resolved.healthy_tokens, ("0", "1", "2", "4"))
    self.assertEqual(resolved.logical_device_spec, "0,1,2,3")
```

Also verify explicit subset mapping, UUID input, all-unhealthy failure,
`auto` flag rewriting, restart-time `--resume_run_dir`, bounded retries, and
that only quarantined devices are passed to the recovery canary.

- [ ] **Step 2: Commit/push red tests and run them on the server**

```bash
git add tests/test_elastic_gpu_supervisor.py
git commit -m "test: define elastic GPU supervisor behavior"
git push
python -m unittest -v tests.test_elastic_gpu_supervisor
```

Expected: import failure for `scripts.elastic_gpu_supervisor`.

- [ ] **Step 3: Implement query-only normal startup**

The normal query is exactly:

```python
[
    "nvidia-smi",
    "--query-gpu=index,uuid,gpu_recovery_action",
    "--format=csv,noheader,nounits",
]
```

The supervisor must not import Torch. It sets `CUDA_VISIBLE_DEVICES` to healthy
physical tokens and replaces each `auto` device argument with dense logical
IDs. It executes the child in the foreground, preserves stdout/stderr, and
restarts only on status 75.

Recovery polling runs in a daemon thread only when the quarantine set is
nonempty. It invokes `nvidia-smi` every 60 seconds and runs the Torch canary
only for a candidate whose recovery action returned to `None`.

- [ ] **Step 4: Commit/push and run the green tests**

```bash
git add scripts/elastic_gpu_supervisor.py
git commit -m "feat: supervise RL on healthy GPUs"
git push
python -m unittest -v tests.test_elastic_gpu_supervisor
```

Expected: all tests pass.

- [ ] **Step 5: Measure the query on the actual server**

```bash
/usr/bin/time -f '%e' python scripts/elastic_gpu_supervisor.py \
  --health-only --json
```

Expected: GPU 3 has action `Reset`, healthy physical tokens are
`0,1,2,4`, logical IDs are `0,1,2,3`, and query wall time is below 0.5 seconds.

### Task 3: Launcher Auto-Device And Background Supervisor Wiring

**Files:**
- Modify: `llama_7B_LayerImportance.sh`
- Modify: `scripts/launcher_gpu_audit.py`
- Modify: `tests/test_launcher_gpu_audit.py`
- Modify: `tests/test_stage2_persistent_launcher.py`
- Modify: `tests/test_stage1_launcher_defaults.py`

- [ ] **Step 1: Add red launcher tests**

Assert:

```python
self.assertIn("--elastic-gpu-mode", script)
self.assertIn("scripts/elastic_gpu_supervisor.py", script)
self.assertIn("--blb_v3_reward_devices", command)
self.assertIn("auto", command)
```

Stage-1-only defaults to `--stage1_rl_devices auto`; layerwise Stage-2 defaults
to `--blb_v3_reward_devices auto`. Explicit subsets remain explicit, elastic
mode `off` retains direct launch, and audit treats `auto` as all visible GPUs.

- [ ] **Step 2: Commit/push red tests and run server launcher suites**

```bash
git add tests/test_launcher_gpu_audit.py \
  tests/test_stage2_persistent_launcher.py \
  tests/test_stage1_launcher_defaults.py
git commit -m "test: define elastic RL launcher defaults"
git push
python -m unittest -v \
  tests.test_launcher_gpu_audit \
  tests.test_stage2_persistent_launcher \
  tests.test_stage1_launcher_defaults
```

Expected: assertions for elastic flags/supervisor fail.

- [ ] **Step 3: Implement launcher integration**

Add:

```bash
ELASTIC_GPU_MODE="auto"
ELASTIC_GPU_RECOVERY_INTERVAL="60"
ELASTIC_GPU_MAX_RESTARTS="8"
```

Before building `CMD`, assign `auto` only when the relevant explicit device
flag is empty. Wrap only RL launches:

```bash
SUPERVISED_CMD=(
  python3 scripts/elastic_gpu_supervisor.py
  --run-dir "$RUN_ROOT"
  --recovery-interval "$ELASTIC_GPU_RECOVERY_INTERVAL"
  --max-restarts "$ELASTIC_GPU_MAX_RESTARTS"
  --
  "${CMD[@]}"
)
```

The existing `nohup`, PID files, lock FD, and log redirection use
`SUPERVISED_CMD`; non-RL and elastic mode `off` use `CMD`.

- [ ] **Step 4: Commit/push and run green launcher suites**

```bash
git add llama_7B_LayerImportance.sh scripts/launcher_gpu_audit.py
git commit -m "feat: auto-select healthy GPUs for RL"
git push
python -m unittest -v \
  tests.test_launcher_gpu_audit \
  tests.test_stage2_persistent_launcher \
  tests.test_stage1_launcher_defaults
```

Expected: all tests pass.

### Task 4: Stage-2 Replica Quarantine And Missing-Trial Retry

**Files:**
- Modify: `blb_stage2_rl/probe_runner.py`
- Modify: `tests/test_probe_runner_process_backend.py`

- [ ] **Step 1: Add red fault-injection tests**

Add a remote stub that returns valid results once and then raises
`BrokenPipeError`. Verify:

```python
self.assertEqual(runner.num_workers, 2)
self.assertEqual(runner.pool_generation, 1)
self.assertEqual(results, expected_canonical_results)
self.assertEqual(retried_tasks, failed_worker_tasks)
self.assertNotIn(successful_worker_tasks[0], retried_tasks)
```

Also verify a child model/shape error remains fatal, local-primary failure
raises `ElasticGPUFailure`, duplicate task identity fails closed, and closing a
quarantined handle is idempotent.

- [ ] **Step 2: Commit/push red tests and run on the server**

```bash
git add tests/test_probe_runner_process_backend.py
git commit -m "test: define Stage-2 elastic probe recovery"
git push
python -m unittest -v tests.test_probe_runner_process_backend
```

Expected: quarantine/retry tests fail while existing process tests pass.

- [ ] **Step 3: Implement the failure-only slow path**

Keep the existing successful hot path unchanged. Only when `errors` is
nonempty:

```python
def _quarantine_process_worker(self, worker_index, operation, exc):
    if worker_index == 0:
        raise ElasticGPUFailure(
            device=self.workers[0].device,
            role="learner-primary",
            operation=operation,
            cause=exc,
        )
    worker = self.workers.pop(worker_index)
    self._process_workers.remove(worker)
    worker.close()
    self.pool_generation += 1
    self._quarantine_events.append(...)
```

Build missing identities from `None` result slots, redistribute those exact
identities over the new live worker set, and repeat until complete or the
primary/final worker fails. No accepted identity is recomputed.

Normal diagnostics remain unchanged. Failure diagnostics add pool generation,
retry count, and quarantined devices.

- [ ] **Step 4: Commit/push and run focused plus Stage-2 contract tests**

```bash
git add blb_stage2_rl/probe_runner.py
git commit -m "perf: make Stage-2 probe workers elastic"
git push
python -m unittest -v tests.test_probe_runner_process_backend
python -m pytest -q tests/test_blb_*.py
```

Expected: all focused tests and the existing BLB contract suite pass.

### Task 5: Dynamic Exact Scheduling And Restart-Safe CUDA RNG

**Files:**
- Modify: `blb_stage2_rl/layerwise_runner.py`
- Modify: `blb_stage2_rl/sequential_runner.py`
- Modify: `tests/test_blb_layerwise_runner.py`

- [ ] **Step 1: Add red generation-change and RNG-remap tests**

Use a fake runner whose `num_workers` changes from 5 to 4 and
`pool_generation` from 0 to 1. Verify the next collection sizes are `1` then
`4`, without crossing a PPO boundary.

Test RNG mapping rules:

```python
restored = map_cuda_rng_states(
    old_states=["learner", "gpu1", "gpu2", "gpu4"],
    old_tokens=["0", "1", "2", "4"],
    new_tokens=["0", "1", "4"],
)
self.assertEqual(restored, ["learner", "gpu1", "gpu4"])
```

When the old learner token is absent, old logical-0 state maps to new
logical-0 so policy sampling continues exactly.

- [ ] **Step 2: Commit/push red tests and run on the server**

```bash
git add tests/test_blb_layerwise_runner.py
git commit -m "test: define dynamic terminal scheduling recovery"
git push
python -m unittest -v tests.test_blb_layerwise_runner
```

Expected: new generation/RNG tests fail.

- [ ] **Step 3: Recompute only at terminal collection boundaries**

Cache the last pool generation. At `if not finalized_drafts`, read the current
generation and worker count; recompute:

```python
terminal_batch_size = resolve_exact_terminal_batch_size(
    requested_terminal_batch_size,
    expected_online_trials,
    live_worker_count,
)
env.configure_terminal_probe_deferral(
    terminal_batch_size > 1 or protected_k1.enabled
)
```

This adds no `nvidia-smi`, CUDA synchronization, or health polling.

- [ ] **Step 4: Capture and restore CUDA RNG by visibility token**

On unchanged visibility, preserve the existing
`cuda_rng_state_all` representation and behavior. Add
`cuda_visible_device_tokens` to the checkpoint. On changed visibility,
restore matching replica states and always transfer the previous learner RNG
state to the new logical learner device. Probe-noise RNG remains trial-seeded
and independent.

After the existing checkpoint commit and status update, call
`raise_if_elastic_restart_requested()` so recovered cards join only between
PPO windows.

- [ ] **Step 5: Commit/push and run focused plus exact batching tests**

```bash
git add blb_stage2_rl/layerwise_runner.py \
  blb_stage2_rl/sequential_runner.py
git commit -m "perf: adapt Stage-2 scheduling to live GPUs"
git push
python -m unittest -v tests.test_blb_layerwise_runner
python -m pytest -q \
  tests/test_sequential_smoke.py \
  tests/test_probe_runner_process_backend.py
```

Expected: all tests pass.

### Task 6: Stage-1 Missing-Episode Retry And Transactional Resume

**Files:**
- Modify: `stage1_rl/parallel_runner.py`
- Modify: `rl_data_points.py`
- Modify: `noise_rl_module_v2.py`
- Modify: `layer_importance_evaluator.py`
- Modify: `tests/test_stage1_parallel_semantics.py`
- Modify: `tests/test_rl_data_points.py`
- Create: `tests/test_stage1_elastic_checkpoint.py`

- [ ] **Step 1: Add red retry and persistence tests**

Verify a worker failure after one accepted global episode retains that episode,
quarantines only the failed worker, and redistributes only missing global IDs.
The returned list remains in exact global order.

Verify `RLDataPointWriter` can commit/recover
`steps.jsonl`, `episodes.jsonl`, and `ppo_updates.jsonl` when explicitly
requested. Verify Stage-1 checkpoint version 2 stores:

```python
{
    "structured_run_id": "...",
    "structured_jsonl_sizes": {...},
    "details_file_sizes": {...},
    "torch_rng_state": ...,
    "cuda_rng_state_all": ...,
    "numpy_rng_state": ...,
    "python_rng_state": ...,
}
```

- [ ] **Step 2: Commit/push red tests and run on the server**

```bash
git add tests/test_stage1_parallel_semantics.py \
  tests/test_rl_data_points.py \
  tests/test_stage1_elastic_checkpoint.py
git commit -m "test: define Stage-1 elastic recovery"
git push
python -m unittest -v \
  tests.test_stage1_parallel_semantics \
  tests.test_rl_data_points \
  tests.test_stage1_elastic_checkpoint
```

Expected: new retry and checkpoint assertions fail.

- [ ] **Step 3: Implement Stage-1 failure-only redistribution**

Retain the normal one-pass thread window unchanged. On classified replica
failure, remove that worker and retry only result indices still `None`.
Unclassified exceptions and a failed primary raise immediately; a
process-wide CUDA failure becomes `ElasticGPUFailure` for the supervisor.

- [ ] **Step 4: Make Stage-1 output checkpoint-coupled**

Persist a stable `rl_data_points_run_id.txt` beside the Stage-1 checkpoint.
Flush and record JSONL/detail-file byte sizes at PPO boundaries. Before resume
writes, validate and truncate provisional tails to those sizes and remove only
post-checkpoint detail files matching the Stage-1 generated filename pattern.
Restore Python/NumPy/Torch RNG state and the canonical episode high-water mark.

Call `raise_if_elastic_restart_requested()` only after the checkpoint and all
committed boundaries are durable.

- [ ] **Step 5: Commit/push and run Stage-1 suites**

```bash
git add stage1_rl/parallel_runner.py rl_data_points.py \
  noise_rl_module_v2.py layer_importance_evaluator.py
git commit -m "perf: make Stage-1 rollout recovery elastic"
git push
python -m unittest -v \
  tests.test_stage1_parallel_semantics \
  tests.test_rl_data_points \
  tests.test_stage1_elastic_checkpoint
python -m pytest -q tests/test_stage1_*.py
```

Expected: all Stage-1 tests pass.

### Task 7: End-To-End Scaling And Fault-Equivalence Gate

**Files:**
- Create: `scripts/elastic_rl_scaling_ab.py`
- Create: `experiments/server_command_runs/elastic_rl_gpu_20260726/README.md`
- Create: evidence files under
  `experiments/server_command_runs/elastic_rl_gpu_20260726/`

- [ ] **Step 1: Add the server-only comparator**

The comparator reads two run directories and checks exact equality for:

- absolute episode/action/trial IDs and seeds
- trial values, terminal metrics, rewards, and candidate decisions
- PPO update rows
- recursive policy/optimizer/checkpoint scientific state
- active candidate-store records
- structured episodes, steps, and PPO rows

It excludes only wall time, PID, device assignment, pool generation,
quarantine/retry, and health-query telemetry.

- [ ] **Step 2: Commit/push the harness**

```bash
git add scripts/elastic_rl_scaling_ab.py
git commit -m "test: add elastic RL equivalence gate"
git push
```

- [ ] **Step 3: Run profile-off 1/2/4-GPU Stage-2 controls**

Use one fixed checkpoint, seed, K=5, 360 episodes, three PPO windows, and
explicit healthy physical subsets:

```bash
python scripts/elastic_rl_scaling_ab.py run-stage2 \
  --checkpoint "$CONTROL_CHECKPOINT" \
  --physical-device-sets 0 0,1 0,1,2,4 \
  --episodes 360 \
  --output-root /hy-tmp/elastic_rl_gpu_ab_20260726
```

Expected: exact scientific equality across GPU counts, monotonic
episodes/hour, and measured speedup/parallel efficiency against one GPU.

- [ ] **Step 4: Inject a Stage-2 replica failure**

Kill one probe child after a configured task identity. Compare against the
no-failure four-GPU control:

```bash
python scripts/elastic_rl_scaling_ab.py inject-stage2-replica-failure \
  --control /hy-tmp/elastic_rl_gpu_ab_20260726/stage2_4gpu \
  --kill-after-task 120:1:2
```

Expected: automatic 4-to-3 worker transition, exact final scientific state,
and no duplicate or missing task identity.

- [ ] **Step 5: Inject a learner failure and recovery**

Terminate the learner after a provisional episode inside a PPO window. The
supervisor must restart with the failed physical token excluded, roll back to
the prior boundary, and replay:

```bash
python scripts/elastic_rl_scaling_ab.py inject-learner-failure \
  --control /hy-tmp/elastic_rl_gpu_ab_20260726/stage2_4gpu \
  --kill-after-episode 181
```

Expected: one reserved-status restart, exact final scientific state, and
checkpoint/JSONL/candidate fingerprint validation passes.

- [ ] **Step 6: Run Stage-1 1/2/4-GPU and failure checks**

Use at least two PPO windows from one fixed Stage-1 checkpoint. Require exact
episode/PPO/checkpoint/structured equality and monotonic throughput.

- [ ] **Step 7: Measure monitor overhead**

Run matched controls with recovery monitor disabled/enabled and no quarantined
card. Because no polling thread is started without quarantine, expected
profile-off wall difference is within noise and below 0.5%.

- [ ] **Step 8: Commit/push evidence only after gates pass**

```bash
git add experiments/server_command_runs/elastic_rl_gpu_20260726
git commit -m "bench: verify elastic RL GPU scaling"
git push
```

### Task 8: Final Aggregate, Three-Way Parity, And Handoff

**Files:**
- Create: `finish1.md` only after every gate below passes.

- [ ] **Step 1: Refresh all remote heads**

```bash
git fetch --all --prune
git for-each-ref --sort=-committerdate \
  --format='%(objectname) %(refname:short) %(subject)' refs/remotes/origin
```

Integrate only completed, non-superseded agent commits. Do not merge active,
dirty, experimental, or unmarked work.

- [ ] **Step 2: Run final server contract and exact gates from the aggregate**

Run the focused suites, full Stage-1/Stage-2 contracts, 1/2/4 scaling,
replica failure, learner restart, and comparator from the exact aggregate
commit.

- [ ] **Step 3: Verify local, remote, and server identity**

For each location record:

```bash
git rev-parse HEAD
git rev-parse HEAD^{tree}
git status --porcelain --untracked-files=no
```

Expected: full commit SHA and tree SHA match, and tracked status is empty.

- [ ] **Step 4: Create and push `finish1.md`**

The marker records final branch, commit, tree, server parity, exact-equivalence
result, 1/2/4 end-to-end throughput, fault-recovery evidence, health-query and
monitor overhead, and evidence paths.

```bash
git add finish1.md
git commit -m "docs: complete elastic RL GPU optimization"
git push
```

- [ ] **Step 5: Let the server pull the marker and recheck parity**

Only after server parity succeeds, publish byte-identical `finish1.md` into
the shared project root for the downstream watcher.
