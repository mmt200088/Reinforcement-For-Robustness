# Stage-2 RL Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce wall time, GPU memory duplication, CPU oversubscription, process RSS, and candidate-store write volume in the `48b03e8` Stage-2 layerwise RL path without changing actions, trials, metrics, rewards, promotion decisions, PPO updates, convergence, or scientific outputs.

**Architecture:** Keep one owning five-GPU `ProbeRunner` and expose keyed F1/F4 views over its workers, with each worker retaining one model and multiple immutable probe batch sets. Make production histories bounded while keeping JSONL authoritative, write compact candidate evidence with backward-compatible logical hydration, and select probe batch size only through an offline exact-parity server gate.

**Tech Stack:** Python 3, PyTorch multiprocessing/CUDA, JSONL persistence, Bash launchers, `unittest`/`pytest`, five RTX 5090 GPUs, Git worktrees.

---

## File Map

- `blb_stage2_rl/probe_runner.py`: shared probe-pool ownership, keyed batch sets, child startup/thread controls, per-call pool telemetry.
- `blb_stage2_rl/runner.py`: explicit F1/F4 probe batch-size construction and train-config wiring.
- `blb_stage2_rl/sequential_runner.py`: reuse the shared pool for F4, bounded production bookkeeping, exact checkpoint/high-water behavior, runtime telemetry.
- `blb_stage2_rl/layerwise_runner.py`: optional history retention and compact F1/F4 evidence calls only; no reward, promotion, PPO, or convergence math changes.
- `blb_stage2_rl/candidate_store.py`: compact record revision, identity-context interning, old/new hydration and indexing.
- `blb_stage2_rl/diagnostics.py`: bounded online windows plus exact cumulative counters while preserving complete JSONL.
- `rl_tune.py`, `layer_importance_evaluator.py`, `llama_7B_LayerImportance.sh`: explicit F1/F4 batch-size CLI plumbing with compatibility fallbacks.
- `presets/mrpc-blb-stage2-rl.conf`: evidence-selected formal MRPC F1/F4 batch sizes; unchanged until the GPU parity gate passes.
- `scripts/stage2_ngpu_speed_ab.sh`, `scripts/stage2_ngpu_ab_compare.py`: baseline-versus-optimized parity/performance gates and exclusion of telemetry-only fields.
- `scripts/stage2_runtime_optimization_gate.sh`: idle-GPU-guarded baseline/optimized and 64/128/256 batch-size orchestration.
- `tests/test_probe_runner_process_backend.py`: pool/view, routing, close, thread, and child-command contracts.
- `tests/test_blb_layerwise_runner.py`: shared F1/F4 construction, bounded history, and unchanged layerwise outputs.
- `tests/test_blb_candidate_store_identity.py`: mixed v1/v2 restore, compaction, recovery, and replay contracts.
- `tests/test_rl_data_points.py`: bounded diagnostics restore and cumulative-count contracts.
- `tests/test_sequential_smoke.py`: config/launcher/runtime integration contracts.
- `experiments/server_command_runs/stage2_rl_runtime_opt_${TS}/`: server-only test, parity, timing, GPU, RSS, and sync evidence, where `TS="$(date +%Y%m%d_%H%M%S)"` is set once by the server gate.

### Task 1: Establish The Isolated Server Test Lane

**Files:**
- Read: `docs/superpowers/specs/2026-07-18-stage2-rl-runtime-optimization-design.md`
- Use: local branch `codex/stage2-rl-runtime-opt-48b03e8`
- Create on server through Git: `/hy-tmp/rfr_stage2_rl_runtime_opt`

- [ ] **Step 1: Verify the local baseline and cleanliness**

Run locally, without executing project code:

```bash
git rev-parse 48b03e8
git merge-base --is-ancestor 48b03e8 HEAD
git status --short
```

Expected: the full baseline SHA is `48b03e869934aa8b3aa904a1fe8b611a1e2d618a`, the ancestry check exits `0`, and status is empty before each implementation batch.

- [ ] **Step 2: Push the current plan commit before server work**

```bash
git push origin codex/stage2-rl-runtime-opt-48b03e8
git ls-remote --heads origin codex/stage2-rl-runtime-opt-48b03e8
```

Expected: the remote SHA exactly equals local `HEAD`.

- [ ] **Step 3: Verify that the active server run is untouched**

Run read-only server inspection:

```bash
ps -p 2236414 -o pid=,stat=,etime=,cmd=
git -C /hy-tmp/rfr_runtime_optimization rev-parse HEAD
```

Expected: if PID `2236414` still exists it remains running from `/hy-tmp/rfr_runtime_optimization`; that checkout remains at `48b03e8`. Do not stop it, update it, or run a GPU test beside it.

- [ ] **Step 4: Create the separate server worktree from the pushed branch**

From the server repository that owns `/hy-tmp/rfr_runtime_optimization`:

```bash
git fetch origin refs/heads/codex/stage2-rl-runtime-opt-48b03e8
git worktree add --detach /hy-tmp/rfr_stage2_rl_runtime_opt FETCH_HEAD
git worktree add --detach /hy-tmp/rfr_stage2_rl_runtime_base \
  48b03e869934aa8b3aa904a1fe8b611a1e2d618a
git -C /hy-tmp/rfr_stage2_rl_runtime_opt rev-parse HEAD
git -C /hy-tmp/rfr_stage2_rl_runtime_base rev-parse HEAD
git -C /hy-tmp/rfr_stage2_rl_runtime_opt status --short
git -C /hy-tmp/rfr_stage2_rl_runtime_base status --short
```

Expected: optimized server SHA equals the pushed SHA, baseline SHA equals `48b03e8`, and both statuses are empty. All CPU/static tests use `CUDA_VISIBLE_DEVICES=""`; GPU gates wait for the active run to release the devices.

### Task 2: Share One Probe Pool Across F1 And F4

**Files:**
- Modify: `blb_stage2_rl/probe_runner.py:183-1110`
- Modify: `blb_stage2_rl/sequential_runner.py:3374-3425, 5448-5520, 5810-5840, 4990-5030`
- Test: `tests/test_probe_runner_process_backend.py`
- Test: `tests/test_blb_layerwise_runner.py`

- [ ] **Step 1: Write failing keyed-batch and ownership tests**

Add process-backend tests with local/remote stubs that record `batch_set_key` and registration commands:

```python
def test_views_route_f1_and_f4_to_the_same_workers(self):
    events = []
    owner, _remote = self._runner(events)
    owner.register_batch_set("F4", ["validation-full"])
    f1 = owner.view("F1")
    f4 = owner.view("F4")

    f1.run_trials(k=2, base_seed=41)
    f4.run_trials(k=2, base_seed=41)

    submitted = [event for event in events if event[:2] == ("remote-submit", "run_trials")]
    self.assertEqual([event[2]["batch_set_key"] for event in submitted], ["F1", "F4"])
    self.assertEqual(f1.pool_id, f4.pool_id)

def test_view_close_never_closes_owner_and_owner_closes_once(self):
    events = []
    owner, remote = self._runner(events)
    owner.view("F1").close()
    owner.view("F4").close()
    self.assertFalse(remote.closed)
    owner.close()
    owner.close()
    self.assertEqual(events.count(("remote-close", None)), 1)
```

Add a layerwise construction test asserting `base_env.probe_runner.pool_id == promotion_env.probe_runner.pool_id`, keys `F1` and `F4`, and one owner with four replica processes for device IDs `0..4`.

- [ ] **Step 2: Commit and push the RED tests**

```bash
git add tests/test_probe_runner_process_backend.py tests/test_blb_layerwise_runner.py
git commit -m "test: pin shared Stage-2 probe pool contracts"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

- [ ] **Step 3: Run the RED tests on the server CPU lane**

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_probe_runner_process_backend.py \
  tests/test_blb_layerwise_runner.py -k 'view or batch_set or shared_probe_pool'
```

Expected: FAIL because `ProbeRunner.view`, `register_batch_set`, `pool_id`, and keyed child commands do not exist.

- [ ] **Step 4: Implement keyed worker batches and child controls**

Keep `ProbeWorker.probe_batches` as the compatibility default and add immutable keyed sets:

```python
def register_batch_set(self, key: str, batches: Sequence[Any]) -> None:
    normalized = str(key).strip()
    if not normalized:
        raise ValueError("probe batch-set key must be nonempty")
    if normalized in self.probe_batch_sets:
        raise ValueError(f"probe batch-set {normalized!r} is already registered")
    self.probe_batch_sets[normalized] = tuple(batches)

def run_trial(self, trial_idx: int, base_seed: int, batch_set_key: str = "F1"):
    batches = self.probe_batch_sets[str(batch_set_key)]
    # Keep existing seed derivation and run_installed_probe_trial call unchanged.
```

In `_probe_process_main`, bind before any CUDA capability/fast-math call and cap helper pools:

```python
device = torch.device(f"cuda:{int(device_id)}")
torch.cuda.set_device(device)
torch.set_num_threads(resolve_probe_intraop_threads())
try:
    torch.set_num_interop_threads(resolve_probe_interop_threads())
except RuntimeError:
    pass
enable_cuda_reward_probe_fast_math()
```

Support `register_batch_set`, and require `batch_set_key` on `run_trials` and `run_action_trial` child messages. Transfer each registered batch set to each replica only once.

For dynamically registered F4 batches, first make one CPU copy in the parent, send only CPU tensors through each pipe, and let each child move that copy to its already-bound target device. Never pickle a CUDA-0 tensor into a replica child.

- [ ] **Step 5: Implement owner and non-owning views**

Add a thin view with no duplicated model/process state:

```python
class ProbeRunnerView:
    def __init__(self, owner: "ProbeRunner", batch_set_key: str):
        self._owner = owner
        self.batch_set_key = str(batch_set_key)

    @property
    def pool_id(self) -> str:
        return self._owner.pool_id

    def run_trials(self, k: int, base_seed: int):
        return self._owner.run_trials(k, base_seed, batch_set_key=self.batch_set_key)

    def close(self) -> None:
        return None
```

Delegate `install_action`, `clear`, `run_action_trials_once`, `num_workers`, `devices`, `backend`, and `last_diagnostics`. `ProbeRunner.close()` remains idempotent and is the only operation that reaps replica processes.

- [ ] **Step 6: Reuse the owner in the layerwise F4 builder**

When the F1 pool is created, store its owner on the base environment and install an F1 view. In `_build_authoritative_validation_env`, register `validation_full` as `F4` and install an F4 view instead of calling `build_probe_runner()` again:

```python
owner = getattr(base_env, "_shared_probe_runner_owner", None)
if owner is not None:
    owner.register_batch_set("F4", validation_full_batches)
    promotion_env.probe_runner = owner.view("F4")
```

The finalizer closes `base_env._shared_probe_runner_owner` once. Preserve all existing pre/post-F4 `clear_installed_blb()` and `_installed_action_hash = None` operations.

- [ ] **Step 7: Run focused tests and commit GREEN**

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_probe_runner_process_backend.py \
  tests/test_blb_layerwise_runner.py -k 'probe or promotion or validation_full'
```

Expected: PASS, with no change to trial split, seed lists, or returned metric ordering.

```bash
git add blb_stage2_rl/probe_runner.py blb_stage2_rl/sequential_runner.py \
  tests/test_probe_runner_process_backend.py tests/test_blb_layerwise_runner.py
git commit -m "perf: share Stage-2 F1 and F4 probe workers"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

### Task 3: Add Explicit Evidence-Gated Probe Batch Sizes

**Files:**
- Modify: `blb_stage2_rl/runner.py:680-780, 2740-2840, 3118-3200`
- Modify: `blb_stage2_rl/sequential_runner.py:5448-5460, 3374-3418`
- Modify: `rl_tune.py:560-630, 1320-1350`
- Modify: `layer_importance_evaluator.py:2340-2380, 3070-3140`
- Modify: `llama_7B_LayerImportance.sh:540-570, 760-800, 1030-1055, 1810-1830`
- Test: `tests/test_sequential_smoke.py`
- Test: `tests/test_blb_stage2_rl_regressions.py`

- [ ] **Step 1: Write failing fallback and plumbing tests**

Pin these contracts:

```python
self.assertEqual(resolve_probe_batch_size(None, evaluator_batch_size=64), 64)
self.assertEqual(resolve_probe_batch_size(128, evaluator_batch_size=64), 128)
self.assertEqual(len(f1_batches), math.ceil(256 / 128))
self.assertEqual(len(f4_batches), math.ceil(len(validation_full) / 256))
```

Add launcher/static assertions for `--blb-v3-probe-batch-size` and `--blb-v3-validation-probe-batch-size`, including positive-integer rejection and forwarding to Python underscore names.

- [ ] **Step 2: Push and prove RED on the server**

```bash
git add tests/test_sequential_smoke.py tests/test_blb_stage2_rl_regressions.py
git commit -m "test: pin Stage-2 probe batch-size plumbing"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

Run:

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_sequential_smoke.py tests/test_blb_stage2_rl_regressions.py \
  -k 'probe_batch_size or validation_probe_batch_size'
```

Expected: FAIL because the new config and CLI attributes do not exist.

- [ ] **Step 3: Implement compatibility-first batch-size plumbing**

Add optional fields to `BLBStage2TrainConfig`:

```python
probe_batch_size: Optional[int] = None
validation_probe_batch_size: Optional[int] = None
```

Resolve `None` to `ev.batch_size`; reject non-positive explicit values. Pass the resolved F1 value to `_build_probe_batches` and the resolved F4 value to `_build_validation_full_batches`. Record effective sizes and batch counts in run context, manifest, and probe diagnostics. Defaults must create exactly the same batches as `48b03e8`.

- [ ] **Step 4: Run tests and commit GREEN**

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_sequential_smoke.py tests/test_blb_stage2_rl_regressions.py \
  -k 'probe or layerwise or validation_full'
```

Expected: PASS.

```bash
git add blb_stage2_rl/runner.py blb_stage2_rl/sequential_runner.py \
  rl_tune.py layer_importance_evaluator.py llama_7B_LayerImportance.sh \
  tests/test_sequential_smoke.py tests/test_blb_stage2_rl_regressions.py
git commit -m "perf: expose Stage-2 F1 and F4 probe batch sizes"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

### Task 4: Bound Production Training And Diagnostics Memory

**Files:**
- Modify: `blb_stage2_rl/layerwise_runner.py:1834-2785`
- Modify: `blb_stage2_rl/sequential_runner.py:4030-5110`
- Modify: `blb_stage2_rl/diagnostics.py:340-640, 850-1210, 1390-1600`
- Test: `tests/test_blb_layerwise_runner.py`
- Test: `tests/test_rl_data_points.py`

- [ ] **Step 1: Write failing long synthetic-history tests**

Add a production-mode layerwise test that runs many cheap fake episodes and asserts no returned duplicate history:

```python
summary = train_layerwise(..., retain_history=False)
self.assertEqual(summary["episode_records"], [])
self.assertEqual(summary["episode_rewards"], [])
self.assertEqual(summary["ppo_metrics"], [])
self.assertEqual(summary["completed_episodes"], episode_count)
```

Add a diagnostics test that records 5,000 synthetic episodes with a 600-row window and checks exact cumulative state:

```python
self.assertEqual(recorder.episode_count, 5000)
self.assertLessEqual(len(recorder._all_episode_returns), 600)
self.assertEqual(recorder.best_episode_return, 4999.0)
self.assertEqual(recorder.worst_episode_return, 0.0)
```

Restore the JSONL into a fresh recorder and require the same counters, high-water IDs, top-K, histogram, and last 600 health rows. Add negative tests for duplicate, decreasing, and gapped episode/update identities.

- [ ] **Step 2: Push and prove RED on the server**

```bash
git add tests/test_blb_layerwise_runner.py tests/test_rl_data_points.py
git commit -m "test: pin bounded Stage-2 runtime history"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_blb_layerwise_runner.py tests/test_rl_data_points.py \
  -k 'bounded_history or high_water or restore_existing'
```

Expected: FAIL because all episode/PPO objects and identity sets are retained.

- [ ] **Step 3: Make `train_layerwise` history optional**

Add `retain_history: bool = True` for direct-call compatibility. Append `records`, `rewards`, and `ppo_diagnostics` only when true. The production call in `sequential_runner.py` passes `retain_history=False`; training math continues to use the rollout buffer, entropy samples for the current update, strict frontier, and convergence tracker exactly as before.

- [ ] **Step 4: Replace caller history with bounded/scalar state**

Use a deque and counters:

```python
recent_episode_records = collections.deque(maxlen=max(1, int(train_cfg.rollout_size)))
completed_episode_count = int(start_episode)
best_reward_so_far = float(resumed_best.get("reward", -math.inf))
```

The episode callback requires `record.episode_index + 1 == completed_episode_count + 1`, increments the count, updates the scalar best, and appends only to the recent deque. The PPO callback derives window statistics from that deque. Manifest/checkpoint/summary counts use the scalar count, not `len(episode_records)`. Replace the current `reward=float(rewards[-1])` construction with the already-computed scalar `episode_reward`, so disabling returned history cannot alter training.

- [ ] **Step 5: Bound `RLDiagnosticsRecorder` without losing full artifacts**

Add `history_window: Optional[int] = None` and `ppo_history_window: Optional[int] = None`. The new layerwise path uses `600` and `10`; legacy callers retain existing defaults. Maintain exact `_episode_count`, `_ppo_update_count`, best/worst return, first/last PPO entropy, histograms, top-K, first-invalid counts, and Pareto state. Trim only rolling arrays after each append:

```python
def _append_bounded(values, value, limit):
    values.append(value)
    if limit is not None and len(values) > limit:
        del values[:len(values) - limit]
```

`restore_existing()` streams every physical row to rebuild cumulative state but keeps only bounded windows. It validates contiguous episode IDs and contiguous PPO `update` IDs, returns counts and high-water marks, and removes the second JSONL scans plus unbounded `existing_diagnostic_episodes`/`existing_diagnostic_updates` sets from `sequential_runner.py`.

- [ ] **Step 6: Run focused and regression tests, then commit GREEN**

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_blb_layerwise_runner.py tests/test_rl_data_points.py \
  tests/test_blb_diagnostics_static.py
```

Expected: PASS; small direct-call tests still receive complete lists because their default is `retain_history=True`.

```bash
git add blb_stage2_rl/layerwise_runner.py blb_stage2_rl/sequential_runner.py \
  blb_stage2_rl/diagnostics.py tests/test_blb_layerwise_runner.py \
  tests/test_rl_data_points.py tests/test_blb_diagnostics_static.py
git commit -m "perf: bound Stage-2 layerwise runtime history"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

### Task 5: Compact Candidate Evidence With Mixed-Store Restore

**Files:**
- Modify: `blb_stage2_rl/candidate_store.py:1-900`
- Modify: `blb_stage2_rl/layerwise_runner.py:1235-1290, 1600-1760, 2220-2305`
- Test: `tests/test_blb_candidate_store_identity.py`
- Test: `tests/test_blb_layerwise_runner.py`

- [ ] **Step 1: Write failing v2, compaction, and mixed-restore tests**

Build one legacy v1 store and one compact store from the same F1/F4 trial groups. Assert identical logical evidence, candidate keys, promotion state, strict selection input, trial order, and seeds. Inspect physical compact rows:

```python
self.assertEqual(row["record_type"], "candidate_trial_group_v2")
self.assertIn("action_indices", row)
self.assertNotIn("raw_action_indices", row)
self.assertNotIn("effective_action_indices", row)
self.assertNotIn("identity_context", row)
self.assertIn("identity_context_hash", row)
self.assertNotIn("boosted_overrides", row["trial_group_metadata"])
```

Require F4/final rows to retain `boosted_overrides`, and require compact F1 bytes to be less than half the equivalent v1 row for the representative 12-layer payload. Append v2 rows after v1 rows, recover to a checkpoint boundary, and require idempotent replay and fingerprint validation to behave exactly as before.

- [ ] **Step 2: Push and prove RED on the server**

```bash
git add tests/test_blb_candidate_store_identity.py tests/test_blb_layerwise_runner.py
git commit -m "test: pin compact candidate evidence contracts"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_blb_candidate_store_identity.py tests/test_blb_layerwise_runner.py \
  -k 'compact or mixed_store or boosted_overrides or candidate_store'
```

Expected: FAIL because every candidate row currently stores three action arrays, a full identity context, and verbose F1 overrides.

- [ ] **Step 3: Add physical v2 records and identity-context interning**

Add internal record types:

```python
_IDENTITY_CONTEXT_RECORD_TYPE = "candidate_identity_context_v1"
_TRIAL_GROUP_RECORD_TYPES = {"candidate_trial_group_v1", "candidate_trial_group_v2"}
_PROMOTION_STATUS_RECORD_TYPES = {
    "candidate_promotion_status_v1", "candidate_promotion_status_v2",
}
```

Before the first compact candidate row for a context, append one internal context record keyed by `sha256_json(identity_context)`. Compact rows physically store one canonical action, hashes/key, `identity_context_hash`, evidence, and metadata. Keep recovery markers, row boundaries, logical generations, committed sizes, and fingerprint algorithms unchanged.

- [ ] **Step 4: Hydrate old and new rows into one logical API**

Build the context index while scanning active physical rows. For v2 candidate rows, return a copy with compatibility aliases restored in memory:

```python
logical["identity_context"] = context_index[logical["identity_context_hash"]]
logical["raw_action_indices"] = list(logical["action_indices"])
logical["effective_action_indices"] = list(logical["action_indices"])
logical["trial_group_metadata"]["identity_context"] = logical["identity_context"]
```

Index v1 and v2 types, hydrate random-offset reads, and hide internal context records from normal candidate iteration. Fail closed for a missing/hash-mismatched context.

- [ ] **Step 5: Use compact status rows and remove only derivable F1 payload**

Add `CandidateStore.append_promotion_status()` so F4 status rows also avoid duplicate action/context fields. In `train_layerwise`, retain the F1 `boosted_overrides_hash` and provenance but omit materialized overrides; F4 promotion and final revalidation retain the complete serialized override list.

- [ ] **Step 6: Run focused/full candidate tests and commit GREEN**

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_blb_candidate_store_identity.py tests/test_blb_layerwise_runner.py \
  tests/test_blb_cost_semantics.py tests/test_blb_optimizer_cost_consistency.py
```

Expected: PASS, including recovery-marker, duplicate-seed, exact replay, and mixed-store cases.

```bash
git add blb_stage2_rl/candidate_store.py blb_stage2_rl/layerwise_runner.py \
  tests/test_blb_candidate_store_identity.py tests/test_blb_layerwise_runner.py
git commit -m "perf: compact Stage-2 candidate evidence"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

### Task 6: Persist Low-Overhead Runtime Telemetry

**Files:**
- Modify: `blb_stage2_rl/probe_runner.py`
- Modify: `blb_stage2_rl/diagnostics.py:113-290`
- Modify: `blb_stage2_rl/sequential_runner.py:4520-4860`
- Modify: `scripts/stage2_ngpu_ab_compare.py:20-55`
- Create: `scripts/stage2_runtime_optimization_gate.sh`
- Test: `tests/test_probe_runner_process_backend.py`
- Test: `tests/test_blb_layerwise_runner.py`
- Create test: `tests/test_stage2_runtime_optimization_gate.py`

- [ ] **Step 1: Write failing telemetry tests**

Require diagnostics to include `pool_id`, `batch_set_key`, `batch_count`, `process_count`, and worker thread limits. Require episode/update persistence to include candidate bytes written and current/peak RSS, while the semantic comparator ignores only these telemetry fields. Add a static harness test that requires an idle-GPU guard, clean baseline/optimized source roots, cases `base64`, `opt64`, `opt128`, `opt256`, and a strict semantic comparator verdict before any speed verdict.

- [ ] **Step 2: Commit, push, and prove RED**

```bash
git add tests/test_probe_runner_process_backend.py tests/test_blb_layerwise_runner.py \
  tests/test_stage2_runtime_optimization_gate.py
git commit -m "test: pin Stage-2 runtime telemetry"
git push origin codex/stage2-rl-runtime-opt-48b03e8
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_probe_runner_process_backend.py tests/test_blb_layerwise_runner.py \
  -k 'pool_telemetry or candidate_bytes or process_rss'
```

Expected: FAIL because pool/resource telemetry is not persisted.

- [ ] **Step 3: Implement telemetry without per-batch synchronization**

Populate pool fields from already-available runner state. Measure candidate bytes once per episode from file-size deltas. Sample RSS only at PPO update boundaries using `/proc/self/statm` plus `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` fallback. Add no CUDA synchronize beyond existing startup/trial boundaries.

Create `scripts/stage2_runtime_optimization_gate.sh` with this interface:

```bash
: "${BASELINE_ROOT:?set BASELINE_ROOT to the clean 48b03e8 worktree}"
: "${OPTIMIZED_ROOT:?set OPTIMIZED_ROOT to the clean optimized worktree}"
: "${ARTIFACT_DIR:?set ARTIFACT_DIR}"
BATCH_SIZES="${BATCH_SIZES:-64 128 256}"
EPISODES="${EPISODES:-600}"
REWARD_DEVICES="${REWARD_DEVICES:-0,1,2,3,4}"
```

The script refuses dirty roots or non-idle GPUs, runs `base64` once from `BASELINE_ROOT`, runs all optimized sizes from `OPTIMIZED_ROOT`, samples `nvidia-smi`, inventories worker PIDs/threads, copies diagnostics/candidate records, normalizes v1/v2 candidate evidence, and writes `verdict.json` plus `verdict.md`. It exits nonzero unless every accepted optimized case has semantic parity with `base64`; speed ranking considers only passing cases.

- [ ] **Step 4: Run tests and commit**

```bash
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_probe_runner_process_backend.py tests/test_blb_layerwise_runner.py \
  tests/test_stage2_ngpu_speed_targeted_first.py \
  tests/test_stage2_runtime_optimization_gate.py
```

Expected: PASS.

```bash
git add blb_stage2_rl/probe_runner.py blb_stage2_rl/diagnostics.py \
  blb_stage2_rl/sequential_runner.py scripts/stage2_ngpu_ab_compare.py \
  scripts/stage2_runtime_optimization_gate.sh \
  tests/test_probe_runner_process_backend.py tests/test_blb_layerwise_runner.py \
  tests/test_stage2_runtime_optimization_gate.py
git commit -m "perf: record Stage-2 runtime resource telemetry"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

### Task 7: Run Server CPU And Full Regression Gates

**Files:**
- Check: all changed source and tests
- Generate on server: `experiments/server_command_runs/stage2_rl_runtime_opt_${TS}/cpu_tests/`

- [ ] **Step 1: Refresh the clean server worktree through Git**

```bash
git -C /hy-tmp/rfr_runtime_optimization fetch \
  origin refs/heads/codex/stage2-rl-runtime-opt-48b03e8
git -C /hy-tmp/rfr_stage2_rl_runtime_opt checkout --detach FETCH_HEAD
cd /hy-tmp/rfr_stage2_rl_runtime_opt
TS="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_DIR="experiments/server_command_runs/stage2_rl_runtime_opt_${TS}"
mkdir -p "$ARTIFACT_DIR/cpu_tests" "$ARTIFACT_DIR/gpu_gate"
printf '%s\n' "$ARTIFACT_DIR" > /hy-tmp/stage2_rl_runtime_opt_artifact_dir.txt
git -C /hy-tmp/rfr_stage2_rl_runtime_opt status --short
```

Expected: empty status and server SHA equal to local/remote SHA.

- [ ] **Step 2: Compile changed modules on the server**

```bash
set -o pipefail
cd /hy-tmp/rfr_stage2_rl_runtime_opt
ARTIFACT_DIR="$(cat /hy-tmp/stage2_rl_runtime_opt_artifact_dir.txt)"
CUDA_VISIBLE_DEVICES="" python3 -m py_compile \
  blb_stage2_rl/probe_runner.py blb_stage2_rl/runner.py \
  blb_stage2_rl/sequential_runner.py blb_stage2_rl/layerwise_runner.py \
  blb_stage2_rl/candidate_store.py blb_stage2_rl/diagnostics.py \
  rl_tune.py layer_importance_evaluator.py scripts/stage2_ngpu_ab_compare.py \
  2>&1 | tee "$ARTIFACT_DIR/cpu_tests/py_compile.log"
```

Expected: exit `0`.

- [ ] **Step 3: Run focused Stage-2 tests on CPU**

```bash
set -o pipefail
cd /hy-tmp/rfr_stage2_rl_runtime_opt
ARTIFACT_DIR="$(cat /hy-tmp/stage2_rl_runtime_opt_artifact_dir.txt)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="" nice -n 10 python3 -m pytest -q \
  tests/test_probe_runner_process_backend.py \
  tests/test_blb_candidate_store_identity.py \
  tests/test_blb_layerwise_runner.py \
  tests/test_blb_layerwise_policy.py \
  tests/test_blb_stage2_rl_regressions.py \
  tests/test_rl_data_points.py \
  tests/test_sequential_smoke.py \
  2>&1 | tee "$ARTIFACT_DIR/cpu_tests/focused_pytest.log"
```

Expected: all runnable tests PASS; only tests with explicit unavailable-hardware/dependency skip conditions SKIP.

- [ ] **Step 4: Run the repository test gate on CPU**

```bash
set -o pipefail
cd /hy-tmp/rfr_stage2_rl_runtime_opt
ARTIFACT_DIR="$(cat /hy-tmp/stage2_rl_runtime_opt_artifact_dir.txt)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES="" nice -n 10 python3 -m pytest -q tests \
  2>&1 | tee "$ARTIFACT_DIR/cpu_tests/full_pytest.log"
```

Expected: no failures. Capture command, SHA, duration, pass/skip counts, and logs under the server artifact directory.

### Task 8: Run Five-GPU Batch Parity And Performance Gates

**Files:**
- Modify after evidence only: `presets/mrpc-blb-stage2-rl.conf`
- Test after evidence only: `tests/test_stage2_stage1_rl_alignment.py`
- Generate on server: `experiments/server_command_runs/stage2_rl_runtime_opt_${TS}/gpu_gate/`

- [ ] **Step 1: Wait for an uncontaminated GPU lane**

```bash
ps -p 2236414 -o pid=,stat=,etime=,cmd=
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
```

Expected before proceeding: PID `2236414` has exited normally and no unrelated compute process owns GPUs `0..4`. Do not terminate a job to satisfy this condition.

- [ ] **Step 2: Benchmark F1/F4 batch sizes 64, 128, and 256**

For each size, use identical fixed actions, F1/F4 seed sets, five/25 trial counts, datasets, and device IDs `0,1,2,3,4`. Save raw per-trial loss/metric1/metric2, all six probabilities, priorities, rewards, promotions, wall times, process/thread inventory, RSS, and `nvidia-smi` samples.

Expected acceptance: no OOM; raw values are exact, or pass the repository's deterministic numerical-equivalence gate; all semantic decisions and selected actions are identical. Reject any size that fails parity even if faster.

Run the committed gate from the optimized worktree:

```bash
ARTIFACT_DIR="$(cat /hy-tmp/stage2_rl_runtime_opt_artifact_dir.txt)/gpu_gate"
BASELINE_ROOT=/hy-tmp/rfr_stage2_rl_runtime_base \
OPTIMIZED_ROOT=/hy-tmp/rfr_stage2_rl_runtime_opt \
ARTIFACT_DIR="$ARTIFACT_DIR" \
BATCH_SIZES="64 128 256" \
EPISODES=600 \
REWARD_DEVICES="0,1,2,3,4" \
bash scripts/stage2_runtime_optimization_gate.sh
```

Expected: `verdict.json` reports semantic parity for every eligible case and names the fastest passing F1/F4 batch-size pair.

- [ ] **Step 3: Pin only the fastest passing batch sizes**

First return the batch-gate verdict through a server artifact commit and fast-forward it into the local source branch. Then, locally only, set `--blb-v3-probe-batch-size` and `--blb-v3-validation-probe-batch-size` in `presets/mrpc-blb-stage2-rl.conf` from `verdict.json`. Add exact preset assertions to `tests/test_stage2_stage1_rl_alignment.py`. If neither 128 nor 256 passes and improves wall time, pin both to 64 explicitly.

Commit and push the local preset decision:

```bash
git add presets/mrpc-blb-stage2-rl.conf tests/test_stage2_stage1_rl_alignment.py
git commit -m "perf: pin verified Stage-2 probe batch sizes"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

Refresh the optimized server worktree only through Git and run the preset test there:

```bash
git -C /hy-tmp/rfr_runtime_optimization fetch \
  origin refs/heads/codex/stage2-rl-runtime-opt-48b03e8
git -C /hy-tmp/rfr_stage2_rl_runtime_opt checkout --detach FETCH_HEAD
cd /hy-tmp/rfr_stage2_rl_runtime_opt
CUDA_VISIBLE_DEVICES="" python3 -m pytest -q \
  tests/test_stage2_stage1_rl_alignment.py -k probe_batch_size
```

Expected: PASS and the effective preset contains exactly the evidence-selected sizes.

- [ ] **Step 4: Run a seeded baseline-versus-optimized end-to-end smoke**

Run the same bounded episode count from clean `48b03e8` and optimized worktrees. Use `scripts/stage2_ngpu_ab_compare.py --require-equal` to require identical non-telemetry episode rows and PPO updates. Additionally compare normalized candidate evidence, F4 promotion/final decisions, and strict winner/frontier.

Expected: semantic parity PASS. Record baseline/optimized wall time, episodes/hour, F1 mean/P50/P95, F4 mean/P95, RSS, candidate bytes/episode, process count, thread count, per-GPU memory, and utilization. Report end-to-end speedup only from this gate, not from microbenchmarks.

- [ ] **Step 5: Inspect the complete diff**

```bash
git diff 48b03e8...HEAD --check
git diff 48b03e8...HEAD --stat
git status --short
```

Expected: no whitespace errors, no uncommitted source changes, and no reward/evaluation/action/trial/PPO/convergence semantic code changes.

### Task 9: Return Server Evidence Through Git And Verify Three-Way Sync

**Files:**
- Create on server: `experiments/server_command_runs/stage2_rl_runtime_opt_${TS}/`
- Update locally through Git only: the same artifact directory

- [ ] **Step 1: Commit server-generated evidence on a results branch**

On the server, create a results branch from the exact optimized source SHA, add only compact logs/JSON/CSV/Markdown evidence (exclude checkpoints, model caches, and large transient persistent directories), commit, and push through Git. Repeat this procedure once after the batch-only verdict if the local preset still needs to be pinned, then again after the final end-to-end gate.

```bash
SOURCE_SHORT_SHA="$(git rev-parse --short=12 HEAD)"
RESULTS_BRANCH="codex/stage2-rl-runtime-opt-results-${SOURCE_SHORT_SHA}"
ARTIFACT_DIR="$(cat /hy-tmp/stage2_rl_runtime_opt_artifact_dir.txt)"
git switch -c "$RESULTS_BRANCH"
git add "$ARTIFACT_DIR"
git commit -m "perf: record Stage-2 runtime optimization evidence"
git push origin "$RESULTS_BRANCH"
```

- [ ] **Step 2: Fast-forward the evidence locally and push the source branch**

Fetch the results branch locally while the source branch still points to its parent, fast-forward merge it onto `codex/stage2-rl-runtime-opt-48b03e8`, and push. Do not cherry-pick: preserving the server artifact commit object is required for exact three-way SHA equality.

```bash
git fetch origin "$RESULTS_BRANCH"
git merge --ff-only "origin/$RESULTS_BRANCH"
git push origin codex/stage2-rl-runtime-opt-48b03e8
```

- [ ] **Step 3: Verify exact local, GitHub, and server SHA equality**

```bash
git rev-parse HEAD
git ls-remote --heads origin codex/stage2-rl-runtime-opt-48b03e8
git -C /hy-tmp/rfr_stage2_rl_runtime_opt rev-parse HEAD
git status --short
```

Expected: all three SHAs are identical; local and server source worktrees are clean. The original active-run checkout remains separate and unmodified.

- [ ] **Step 4: Write the final audit**

Report exact changed components, parity gates, test totals, baseline and optimized wall times, measured end-to-end speedup, GPU memory/process/thread reduction, RSS reduction, candidate-store byte reduction, selected batch sizes, artifact path, and exact three-way SHA. State clearly if a GPU gate remained pending; do not estimate or extrapolate a project speedup without completed end-to-end evidence.
