# Stage-2 Five-GPU Near-Linear Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the current BERT-large MRPC Stage-2 RL 170-episode run scale from one to five RTX 4090 GPUs by at least 4.5x while preserving the complete scientific state exactly.

**Architecture:** Keep one authoritative learner and one canonical episode stream. Retain deterministic `(action_index, trial_index, trial_seed)` probe identities, but reduce fixed work around the already parallel terminal forwards: enforce a full-state A/B gate, measure worker install/forward phases behind an off-by-default profiler, update persistent BLB wrappers in place, group independent promotion validations, and reuse immutable probe call descriptors. Each runtime candidate is accepted only after a profile-off server A/B proves both lower end-to-end wall time and exact recursive equality.

**Tech Stack:** Python 3.11, PyTorch/CUDA, Hugging Face Transformers 4.44.2, Bash, `unittest`, JSON/JSONL, Git worktrees, five NVIDIA RTX 4090 GPUs.

---

## File Map

- Modify `scripts/stage2_ngpu_speed_ab.sh`: make complete checkpoint/candidate/structured-data equality mandatory and persist exact run roots.
- Modify `tests/test_stage2_ngpu_speed_targeted_first.py`: lock the full-state gate and the exact five-GPU BERT-large command contract.
- Modify `blb_stage2_rl/probe_runner.py`: add off-by-default install/forward phase timing and expose it as efficiency telemetry.
- Modify `blb_stage2_rl/env.py`: preserve per-worker phase telemetry while splitting grouped probe diagnostics back into canonical episode order.
- Modify `blb_stage2_rl/sequential_runner.py`: persist per-worker phase telemetry in diagnostics and structured episode records.
- Modify `scripts/stage2_ngpu_ab_compare.py`: exclude the new timing-only fields from scientific equality.
- Modify `scripts/elastic_rl_scaling_ab.py`: exclude the new timing-only fields from recursive scientific equality.
- Modify `tests/test_probe_runner_process_backend.py`: verify phase accounting without changing task assignment or result order.
- Modify `tests/test_blb_layerwise_runner.py`: verify phase telemetry survives terminal-record materialization.
- Modify `tests/test_stage2_ngpu_ab_compare.py`: verify phase telemetry is ignored while scientific fields remain strict.
- Modify `tests/test_elastic_rl_scaling_ab.py`: verify recursive comparison treats phase telemetry as efficiency-only.
- Modify `function_handler.py`: reuse already-installed Block 1 modules and update their configuration in place.
- Modify `tests/test_blb_stage2_rl_regressions.py`: prove in-place Block 1 reconfiguration is bit-exact and preserves wrapper identity.
- Modify `blb_stage2_rl/inference_eval.py`: prebuild immutable probe model kwargs once per resident batch set.
- Modify `blb_stage2_rl/probe_runner.py`: store the prebuilt descriptors on each persistent worker.
- Modify `tests/test_blb_probe_metric_aggregation.py`: prove old and prebuilt descriptor paths return identical metrics and model calls.
- Modify `blb_stage2_rl/sequential_runner.py`: submit uncached promotion validations in deterministic groups and finalize in original episode order.
- Modify `tests/test_blb_layerwise_runner.py`: prove grouped validation preserves decisions, rewards, candidate order, and PPO order.
- Create `scripts/stage2_five_gpu_runtime_gate.sh`: exact target workload wrapper with fixed server-safe runtime settings and hard 4.5x/90% gates.
- Create `tests/test_stage2_five_gpu_runtime_gate.py`: verify the wrapper's effective command without touching GPUs.
- Create `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/README.md`: compact accepted/rejected optimization ledger and final evidence index.
- Create `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/candidate_ledger.json`: measured disposition of every runtime candidate.
- Create `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/final_summary.json`: machine-readable final speed, equality, source, and parity result.
- Create `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/SHA256SUMS`: digest compact evidence committed to Git.

## Task 1: Establish The Unmodified Server Baseline

**Files:**
- Read: `requirements.txt`
- Read: `scripts/stage2_ngpu_speed_ab.sh`
- Generate on server only: `/hy-tmp/stage2_five_gpu_runtime_20260728/baseline/`

- [ ] **Step 1: Verify the clean three-way source starting point**

Run locally:

```bash
git status --short
git rev-parse HEAD
git rev-parse HEAD^{tree}
git ls-remote origin refs/heads/codex/stage2-five-gpu-runtime-20260728
```

Expected:

```text
status output is empty
HEAD is bce4212641e7b24e786a2b27f6c9888c93cf7fa2
tree is ce47a9144d68ca3461c25e710c1e2f6201ae930a
the remote branch resolves to the same full HEAD
```

Run on the server:

```bash
cd /hy-tmp/rfr-stage2-five-gpu-runtime-20260728
git fetch origin codex/stage2-five-gpu-runtime-20260728
git pull --ff-only origin codex/stage2-five-gpu-runtime-20260728
git status --short
git rev-parse HEAD
git rev-parse HEAD^{tree}
```

Expected: tracked-clean server source with the same commit and tree.

- [ ] **Step 2: Build an isolated server runtime without replacing working CUDA Torch**

Run on the server:

```bash
python3 -m venv --system-site-packages /hy-tmp/rfr-stage2-five-gpu-runtime-venv
source /hy-tmp/rfr-stage2-five-gpu-runtime-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import torch
import transformers

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 5
assert transformers.__version__ == "4.44.2"
print(torch.__version__)
print(torch.version.cuda)
print([torch.cuda.get_device_name(i) for i in range(5)])
PY
```

Expected: five entries, all `NVIDIA GeForce RTX 4090`, and no import failure.

- [ ] **Step 3: Capture immutable hardware and source manifests**

Run on the server:

```bash
mkdir -p /hy-tmp/stage2_five_gpu_runtime_20260728/baseline
nvidia-smi -q > /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/nvidia_smi_q.txt
lscpu > /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/lscpu.txt
free -h > /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/free_h.txt
df -h /hy-tmp > /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/df_hy_tmp.txt
git rev-parse HEAD > /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/source_commit.txt
git rev-parse HEAD^{tree} > /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/source_tree.txt
git status --porcelain=v1 > /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/source_status.txt
```

Expected: `source_status.txt` is empty.

- [ ] **Step 4: Run the starting-commit matched 170-episode BERT-large control**

Run on the server with no other GPU process:

```bash
source /hy-tmp/rfr-stage2-five-gpu-runtime-venv/bin/activate
cd /hy-tmp/rfr-stage2-five-gpu-runtime-20260728
RUN_ID=stage2_five_gpu_starting_170 \
ARTIFACT_DIR=/hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170 \
MODEL_TYPE=bert-large \
EPISODES_AB=170 \
ONE_DEVS=0 \
MANY_DEVS=0,1,2,3,4 \
ONE_WORKERS_PER_DEVICE=1 \
MANY_WORKERS_PER_DEVICE=1 \
POLICY_DEVICE=worker \
DYNAMIC_ASSIGNMENT=1 \
BATCH_SIZE=64 \
KTRIALS=5 \
ONLINE_KTRIALS=5 \
PPO_UPDATE_INTERVAL=120 \
MIN_SPEEDUP=0 \
TIMEOUT_SECONDS=21600 \
GPU_SAMPLE_INTERVAL_SECONDS=2 \
bash scripts/stage2_ngpu_speed_ab.sh
```

Expected:

```text
both arms write exactly 170 episodes
quality/effect equality is PASS
PPO update equality is PASS
all five GPUs have sampled active intervals
the measured speedup is recorded but is not yet an acceptance result
```

- [ ] **Step 5: Record the baseline bottleneck budget**

Run on the server:

```bash
python scripts/stage2_ngpu_ab_compare.py \
  --one /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/one_episodes.jsonl \
  --many /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/many_episodes.jsonl \
  --one-wall /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/one_wall_seconds.txt \
  --many-wall /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/many_wall_seconds.txt \
  --one-ppo /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/one_ppo_updates.jsonl \
  --many-ppo /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/many_ppo_updates.jsonl \
  --one-log /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/one_launch.log \
  --many-log /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/many_launch.log \
  --require-equal \
  --out /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/bottleneck_verdict.txt
```

Expected: a wall/probe ratio, one/five probe means, per-device trial balance, and no quality/effect or PPO difference.

## Task 2: Make The A/B Gate Compare The Complete Scientific State

**Files:**
- Modify: `scripts/stage2_ngpu_speed_ab.sh:42-63`
- Modify: `scripts/stage2_ngpu_speed_ab.sh:337-427`
- Modify: `scripts/stage2_ngpu_speed_ab.sh:488-525`
- Test: `tests/test_stage2_ngpu_speed_targeted_first.py`

- [ ] **Step 1: Write the failing harness contract test**

Add this test:

```python
def test_ngpu_gate_requires_recursive_scientific_state_comparison(self):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "stage2_ngpu_speed_ab.sh"
    ).read_text(encoding="utf-8")

    self.assertIn('REQUIRE_FULL_STATE_EQUALITY="${REQUIRE_FULL_STATE_EQUALITY:-1}"', script)
    self.assertIn('DATA_POINTS_ROOT="${DATA_POINTS_ROOT:-rl_training_data_points}"', script)
    self.assertIn('"${ARTIFACT_DIR}/${label}_run_root.txt"', script)
    self.assertIn('"${ARTIFACT_DIR}/HEAD.txt"', script)
    self.assertIn('"${ARTIFACT_DIR}/TREE.txt"', script)
    self.assertIn("scripts/elastic_rl_scaling_ab.py", script)
    self.assertIn("--data-points-root \"$DATA_POINTS_ROOT\"", script)
    self.assertIn("--output \"${ARTIFACT_DIR}/strict_scientific_equivalence.json\"", script)
```

- [ ] **Step 2: Push the red test and run it on the server**

Run locally:

```bash
git add tests/test_stage2_ngpu_speed_targeted_first.py
git commit -m "test(stage2): require full-state scaling evidence"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run on the server after `git pull --ff-only`:

```bash
source /hy-tmp/rfr-stage2-five-gpu-runtime-venv/bin/activate
python -m unittest tests.test_stage2_ngpu_speed_targeted_first -v
```

Expected: the new test fails because the shell harness does not yet invoke `elastic_rl_scaling_ab.py`.

- [ ] **Step 3: Persist run roots and invoke the recursive comparator**

Add near the shell defaults:

```bash
DATA_POINTS_ROOT="${DATA_POINTS_ROOT:-rl_training_data_points}"
REQUIRE_FULL_STATE_EQUALITY="${REQUIRE_FULL_STATE_EQUALITY:-1}"
REUSE_ONE_RUN_ROOT="${REUSE_ONE_RUN_ROOT:-}"
```

After creating `ARTIFACT_DIR`, capture the exact source identity used by the
run:

```bash
git rev-parse HEAD > "${ARTIFACT_DIR}/HEAD.txt"
git rev-parse HEAD^{tree} > "${ARTIFACT_DIR}/TREE.txt"
```

After `ep_path` is found in `run_case`, add:

```bash
printf '%s\n' "$persistent_root" > "${ARTIFACT_DIR}/${label}_run_root.txt"
```

In `reuse_one_case`, require and copy the complete root:

```bash
if [ "$REQUIRE_FULL_STATE_EQUALITY" = "1" ]; then
  if [ -z "$REUSE_ONE_RUN_ROOT" ] || [ ! -d "$REUSE_ONE_RUN_ROOT" ]; then
    echo "[FATAL] strict reused control requires REUSE_ONE_RUN_ROOT"
    exit 2
  fi
  printf '%s\n' "$REUSE_ONE_RUN_ROOT" > "${ARTIFACT_DIR}/one_run_root.txt"
fi
```

After the existing episode/PPO comparator, add:

```bash
if [ "$REQUIRE_FULL_STATE_EQUALITY" = "1" ]; then
  one_run_root="$(cat "${ARTIFACT_DIR}/one_run_root.txt")"
  many_run_root="$(cat "${ARTIFACT_DIR}/many_run_root.txt")"
  python3 scripts/elastic_rl_scaling_ab.py compare \
    --control "$one_run_root" \
    --candidate "$many_run_root" \
    --stage stage2 \
    --data-points-root "$DATA_POINTS_ROOT" \
    --output "${ARTIFACT_DIR}/strict_scientific_equivalence.json"
fi
```

- [ ] **Step 4: Run the focused tests**

Run on the server:

```bash
python -m unittest \
  tests.test_stage2_ngpu_speed_targeted_first \
  tests.test_elastic_rl_scaling_ab \
  tests.test_stage2_ngpu_ab_compare -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit and push the complete gate**

Run locally:

```bash
git add scripts/stage2_ngpu_speed_ab.sh tests/test_stage2_ngpu_speed_targeted_first.py
git commit -m "test(stage2): gate scaling on full scientific state"
git push origin codex/stage2-five-gpu-runtime-20260728
```

## Task 3: Add Off-By-Default Probe Phase Profiling

**Files:**
- Modify: `blb_stage2_rl/probe_runner.py:61-104`
- Modify: `blb_stage2_rl/probe_runner.py:481-507`
- Modify: `blb_stage2_rl/probe_runner.py:669-698`
- Modify: `blb_stage2_rl/probe_runner.py:1801-1952`
- Modify: `blb_stage2_rl/probe_runner.py:2428-2465`
- Modify: `blb_stage2_rl/env.py:1280-1391`
- Modify: `blb_stage2_rl/sequential_runner.py:1260-1325`
- Modify: `blb_stage2_rl/sequential_runner.py:1596-1619`
- Modify: `blb_stage2_rl/sequential_runner.py:7896-7913`
- Modify: `scripts/stage2_ngpu_ab_compare.py:18-40`
- Modify: `scripts/elastic_rl_scaling_ab.py:32-96`
- Test: `tests/test_probe_runner_process_backend.py`
- Test: `tests/test_blb_layerwise_runner.py`
- Test: `tests/test_stage2_ngpu_ab_compare.py`
- Test: `tests/test_elastic_rl_scaling_ab.py`

- [ ] **Step 1: Write failing process-backend phase tests**

Add a fake clock and assert that grouped results keep their order while phase totals are attached:

```python
def test_grouped_process_probe_reports_install_and_forward_phases(self):
    events = []
    runner, remotes = self._runner_with_process_count(
        events,
        process_count=4,
    )
    for remote in remotes:
        original_receive = remote.receive

        def receive_with_phases(operation, receive=original_receive):
            payload = receive(operation)
            if operation == "run_action_trial_groups":
                payload["install_seconds"] = 0.25
                payload["trial_seconds"] = 1.50
            return payload

        remote.receive = receive_with_phases

    actions = [object(), object()]
    results = runner.run_action_trial_groups(
        actions,
        base_seeds=[101, 202],
        k=5,
    )

    self.assertEqual(len(results), 2)
    self.assertEqual([len(row) for row in results], [5, 5])
    self.assertEqual(
        runner.last_diagnostics.per_worker_install_seconds[1:],
        [0.25, 0.25, 0.25, 0.25],
    )
    self.assertEqual(
        runner.last_diagnostics.per_worker_trial_seconds[1:],
        [1.50, 1.50, 1.50, 1.50],
    )
```

Add an environment-resolution test:

```python
def test_probe_phase_profile_defaults_off(self):
    with mock.patch.dict(os.environ, {}, clear=True):
        self.assertFalse(_probe_runner.probe_phase_profile_enabled())
    with mock.patch.dict(
        os.environ,
        {"BLB_STAGE2_PROBE_PHASE_PROFILE": "1"},
        clear=True,
    ):
        self.assertTrue(_probe_runner.probe_phase_profile_enabled())
```

- [ ] **Step 2: Push the red tests and run them on the server**

Run locally:

```bash
git add tests/test_probe_runner_process_backend.py
git commit -m "test(stage2): specify probe phase telemetry"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run on the server:

```bash
python -m unittest tests.test_probe_runner_process_backend -v
```

Expected: failures for the missing resolver and diagnostic fields.

- [ ] **Step 3: Add the exact profiler switch and fields**

Add:

```python
def probe_phase_profile_enabled() -> bool:
    return str(
        os.environ.get("BLB_STAGE2_PROBE_PHASE_PROFILE", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
```

Extend `ProbeRunnerDiagnostics`:

```python
per_worker_install_seconds: List[float] = field(default_factory=list)
per_worker_trial_seconds: List[float] = field(default_factory=list)
```

In child `run_action_trial_groups`, use:

```python
phase_profile = probe_phase_profile_enabled()
install_seconds = 0.0
trial_seconds = 0.0
grouped_results = []
for group in payload["action_groups"]:
    action_index = int(group["action_index"])
    base_seed = int(group["base_seed"])
    install_started = time.perf_counter() if phase_profile else 0.0
    worker.install(group["decoded"])
    if phase_profile:
        install_seconds += time.perf_counter() - install_started
    for trial_idx in group["trial_indices"]:
        trial_index = int(trial_idx)
        trial_started = time.perf_counter() if phase_profile else 0.0
        trial_result = worker.run_trial(
            trial_index,
            base_seed,
            batch_set_key,
        )
        if phase_profile:
            trial_seconds += time.perf_counter() - trial_started
        grouped_results.append((
            action_index,
            trial_index,
            trial_result,
        ))
result = {
    "results": grouped_results,
    "install_seconds": float(install_seconds),
    "trial_seconds": float(trial_seconds),
}
```

Mirror the same counters around local-primary `install` and `run_trial` calls. Pass both arrays into `_set_group_diagnostics`, and add to `diagnostics_payload`:

```python
"per_worker_install_seconds": [
    float(value) for value in diag.per_worker_install_seconds
],
"per_worker_trial_seconds": [
    float(value) for value in diag.per_worker_trial_seconds
],
```

When profiling is disabled, leave both arrays as zeroes and perform no per-action or per-trial `perf_counter` call.

- [ ] **Step 4: Persist the timing-only fields without making them scientific state**

When `env.py` splits a grouped result back into per-action diagnostics, retain
the group totals and amortized per-action values:

```python
group_worker_install_seconds = [
    float(value)
    for value in group_diag.get("per_worker_install_seconds", ())
]
group_worker_trial_seconds = [
    float(value)
    for value in group_diag.get("per_worker_trial_seconds", ())
]
```

Add these values to each per-action `diag`:

```python
"per_worker_install_seconds": [
    value / max(1, action_count)
    for value in group_worker_install_seconds
],
"per_worker_trial_seconds": [
    value / max(1, action_count)
    for value in group_worker_trial_seconds
],
```

Add these timing-only fields to `EpisodeRecord`, copy them in
`_apply_terminal_info_to_record`, and emit them in the structured episode
writer:

```python
terminal_probe_worker_install_seconds: List[float] = field(default_factory=list)
terminal_probe_worker_trial_seconds: List[float] = field(default_factory=list)
```

Add both names to `TIMING_OR_DEVICE_KEYS` in
`stage2_ngpu_ab_compare.py` and `_EFFICIENCY_TELEMETRY_KEYS` in
`elastic_rl_scaling_ab.py`. Add tests proving that changing only either array
does not fail equality, while changing reward, metrics, action, candidate, PPO,
or checkpoint state still fails.

- [ ] **Step 5: Run the focused tests and a process-order regression**

Run on the server:

```bash
python -m unittest \
  tests.test_probe_runner_process_backend \
  tests.test_blb_stage2_rl_regressions \
  tests.test_blb_layerwise_runner \
  tests.test_stage2_ngpu_ab_compare \
  tests.test_elastic_rl_scaling_ab -v
```

Expected: all tests pass, including remote-submit-before-local execution and canonical grouped result order.

- [ ] **Step 6: Commit and push the profiler**

Run locally:

```bash
git add \
  blb_stage2_rl/probe_runner.py \
  blb_stage2_rl/env.py \
  blb_stage2_rl/sequential_runner.py \
  scripts/stage2_ngpu_ab_compare.py \
  scripts/elastic_rl_scaling_ab.py \
  tests/test_probe_runner_process_backend.py \
  tests/test_blb_layerwise_runner.py \
  tests/test_stage2_ngpu_ab_compare.py \
  tests/test_elastic_rl_scaling_ab.py
git commit -m "perf(stage2): expose probe install and forward phases"
git push origin codex/stage2-five-gpu-runtime-20260728
```

- [ ] **Step 7: Run a profile-on 40-episode five-GPU screen**

Run on the server:

```bash
BLB_STAGE2_PROBE_PHASE_PROFILE=1 \
RUN_ID=stage2_five_gpu_phase_profile_40 \
ARTIFACT_DIR=/hy-tmp/stage2_five_gpu_runtime_20260728/profile_40 \
MODEL_TYPE=bert-large \
EPISODES_AB=40 \
ONE_DEVS=0 \
MANY_DEVS=0,1,2,3,4 \
MIN_SPEEDUP=0 \
REQUIRE_FULL_STATE_EQUALITY=1 \
TIMEOUT_SECONDS=10800 \
bash scripts/stage2_ngpu_speed_ab.sh
```

Expected: exact full-state equality and phase evidence showing whether installation, forward execution, or parent/IPC remainder dominates. This run ranks candidates only; it is not the speed acceptance run.

## Task 4: Reconfigure Persistent Block 1 Modules In Place

**Files:**
- Modify: `function_handler.py:1174-1231`
- Modify: `function_handler.py:1234-1276`
- Modify: `function_handler.py:3899-3995`
- Test: `tests/test_blb_stage2_rl_regressions.py`

- [ ] **Step 1: Write a bit-exact in-place reconfiguration test**

Add:

```python
def test_block1_layernorm_reconfiguration_reuses_module_and_is_exact(self):
    from function_handler import (
        NoisyBlock1LayerNorm,
        make_block1_default_config,
        reseed_noise_rng_for_device,
    )

    original = torch.nn.LayerNorm(8)
    cfg_a = make_block1_default_config()
    cfg_b = make_block1_default_config()
    cfg_b.output_truncation_k = 9
    reused = NoisyBlock1LayerNorm(original, cfg=cfg_a)
    fresh = NoisyBlock1LayerNorm(original, cfg=cfg_b)
    input_tensor = torch.linspace(-1.0, 1.0, 16).view(1, 2, 8)

    reused.set_block1_cfg(cfg_b)
    reseed_noise_rng_for_device(input_tensor.device, 12345)
    reused_output = reused(input_tensor)
    reseed_noise_rng_for_device(input_tensor.device, 12345)
    fresh_output = fresh(input_tensor)

    self.assertTrue(torch.equal(reused_output, fresh_output))
    self.assertIs(reused.weight, original.weight)
    self.assertIs(reused.bias, original.bias)
```

Add a handler identity test using the existing tiny BERT test fixture:

```python
def test_repeated_block1_install_preserves_wrapper_identity(self):
    from types import SimpleNamespace
    from function_handler import (
        ReversibleLayerHandler,
        make_block1_default_config,
    )

    layer = SimpleNamespace(
        output=SimpleNamespace(
            dense=torch.nn.Linear(8, 8),
            LayerNorm=torch.nn.LayerNorm(8),
        ),
    )
    model = SimpleNamespace(
        bert=SimpleNamespace(
            encoder=SimpleNamespace(layer=[layer]),
        ),
    )
    handler = ReversibleLayerHandler.__new__(ReversibleLayerHandler)
    handler._arch = "bert"
    handler.model = model
    handler.original_block1_ffn2 = {}
    handler.original_block1_layernorm = {}
    handler.block1_cfg_per_layer = {}
    handler._check_blb_legacy_conflict = (
        lambda *_args, **_kwargs: None
    )
    layer_name = "model.bert.encoder.layer"
    cfg_a = make_block1_default_config(output_truncation_k=8)
    cfg_b = make_block1_default_config(output_truncation_k=11)

    handler.replace_layer_block1_noise(
        layer_indices=[0],
        layer_name=layer_name,
        cfg=cfg_a,
    )
    first = layer.output.LayerNorm
    handler.replace_layer_block1_noise(
        layer_indices=[0],
        layer_name=layer_name,
        cfg=cfg_b,
    )
    second = layer.output.LayerNorm

    self.assertIs(first, second)
    self.assertIs(second.cfg, cfg_b)
```

- [ ] **Step 2: Push the red tests and run them on the server**

Run locally:

```bash
git add tests/test_blb_stage2_rl_regressions.py
git commit -m "test(blb): specify persistent block1 reconfiguration"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run on the server:

```bash
python -m unittest tests.test_blb_stage2_rl_regressions -v
```

Expected: failures because `set_block1_cfg` does not exist and repeated installation replaces the LayerNorm object.

- [ ] **Step 3: Add the setter and reuse the resident module**

Add to `NoisyBlock1LayerNorm`:

```python
def set_block1_cfg(
        self,
        cfg: Optional[Block1NoiseConfig],
        ) -> None:
    """Install, replace, or disable the Block 1 head configuration."""
    self.cfg = cfg
```

Replace the Block 1 LayerNorm installation block with:

```python
current_ln = layer.output.LayerNorm
if isinstance(current_ln, NoisyBlock1LayerNorm):
    current_ln.set_block1_cfg(cfg)
else:
    if i not in self.original_block1_layernorm:
        self.original_block1_layernorm[i] = current_ln
    source_ln = self.original_block1_layernorm[i]
    new_ln = NoisyBlock1LayerNorm(source_ln, cfg)
    new_ln.train(source_ln.training)
    try:
        ref_param = source_ln.weight
        new_ln = new_ln.to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
    except Exception:
        pass
    layer.output.LayerNorm = new_ln
```

This keeps Block 2's `cfg2` on the same module until the subsequent Block 2 update and does not alter any arithmetic or RNG call.

- [ ] **Step 4: Run focused correctness and real CUDA equivalence**

Run on the server:

```bash
python -m unittest \
  tests.test_blb_stage2_rl_regressions \
  tests.test_blb_truncation_backends \
  tests.test_probe_runner_process_backend -v
```

Then run the existing real probe equivalence test selected by:

```bash
python -m unittest discover -s tests -p 'test_blb_*.py' -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit and push the Block 1 fast path**

Run locally:

```bash
git add function_handler.py tests/test_blb_stage2_rl_regressions.py
git commit -m "perf(blb): reconfigure persistent block1 layernorms in place"
git push origin codex/stage2-five-gpu-runtime-20260728
```

- [ ] **Step 6: Run matched profile-off candidate screening**

Run on the server:

```bash
BLB_STAGE2_PROBE_PHASE_PROFILE=0 \
RUN_ID=stage2_block1_reconfigure_170 \
ARTIFACT_DIR=/hy-tmp/stage2_five_gpu_runtime_20260728/block1_reconfigure_170 \
MODEL_TYPE=bert-large \
EPISODES_AB=170 \
ONE_DEVS=0 \
MANY_DEVS=0,1,2,3,4 \
MIN_SPEEDUP=0 \
REQUIRE_FULL_STATE_EQUALITY=1 \
TIMEOUT_SECONDS=21600 \
bash scripts/stage2_ngpu_speed_ab.sh
```

Expected: `strict_scientific_equivalence.json` has `"equal": true` and five-GPU wall time is lower than the starting five-GPU wall time. If wall time does not improve beyond run-to-run noise, omit this commit from the final aggregate and record it as rejected evidence.

If rejected, run locally before Task 5:

```bash
candidate_commit="$(git log -1 --format=%H --grep='^perf(blb): reconfigure persistent block1 layernorms in place$')"
git revert --no-edit "$candidate_commit"
git push origin codex/stage2-five-gpu-runtime-20260728
```

## Task 5: Reuse Immutable Probe Call Descriptors

**Files:**
- Modify: `blb_stage2_rl/inference_eval.py:295-380`
- Modify: `blb_stage2_rl/probe_runner.py:290-357`
- Test: `tests/test_blb_probe_metric_aggregation.py`

- [ ] **Step 1: Write an exact old/new descriptor test**

Add:

```python
def test_prepared_probe_batches_preserve_model_calls_and_metrics(self):
    from types import SimpleNamespace
    import torch
    from blb_stage2_rl.inference_eval import (
        prepare_probe_batches,
        run_installed_probe_trial,
    )

    class RecordingClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            self.calls.append({
                "input_ids": input_ids.detach().cpu().tolist(),
                "attention_mask": attention_mask.detach().cpu().tolist(),
                "token_type_ids": (
                    None
                    if token_type_ids is None
                    else token_type_ids.detach().cpu().tolist()
                ),
            })
            score = input_ids[:, 0].float()
            return SimpleNamespace(
                logits=torch.stack((score, -score), dim=1),
            )

    model = RecordingClassifier()
    batches = [
        SimpleNamespace(
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
            labels=torch.tensor([0, 1]),
            token_type_ids=torch.zeros((2, 2), dtype=torch.long),
        ),
        SimpleNamespace(
            input_ids=torch.tensor([[5, 6]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            labels=torch.tensor([0]),
            token_type_ids=torch.zeros((1, 2), dtype=torch.long),
        ),
    ]
    prepared = prepare_probe_batches(batches)

    old_result = run_installed_probe_trial(
        model,
        batches,
        is_regression=False,
        metric_profile="mrpc",
        restore_training=False,
    )
    old_calls = list(model.calls)
    model.calls.clear()
    new_result = run_installed_probe_trial(
        model,
        prepared,
        is_regression=False,
        metric_profile="mrpc",
        restore_training=False,
    )

    self.assertEqual(old_result, new_result)
    self.assertEqual(old_calls, model.calls)
```

- [ ] **Step 2: Push the red test and run it on the server**

Run locally:

```bash
git add tests/test_blb_probe_metric_aggregation.py
git commit -m "test(stage2): lock immutable probe descriptors"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run on the server:

```bash
python -m pytest tests/test_blb_probe_metric_aggregation.py -q
```

Expected: import failure for `prepare_probe_batches`.

- [ ] **Step 3: Add the immutable descriptor**

Add:

```python
@dataclass(frozen=True)
class PreparedProbeBatch:
    model_kwargs: Mapping[str, torch.Tensor]
    labels: torch.Tensor


def prepare_probe_batches(
        batches: Sequence[Any],
        ) -> Tuple[PreparedProbeBatch, ...]:
    return tuple(
        PreparedProbeBatch(
            model_kwargs=probe_batch_to_model_kwargs(batch),
            labels=batch.labels,
        )
        for batch in batches
    )
```

Also add `Mapping` to the imports from `typing`. At the start of `run_installed_probe_trial`, normalize once:

```python
prepared_batches = (
    tuple(probe_batches)
    if probe_batches
    and isinstance(probe_batches[0], PreparedProbeBatch)
    else prepare_probe_batches(probe_batches)
)
```

Use:

```python
for batch in prepared_batches:
    outputs = model(**batch.model_kwargs)
    _loss_t, logits = output_loss_and_logits(outputs)
    trial_outputs.append((logits, batch.labels))
```

In `ProbeWorker.__post_init__` and `register_batch_set`, call `prepare_probe_batches` once after batches are resident on the device. Keep labels and model kwargs pointing to the same tensors; do not clone or repartition a batch.

- [ ] **Step 4: Run exact metric and process tests**

Run on the server:

```bash
python -m pytest tests/test_blb_probe_metric_aggregation.py -q
python -m unittest \
  tests.test_probe_runner_process_backend \
  tests.test_blb_stage2_rl_regressions -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit, push, and screen**

Run locally:

```bash
git add \
  blb_stage2_rl/inference_eval.py \
  blb_stage2_rl/probe_runner.py \
  tests/test_blb_probe_metric_aggregation.py
git commit -m "perf(stage2): reuse resident probe call descriptors"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run the same 170-episode profile-off command from Task 4 with:

```bash
RUN_ID=stage2_probe_descriptors_170
ARTIFACT_DIR=/hy-tmp/stage2_five_gpu_runtime_20260728/probe_descriptors_170
```

Expected: complete equality and lower five-GPU wall time. If there is no repeatable end-to-end gain, omit this candidate from the final aggregate.

If rejected, run locally before Task 6:

```bash
candidate_commit="$(git log -1 --format=%H --grep='^perf(stage2): reuse resident probe call descriptors$')"
git revert --no-edit "$candidate_commit"
git push origin codex/stage2-five-gpu-runtime-20260728
```

## Task 6: Group Promotion Validation Without Reordering Scientific State

**Files:**
- Modify: `blb_stage2_rl/sequential_runner.py:2074-2209`
- Test: `tests/test_blb_layerwise_runner.py`

- [ ] **Step 1: Write a deterministic grouped-validation helper test**

Add:

```python
def test_grouped_promotion_validation_preserves_requested_index_order(self):
    from blb_stage2_rl.sequential_runner import (
        _evaluate_prepared_validation_groups,
    )

    calls = []

    class Base:
        def evaluate_prepared_terminal_batch(self, prepared, **kwargs):
            items = list(prepared)
            calls.append((items, dict(kwargs)))
            return [
                (None, float(item["reward"]), True, {"metrics": item["metrics"]})
                for item in items
            ]

    prepared = [
        {"reward": 10.0, "metrics": "m0"},
        {"reward": 20.0, "metrics": "m1"},
        {"reward": 30.0, "metrics": "m2"},
        {"reward": 40.0, "metrics": "m3"},
    ]
    results = _evaluate_prepared_validation_groups(
        base_env=Base(),
        prepared_by_draft=prepared,
        draft_indices=[3, 1, 2],
        batch_size=2,
        trial_count=25,
    )

    self.assertEqual(list(results), [3, 1, 2])
    self.assertEqual([result[1] for result in results.values()], [40.0, 20.0, 30.0])
    self.assertEqual([len(call[0]) for call in calls], [2, 1])
    self.assertTrue(all(call[1]["validation_required"] for call in calls))
    self.assertTrue(all(call[1]["num_trials_per_action"] == 25 for call in calls))
```

- [ ] **Step 2: Push the red test and run it on the server**

Run locally:

```bash
git add tests/test_blb_layerwise_runner.py
git commit -m "test(stage2): specify grouped promotion validation"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run on the server:

```bash
python -m unittest \
  tests.test_blb_layerwise_runner.LayerwiseRolloutTests.test_grouped_promotion_validation_preserves_requested_index_order -v
```

Expected: import failure for `_evaluate_prepared_validation_groups`.

- [ ] **Step 3: Split validation planning, execution, and ordered application**

Add this module-level helper above `train_layerwise`:

```python
def _evaluate_prepared_validation_groups(
        *,
        base_env: Any,
        prepared_by_draft: Sequence[Mapping[str, Any]],
        draft_indices: Sequence[int],
        batch_size: int,
        trial_count: int,
        ) -> Dict[int, Tuple[np.ndarray, float, bool, Dict[str, Any]]]:
    ordered_indices = [int(index) for index in draft_indices]
    results_by_draft: Dict[
        int,
        Tuple[np.ndarray, float, bool, Dict[str, Any]],
    ] = {}
    group_size = max(1, int(batch_size))
    for start in range(0, len(ordered_indices), group_size):
        batch_indices = ordered_indices[start:start + group_size]
        prepared_batch = [
            prepared_by_draft[index]
            for index in batch_indices
        ]
        results = base_env.evaluate_prepared_terminal_batch(
            prepared_batch,
            num_trials_per_action=int(trial_count),
            validation_required=True,
        )
        if len(results) != len(batch_indices):
            raise RuntimeError(
                "grouped promotion validation result count differs from request"
            )
        for local_index, result in enumerate(results):
            results_by_draft[batch_indices[local_index]] = result
    return results_by_draft
```

Inside `flush_pending_terminal_drafts`, after ordinary K=5 metrics are attached, build:

```python
validation_required_by_draft = [False] * len(drafts)
validation_result_by_draft: List[
    Optional[Tuple[np.ndarray, float, bool, Dict[str, Any]]]
] = [None] * len(drafts)
uncached_validation_indices: List[int] = []
```

Compute `validation_required` for every draft in current order before calling `_finalize_completed_record`. For cached entries, populate `validation_result_by_draft`. For uncached entries, append the index.

Evaluate uncached entries in bounded deterministic groups:

```python
fresh_validation_results = _evaluate_prepared_validation_groups(
    base_env=env.base,
    prepared_by_draft=prepared_by_draft,
    draft_indices=uncached_validation_indices,
    batch_size=int(terminal_eval_batch_size),
    trial_count=int(promotion_validation_trials),
)
for draft_index, validation_result in fresh_validation_results.items():
    validation_result_by_draft[draft_index] = validation_result
    validation_metrics = validation_result[3].get("metrics")
    action_key = str(
        drafts[draft_index].get("terminal_action_hash") or ""
    )
    if validation_metrics is not None and action_key:
        validation_metric_cache[action_key] = validation_metrics
```

Finally iterate `drafts` in original order, apply each stored validation result using the existing priority comparison, then call `_finalize_completed_record(draft)`. Do not update `best_online_reward_seen`, `best_online_cost_rank_seen`, candidate state, or PPO state before all validation requirements for the flush have been computed, matching the existing prefix semantics.

- [ ] **Step 4: Run focused order, cache, promotion, and PPO tests**

Run on the server:

```bash
python -m unittest \
  tests.test_blb_layerwise_runner \
  tests.test_blb_stage2_rl_regressions \
  tests.test_probe_runner_process_backend -v
python -m pytest tests/test_blb_probe_metric_aggregation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit, push, and screen**

Run locally:

```bash
git add blb_stage2_rl/sequential_runner.py tests/test_blb_layerwise_runner.py
git commit -m "perf(stage2): group independent promotion validations"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run the same profile-off 170-episode A/B with:

```bash
RUN_ID=stage2_grouped_validation_170
ARTIFACT_DIR=/hy-tmp/stage2_five_gpu_runtime_20260728/grouped_validation_170
```

Expected: exact full-state equality and lower five-GPU wall time. Reject the candidate if validation decisions, candidate records, PPO/checkpoint state, or wall performance differ adversely.

If rejected, run locally before Task 7:

```bash
candidate_commit="$(git log -1 --format=%H --grep='^perf(stage2): group independent promotion validations$')"
git revert --no-edit "$candidate_commit"
git push origin codex/stage2-five-gpu-runtime-20260728
```

## Task 7: Add The Exact Five-GPU Production Gate

**Files:**
- Create: `scripts/stage2_five_gpu_runtime_gate.sh`
- Create: `tests/test_stage2_five_gpu_runtime_gate.py`

- [ ] **Step 1: Write the failing wrapper test**

Create:

```python
from pathlib import Path
import os
import subprocess
import unittest


class Stage2FiveGPURuntimeGateTests(unittest.TestCase):
    def test_preflight_prints_exact_target_contract(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PRINT_EFFECTIVE_COMMANDS"] = "1"
        completed = subprocess.run(
            ["bash", "scripts/stage2_five_gpu_runtime_gate.sh"],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        self.assertIn("--model-type bert-large", output)
        self.assertIn("--stage2-search-episodes 170", output)
        self.assertIn("--stage2-k-trials 5", output)
        self.assertIn("--blb-v3-online-k-trials 5", output)
        self.assertIn("--ppo-update-interval 120", output)
        self.assertIn("--batch-size 64", output)
        self.assertIn("many=0,1,2,3,4", output)
```

- [ ] **Step 2: Push the red test and run it on the server**

Run locally:

```bash
git add tests/test_stage2_five_gpu_runtime_gate.py
git commit -m "test(stage2): specify five-GPU production gate"
git push origin codex/stage2-five-gpu-runtime-20260728
```

Run on the server:

```bash
python -m unittest tests.test_stage2_five_gpu_runtime_gate -v
```

Expected: failure because the wrapper does not exist.

- [ ] **Step 3: Create the fixed-contract wrapper**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLB_STAGE2_PROBE_INTRAOP_THREADS=1
export BLB_STAGE2_PROBE_INTEROP_THREADS=1
export BLB_STAGE2_PROBE_BACKEND=process
export BLB_STAGE2_PROBE_PHASE_PROFILE=0

RUN_ID="${RUN_ID:-stage2_five_gpu_runtime_$(date +%Y%m%d_%H%M%S)}"
ARTIFACT_DIR="${ARTIFACT_DIR:-experiments/server_command_runs/${RUN_ID}}"

RUN_ID="$RUN_ID" \
ARTIFACT_DIR="$ARTIFACT_DIR" \
MODEL_TYPE=bert-large \
EPISODES_AB=170 \
ONE_DEVS=0 \
MANY_DEVS=0,1,2,3,4 \
ONE_WORKERS_PER_DEVICE=1 \
MANY_WORKERS_PER_DEVICE=1 \
POLICY_DEVICE=worker \
DYNAMIC_ASSIGNMENT=1 \
BATCH_SIZE=64 \
PROBE_SIZE=256 \
KTRIALS=5 \
ONLINE_KTRIALS=5 \
PPO_UPDATE_INTERVAL=120 \
MIN_SPEEDUP=4.5 \
REQUIRE_FULL_STATE_EQUALITY=1 \
REQUIRE_IDLE_GPUS=1 \
GPU_SAMPLE_INTERVAL_SECONDS=2 \
TIMEOUT_SECONDS=21600 \
bash scripts/stage2_ngpu_speed_ab.sh
```

Do not change batch size, K, validation trials, reward settings, seed, or dataset.

- [ ] **Step 4: Run shell syntax and preflight tests**

Run on the server:

```bash
bash -n scripts/stage2_five_gpu_runtime_gate.sh
PRINT_EFFECTIVE_COMMANDS=1 bash scripts/stage2_five_gpu_runtime_gate.sh
python -m unittest tests.test_stage2_five_gpu_runtime_gate -v
```

Expected: syntax check and test pass without querying or occupying GPUs.

- [ ] **Step 5: Commit and push**

Run locally:

```bash
git add scripts/stage2_five_gpu_runtime_gate.sh tests/test_stage2_five_gpu_runtime_gate.py
git commit -m "perf(stage2): add strict five-GPU runtime gate"
git push origin codex/stage2-five-gpu-runtime-20260728
```

## Task 8: Select Only Proven Runtime Candidates

**Files:**
- Read: `/hy-tmp/stage2_five_gpu_runtime_20260728/*/stage2_ngpu_gate_verdict.txt`
- Read: `/hy-tmp/stage2_five_gpu_runtime_20260728/*/strict_scientific_equivalence.json`
- Create locally after copying compact evidence: `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/README.md`
- Create locally: `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/candidate_ledger.json`

- [ ] **Step 1: Build a candidate ledger from external wall time**

For each candidate, record:

```text
commit
one-GPU wall seconds
five-GPU wall seconds
five-GPU episodes/hour
speedup
parallel efficiency
strict scientific equality
install/forward/IPC phase totals from the profile-on run
accepted or rejected
```

Acceptance rule:

```text
strict scientific equality must be true
the candidate five-GPU median wall time must be lower than its matched parent
no single-GPU regression may exceed the saved five-GPU wall-time gain
```

- [ ] **Step 2: Remove rejected candidate commits from the final line**

Each rejected candidate was reverted immediately in Tasks 4-6. Verify that no rejected implementation remains in the source tree:

```bash
git status --short
git log --format='%H%x09%s' \
  bce4212641e7b24e786a2b27f6c9888c93cf7fa2..HEAD
git diff --check bce4212641e7b24e786a2b27f6c9888c93cf7fa2..HEAD
```

Write `candidate_ledger.json` with one row for each of:

```text
block1_reconfigure_170
probe_descriptors_170
grouped_validation_170
```

Each row records its exact commit, revert commit when rejected, parent and candidate wall times, strict equality boolean, measured speedup, and `accepted` boolean. A rejected row must have a nonempty `revert_commit`; an accepted row must have a strictly lower matched five-GPU wall time.

- [ ] **Step 3: Push the clean accepted line**

Run locally:

```bash
git status --short
git log --oneline --decorate -12
git push origin codex/stage2-five-gpu-runtime-20260728
```

Expected: clean status and a remote branch whose active source contains no rejected runtime implementation.

## Task 9: Refresh All Agent Heads Before Final Deployment

**Files:**
- Read: all remote branch heads
- Modify only if required: clean final aggregate branch

- [ ] **Step 1: Refresh every remote head and identify the newest completed aggregate**

Run locally:

```bash
git fetch --all --prune
git for-each-ref \
  --sort=-committerdate \
  --format='%(committerdate:iso8601) %(objectname) %(refname:short)' \
  refs/remotes/origin
```

Expected: a dated, complete list. Use project handoff markers and ancestry to distinguish completed aggregates from isolated agent branches.

- [ ] **Step 2: Integrate only completed, non-superseded commits**

Create a clean integration branch from the current completed aggregate and merge the reviewed runtime branch:

```bash
git switch -c codex/stage2-five-gpu-runtime-aggregate origin/jk_standard_rl
git merge --no-ff codex/stage2-five-gpu-runtime-20260728 \
  -m "merge: aggregate stage2 five-GPU runtime work"
git diff --check origin/jk_standard_rl..HEAD
git push -u origin codex/stage2-five-gpu-runtime-aggregate
```

If the merge conflicts, stop before resolving and re-audit the newest completed
aggregate against the runtime branch; do not guess between scientific changes.
Do not merge dirty worktrees, experiment-only branches, or incomplete agent
commits.

- [ ] **Step 3: Re-run focused tests after any runtime-relevant integration**

Push, pull on the server, then run:

```bash
python -m unittest \
  tests.test_stage2_five_gpu_runtime_gate \
  tests.test_stage2_ngpu_speed_targeted_first \
  tests.test_elastic_rl_scaling_ab \
  tests.test_stage2_ngpu_ab_compare \
  tests.test_probe_runner_process_backend \
  tests.test_blb_stage2_rl_regressions \
  tests.test_blb_layerwise_runner -v
python -m pytest tests/test_blb_probe_metric_aggregation.py -q
```

Expected: all pass. If integration changes `function_handler.py`, `probe_runner.py`, `inference_eval.py`, `sequential_runner.py`, launcher arguments, presets, or checkpoint writers, repeat the final A/B in Task 10.

## Task 10: Run The Final 170-Episode One-Vs-Five Gate

**Files:**
- Generate on server: `/hy-tmp/stage2_five_gpu_runtime_20260728/final_170/`
- Generate on server: `/hy-tmp/stage2_five_gpu_runtime_20260728/final_reference_compare.json`

- [ ] **Step 1: Deploy the final aggregate only through Git**

Run on the server:

```bash
cd /hy-tmp/rfr-stage2-five-gpu-runtime-20260728
git fetch origin codex/stage2-five-gpu-runtime-aggregate
git switch codex/stage2-five-gpu-runtime-aggregate
git pull --ff-only origin codex/stage2-five-gpu-runtime-aggregate
git status --short
git rev-parse HEAD
git rev-parse HEAD^{tree}
```

Expected: tracked-clean server source. The commit and tree must match local and remote.

- [ ] **Step 2: Run the focused suite and then the full project suite**

Run on the server:

```bash
source /hy-tmp/rfr-stage2-five-gpu-runtime-venv/bin/activate
python -m unittest \
  tests.test_stage2_five_gpu_runtime_gate \
  tests.test_stage2_ngpu_speed_targeted_first \
  tests.test_elastic_rl_scaling_ab \
  tests.test_stage2_ngpu_ab_compare \
  tests.test_probe_runner_process_backend \
  tests.test_blb_stage2_rl_regressions \
  tests.test_blb_layerwise_runner -v \
  2>&1 | tee /hy-tmp/stage2_five_gpu_runtime_20260728/focused_tests.log
python -m pytest tests/test_blb_probe_metric_aggregation.py -q \
  2>&1 | tee -a /hy-tmp/stage2_five_gpu_runtime_20260728/focused_tests.log
python -m unittest discover -s tests -v \
  2>&1 | tee /hy-tmp/stage2_five_gpu_runtime_20260728/full_tests.log
```

Expected: no new failure.

- [ ] **Step 3: Run the production gate with profiling disabled**

Run on the server with all five GPUs idle:

```bash
RUN_ID=stage2_five_gpu_runtime_final_170 \
ARTIFACT_DIR=/hy-tmp/stage2_five_gpu_runtime_20260728/final_170 \
bash scripts/stage2_five_gpu_runtime_gate.sh
```

Expected:

```text
one-GPU episodes = 170
five-GPU episodes = 170
speedup >= 4.500x
parallel efficiency >= 0.900
strict_scientific_equivalence.json has equal=true and diffs=[]
each GPU receives the deterministic K=5 share
```

- [ ] **Step 4: Compare optimized one-GPU scientific state to the starting reference**

Run on the server:

```bash
python scripts/elastic_rl_scaling_ab.py compare \
  --control /hy-tmp/stage2_five_gpu_runtime_20260728/baseline/ab_170/persistent_one \
  --candidate /hy-tmp/stage2_five_gpu_runtime_20260728/final_170/persistent_one \
  --stage stage2 \
  --data-points-root rl_training_data_points \
  --output /hy-tmp/stage2_five_gpu_runtime_20260728/final_reference_compare.json
```

Expected:

```json
{
  "equal": true,
  "diffs": []
}
```

This proves that the optimized source did not change scientific state, while the final one-vs-five comparison proves GPU-count invariance.

- [ ] **Step 5: Do not declare completion if the hard speed gate misses**

If speedup is below 4.5x, retain the run as profile evidence, return to Task 3 with the measured largest phase, and add one new isolated runtime candidate. Do not weaken `MIN_SPEEDUP`, increase episode count, omit startup time, reduce K, reduce validation, or compare against a different configuration.

## Task 11: Archive Evidence And Verify Three-Way Parity

**Files:**
- Create: `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/README.md`
- Create: `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/candidate_ledger.json`
- Create: `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/final_summary.json`
- Create: `experiments/server_command_runs/stage2_five_gpu_runtime_20260728/SHA256SUMS`

- [ ] **Step 1: Copy only compact evidence from the server**

Copy through the approved artifact channel:

```text
source commit and tree
effective commands
one/five wall seconds
gate verdict
strict scientific equivalence JSON
starting-reference equivalence JSON
GPU utilization summaries
focused/full test summaries
accepted/rejected candidate ledger
hardware manifest
```

Do not commit checkpoints, model caches, raw training directories, or raw `nvidia-smi` streams.

- [ ] **Step 2: Write the machine-readable final summary from measured files**

After compact final evidence has been copied into
`experiments/server_command_runs/stage2_five_gpu_runtime_20260728/final_170/`,
run locally:

```bash
python - <<'PY'
import json
import re
from pathlib import Path

root = Path(
    "experiments/server_command_runs/"
    "stage2_five_gpu_runtime_20260728"
)
verdict_text = (
    root / "final_170" / "stage2_ngpu_gate_verdict.txt"
).read_text(encoding="utf-8")
strict = json.loads(
    (root / "final_170" / "strict_scientific_equivalence.json").read_text(
        encoding="utf-8"
    )
)
reference = json.loads(
    (root / "final_reference_compare.json").read_text(encoding="utf-8")
)
ledger = json.loads(
    (root / "candidate_ledger.json").read_text(encoding="utf-8")
)
runtime_commit = (
    root / "final_170" / "HEAD.txt"
).read_text(encoding="utf-8").strip()
runtime_tree = (
    root / "final_170" / "TREE.txt"
).read_text(encoding="utf-8").strip()

def number(label):
    match = re.search(
        rf"^{re.escape(label)}:\s*([-+0-9.eE]+)x?\s*$",
        verdict_text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"missing verdict field: {label}")
    return float(match.group(1))

one_wall = number("1GPU wall_s")
five_wall = number("NGPU wall_s")
speedup = number("speedup")
payload = {
    "schema": "stage2_five_gpu_runtime_final_v1",
    "model": "bert-large",
    "dataset": "mrpc",
    "episodes": 170,
    "one_gpu_wall_seconds": one_wall,
    "five_gpu_wall_seconds": five_wall,
    "one_gpu_episodes_per_hour": number("1GPU episodes/hour"),
    "five_gpu_episodes_per_hour": number("NGPU episodes/hour"),
    "speedup": speedup,
    "parallel_efficiency": speedup / 5.0,
    "strict_one_vs_five_equal": bool(strict["equal"]),
    "strict_start_vs_optimized_equal": bool(reference["equal"]),
    "runtime_source_commit": runtime_commit,
    "runtime_source_tree": runtime_tree,
    "runtime_server_tracked_clean": True,
    "accepted_candidates": [
        row["name"] for row in ledger["candidates"] if row["accepted"]
    ],
    "rejected_candidates": [
        row["name"] for row in ledger["candidates"] if not row["accepted"]
    ],
    "evidence_paths": [
        "final_170/stage2_ngpu_gate_verdict.txt",
        "final_170/strict_scientific_equivalence.json",
        "final_reference_compare.json",
        "candidate_ledger.json",
        "focused_tests.log",
        "full_tests.log",
    ],
}
if payload["speedup"] < 4.5:
    raise RuntimeError("final speedup is below 4.5x")
if payload["parallel_efficiency"] < 0.90:
    raise RuntimeError("final parallel efficiency is below 90%")
if not payload["strict_one_vs_five_equal"]:
    raise RuntimeError("one-vs-five scientific state differs")
if not payload["strict_start_vs_optimized_equal"]:
    raise RuntimeError("starting-vs-optimized scientific state differs")
(root / "final_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
```

Expected: the command refuses to generate a passing summary if speed or either exact-equivalence gate fails.
The summary deliberately records the immutable runtime commit/tree rather than
the later evidence commit, avoiding an impossible self-referential commit hash.
Final local/Git/server evidence-commit parity is verified after Step 4.

- [ ] **Step 3: Generate and verify SHA-256 digests**

Run locally:

```bash
cd experiments/server_command_runs/stage2_five_gpu_runtime_20260728
find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 > SHA256SUMS
shasum -a 256 -c SHA256SUMS
```

Expected: every compact evidence file reports `OK`.

- [ ] **Step 4: Commit and push the evidence**

Run locally:

```bash
git add experiments/server_command_runs/stage2_five_gpu_runtime_20260728
git commit -m "docs(stage2): archive five-GPU runtime evidence"
git push origin codex/stage2-five-gpu-runtime-aggregate
```

- [ ] **Step 5: Pull the evidence commit on the server and verify final parity**

Run on the server:

```bash
git pull --ff-only origin codex/stage2-five-gpu-runtime-aggregate
git status --short
git rev-parse HEAD
git rev-parse HEAD^{tree}
```

Run locally:

```bash
git status --short
git rev-parse HEAD
git rev-parse HEAD^{tree}
git ls-remote origin refs/heads/codex/stage2-five-gpu-runtime-aggregate
```

Expected:

```text
local status is empty
server status is empty
local, remote, and server full commit IDs are identical
local and server tree IDs are identical
```

- [ ] **Step 6: Report only the proven outcome**

The final report must state:

```text
measured one-GPU and five-GPU wall times
measured episodes/hour
measured speedup and parallel efficiency
remaining serial fraction and worker straggler evidence
strict one-vs-five and start-vs-optimized equality results
focused/full server test counts
final commit/tree and three-way parity
compact evidence path
```

Do not claim RL-quality improvement from this 170-episode performance gate.

## Task 12: Preserve Five Healthy GPUs And Balance Degraded Pools

**Files:**
- Modify: `function_handler.py`
- Modify: `blb_stage2_rl/inference_eval.py`
- Modify: `blb_stage2_rl/probe_runner.py`
- Modify: `tests/test_elastic_gpu_supervisor.py`
- Modify: `tests/test_blb_probe_metric_aggregation.py`
- Modify: `tests/test_probe_runner_process_backend.py`
- Modify only if telemetry requires it:
  `blb_stage2_rl/env.py`, `blb_stage2_rl/sequential_runner.py`,
  `scripts/elastic_rl_scaling_ab.py`, and their focused tests.

- [ ] **Step 1: Lock the all-healthy invariant with red tests**

Require an all-healthy five-device health snapshot to preserve candidate order,
produce logical devices `0,1,2,3,4`, and create five probe workers. Require
K=5 with five workers to choose whole-trial scheduling and never batch
sharding.

- [ ] **Step 2: Specify exact per-batch metric contributions**

Write tests proving that finalizing ordered per-batch contributions returns the
same loss, metric1, and weighted-F1 metric2 as the existing complete-trial
path. Duplicate, missing, or out-of-order identities must fail closed.

- [ ] **Step 3: Specify deterministic CUDA generator offset replay**

Write server-only CUDA tests proving that a later probe batch executed after
reseed plus exact noise/truncation generator offsets is bitwise equal to the
same batch in a sequential complete trial, including the next generator state.

- [ ] **Step 4: Implement guarded uneven-pool scheduling**

Keep the existing complete-trial process path unchanged for five healthy
workers and K=5. For uneven smaller pools, calibrate generator offset deltas
with one inference-only sample, distribute `(trial, batch)` identities
round-robin, verify offsets on every result, and finalize in canonical order.
Discard all partial outputs and replay complete trials if any guard fails.

- [ ] **Step 5: Verify failure handling and zero hot-path health polling**

Prove a failed replica retries only missing batch identities on the remaining
pool, while all-healthy execution performs no per-episode `nvidia-smi` query.
The existing 60-second recovery monitor remains dormant when there are no
quarantined devices.

- [ ] **Step 6: Run matched server A/B**

Run profile-off one-, three-, four-, and five-GPU probes from the same source,
seed, action, and batch set. Acceptance requires:

- five healthy GPUs use all five workers and retain at least the already
  measured `4.9299x` terminal-probe speedup;
- four GPUs improve materially over the current `2.5257x` discrete-task limit;
- three GPUs improve materially over whole-trial scheduling;
- every arm passes recursive scientific-state equality with `diffs=[]`.

- [ ] **Step 7: Commit, push, and include in the final 170-episode gate**

Commit each red/green stage and push it before server deployment. Reject the
degraded scheduler if exact replay cannot be proved; do not weaken or replace
the all-healthy five-GPU path.
