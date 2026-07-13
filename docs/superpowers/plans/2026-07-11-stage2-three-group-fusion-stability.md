# Stage-2 Three-Group Fusion Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact Block2/4/5 fusion-one fixed action, evaluate control and both treatments in five independent five-trial MRPC experiments, and produce a gated three-group stability report.

**Architecture:** Extend the torch-free fusion-map report generator so it emits the third fixed action without changing existing groups. Add opt-in paired-seed and raw-trial diagnostics to the existing unified RL-path evaluation script while preserving its default behavior. Add a dedicated five-run report/aggregation script; do not reuse or alter the older two-profile pair report.

**Tech Stack:** Python 3, `unittest`, JSON/JSONL-style structured artifacts, existing `BLBStage2SequentialEnv`/`BLBStage2Env` evaluation chain, Git worktrees, one-GPU CUDA server execution.

---

## File Map

- Modify `scripts/report_fusion_count_map.py`: generate the exact B2=1/B4=1/B5=1 action and fail if a required map lacks fusion count 1.
- Modify `tests/test_report_fusion_count_map.py`: lock the new action schedule and no-clamping behavior.
- Modify `scripts/run_fusion_count_action_eval_rlpath.py`: add opt-in shared group seeds, capture raw trial metrics without changing the core env, and retain per-step replan evidence.
- Modify `tests/test_run_fusion_count_action_eval_rlpath.py`: verify seed behavior, trial capture, and replan evidence.
- Create `scripts/render_three_group_fusion_stability_report.py`: validate five run payloads, compute pooled and paired statistics, and render JSON/HTML.
- Create `tests/test_render_three_group_fusion_stability_report.py`: verify 25-trial pooling, all three comparisons, gates, and readable output.
- Create after server execution: `experiments/server_command_runs/stage2_three_group_fusion_stability_<timestamp>/`: exact command, per-run raw results/logs, summary JSON/HTML, and verification evidence.
- Copy final HTML to `reports/html_reports/<timestamp>_stage2_three_group_fusion_stability.html`.

### Task 1: Generate The Exact Three-Block Fusion-One Action

**Files:**
- Modify: `tests/test_report_fusion_count_map.py`
- Modify: `scripts/report_fusion_count_map.py:560-630`

- [ ] **Step 1: Write the failing exact-action tests**

Add tests that build small Block1/2/4/5 graphs and assert the new group uses fusion count 1 only for Blocks 2, 4, and 5:

```python
def _two_option_graph(graph_key, block_idx):
    return {
        "graph_key": graph_key,
        "block_idx": block_idx,
        "options": [
            {"option_id": 0, "fusion_count": 0},
            {"option_id": 1, "fusion_count": 1},
        ],
    }

def test_group_specs_adds_exact_b2_b4_b5_fusion_one(self):
    graphs = {
        "block1_mrpc": _two_option_graph("block1_mrpc", 1),
        "block2_mrpc": _two_option_graph("block2_mrpc", 2),
        "block4": _two_option_graph("block4", 4),
        "block5_n4": _two_option_graph("block5_n4", 5),
    }
    schedule = [
        {"step_idx": i, "layer_idx": 0, "block_idx": graph["block_idx"], "graph_key": key}
        for i, (key, graph) in enumerate(graphs.items())
    ]
    specs = {spec["name"]: spec for spec in report._group_specs(graphs, schedule)}
    actual = specs["block2_block4_block5_all_layers_fusion1"]
    self.assertEqual(actual["fusion_count_by_graph"], {
        "block1_mrpc": 0,
        "block2_mrpc": 1,
        "block4": 1,
        "block5_n4": 1,
    })

def test_group_specs_rejects_missing_required_fusion_one(self):
    graphs = {
        "block2_mrpc": _two_option_graph("block2_mrpc", 2),
        "block4": {
            "graph_key": "block4",
            "block_idx": 4,
            "options": [{"option_id": 0, "fusion_count": 0}],
        },
    }
    with self.assertRaisesRegex(ValueError, "block4.*fusion count 1"):
        report._group_specs(graphs, [])
```

- [ ] **Step 2: Commit and push the red tests**

```bash
git add tests/test_report_fusion_count_map.py
git commit -m "test: require three-block fusion action"
git push origin codex/fixed-b2-b5-fusion
```

- [ ] **Step 3: Run the exact red snapshot on the server**

Run from an isolated server worktree checked out at the pushed test commit:

```bash
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m unittest \
  tests.test_report_fusion_count_map.FusionCountMapReportTest.test_group_specs_adds_exact_b2_b4_b5_fusion_one \
  tests.test_report_fusion_count_map.FusionCountMapReportTest.test_group_specs_rejects_missing_required_fusion_one -v
```

Expected: both tests fail because the group does not exist and missing fusion count 1 is not rejected.

- [ ] **Step 4: Implement minimal exact group generation**

After the existing Block2/Block5 combined group, add an exact fusion-one group. Reuse the existing `choose()` cache and reject clamping:

```python
    exact_one_options = {}
    exact_one_counts = {}
    exact_one_targets = []
    for graph_key in graph_order:
        required = (
            graph_key == "block2_mrpc"
            or graph_key == "block4"
            or graph_key.startswith("block5_")
        )
        target = 1 if required else 0
        opt, count, clamped = choose(graph_key, target)
        if required and (clamped or count != 1):
            raise ValueError(
                f"{graph_key} does not provide required fusion count 1"
            )
        exact_one_options[graph_key] = opt
        exact_one_counts[graph_key] = count
        if required:
            exact_one_targets.append(graph_key)
    specs.append({
        "name": "block2_block4_block5_all_layers_fusion1",
        "family": "combined",
        "target_graphs": exact_one_targets,
        "target_fusion_count": 1,
        "option_by_graph": exact_one_options,
        "fusion_count_by_graph": exact_one_counts,
        "occurrence_counts": dict(occurrence_counts),
    })
```

- [ ] **Step 5: Commit and push the implementation**

```bash
git add scripts/report_fusion_count_map.py
git commit -m "Add exact three-block fusion action"
git push origin codex/fixed-b2-b5-fusion
```

- [ ] **Step 6: Verify Task 1 on the server**

```bash
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m unittest \
  tests.test_report_fusion_count_map -v
```

Expected: all `test_report_fusion_count_map` tests pass.

### Task 2: Add Paired Seeds And Auditable Trial/Replan Evidence

**Files:**
- Modify: `tests/test_run_fusion_count_action_eval_rlpath.py`
- Modify: `scripts/run_fusion_count_action_eval_rlpath.py:190-450,520-610`

- [ ] **Step 1: Write failing tests for opt-in paired seeds and evidence**

Add pure helper tests and extend the existing fake sequential environment test:

```python
def test_group_seed_can_be_shared_without_changing_default(self):
    import scripts.run_fusion_count_action_eval_rlpath as rlpath

    self.assertEqual(rlpath._group_seed(100, 2, shared=False), 102)
    self.assertEqual(rlpath._group_seed(100, 2, shared=True), 100)

def test_trial_metric_payload_preserves_each_trial(self):
    import scripts.run_fusion_count_action_eval_rlpath as rlpath

    self.assertEqual(
        rlpath._trial_metric_payload([0.2, 0.3], [0.8, 0.9], [0.7, 0.8]),
        {"loss": [0.2, 0.3], "metric1": [0.8, 0.9], "metric2": [0.7, 0.8]},
    )
```

In `FakeSeqEnv.commit_step`, set the captured values immediately before
returning the terminal transition (the production path clears stale values at
the beginning of each group):

```python
self.base.fixed_eval_trial_metrics = {
    "loss": [0.29, 0.31],
    "metric1": [0.87, 0.89],
    "metric2": [0.86, 0.88],
}
```

and return this from `evaluate_step`:

```python
"replan_application": {
    "applied_before_forward": True,
    "model_uses_replan_config": True,
},
```

Then assert `_run_group()` preserves both fields:

```python
self.assertTrue(
    result["step_records"][0]["replan_application"]["model_uses_replan_config"]
)
self.assertEqual(result["trial_metrics"]["loss"], [0.29, 0.31])
```

- [ ] **Step 2: Commit/push and run the red tests on the server**

```bash
git add tests/test_run_fusion_count_action_eval_rlpath.py
git commit -m "test: require paired fixed-eval evidence"
git push origin codex/fixed-b2-b5-fusion
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m unittest \
  tests.test_run_fusion_count_action_eval_rlpath -v
```

Expected: new tests fail because the helpers and output fields do not exist.

- [ ] **Step 3: Implement dependency-light helpers**

Add helpers near `_metric_dict`:

```python
def _group_seed(base_seed: int, group_index: int, *, shared: bool) -> int:
    return int(base_seed) if shared else int(base_seed) + int(group_index)


def _trial_metric_payload(losses, metric1s, metric2s) -> Dict[str, List[float]]:
    return {
        "loss": [float(value) for value in losses],
        "metric1": [float(value) for value in metric1s],
        "metric2": [float(value) for value in metric2s],
    }
```

- [ ] **Step 4: Capture raw trials only inside the fixed-eval script**

Inside `_build_seq_env`, subclass the runtime-loaded env without editing `blb_stage2_rl/env.py`:

```python
    class RecordingBLBStage2Env(BLBStage2Env):
        def _aggregate_probe_trials(self, losses, metric1s, metric2s):
            self.fixed_eval_trial_metrics = _trial_metric_payload(
                losses, metric1s, metric2s,
            )
            return super()._aggregate_probe_trials(losses, metric1s, metric2s)

```

Then change only the existing constructor class name, retaining every argument:

```python
-    base_env = BLBStage2Env(
+    base_env = RecordingBLBStage2Env(
```

This records diagnostics only; it delegates metric calculation unchanged to
the production env.

- [ ] **Step 5: Preserve per-step replan and terminal trial evidence**

Before starting `_run_group`, clear stale trial data. Add the replan payload to each step and the captured trial arrays to the group result:

```python
    seq_env.base.fixed_eval_trial_metrics = {}
    seq_env.reset(seed=int(seed))
    seq_env.base.probe_noise_seed = int(seed)
```

Add this field to the existing step-record dictionary:

```python
"replan_application": to_jsonable(
    eval_info.get("replan_application") or {},
    stringify_unknown=True,
    preserve_native=True,
),
```

Add this field to the existing returned group-result literal:

```python
"trial_metrics": to_jsonable(
    getattr(seq_env.base, "fixed_eval_trial_metrics", {}) or {},
    stringify_unknown=True,
    preserve_native=True,
),
```

- [ ] **Step 6: Add the opt-in CLI and audit metadata**

Add a default-off flag:

```python
parser.add_argument(
    "--shared-group-seed",
    action="store_true",
    help="use the same deterministic probe seed for every compared group",
)
```

Use it in the group loop and record it in the top-level result:

```python
for idx, cfg in enumerate(unique):
    group_seed = _group_seed(
        args.seed, idx, shared=bool(args.shared_group_seed),
    )
    result_by_key[rlpath_config_group_key(cfg)] = _run_group(
        seq_env, cfg, seed=group_seed,
    )

combined.update({
    "seed": int(args.seed),
    "shared_group_seed": bool(args.shared_group_seed),
})
```

Default callers retain the current `seed + group_index` behavior.

- [ ] **Step 7: Commit/push and verify Task 2 on the server**

```bash
git add scripts/run_fusion_count_action_eval_rlpath.py
git commit -m "Record paired fusion evaluation evidence"
git push origin codex/fixed-b2-b5-fusion
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m unittest \
  tests.test_run_fusion_count_action_eval_rlpath -v
```

Expected: all RL-path evaluator tests pass.

### Task 3: Build The Five-Run Three-Group Report

**Files:**
- Create: `tests/test_render_three_group_fusion_stability_report.py`
- Create: `scripts/render_three_group_fusion_stability_report.py`

- [ ] **Step 1: Write the synthetic payload fixture**

Generate the real 47-step schedule in the test: layer 0 has Blocks 2/4/5, and layers 1-11 have Blocks 1/2/4/5. Each group receives five raw trial values and matching replan evidence.

```python
GROUPS = {
    "all_fusion0": {2: 0, 4: 0, 5: 0},
    "block2_block5_all_layers_fusionmax": {2: 1, 4: 0, 5: 1},
    "block2_block4_block5_all_layers_fusion1": {2: 1, 4: 1, 5: 1},
}

def _steps(pattern):
    rows = []
    schedule = [(0, block) for block in (2, 4, 5)]
    schedule.extend(
        (layer, block)
        for layer in range(1, 12)
        for block in (1, 2, 4, 5)
    )
    for index, (layer, block) in enumerate(schedule):
        fusion = pattern.get(block, 0)
        rows.append({
            "step_idx": index,
            "layer_idx": layer,
            "block_idx": block,
            "graph_key": "block5_n4" if block == 5 else f"block{block}",
            "k_value": 13,
            "valid": True,
            "fusion_count_replan": fusion,
            "boosted": bool(fusion),
            "replan_application": {
                "applied_before_forward": True,
                "model_uses_replan_config": True,
            },
        })
    return rows
```

- [ ] **Step 2: Write failing aggregation and gate tests**

Cover these behaviors:

```python
def test_build_summary_pools_25_trials_and_three_pairings(self):
    summary = build_summary(
        run_payloads=[_payload(seed) for seed in EXPECTED_SEEDS],
        source_commit="abc123",
    )
    self.assertEqual(summary["experiment"]["total_evaluations"], 75)
    self.assertEqual(
        set(summary["paired"]),
        {"b2b5_minus_control", "b2b4b5_minus_control", "b2b4b5_minus_b2b5"},
    )
    self.assertEqual(
        summary["aggregate"]["all_fusion0"]["loss"]["trial_count"], 25,
    )
    self.assertTrue(summary["all_gates_pass"])

def test_build_summary_fails_gate_on_wrong_block4_pattern(self):
    payloads = [_payload(seed) for seed in EXPECTED_SEEDS]
    target = payloads[0]["group_results"][2]
    target["step_records"][1]["fusion_count_replan"] = 0
    summary = build_summary(run_payloads=payloads, source_commit="abc123")
    self.assertFalse(summary["all_gates_pass"])

def test_cli_renders_group_definitions_k_and_standard_deviations(self):
    # write five payloads, call main(), and inspect JSON/HTML
    self.assertIn("B2=1, B4=1, B5=1", html)
    self.assertIn("K=13", html)
    self.assertIn("mean +/- std", html)
```

- [ ] **Step 3: Commit/push and run the red report tests on the server**

```bash
git add tests/test_render_three_group_fusion_stability_report.py
git commit -m "test: specify three-group stability report"
git push origin codex/fixed-b2-b5-fusion
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m unittest \
  tests.test_render_three_group_fusion_stability_report -v
```

Expected: import failure because the report module does not exist.

- [ ] **Step 4: Implement strict run/group validation**

In the new report script, define the required seeds, groups, totals, and comparisons. Validate exactly five runs, five trials per group, 408 examples, GELU4/Softmax6, K=13, three unique actions, 47 valid steps, canonical install path, replan application, expected fusion patterns/totals, and `block5_n4` usage.

```python
EXPECTED_SEEDS = [20260721, 20261721, 20262721, 20263721, 20264721]
GROUP_SPECS = {
    "all_fusion0": {"pattern": {2: 0, 4: 0, 5: 0}, "total": 0},
    "block2_block5_all_layers_fusionmax": {
        "pattern": {2: 1, 4: 0, 5: 1}, "total": 24,
    },
    "block2_block4_block5_all_layers_fusion1": {
        "pattern": {2: 1, 4: 1, 5: 1}, "total": 36,
    },
}
PAIR_SPECS = {
    "b2b5_minus_control": (
        "block2_block5_all_layers_fusionmax", "all_fusion0",
    ),
    "b2b4b5_minus_control": (
        "block2_block4_block5_all_layers_fusion1", "all_fusion0",
    ),
    "b2b4b5_minus_b2b5": (
        "block2_block4_block5_all_layers_fusion1",
        "block2_block5_all_layers_fusionmax",
    ),
}
```

Return gate details instead of silently excluding a bad run. `main()` writes artifacts and returns 1 when `all_gates_pass` is false.

- [ ] **Step 5: Implement exact pooled statistics from raw trials**

Flatten the five five-value lists for each metric and use population statistics, matching the production per-run `ddof=0` convention:

```python
from statistics import fmean, pstdev

pooled = [
    float(value)
    for run in runs
    for value in run["trial_metrics"][metric_key]
]
aggregate[metric_name] = {
    "trial_count": len(pooled),
    "pooled_25_mean": fmean(pooled),
    "pooled_25_std": pstdev(pooled),
    "run_mean_std": pstdev(run_means),
}
```

For each pair, compute five run-mean deltas, their mean/std, and the number of runs in which the first group is better. Loss is better when the delta is negative; accuracy/F1 are better when positive.

- [ ] **Step 6: Implement readable HTML and CLI**

The HTML must include:

- source commit and exact five seeds;
- GELU4, Softmax6, K=13, validation size 408;
- all three fusion definitions and totals;
- per-run `mean +/- std` rows for all groups;
- pooled 25-trial means/stds;
- three pairwise delta tables;
- protocol/replan/trial gates;
- a compact layer/block action table.

CLI:

```python
parser.add_argument("--run-json", action="append", required=True)
parser.add_argument("--source-commit", required=True)
parser.add_argument("--output-json", required=True)
parser.add_argument("--output-html", required=True)
```

- [ ] **Step 7: Commit/push and verify Task 3 on the server**

```bash
git add scripts/render_three_group_fusion_stability_report.py \
  tests/test_render_three_group_fusion_stability_report.py
git commit -m "Add three-group fusion stability report"
git push origin codex/fixed-b2-b5-fusion
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m unittest \
  tests.test_render_three_group_fusion_stability_report -v
```

Expected: all report tests pass.

### Task 4: Verify The Complete Source Snapshot On The School Server

**Files:**
- Verify: all files changed in Tasks 1-3
- Create as artifact: `verification.txt`

- [ ] **Step 1: Confirm the school server is safe to use**

Connect to `root@100.64.229.185:8722` and record, without terminating anything:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi pmon -c 1
ps -eo user,pid,etimes,cmd --sort=-etimes
```

If another user's GPU job is active, postpone the full evaluation. CPU-only focused tests may run only if they do not interfere with that job.

- [ ] **Step 2: Check out the exact pushed implementation commit**

Create a new isolated worktree under `/var/tmp/root-home/rfr_runs/` from
`origin/codex/fixed-b2-b5-fusion`:

```bash
BASE=/var/tmp/root-home/Reinforcement-For-Robustness
git -C "$BASE" fetch origin codex/fixed-b2-b5-fusion
SOURCE_COMMIT=$(git -C "$BASE" rev-parse origin/codex/fixed-b2-b5-fusion)
RUNROOT="/var/tmp/root-home/rfr_runs/stage2_three_group_${SOURCE_COMMIT:0:7}_$(date +%Y%m%d_%H%M%S)"
git -C "$BASE" worktree add --detach "$RUNROOT" "$SOURCE_COMMIT"
cd "$RUNROOT"
```

Record:

```bash
git rev-parse HEAD
git status --short
```

Expected: HEAD equals the local pushed commit and status is clean.

- [ ] **Step 3: Run the focused combined suite**

```bash
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m unittest \
  tests.test_report_fusion_count_map \
  tests.test_run_fusion_count_action_eval_rlpath \
  tests.test_render_three_group_fusion_stability_report -v
```

Expected: all tests pass with exit code 0. Save stdout/stderr and exit code in `verification.txt`.

- [ ] **Step 4: Generate and audit current GELU4 action configs**

Create the output root inside the detached source worktree before generating
artifacts:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT="$PWD/experiments/server_command_runs/stage2_three_group_fusion_stability_${TS}"
mkdir -p "$OUT/selected_actions"
printf '%s\n' "$(git rev-parse HEAD)" > "$OUT/SOURCE_SYNC_COMMIT"
```

```bash
PY=/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python
$PY scripts/report_fusion_count_map.py \
  --map-dir blb_stage2_rl/fusion_maps/mrpc \
  --output-dir "$OUT/map_report" \
  --html "$OUT/map_report.html" \
  --json "$OUT/map_report.json" \
  --profile mrpc \
  --gelu '[4,4,4,4,4,4,4,4,4,4,4,4]' \
  --softmax '[6,6,6,6,6,6,6,6,6,6,6,6]'
```

Copy only these generated JSON files into `$OUT/selected_actions/`:

```bash
cp "$OUT/map_report/action_configs/all_fusion0.json" "$OUT/selected_actions/"
cp "$OUT/map_report/action_configs/block2_block5_all_layers_fusionmax.json" "$OUT/selected_actions/"
cp "$OUT/map_report/action_configs/block2_block4_block5_all_layers_fusion1.json" "$OUT/selected_actions/"
test "$(find "$OUT/selected_actions" -maxdepth 1 -name '*.json' | wc -l)" -eq 3
```

Audit that exactly three files exist and their generated group metadata matches totals 0, 24, and 36. This copies generated artifacts; it does not hand-author server action configs.

```bash
"$PY" - "$OUT/selected_actions" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "all_fusion0": ({"block2_mrpc": 0, "block4": 0, "block5_n4": 0}, 0),
    "block2_block5_all_layers_fusionmax": (
        {"block2_mrpc": 1, "block4": 0, "block5_n4": 1}, 24,
    ),
    "block2_block4_block5_all_layers_fusion1": (
        {"block2_mrpc": 1, "block4": 1, "block5_n4": 1}, 36,
    ),
}
for name, (pattern, total) in expected.items():
    payload = json.loads((root / f"{name}.json").read_text())
    group = payload["group"]
    counts = group["fusion_count_by_graph"]
    occurrences = group["occurrence_counts"]
    for key, value in pattern.items():
        assert int(counts[key]) == value, (name, key, counts[key])
    actual_total = sum(
        int(counts[key]) * int(occurrences.get(key, 0))
        for key in counts
    )
    assert actual_total == total, (name, actual_total, total)
print("selected_action_gate=PASS")
PY
```

### Task 5: Run Five Independent Three-Group Evaluations

**Files:**
- Create on server then pull locally: `experiments/server_command_runs/stage2_three_group_fusion_stability_<timestamp>/run_*/results.json`
- Create: per-run `results.html`, `run.log`, and status metadata

- [ ] **Step 1: Write immutable run metadata outside source files**

Record `SOURCE_SYNC_COMMIT`, the full shell command, environment variables, GPU inventory, Python path, action-config hashes, and seed schedule under `$OUT`. Create a dummy original comparison artifact containing only:

```bash
printf '%s\n' '{"group_results": []}' > "$OUT/dummy_original.json"
printf '%s\n' "$PY" > "$OUT/PYTHON_PATH"
printf '%s\n' '20260721 20261721 20262721 20263721 20264721' > "$OUT/SEEDS"
sha256sum "$OUT"/selected_actions/*.json > "$OUT/action_sha256.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv > "$OUT/gpu_before.csv"
```

This file satisfies the evaluator's legacy comparison input; it does not define or alter actions.

- [ ] **Step 2: Run the five seed loop sequentially on GPU 0**

Use the exact established seed schedule and define each run directory
explicitly:

```bash
set -euo pipefail
SEEDS=(20260721 20261721 20262721 20263721 20264721)
for INDEX in 1 2 3 4 5; do
  SEED=${SEEDS[$((INDEX - 1))]}
  RUN=$(printf '%s/run_%02d_seed_%s' "$OUT" "$INDEX" "$SEED")
  mkdir -p "$RUN"
  CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/var/tmp/root-home/hf_cache \
  TRANSFORMERS_CACHE=/var/tmp/root-home/hf_cache/hub \
  HF_DATASETS_CACHE=/var/tmp/root-home/hf_cache/datasets \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" scripts/run_fusion_count_action_eval_rlpath.py \
    --dataset mrpc \
    --model-type bert-base \
    --action-dir "$OUT/selected_actions" \
    --original-json "$OUT/dummy_original.json" \
    --output-json "$RUN/results.json" \
    --output-html "$RUN/results.html" \
    --run-output-dir "$RUN/runtime" \
    --stage1-gelu '[4,4,4,4,4,4,4,4,4,4,4,4]' \
    --stage1-softmax '[6,6,6,6,6,6,6,6,6,6,6,6]' \
    --repeat 5 \
    --probe-size 408 \
    --batch-size 64 \
    --seed "$SEED" \
    --shared-group-seed \
    > "$RUN/run.log" 2>&1
done
```

Expected per run: `groups=3`, `unique_group_actions=3`, three `[run]` markers, exit code 0, and no traceback/OOM/invalid action.

- [ ] **Step 3: Validate each result before advancing**

Run the report script only after all five raw result files exist. If a run is missing five raw trials, has a non-finite metric, does not show 47 valid K=13 steps, or has the wrong fusion pattern, stop and preserve logs; do not substitute a historical run.

- [ ] **Step 4: Aggregate and render final artifacts**

```bash
$PY scripts/render_three_group_fusion_stability_report.py \
  --run-json "$OUT/run_01_seed_20260721/results.json" \
  --run-json "$OUT/run_02_seed_20261721/results.json" \
  --run-json "$OUT/run_03_seed_20262721/results.json" \
  --run-json "$OUT/run_04_seed_20263721/results.json" \
  --run-json "$OUT/run_05_seed_20264721/results.json" \
  --source-commit "$(cat "$OUT/SOURCE_SYNC_COMMIT")" \
  --output-json "$OUT/three_group_summary.json" \
  --output-html "$OUT/three_group_report.html"
```

Expected: exit code 0 and `all_gates_pass=true`.

### Task 6: Return, Verify, And Publish Compact Artifacts

**Files:**
- Create: `experiments/server_command_runs/stage2_three_group_fusion_stability_<timestamp>/...`
- Create: `reports/html_reports/<timestamp>_stage2_three_group_fusion_stability.html`

- [ ] **Step 1: Copy compact artifacts from server to the clean local worktree**

Include raw JSON, HTML, logs, action JSONs, map report, command/status/verification files, and source commit. Exclude model checkpoints, Hugging Face caches, runtime caches, `.pt`, and temporary dataset files.

- [ ] **Step 2: Verify copied artifact identity**

Compare file counts and SHA-256 hashes against the server manifest. Confirm `SOURCE_SYNC_COMMIT` equals the implementation commit and the summary reports 5 runs, 3 groups, 25 trials/group, 75 total evaluations, and passing gates.

- [ ] **Step 3: Commit and push the artifacts from local**

```bash
git add experiments/server_command_runs/stage2_three_group_fusion_stability_* \
  reports/html_reports/*_stage2_three_group_fusion_stability.html
git commit -m "Record three-group fusion stability results"
git push origin codex/fixed-b2-b5-fusion
```

- [ ] **Step 4: Final Git and result audit**

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/codex/fixed-b2-b5-fusion
```

Expected: clean worktree and local HEAD equals remote branch. Report the local HTML/JSON paths and a chat table containing all five run results, pooled means/stds, and all three paired comparisons.
