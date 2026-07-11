# Stage-2 Per-Example Correctness And Logits Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the exact MRPC input identities, labels, predictions, correctness, and final logits for all 30,600 rows in the existing three-group fixed-fusion experiment, then deliver one self-contained searchable HTML report on the user's Desktop.

**Architecture:** Add an opt-in, read-only model-forward recorder to the existing canonical fixed-action evaluator. Keep capture and MRPC identity resolution in a focused helper module, leave Stage-2 RL core files unchanged, and use a separate torch-free aggregation/report module that reuses the existing three-group metric gates before adding per-example gates and HTML data browsing.

**Tech Stack:** Python 3.10, PyTorch forward hooks, Hugging Face MRPC validation data, JSONL, existing `BLBStage2SequentialEnv`/`BLBStage2Env` evaluation path, `unittest`, static HTML/CSS/JavaScript, one RTX 4090 GPU server.

---

## File Map

- Create `scripts/fusion_count_prediction_capture.py`: dependency-light MRPC identity catalog, read-only forward recorder, row builder, and streaming JSONL writer.
- Create `tests/test_fusion_count_prediction_capture.py`: identity collision, batching, logits, prediction, correctness, and row-count tests.
- Modify `scripts/run_fusion_count_action_eval_rlpath.py`: opt-in `--prediction-jsonl`, validation identity catalog construction, hook lifecycle, and prediction-artifact metadata.
- Modify `tests/test_run_fusion_count_action_eval_rlpath.py`: opt-in integration and no-capture default-behavior tests.
- Create `scripts/render_three_group_per_example_logits_report.py`: strict detailed-row gates, prior-result equivalence, per-input aggregation, and standalone HTML rendering.
- Create `tests/test_render_three_group_per_example_logits_report.py`: 30,600-row accounting, metric reconstruction, previous-result comparison, malformed-row rejection, and HTML controls.
- Create on the server and pull locally: `experiments/server_command_runs/stage2_three_group_per_example_logits_${TS}/`.
- Create in the repository: `reports/html_reports/${TS}_stage2_three_group_per_example_logits.html`.
- Copy the final report to `/Users/pengjunkai/Desktop/20260711_mrpc_three_group_per_example_logits.html`.

The implementation must not modify `blb_stage2_rl/env.py`, `blb_stage2_rl/runner.py`, reward code, action mapping, replan, bridge installation, or RL training behavior.

### Task 1: Build The Prediction Capture Primitive

**Files:**
- Create: `tests/test_fusion_count_prediction_capture.py`
- Create: `scripts/fusion_count_prediction_capture.py`

- [ ] **Step 1: Write failing identity and capture tests**

Add tests that use NumPy arrays and a `SimpleNamespace(logits=logit_array)`
model output, so the helper stays importable without loading PyTorch at module
import time:

```python
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


class PredictionCaptureTest(unittest.TestCase):
    def test_identity_catalog_resolves_duplicate_token_rows_once_per_trial(self):
        from scripts.fusion_count_prediction_capture import ExampleIdentityCatalog

        rows = [
            {"idx": 7, "input_ids": [101, 10, 102], "token_type_ids": [0, 0, 0], "labels": 1},
            {"idx": 9, "input_ids": [101, 10, 102], "token_type_ids": [0, 0, 0], "labels": 1},
        ]
        catalog = ExampleIdentityCatalog.from_tokenized_rows(rows)
        resolver = catalog.new_trial_resolver()

        self.assertEqual(resolver.resolve([101, 10, 102, 0], [1, 1, 1, 0], [0, 0, 0, 0], 1), 7)
        self.assertEqual(resolver.resolve([101, 10, 102, 0], [1, 1, 1, 0], [0, 0, 0, 0], 1), 9)
        resolver.assert_complete()

    def test_forward_recorder_partitions_batches_into_trials_and_writes_rows(self):
        from scripts.fusion_count_prediction_capture import (
            ExampleIdentityCatalog,
            ForwardPredictionRecorder,
        )

        catalog = ExampleIdentityCatalog.from_tokenized_rows([
            {"idx": 10, "input_ids": [101, 11, 102], "token_type_ids": [0, 0, 0], "labels": 0},
            {"idx": 11, "input_ids": [101, 12, 102], "token_type_ids": [0, 0, 0], "labels": 1},
        ])
        recorder = ForwardPredictionRecorder(catalog=catalog, probe_batch_count=1)
        recorder.begin_group(run_seed=100, group="all_fusion0")
        recorder.hook(
            None,
            (),
            {
                "input_ids": np.asarray([[101, 11, 102, 0], [101, 12, 102, 0]]),
                "attention_mask": np.asarray([[1, 1, 1, 0], [1, 1, 1, 0]]),
                "token_type_ids": np.zeros((2, 4), dtype=np.int64),
                "labels": np.asarray([0, 1]),
            },
            SimpleNamespace(logits=np.asarray([[2.0, -1.0], [-0.5, 0.8]], dtype=np.float32)),
        )
        rows = recorder.finish_group(trial_seeds=[123])

        self.assertEqual([row["dataset_idx"] for row in rows], [10, 11])
        self.assertEqual([row["predicted_label"] for row in rows], [0, 1])
        self.assertTrue(all(row["correct"] for row in rows))
        self.assertEqual(rows[0]["logits"], [2.0, -1.0])
        self.assertEqual(rows[0]["input_ids"], [101, 11, 102, 0])
        self.assertEqual(rows[0]["trial_seed"], 123)

    def test_jsonl_writer_emits_strict_json_rows(self):
        from scripts.fusion_count_prediction_capture import PredictionJsonlWriter

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "predictions.jsonl"
            with PredictionJsonlWriter(path) as writer:
                writer.write_rows([{
                    "schema_version": "fusion-count-per-example-v1",
                    "dataset_idx": 1,
                    "logits": [0.1, 0.2],
                    "correct": True,
                }])
            payload = json.loads(path.read_text().strip())

        self.assertEqual(payload["dataset_idx"], 1)
        self.assertEqual(payload["logits"], [0.1, 0.2])
```

- [ ] **Step 2: Push the red-test commit and verify RED on the server**

```bash
git add tests/test_fusion_count_prediction_capture.py
git commit -m "test: specify per-example prediction capture"
git push origin HEAD:refs/heads/codex/fixed-b2-b5-fusion
```

On the school server, check out that exact commit in the detached experiment worktree and run:

```bash
PY=/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python
$PY -m unittest tests.test_fusion_count_prediction_capture -v
```

Expected: FAIL because `scripts.fusion_count_prediction_capture` does not exist.

- [ ] **Step 3: Implement the minimal dependency-light capture module**

Implement these exact public interfaces in
`scripts/fusion_count_prediction_capture.py`:

- `PREDICTION_ROW_SCHEMA = "fusion-count-per-example-v1"`;
- `ExampleIdentityCatalog.from_tokenized_rows(rows)`,
  `new_trial_resolver()`, and read-only `dataset_indices`;
- `TrialIdentityResolver.resolve(input_ids, attention_mask, token_type_ids,
  label)` and `assert_complete()`;
- `ForwardPredictionRecorder(catalog=catalog,
  probe_batch_count=probe_batch_count)`,
  `begin_group(run_seed=run_seed, group=group_name)`,
  `hook(module, args, kwargs, output)`,
  `finish_group(trial_seeds=trial_seeds)`, and `abort_group()`;
- `PredictionJsonlWriter(path)` with `write_rows(rows)`, `close()`, and context
  manager methods.

Implementation requirements:

- build identity keys from non-padding token IDs, matching non-padding
  token-type IDs, and gold label;
- retain complete padded tensors in output rows;
- accept NumPy arrays, Python lists, or tensor-like objects implementing
  `detach()`, `float()`, `cpu()`, and `tolist()`;
- obtain logits from `output.logits`, falling back to `output[1]`;
- require exactly two finite logits for MRPC;
- partition `captured_batch_count` by `probe_batch_count` and require exactly
  one complete partition per supplied trial seed;
- reset the duplicate resolver for every trial;
- compute prediction with deterministic two-logit argmax and correctness from
  the gold label;
- assign `probe_position` from 0 through 407 within every trial;
- reject missing IDs, duplicate reuse, extra forwards, malformed shapes, and
  non-finite logits;
- write JSON with `allow_nan=False` and stream one row per line.

- [ ] **Step 4: Run the focused test and verify GREEN on the server**

```bash
$PY -m unittest tests.test_fusion_count_prediction_capture -v
```

Expected: all capture tests PASS.

- [ ] **Step 5: Commit and push Task 1**

```bash
git add scripts/fusion_count_prediction_capture.py tests/test_fusion_count_prediction_capture.py
git commit -m "Add per-example prediction capture"
git push origin HEAD:refs/heads/codex/fixed-b2-b5-fusion
```

### Task 2: Connect Capture To The Canonical Fixed Evaluator

**Files:**
- Modify: `tests/test_run_fusion_count_action_eval_rlpath.py`
- Modify: `scripts/run_fusion_count_action_eval_rlpath.py`

- [ ] **Step 1: Write failing opt-in integration tests**

Add tests for three behaviors:

```python
def test_run_group_arms_and_finishes_prediction_recorder(self):
    # FakeSeqEnv has one valid terminal step and terminal_probe trial seeds.
    # FakeRecorder records begin_group/finish_group calls.
    result = rlpath._run_group(env, cfg, seed=42, prediction_recorder=recorder)
    self.assertEqual(recorder.begin_calls, [(42, "fixed_b2")])
    self.assertEqual(recorder.finish_calls, [[42, 2654435739]])
    self.assertEqual(result["prediction_capture"]["row_count"], 2)


def test_run_group_aborts_prediction_recorder_on_commit_error(self):
    with self.assertRaisesRegex(RuntimeError, "terminal failure"):
        rlpath._run_group(env, cfg, seed=42, prediction_recorder=recorder)
    self.assertEqual(recorder.abort_count, 1)


def test_prediction_capture_is_disabled_by_default(self):
    parser = rlpath._parser()
    args = parser.parse_args(required_minimum_args)
    self.assertEqual(args.prediction_jsonl, "")
```

Also add a catalog test around `_tokenize_glue()` showing that the original
MRPC `idx` remains available to the returned identity catalog even though the
formatted model dataset exposes only model columns.

- [ ] **Step 2: Run the focused evaluator tests on the server and verify RED**

```bash
$PY -m unittest \
  tests.test_run_fusion_count_action_eval_rlpath.FusionCountActionEvalRLPathTest.test_run_group_arms_and_finishes_prediction_recorder \
  tests.test_run_fusion_count_action_eval_rlpath.FusionCountActionEvalRLPathTest.test_run_group_aborts_prediction_recorder_on_commit_error \
  tests.test_run_fusion_count_action_eval_rlpath.FusionCountActionEvalRLPathTest.test_prediction_capture_is_disabled_by_default -v
```

Expected: FAIL because `_run_group` has no recorder argument and the CLI has no
prediction output option.

- [ ] **Step 3: Implement opt-in evaluator wiring without changing defaults**

Make these surgical changes:

1. Add `_parser()` so parser defaults remain testable and add:

```python
parser.add_argument(
    "--prediction-jsonl",
    default="",
    help="optional per-example prediction JSONL captured from terminal probe forwards",
)
```

2. While tokenizing validation data, build an `ExampleIdentityCatalog` before
   `set_format()` hides `idx`, then attach it to the evaluator as
   `ev.fixed_eval_identity_catalog`.
3. After `_build_seq_env`, create `ForwardPredictionRecorder` only when
   `--prediction-jsonl` is non-empty. Register exactly one top-level hook with:

```python
handle = ev.model.register_forward_hook(recorder.hook, with_kwargs=True)
```

4. Extend `_run_group(seq_env, cfg, *, seed, prediction_recorder=None,
   prediction_writer=None)`:

```python
if prediction_recorder is not None:
    prediction_recorder.begin_group(run_seed=seed, group=str(cfg["name"]))
try:
    # Execute the existing canonical 47-step loop without changing its body.
    if prediction_recorder is not None:
        trial_seeds = terminal_probe["per_worker_trial_seeds"][0]
        prediction_rows = prediction_recorder.finish_group(trial_seeds=trial_seeds)
        prediction_writer.write_rows(prediction_rows)
except Exception:
    if prediction_recorder is not None:
        prediction_recorder.abort_group()
    raise
```

Pass the writer explicitly rather than storing it on the Stage-2 environment.
Remove the hook in `finally`, even when evaluation fails.
5. Add top-level result metadata:

```python
"prediction_artifact": {
    "schema_version": PREDICTION_ROW_SCHEMA,
    "path": str(prediction_jsonl),
    "row_count": writer.row_count,
    "dataset_indices": catalog.dataset_indices,
},
```

When `--prediction-jsonl` is absent, do not build the catalog recorder, do not
register a hook, and preserve existing evaluator output behavior.

- [ ] **Step 4: Run evaluator and capture tests on the server**

```bash
$PY -m unittest \
  tests.test_fusion_count_prediction_capture \
  tests.test_run_fusion_count_action_eval_rlpath -v
```

Expected: all tests PASS, including dependency-light import tests.

- [ ] **Step 5: Commit and push Task 2**

```bash
git add scripts/run_fusion_count_action_eval_rlpath.py \
  tests/test_run_fusion_count_action_eval_rlpath.py
git commit -m "Capture fixed-eval logits by input"
git push origin HEAD:refs/heads/codex/fixed-b2-b5-fusion
```

### Task 3: Build Strict Per-Example Aggregation And HTML

**Files:**
- Create: `tests/test_render_three_group_per_example_logits_report.py`
- Create: `scripts/render_three_group_per_example_logits_report.py`

- [ ] **Step 1: Write failing aggregation and report tests**

Create compact two-example fixtures for helper-level tests and one generated
strict fixture covering all 30,600 rows. Tests must assert:

```python
def test_build_prediction_summary_recomputes_metrics_and_input_aggregates(self):
    summary = report.build_prediction_summary(
        run_payloads=current_runs,
        prediction_rows=rows,
        prior_run_payloads=prior_runs,
        source_commit="abc123",
        expected_examples=2,
    )
    self.assertTrue(summary["all_gates_pass"])
    self.assertEqual(summary["row_count"], 5 * 3 * 5 * 2)
    self.assertEqual(summary["groups"]["all_fusion0"]["inputs"]["10"]["correct_count"], 25)
    self.assertEqual(summary["trial_results"][0]["incorrect_dataset_indices"], [])


def test_prediction_gate_rejects_wrong_argmax_duplicate_idx_and_nonfinite_logits(self):
    # Mutate one row at a time and assert structured failures:
    # prediction_argmax, duplicate_dataset_idx, non_finite_logits.


def test_prior_metric_gate_rejects_changed_trial_metric(self):
    current_runs[0]["group_results"][0]["trial_metrics"]["metric1"][0] += 0.001
    summary = report.build_prediction_summary(
        run_payloads=current_runs,
        prediction_rows=rows,
        prior_run_payloads=prior_runs,
        source_commit="abc123",
        expected_examples=2,
    )
    self.assertFalse(summary["all_gates_pass"])
    self.assertIn("prior_trial_metric_mismatch", failure_codes(summary))


def test_html_embeds_all_rows_and_filter_controls(self):
    html_text = report.render_html(summary, rows)
    self.assertIn('id="seed-filter"', html_text)
    self.assertIn('id="group-filter"', html_text)
    self.assertIn('id="trial-filter"', html_text)
    self.assertIn('id="correct-filter"', html_text)
    self.assertIn('id="dataset-idx-filter"', html_text)
    self.assertIn('id="prediction-data"', html_text)
    self.assertIn("input_ids", html_text)
    self.assertIn("logits", html_text)
```

The strict CLI fixture generates five seeds, three groups, five trials, and 408
rows per trial with short synthetic token vectors. It verifies an exact row
count of 30,600 without storing a large fixture in Git.

- [ ] **Step 2: Run report tests on the server and verify RED**

```bash
$PY -m unittest tests.test_render_three_group_per_example_logits_report -v
```

Expected: FAIL because the report module does not exist.

- [ ] **Step 3: Implement strict prediction gates and aggregation**

The report CLI is:

```text
--run-json PATH            repeated five times
--prediction-jsonl PATH    repeated five times
--prior-run-json PATH      repeated five times
--source-commit SHA
--output-json PATH
--output-html PATH
```

Reuse these symbols from `scripts.render_three_group_fusion_stability_report`:

```python
EXPECTED_SEEDS
GROUP_SPECS
METRICS
build_summary
```

Implement gates named:

- `base_three_group`: existing result/replan/fusion/K gates all pass;
- `prediction_completeness`: exact five files, 30,600 rows, and exact hierarchy;
- `input_identity`: every trial has the same 408 unique dataset IDs and stable
  token inputs/gold labels per ID;
- `logits_prediction`: two finite logits, argmax prediction, and correct flag;
- `recomputed_metrics`: row-derived Accuracy and Weighted F1 equal trial
  metrics within `1e-12`, and row-derived cross entropy equals trial loss
  within `1e-6`;
- `shared_trial_seeds`: all groups use the same five trial seeds in a run;
- `prior_equivalence`: every current raw trial loss/Accuracy/F1 matches the
  committed prior result within `1e-9`.

For every run/group/trial, emit:

```python
{
    "seed": seed,
    "group": group,
    "trial_index": trial_index,
    "trial_seed": trial_seed,
    "correct_count": correct_count,
    "incorrect_count": 408 - correct_count,
    "correct_dataset_indices": correct_ids,
    "incorrect_dataset_indices": incorrect_ids,
    "recomputed_loss": recomputed_loss,
    "recomputed_accuracy": recomputed_accuracy,
    "recomputed_weighted_f1": recomputed_weighted_f1,
}
```

For every group and `dataset_idx`, aggregate all 25 rows into correctness count
and rate plus two-logit mean and population standard deviation.

Always write strict diagnostic JSON and HTML. Return exit code 1 when any gate
fails.

- [ ] **Step 4: Implement the self-contained paginated HTML**

The HTML must contain static protocol/gate/group/trial summary tables and an
embedded strict JSON payload:

```html
<script type="application/json" id="prediction-data">JSON_PAYLOAD_ESCAPED</script>
```

Client JavaScript parses this element, applies select/input filters, and renders
100 rows per page. Required controls are `seed-filter`, `group-filter`,
`trial-filter`, `correct-filter`, and `dataset-idx-filter`. The detail table
shows dataset ID, full input IDs, attention mask, token-type IDs, gold label,
prediction, correctness, and both logits. Never create all 30,600 `<tr>` nodes
at once.

Serialize embedded JSON with compact separators and replace `</` with `<\/` so
data cannot terminate the script element.

- [ ] **Step 5: Run report tests and full existing torch-free suites on server**

```bash
$PY -m unittest \
  tests.test_render_three_group_per_example_logits_report \
  tests.test_render_three_group_fusion_stability_report \
  tests.test_run_fusion_count_action_eval_rlpath \
  tests.test_fusion_count_prediction_capture \
  tests.test_report_fusion_count_map -v
```

Expected: all tests PASS with no regression.

- [ ] **Step 6: Commit and push Task 3**

```bash
git add scripts/render_three_group_per_example_logits_report.py \
  tests/test_render_three_group_per_example_logits_report.py
git commit -m "Add per-example logits stability report"
git push origin HEAD:refs/heads/codex/fixed-b2-b5-fusion
```

### Task 4: Run An Exact-Snapshot Capture Gate On The Server

**Files:**
- Create on server: `/var/tmp/root-home/rfr_runs/stage2_per_example_gate_${SOURCE_COMMIT:0:7}_${TS}/`
- Create on server: first-seed result and prediction JSONL

- [ ] **Step 1: Prepare an isolated exact Git snapshot**

Do not touch the dirty shared server worktree and do not stop unrelated jobs:

```bash
SERVER_REPO=/var/tmp/root-home/Reinforcement-For-Robustness
git -C "$SERVER_REPO" fetch origin codex/fixed-b2-b5-fusion
SOURCE_COMMIT=$(git -C "$SERVER_REPO" rev-parse origin/codex/fixed-b2-b5-fusion)
TS=$(date +%Y%m%d_%H%M%S)
RUNROOT=/var/tmp/root-home/rfr_runs/stage2_per_example_gate_${SOURCE_COMMIT:0:7}_${TS}
git -C "$SERVER_REPO" worktree add --detach "$RUNROOT" "$SOURCE_COMMIT"
printf '%s\n' "$SOURCE_COMMIT" > "$RUNROOT/SOURCE_SYNC_COMMIT"
```

- [ ] **Step 2: Run the complete focused server test gate**

```bash
cd "$RUNROOT"
PY=/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python
$PY -m unittest \
  tests.test_fusion_count_prediction_capture \
  tests.test_run_fusion_count_action_eval_rlpath \
  tests.test_render_three_group_per_example_logits_report \
  tests.test_render_three_group_fusion_stability_report \
  tests.test_report_fusion_count_map -v 2>&1 | tee server_tests.log
```

Expected: all tests PASS.

- [ ] **Step 3: Generate and gate exactly three current action configs**

Use `scripts/report_fusion_count_map.py` with MRPC GELU4/Softmax6, then copy only:

```text
all_fusion0.json
block2_block5_all_layers_fusionmax.json
block2_block4_block5_all_layers_fusion1.json
```

Run the same exact 0/24/36 action gate used by the prior experiment before any
model forward.

- [ ] **Step 4: Run seed 20260721 with detailed capture**

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/var/tmp/root-home/.cache/huggingface \
TRANSFORMERS_CACHE=/var/tmp/root-home/.cache/huggingface/hub \
HF_DATASETS_CACHE=/var/tmp/root-home/.cache/huggingface/datasets \
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
TOKENIZERS_PARALLELISM=false \
$PY scripts/run_fusion_count_action_eval_rlpath.py \
  --dataset mrpc \
  --model-type bert-base \
  --action-dir "$RUNROOT/selected_actions" \
  --original-json "$RUNROOT/dummy_original.json" \
  --output-json "$RUNROOT/gate/results.json" \
  --output-html "$RUNROOT/gate/results.html" \
  --run-output-dir "$RUNROOT/gate/runtime" \
  --prediction-jsonl "$RUNROOT/gate/predictions.jsonl" \
  --stage1-gelu '[4,4,4,4,4,4,4,4,4,4,4,4]' \
  --stage1-softmax '[6,6,6,6,6,6,6,6,6,6,6,6]' \
  --repeat 5 --probe-size 408 --batch-size 64 \
  --seed 20260721 --shared-group-seed
```

- [ ] **Step 5: Gate the first-seed capture before the full rerun**

Require:

- 6,120 JSONL rows;
- three groups, five trials each, 408 rows per trial;
- exact dataset-ID set in every trial;
- all logits finite and prediction/correctness internally consistent;
- row-derived Accuracy/F1 equal evaluator trial metrics;
- current raw trial metrics match
  `experiments/server_command_runs/stage2_three_group_fusion_stability_20260711_205542/runs/seed_20260721/results.json`
  within the specified tolerances.

Stop and preserve artifacts on any failure.

### Task 5: Rerun All 75 Evaluations And Render The Report

**Files:**
- Create on server: `experiments/server_command_runs/stage2_three_group_per_example_logits_${TS}/`
- Create per run: `results.json`, `results.html`, `predictions.jsonl`, logs, and status
- Create aggregate: `prediction_summary.json`, `three_group_per_example_logits.html`

- [ ] **Step 1: Record immutable metadata**

Record source SHA, Python path, server/GPU inventory, exact environment, five
seeds, action hashes, prior-result hashes, and full commands under the new
output directory.

Define the output location once and retain `TS` in metadata:

```bash
TS=$(date +%Y%m%d_%H%M%S)
OUT="$RUNROOT/experiments/server_command_runs/stage2_three_group_per_example_logits_${TS}"
mkdir -p "$OUT"
printf '%s\n' "$TS" > "$OUT/TIMESTAMP"
```

- [ ] **Step 2: Run five seeds sequentially on GPU 0**

For each seed in the exact schedule, run the Task 4 evaluator command with a
seed-specific `results.json` and `predictions.jsonl`. Keep all three groups in
one process so they share the same model, probe subset, and trial-seed stream.

Expected per run:

- three `[run]` markers;
- 15 model trials and 6,120 prediction rows;
- no traceback, OOM, invalid action, or failed replan/model-use evidence;
- exit code 0.

- [ ] **Step 3: Aggregate current, prediction, and prior artifacts**

Run:

```bash
$PY scripts/render_three_group_per_example_logits_report.py \
  --run-json "$OUT/runs/seed_20260721/results.json" \
  --run-json "$OUT/runs/seed_20261721/results.json" \
  --run-json "$OUT/runs/seed_20262721/results.json" \
  --run-json "$OUT/runs/seed_20263721/results.json" \
  --run-json "$OUT/runs/seed_20264721/results.json" \
  --prediction-jsonl "$OUT/runs/seed_20260721/predictions.jsonl" \
  --prediction-jsonl "$OUT/runs/seed_20261721/predictions.jsonl" \
  --prediction-jsonl "$OUT/runs/seed_20262721/predictions.jsonl" \
  --prediction-jsonl "$OUT/runs/seed_20263721/predictions.jsonl" \
  --prediction-jsonl "$OUT/runs/seed_20264721/predictions.jsonl" \
  --prior-run-json "$PRIOR/runs/seed_20260721/results.json" \
  --prior-run-json "$PRIOR/runs/seed_20261721/results.json" \
  --prior-run-json "$PRIOR/runs/seed_20262721/results.json" \
  --prior-run-json "$PRIOR/runs/seed_20263721/results.json" \
  --prior-run-json "$PRIOR/runs/seed_20264721/results.json" \
  --source-commit "$SOURCE_COMMIT" \
  --output-json "$OUT/prediction_summary.json" \
  --output-html "$OUT/three_group_per_example_logits.html"
```

Expected: exit 0 and every gate passes.

- [ ] **Step 4: Verify the final HTML data contract on the server**

Assert that the HTML is non-empty, contains the five filter controls, embeds
exactly 30,600 rows, displays GELU4/Softmax6/K13 and 0/24/36 fusion totals, and
contains both correct and incorrect records with final logits. Extract the
inline JavaScript to a temporary file and run `node --check` when Node is
available; otherwise rely on the server `unittest` HTML/JavaScript contract
tests and record that Node syntax validation was unavailable.

### Task 6: Pull, Verify, Publish, And Copy To Desktop

**Files:**
- Create: `experiments/server_command_runs/stage2_three_group_per_example_logits_${TS}/`
- Create: `reports/html_reports/${TS}_stage2_three_group_per_example_logits.html`
- Copy: `/Users/pengjunkai/Desktop/20260711_mrpc_three_group_per_example_logits.html`

- [ ] **Step 1: Generate a server SHA-256 manifest**

Exclude caches and model/checkpoint files. Include raw prediction JSONL,
result JSONs, summary JSON, HTML, logs, action configs, test evidence, and gate
evidence.

- [ ] **Step 2: Pull the compact artifact directory as one tar stream**

Use a tar stream rather than recursive SCP because the prior SCP transfer
silently omitted ignored/log files:

```bash
ssh -p 8722 root@100.64.229.185 \
  tar -C "$SERVER_ARTIFACT_PARENT" -cf - "$SERVER_ARTIFACT_NAME" |
  tar -C experiments/server_command_runs -xf -
```

- [ ] **Step 3: Verify every local artifact hash and gate**

```bash
ARTIFACT_NAME="stage2_three_group_per_example_logits_${TS}"
cd "experiments/server_command_runs/$ARTIFACT_NAME"
sha256sum -c SHA256SUMS
jq -e '.all_gates_pass == true and .row_count == 30600' prediction_summary.json
```

Also verify local file count, source SHA, five seeds, 75 trials, 408 examples
per trial, and HTML/summary hashes.

- [ ] **Step 4: Copy the final HTML to repository reports and Desktop**

```bash
cp three_group_per_example_logits.html \
  "../../../reports/html_reports/${TS}_stage2_three_group_per_example_logits.html"
cp three_group_per_example_logits.html \
  /Users/pengjunkai/Desktop/20260711_mrpc_three_group_per_example_logits.html
cmp -s three_group_per_example_logits.html \
  /Users/pengjunkai/Desktop/20260711_mrpc_three_group_per_example_logits.html
```

- [ ] **Step 5: Commit and push all compact artifacts**

Force-add ignored evaluation logs that belong to the manifest:

```bash
git add experiments/server_command_runs/stage2_three_group_per_example_logits_* \
  reports/html_reports/*_stage2_three_group_per_example_logits.html
git add -f experiments/server_command_runs/stage2_three_group_per_example_logits_*/**/*.log
git commit -m "Record per-example fusion logits results"
git push origin HEAD:refs/heads/codex/fixed-b2-b5-fusion
```

- [ ] **Step 6: Perform fresh completion verification**

Require a clean worktree, local HEAD equal to the remote branch SHA, all server
tests passing, all prediction/report gates passing, all artifact hashes
matching, and the Desktop HTML byte-identical to the committed report copy.

The final chat summary reports the Desktop path, row count, correct/incorrect
counts by group, examples that change correctness across groups/noise trials,
and the implementation/result commit SHAs.
