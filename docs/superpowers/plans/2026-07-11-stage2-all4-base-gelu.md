# Stage-2 Configurable All-GELU4 Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make all-GELU4 the current Stage-2 prerequisite default while retaining explicit `stage1_result`, `json`, and `manual` alternatives.

**Architecture:** The launcher selects one Stage-2 configuration source and passes it through `rl_tune.py` to `LayerImportanceEvaluator`. A dedicated Stage-2 resolver returns one GELU/Softmax pair that is reused by training and its saved/final evaluation path; GELU4 naturally selects `block5_n4` in every layer through the existing action schedule.

**Tech Stack:** Bash launcher, Python, NumPy, `unittest`/pytest, existing BLB Stage-2 action-space and fusion-map modules.

---

### Task 1: Add Failing Configuration And Alignment Contracts

**Files:**
- Create: `tests/test_stage2_all4_base_config.py`
- Modify: `tests/test_stage2_persistent_launcher.py`
- Test: `tests/test_stage2_all4_base_config.py`
- Test: `tests/test_stage2_persistent_launcher.py`

- [ ] **Step 1: Add a resolver contract for the new source**

Create a focused server-side test that constructs `LayerImportanceEvaluator`
with `__new__`, sets `total_layers=12`, and verifies:

```python
ev.stage2_fixed_config_source = "all4"
ev.stage2_fixed_config_path = ""
ev.stage2_manual_gelu = None
ev.stage2_manual_softmax = None
gelu, softmax, label, source = ev._resolve_stage2_fixed_stage1_config()
assert gelu.tolist() == [4] * 12
assert softmax.tolist() == [6] * 12
assert source == "stage2_all4"
assert "all4" in label.lower()
```

- [ ] **Step 2: Add switch-back and Block5 contracts**

Patch the Stage-2 resolver dependency and assert that
`stage2_fixed_config_source="stage1_result"` delegates to the prior searched
configuration path. Build the real action schedule with `[4] * 12` and assert
every Block 5 step uses `block5_n4`:

```python
schedule = step_schedule(12, gelu_degree_per_layer=[4] * 12, profile="mrpc")
assert {s.graph_key_suffix for s in schedule if s.block_idx == 5} == {"block5_n4"}
```

Read each committed profile map directory and assert `block5_n4.json` exists,
has `graph_key == "block5_n4"`, and `gelu_degree == 4`.

- [ ] **Step 3: Add launcher default and explicit override tests**

Use the existing fake-`python` launcher harness. A Stage-2-only invocation with
no Stage-2 fixed-config flags must capture:

```text
--stage2_fixed_config_source all4
--stage2_fixed_config_path ""
```

An invocation with explicit `--stage2-fixed-config-source json` and
`--stage2-fixed-config glue_final_configs_best_ppo.json` must retain `json` and
that path.

- [ ] **Step 4: Add static end-to-end plumbing assertions**

Parse/read `rl_tune.py` and assert its `LayerImportanceEvaluator(...)` call
contains these exact keyword bindings:

```python
stage2_fixed_config_source=stage2_fixed_config_source
stage2_fixed_config_path=stage2_fixed_config_path
stage2_manual_gelu=parsed_stage2_manual_gelu
stage2_manual_softmax=parsed_stage2_manual_softmax
```

Also assert both fixed-action scripts define `DEFAULT_STAGE1_GELU = [4] * 12`.

- [ ] **Step 5: Commit and push the red tests**

```bash
git add tests/test_stage2_all4_base_config.py tests/test_stage2_persistent_launcher.py
git commit -m "test: define configurable Stage-2 GELU4 base"
git push origin HEAD
```

- [ ] **Step 6: Verify the tests fail on the server**

Run from an exact archive/checkout of the red-test commit:

```bash
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m pytest \
  tests/test_stage2_all4_base_config.py \
  tests/test_stage2_persistent_launcher.py -q
```

Expected: failures show unsupported/missing `all4`, missing evaluator plumbing,
and old fixed-action defaults.

### Task 2: Implement The Single Stage-2 Configuration Source

**Files:**
- Modify: `llama_7B_LayerImportance.sh`
- Modify: `rl_tune.py`
- Modify: `layer_importance_evaluator.py`
- Test: `tests/test_stage2_all4_base_config.py`
- Test: `tests/test_stage2_persistent_launcher.py`

- [ ] **Step 1: Extend and default the launcher source**

Allow `all4` in the source case statement and help text. When no Stage-2
source/path/manual array is explicitly provided, set:

```bash
STAGE2_FIXED_CONFIG_SOURCE="all4"
STAGE2_FIXED_CONFIG=""
```

Keep the existing inference rules for explicit manual arrays and explicit JSON
paths. Reject explicit path/manual values with `all4` to avoid ambiguous
precedence.

- [ ] **Step 2: Parse and forward the dedicated Stage-2 values**

In `rl_tune.py`, parse the dedicated arrays separately:

```python
parsed_stage2_manual_gelu = parse_degree_config(stage2_manual_gelu)
parsed_stage2_manual_softmax = parse_degree_config(stage2_manual_softmax)
```

Pass source, path, and parsed arrays to `LayerImportanceEvaluator` rather than
letting Stage-2 reuse `final_eval_config_*` implicitly.

- [ ] **Step 3: Store and validate Stage-2 resolver inputs**

Add constructor parameters to `LayerImportanceEvaluator` and normalize them:

```python
self.stage2_fixed_config_source = (
    str(stage2_fixed_config_source or "all4").strip().lower()
)
self.stage2_fixed_config_path = str(stage2_fixed_config_path or "").strip()
self.stage2_manual_gelu = stage2_manual_gelu
self.stage2_manual_softmax = stage2_manual_softmax
```

Accept only `all4`, `stage1_result`, `json`, or `manual` and reject incomplete
manual arrays or a missing JSON path before Stage-2 begins.

- [ ] **Step 4: Add the dedicated resolver**

Implement `_build_stage2_fixed_config_resolver()` using
`UnifiedFinalEvaluationModule`, mapping only `stage1_result` to its existing
`search` resolver behavior. Update `_resolve_stage2_fixed_stage1_config()`:

```python
if self.stage2_fixed_config_source == "all4":
    gelu = np.full(self.total_layers, 4, dtype=int)
    softmax = np.full(self.total_layers, FIXED_SOFTMAX_DEGREE, dtype=int)
    return gelu, softmax, "Stage-2 all4 (softmax fixed deg6)", "stage2_all4"
```

For `stage1_result`, preserve decoupled record lookup. For `json` and `manual`,
use the dedicated resolver. Continue forcing Softmax to degree 6 for every
source.

- [ ] **Step 5: Run the focused server tests**

```bash
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m pytest \
  tests/test_stage2_all4_base_config.py \
  tests/test_stage2_persistent_launcher.py -q
```

Expected: the resolver, source switching, launcher, plumbing, and map-alignment
tests pass.

- [ ] **Step 6: Commit the core implementation**

```bash
git add llama_7B_LayerImportance.sh rl_tune.py layer_importance_evaluator.py \
  tests/test_stage2_all4_base_config.py tests/test_stage2_persistent_launcher.py
git commit -m "Use configurable GELU4 base for Stage-2"
git push origin HEAD
```

### Task 3: Align Fixed Experiments And Server Commands

**Files:**
- Modify: `scripts/run_fusion_count_action_eval.py`
- Modify: `scripts/run_fusion_count_action_eval_rlpath.py`
- Modify: `SERVER_COMMAND.md`
- Modify: `AGENTS.md`
- Test: `tests/test_stage2_all4_base_config.py`
- Test: `tests/test_run_fusion_count_action_eval.py`
- Test: `tests/test_run_fusion_count_action_eval_rlpath.py`

- [ ] **Step 1: Change fixed-action defaults without removing overrides**

Set both script constants to:

```python
DEFAULT_STAGE1_GELU = [4] * 12
```

Keep `--stage1-gelu` unchanged so historical vectors remain explicitly
selectable.

- [ ] **Step 2: Change executable Stage-2 commands to the new source**

Replace runnable command arguments that select the old PPO JSON with:

```bash
--stage2-fixed-config-source all4
```

Do not rewrite historical result prose or the Stage-1 best JSON artifact.

- [ ] **Step 3: Record the active project contract**

Add a concise `AGENTS.md` note stating that Stage-2 currently defaults to all4,
that `stage1_result` disables this choice, and that Block5 must resolve to
`block5_n4` under all4.

- [ ] **Step 4: Run focused and adjacent tests on the server**

```bash
/var/tmp/root-home/miniconda3/envs/llm_ist/bin/python -m pytest \
  tests/test_stage2_all4_base_config.py \
  tests/test_stage2_persistent_launcher.py \
  tests/test_run_fusion_count_action_eval.py \
  tests/test_run_fusion_count_action_eval_rlpath.py \
  tests/test_blb_fusion_count_map.py -q
```

Expected: all selected tests pass or only documented environment skips occur.

- [ ] **Step 5: Commit and push tool/command alignment**

```bash
git add scripts/run_fusion_count_action_eval.py \
  scripts/run_fusion_count_action_eval_rlpath.py SERVER_COMMAND.md AGENTS.md \
  tests/test_stage2_all4_base_config.py
git commit -m "Align Stage-2 tools with GELU4 base"
git push origin HEAD
```

### Task 4: Verify The Exact Snapshot On The Server

**Files:**
- Create locally after retrieval: `experiments/server_command_runs/stage2_all4_gate_<timestamp>/`

- [ ] **Step 1: Package the exact pushed commit**

Create an isolated server runroot containing the pushed commit and write its
full SHA to `SOURCE_SYNC_COMMIT`. Do not edit source on the server.

- [ ] **Step 2: Run a Stage-2 startup gate**

Launch a short approved Stage-2-only gate using:

```bash
--stage2-fixed-config-source all4
```

Capture logs proving GELU is `[4] * 12`, Softmax is `[6] * 12`, every layer's
Block5 graph is `block5_n4`, maps load, the model reaches the Stage-2 rollout,
and no fallback Stage-1 record/JSON is read.

- [ ] **Step 3: Pull compact evidence and audit it locally**

Retrieve source SHA, launcher argv/log, resolver log, map gate output, and the
short-run status while excluding model checkpoints and caches.

- [ ] **Step 4: Commit and push compact evidence**

```bash
git add experiments/server_command_runs/stage2_all4_gate_<timestamp>
git commit -m "Record Stage-2 GELU4 configuration gate"
git push origin HEAD
```

- [ ] **Step 5: Report the switch semantics**

Report the verified default and the exact opt-out command:

```bash
--stage2-fixed-config-source stage1_result
```
