# MRPC All-Fusion1 Installed SF Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the B2/B4/B5 all-fusion1 configuration reaches the model through the canonical Stage-2 chain, capture its final post-replan installed SF values, and publish a self-contained HTML keyed by MRPC validation rows `0..407`.

**Architecture:** Add one dependency-light audit/report module and one server entrypoint. Wrap the existing fixed evaluator's `bridge.apply` boundary, call the original installer, verify function-handler object identity, and serialize only those post-replan cfg objects. Translate historical GLUE indices to unshuffled validation ordinals and render the existing 25-draw outcomes without exposing obsolete identifiers.

**Tech Stack:** Python 3.10, existing PyTorch/Transformers/Datasets Stage-2 fixed evaluator, JSON/JSONL, `unittest`, static HTML/CSS/JavaScript.

---

## File Map

- Create `scripts/fusion_count_installed_sf_audit.py`: cfg serialization,
  bridge wrapper, validation-row mapping, gates, aggregation, and HTML renderer.
- Create `scripts/run_mrpc_allfusion1_installed_sf_audit.py`: server-only exact
  evaluator orchestration and artifact writer.
- Create `tests/test_fusion_count_installed_sf_audit.py`: torch-free unit tests
  for provenance, identity verification, mapping, gates, and report content.
- Create on server then pull locally under
  `experiments/server_command_runs/stage2_mrpc_allfusion1_installed_sf_audit_<ts>/`.
- Create `reports/html_reports/<ts>_mrpc_allfusion1_installed_sf_audit.html`.
- Copy final HTML to
  `/Users/pengjunkai/Desktop/20260712_mrpc_allfusion1_actual_sf_audit.html`.

Do not modify Stage-2 RL, reward, replan, bridge, handler, or model code.

### Task 1: Specify The Installed-Config Capture Contract

**Files:**
- Create: `tests/test_fusion_count_installed_sf_audit.py`
- Create: `scripts/fusion_count_installed_sf_audit.py`

- [ ] **Step 1: Write failing serializer and bridge-wrapper tests**

Use fake noise points, cfg objects, bridge, and handler objects. Require scalar
and tuple SF serialization, explicit fused-away rows, and rejection of any
provenance other than `post_replan_bridge_apply`:

```python
rows = serialize_installed_cfgs(
    block2_cfgs=[cfg], block4_cfgs=[cfg], block5_cfgs=[cfg],
    provenance="post_replan_bridge_apply",
)
self.assertEqual(rows[0]["scaling_factor"], 21)
self.assertEqual(rows[1]["installation_state"], "fused_away")
with self.assertRaisesRegex(ValueError, "authoritative"):
    serialize_installed_cfgs(..., provenance="map_option")
```

Wrap a fake `bridge.apply`, require the original call to occur, and assert the
handler's active cfg objects are identical to the supplied cfg objects before
the capture is accepted.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest tests.test_fusion_count_installed_sf_audit -v
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the minimal capture primitives**

Implement:

```python
AUTHORITATIVE_PROVENANCE = "post_replan_bridge_apply"
serialize_installed_cfgs(...)
InstalledConfigCapture(original_apply, handler)
InstalledConfigCapture.apply(...)
InstalledConfigCapture.assert_complete()
```

The wrapper must call the original installer first, verify active layer sets
and object identity, then deep-serialize primitive audit rows. It must reject
zero or multiple conflicting captures.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same unittest command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/fusion_count_installed_sf_audit.py tests/test_fusion_count_installed_sf_audit.py
git commit -m "Add installed SF capture audit"
```

### Task 2: Build Stable MRPC Row Aggregation And Minimal HTML

**Files:**
- Modify: `tests/test_fusion_count_installed_sf_audit.py`
- Modify: `scripts/fusion_count_installed_sf_audit.py`

- [ ] **Step 1: Write failing row-mapping and report tests**

Specify a strict bijection from unshuffled source rows to `0..N-1`, translate
historical prediction rows by source `idx`, and aggregate exactly 25 outcomes
per group. Verify rendered HTML contains `Validation row 0` and row `407` but
does not contain identifier labels `dataset_idx`, `input_ids`, or
`probe_position`.

- [ ] **Step 2: Run focused tests and verify RED**

Run the focused unittest. Expected: missing mapping/report functions.

- [ ] **Step 3: Implement mapping, gates, and renderer**

Implement:

```python
build_validation_row_lookup(source_rows)
aggregate_prediction_rows(prediction_paths, row_lookup, expected_trials=25)
validate_action_result(result)
render_audit_html(payload)
```

Each aggregate row must retain only `validation_row_id`, gold label, and trial
outcomes needed for correctness/logit display. Never expose source `idx`, token
arrays, or probe order in the HTML payload.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the focused unittest and inspect the generated fixture HTML.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/fusion_count_installed_sf_audit.py tests/test_fusion_count_installed_sf_audit.py
git commit -m "Add MRPC row audit report"
```

### Task 3: Orchestrate The Canonical Server Audit

**Files:**
- Create: `scripts/run_mrpc_allfusion1_installed_sf_audit.py`
- Modify: `tests/test_fusion_count_installed_sf_audit.py`

- [ ] **Step 1: Write a failing orchestration seam test**

Use patched fixed-evaluator builders to require that the entrypoint:

1. loads only `block2_block4_block5_all_layers_fusion1.json`;
2. installs the wrapper after environment construction;
3. calls `_run_group_canonical` exactly once;
4. restores the original bridge method in `finally`;
5. writes an audit JSON and HTML only after all gates pass.

- [ ] **Step 2: Run the focused test and verify RED**

Run the focused unittest. Expected: missing entrypoint module.

- [ ] **Step 3: Implement the server entrypoint**

Add CLI options for action config, historical artifact root, output JSON/HTML,
dataset/model cache paths, and seed. Reuse the fixed evaluator's existing
builders and canonical group runner; do not duplicate action mapping or model
evaluation logic. Replace only the instance's `bridge.apply` method during the
candidate run and restore it in `finally`.

- [ ] **Step 4: Run local torch-free verification**

```bash
python -m unittest tests.test_fusion_count_installed_sf_audit -v
python -m py_compile scripts/fusion_count_installed_sf_audit.py scripts/run_mrpc_allfusion1_installed_sf_audit.py
```

Expected: all tests pass and compilation succeeds.

- [ ] **Step 5: Commit and push the verified source snapshot**

```bash
git add scripts/run_mrpc_allfusion1_installed_sf_audit.py tests/test_fusion_count_installed_sf_audit.py
git commit -m "Add canonical MRPC installed SF audit"
git push origin HEAD:refs/heads/codex/block-sf-audit
```

### Task 4: Run On Server And Publish Artifacts

**Files:**
- Create from server output:
  `experiments/server_command_runs/stage2_mrpc_allfusion1_installed_sf_audit_<ts>/`
- Create: `reports/html_reports/<ts>_mrpc_allfusion1_installed_sf_audit.html`

- [ ] **Step 1: Check out the exact pushed commit on the school server**

Use a separate server worktree and record `SOURCE_SYNC_COMMIT`. Do not edit
source on the server and do not stop unrelated processes.

- [ ] **Step 2: Run focused tests on the server**

Run the exact Python environment's unittest and record the exit code/log.
Expected: all audit tests pass.

- [ ] **Step 3: Run one full-validation canonical all-fusion1 audit**

Use GELU4/Softmax6/K13 and the committed all1 action. Expected gates: 47 valid
steps, fusion totals B2=12/B4=12/B5=12, all handler identities true, 408 finite
evaluation examples, and installed-SF provenance exclusively from bridge args.

- [ ] **Step 4: Verify artifacts before copying**

Parse the JSON and assert layers `0..11`, rows `0..407`, 25 outcomes per
historical group/row, no obsolete identifier labels in HTML, and matching
SHA-256 manifest entries.

- [ ] **Step 5: Pull, commit, push, and copy the final HTML**

Copy compact artifacts locally, create the repository report copy, commit and
push them from the local worktree, then copy the verified HTML to the Desktop
path specified in the design.
