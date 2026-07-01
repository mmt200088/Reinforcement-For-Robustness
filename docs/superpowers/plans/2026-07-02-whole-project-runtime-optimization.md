# Whole-Project Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the entire project runtime path, from launch to reports, while preserving research semantics and validation protocols.

**Architecture:** Start with project-wide observability, then optimize one flow stage at a time behind tests and parity gates. Hardware-sensitive changes require server A/B evidence before becoming defaults.

**Tech Stack:** Bash launcher, Python 3 stdlib tooling, PyTorch/Hugging Face hot paths, JSON/JSONL artifacts, unittest, ruff, git-synced server workflow.

---

## File Structure

- Create `scripts/project_optimization_audit.py`: dependency-free whole-flow inventory and artifact summary.
- Create `tests/test_project_optimization_audit.py`: unit tests for inventory, artifact discovery, and CLI output.
- Modify `llama_7B_LayerImportance.sh`: only for launcher/server resource checks after tests prove behavior.
- Modify `stage1_rl/*` and `layer_importance_evaluator.py`: only for Stage-1 evaluation/rollout improvements with Stage-1 parity evidence.
- Modify `blb_stage2_rl/*`: only after coordinating with concurrent Stage-2 RL work; require 1GPU vs NGPU gates.
- Modify `Rescale_optimizer/rescale_optimizer/*`: only for profiled CPU hot paths or safe memoization.
- Modify `Paean/*` and `final_evaluation_module.py`: only for final-eval batching, reuse, or independent-config scheduling.
- Modify `rl_data_points.py`, reports, and verifier scripts: only for hot-path/report decoupling and artifact integrity.

## Task 1: Whole-Flow Optimization Audit Tool

**Files:**

- Create: `scripts/project_optimization_audit.py`
- Create: `tests/test_project_optimization_audit.py`
- Verify: `docs/superpowers/specs/2026-07-02-whole-project-runtime-optimization-design.md`

- [x] **Step 1: Write failing tests**

Create tests that build a temporary mini-repo with representative files:

```python
root = Path(td)
(root / "llama_7B_LayerImportance.sh").write_text("#!/usr/bin/env bash\n")
(root / "presets").mkdir()
(root / "presets" / "mrpc-blb-stage2-rl.conf").write_text("--stage2-rl-variant\nblb_v3\n")
(root / "blb_stage2_rl").mkdir()
(root / "blb_stage2_rl" / "parallel_runner.py").write_text("")
(root / "run" / "diagnostics").mkdir(parents=True)
(root / "run" / "diagnostics" / "episodes.jsonl").write_text('{"episode":0}\n')
```

Assert that `build_project_audit(root)` reports:

- stage names include `launcher`, `stage1`, `stage2`, `rescale`, `paean`, `artifacts`.
- existing files are marked `present=True`.
- artifact summary counts one `episodes.jsonl`.
- CLI writes JSON and Markdown.

Run:

```bash
python3 -m unittest tests.test_project_optimization_audit -v
```

Expected before implementation: FAIL because the script does not exist.

- [x] **Step 2: Implement the audit tool**

Implement these functions:

```python
def build_project_audit(root: str | Path, artifact_roots: Sequence[str | Path] = ()) -> dict:
    ...

def render_markdown(report: Mapping[str, object]) -> str:
    ...

def main(argv: Sequence[str] | None = None) -> int:
    ...
```

The tool must be stdlib-only, deterministic, and safe on a dirty worktree. It
must not import torch or training modules.

- [x] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_project_optimization_audit -v
python3 -m py_compile scripts/project_optimization_audit.py
python3 -m ruff check scripts/project_optimization_audit.py tests/test_project_optimization_audit.py
```

Expected: all pass.

- [x] **Step 4: Commit**

```bash
git add scripts/project_optimization_audit.py tests/test_project_optimization_audit.py docs/superpowers/specs/2026-07-02-whole-project-runtime-optimization-design.md docs/superpowers/plans/2026-07-02-whole-project-runtime-optimization.md
git commit -m "Add whole-project optimization plan and audit"
```

## Task 2: Launcher and Server Resource Gates

**Files:**

- Modify: `scripts/launcher_gpu_audit.py`
- Modify: `llama_7B_LayerImportance.sh`
- Test: `tests/test_launcher_gpu_audit.py`
- Test: `tests/test_stage2_persistent_launcher.py`

- [x] **Step 1: Extend tests**

Add tests for these cases:

- Stage-1 `--stage1-rl-devices` set to fewer devices than `CUDA_VISIBLE_DEVICES`.
- Stage-2 `--stage2-rl-devices` and `--blb-v3-reward-devices` disagree.
- `RFR_GPU_AUDIT_STRICT=1` fails only when warnings exist.

- [x] **Step 2: Implement warnings only**

Keep default behavior non-fatal. Add strict failure only through
`RFR_GPU_AUDIT_STRICT=1`.

- [x] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_launcher_gpu_audit tests.test_stage2_persistent_launcher -v
bash -n llama_7B_LayerImportance.sh
python3 -m ruff check scripts/launcher_gpu_audit.py tests/test_launcher_gpu_audit.py tests/test_stage2_persistent_launcher.py
```

## Task 3: Stage-1 Evaluation and Rollout Throughput

**Files:**

- Modify: `stage1_rl/eval_cache.py`
- Modify: `stage1_rl/parallel_runner.py`
- Modify: `layer_importance_evaluator.py`
- Create: `scripts/stage1_parallel_report.py`
- Test: `tests/test_stage1_eval_accel.py`
- Test: `tests/test_stage1_parallel_semantics.py`
- Test: `tests/test_stage1_parallel_report.py`

- [ ] **Step 1: Baseline current behavior**

Run focused local tests:

```bash
python3 -m unittest tests.test_stage1_eval_accel tests.test_stage1_parallel_semantics -v
```

- [ ] **Step 2: Add timing fields**

Add Stage-1 window diagnostics for cache hit rate, worker wall seconds,
model-forward wall seconds, and report-write wall seconds. Write them to the
existing Stage-1 log/status path, not to a new hot-path report.

Progress 2026-07-02: added torch-free `scripts/stage1_parallel_report.py` to
summarize existing Stage-1 rollout/cache/component timing logs into JSON and
Markdown for server 1GPU vs 4GPU evidence. Existing
`tests.test_stage1_parallel_semantics` currently fails against the dirty
`layer_importance_evaluator.py` worktree because `_stage1_collect_episode_in_worker`
no longer contains `SOS_TOKEN_SOFTMAX`; this optimization pass did not modify
that core file.

- [ ] **Step 3: Optimize only proven redundant work**

Allowed changes:

- Share deterministic cache for worker evals.
- Avoid rebuilding identical Stage-1 GELU/Softmax installs when the config
  hash is unchanged.
- Keep worker seeding and validation_full split unchanged.

- [ ] **Step 4: Verify**

Run the local tests above, then run a server 1GPU vs 4GPU smoke before changing
defaults.

## Task 4: Stage-2 BLB RL GPU Scheduling and Diagnostics

**Files:**

- Modify: `scripts/stage2_ngpu_ab_compare.py`
- Modify: `scripts/gpu_utilization_report.py`
- Modify only after coordination: `blb_stage2_rl/parallel_runner.py`,
  `blb_stage2_rl/probe_runner.py`, `blb_stage2_rl/sequential_runner.py`
- Test: `tests/test_stage2_ngpu_ab_compare.py`
- Test: `tests/test_gpu_utilization_report.py`
- Test: `tests/test_stage2_parallel_runner.py`

- [x] **Step 1: Strengthen evidence tools**

Ensure reports include:

- per-device episode counts.
- per-device terminal probe wall means.
- policy rollout wall mean.
- replan wall mean.
- JSONL write/report render wall time when present.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now reports
per-device probe episode counts, per-device terminal probe wall statistics,
global policy rollout wall statistics, replan/optimizer wall statistics, and
optional JSONL/report hot-path wall fields when they are present in
`episodes.jsonl`.

- [x] **Step 2: Do not change core RL during concurrent edits**

Until the Stage-2 RL agent handoff is clear, restrict work to tools and gates.

- [ ] **Step 3: Server A/B before defaults**

Use `SERVER_COMMAND.md` to run 1GPU vs NGPU parity and speed checks. Promote a
new default only when effect equality passes and wall-clock evidence improves.

## Task 5: Rescale Optimizer and Fusion Map Runtime

**Files:**

- Modify: `Rescale_optimizer/rescale_optimizer/replan_interface.py`
- Modify: `Rescale_optimizer/rescale_optimizer/replan.py`
- Modify: `scripts/blb_build_fusion_count_map.py`
- Modify: `scripts/report_fusion_count_map.py`
- Test: `tests/test_rescale_optimizer_bridge_cache.py`
- Test: `tests/test_blb_fusion_count_map.py`

- [ ] **Step 1: Profile before editing**

Use existing fusion-map build logs and local unit tests to identify whether
time is in graph loading, feasibility DAG build, replan calls, or summary
parsing.

- [ ] **Step 2: Apply safe reuse**

Allowed changes:

- Cache loaded profile graph data inside `ReplanSession`.
- Cache feasibility DAGs keyed by graph/config hash.
- Stream map summaries without loading sidecars as maps.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_rescale_optimizer_bridge_cache tests.test_blb_fusion_count_map -v
```

Use server only for large fusion-map wall-clock evidence.

## Task 6: Paean Final Evaluation Throughput

**Files:**

- Modify: `Paean/run_final_eval.py`
- Modify: `Paean/action_grid.py`
- Modify: `Paean/blb_action_eval.py`
- Modify: `final_evaluation_module.py`
- Test: `tests/test_final_eval_layout.py`
- Test: `tests/test_blb_final_eval_feasibility.py`

- [x] **Step 1: Add final-eval plan diagnostics**

Expose how many configs, repeats, random controls, and expected model loads a
Paean run will perform before launch.

- [ ] **Step 2: Optimize shared work**

Allowed changes:

- Group action-grid configs by shared Stage-1 install.
- Reuse model/tokenizer initialization inside one final-eval process.
- Schedule independent configs across visible GPUs only after local tests and
  server smoke show no metric drift.

- [ ] **Step 3: Verify**

Run final-eval unit tests locally and a server repeated final-eval smoke for
the same fixed action before/after.

## Task 7: Structured Data and Report Decoupling

**Files:**

- Modify: `rl_data_points.py`
- Modify: `scripts/verify_stage2_persistent_outputs.py`
- Modify report generators under `reports/` or `tools/`
- Test: `tests/test_rl_data_points.py`
- Test: `tests/test_stage2_persistent_output_verifier.py`

- [ ] **Step 1: Protect data completeness**

Add tests that fail if required structured fields are dropped from Stage-1 or
Stage-2 mirrored data.

- [ ] **Step 2: Move expensive rendering out of hot paths**

Keep JSON/JSONL writes in training; move PNG/HTML/NPZ rendering to post-run
commands unless the user explicitly requests live rendering.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_rl_data_points tests.test_stage2_persistent_output_verifier -v
```

## Task 8: Server Evidence and Promotion Loop

**Files:**

- Modify: `SERVER_COMMAND.md`
- Use: `scripts/project_optimization_audit.py`
- Use: `scripts/gpu_utilization_report.py`
- Use: `scripts/stage2_ngpu_ab_compare.py`

- [ ] **Step 1: Write one server command per promoted optimization**

Each command must record:

- source commit.
- GPU inventory.
- exact command.
- output artifact directory.
- timing summary.
- semantic parity/eval evidence.

- [ ] **Step 2: Pull artifacts back locally**

Import compact summaries into `experiments/server_command_runs/` or
`reports/html_reports/` as appropriate.

- [ ] **Step 3: Commit/push source and evidence**

Never leave canonical source changes only on the server.

## Completion Audit

Before marking the objective complete, verify:

- This plan has no unchecked high-impact phases.
- Every implemented optimization has tests and timing evidence.
- Server-side hardware-utilization claims have server artifacts.
- Stage-1 and Stage-2 validation protocols remain intact.
- `git status` shows no uncommitted source changes made by this optimization
  work.
