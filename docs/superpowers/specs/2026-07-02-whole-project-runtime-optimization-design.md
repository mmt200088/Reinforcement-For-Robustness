# Whole-Project Runtime Optimization Design

## Objective

Optimize the project end to end for runtime efficiency and hardware
utilization, without changing research semantics or weakening validation
protocols. The work covers every active flow: launcher/presets, Stage-1
plaintext RL, Stage-2 BLB RL, Rescale_optimizer and fusion-map paths, Paean
final evaluation, structured data capture, report generation, server command
execution, and artifact sync.

## Current Flow Map

The authoritative user-facing entrypoints are:

- `llama_7B_LayerImportance.sh` for training, search, compare, and compatible
  eval dispatch.
- `Paean/run_final_eval.sh` and `Paean/run_final_eval.py` for standalone final
  evaluation.
- `SERVER_COMMAND.md` for approved server-side command execution.

The active optimization chain is:

```text
launcher/presets/server bridge
  -> Stage-1 plaintext RL and validation_full evaluation
  -> Stage-2 BLB RL action decode, rollout, reward probe, PPO update
  -> Rescale_optimizer replan / fusion-map cost source
  -> Paean final evaluation and random/range controls
  -> structured JSONL/JSON/NPZ/HTML artifacts
  -> local/server git sync and reports
```

## Non-Negotiable Invariants

- Preserve Stage-1 and Stage-2 validation-only protocol unless the user
  explicitly changes it.
- Do not use GRPO for new Stage-1 or Stage-2 runs.
- Do not change reward, action, or final-eval semantics merely to improve speed.
- Do not force GPU onto pure-Python Rescale_optimizer enumeration unless a
  correct GPU-equivalent implementation is designed and verified.
- Do not edit server source directly; source changes are local commit/push,
  then server pull or verified source snapshot.
- Avoid core Stage-2 RL algorithm files while another agent is actively editing
  them, unless the user coordinates a handoff.
- Every long-running RL run must preserve structured `rl_training_data_points/`
  output.

## Optimization Strategy

Optimization happens in three layers.

1. **Measurement and gates.** Build one project-wide view of flow files,
   artifact evidence, GPU utilization, timing fields, and missing diagnostics.
   This prevents local micro-optimizations from hiding a downstream bottleneck.

2. **Low-conflict runtime improvements.** Prefer launcher checks, artifact
   writers, profiling tools, report decoupling, caching, batching, and
   scheduling improvements before touching RL semantics.

3. **Hot-path optimizations with parity evidence.** For Stage-1, Stage-2,
   Rescale, and Paean hot paths, require focused tests plus server A/B evidence
   where hardware behavior matters.

## Flow-Specific Optimization Surfaces

### Launcher / Presets / Server Bridge

Files:

- `llama_7B_LayerImportance.sh`
- `presets/*.conf`
- `Paean/presets/*.conf`
- `SERVER_COMMAND.md`
- `scripts/launcher_gpu_audit.py`
- `scripts/project_optimization_audit.py`

Optimization targets:

- Warn or fail early when visible GPUs are not forwarded to RL/final-eval flags.
- Validate that `stage2-k-trials` is compatible with reward devices.
- Capture source commit, dirty-state summary, device inventory, and artifact
  roots before server runs.
- Keep server commands scoped to the active objective and avoid idle hardware.

### Stage-1 Plaintext RL

Files:

- `layer_importance_evaluator.py`
- `stage1_rl/parallel_runner.py`
- `stage1_rl/eval_cache.py`
- `function_handler.py`
- `tests/test_stage1_*.py`

Optimization targets:

- Keep validation_full semantics while reducing redundant model forward passes.
- Reuse deterministic Stage-1 eval cache across single-GPU and N-GPU paths.
- Improve data transfer and DataLoader settings only when benchmark evidence
  shows benefit.
- Keep four-GPU rollout collection semantically equivalent to one-GPU seeded
  collection.
- Move heavy report rendering out of the training hot path.

### Stage-2 BLB RL

Files:

- `blb_stage2_rl/parallel_runner.py`
- `blb_stage2_rl/probe_runner.py`
- `blb_stage2_rl/sequential_runner.py`
- `blb_stage2_rl/sequential_env.py`
- `blb_stage2_rl/candidate_store.py`
- `blb_stage2_rl/diagnostics.py`
- `scripts/stage2_ngpu_*.py`
- `scripts/gpu_utilization_report.py`

Optimization targets:

- Prefer episode-parallel GPU workers for fusion-count mode when parity gates
  pass.
- Balance worker assignment, K trials, and reward devices to avoid idle GPUs.
- Cache deterministic action decode / effective config / cost identity where
  safe.
- Keep JSONL diagnostics sufficient for paper figures while minimizing per-step
  synchronous I/O.
- Use 1GPU vs NGPU gates for rollout signature, PPO-visible equality, and
  terminal metric equality.

### Rescale_optimizer / Fusion Maps

Files:

- `Rescale_optimizer/rescale_optimizer/replan_interface.py`
- `Rescale_optimizer/rescale_optimizer/replan.py`
- `scripts/blb_build_fusion_count_map.py`
- `blb_stage2_rl/fusion_count_map.py`
- `scripts/report_fusion_count_map.py`

Optimization targets:

- Reuse `ReplanSession` and loaded graph/baseline data.
- Memoize repeated replan payloads only when keys include all semantic inputs.
- Keep large fusion-map builds streaming; do not retain huge intermediate
  valid-config lists.
- Separate expensive map builds from summary/report parsing.
- Use CPU process/thread parallelism deliberately for pure-Python enumeration.

### Paean Final Evaluation

Files:

- `Paean/run_final_eval.py`
- `Paean/config.py`
- `Paean/action_grid.py`
- `Paean/blb_action_eval.py`
- `Paean/final_eval_layout.py`
- `final_evaluation_module.py`

Optimization targets:

- Reuse model and tokenizer initialization across repeated configs.
- Batch action-grid/range evaluations by shared Stage-1 and Stage-2 settings.
- Add optional N-GPU scheduling for independent final-eval configs.
- Decouple GLUE submission/report rendering from the evaluation hot path.
- Preserve standalone Paean output layout and compatibility dispatch from the
  main launcher.

### Structured Data / Reports / Artifact Sync

Files:

- `rl_data_points.py`
- `rl_training_data_points/`
- `reports/`
- `experiments/server_command_runs/`
- `tools/paper_figures.py`
- `scripts/verify_stage2_persistent_outputs.py`

Optimization targets:

- Keep raw JSON/JSONL complete enough to redraw figures without rerunning
  training.
- Write compact hot-path records first; render PNG/HTML/NPZ after training.
- Add project-wide artifact summaries that make missing diagnostics obvious.
- Keep local/server artifact sync reproducible and tied to source commits.

## Verification Model

Every optimization gets at least one of these evidence types:

- Unit tests for pure parsing, caching, scheduling, or artifact logic.
- `bash -n` for launcher changes.
- `py_compile` and `ruff` for new Python tooling.
- 1GPU vs NGPU equality gate for Stage-1 or Stage-2 rollout changes.
- Server A/B wall-clock evidence for actual GPU/CPU scheduling changes.
- Final-eval metric equality or bounded stochastic repeat evidence for Paean
  changes.

## Initial Deliverables

1. `scripts/project_optimization_audit.py`: project-wide static flow and
   artifact evidence summary.
2. `tests/test_project_optimization_audit.py`: deterministic tests for the
   audit tool.
3. `docs/superpowers/plans/2026-07-02-whole-project-runtime-optimization.md`:
   implementation plan that sequences all optimization phases.

## Completion Definition

The goal is complete only when the project has:

- A committed whole-project optimization plan.
- Committed tooling that can report current optimization evidence across the
  full flow.
- Implemented optimizations for launcher/server, Stage-1, Stage-2, Rescale,
  Paean, reports, and artifact sync.
- Verification evidence for every changed flow, including server A/B evidence
  where hardware utilization is the claim.
- No known remaining high-impact optimization phase in the plan left
  unimplemented.
