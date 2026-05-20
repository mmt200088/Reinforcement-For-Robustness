# Stage2 RL First-10k Curve Optimization Plan

> **For agentic workers:** This is a research/debugging loop. Do not declare
> completion from a short unit test or a partial smoke. Use server evidence.

**Goal:** Optimize and monitor the first 10,000 BLB Stage-2 sequential RL
episodes while preserving the no-collapse safety achieved in the 600-episode
run.

**Hypothesis:** The 600-episode run fixed catastrophic collapse, but entropy and
clip fraction show the policy may narrow too early. Exposing curriculum and
entropy knobs plus adding an online watchdog lets us test a wider, slower
exploration schedule over 10k episodes without changing server code directly.

## Task 1: Expose 10k Curve Knobs

Files:
- `llama_7B_LayerImportance.sh`
- `rl_tune.py`
- `layer_importance_evaluator.py`
- `blb_stage2_rl/runner.py`

Steps:

- Add launcher flags for neighbor ramp, max mutations, max radius, neighbor
  sampling, warmstart bias gain, entropy coefficient, anchor entropy, and
  entropy ramp.
- Forward those values through `rl_tune.py` and `LayerImportanceEvaluator`.
- Map evaluator attributes into `BLBStage2TrainConfig`.
- Verify that the launcher help, parser, command construction, and startup
  display mention the new knobs.

## Task 2: Structure Per-Episode Health

Files:
- `blb_stage2_rl/diagnostics.py`
- `blb_stage2_rl/sequential_runner.py`

Steps:

- Add terminal priority/loss/metric and safe-neighbor fields to
  `EpisodeStats`.
- Populate those fields from `EpisodeRecord` when writing `episodes.jsonl`.
- Keep the writer append-only and best-effort.

## Task 3: Update Server Run Command

Files:
- `SERVER_COMMAND.md`
- optional helper script under `scripts/`

Steps:

- Run server tests first.
- Launch a fresh 10k dual-GPU Stage-2 sequential RL run with final eval skipped.
- Use an online watchdog that writes `monitor_live.json` and
  `monitor_events.jsonl`.
- Stop the run if hard failure criteria appear; otherwise let 10k finish.
- Generate HTML and JSON reports from the copied artifacts.

## Task 4: Verification And Iteration

Required checks:

- Local narrow unit tests.
- `bash -n` for launcher and server helper scripts.
- Server sequential smoke tests and BLB contract tests.
- Server 10k monitor summary.

If the 10k run stalls, collapses, or shows poor curve quality, do not patch the
server. Diagnose the artifact locally, adjust the local code or
`SERVER_COMMAND.md`, push, let the server pull, and repeat.

User-corrected interpretation: do not reject a run for occasional negative
reward spikes or isolated P1(acc) episodes. Reject it when reward rolling
averages collapse, P1 becomes frequent/consecutive, loss hits collapse
sentinels, invalid steps return, GPU reward probing fails, or search progress
stalls for a long window.
