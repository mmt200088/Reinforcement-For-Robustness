# Whole-Project Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the entire project runtime path, from launch to reports, while preserving research semantics and validation protocols.

**Architecture:** Start with project-wide observability, then optimize one flow stage at a time behind tests and parity gates. Hardware-sensitive changes require server A/B evidence before becoming defaults.

**Tech Stack:** Bash launcher, Python 3 stdlib tooling, PyTorch/Hugging Face hot paths, JSON/JSONL artifacts, unittest, ruff, git-synced server workflow.

---

## Current Execution Spine

This plan is executed as one main chain, not as isolated micro-optimizations.
Each optimization must either reduce time/memory in the active training/eval
flow, improve hardware utilization evidence, or remove overhead from required
post-run artifacts without weakening the validation protocol.

### Main Chain Order

1. **Keep runs launchable and observable.** Launcher/preset/resource snapshot
   work comes first because every server run depends on correct device flags,
   source identity, and artifact roots.
2. **Keep required artifact writes cheap.** JSON/JSONL/NPZ data that is needed
   for paper figures remains mandatory, but expensive HTML/PNG/PDF rendering
   should happen after training unless live rendering is explicitly requested.
3. **Reduce shared GPU forward overhead.** `function_handler.py`,
   `layer_importance_evaluator.py`, Paean final eval, and Stage-2 probe
   installs are shared by Stage-1, Stage-2, and final-eval paths; safe tensor
   allocation and synchronization reductions here have broad effect.
4. **Optimize pure-Python cost/replan/report loops.** Rescale/fusion-map and
   report tools stay CPU-bound; use streaming, session reuse, bounded heaps,
   and exact-cache keys rather than GPU rewrites.
5. **Promote hardware defaults only with server evidence.** Any change that
   claims better GPU utilization, worker assignment, or batch/device defaults
   needs 1GPU-vs-NGPU parity plus wall-clock evidence before becoming a
   default.

### Current Priority Queue

| Priority | Flow | Next concrete work | Verification gate |
| --- | --- | --- | --- |
| P0 | Plan and audit | Keep this plan synchronized with real commits and missing gates. | `python3 -m unittest tests.test_project_optimization_audit -v` and plan self-review. |
| P1 | Shared GPU forward | Continue removing redundant full-shape allocations, scalar syncs, and repeated install work in `function_handler.py` and `layer_importance_evaluator.py` without changing noise/reward semantics. | Source/behavior tests in `tests/test_stage1_eval_accel.py`, `tests/test_stage1_approx_reuse.py`, and server torch parity when available. |
| P1 | Stage-2/Paean evaluation | Reuse parsed action/config metadata and avoid retaining large unused action payloads while final-eval groups are scheduled. | `tests/test_paean_action_grid.py`, `tests/test_run_fusion_count_action_eval.py`, `tests/test_run_fusion_count_action_eval_rlpath.py`, and a fixed-action server smoke before default changes. |
| P1 | Structured artifacts | Keep required `rl_training_data_points/`, `episodes.jsonl`, PPO diagnostics, and manifests complete while streaming verification/report generation. | `tests/test_rl_data_points.py`, `tests/test_stage2_persistent_output_verifier.py`, and generated evidence bundle checks. |
| P2 | Rescale/fusion maps | Reuse session-loaded graph/baseline data, stream large map builds and summaries, and avoid parsing sidecars as maps. | `tests/test_rescale_optimizer_bridge_cache.py`, `tests/test_blb_fusion_count_map.py`, fusion-map report tests, and server build logs for large maps. |
| P2 | Stage-1 rollout | Add or consume timing fields for per-worker rollout, cache hit rate, forward wall, and report write wall before changing worker defaults. | Stage-1 local semantics tests plus server 1GPU/4GPU speed and deterministic-result evidence. |
| P2 | Stage-2 GPU scheduling | Leave core Stage-2 RL algorithm changes to the active Stage-2 agent; optimize only tooling/gates unless a handoff is explicit. | `stage2_ngpu_ab_compare.py`, GPU utilization reports, rollout signatures, and PPO-visible equality gates. |

### Recent Cross-Flow Runtime Commits

- `ed77481` batched installed inference metric transfers, reducing repeated
  CPU/GPU scalar transfers in installed-model evaluation.
- `edbd766` and `794be01` cached Stage-1 GELU and Stage-2 fusion action masks.
- `63b8f22`, `66ee403`, `990a89c`, `ebeddae`, and related commits deferred or
  batched scalar synchronization in Stage-1/Stage-2 PPO paths.
- `980ca8d`, `50c9907`, and `70a2655` removed unnecessary full-shape tensor
  allocations in shared GELU/LN/softmax noisy forward paths.
- `9468f58`, `51cd4a0`, `58c6666`, `9273440`, and `b0a8697` centralized JSON
  artifact read/write helpers so report/final-eval code avoids duplicated
  parsing and formatting paths.

### Standing Verification Rules

- Local tests can prove pure-Python parsing, caching, and source-level
  allocation constraints.
- Local torch-less skips are not hardware evidence; any GPU utilization claim
  requires a server run with visible GPU inventory and sampled utilization.
- Result-equivalence claims for Stage-1/Stage-2 require the validation_full
  protocol to remain unchanged and must not use train data or proxies.
- When a concurrent agent owns Stage-2 RL logic, restrict this optimization
  work to shared tooling, forward-path allocation, report/artifact paths, and
  explicit handoff-safe files.

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

Progress 2026-07-02: `scripts/project_optimization_audit.py` now classifies
artifact evidence filenames with direct deterministic rules instead of running
every file through every glob pattern. A local 200k-name parity benchmark kept
the same counts and reduced classification time from `0.622s` to `0.093s`
(`6.69x`).

- [x] **Step 4: Commit**

```bash
git add scripts/project_optimization_audit.py tests/test_project_optimization_audit.py docs/superpowers/specs/2026-07-02-whole-project-runtime-optimization-design.md docs/superpowers/plans/2026-07-02-whole-project-runtime-optimization.md
git commit -m "Add whole-project optimization plan and audit"
```

## Task 2: Launcher and Server Resource Gates

**Files:**

- Modify: `scripts/launcher_gpu_audit.py`
- Create: `scripts/server_resource_snapshot.py`
- Modify: `llama_7B_LayerImportance.sh`
- Test: `tests/test_launcher_gpu_audit.py`
- Test: `tests/test_server_resource_snapshot.py`
- Test: `tests/test_stage2_persistent_launcher.py`

- [x] **Step 1: Extend tests**

Add tests for these cases:

- Stage-1 `--stage1-rl-devices` set to fewer devices than `CUDA_VISIBLE_DEVICES`.
- Stage-2 `--stage2-rl-devices` and `--blb-v3-reward-devices` disagree.
- `RFR_GPU_AUDIT_STRICT=1` fails only when warnings exist.

- [x] **Step 2: Implement warnings only**

Keep default behavior non-fatal. Add strict failure only through
`RFR_GPU_AUDIT_STRICT=1`.

Progress 2026-07-02: added torch-free `scripts/server_resource_snapshot.py`
for pre-run GPU inventory/utilization, CPU/load, and git dirty-state evidence.
`scripts/optimization_evidence_bundle.py` now includes this snapshot in every
bundle.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now parses offline
`nvidia-smi` CSV input from a line iterator instead of `Path.read_text()`,
keeping server evidence capture memory-bounded if a sampled GPU log is supplied
instead of a tiny one-shot inventory.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now skips blank
offline `nvidia-smi` CSV rows with `isspace()` before handing lines to
`csv.reader`, instead of allocating stripped copies during row filtering. A
local 100k valid-row / 100k blank-row GPU CSV benchmark preserved the collapsed
GPU summary and reduced parsing from `2.666752s` to `2.537524s`, with traced
peak allocation dropping from `27,575B` to `23,068B`.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now detects sampled
`nvidia-smi` CSV headers and collapses repeated samples into one max-util/max
memory row per GPU. This keeps snapshot `gpu_count` tied to unique devices and
keeps evidence bundles small when they reuse a long sampled GPU log. A local
100k-sample CSV benchmark reduced parsed rows from `100001` to `4`, traced
peak memory from `28.24MB` to `0.05MB`, and serialized GPU JSON from about
`10.5MB` to `504B`.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now bounds its
best-effort `nvidia-smi` and git subprocess calls with a 5-second timeout.
This keeps pre-run resource evidence collection from stalling a server launch
when the GPU driver CLI or git status is slow, while preserving the existing
empty-output fallback behavior.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now summarizes
`git status --porcelain` output by counting newline-delimited records and
retaining only the first 20 examples, instead of materializing every dirty line
with `splitlines()`. A regression test guards against calling `splitlines()` on
the dirty-status payload. A local 200k-row synthetic dirty-status benchmark
preserved count/examples and improved the summary from `0.087380s` / `17.21MB`
to `0.002384s` / near-zero traced allocations (`36.66x`).

Progress 2026-07-02: `scripts/launcher_gpu_audit.py` now bounds fallback
`nvidia-smi` GPU discovery with a 5-second timeout. This prevents the
non-fatal GPU audit gate from stalling expensive RL launchers when the server
driver/CLI is slow or wedged and `CUDA_VISIBLE_DEVICES` was not set.

Progress 2026-07-02: `scripts/launcher_gpu_audit.py` now parses comma device
lists and fallback `nvidia-smi` output through one shared single-strip helper,
so each nonblank token is trimmed once instead of twice. A regression test
counts `strip()` calls on synthetic `nvidia-smi` output. A local 200k-line
synthetic parser benchmark preserved device IDs and improved parsing from
`0.013890s` to `0.010055s` (`1.38x`).

Progress 2026-07-02: startup/preset parsing now streams configuration files
instead of materializing full `read().splitlines()` lists. `tools/validate_preset.py`
parses preset files with a one-line lookahead iterator, and `Paean/config.py`
streams final-eval preset lines through `shlex`. Local synthetic preset
benchmarks preserved parsed arguments/flags; `validate_preset` improved from
`0.2321s` / `25.81MB` to `0.2060s` / `21.16MB`, and Paean preset parsing
reduced peak memory from `21.50MB` to `10.62MB`.

Progress 2026-07-02: `scripts/server_resource_snapshot.py`
`parse_nvidia_smi_csv()` now streams text through a small line iterator instead
of materializing `splitlines()`. This keeps resource snapshot parsing
memory-bounded when callers pass long sampled `nvidia-smi` text. A local 100k
sample benchmark preserved collapsed GPU rows while reducing traced peak memory
from `9.53MB` to `0.02MB` and wall time from `1.0655s` to `0.9515s`.

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

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now parses Stage-1
training logs from a single-pass line iterator, and
`scripts/optimization_evidence_bundle.py` streams Stage-1 log files into that
parser instead of concatenating all logs into one large in-memory string.

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now aggregates
Stage-1 rollout windows, total timing windows, cache status, and worker speedup
with running counters instead of retaining every parsed window row before
rendering the report.

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now skips regex
matching for unrelated training-log lines that do not contain
`[stage1-rollout`, and `parse_log_text()` streams text through a small line
iterator instead of materializing `splitlines()`. A local noisy 250k-line log
benchmark preserved the full summary; the file/line parser path improved from
`1.1329s` to `0.8107s` (`1.40x`), while the text-entry path reduced
`splitlines()` peak memory from `38.77MB` to `0.01MB`.

Progress 2026-07-02: `scripts/stage1_approx_reuse_benchmark.py` now summarizes
install/forward/total timing means and speedups in one helper using
`math.fsum()`, avoiding repeated `statistics.mean()` calls and intermediate
millisecond/total timing lists after a server benchmark completes. A local
200k-row timing-summary benchmark produced matching rounded results, reduced
summary time from `2.133355s` to `0.124024s` (`17.20x`), and eliminated the
`18.39MB` traced peak from list materialization.

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now checks Stage-1
worker episode-count imbalance without materializing `worker_episode_counts`
into a list and without scanning the minimum twice. For real `dict_values`
inputs it uses built-in `min/max` over the existing view; single-pass iterables
fall back to one-pass aggregation. A local 200k-worker synthetic count
benchmark preserved the warning result, reduced time from `2.868017s` to
`1.976543s` (`1.45x`), and cut traced peak allocation from `1,600,128B` to
`72B`.

Progress 2026-07-03: `_select_piecewise_gelu_output()` in `function_handler.py`
now uses scalar `0.0` for the low/NaN branch instead of allocating
`torch.zeros_like(x)`. The helper is shared by regular Stage-1
`PolynomialGELU` and Block-5 noisy GELU, so this removes one full-shape zero
tensor allocation from both installed forward paths while preserving the same
piecewise boundaries.

Progress 2026-07-03: Block-2 QK-merge, Block-2 BSGS, and Block-4 input /
softmax-V ones-mask encode hooks in `function_handler.py` now sample the
same-shape encode noise against the current tensor and `add_(1.0)` instead of
allocating `torch.ones_like(...)` before adding noise. This removes five
full-shape ones tensor allocations from shared installed forward paths while
preserving the CKKS `ones + encode-noise` mask semantics.

Progress 2026-07-03: BERT and GPT-2 approximation-softmax lower-bound masks in
`function_handler.py` now use scalar `0.0` in `torch.where(...)` instead of
allocating `torch.zeros_like(x)` before discarding below-band values. This
removes two full-shape zero tensor allocations from shared approximate-softmax
forward paths while preserving the same lower-bound zeroing semantics.

Progress 2026-07-03: single-GPU `LayerImportanceEvaluator.evaluate_model()` now
uses the same `Stage1EvalCache` helper as the worker path and keys cache entries
by the resolved split name. This avoids duplicate validation_full forwards when
callers reach the same Stage-1 plaintext configuration through equivalent
`use_train=False` / `split="validation_full"` entrypoints, while preserving the
existing install/forward semantics.

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
- Modify: `scripts/stage2_reward_probe_scaling_report.py`
- Modify only after coordination: `blb_stage2_rl/parallel_runner.py`,
  `blb_stage2_rl/probe_runner.py`, `blb_stage2_rl/sequential_runner.py`
- Test: `tests/test_stage2_ngpu_ab_compare.py`
- Test: `tests/test_gpu_utilization_report.py`
- Test: `tests/test_stage2_reward_probe_scaling_report.py`
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

Progress 2026-07-02: the same report now streams `episodes.jsonl` through a
single-pass row summarizer instead of loading every episode into memory first.
This keeps report generation and evidence bundling lightweight for 60k+ episode
Stage-2 runs while preserving the existing CLI output shape.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now summarizes
optional `nvidia-smi` CSV samples with running per-device aggregates instead of
keeping every utilization sample in memory before computing means and maxima.
This keeps GPU evidence reports memory-bounded for long server sampling logs.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now normalizes
`nvidia-smi` CSV headers once per file and uses a raw-field lookup for every
sample row, instead of regex-normalizing the header keys again for each row.
A local 200k-row sampled GPU CSV benchmark preserved the per-device summary
and reduced CSV post-processing from `4.274366s` to `2.521548s` (`1.70x`),
with traced peak allocation dropping from `62,023B` to `51,313B`.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now parses sampled
`nvidia-smi` CSV files with `csv.reader` and precomputed column indices instead
of `csv.DictReader`, avoiding one per-row dictionary allocation in long server
GPU sampling logs. A local 100k-row / 4-GPU sampled CSV benchmark preserved the
per-device utilization summary and reduced post-processing from `1.216738s` /
`0.05MiB` peak to `1.002973s` / `0.04MiB`.

Progress 2026-07-02: the same report now aggregates Stage-2 episode timing
fields (`terminal_probe_wall_seconds`, policy rollout, replan/optimizer,
per-device probe walls, and optional hot-path write/render timings) with
running count/sum/min/max stats instead of retaining every float in lists. A
local 100k-row parity benchmark kept the same stats and reduced timing-summary
peak memory from `15.28MB` to `0.02MB`.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now uses a streaming
sorted `os.walk()` fallback to locate nested `episodes.jsonl` files instead of
`Path.rglob("episodes.jsonl")`. A local 241-directory/2401-file synthetic run
tree preserved the same discovery result and reduced fallback lookup from
`0.0084s`/`0.05MB` to `0.0052s`/`0.02MB`.

Progress 2026-07-02: `_find_episodes_path()` in
`scripts/gpu_utilization_report.py` now avoids sorting each directory's
`filenames` list during fallback search because it only needs an
`"episodes.jsonl"` membership check. Directory traversal order remains stable
via `dirnames.sort()`. A local 120k-filename synthetic directory preserved the
same discovered path and reduced fallback lookup from `0.021896s` to
`0.000526s` (`41.66x`), with traced peak memory dropping from `1.37MB` to
`0.92MB`.

Progress 2026-07-02: extracted the reward-probe scaling benchmark postprocessor
from `scripts/stage2_reward_probe_scaling_benchmark.sh` into
`scripts/stage2_reward_probe_scaling_report.py`. The new report script is
unit-tested and streams `runs.jsonl`, per-run `episodes.jsonl`, and sampled
`nvidia-smi` CSV files instead of reading them into large strings.

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now
computes probe wall/speedup means with running totals while retaining the wall
list only for exact median. A local 100k-row parity benchmark reduced report
post-processing from `1.282s`/`7.21MB` to `1.074s`/`4.14MB` (`1.74x` lower
peak memory).

Progress 2026-07-02: the same reward-probe scaling report now sorts the retained
probe-wall list in place and computes the median directly instead of calling
`statistics.median()`, which copies and sorts another list. A local 80k-row
episode JSONL benchmark produced identical summaries, reduced total time from
`0.798515s` to `0.767511s` (`1.04x`), and cut traced peak memory from `3.42MB`
to `2.81MB` (`1.22x`).

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now
normalizes sampled `nvidia-smi` CSV headers once per file and uses raw-field
lookups for every sample row, instead of regex-normalizing field names on each
row. A local 200k-row sampled GPU CSV benchmark preserved max util/memory maps
and reduced GPU-sample post-processing from `3.853528s` to `2.066302s`
(`1.86x`), with traced peak allocation dropping from `60,023B` to `49,098B`.

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now parses
sampled `nvidia-smi` CSV files with `csv.reader` and precomputed column indices
instead of `csv.DictReader`, avoiding one dictionary allocation per sample row.
A local 120k-row / 4-GPU sampled CSV benchmark preserved max util/memory maps
and reduced post-processing from `1.039389s` / `0.05MB` peak to `0.910350s` /
`0.04MB` (`1.14x`).

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py`
`render_html()` now iterates the summary `runs` collection directly instead of
wrapping it with `list()` before rendering rows. This preserves HTML output and
lets future streaming callers avoid one full run-list materialization. A local
100k-row render benchmark had comparable wall time (`1.530112s` vs
`1.540720s`) while reducing traced peak memory from `29.26MB` to `28.50MB`.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` and
`scripts/stage2_reward_probe_scaling_report.py` now parse JSONL episode lines
directly instead of allocating `line.strip()` copies before `json.loads()`,
while still skipping whitespace-only lines. Local 80k-row long-line benchmarks
preserved report summaries and reduced `gpu_utilization_report` from
`3.0698s` to `2.8886s` (`1.06x`) and reward-probe scaling episode summary
from `1.3052s` to `1.2424s` (`1.05x`).

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now uses a
precompiled hot-path float regex for episode timing fields and sampled
`nvidia-smi` values, instead of dispatching through `re.search()` for every
numeric parse. A local 200k-value mixed GPU/timing benchmark preserved parsed
values and reduced `_float_value()` from `0.103639s` to `0.072524s` (`1.43x`).

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now uses
the same precompiled float regex for probe wall/speedup values, trial counts,
and sampled `nvidia-smi` utilization/memory fields. A local 200k-value mixed
scaling-report benchmark preserved parsed values and reduced `_float_value()`
from `0.108688s` to `0.072969s` (`1.49x`).

Progress 2026-07-02: `blb_stage2_rl/fusion_cost.py`
`compute_fusion_cost_saving()` now reuses the per-call fusion/truncation maxima
it already accumulates to derive the default `max_actual`, instead of rescanning
the same block choices through `max_actual_for_choices()`. A local 47-block
hot-path benchmark preserved all reward-cost outputs and reduced the full
calculation from `1.867492s` to `1.701235s` over 10k iterations (`1.10x`).

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now analyzes ordered
Stage-2 `episodes.jsonl` files with a streaming two-pass path for summary and
bounded-window rows, avoiding materializing full 60k+ episode lists during A/B
HTML generation. If an input file is out of episode order, it falls back to the
legacy load-and-sort path to preserve report semantics.

Progress 2026-07-02: the ordered Stage-2 A/B path now combines episode-order
scanning and bounded-window aggregation in the first pass, then performs only
one second pass for tail-aware summary statistics. This reduces ordered
`episodes.jsonl` reads from three passes to two while preserving the legacy
list-based summary/window output.

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now checks common
`blb_stage2_best_action_full.json` locations directly before falling back to a
recursive directory walk. A local synthetic run tree with 402 directories and
2001 files preserved the same best-action payload and reduced common-path
lookup from `0.0064s`/`0.03MB` to `0.0001s`/`0.01MB`.

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now skips blank JSONL
lines with `isspace()` and passes nonblank episode lines directly to
`json.loads()` instead of allocating a stripped copy for every row. A local
80k-row long-line benchmark preserved row count and reduced episode parse time
from `0.434637s` to `0.415472s` (`1.05x`).

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now computes each
per-window stats row with a single chunk scan and running counters instead of
building temporary value and boolean lists for every metric. A local 200k-row,
400-window benchmark preserved identical stats and reduced window aggregation
from `0.152045s` to `0.142315s` (`1.07x`), with traced peak allocation dropping
from `381,192B` to `359,168B`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` and
`scripts/run_fusion_count_action_eval_rlpath.py` now keep the first unique
fusion-count action config by reference during deduplication instead of
eagerly shallow-copying every input config via `dict(cfg)` and then copying
the unique values into a list. The returned `dict_values` view still supports
`len()` and ordered iteration for the launch loops. Local 80k-config duplicate
benchmarks preserved canonical selection; standalone Paean action eval
deduplication improved from `1.340131s` to `0.819613s` (`1.64x`, peak
`15,592B` to `6,888B`), while RL-path deduplication improved from `19.053310s`
to `18.179581s` (`1.05x`, peak `108,156B` to `50,443B`).

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
- Modify: `scripts/blb_f0_scan_feasible_domain.py`
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

Progress 2026-07-02: `ReplanSession.from_profile()` now passes the static
skeleton baselines it already loaded into `ReplanSession.__init__()`, avoiding a
second read/parse of the same `static_skeletons_<profile>.json` archive during
session construction. This reduces repeated worker/session setup overhead in
Rescale/fusion-map paths without changing replan semantics.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now filters fusion
map candidates by block-map filename before opening JSON files, so post-build
sidecars such as `map_summary.json` are not parsed as maps. This keeps fusion
map reporting focused on real `block*.json` maps and avoids unnecessary
sidecar reads after large server builds.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now discovers map
files with `os.scandir()` and filename filtering before constructing `Path`
objects, instead of `Path.glob("*.json")` followed by sidecar filtering. A
regression test patches `Path.glob()` out of `_load_maps()`. A local directory
benchmark with 3000 map files plus 3000 sidecar/non-map files preserved map
ordering and improved discovery from `0.059813s` / `3.15MB` to `0.027619s` /
`0.86MB` (`2.17x`).

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now precomputes the
static all-max baseline action, layer width, and block offsets once per
action-config report instead of recomputing them for every generated group. A
local 12-layer / 5000-group splice benchmark preserved generated actions and
reduced action splice time from `1.613343s` / `15.57MiB` peak to `0.853562s` /
`14.08MiB`.

Progress 2026-07-02: `scripts/blb_f0_scan_feasible_domain.py` now lazily
imports its torch/optimizer-heavy execution dependencies and uses
`heapq.nsmallest()` for the masked-random and multi-random best-cost summaries
instead of sorting every valid sampled row. On a 300k-row local top-20
benchmark, the summary selection stayed identical while median time dropped
from `0.280s` to `0.069s` and traced peak memory dropped from `22.76MB` to
near zero.

Progress 2026-07-02: `scripts/blb_f0_scan_feasible_domain.py` now builds the
per-slot summary rows with one pass over `per_slot_scan` rows instead of
rescanning all rows once per slot. A local 16k-row / 800-slot synthetic
benchmark preserved the summary output and reduced this step from `1.8017s` to
`0.0195s`.

Progress 2026-07-02: `scripts/blb_build_fusion_count_map.py` now consumes golden
enumeration shard results with `Pool.imap_unordered()` and merges the result
stream directly, instead of `Pool.map()` batching every shard result before
reduction. A synthetic 80-shard/120k-row local benchmark kept identical merged
counts while reducing traced peak memory from `69.60MB` to `59.74MB` and median
merge wall time from `0.718s` to `0.644s`; the real benefit is avoiding a second
parent-process copy of large golden fallback shard payloads.

Progress 2026-07-02: `blb_stage2_rl/fusion_enum.py` golden
`enumerate_shard()` now decodes only the product ranks assigned to the shard,
instead of iterating the entire Cartesian product in every worker and skipping
`num_shards - 1` out of every `num_shards` combos. The shard partition is
unchanged (`rank % num_shards == shard_idx`) and a 1,000,000-combo / 64-shard
local iterator benchmark produced identical assigned combos while reducing one
worker's pure iteration time from `0.0646s` to `0.0253s` (`2.55x`).

Progress 2026-07-02: `blb_stage2_rl/fusion_enum.py` degeneracy probes now reuse
the already-computed all-min corner result and stream random probes, avoiding one
duplicate real replan call and a temporary probe list on every over-budget
degeneracy check. A local call-count check with `num_random=5` now performs the
expected `7` evals (`baseline + corner + 5 random`) instead of the previous `8`.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now selects the best
option for a requested fusion count with targeted streaming scans instead of
building per-count candidate buckets and sorting the selected bucket. A local
250k-option benchmark preserved the selected option for exact, clamped, and
max targets; exact target selection improved from `0.5034s` / `6.17MB` to
`0.3138s` / near-zero traced allocations.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now detects
already-ordered fusion map option lists and reuses the list directly instead of
unconditionally sorting by `option_id` while building the HTML/JSON report.
Unordered inputs still fall back to the old sort semantics. A local
300k-option ordered-list benchmark preserved rows and reduced extra allocation
from `4.58MB` to near zero with comparable wall time (`0.0327s` -> `0.0325s`).

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now caches
`_choose_option()` results inside `_group_specs()` by `(graph_key, target)`.
The report still emits the same fixed-action groups, but repeated global,
one-hot, combined, and partial-block4 specs no longer rescan the same graph
options for `0`, `1`, and `max` targets. A local seven-graph / 25k-options-per
graph benchmark preserved the exact group specs and reduced `_group_specs()`
from `0.463743s` to `0.103732s` (`4.47x`).

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now decodes each
graph's fusion0/base option action indices and real slots once while building
the report payload, then reuses that baseline for every option summary. A local
8000-option / 24-slot payload benchmark preserved the exact graph payload and
reduced payload construction from `0.536779s` to `0.498236s` (`1.08x`).

Progress 2026-07-02: `scripts/blb_verify_boosted_install.py` now lazily imports
the torch/rescale install-path dependencies only after it finds a non-skipped map
with boosted fusion options, and the map loop passes the already-loaded JSON
payload into `verify_map()`. This removes one duplicate full JSON parse per
verified fusion map and lets degenerate/no-boost maps skip without loading torch.
A local 7-map synthetic JSON benchmark for the eliminated parse path preserved
the checked option count and reduced median time from `6.84s` to `3.44s`, with
traced peak memory down from `60.37MB` to `42.10MB`.

Progress 2026-07-02: `scripts/blb_verify_boosted_install.py` now discovers
fusion maps with `os.scandir()` and canonical block-map filename filtering
before opening JSON payloads. Post-build sidecars such as `map_summary.json`
and hidden summaries are no longer read or passed to the boosted-install
verifier. A local 7-map / 3000-sidecar benchmark reduced the discover+load
phase from `0.159301s` / `3.18MB` to `0.015133s` / `0.01MB` (`10.53x`, with
candidate payloads dropping from `3006` to `6` after degenerate skips).

Progress 2026-07-02: `scripts/blb_orphan_slot_audit.py` now caches parsed ASTs
for `function_handler.py` and `rescale_optimizer_bridge.py` across all block
loaders in one audit process. The static slot/cfg/t_new extraction output is
unchanged, but the audit no longer rereads and reparses the same bridge source
for every block. A real-source local benchmark over blocks 1..5 reduced the
static extraction phase from `0.471s` to `0.231s` (`2.04x`).

Progress 2026-07-02: `scripts/blb_orphan_slot_audit.py` now discovers
Rescale graph JSON files with one cached `os.scandir()` result per config
directory, then filters that sorted filename tuple for each block. This avoids
five repeated directory scans in one audit run and keeps graph discovery
independent of `Path.glob()`. A local 603-graph / 3000-sidecar benchmark
preserved graph ordering while improving discovery from `0.026675s` /
`1.95MB` to `0.014171s` / `0.41MB` (`1.88x`, `4.76x` lower peak memory).

Progress 2026-07-02: `scripts/blb_f0_scan_feasible_domain.py` now sorts each
slot's allowed action indices once while building the suggested action mask and
reuses that list for both `allowed_indices` and `allowed_values`. This removes
one duplicate filter/sort pass per slot in F0 feasible-domain scans. A local
20k-slot synthetic mask benchmark preserved the full mask payload and reduced
runtime from `9.584897s` to `8.428960s` (`1.14x`) over 80 builds, with traced
peak memory unchanged at `13.36MiB`.

Progress 2026-07-02: `scripts/blb_apply_precision_boost.py` no longer
materializes an unused pre-boost snapshot of every fusion-map option before
calling the deterministic precision-boost pass. A 300k-option synthetic
benchmark preserved the effective option count while eliminating `0.089972s`
of tuple-copy time and `25.22MB` of traced peak allocation.

Progress 2026-07-02: `Rescale_optimizer/scripts/batch_run_configs.py` and
`Rescale_optimizer/scripts/check_compress_headroom.py` now discover config JSON
files with `os.scandir()` and filename filtering instead of materializing
`Path.glob("*.json")` results. The new path preserves sorted real-file config
discovery and skips `.json` directories before optimizer work starts. A local
4000-config / 4000-sidecar / 200-json-directory benchmark reduced batch
discovery from `0.052645s` / `4.31MB` to `0.031833s` / `1.18MB`, and headroom
discovery from `0.071282s` / `5.20MB` to `0.046929s` / `2.41MB`.

Progress 2026-07-02: `Rescale_optimizer/scripts/update_noise_tables_from_csv.py`
now uses the same `os.scandir()` real-file config discovery for noise-table
maintenance runs, avoiding full `Path.glob("*.json")` materialization and
skipping `.json` directories before any file rewrite attempt. A local
4000-config / 4000-sidecar / 200-json-directory benchmark preserved real config
ordering and reduced discovery from `0.073012s` / `5.20MB` to `0.030925s` /
`1.18MB`.

Progress 2026-07-02: `Rescale_optimizer/scripts/update_noise_tables_from_csv.py`
now parses measured-noise CSV rows with `csv.reader` and precomputed header
indices instead of `csv.DictReader`, avoiding one dictionary allocation per
measured `(N, scale_bits)` row. A local 200k-row synthetic noise CSV benchmark
preserved the loaded table and reduced parsing from `0.996201s` / `0.07MB` peak
to `0.723280s` / `0.06MB` (`1.38x`).

Progress 2026-07-02: `blb_stage2_rl/fusion_count_map.py` runtime
`FusionCountMap.load()` now discovers canonical block-map JSON files with
`os.scandir()` and filename filtering before parsing payloads. Runtime map
loading no longer opens sidecars such as `map_summary.json`, and `.json`
directories are ignored before any read attempt. A local 1000-map /
3000-sidecar benchmark preserved loaded graph count and reduced map loading
from `0.208784s` / `3.57MB` to `0.068689s` / `2.54MB` (`3.04x`).

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

Progress 2026-07-02: `UnifiedFinalEvaluationModule` now caches the loaded
final-eval JSON config map per `config_path` inside one module instance. This
prevents Stage-1 and Stage-2 JSON resolution from opening/parsing the same
config file twice in a single final-eval flow while preserving config-source
semantics.

Progress 2026-07-02: `UnifiedFinalEvaluationModule` now caches the Stage-2
total-cost count-solution maps used by random final-eval groups. Repeated
Stage2Budget/Budget sampling in one final-eval run reuses the same per-noise
domain enumerations instead of rebuilding all seven maps for every sampled
config.

Progress 2026-07-02: Stage-2 Equiv random final-eval sampling now uses the
same cached exact count-solution maps instead of running iterative random
cost-matching searches for every noise type and trial. It still falls back to
the old matcher if an exact target key is unavailable.

Progress 2026-07-02: Stage2Budget/Budget random final-eval sampling now caches
the Stage-2 total-cost combination plan for a reused solution-map set. Repeated
samples no longer rebuild the same suffix reachability table or rescan cost
keys for every noise domain.

Progress 2026-07-02: Stage1Budget/Budget random final-eval sampling now caches
the feasible GELU/Softmax total-cost pair list for a reused solution-map pair
and target. Repeated Stage-1 budget samples no longer rescan the GELU cost-key
domain for every random control.

Progress 2026-07-02: final-eval random comparison generation now builds
Stage-1 GELU/Softmax solution maps lazily. Stage2Budget-only runs and other
paths that do not need Stage-1 budget/equivalence controls skip the Stage-1
cost-solution enumeration entirely.

Progress 2026-07-02: `UnifiedFinalEvaluationModule` Stage2Budget/Budget random
sampling now caches feasible Stage-2 count-combo keys by `(noise-domain index,
remaining target cost)` inside the reused combo plan. Repeated random controls
with the same target no longer rescan every cost key for every domain on every
trial. A local 5000-sample synthetic benchmark kept valid samples while reducing
key scans from `25000` to `478` and sampler wall time from `0.168s` to
`0.047s`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` now folds
duplicate/no-op fusion-count final-eval groups with a shallow top-level
candidate copy instead of `copy.deepcopy()` for every alias. The combined JSON
still carries the same nested candidate diagnostics, but repeated groups no
longer clone large result payloads. A local 1200-alias synthetic combined-report
benchmark reduced merge cost from `2.0259s` / `68.41MB` to `0.0048s` /
`0.86MB`.

Progress 2026-07-02: `Paean/final_eval_layout.py` now precomputes quantized
GELU choice costs for Stage-1 same-cost peer sampling and checks sampled index
costs before materializing a GELU vector. The accepted peer sequence is
unchanged for the same seed. A local same-process benchmark over 12-layer
sampler cases kept outputs identical and improved runtime by `1.48x` to
`2.26x`; the common all-degree-1 case reduced traced peak memory from
`0.782MB` to `0.024MB`.

Progress 2026-07-02: the same sampler now short-circuits unique min/max GELU
cost extremes, where the selected vector is the only possible same-cost vector
and peers must be empty. This avoids burning `max_attempts` on impossible
final-eval controls. Same-process benchmarks kept outputs identical and reduced
all-degree-4 from `0.2044s` to `0.00014s` and all-degree-0 from `0.1829s` to
`0.00011s`.

Progress 2026-07-02: `Paean/run_final_eval.py --list-presets` now reads only
the first line of each preset with `readline()` instead of materializing the
whole preset through `read_text().splitlines()[0]`. A local 200k-line synthetic
preset benchmark preserved the displayed summary while reducing the lookup
from `0.0400s` / `14.92MB` to `0.0002s` / `0.02MB`.

Progress 2026-07-02: `Paean/config.py` now lists final-eval preset names with
a single `os.scandir()` pass instead of `Path.glob("*.conf")` plus per-entry
`Path` objects. The CLI behavior is unchanged, but startup/list-presets paths
avoid extra filesystem wrapper allocation. A local 4000-preset /
4000-nonpreset benchmark preserved sorted preset names while improving
discovery from `0.058477s` / `2.48MB` to `0.010787s` / `0.29MB` (`5.42x`,
`8.41x` lower peak memory).

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` now drops the
full parsed action-config payload after deriving the fields it actually uses
(`name`, `path`, `group`, and `action_hash`). Final-eval launch/report behavior
is unchanged, but large `action_vec`/`slots` diagnostics no longer stay resident
for the whole combined-report flow. A local 120-config synthetic benchmark kept
the same action hashes and reduced retained loader memory from `16.90MB` to
`0.17MB` (`101.86x`), with comparable wall time (`0.3065s` -> `0.3008s`).

Progress 2026-07-02: `scripts/run_fusion_count_action_eval_rlpath.py` now keeps
its action-config records dependency-light and payload-light: module import no
longer pulls torch/HF/RL dependencies, and `_load_action_configs()` retains only
the group metadata, config path/name, and `baseline_k_index` needed by
deduplication and `_run_group()`. A local 120-config synthetic benchmark with
large unused payload fields preserved config count while reducing retained
current memory from `254.69MB` to `0.65MB` and traced peak memory from
`255.42MB` to `5.60MB`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` and
`scripts/run_fusion_count_action_eval_rlpath.py` now discover action config
JSON files with `os.scandir()` and pre-filter `_`/`._` sidecars before building
`Path` objects, instead of using `Path.glob("*.json")` and then filtering.
Regression tests patch `Path.glob()` out of both loaders. A local directory
benchmark with 3000 valid configs plus 2000 sidecar/non-JSON files preserved
config ordering and improved discovery from `0.036433s` / `2.17MB` to
`0.020808s` / `0.86MB` (`1.75x`).

Progress 2026-07-02: `scripts/run_fusion_count_action_eval_rlpath.py` now
computes each action config's canonical group key once at load time and reuses
it during deduplication, result indexing, and requested-group result backfill.
This removes repeated JSON key serialization from the RL-path fusion-count
comparison driver. A local 80k-config duplicate benchmark covering those three
keyed phases preserved canonical selection and reduced runtime from
`25.343264s` to `1.572615s` (`16.12x`), with traced peak allocation dropping
from `0.78MiB` to `0.70MiB`.

Progress 2026-07-02: `Paean/action_grid.py` now precomputes the truncation
`K_LEVELS` value-to-action-index map and the invalid-value choices string at
module import. Action-range and slot decoding no longer allocates
`list(K_LEVELS)` for every K value conversion. A local same-signature 1M-call
benchmark preserved indexes and reduced K lookup time from `0.293003s` to
`0.213070s` (`1.38x`), with traced peak allocation dropping from `180B` to
`156B` over 10k calls.

Progress 2026-07-02: `Paean/action_grid.py` now caches non-K scaling-factor
value-to-action-index tables by `(kind, max_sf)`. Repeated action-range and
slot decoding no longer rebuilds the same SF choices list or scans it twice for
membership plus index lookup. A local same-signature 1M-call benchmark
preserved indexes and reduced lookup time from `1.802358s` to `0.447846s`
(`4.02x`), with traced peak allocation dropping from `404B` to `124B` over
10k calls.

Progress 2026-07-02: `Paean/action_grid.py` now caches selector-to-slot
resolution by `(num_layers, selector)` during action-grid expansion. Repeated
final-eval action-range candidates no longer rescan every layer's action-space
fields for the same selector before setting values. A local 200k-call repeated
selector benchmark preserved vector writes and reduced setter time from
`4.215663s` to `1.279825s` (`3.29x`), with traced peak allocation dropping
from `1,576B` to `124B` over 10k calls.

- [ ] **Step 3: Verify**

Run final-eval unit tests locally and a server repeated final-eval smoke for
the same fixed action before/after.

## Task 7: Structured Data and Report Decoupling

**Files:**

- Modify: `rl_data_points.py`
- Modify: `scripts/verify_stage2_persistent_outputs.py`
- Create: `scripts/optimization_evidence_bundle.py`
- Modify report generators under `reports/` or `tools/`
- Test: `tests/test_rl_data_points.py`
- Test: `tests/test_stage2_persistent_output_verifier.py`
- Test: `tests/test_optimization_evidence_bundle.py`

- [x] **Step 1: Protect data completeness**

Add tests that fail if required structured fields are dropped from Stage-1 or
Stage-2 mirrored data.

Progress 2026-07-02: strengthened
`scripts/verify_stage2_persistent_outputs.py` so Stage-2 persistent verification
fails when `episodes.jsonl` or `ppo_updates.jsonl` drops required reward,
metric, cost, action-summary, PPO, or timing fields. This did not modify the
dirty `rl_data_points.py` worktree file.

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now reads JSONL
diagnostics line by line instead of building a full `read_text().splitlines()`
string first, and it treats a missing/empty `nvidia_log` path as no GPU samples
instead of trying to open the current directory. This keeps long-run monitoring
memory-bounded and avoids a default-path crash during post-run summaries.

Progress 2026-07-02: the same monitor now skips blank JSONL lines with
`isspace()` instead of allocating `strip()` copies while loading episode and PPO
diagnostics. A local 80k-row long-line benchmark preserved parsed row count and
reduced `_read_jsonl()` from `0.497555s` to `0.475890s` (`1.05x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now writes
`reward_windows.csv` with one-pass rolling accumulators and monotonic queues
for mean/min/max, instead of repeatedly slicing and sorting each prefix window.
This reduces final monitor CSV rendering CPU work for 10k/60k episode runs.
Local 20k-row mixed-reward microbenchmark produced identical CSV output and
reduced wall time from `9.580s` to `0.186s` (`51.50x`).

Progress 2026-07-02: `_window()` in `scripts/stage2_first10k_monitor.py` now
reuses the sorted tail for min/max bounds and uses `math.fsum()` for the mean,
avoiding separate `statistics.mean()`, `min()`, and `max()` scans while
preserving the summary values. A local 1000-point window benchmark over 3000
calls produced identical output and reduced time from `3.776341s` to
`0.181783s` (`20.77x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now computes
full-run and recent PPO means in `build_summary()` with `math.fsum()/len`
instead of Python's slower `statistics.mean()` exact-rational path. A local
60k-value benchmark over 400 repeats produced the same mean and reduced time
from `27.821296s` to `0.499612s` (`55.69x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now aggregates
nvidia-smi GPU samples while reading the CSV instead of retaining every parsed
row and grouping in a second pass. A local 30k-row CSV comparison produced
identical GPU summaries and reduced traced peak memory from `9.15MB` to
`1.08MB` (`8.48x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` final mode now loads
episode and PPO JSONL rows once and reuses those rows for the summary,
`reward_windows.csv`, and `episode_health_windows.csv`. This avoids rereading
large Stage-2 episode logs during final monitor report generation.

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now streams
`reward_windows.csv` rows directly to `csv.DictWriter.writerow()` after each
rolling-window update instead of buffering the full row list before
`writerows()`. A local 100k-row parity benchmark kept the CSV output identical
and reduced write peak memory from `141.65MB` to `0.25MB` while slightly
improving wall time from `2.460s` to `2.335s`.

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now parses sampled
`nvidia-smi` CSV logs with `csv.reader` and header indices instead of
`csv.DictReader`, avoiding one per-row dictionary allocation in final monitor
summaries. A local 120k-row / 4-GPU CSV benchmark preserved the GPU summary and
reduced `_gpu_stats()` from `0.562882s` / `4.22MB` to `0.333455s` / `4.24MB`
(`1.69x`).

Progress 2026-07-02: `scripts/verify_stage2_persistent_outputs.py` now counts
Stage-2 detail batch files without materializing and sorting the full file list.
The verifier only needs the count for its gate output, so long runs with many
detail batches avoid unnecessary path-list allocation. A local 8000-detail-file
benchmark preserved the count and reduced detail discovery from `0.1079s` /
`3.51MB` to `0.0805s` / `0.56MB`.

Progress 2026-07-02: `scripts/verify_stage2_persistent_outputs.py` now skips
blank `episodes.jsonl` and `ppo_updates.jsonl` rows with `isspace()` instead of
allocating a stripped copy of every diagnostics line while checking required
fields. A local 80k-row long-line benchmark preserved row/failure counts and
reduced required-field scanning from `0.658321s` to `0.642286s` (`1.02x`).

Progress 2026-07-02: `scripts/verify_stage2_persistent_outputs.py` now checks
required JSONL fields with a complete-row fast path that returns `None` and only
constructs the missing-field tuple for rows that actually fail the schema gate.
This keeps the persistent-output verifier's hot loop allocation-free on healthy
60k+ episode logs. A local 200k complete-row field-check benchmark reduced the
field-scan loop from `0.173905s` to `0.118869s` (`1.46x`); full JSONL parse is
still dominated by `json.loads`, so end-to-end 100k-row verification improved
modestly from `1.253700s` to `1.236978s`.

Progress 2026-07-02: `tools/paper_figures.py` now lazily loads run artifacts
based on the requested `--figs`. For example, `--figs cost_vs_accuracy` no
longer reads `episodes.jsonl`, `ppo_updates.jsonl`, best-action JSON, baseline
JSON, first-invalid counts, or action histograms before rendering the
top-candidate scatter. A local 100k-episode synthetic run comparison reduced
run loading for that path from `0.464s`/`39.39MB` to effectively `0.000s`/
`0.00MB`.

Progress 2026-07-02: `tools/paper_figures.py` no longer rereads each
`top_candidates.jsonl` only to decide whether the cost-vs-accuracy scatter
needs a legend; it reuses a loop-local `has_points` flag. A local 100k-row
top-candidate benchmark reduced that figure's candidate-read path from
`0.9476s`/`78.58MB` to `0.4444s`/`39.28MB`.

Progress 2026-07-02: `tools/paper_figures.py` now projects JSONL rows to the
fields each figure needs while loading run data: `episodes.jsonl` keeps only
`total_reward`, `ppo_updates.jsonl` keeps the four plotted PPO metrics, and
`top_candidates.jsonl` keeps only `total_bits`/`total_reward` for the scatter.
A local 100k-row synthetic episode log with 20 unused debug fields preserved
row count and reduced traced peak memory from `368.41MB` to `25.19MB`.

Progress 2026-07-02: `tools/paper_figures.py` now streams
`top_candidates.jsonl` directly into the cost-vs-accuracy scatter's `xs`/`ys`
lists instead of first building a list of projected row dictionaries and then
splitting it into columns. A local 120k-row top-candidate benchmark preserved
the plotted points and reduced this read path from `0.8249s` / `39.78MB` to
`0.6445s` / `7.81MB`.

Progress 2026-07-02: `tools/paper_figures.py` now skips whitespace-only JSONL
lines with `isspace()` before calling `json.loads()` in both projected-row and
XY readers. Nonblank lines are still passed through unstripped, preserving the
existing low-copy parse path. A local 100k valid-row / 50k blank-row episode
benchmark preserved projected rows and reduced read time from `0.716423s` to
`0.468636s` (`1.53x`).

Progress 2026-07-02: `blb_stage2_rl/candidate_store.py` now skips blank
candidate JSONL rows with `isspace()` and passes nonblank rows directly to
`json.loads()` instead of allocating stripped copies while loading append-only
candidate stores. A local 100k-row synthetic `top_candidates.jsonl` benchmark
preserved record count and reduced `read_all()` from `0.720805s` to
`0.691242s` (`1.04x`).

Progress 2026-07-02: `tools/aggregate_seeds.py` now finds the latest Paean
`blb_action_final_eval_results_*.json` by streaming `os.walk()` and retaining
only the newest path, instead of recursive `glob` plus sorting a materialized
candidate list. A local 6000-candidate synthetic final-eval tree kept the same
latest result and reduced lookup peak memory from `1.28MB` to `0.02MB`, with
wall time improving from `0.0838s` to `0.0613s`.

Progress 2026-07-02: `tools/aggregate_seeds.py` now builds one filtered
legacy persistent-dir index for the seed list instead of running a persistent
tree glob once per seed. A local synthetic tree with 5000 persistent dirs and
100 queried seed tags returned the same 100 matches while reducing directory
discovery wall time from `4.5251s` to `0.0379s` and traced peak memory from
`0.11MB` to `0.05MB`.

Progress 2026-07-02: `tools/aggregate_seeds.py` now scans
`diagnostics_summary.md` for the last-50 invalid-rate line with bounded 1MiB
chunks instead of reading the whole Markdown file into memory. A local
200k-line synthetic summary preserved the parsed rate and improved the lookup
from `0.0015s` / `8.40MB` to `0.0011s` / `4.01MB`.

Progress 2026-07-02: `tools/aggregate_seeds.py` now parses the multi-seed
seed-list with `line.split()` directly instead of allocating a stripped copy of
every line before splitting. A regression test guards that the parser does not
call `strip()` per line. A local 200k-line synthetic seed-list benchmark
preserved parsed `(seed, run_tag)` rows and improved parsing from `0.284336s`
to `0.239681s` (`1.19x`).

Progress 2026-07-02: `tools/aggregate_seeds.py` now formats aggregate
`mean ± std` values with `math.fsum()` plus a direct sample-standard-deviation
calculation instead of `statistics.mean()` / `statistics.stdev()`. A local
200k-value benchmark produced identical formatted output and reduced formatting
time from `0.909304s` to `0.036743s` (`24.75x`).

Progress 2026-07-02: `tools/experiments_log.py` now bounds both best-effort
git provenance subprocesses in `_git_info()` with a 5-second timeout. This
prevents run registration/index rebuild from hanging indefinitely on a slow or
wedged git command while preserving the existing fallback behavior.

Progress 2026-07-02: `tools/experiments_log.py` now streams
`experiments/registry.jsonl` rows through `_iter_records()` for query and index
rebuild instead of first materializing every append-only record in a list.
A local 100k-row registry with 1000 unique run IDs preserved the query result
and reduced peak memory from `98.00MB` to `1.08MB`, with wall time improving
from `0.7549s` to `0.6560s`.

Progress 2026-07-02: `_iter_records()` in `tools/experiments_log.py` now skips
blank registry JSONL rows with `isspace()` and passes nonblank rows directly to
`json.loads()` instead of allocating stripped line copies. A local 100k-row
registry benchmark preserved row count and reduced raw registry iteration from
`0.688479s` to `0.666293s` (`1.03x`).

Progress 2026-07-02: `tools/experiments_log.py` now computes the
best-by-dataset table for `experiments/index.md` with one streaming max pass
over latest records, instead of building per-dataset buckets and sorting each
bucket. A local 250k-record synthetic registry preserved the selected run IDs
while reducing that index substep from `0.132s`/`1.44MB` to `0.113s`/`0.01MB`.

Progress 2026-07-02: `tools/experiments_log.py` query now applies filters
before ordering and uses a bounded heap for `--last-n`, instead of sorting
every latest run record before filtering. A local 120k-record registry
preserved `last_n=20` results while reducing query wall time from `0.9792s` to
`0.8729s`; peak memory is still dominated by latest-run de-duplication.

Progress 2026-07-02: `tools/experiments_log.py` now delays materializing
registry rows while selecting the latest record per `run_id` and best record
per dataset. Overwritten records are kept as mapping references during the scan
and only final winners are copied. A local 100k-row synthetic registry kept the
same selected run IDs while improving latest-run de-duplication from `0.1253s`
to `0.0454s` (`2.76x`) and best-by-dataset selection from `0.1291s` to
`0.0613s` (`2.11x`).

Progress 2026-07-02: `tools/experiments_log.py query` now keeps latest
registry records as mapping references through filtering and materializes dicts
only for rows returned to the caller. A local 100k-record synthetic registry
with 1k matching `mrpc` rows preserved query output while reducing filtered
query time from `0.056000s` to `0.030052s` (`1.86x`) and traced peak
allocation from `42.06MB` to `7.88MB`.

Progress 2026-07-02: `tools/experiments_log.py rebuild` now prepares and sorts
latest registry records as mapping references instead of copying every latest
record into a fresh dict before rendering `experiments/index.md`. A local
100k-record synthetic index-prep benchmark preserved selected rows while
reducing prep/sort time from `0.092343s` to `0.050433s` (`1.83x`) and traced
peak allocation from `42.04MB` to `7.86MB`.

Progress 2026-07-02: `tools/experiments_log.py rebuild` now writes
`experiments/index.md` incrementally instead of accumulating every Markdown row
in a list and joining the full document before one final write. A local
50k-record synthetic rebuild produced byte-identical index output with
comparable wall time (`0.472218s` to `0.473366s`) while reducing traced peak
allocation from `166.66MB` to `131.88MB`.

Progress 2026-07-02: `tools/experiments_log.py rebuild` now computes status
counts, dataset counts, and best-by-dataset rows in one pass over the sorted
latest registry records, instead of scanning the same latest list again through
`_best_by_dataset()`. A local 120k-row synthetic rebuild preserved index output
and reduced the aggregation/render-prep path from `2.113363s` to `2.021970s`
(`1.05x`).

Progress 2026-07-02: `_git_info()` in `tools/experiments_log.py` now treats
raw `git status --porcelain` emptiness as the dirty-state signal instead of
calling `strip()` on the whole status payload. Git porcelain output is empty
for clean trees and non-empty for dirty trees, so run registration keeps the
same semantics while avoiding a full dirty-status copy. A local 200k-row
synthetic dirty-status benchmark preserved the result and improved the check
from `0.010460s` / `11.89MB` to `0.000009s` / `520B` traced peak allocation.

Progress 2026-07-02: `scripts/blb_export_action_registry.py` now lazily imports
`blb_stage2_rl.action_space` only when registry generation actually needs it,
so dependency-light imports and help/static tooling no longer pull the torch
stack. `build_registry_payload()` also computes `per_layer_field_offsets()`
once and reuses the result for expected slot count, block counts, and summary
slot count. A synthetic 120k-offset benchmark preserved the registry summary
and reduced the offset-counting path from `0.447s` to `0.226s` (`1.97x`).

- [ ] **Step 2: Move expensive rendering out of hot paths**

Keep JSON/JSONL writes in training; move PNG/HTML/NPZ rendering to post-run
commands unless the user explicitly requests live rendering.

Progress 2026-07-02: `blb_stage2_rl/persistence.py` now supports
`render_plots=False` and the `RFR_STAGE2_RENDER_PLOTS=0` environment switch
for Stage-2 curve generation. Training can keep writing the required NPZ data
while skipping synchronous PNG/PDF matplotlib rendering; the offline
`scripts/blb_regen_stage2_outputs.py` path forces `render_plots=True` so
post-run reports can regenerate the inspection artifacts later. A local
5000-episode benchmark reduced `write_training_curves()` from `1.999s` /
`44.88MB` with PNG/PDF rendering to `0.002s` / `0.30MB` when plot rendering is
disabled.

Progress 2026-07-02: the Stage-2 curve persistence path now checks sequence
length with `len()` where available and converts each NPZ series with one
`list()` pass, instead of using `len(list(seq))` before converting the same
series again. A regression test now enforces that `render_plots=False` iterates
each supplied NPZ sequence at most once. A warmed 60k-episode local parity
benchmark kept the NPZ byte size identical (`3374656` bytes) with comparable
median write time (`0.0123s` -> `0.0116s`), while removing avoidable temporary
copies from the training hot path.

Progress 2026-07-02: Stage-2 live trace CSV schema migration now streams old
rows directly into the migrated file instead of materializing all rows in a
list and calling `writerows()`. A regression test patches `writerows()` out of
the migration path, and a local 50k-row old-schema trace benchmark preserved
the migrated row count while reducing migration peak memory from `137.08MB` to
`0.19MB` and wall time from `0.980s` to `0.877s`.

Progress 2026-07-02: `append_blb_episode_trace_row()` now caches trace CSV
paths whose schema has already been confirmed or freshly written in the current
process, so repeated PPO-update trace appends skip the redundant header
inspection. A regression test verifies the second append does not call schema
migration again after the first append created a current-schema trace. A local
3000-row append benchmark preserved CSV byte size while reducing append wall
time from `0.5079s` to `0.3204s`.

Progress 2026-07-02: `scripts/blb_regen_stage2_outputs.py` now lazily
materializes ADR-014 optional diagnostic series while reading `episodes.jsonl`.
Older runs that do not contain those fields no longer allocate nine extra
all-zero lists before plotting. A local 100k-row legacy-log benchmark preserved
the base reward/loss/metric series and reduced parsing from `1.068s` /
`25.97MB` to `0.973s` / `19.10MB`.

Progress 2026-07-02: the same offline regenerator now skips blank
`episodes.jsonl` and `ppo_updates.jsonl` lines with `isspace()` and passes
nonblank rows directly to `json.loads()` instead of allocating stripped copies.
Local 80k-row long-line benchmarks preserved parsed row counts and reduced
episode parsing from `0.776086s` to `0.755639s` (`1.03x`) and entropy parsing
from `0.300351s` to `0.283807s` (`1.06x`).

Progress 2026-07-02: `scripts/blb_regen_stage2_outputs.py` now parses
baseline reference values from `blb_stage2_report.md` and
`diagnostics_summary.md` by scanning lines and stopping once the needed values
are found, instead of reading the whole Markdown files. A local synthetic
report with the baseline table followed by 200k tail lines preserved parsed
baselines and reduced parsing from `0.0006s` / `4.20MB` to `0.0001s` /
`0.02MB`.

Progress 2026-07-03: `rl_local_optimum.py` now materializes episode returns,
optional entropy/best-score series, and collapse-attribution fusion/margin
series at most once before computing local-optimum and HOT/COLD reports. This
keeps Stage-1/Stage-2 shared health-report generation compatible with
one-shot iterators from offline regenerators and removes repeated full-list
copies from long-run report paths.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_rl_data_points tests.test_stage2_persistent_output_verifier -v
```

## Task 8: Server Evidence and Promotion Loop

**Files:**

- Modify: `SERVER_COMMAND.md`
- Use: `scripts/project_optimization_audit.py`
- Use: `scripts/server_resource_snapshot.py`
- Use: `scripts/gpu_utilization_report.py`
- Use: `scripts/stage1_parallel_report.py`
- Use: `scripts/optimization_evidence_bundle.py`
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

Progress 2026-07-02: added torch-free
`scripts/optimization_evidence_bundle.py`, which packages project audit,
Stage-1 parallel timing, Stage-2 GPU utilization, and Stage-2 persistent
verification outputs into one manifest/index, with optional `--tar-gz` archive
output. This reduces server post-run manual stitching before evidence
promotion. The project audit artifact scan now walks each artifact root once
instead of once per pattern, reducing post-run evidence packaging overhead on
large `experiments/` trees. Stage-1 logs and Stage-2 episode diagnostics are
now streamed by the evidence tools, reducing peak memory during long-run bundle
generation. Actual server A/B artifact pullback is still pending.

Progress 2026-07-02: `scripts/optimization_evidence_bundle.py` now writes
optional tar.gz archives with a streaming `os.walk` traversal instead of
materializing and sorting every path under the bundle directory. A local 2k-file
archive comparison produced identical entries and reduced traced peak memory
from `2.76MB` to `1.70MB` (`1.63x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now hashes the
canonical Rescale_optimizer source/config subset by streaming file chunks
instead of calling `Path.read_bytes()` for every `.py`/`.json` file. This keeps
Trust-0 manifest generation memory-bounded for larger optimizer/config trees
while preserving the manifest hash format.

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now also streams the
canonical Rescale_optimizer file traversal with a heap-ordered directory walk
instead of collecting all `.py`/`.json` paths before hashing. A local 2k-file
Rescale tree comparison produced the same canonical hash and reduced traced
peak memory from `2.05MB` to `1.08MB` (`1.90x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now uses the same
heap-ordered traversal for generic directory hashes, replacing
`sorted(path.rglob("*"))` in `_dir_sha256()`. A local 2k-file directory
comparison preserved the full-tree hash, kept cache-directory skip semantics,
and reduced traced peak memory from `2.21MB` to `1.10MB` (`2.02x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now bounds all
best-effort git subprocess calls in `_run_git()` with a 5-second timeout. This
keeps Trust-0 manifest generation from stalling a server command indefinitely
if git status/diff/upstream resolution hangs, while preserving the existing
`None` fallback on failure.

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now reuses the
already-cleaned `git status --short --branch` string for manifest dirty-state
detection instead of calling `strip()` a second time in `build_manifest()`.
This preserves `status_short` and dirty semantics while avoiding another full
status edge scan in evidence packaging. A local 200k-row synthetic status
benchmark preserved the dirty result and improved the second-check path from
`0.078152s` to `0.060709s` over 200k repeats (`1.29x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now builds
`per_layer_field_offsets()` once per manifest and reuses that tuple for both
per-block slot counts and total per-layer slot count. A local 120k-offset
synthetic benchmark preserved the manifest-derived counts, improved best time
from `0.349566s` to `0.222708s` (`1.57x`), and cut traced peak memory from
`30.30MB` to `15.18MB` (`2.00x`).

Progress 2026-07-02: `_dir_sha256()` in `scripts/blb_make_run_manifest.py` now
prunes skip directories such as `.git`, `__pycache__`, `.pytest_cache`, and
`.mypy_cache` before calling `iterdir()` on them, while preserving the existing
full-tree hash for included files. A local synthetic tree with 30 kept files and
3600 skipped `.git` files preserved the digest and reduced hash time from
`0.3584s` to `0.0105s`.

Progress 2026-07-02: `scripts/blb_phase0_preflight.py` now scans source/config
files line by line when building `blb_entrypoints_grep.txt` instead of
materializing each file with `read_text().splitlines()`. A local 100k-line
source-file benchmark preserved the same 10 entrypoint matches and reduced
traced peak memory from `8.36MB` to `0.02MB`.

Progress 2026-07-02: `scripts/blb_phase0_preflight.py` now reuses the same
repository walk to generate `repo_file_list.txt`, `repo_code_config_files.txt`,
and `blb_entrypoints_grep.txt`, instead of walking the tree once for file lists
and again for grep. A local synthetic 2400-file repo preserved all three report
outputs and reduced Phase-0 report generation from `0.1505s` / `0.78MB` to
`0.1234s` / `0.30MB`.

Progress 2026-07-02: `scripts/blb_verify_noise_install.py` now caches the
torch-free noise-variance table extracted from `function_handler.py` after the
first AST parse in a process. Repeated verifier/test calls no longer reread and
reparse the same `_NOISE_STD_RAW` literal while the variance lookup table shape
and values remain unchanged. A real-source local benchmark over 80 repeated
loads reduced table extraction from `3.228s` to `0.040s` (`80.43x`).

Progress 2026-07-02: `scripts/project_optimization_audit.py` now walks
artifact roots with a streaming sorted `os.walk()` iterator instead of
`Path.rglob("*")`. A local synthetic 3400-file artifact tree preserved the same
evidence counts and reduced scan cost from `0.1523s`/`1.89MB` to
`0.1270s`/`0.58MB`.

Progress 2026-07-02: `_iter_files()` in `scripts/project_optimization_audit.py`
now keeps deterministic directory traversal with `dirnames.sort()` but avoids
sorting each directory's `filenames` list before artifact classification. A
local 120k-filename synthetic artifact directory preserved evidence counts,
reduced scan time from `2.215341s` to `2.059308s` (`1.08x`), and cut traced
peak memory from `8.43MB` to `0.92MB`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval_rlpath.py` now
uses copy-on-convert behavior in `_jsonable()`, so already JSON-native
RL-path report trees such as `step_records`, `fusion_action_steps`, and
terminal diagnostics are reused instead of being recursively cloned before
`json.dumps()`. A local 800-group synthetic RL-path report benchmark preserved
the converted payload semantics and reduced conversion from `2.391359s` /
`37.62MiB` peak to `0.485470s` / near-zero traced allocation.

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
