# Decouple Stage-1 / Stage-2 RL + new persistence layout

- **Date:** 2026-05-30
- **Status:** Approved 2026-06-01 (grilled); **core IMPLEMENTED locally 2026-06-01**
  (py_compile + `bash -n` + torch-free tests green; server behavioral verification
  pending). Remaining: light tool discovery + registry `record_dir` (§12.5), and the
  secondary docs (ARCHITECTURE §4 / GLOBALS). New module `config/run_layout.py`
  (+ `tests/test_run_layout.py`).
- **Scope:** Canonical Stage-1 RL (GELU/Softmax degree search) and canonical
  Stage-2 RL (blb_v3 BLB). NOT GA / greedy / general-rl / compare / legacy
  `noise_rl_module_v2` Stage-2.

## 1. Problem / motivation

Today Stage-1 and Stage-2 are **chained** inside one launcher invocation and
**share one persistent run dir**:
`Parting Chapter/persistent/{algorithm}/{model}/{dataset}/{constraint_slug}/`
with `stage1/` + `stage2_noise|blb_stage2/` side by side. The chaining +
per-constraint-slug subdirs make "fresh start" semantics tangled (fresh-stage1
vs fresh-stage2 vs fresh-start vs whole-run resume all interact), and a Stage-2
re-run is entangled with Stage-1 state.

Goal: **fully decouple** the two stages into independent runs with independent,
clean fresh/resume lifecycles, and give each stage a **single persistence dir
per fine-tuned-model+dataset** plus a **`record/` archive** of completed runs.

## 2. Locked decisions (from brainstorming)

1. **Stage-2 ← Stage-1 input:** Stage-2 reads the Stage-1 best config from
   `stage1/record/` (highest run-number for the combo by default, or
   `--stage1-run-id "<run-id>"`), overridable by `--stage2-fixed-config JSON`.
   If no record and no JSON → hard error (run Stage-1 first).
2. **Run lifecycle:** resume by default; `--fresh` starts a new run; the record
   entry is written **on completion**, numbered `N = (existing record entries
   for combo) + 1`.
3. **Scope:** RL Stage-1 + Stage-2 only. Old `persistent/`, `runs/`, GA/greedy/
   general/compare, and legacy v2 Stage-2 are untouched. **No migration** of
   existing data.
4. **Naming:** spaces, exactly as specified. Combo folder
   `{model_type with '-'→' '} {dataset}` → `bert base mrpc`. Run-id
   `{combo} {N} {YYYYMMDD}` → `bert base rte 1 20260530`.
5. **Chained `train` mode:** removed. Stages are always separate invocations;
   the old chained mode errors with guidance.
6. **Metric curve:** the record gets reward + entropy + metric curves; the
   metric curve is derived from logged per-episode metric1 when no plot exists.

## 3. New directory layout

```
Parting Chapter/                          ← RL output root (unchanged name)
├── stage1/
│   ├── bert base mrpc/                    ← single persistence/working dir per combo
│   ├── bert base sst2/    bert base rte/
│   ├── bert large mrpc/   bert large sst2/  bert large rte/
│   └── record/
│       ├── bert base rte 1 20260530/      ← archived completed run #1
│       ├── bert base rte 2 20260531/
│       └── bert base mrpc 1 20260530/
└── stage2/                               ← identical shape, mirrors stage1
    ├── bert base mrpc/ …
    └── record/ …
```

- The persistence/working dir `stage{1,2}/{combo}/` keeps each stage's existing
  internal product layout (checkpoints, status JSON, live curves, logs,
  metadata.json). It is the resume target.
- `record/{run-id}/` is a **read-only snapshot** created on completion.
- Coexists with the untouched legacy `Parting Chapter/persistent/`, `runs/`, etc.

### Constraint tolerances leave the path

`s1t/s2t/s2st` are no longer in the directory name (one dir per combo). They are
written into the run's `metadata.json` and into the record. **Guard:** on a
resume (no `--fresh`), if the persisted `metadata.json` constraints differ from
the current invocation, abort with a clear message telling the user to pass
`--fresh` (so a different-constraint run is never silently resumed).

## 4. Run lifecycle

State of `stage{1,2}/{combo}/` is one of: *empty*, *in-progress* (checkpoint
present, no `COMPLETED` marker), *completed* (`COMPLETED` marker present).

| Invocation | empty | in-progress | completed |
|---|---|---|---|
| (no flag) | start new run | **resume** | error: "run already completed; use --fresh" |
| `--fresh` | start new run | wipe + start new run | wipe + start new run |

- A run writes a `COMPLETED` marker (and the record snapshot) only after training
  **and** its completion snapshot (config + curves + basic single-eval metric)
  finish. The heavy same-cost final-eval is a separate standalone tool (see the
  decoupled-final-eval spec) and is not part of completion.
- `next_run_number(stage, combo)` scans `record/` for entries whose name starts
  with `"{combo} "` and parses the integer token immediately before the date;
  returns max+1 (1 if none).
- Graceful-stop (SIGINT / `STOP_RL`) leaves the run *in-progress* (no record, no
  COMPLETED marker) so the next plain invocation resumes — unchanged behavior,
  now per-stage.

## 5. Decoupling + Stage-2 ← Stage-1 wiring

- `--mode stage1-only` → `run_output_dir = Parting Chapter/stage1/{combo}`.
- `--mode stage2-only` → `run_output_dir = Parting Chapter/stage2/{combo}`.
- The chained `train`/`eval` mode is removed from the launcher (hard error with
  guidance). `rl_tune.py` no longer runs Stage-1→Stage-2 in one process.
- **Stage-2 input resolution** (in `blb_stage2_rl` baseline bootstrap path):
  1. If `--stage2-fixed-config` given → use it (current behavior).
  2. Else load `stage1/record/{combo} <N> *` (max N, or `--stage1-run-id`) →
     `final_config.json` → extract `gelu_degree_per_layer` /
     `softmax_degree_per_layer`.
  3. Else hard error.
- **Per-stage completion snapshot** (REVISED 2026-05-30 — final-eval is now a
  standalone tool; see `2026-05-30-decoupled-standalone-final-eval-design.md`):
  on completion each stage snapshots **config + curves + a basic single-eval
  metric** into `record/{run-id}/`. It does NOT auto-run the heavy same-cost
  51-group final-eval or the GLUE submission; those belong to the standalone
  final-eval tool.

## 6. Record contents

Each `record/{run-id}/`:

- `final_config.json` — Stage-1: best `gelu_degree_per_layer` /
  `softmax_degree_per_layer` (+ per-layer cost). Stage-2: best
  `blb_v3_best_action_vec` + slot view (`action_io` slots) + decoded SF/K,
  **plus the prerequisite `gelu_degree_per_layer` / `softmax_degree_per_layer`**
  it was trained against — a Stage-2 config binds to exactly one Stage-1, so the
  standalone Stage-2 final-eval reads both from here. (REVISED 2026-05-30.)
- `final_eval.json` — a **basic single-eval** snapshot of the best config on
  validation_full (loss, metric1, metric2) + cost (Stage-1: degree cost;
  Stage-2: Rescale_optimizer total_bits / fusion / sum_k) + **% change vs
  baseline** (baseline = original no-approx model for Stage-1; static_skeletons
  all-max action for Stage-2). The full same-cost 51-group comparison lives in
  the standalone final-eval tool, not here. (REVISED 2026-05-30.)
- `reward_curve.png`, `entropy_curve.png`, `metric_curve.png` (metric curve
  derived from per-episode metric1 when not already plotted).
- `report.md` — short human summary (config + metrics + % deltas + curve refs).
- `metadata.json` — model, dataset, run-id, run-number, timestamp, constraints,
  source code commit, episode count.

## 7. Launcher interface

```bash
# Stage-1 (writes Parting Chapter/stage1/bert base mrpc/, records on completion)
bash llama_7B_LayerImportance.sh run rl --preset bert-base-mrpc-stage1-rl [--fresh]

# Stage-2 (reads stage1 record by default; writes Parting Chapter/stage2/bert base mrpc/)
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl [--fresh] \
    [--stage1-run-id "bert base mrpc 2 20260530"] [--stage2-fixed-config path.json]
```

- New flag `--stage1-run-id` (Stage-2 only): pick a specific Stage-1 record.
- `--fresh` semantics per §4 (wipe + new run). The old `--fresh-stage1` /
  `--fresh-stage2` / `--fresh-start` triplet collapses to a single `--fresh`
  per stage (each stage is its own run now); legacy flags error with guidance.
- Resume detection: presence of an in-progress checkpoint in
  `stage{1,2}/{combo}/` (no slug lookup).

## 8. Implementation architecture

New module **`config/run_layout.py`** — the single source of truth for the new
layout (pure path/string logic + small fs helpers, mirrors `config/paths.py`
style):

- `combo_name(model_type, dataset) -> "bert base mrpc"`
- `stage_persistent_dir(stage, model_type, dataset, root) -> ".../stageN/{combo}"`
- `stage_record_root(stage, root) -> ".../stageN/record"`
- `next_run_number(stage, model_type, dataset, root) -> int`
- `run_id(model_type, dataset, n, timestamp) -> "bert base rte 1 20260530"`
- `make_record_dir(...)` / `snapshot_run_to_record(working_dir, record_dir, artifacts)`
- `COMPLETED_MARKER` constant + helpers `mark_completed` / `is_completed`.

`config/paths.py` gains `STAGE1_SUBDIR="stage1"`, `STAGE2_SUBDIR="stage2"`,
`RECORD_SUBDIR="record"`, `COMPLETED_MARKER_FILENAME`.

Touchpoints (route through `run_layout`):

- `llama_7B_LayerImportance.sh` — replace `PERSISTENT_DIR` slug construction
  (lines ~1257-1320) with stage-aware combo dir; remove chained mode; new
  resume/fresh logic; `--stage1-run-id` passthrough; emit `run_output_dir`.
- `rl_tune.py` — drop Stage-1→Stage-2 chaining; thread stage + run_output_dir;
  trigger per-stage completion snapshot (config + curves + basic single-eval
  metric → record); the heavy final-eval auto-trigger is removed.
- `layer_importance_evaluator.py` — `run_output_dir` anchor for Stage-1
  (`compute_output_layout`), record snapshot on completion, metric-curve helper.
- `blb_stage2_rl/runner.py` — `resolve_blb_persistence_dir` → `stage2/{combo}`;
  Stage-1-record degree resolution; completion snapshot.
- `blb_stage2_rl/sequential_runner.py`, `blb_stage2_rl/persistence.py` —
  record snapshot wiring + metric curve.
- `tools/status_board.py`, `tools/experiments_log.py`, `tools/aggregate_seeds.py`,
  `tools/paper_figures.py` — discover the new `stage{1,2}/` + `record/` layout
  (in addition to legacy `persistent/`).
- Docs: `CLAUDE.md` (persistence section + critical mental model), `docs/
  ARCHITECTURE.md` §4, `docs/GLOBALS.md`.

## 9. Backward compatibility / out of scope

- Existing `Parting Chapter/persistent/...` data and the server's active runs are
  untouched and stay readable. No auto-migration.
- GA / greedy / general-rl / compare and legacy `noise_rl_module_v2` Stage-2 keep
  the old `persistent/` layout (their slug logic is unchanged).
- `experiments/registry.jsonl` keeps its schema; new runs add a `record_dir`
  field pointing at the new record entry.

## 10. Testing strategy

- Torch-free unit tests for `config/run_layout.py`: combo naming (spaces),
  `next_run_number` scan over `record/` with spaced names (incl. coexistence of
  `bert base rte 1` and `bert base mrpc 1`; second `bert base rte` → 2),
  run-id format, completed-marker round-trip, constraint-mismatch guard.
- A snapshot test: given a fake working dir with curves/config, `snapshot_run_to_
  record` produces the documented record contents.
- Stage-2 input resolution test: picks max-N stage1 record; respects
  `--stage1-run-id`; errors when absent and no JSON.
- Launcher: `bash -n` + a dry-run that asserts the computed `run_output_dir` for
  each stage/combo and that the chained mode errors.

## 11. Risks / considerations

- **Spaces in paths** touch every shell line that handles the dir — must quote
  consistently; the `next_run_number` parser must split on the trailing
  ` {N} {YYYYMMDD}` carefully (combo itself contains spaces).
- **Server interruption:** the layout change ships via git; the server's active
  legacy run keeps using the old path until restarted with the new flags. We do
  not move running data.
- **Constraint-in-metadata** (not in path) means re-running a combo with a new
  tolerance requires `--fresh`; the guard prevents silent mis-resume.
- Removing the chained mode may break user muscle-memory / old presets that set
  `--mode train`; those now error with a clear message.

## 12. Grilled refinements (2026-06-01, locked by user)

A grill-me pass before implementation locked five remaining branches:

1. **Working-dir internal layout = FLATTEN.** Products land directly under
   `stage1/{combo}/…` and `stage2/{combo}/progress/…` — no redundant inner
   `stage1/` / `stage2_noise/` nesting. The legacy `resolve_run_output_layout`
   stays for old `persistent/` runs; the new code paths use a stage-aware
   flattened layout from `config/run_layout.py`.
2. **Chained mode removal = require explicit `--mode`.** `run rl` must pass
   `--mode stage1-only` or `--mode stage2-only`. `train` / `eval` / `search-only`
   (the chained values) error with guidance. The `eval` *subcommand* is
   untouched — it already `exec`s `Paean/run_final_eval.sh` before `--mode`
   parsing (launcher line ~286). The canonical Stage-2 preset
   `mrpc-blb-stage2-rl.conf` switches to `--mode stage2-only` and drops the
   fixed-JSON Stage-1 source (`glue_final_configs_best_ppo.json`) in favour of
   the default `stage1/record/` read; the JSON path stays available as an
   explicit `--stage2-fixed-config` override. Only `SEARCH_ALGORITHM=rl` uses
   the new layout; `ga` / `greedy` / `general-rl` keep legacy `persistent/`.
3. **Final-eval coupling = remove the heavy auto-trigger now.** On completion
   each stage writes only a *basic snapshot* to `record/{run-id}/`: Stage-1 =
   one exact plaintext eval on `validation_full`; Stage-2 = mean over
   `--stage2-k-trials` MC noise trials on `validation_full` (no single-trial
   "best" claim). The heavy same-cost 51-group comparison is the separate
   standalone final-eval tool (its own approved spec), run manually later.
   Interim consequence accepted: no automatic same-cost comparison after
   training until that tool ships.
4. **degree-0 / ReLU → Stage-2 guard = DEFERRED.** This change makes Stage-2
   auto-read the Stage-1 record by default, so a Stage-1 record containing a
   gelu degree-0 (ReLU) layer can now reach Stage-2. We do NOT add the task-D
   guard in this change; a degree-0 record fed to Stage-2 will crash at the
   `block5_n0` lookup (cryptic). Accepted for now; task D remains a separate
   future item.
5. **Secondary read-tools = light discovery.** `status_board`,
   `experiments_log`, `aggregate_seeds`, `paper_figures` gain discovery of the
   new `stage{1,2}/record/` layout (in addition to legacy `persistent/`) so new
   runs/records show up; `experiments/registry.jsonl` rows gain a `record_dir`
   field. No deeper cross-stage aggregation / new figure types this round.
