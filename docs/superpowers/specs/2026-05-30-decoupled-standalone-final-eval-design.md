# Decoupled, standalone Stage-1 / Stage-2 final-eval

- **Date:** 2026-05-30
- **Status:** Design (approved; awaiting spec review → writing-plans)
- **Depends on:** `2026-05-30-decouple-stage1-stage2-persistence-design.md`
  (the `record/` archive, combo naming, and a Stage-2 record carrying its
  prerequisite Stage-1 config). **Revises** that spec in two places (see §8).
- **Scope:** Canonical Stage-1 RL final-eval and canonical Stage-2 (blb_v3)
  final-eval, as standalone validation tools. NOT GA / greedy / general / compare
  final-eval, NOT the legacy `Paean/outputs/` consumers.

## 1. Problem / motivation

Today final-eval is (a) **chained to training** — `UnifiedFinalEvaluationModule`
runs at the end of an RL run via `Paean/embedded.run_embedded_final_eval` — and
(b) **Stage-1 + Stage-2 are merged** in one module, writing under
`Paean/outputs/{dataset}/{algorithm}/{run}/final_eval/`.

Goal: make final-eval a **standalone, full-validation-set evaluation** decoupled
from training, with **separate Stage-1 and Stage-2** tools, each producing a
**same-cost peer-group comparison** that shows where the selected optimal config
ranks among same-cost alternatives.

## 2. Domain anchor (why Stage-2 needs Stage-1)

A Stage-2 config is strictly bound to exactly one Stage-1 config: the Stage-1
GELU/Softmax degrees fully determine Block 3 (`block3_exp_n<softmax>`) and Block 5
(`block5_n<gelu>`). One Stage-1 → many Stage-2; one Stage-2 → exactly one
Stage-1. So the Stage-2 final-eval must evaluate the **stage1+stage2** config
together, hold Stage-1 **fixed** during cost-matched sampling, and read Stage-1
from the Stage-2 record (never re-derive it). (Recorded in AGENTS.md Critical
Mental Model #9 and project memory `stage1-stage2-binding`.)

## 3. Locked decisions (from brainstorming)

1. **Decoupling:** RL training completion records **config + curves + a basic
   single-eval metric snapshot** only. The heavy 51-group validation is a
   **separate standalone tool**. The training-embedded auto-trigger is removed.
2. **Stage-1 same-cost domain:** sample RL-selectable degrees (gelu∈{1,2,4},
   softmax∈{2..6}) with total Stage-1 cost **exactly** equal to the selected
   config; if a near-extreme cost yields <50 distinct configs, use as many as
   exist and report the shortfall.
3. **Archive dir name:** `record` (consistent with the persistence spec;
   "report" was shorthand for the same thing).
4. **Plot style:** per metric, a **sorted bar chart** with the selected config's
   bar highlighted (rank read off directly).

## 4. Output layout

```
Paean/
├── stage1/
│   ├── bert base mrpc 1 20260530/        ← one standalone Stage-1 FE run
│   └── bert base rte 1 20260530/
└── stage2/
    └── bert base mrpc 1 20260530/        ← one standalone Stage-2 FE run
```

- Each standalone final-eval run is one **flat numbered dir** (no `record/`
  sublevel — these *are* the FE outputs, not RL working dirs).
- Run dir = `{combo} {N} {YYYYMMDD}`, combo = `{model_type '-'→' '} {dataset}` →
  `bert base mrpc`. `N` = (existing `Paean/stage{1,2}/{combo} *` dirs) + 1, an
  independent "final-eval 序号" (unrelated to the RL run number).
- Replaces today's `Paean/outputs/{dataset}/{algorithm}/{run}/final_eval/` for
  these two standalone tools.

### Input interface
- `--stage stage1|stage2` selects the tool.
- `--record-dir <path>` points at the RL `record` run dir holding the optimal
  config (matches "通过文件目录指定"). Convenience `--run-id "bert base mrpc 2
  20260530"` resolves into `Parting Chapter/stage{1,2}/record/<run-id>/`.
- `--repeat` (Stage-2 trials/group, default 50), `--cost-match-count` (default
  50), and existing knobs remain.

## 5. Stage-1 standalone final-eval

Module `Paean/stage1_final_eval.py::Stage1FinalEvaluationModule`.

1. Read `final_config.json` from the record → `gelu_degree_per_layer`,
   `softmax_degree_per_layer`.
2. New sampler `build_cost_matched_stage1_configs(selected_gelu,
   selected_softmax, num_layers, count=50, max_attempts=...)` in
   `Paean/action_grid.py`: produce up to 50 **distinct** RL-domain degree vectors
   with total Stage-1 cost (Σ `GELU_COST[g]` + `SOFTMAX_COST[s]`) **exactly**
   equal to the selected config. Randomized sampling with dedup against the
   selected and each other; cap by `max_attempts`; return `(configs,
   shortfall_count)`.
3. Evaluate all 51 (50 + selected) **once each** on `validation_full` via the
   evaluator's `evaluate_model(gelu, softmax, split="validation_full")` — **no
   noise, no Stage-2** (plaintext approximated model only). Each group →
   `(loss, m1, m2)`.
4. Outputs in the run dir:
   - 3 sorted-bar PNGs: `stage1_loss_compare.png`, `stage1_m1_compare.png`,
     `stage1_m2_compare.png` — groups sorted by the metric, **selected bar
     highlighted** + annotated with its rank.
   - `stage1_final_eval.json` — per-group config + (loss, m1, m2) + cost, the
     selected group flagged, plus the shortfall count.
   - `report.md` — selected config, its rank per metric, baseline % deltas.

## 6. Stage-2 standalone final-eval

Adapt `Paean/blb_action_eval.py::BLBActionFinalEvaluationModule` into a
standalone entry (it already does cost-matched valid sampling + repeated trials
on `validation_full`).

1. Read the Stage-2 record's `final_config.json` → `blb_v3_best_action_vec`
   **and** the prerequisite `gelu_degree_per_layer` / `softmax_degree_per_layer`
   (§8 guarantees they are stored together).
2. Cost-matched sampling via `build_cost_matched_random_action_candidates`, with
   **Stage-1 held fixed** to the record's prerequisite Stage-1; only Stage-2 cost
   matched (`total_bits_sum` + `total_fusion_count` + `sum_truncation_k`); every
   accepted group must be `Rescale_optimizer`-**valid** (no invalid_chain) — this
   is the module's existing rejection criterion.
3. Each of the 51 groups evaluated **`--repeat` (default 50)** times on
   `validation_full` with the stage1+stage2 install → per-group mean + std of
   (loss, m1, m2).
4. Outputs in the run dir:
   - 6 sorted-bar PNGs: `stage2_{loss,m1,m2,loss_std,m1_std,m2_std}_compare.png`
     — selected highlighted.
   - `stage2_final_eval.json` — per-group action + cost + (mean,std) of metrics,
     selected flagged, sampling diagnostics (attempts, rejects, shortfall).
   - `report.md` + the existing GLUE-submission path stays available but is
     **not** auto-run by FE (it belonged to the training chain).

## 7. Module architecture / blast radius

- New `Paean/stage1_final_eval.py` (`Stage1FinalEvaluationModule`).
- New `build_cost_matched_stage1_configs` in `Paean/action_grid.py`.
- Adapt `Paean/blb_action_eval.py`: standalone constructor (read-from-record,
  fixed-stage1), 6-plot output, sorted-bar-highlight plots.
- Shared `Paean/final_eval_layout.py`: `paean_stage_run_dir(stage, model_type,
  dataset)` + `next_final_eval_number(...)` (scan `Paean/stage{1,2}/`), and a
  shared `sorted_bar_highlight(values, labels, selected_idx, ...)` plot helper
  used by both Stage-1 and Stage-2.
- `Paean/run_final_eval.py` + `run_final_eval.sh`: add `--stage stage1|stage2
  --record-dir … [--run-id …]`; route to the two modules; drop the
  training-coupled path.
- **Remove the training auto-trigger:** `Paean/embedded.run_embedded_final_eval`
  call sites in the RL completion path (and the `UnifiedFinalEvaluationModule`
  invocation from training) are removed; `UnifiedFinalEvaluationModule` itself
  may stay for legacy `Paean/outputs/` callers but is no longer wired into
  training.
- Docs: CLAUDE.md (final-eval section), `docs/ARCHITECTURE.md`, AGENTS.md
  (Final Eval Routing + Persistence).

## 8. Revisions to the persistence spec

In `2026-05-30-decouple-stage1-stage2-persistence-design.md`:

- **§5 "Per-stage completion chain":** change from "completion runs final-eval →
  record" to "completion snapshots **config + curves + a basic single-eval
  metric** into the record; the full final-eval is the standalone tool in this
  spec." Stage-2 completion no longer auto-runs BLB final-eval + GLUE.
- **§6 "Record contents":** the Stage-2 `final_config.json` must store the
  prerequisite `gelu_degree_per_layer` / `softmax_degree_per_layer` alongside
  `blb_v3_best_action_vec`, so the Stage-2 standalone FE reads both from one
  place.

## 9. Out of scope

- GA / greedy / general / compare final-eval, and `UnifiedFinalEvaluationModule`'s
  legacy `Paean/outputs/` behavior, are untouched.
- No migration of existing `Paean/outputs/` runs.
- GLUE submission generation stays available (`generate_glue_submission.py`,
  `Paean.blb_action_eval` glue path) but is not auto-run by standalone FE.

## 10. Testing

- Torch-free: `build_cost_matched_stage1_configs` — exact-cost equality,
  dedup, RL-domain only, shortfall reporting near cost extremes, deterministic
  under a seed.
- Torch-free: `next_final_eval_number` scan over spaced `Paean/stage{1,2}/`
  dirs (coexistence of `bert base rte 1` and `bert base mrpc 1`; second
  `bert base rte` → 2).
- Torch-free: `sorted_bar_highlight` data shaping (sort order, selected index
  tracking through the sort, rank computation).
- Torch-free: Stage-2 record round-trip carrying stage1+stage2.
- A small integration smoke (torch, server): Stage-1 FE on a tiny config
  produces 51 results + 3 plots; Stage-2 FE produces mean/std + 6 plots.

## 11. Risks / considerations

- **Stage-1 same-cost shortfall:** a cost-optimal Stage-1 config can sit near a
  cost extreme; the sampler must report how many of 50 it actually filled, and
  the plots/report must state the realized group count.
- **Stage-2 sampling cost:** 50 valid same-cost groups × 50 trials ×
  validation_full is heavy; reuse the module's existing attempt caps and
  full-val batching. This is a server job, not local.
- **Spaces in run-dir names** (combo has spaces) — the `next_final_eval_number`
  parser must split the trailing ` {N} {YYYYMMDD}` carefully.
- **Dependency ordering:** this spec needs the persistence refactor's `record`
  layout in place first; implementation order is persistence → final-eval.
