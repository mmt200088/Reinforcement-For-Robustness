# Changelog

All notable changes to this project are recorded here, in reverse chronological
order. Format inspired by [Keep a Changelog](https://keepachangelog.com/),
slightly relaxed for a research codebase.

Versioning is informal so far (no git tags yet). Each entry below ties to
one or more commits on the `jk_standard_rl` branch. The rationale for major
decisions lives in `docs/adr/`.

## Conventions

- **Added** — new features / files / modules
- **Changed** — behavior changes (back-compat: usually yes; if not, called out)
- **Removed** — deletions (modules, fields, flags)
- **Fixed** — bug fixes
- **Docs** — documentation-only changes
- **Infra** — CI / build / deps / packaging

When in doubt, check the linked commit's full message for surgical details.

---

## [Unreleased] — 2026-06-13 (pm2)

### Added
- **Stage-2 RL outputs aligned with Stage-1** (图片 / 中间结果 / 细节输出 / 归档).
  The default *sequential* Stage-2 path early-returns before the decoupled
  archive block in `runner.py`, so finished Stage-2 runs were missing the
  `record/` archive, `metadata.json`, `COMPLETED` marker, `final_config.json` /
  `final_eval.json`, a Stage-1-style training curve, an entropy curve, and a
  local-optimum detection report that Stage-1 produces. `run_sequential_via_runner`
  now (gated on `ev.decoupled_layout`): upgrades `blb_stage2_training_curve.png`
  to the Stage-1 multi-panel style (Reward / Loss / metric1 / metric2 /
  fusion_count / avg_K, each raw + Moving Avg + Baseline) + a separate
  `blb_stage2_entropy_curve.png`; writes `blb_stage2_search_log.txt` (Stage-1
  `pruning_search_log.txt` format); archives into
  `Parting Chapter/stage2/record/{combo N date}/` + `COMPLETED` + combo-level
  `metadata.json` + `best_policy/` — matching Stage-1.
- `rl_local_optimum.py` (torch-free): `detect_rl_local_optimum` moved here from
  `layer_importance_evaluator.py` (re-exported there → Stage-1 +
  `noise_rl_module_v2` unchanged) + a `write_local_optimum_report` helper.
- `scripts/blb_regen_stage2_outputs.py`: offline (torch-free) regenerator that
  rebuilds the upgraded curves + entropy + detection report from a finished run's
  `diagnostics/episodes.jsonl(.gz)` + `ppo_updates.jsonl`. Backfills history and
  gives local eyeball verification without a server run.
- `tests/test_blb_stage2_outputs.py`: torch-free coverage (upgraded curve fn +
  back-compat, detection report, regenerator plain/gz, stage-2 archive shape).

### Changed
- `persistence.write_training_curves` gained optional per-episode series +
  `baselines` + `entropy_series`/`entropy_episodes`; absent → degrades to the
  legacy reward panel (back-compat). New constants `BLB_ENTROPY_CURVE_PNG`,
  `BLB_SEARCH_LOG_TXT`.

---

## [Unreleased] — 2026-06-13 (pm)

### Changed
- **Stage-2 fusion reward: Stage-1-style log-barrier accuracy boundary**
  (ADR-013). The 3rd 60k run collapsed HOT (fusion ran away 1.4→35, accuracy
  destroyed, back half frozen at -6.95). `reward.accuracy_margin_barrier`
  replaces ADR-012's near-miss tier (P1) and the linear P3 metric-margin (P3)
  with the two-piece log-barrier from Stage-1: a steep ≤0 restoring penalty as
  the margin thins (→ interior reward peak at positive-margin headroom, no
  overshoot) and a continuous monotone violated region (→ recovery gradient, no
  flat cliff floor). Cost weights unchanged (80:150:130:40); the barrier is the
  sole restoring force. `MARGIN_REF` (default 0.25 ≈ 1.8 probe-σ) is the
  aggressiveness knob. Priority / rank-key / selection bit-identical; item 7 and
  1==N preserved. `acc_barrier_enabled=False` restores the ADR-012 path. Locked
  by `tests/test_blb_log_barrier_reward.py`.
- **Per-block-type fusion diagnostics**: `fusion_count_b2/b4/b5` now recorded in
  `episodes.jsonl` (both serial and episode-parallel paths) so a runaway block
  type (e.g. the accuracy-toxic block4) is a one-glance read.

---

## [Unreleased] — 2026-06-13

### Changed
- **Stage-1 eval acceleration (bert-large focus)**: TF32 fast matmul enabled
  for Stage-1 scoring (same `enable_cuda_reward_probe_fast_math` setting the
  Stage-2 reward probe has used since 2026-05); `PolynomialGELU._poly` is now
  Horner (no more (deg+1)× stacked-powers intermediate on the FFN-wide
  activation); `approximation_exponential` (BERT + GPT-2) uses repeated
  squaring instead of a scalar `powf` kernel; `_run_evaluation` defers all
  GPU→CPU syncs to one point after the forward loop (bit-identical, locked by
  `tests/test_stage1_eval_accel.py`).
- **Stage-1 multi-GPU worker eval cache** (`stage1_rl/eval_cache.py`):
  the worker path now shares a lock-protected deterministic eval cache, so
  repeated (gelu, softmax) configs skip the whole install + forward. Exact
  same floats on hit → `rollout_sig` / GPU-count-independence unaffected.
  Per-window hit-rate logged as `[stage1-rollout] … eval_cache hits=…`.

---

## [Unreleased] — 2026-05-16

### Infra
- **CI workflow** (`.github/workflows/ci.yml`): runs torch-free unit tests
  (matrix py 3.10 / 3.11), `ruff check` + `ruff format --check`, `pip-audit`
  (advisory), and a docs-sanity job that verifies ADR index coverage + HTML
  guide tag balance. Concurrency-cancel so quick pushes don't pile up.
- **Ruff config** (`pyproject.toml [tool.ruff]`): conservative baseline
  (E, F, I, B, UP, SIM, RUF). Legacy giants (`layer_importance_evaluator.py`,
  `noise_rl_module_v2.py`) exempted via `per-file-ignores` until they're
  refactored. `[tool.ruff.format]` matches Black defaults.
- **Makefile**: one-liner shortcuts — `make test` / `lint` / `format` /
  `audit` / `docker` / `train` / `train-multi-seed` / `index` / `figures` /
  `changelog` / `clean`. `make help` lists everything.
- **CHANGELOG.md** (this file): chronological release notes alongside the
  per-decision ADR records.

### Added
- `tools/run_multi_seed.sh` + `tools/aggregate_seeds.py` — multi-seed
  sweeps with per-seed isolated persistent dirs (`--run-tag` slug suffix),
  auto-aggregated `seed_summary.{md,json}` (mean ± std, per-seed table,
  failure log).
- `tools/experiments_log.py` — append-only `experiments/registry.jsonl`
  + auto-rebuilt `experiments/index.md` index. Subcommands:
  `register` / `rebuild` / `query`. `sequential_runner.py` auto-registers
  at training end via subprocess hook (resilient to Ctrl-C).
- `tools/paper_figures.py` — paper-friendly figures: `training_curves`
  (multi-seed mean ± std band), `invalid_heatmap`, `best_vs_baseline`,
  `action_histogram`, `ppo_dynamics`, `cost_vs_accuracy`. Times serif,
  Wong 2011 colorblind palette, 300 DPI PNG + vector PDF, optional
  LaTeX `booktabs` summary table.
- `docs/adr/` (Architecture Decision Records) with 6 initial ADRs:
  001 per-block sequential PPO · 002 hard-priority reward ·
  003 per-block K baseline · 004 static_skeletons baseline ·
  005 SF/K-first outputs · 006 F0/F1/F4 fidelity ladder.
  `_TEMPLATE.md` + `README.md` (when-to-write rules, status workflow).
- `blb_stage2_rl/diagnostics.py` (`RLDiagnosticsRecorder`) writes
  `<progress_dir>/diagnostics/` with `diagnostics_summary.md`,
  `episodes.jsonl`, `ppo_updates.jsonl`, `top_candidates.jsonl`,
  `first_invalid_counts.json`, `action_histogram.npz`,
  `baseline_action_vec.json`, `best_action_vec.json`. Auto-flag warnings
  (learning regression / training stall / first-invalid concentration /
  policy collapse).
- `blb_stage2_rl/action_io.py` — bidirectional `action_vec ↔ slots_list`
  converter, used by diagnostics, persistence, and Paean's
  `load_action_grid_config`.
- `requirements.txt`, `pyproject.toml`, `Dockerfile`, `.dockerignore`,
  `docs/SETUP.md` — env setup story (Docker / venv parity, GLUE
  pre-download, "report your env" snippet).
- `--blb-v3-seed N` / `--run-tag SUFFIX` launcher flags (multi-seed support).

### Changed
- **All training artifacts now SF/K-first.** `blb_stage2_best_action_full.{json,md}`,
  `blb_stage2_baseline_action_full.{json,md}`, `blb_stage2_report.md`, and
  `blb_stage2_status.json` (`best.slots`, `best.slots_by_layer`) lead with
  decoded `scaling_factor` / `truncation_bits`. `action_index` is kept as a
  sanity-check sidekick column. The flat `action_vec` is retained as a
  fallback field for Paean's old reader.
- **Per-block K baseline** changed from uniform `K=13` to
  `{B1=13, B2=10, B3=13, B4=10, B5=13}` (see ADR-003).
  L=12 baseline `avg_k ≈ 11.78`.
- **Paean `--action-config`** now accepts 4 schemas: full slots list, slots
  dict (label → value), `base + overrides`, legacy `action_vec`. Forgiving
  snap with stderr warning when a requested SF isn't in the noise table.
- **Sequential per-block PPO** is the default Stage-2 path
  (`--blb-v3-sequential-rl true`). Single-shot reachable via
  `--blb-v3-no-sequential-rl`.
- **CLAUDE.md** + `reports/session_summary/blb_stage2_rl_guide.html`
  refreshed to cover the long-term research workflow (multi-seed,
  experiments log, paper figures, ADRs, Docker setup, SF/K-first artifacts).

### Removed
- **F2 / F3 fidelity tiers**: `candidate_store.FIDELITY_ORDER` simplified
  to `{F0, F1, F4}`. Old JSONL records with `fidelity="F2"`/`"F3"` get
  `fidelity_rank == -1` (legacy; not promotable). See ADR-006.
- `blb_stage2_rl/default_invoker.py` — already-deleted file, cleaned up.

### Fixed
- Sequential RL resume hardening (3 issues): auto-detect existing
  `blb_stage2_rl_checkpoint_live.pt`, restore `best_reward` +
  `best_action_vec`, skip checkpoints whose `rl_variant` ≠
  `blb_v3_sequential` with a clear warning.
- `remaining_episodes = max(0, total_episodes - start_episode)` — guard
  against negative on over-resume.
- Seed offset on resume: `seed_for_this_run = seed + start_episode` so
  resumed runs don't replay the same RNG.

---

## 2026-05-15 — Sequential per-block RL · default

### Added
- `blb_stage2_rl/sequential_env.py` (`BLBStage2SequentialEnv`,
  `SequentialEnvConfig`).
- `blb_stage2_rl/sequential_policy.py` (`BLBStage2SequentialPolicy`,
  `SequentialRolloutBuffer`, `sequential_ppo_update`).
- `blb_stage2_rl/sequential_runner.py` (`train_sequential`,
  `run_sequential_via_runner`).
- `--blb-v3-sequential-rl true|false` and supporting CLI flags
  (`--blb-v3-sequential-invalid-penalty`,
   `--blb-v3-sequential-cost-shaping-coeff`,
   `--blb-v3-sequential-fusion-shaping-coeff`,
   `--blb-v3-sequential-early-terminate-on-invalid`).
- Helpers in `action_space.py`: `step_schedule`,
  `splice_step_action_into_full_vec`, `step_schedule_max_dim`.

### Changed
- `BLBStage2RLRunner.run` dispatches to `run_sequential_via_runner` when
  `train_cfg.sequential_rl=True`. Default flipped to `True`.

Commits: `d8efed0` "Add per-block sequential RL as the default BLB Stage-2 path".

---

## Earlier

The pre-2026-05-15 history is in `git log` (see `docs/STATUS.md` for the
running snapshot). Highlights:

- **2026-05-14** — `HeuristicStubInvoker` deleted; every reward number now
  passes through real `Rescale_optimizer.replan_with_user_actions`. See
  ADR-004 for the rationale.
- **2026-05** — first-input fresh slot deprecated (`effective=False`);
  retained in action vector for backward compat.
- **earlier** — Stage-1 GTrXL PPO over GELU/Softmax polynomial degrees;
  Stage-2 (legacy_v2) single-N PPO. See `docs/BLB_stage2_rl_FULL_FLOW.md`
  for the full pre-blb_v3 history.

---

## How to add an entry

1. Find your commit's category (Added / Changed / Removed / Fixed / Docs / Infra).
2. Prepend the new line under the appropriate section in **[Unreleased]**.
3. When the next git tag is cut, move everything under [Unreleased] into a
   new dated section.
4. Link related ADRs by number (e.g. "see ADR-005").
5. Don't expand more than 2 lines per entry — link to the commit / ADR for
   the long version.
