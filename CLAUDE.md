# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research codebase that searches for an optimal noise / approximation configuration for **CKKS + MPC privacy-preserving inference** of BERT (and to a lesser extent GPT-2). Two search phases per task:

- **Stage 1** — picks GELU / Softmax polynomial-approximation degrees per layer.
- **Stage 2 (BLB)** — picks per-layer scaling factors for every CKKS noise point in 5 blocks (encode / fresh / rescale, plus rotation flags) plus a per-block MPC↔HE truncation `k`. The new "BLB Stage 2 RL" (`blb_v3`) is the final/canonical implementation; the old single-N implementation (`legacy_v2` in `noise_rl_module_v2.py`) is kept only for reproducing past experiments.

Both stages are PPO; a GA path (`genetic_search_module.py`) and a greedy path (`greedy_search_module.py`) exist as alternatives. After search, results go through `final_evaluation_module.py` (`UnifiedFinalEvaluationModule`, the merged Stage-1 + Stage-2 final eval).

## Critical mental model (read first)

These facts override naive readings of the code. Full rationale: `project_understanding_blb_stage2_rl.md`.

1. **Plaintext SIMULATION, not real ciphertext computation.** The model still runs as PyTorch fp32; at every position where CKKS encode / fresh / rescale / rotation or MPC truncation *would* happen, we inject Gaussian noise (variance from `NOISE_VARIANCE_TABLE_BY_N`) or fixed-point truncation. We are searching for a *schedule* that would be safe under the real protocol, not running the protocol.
2. **The 5 BLB blocks and their CKKS scale points are FIXED operations.** RL never decides whether an operation happens. Every must-exist slot must receive an action. Any "mask" is an *index mask* (restrict which action indices a slot may pick), never an *operation mask*.
3. **Actions are integer INDICES, not scaling factors.** Policy outputs `a_j ∈ {0,…,m_j-1}` per slot; a decode rule (`sf_from(idx, max_sf, levels) = max_sf - 2*(levels-1-idx)` for SF slots; the `K_LEVELS` table for K slots) maps each index to the actual `scaling_factor` or truncation `k`. Always log both `action_index` AND `decoded_value`. K decoding is **non-monotonic** in some checkpoints — find all-max via the largest `k` value, not the largest index.
4. **Slot kinds carry distinct semantics.** `F` (fresh ciphertext), `W` (weight encode), `M` (mask encode), `S` (scalar encode), `R` (rescale target scale), `K` (block-output truncation bits). Same `scaling_factor` produces *different* noise variance under different distributions — never collapse kinds into plain integers.
5. **Rotation noise has NO independent action.** Its scale is bound to the *current* scale (post-rescale, if a rescale precedes it). If a rescale wasn't selected/executed by `Rescale_optimizer`, its trailing rotation noise also must not be added. Don't invent a freestanding rotation SF action.
6. **Two episode formulations.** The legacy single-shot path (`BLBStage2Env`, `BLBStage2Policy`, `runner.BLBStage2RLRunner`) treats every `env.step(action_vec)` as a horizon-1 episode that emits the full 577-dim config in one go (so GAE degenerates to `A = r − V(s)`). The new sequential path (`BLBStage2SequentialEnv`, `BLBStage2SequentialPolicy`, `sequential_runner.train_sequential`) decomposes the same action into a horizon-N sequence — one `(layer, block)` per step, in the order `L0:B2→B3→B4→B5, L1:B1→B5, …, L11:B1→B5` (59 steps for L=12; layer 0 has no block 1). first-input fresh is folded into step 0 (layer 0 block 2). Per-step reward = ReplanSession cost on just that block (invalid → big penalty); terminal step adds the existing full hard-priority reward. The single-shot path stays canonical for tests, F0 scan, and candidate store; sequential is opt-in for training when the action space is too wide for cold-start PPO to converge.
7. **Reward is hard-priority, not weighted-sum.** `invalid → accuracy → stability → cost`. Cost reward must never offset an accuracy or stability violation. Final-best selection should use a tuple rank key `(invalid_flag, acc_violation, stab_violation, normalized_cost, …)`, not raw PPO reward.
8. **`Rescale_optimizer` is the source of truth for modulus-chain validity and cost.** Every reward number, including per-step ones in the sequential path, must come from a real `replan_with_user_actions` (via `ReplanSession`). The legacy `HeuristicStubInvoker` was deleted on 2026-05-14; `InProcessInvoker.from_profile(...)` is hardcoded for training. If the real package fails to load, training aborts (no fallback).

The **"59 required slots"** is the user-stated target. Older `action_space.py` field tables export ~73 fields/layer and stale doc comments still say 94. **Trust `scripts/blb_export_action_registry.py` over comments** before changing slot counts; classify discrepancies as `required / effective-extra / compat-extra / inactive` rather than deleting.

## Common commands

**All training / evaluation goes through one launcher** (`bash llama_7B_LayerImportance.sh ...`); do not call the underlying `rl_tune*.py` directly. Subcommands and presets:

```bash
# List presets
bash llama_7B_LayerImportance.sh --list-presets

# First BLB Stage-2 RL run (must pass --fresh first time for a parameter combo)
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh

# Resume same parameter combo (auto-detects persistent dir, no --resume-from needed)
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl

# Final eval — independent module (preferred). Old `llama_7B_LayerImportance.sh eval ...`
# is now a compatibility shim that delegates to this entrypoint.
bash Paean/run_final_eval.sh --preset mrpc-final-eval-only
bash Paean/run_final_eval.sh --preset mrpc-blb-max-final-eval
# BLB action grid: cartesian sweep over decoded fields, with per-block/per-layer selectors
bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed \
  --range block3.truncation=8,9,10,11,12,13 \
  --action-fixed layer2.block5.wffn1_sf=18
# Compatibility entrypoint (same effect):
bash llama_7B_LayerImportance.sh eval --preset mrpc-final-eval-only

# RL vs GA comparison from persistent dirs
bash llama_7B_LayerImportance.sh compare --dataset mrpc

# Cross-task general policy
bash llama_7B_LayerImportance.sh general train --general-rl-tasks mrpc,cola,rte,stsb --fresh
```

`--mode` is the safe wrapper for the various skip flags: `train` / `eval` / `stage2-only` / `stage1-only` / `search-only`. Old `--skip-*` flags still work but conflict-checked.

**Paean final-eval module** (`Paean/run_final_eval.{sh,py}`) is now the standalone final-eval entrypoint, separate from training. Knobs:
- Presets live in `Paean/presets/` (`mrpc-final-eval-only.conf`, `mrpc-blb-action-range.conf`, `mrpc-blb-max-final-eval.conf`, `mrpc-blb-baseline-truncation-sweep.conf`, `mrpc-blb-baseline-fixed.conf`, `default.conf`).
- Standalone mode does NOT generate random/perm/equiv/budget controls unless you pass `--random`; the `--perm-trials` / `--cost-trials` / `--budget-trials` defaults are `0` here (training-triggered passive final-eval still defaults to `10` each).
- `--action-config PATH` loads a BLB action JSON; `--range NAME=v1,v2,...` (repeatable) expands a cartesian grid; `--action-fixed NAME=v` (repeatable) pins a slot. Names support global / per-block / per-layer selectors (`truncation=…`, `block3.truncation=…`, `layer2.block5.wffn1_sf=…`).
- Outputs land under `Paean/outputs/{dataset}/{algorithm}/{run}/final_eval/` by default.
- Passive (training-end) final-eval is configured via `--final-eval-preset NAME` (resolved against `Paean/presets/`); training-side `--random-seed` / `--budget` / `--final-eval-repeat` do NOT control it.

**Status board** (aggregates running RL/GA/general/compare jobs into one markdown):

```bash
python tools/status_board.py --write-md   # rewrites docs/STATUS.md
```

**BLB sidecar tools** (in `scripts/`; do NOT replace the launcher — they ride alongside it):

```bash
# Confirm entrypoints, env, data, paths before any long run
python scripts/blb_phase0_preflight.py

# Export the actual slot registry; reconcile against the user's required-59
python scripts/blb_export_action_registry.py
# → reports/blb_opt/phase1_registry/slot_registry_required59_or_mismatch.md

# Offline-evaluate a single candidate action (supports F0/F1 fidelity ladder)
python scripts/blb_eval_action.py ...

# F0 sweep: scan feasible action domain via Rescale_optimizer (no model forward)
python scripts/blb_f0_scan_feasible_domain.py ...

# Compare invoker modes (in-process vs subprocess vs heuristic) on the same action
python scripts/blb_compare_optimizer_modes.py ...

# Snapshot a reproducible run manifest (env, code rev, configs, hashes)
python scripts/blb_make_run_manifest.py ...

# Orphan-slot audit: action slot ↔ cfg field ↔ bridge ↔ graph node, both
# delta_overrides and t_new paths. Pure AST + JSON (torch-free).
python scripts/blb_orphan_slot_audit.py --profile mrpc
# → reports/blb_opt/orphan_slots/audit_mrpc.{md,json}

# Noise-install verifier: drives ReplanSession + apply_optimizer_output_to_cfg
# end-to-end and emits an HTML with per-(layer,block,graph_node) install plan
# (distribution, SF, N, σ²) plus per-config valid/fusion/total_bits/effective_rotations.
# Smoke mode is torch-free; full mode needs torch + transformers.
python scripts/blb_verify_noise_install.py --mode smoke --profile mrpc --num-layers 12
python scripts/blb_verify_noise_install.py --mode full  --profile mrpc --num-layers 12 \
  --stage1 '{"gelu_degree_per_layer":[4,...],"softmax_degree_per_layer":[4,...]}'
# → reports/blb_opt/noise_install_verify/{smoke,full}_<profile>_<ts>.html
```

The candidate store (`blb_stage2_rl/candidate_store.py`) is the canonical place to persist `action_index + decoded_value + N + distribution + block + operation + metrics + rank_key`. Never log only the index.

**Tests** (plain `unittest` files, no pytest config):

```bash
# Single test file
python tests/test_blb_registry_artifact_consistency.py
python tests/test_blb_baseline_bootstrap.py
python tests/test_blb_optimizer_cost_consistency.py
python tests/test_blb_warmstart_resume.py

# All (best-effort)
python -m unittest discover -s tests -v
```

Most BLB tests are torch-free — they exercise action-space / registry / bridge / cost / mask / threshold / candidate-store / warmstart logic in isolation, and `test_blb_f0_scan.py` drives the F0 offline scan path. Only `test_blb_action_mask.py` and `test_blb_stage2_rl_regressions.py` pull in torch + transformers; `test_glue_dataset_loading.py` needs `datasets` + a populated GLUE cache (`GLUE_LOCAL_DATASET_DIR`).

## Submodules

`.gitmodules` declares **EzPC**, **LLM-Adapters**, **importance-aware-sparse-tuning-IST-paper**. `Rescale_optimizer/` is **NOT** a submodule — it is checked into the repo directly. If a fresh clone shows it as empty, see commit `3af56dd` (it was de-submoduled).

## Architecture

### Two-stage RL search → unified final eval

```
launcher (llama_7B_LayerImportance.sh)
   └─ rl_tune.py (CLI)
        ├─ Stage-1: layer_importance_evaluator.LayerImportanceEvaluator
        │    └─ GTrXL PPO over GELU/Softmax degrees per layer
        ├─ Stage-2 (variant=blb_v3, default):
        │    └─ blb_stage2_rl.BLBStage2RLRunner            ← the "final" path
        │         ├─ blb_rl_bridge.BLBNoiseRLBridge        ← cfg → model noise hooks
        │         ├─ rescale_optimizer_bridge              ← cfg → modulus chain cost
        │         └─ blb_stage2_rl.persistence             ← status board / curves / report
        └─ Stage-2 (variant=legacy_v2):
             └─ noise_rl_module_v2.NoiseRLModuleV2          ← old single-N PPO
        finally:
        └─ final_evaluation_module.UnifiedFinalEvaluationModule
```

`layer_importance_evaluator.py` and `noise_rl_module_v2.py` import each other (graceful-stop helpers, checkpoint filename constants); see `docs/GLOBALS.md`.

### BLB Stage-2 RL action flow (the path that actually matters)

```
PPO policy.sample_action(obs)
   → action_vec  (numpy int, slots-per-layer × L layers + 1 first-input)
       ▲ slots-per-layer is whatever the registry exports today; old comments say 94,
         current action_space exports ~73, the user-stated required count is 59 —
         verify with scripts/blb_export_action_registry.py before relying on a number.
   → action_space.action_vector_to_cfgs()
        → decoded.cfgs_dict() = {"block1": [L cfgs], ..., "block5": [L cfgs]}
   ┌──────────────────────────────────────┬──────────────────────────────────────┐
   │ cost path (modulus chain)            │ model forward path                    │
   │ env.step → build_optimizer_requests  │ env.step → bridge.apply(cfg, ...)    │
   │  → rescale_bridge.evaluate_blocks()  │  → handler.replace_layer_blockN_noise │
   │     → InProcess/Subprocess/Heuristic │  → model(**batch)                     │
   │       Invoker → replan_with_user_…   │     ↑ cfg's *_rescale.scaling_factor  │
   │  → opt_signals (total_bits, fusion,  │       drives per-noise-point sampling │
   │     invalid_chain)                   │       via NOISE_VARIANCE_TABLE_BY_N   │
   └──────────────────────────────────────┴──────────────────────────────────────┘
   → reward.compute_reward(metrics, opt_signals, ...) — three-priority reward
```

Two non-obvious wires (both fixed; if you regress them, the optimizer becomes blind to RL choices):

1. `RescaleOptimizerBridge.evaluate(...)` strips the `_L<i>` layer suffix from `config_name` before calling the invoker. RL uses layered names (`block1_mrpc_L0`); invoker baselines are keyed by graph (`block1_mrpc`). See `_strip_layer_suffix` in `rescale_optimizer_bridge.py`.
2. `RescaleOptimizerBridge.evaluate(...)` auto-derives `t_new` from cfg's fresh + rescale `scaling_factor` fields when caller doesn't pass one (`auto_t_new_from_cfg=True`, default). The mapping table `DEFAULT_CFG_TO_T_NEW_MAP` is keyed by `(block, profile)` (mrpc only out-of-box); add entries when supporting other profiles.

### Per-block conceptual scope

Field-level details live in the registry, not here, but the conceptual scope of each block is:

- **Block 1** — post-FFN / GELU output / Wffn2 / LayerNorm mean-variance head. Slots: GELU output fresh, Wffn2 weight encode, mean/variance scalar encodes, several rescales, block-end K. **NOT installed at layer 0** (no upstream FFN2 there; the first HE config is treated as lossless and aligned with Rescale_optimizer).
- **Block 2** — LayerNorm tail + Wq/Wk/Wv projections + QK BSGS/mask/merge. Watch for tied Wq/Wk/mask groups if the optimizer requires them — registry must mark `tied_group`. **Fully active for layer 0** — the layer-0 X goes directly into block 2's LayerNorm.
- **Block 3** — Softmax exponential approximation `exp(x) ≈ (1 + x/2^n)^(2^n)`. Some `square` rescales become inactive when Softmax degree is low — registry's `effective` flag is authoritative.
- **Block 4** — Softmax × V, Wo, post-attention LayerNorm head.
- **Block 5** — LayerNorm tail + Wffn1 + GELU polynomial chain. High-order GELU coefficient/power rescales become inactive at low GELU degree.
- **first-input fresh** — DEPRECATED (2026-05). Originally a layer-0 input fresh slot; since the first HE config is now treated as lossless, this slot stays in the action vector for backward compat but `effective=False` and is not installed.

### Persistent directories (two distinct trees)

- **Old stage-2 RL + Stage-1 RL + GA + general-RL**: `Parting Chapter/persistent/{algorithm}/{model}/{dataset}/{accuracy_slug}/...` (`accuracy_slug` is e.g. `s1t0.005_s2t0.05_s2st0.05`). Same parameters → same dir → auto-resume. See `docs/ARCHITECTURE.md` §4.
- **BLB Stage-2 RL (blb_v3)**: `Parting Chapter/persistent/{algorithm}/{model}/{dataset}/{accuracy_slug}/blb_stage2/progress/`. The runner overrides `evaluator.noise_stage_progress_dir` at the start of `run()` so all BLB checkpoints / status board / curves / final report land inside the active persistent run directory. See `resolve_blb_persistence_dir()` in `blb_stage2_rl/runner.py`.

In each BLB run dir you'll find: `blb_stage2_rl_checkpoint_{live,final}.pt`, `blb_stage2_best_cfg.pkl`, `blb_stage2_status.json` (atomically rewritten — safe to `tail -f` / `cat`), `blb_stage2_training_curve.{npz,png}`, `blb_stage2_report.md`, plus `blb_stage2_error.txt` if the loop crashed.

### Graceful stop / resume

Stage-2 RL (both variants) honors **SIGINT** and a stop-flag file (`STOP_RL` in the progress dir). The next PPO update boundary saves a checkpoint then exits with code 0. Re-running the same launcher invocation auto-resumes; the BLB runner restores PPO net + optimizer + episode counter + best reward + `episode_returns` + RNG state.

### Baseline bootstrap from `static_skeletons` archive

Before training, the runner reads `Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json` to derive the BLB baseline (max-action). For each layer it picks the correct graph entry based on Stage-1 (`gelu_degree[layer]`, `softmax_degree[layer]`):

- Block 1 → `block1_<dataset>` (skipped for layer 0 — no upstream FFN2)
- Block 2 → `block2_<dataset>`
- Block 3 → `block3_exp_n<softmax_degree[layer]>`
- Block 4 → `block4`
- Block 5 → `block5_n<gelu_degree[layer]>`

From each entry, RL extracts: fresh SF (`cut_point_sf[0].sf`), encode SFs (`propagation_deltas[*].delta` numeric), rescale SFs (`cut_point_sf[*].sf_post`). The extracted SFs are written into a calibrated `MaxSFsTable` so that **`make_all_max_action_vector` produces the exact RO baseline** when decoded.

API: `blb_stage2_rl.baseline_bootstrap.load_static_skeletons_baseline(...)` + `static_skeletons_baseline_to_action(...)`. The runner treats this archive as the only allowed BLB Stage-2 RL baseline source: if the archive is missing or a required graph key is absent, training stops instead of falling back to an estimated all-max baseline. See `docs/blb_baseline_handover_protocol.md` §0 for the full schema.

### Rescale_optimizer integration

`rescale_optimizer_bridge.py` wraps the local `Rescale_optimizer/rescale_optimizer/` package. `InProcessInvoker` is now a thin adapter over `rescale_optimizer.ReplanSession` (graphs + baselines preloaded once; per-call cost is one `replan_with_user_actions`). For BLB Stage-2 RL training the choice is **NOT** runtime-configurable — `blb_stage2_rl/runner.py` hardcodes `rescale_invoker_kind="in_process_real"`, and if `InProcessInvoker.from_profile(...)` fails (missing package, missing baseline) training aborts. The `HeuristicStubInvoker` has been deleted — every reward number now passes through real Rescale_optimizer math. Remaining invoker kinds:

- `InProcessInvoker` — adapter over `ReplanSession`; ms per call. Required for publishable results.
- `SubprocessInvoker` — forks a `python` worker; hundreds of ms per call; debug-isolation only.
- `StubInvoker` — canned responses for unit tests.

`build_optimizer_requests(profile, cfgs_dict)` produces `{"block1_mrpc_L0": ("block1", cfg), ...}`. To support a profile beyond `mrpc`, extend `default_block{1..5}_cfg_to_delta` (graph-node names) and `DEFAULT_CFG_TO_T_NEW_MAP` (skeleton-position → cfg field).

### Optimizer-driven cfg override (new path)

After `bridge.evaluate_blocks(...)`, `env.step` reads each result's `new_compact_config` and writes back into the action-decoded cfg via `apply_optimizer_output_to_cfg`. This means **the model installs noise at the SFs Rescale_optimizer actually settled on**, not at the SFs the action proposed — including the optimizer's snapping/repair, fused-away rescale points (cfg field → None), CTPT_MUL propagation deltas, and effective rotations. Previously only effective rotations flowed back this way. The inverse mapping `graph_node → cfg_attribute` lives in `GRAPH_NODE_TO_CFG_ATTR` next to `default_block{1..5}_cfg_to_delta` and must stay in sync. The override list per (block, layer) is surfaced in `env.step` info dict as `optimizer_cfg_overrides`.

For Block 2 specifically, action_space binds Q-side fields (`wq_sf`, `q_mask{1,2}_sf`) to their K-side counterparts (`wk_sf`, `kt_mask{1,2}_sf`). The optimizer override path only touches the K-side cfg fields (those are the names in `GRAPH_NODE_TO_CFG_ATTR[2]`), so `env.step` calls `sync_block2_qk_binding(cfg)` immediately after `apply_optimizer_output_to_cfg` to mirror the K-side updates onto Q-side fields. Without this sync, `function_handler.handler_block2` would install Q noise at the pre-override RL SF while K noise uses the post-override SF.

### Sequential per-block RL (opt-in, additive)

The default training path treats every `env.step(action_vec)` as a horizon-1 episode that decides all 577 dims at once. With ~5^577 search space, cold-start PPO struggles. The opt-in sequential path decomposes this into 59 horizon-N steps:

```
horizon = 4 + (L-1) * 5    # 59 for L=12
order   = (L0,B2) -> (L0,B3) -> (L0,B4) -> (L0,B5)
        -> (L1,B1) -> (L1,B2) -> ... -> (L11,B5)
```

`first_input fresh` is folded into step 0 (so step 0 has 13 slot decisions; subsequent steps have 7-12). Layer 0 has no block 1.

Three additive modules implement this:

- `blb_stage2_rl/sequential_env.py` — `BLBStage2SequentialEnv` wraps an existing `BLBStage2Env`. Each `step()` splices the per-step action into a `_pending_full_vec`, calls `ReplanSession` on **just that one block's graph** for an immediate dense cost signal, applies `apply_optimizer_output_to_cfg` + `sync_block2_qk_binding` (block 2) to that block's cfg, then either returns shaped per-step reward or, on the terminal step, hands the assembled full vec to `BLBStage2Env.step()` for the existing model forward + full hard-priority reward. Per-step reward shaping is configurable via `SequentialEnvConfig` (invalid_penalty, cost_shaping_coeff, fusion_shaping_coeff, early_terminate_on_invalid).
- `blb_stage2_rl/sequential_policy.py` — `BLBStage2SequentialPolicy` shares a trunk + a single `MultiDiscrete` head sized to `step_schedule_max_dim` (13). Per-step `(slot_mask, num_levels)` mask out padding so the same head serves every block. Includes `SequentialRolloutBuffer` with proper GAE-λ over horizon-N episodes and `sequential_ppo_update`.
- `blb_stage2_rl/sequential_runner.py` — `train_sequential(...)` drives the env+policy+buffer end-to-end. Deliberately a thin standalone driver (does NOT touch persistence / status board / candidate store) so it can run alongside the existing `BLBStage2RLRunner`.

Helpers in `action_space.py`: `step_schedule(num_layers, profile, attn_degree_per_layer, gelu_degree_per_layer)` returns the ordered list of `BlockStepSpec` with each step's slot dims, kinds, full-vec offsets, and the graph-key suffix (e.g. `block3_exp_n4`) that ReplanSession should be called against. `splice_step_action_into_full_vec(...)` writes per-step actions into the legacy 577-dim vector. `step_schedule_max_dim(num_layers)` returns the max slots per step (13 for L=12).

Smoke-tested torch-free; runner integration is left to the launcher (the existing `BLBStage2RLRunner` is 2866 lines and not factored for this — `sequential_runner.train_sequential` is the path forward).

## Verification: F0 → F1 → F4 fidelity ladder

The active ladder is **three tiers** (F2 / F3 were deprecated and removed
2026-05-16 — old candidate-store JSONL records still carry those strings but
they rank as legacy and aren't promotable):

- **F0** — optimizer-only: decode action, call `Rescale_optimizer`, collect `valid / total_bits / fusion_count`. No model forward. For registry checks, sensitivity scans, cheap candidate filtering.
- **F1** — small probe + few MC trials, online during training: catches obvious accuracy collapses cheaply. This is where the per-episode reward signal comes from.
- **F4** — final eval: full/near-full validation set, real BLB install, frozen report. **Only F4 numbers belong in "best" claims.**

The runner's "final eval" path must install the actual BLB best action (decode → `bridge.apply` → real `Rescale_optimizer`), not silently fall back to a legacy all-max baseline. If you change runner glue, verify this path explicitly.

## Conventions worth knowing

- **Don't directly call `rl_tune*.py`**. The launcher does conflict checks (e.g. `legacy_v2` rejects BLB-only flags), generates the persistent slug, and creates `LATEST_PID` / `LATEST_RUN_DIR` markers under `Parting Chapter/persistent/`.
- **First time for a parameter combo always needs `--fresh`**. The launcher refuses to start otherwise to prevent accidental overwrites.
- **MC repeated evaluation, not single trial.** Multi-trial probe (sampling RNG independent of `torch.manual_seed`) + per-slot entropy logging beat single-shot rewards. A single noise trial is not evidence.
- **Warmstart toward all-max baseline.** Action space is huge; uniform-random rollouts produce mostly invalid candidates. Bias the actor toward each slot's all-max index at init — this constrains the *prior*, not the search.
- **GLUE network instability.** `rl_tune.py` honors `GLUE_LOCAL_DATASET_DIR` / `GLUE_DATASET_DIR` / `DatasetDict.save_to_disk` dirs / local parquet / HF cache `local_files_only=True` fallback. Pre-stage data and `export GLUE_LOCAL_DATASET_DIR=...` before remote long runs.
- Logs/curves/checkpoints under `rl_results/` are mostly gitignored; un-ignored exceptions are explicit in `.gitignore` (e.g. `pruning_search_log.txt`, `persistent/**/*.csv`). Don't add new untracked artifact patterns blindly.
- The Windows console may be GBK; `BLBStage2RLRunner._make_log_safe` wraps `evaluator.log` so non-GBK chars fall back without crashing stdout (file logs stay UTF-8). Matplotlib plot titles are intentionally ASCII (the markdown report carries the Chinese). Unicode bullets (▸) in new console output get replaced with `?` in stdout but are preserved in log files.
- `GLOBALS.md` lists where global path / hyperparameter constants live. `config/paths.py` and `config/constants.py` exist as the future single source of truth, but most modules still hardcode their own — change with care.

## Hard taboos when modifying BLB Stage-2

In addition to the Critical mental model items above, these mistakes specifically tend to slip past code review:

1. Declaring a final best from any path that didn't go through real `Rescale_optimizer.replan_with_user_actions` (`HeuristicStubInvoker` was deleted on 2026-05-14; `InProcessInvoker.from_profile(...)` is hardcoded).
2. Selecting "best" from a single noise trial — final-eval must be MC-repeated.
3. Letting "final eval" silently evaluate a legacy all-max baseline instead of the BLB best.
4. Replacing the launcher entrypoint instead of riding alongside it (sidecars only).
5. Large multi-module rewrites without F0 / F1 verification between steps (and F4 before claiming a result).
6. Forgetting to call `sync_block2_qk_binding(cfg)` after any code path that mutates `wk_encode` / `kt_mask{1,2}_encode` SFs (e.g. a new override hook). Block 2's Q/K binding is action-space convention, not a cfg-level invariant — every mutation site must restore it explicitly.
7. Hard-coding episode horizon=1 when the sequential RL path is opt-in. The single-shot `BLBStage2Env` and the sequential `BLBStage2SequentialEnv` co-exist; reusable helpers must work with both (e.g. `apply_optimizer_output_to_cfg` is per-block, so it's already compatible).

## When you're investigating something specific

- "What's the project actually optimizing, semantically" → `project_understanding_blb_stage2_rl.md` (canonical conceptual reference).
- "Where is X persisted / where do logs go" → `docs/ARCHITECTURE.md` §4, then `config/paths.py`.
- "How does the launcher decide it's a resume" → `llama_7B_LayerImportance.sh` (large; search for `accuracy_slug`).
- "What action dim corresponds to what cfg field" → `blb_stage2_rl/action_space.py` top half (`_BLOCK{1..5}_FIELDS`), then `scripts/blb_export_action_registry.py` for the authoritative registry.
- "How does cfg get installed into the model" → `blb_rl_bridge.BLBNoiseRLBridge.apply()`.
- "How does the modulus chain see RL choices" → `RescaleOptimizerBridge.evaluate` + the two helpers `_strip_layer_suffix` and `cfg_to_t_new_from_table` in `rescale_optimizer_bridge.py`.
- "How is the optimizer's return mirrored back into cfg" → `apply_optimizer_output_to_cfg` (+ `sync_block2_qk_binding` for Block 2's Q/K binding) in `rescale_optimizer_bridge.py`; the env loop in `blb_stage2_rl/env.py:step` and `blb_stage2_rl/sequential_env.py:step` both call them.
- "Where's the per-block sequential RL path" → `blb_stage2_rl/sequential_env.py` (env), `…/sequential_policy.py` (actor-critic + buffer + GAE + ppo_update), `…/sequential_runner.py` (`train_sequential` driver). Schedule helpers in `action_space.py`: `step_schedule`, `splice_step_action_into_full_vec`, `step_schedule_max_dim`.
- "How is one candidate stored / ranked" → `blb_stage2_rl/candidate_store.py`.
- "Offline-evaluate a candidate without retraining" → `scripts/blb_eval_action.py`.
- "BLB Stage-2 RL docs" → `docs/BLB_stage2_rl_README.md` (launcher knobs) / `…_FULL_FLOW.md` (run-to-resume logic) / `…_INTERNAL_FLOW.md` (per-module call stack) / `…_spec.md` (design rationale, math). The 4 docs disambiguate themselves at the top.
