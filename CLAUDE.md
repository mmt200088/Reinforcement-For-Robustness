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
6. **Single-step episode.** One `env.step(action_vec)` produces the entire model's config across all layers × all 5 blocks × all slots, plus `first-input fresh` (layer-0 embedding entry, not inside any block). With horizon=1, GAE degenerates to `A = r − V(s)` — that's correct, not a bug.
7. **Reward is hard-priority, not weighted-sum.** `invalid → accuracy → stability → cost`. Cost reward must never offset an accuracy or stability violation. Final-best selection should use a tuple rank key `(invalid_flag, acc_violation, stab_violation, normalized_cost, …)`, not raw PPO reward.
8. **`Rescale_optimizer` is the source of truth for modulus-chain validity and cost.** The `HeuristicStubInvoker` keeps PPO learning when the real package is unavailable, but its numbers are NOT real chain cost — never publish a "best" that was confirmed only against the heuristic.

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

# Final eval only
bash llama_7B_LayerImportance.sh eval --dataset mrpc --algorithm rl \
  --config glue_final_configs_best_ppo.json --eval-repeat 50

# RL vs GA comparison from persistent dirs
bash llama_7B_LayerImportance.sh compare --dataset mrpc

# Cross-task general policy
bash llama_7B_LayerImportance.sh general train --general-rl-tasks mrpc,cola,rte,stsb --fresh
```

`--mode` is the safe wrapper for the various skip flags: `train` / `eval` / `stage2-only` / `stage1-only` / `search-only`. Old `--skip-*` flags still work but conflict-checked.

**Status board** (aggregates running RL/GA/general/compare jobs into one markdown):

```bash
python tools/status_board.py --write-md   # rewrites docs/STATUS.md
```

**BLB sidecar tools** (do NOT replace the launcher; they ride alongside it):

```bash
# Confirm entrypoints, env, data, paths before any long run
python scripts/blb_phase0_preflight.py

# Export the actual slot registry; reconcile against the user's required-59
python scripts/blb_export_action_registry.py
# → reports/blb_opt/phase1_registry/slot_registry_required59_or_mismatch.md

# Offline-evaluate a single candidate action (supports F0/F1 fidelity ladder)
python scripts/blb_eval_action.py ...
```

The candidate store (`blb_stage2_rl/candidate_store.py`) is the canonical place to persist `action_index + decoded_value + N + distribution + block + operation + metrics + rank_key`. Never log only the index.

**Tests** (plain `unittest` files, no pytest config):

```bash
# Single test file
python tests/test_blb_stage2_rl.py
python tests/test_rescale_optimizer_bridge_real.py
python tests/test_bridge_action_flow_real.py
python tests/test_blb_persistence.py

# All (best-effort)
python -m unittest discover -s tests -v
```

The four real-invoker tests above are independent of torch — they stub `function_handler` so they can exercise the `Rescale_optimizer` integration in isolation. `test_blb_stage2_rl.py` is the only one that needs torch + transformers.

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

- **Block 1** — post-FFN / GELU output / Wffn2 / LayerNorm mean-variance head. Slots: GELU output fresh, Wffn2 weight encode, mean/variance scalar encodes, several rescales, block-end K.
- **Block 2** — LayerNorm tail + Wq/Wk/Wv projections + QK BSGS/mask/merge. Watch for tied Wq/Wk/mask groups if the optimizer requires them — registry must mark `tied_group`.
- **Block 3** — Softmax exponential approximation `exp(x) ≈ (1 + x/2^n)^(2^n)`. Some `square` rescales become inactive when Softmax degree is low — registry's `effective` flag is authoritative.
- **Block 4** — Softmax × V, Wo, post-attention LayerNorm head.
- **Block 5** — LayerNorm tail + Wffn1 + GELU polynomial chain. High-order GELU coefficient/power rescales become inactive at low GELU degree.
- **first-input fresh** — layer-0 input from embedding; not inside any block but part of the action vector.

### Persistent directories (two distinct trees)

- **Old stage-2 RL + Stage-1 RL + GA + general-RL**: `Parting Chapter/persistent/{algorithm}/{model}/{dataset}/{accuracy_slug}/...` (`accuracy_slug` is e.g. `s1t0.005_s2t0.05_s2st0.05`). Same parameters → same dir → auto-resume. See `docs/ARCHITECTURE.md` §4.
- **BLB Stage-2 RL (blb_v3)**: `Parting Chapter/persistent/{algorithm}/{model}/{dataset}/{accuracy_slug}/blb_stage2/progress/`. The runner overrides `evaluator.noise_stage_progress_dir` at the start of `run()` so all BLB checkpoints / status board / curves / final report land inside the active persistent run directory. See `resolve_blb_persistence_dir()` in `blb_stage2_rl/runner.py`.

In each BLB run dir you'll find: `blb_stage2_rl_checkpoint_{live,final}.pt`, `blb_stage2_best_cfg.pkl`, `blb_stage2_status.json` (atomically rewritten — safe to `tail -f` / `cat`), `blb_stage2_training_curve.{npz,png}`, `blb_stage2_report.md`, plus `blb_stage2_error.txt` if the loop crashed.

### Graceful stop / resume

Stage-2 RL (both variants) honors **SIGINT** and a stop-flag file (`STOP_RL` in the progress dir). The next PPO update boundary saves a checkpoint then exits with code 0. Re-running the same launcher invocation auto-resumes; the BLB runner restores PPO net + optimizer + episode counter + best reward + `episode_returns` + RNG state.

### Rescale_optimizer integration

`rescale_optimizer_bridge.py` wraps the local `Rescale_optimizer/rescale_optimizer/` package (graph + `replan_with_user_actions`). Four invoker kinds:

- `InProcessInvoker` (recommended; **required for publishable results**) — `import rescale_optimizer`, ms per call. Use `InProcessInvoker.from_profile(rescale_optimizer_root="Rescale_optimizer", profile="mrpc")`.
- `SubprocessInvoker` — forks `python scripts/replan_what_if.py`; hundreds of ms per call; for debug isolation.
- `StubInvoker` — canned responses; for unit tests.
- `HeuristicStubInvoker` (`blb_stage2_rl/default_invoker.py`) — bridge fallback when `Rescale_optimizer` isn't available; reward stays monotonic so PPO can still learn (see mental-model item 8 for the constraint).

`build_optimizer_requests(profile, cfgs_dict)` produces `{"block1_mrpc_L0": ("block1", cfg), ...}`; the bridge handles the rest. To support a profile beyond `mrpc`, extend `default_block{1..5}_cfg_to_delta` (graph-node names) and `DEFAULT_CFG_TO_T_NEW_MAP` (skeleton-position → cfg field).

## Verification: F0–F4 fidelity ladder

Don't grade a candidate at one fidelity. Climb the ladder:

- **F0** — optimizer-only: decode action, call `Rescale_optimizer`, collect `valid / total_bits / fusion_count`. No model forward. For registry checks, sensitivity scans, cheap candidate filtering.
- **F1** — small probe, low trial count: catch obvious accuracy collapses cheaply.
- **F2** — medium probe, more trials: validate F1 winners aren't lucky; check `loss_std` and `metric_min`.
- **F3** — confirmation: large probe, more trials, multiple seeds. Required before promoting an incumbent.
- **F4** — final eval: full/near-full validation set, real BLB install, frozen report. Only F4 numbers belong in "best" claims.

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

1. Declaring a final best from a `HeuristicStubInvoker` run (the heuristic is for monotonic-reward training only).
2. Selecting "best" from a single noise trial — final-eval must be MC-repeated.
3. Letting "final eval" silently evaluate a legacy all-max baseline instead of the BLB best.
4. Replacing the launcher entrypoint instead of riding alongside it (sidecars only).
5. Large multi-module rewrites without F0/F1/F2 verification between steps.

## When you're investigating something specific

- "What's the project actually optimizing, semantically" → `project_understanding_blb_stage2_rl.md` (canonical conceptual reference).
- "Where is X persisted / where do logs go" → `docs/ARCHITECTURE.md` §4, then `config/paths.py`.
- "How does the launcher decide it's a resume" → `llama_7B_LayerImportance.sh` (large; search for `accuracy_slug`).
- "What action dim corresponds to what cfg field" → `blb_stage2_rl/action_space.py` top half (`_BLOCK{1..5}_FIELDS`), then `scripts/blb_export_action_registry.py` for the authoritative registry.
- "How does cfg get installed into the model" → `blb_rl_bridge.BLBNoiseRLBridge.apply()`.
- "How does the modulus chain see RL choices" → `RescaleOptimizerBridge.evaluate` + the two helpers `_strip_layer_suffix` and `cfg_to_t_new_from_table` in `rescale_optimizer_bridge.py`.
- "How is one candidate stored / ranked" → `blb_stage2_rl/candidate_store.py`.
- "Offline-evaluate a candidate without retraining" → `scripts/blb_eval_action.py`.
- "BLB Stage-2 RL docs" → `docs/BLB_stage2_rl_README.md` (launcher knobs) / `…_FULL_FLOW.md` (run-to-resume logic) / `…_INTERNAL_FLOW.md` (per-module call stack) / `…_spec.md` (design rationale, math). The 4 docs disambiguate themselves at the top.
