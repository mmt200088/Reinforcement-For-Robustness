# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research codebase that searches for an optimal noise / approximation configuration for **CKKS + MPC privacy-preserving inference** of BERT (and to a lesser extent GPT-2). Two search phases per task:

- **Stage 1** — picks GELU / Softmax polynomial-approximation degrees per layer.
- **Stage 2 (BLB)** — picks per-layer scaling factors for every CKKS noise point in 5 blocks (encode / fresh / rescale, plus rotation flags) plus a per-block MPC↔HE truncation `k`. The new "BLB Stage 2 RL" (`blb_v3`) is the final/canonical implementation; the old single-N implementation (`legacy_v2` in `noise_rl_module_v2.py`) is kept only for reproducing past experiments.

Both stages are PPO; a GA path (`genetic_search_module.py`) and a greedy path (`greedy_search_module.py`) exist as alternatives. After search, results go through `final_evaluation_module.py` (`UnifiedFinalEvaluationModule`, the merged Stage-1 + Stage-2 final eval).

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
   → action_vec  (numpy int, ~94 dims/layer × L layers + 1)
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

### Persistent directories (two distinct trees)

- **Old stage-2 RL + Stage-1 RL + GA + general-RL**: `rl_results/persistent/{algorithm}/{model}/{dataset}/{accuracy_slug}/...` (`accuracy_slug` is e.g. `s1t0.005_s2t0.05_s2st0.05`). Same parameters → same dir → auto-resume. See `docs/ARCHITECTURE.md` §4.
- **BLB Stage-2 RL (blb_v3)**: `Parting Chapter/<run_basename>/blb_stage2/progress/`. The runner overrides `evaluator.noise_stage_progress_dir` at the start of `run()` so all BLB checkpoints / status board / curves / final report land here, isolated from legacy. See `resolve_blb_persistence_dir()` in `blb_stage2_rl/runner.py`.

In each BLB run dir you'll find: `blb_stage2_rl_checkpoint_{live,final}.pt`, `blb_stage2_best_cfg.pkl`, `blb_stage2_status.json` (atomically rewritten — safe to `tail -f` / `cat`), `blb_stage2_training_curve.{npz,png}`, `blb_stage2_report.md`, plus `blb_stage2_error.txt` if the loop crashed.

### Graceful stop / resume

Stage-2 RL (both variants) honors **SIGINT** and a stop-flag file (`STOP_RL` in the progress dir). The next PPO update boundary saves a checkpoint then exits with code 0. Re-running the same launcher invocation auto-resumes; the BLB runner restores PPO net + optimizer + episode counter + best reward + `episode_returns` + RNG state.

### Rescale_optimizer integration

`rescale_optimizer_bridge.py` wraps the local `Rescale_optimizer/rescale_optimizer/` package (graph + `replan_with_user_actions`). Four invoker kinds:

- `InProcessInvoker` (recommended) — `import rescale_optimizer`, ms per call. Use `InProcessInvoker.from_profile(rescale_optimizer_root="Rescale_optimizer", profile="mrpc")`.
- `SubprocessInvoker` — forks `python scripts/replan_what_if.py`; hundreds of ms per call; for debug isolation.
- `StubInvoker` — canned responses; for unit tests.
- `HeuristicStubInvoker` (`blb_stage2_rl/default_invoker.py`) — bridge fallback when `Rescale_optimizer` isn't available; reward stays monotonic so PPO can still learn, but **it is not the real modulus-chain cost**.

`build_optimizer_requests(profile, cfgs_dict)` produces `{"block1_mrpc_L0": ("block1", cfg), ...}`; the bridge handles the rest. To support a profile beyond `mrpc`, extend `default_block{1..5}_cfg_to_delta` (graph-node names) and `DEFAULT_CFG_TO_T_NEW_MAP` (skeleton-position → cfg field).

## Conventions worth knowing

- **Don't directly call `rl_tune*.py`**. The launcher does conflict checks (e.g. `legacy_v2` rejects BLB-only flags), generates the persistent slug, and creates `LATEST_PID` / `LATEST_RUN_DIR` markers under `rl_results/persistent/`.
- **First time for a parameter combo always needs `--fresh`**. The launcher refuses to start otherwise to prevent accidental overwrites.
- Logs/curves/checkpoints under `rl_results/` are mostly gitignored; un-ignored exceptions are explicit in `.gitignore` (e.g. `pruning_search_log.txt`, `persistent/**/*.csv`). Don't add new untracked artifact patterns blindly.
- The Windows console may be GBK; `BLBStage2RLRunner._make_log_safe` wraps `evaluator.log` so non-GBK chars fall back without crashing stdout (file logs stay UTF-8). Matplotlib plot titles are intentionally ASCII (the markdown report carries the Chinese). If you add new console output that may include unicode bullets (▸), they will be replaced with `?` in stdout but preserved in log files.
- `GLOBALS.md` lists where global path / hyperparameter constants live. `config/paths.py` and `config/constants.py` exist as the future single source of truth, but most modules still hardcode their own — change with care.

## When you're investigating something specific

- "Where is X persisted / where do logs go" → `docs/ARCHITECTURE.md` §4, then `config/paths.py`.
- "How does the launcher decide it's a resume" → `llama_7B_LayerImportance.sh` (large; search for `accuracy_slug`).
- "What action dim corresponds to what cfg field" → `blb_stage2_rl/action_space.py` top half (`_BLOCK{1..5}_FIELDS`).
- "How does cfg get installed into the model" → `blb_rl_bridge.BLBNoiseRLBridge.apply()`.
- "How does the modulus chain see RL choices" → `RescaleOptimizerBridge.evaluate` + the two helpers `_strip_layer_suffix` and `cfg_to_t_new_from_table` in `rescale_optimizer_bridge.py`.
- "BLB Stage-2 RL spec / design rationale" → `docs/BLB_stage2_rl_spec.md`; runtime flow → `docs/BLB_stage2_rl_FULL_FLOW.md`; user-facing knobs → `docs/BLB_stage2_rl_README.md`.
