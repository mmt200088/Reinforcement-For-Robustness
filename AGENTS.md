# AGENTS.md

This file is the project-level working memory for Codex in this repository.
It should describe the code as it exists, not as older docs or comments once
described it. When this file conflicts with current code, verify with the code
and update this file.

## Operating Discipline

For future work in this repository, follow the local `karpathy-guidelines` and
`grill-me` skills as standing collaboration rules:

- Think before coding: state assumptions, expose uncertainty, prefer the simple
  solution, and define verifiable success criteria.
- Make surgical changes only. Do not refactor, clean up, or delete unrelated
  code unless explicitly asked.
- For plans, designs, or ambiguous implementation choices, grill one decision
  at a time. If code can answer the question, inspect the code instead of asking
  the user. When asking, include the recommended answer.
- For coding tasks, implement only what was requested and verify with the
  narrowest meaningful command or test.
- For the current Stage-2 RL collapse task, operate in goal mode rather than
  one-shot bugfix mode. The goal is not just "tests pass"; RL must train after
  the anchor without collapse, the reward curve must look like a normal RL
  curve, terminal metrics must not hit collapse sentinels such as
  `loss_mean=100`, priority must not enter sustained P1(acc), and monitored
  parameters must not jump pathologically. If a server run shows a new abnormal
  point, design the next experiment, inspect evidence, apply the real fix
  locally, and repeat the git-synced server run loop.
- Keep this `AGENTS.md` current as the shared project memory for Codex and
  Claude Code. After each user message that adds or changes project facts,
  workflow rules, run state, architecture notes, or operating constraints,
  update this file before finishing the turn.

### Local/Git/Server Workflow

Code changes must be made locally first. The server is for running jobs and
producing results only.

Required flow:

1. Edit code in the local workspace.
2. Commit/push the local changes to git.
3. On the server, pull from git before running.
4. Run training/evaluation on the server.
5. Push generated results/artifacts from the server to git.
6. Pull those results back into the local workspace.

Do not directly patch source code on the server except for emergency inspection
or a throwaway diagnostic that will not be kept. Any real fix must be applied
locally, pushed to git, then pulled by the server.

Collaboration protocol for future Codex + Claude Code work:

- Codex and Claude Code may both help modify this repository, but canonical
  source edits happen only in the local workspace.
- The server must not be used as a source-editing workspace. Do not edit,
  format, patch, or commit `.py`, launcher, config, test, or documentation
  source there unless the user explicitly changes this protocol.
- The server may only pull code from git, run commands/experiments, produce
  logs/checkpoints/reports/results, and push or hand back those generated
  artifacts.
- Normal synchronization is git-only: local source edit -> local commit/push ->
  server pull -> server run -> server push generated artifacts/results -> local
  pull.
- If a server-side run exposes a code bug, document the diagnosis and reproduce
  the real fix locally; do not keep a server-side source patch as canonical.

### Server Command Bridge

Use `SERVER_COMMAND.md` as the normal bridge for server-side command execution.
The server-side agent watches that file, reads the first fenced `bash` code
block under the active command section, and runs it from the repository root.

When a server run is needed:

1. Edit `SERVER_COMMAND.md` locally.
2. Put the exact command to run in the first fenced `bash` code block.
3. Update the human-readable metadata/checklist below it when useful.
4. Commit and push the file.
5. Let the server agent pull/sync and run the command.

Do not SSH in just to launch routine training/evaluation commands. Do not use
the server bridge to edit source code; source changes still follow the local
edit → git push → server pull flow above.

### New GPUShare Server State

As of 2026-05-19, a new GPUShare server was prepared for this project at
`ssh -p 46587 root@i-1.gpushare.com`. Do not store the password in any config
or project file.

Current verified server facts:

- OS/container: Ubuntu 22.04.5 style container environment, no systemd.
- GPUs: 2x NVIDIA GeForce RTX 5090, driver 580.159.03, CUDA runtime visible.
- Work directory: `/hy-tmp/Reinforcement-For-Robustness`.
- Checkout: sparse `jk_standard_rl` clone from
  `https://github.com/mmt200088/Reinforcement-For-Robustness.git` at commit
  `a28d837`.
- Runtime/cache env used for successful runs:
  `HF_HOME=/hy-tmp/hf_cache`, `HF_ENDPOINT=https://hf-mirror.com`,
  `HF_HUB_DISABLE_XET=1`, `GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data`.
- Python environment: system Python 3.11.12 with PyTorch 2.9.1+cu128. The
  project needed `transformers==4.44.2` because newer 4.57.x rejects
  `TrainingArguments(evaluation_strategy=...)`.
- GitHub HTTPS transport on this server needs repo-local Git settings:
  `git config --local http.version HTTP/1.1` and
  `git config --local protocol.version 0`. Without them, `git pull` can fail
  with `RPC failed; curl 16 Error in the HTTP2 framing layer` and
  `fatal: expected flush after ref listing`.

### Two-GPU Reward-Probe Parallelism

The server has two visible GPUs:

- GPU 0: NVIDIA GeForce RTX 5090, about 32607 MiB.
- GPU 1: NVIDIA GeForce RTX 5090, about 32607 MiB.

The two-GPU optimization is not two independent RL jobs. The target is still
one RL job where, after the policy selects one BLB action, the model-forward
reward probe trials for that same action run concurrently across both GPUs.
This should accelerate the repeated inference tests used to compute the PPO
reward for one action.

Current implementation facts:

- `--stage2-k-trials` controls the number of Stage-2 reward noise trials. It
  defaults to 5 and maps into `BLBStage2TrainConfig.num_trials_per_step`.
- Enable the parallel reward probe with `--blb-v3-reward-devices 0,1` plus
  `CUDA_VISIBLE_DEVICES=0,1`. Leaving `--blb-v3-reward-devices` unset preserves
  the original single-GPU code path.
- `blb_stage2_rl/probe_runner.py::parse_device_ids(...)` accepts all launcher
  forms observed in practice: `"0,1"`, Python Fire tuple `(0, 1)`, list
  `[0, 1]`, int `0`, and stringified `"(0, 1)"`/`"[0, 1]"`. Invalid non-empty
  specs raise instead of silently falling back to single GPU.
- `BLBStage2RLRunner._build_train_config_from_evaluator(...)` fills
  `BLBStage2TrainConfig.reward_devices`. `sequential_runner.py` attaches a
  `ProbeRunner` when that list has at least two devices and logs
  `[multi-gpu] reward probe enabled: devices=[0, 1]`.
- `BLBStage2Env.step(...)` applies the selected BLB config, installs that same
  decoded action on every `ProbeRunner` worker, then calls
  `self._eval_on_probe(self.env_cfg.num_trials_per_step)`.
- `BLBStage2Env._eval_on_probe(k_trials)` delegates to `ProbeRunner.run_trials`
  when a runner is attached. The runner splits trials round-robin. For the
  default five trials on two GPUs, GPU 0 runs `[0, 2, 4]` and GPU 1 runs
  `[1, 3]`, then returns results in trial order for the existing aggregation.
- Sequential RL terminal reward reaches the same path through
  `BLBStage2SequentialEnv` -> assembled full action vector ->
  `BLBStage2Env.step(...)`. Per-block dense optimizer shaping is not the target
  for GPU parallelism; only the terminal/full model-forward reward probe is.

Implementation constraints to preserve:

- Preserve one PPO learner, one action stream, one persistent run directory,
  and one reward per selected action.
- Do not solve this by running two separate launcher processes with different
  `--run-tag` values; that tests different actions/seeds and does not speed up
  a single action's reward.
- Do not assume `CUDA_VISIBLE_DEVICES=0,1` alone is enough. PyTorch
  `torch.device("cuda")` means the first visible GPU unless the reward probe
  explicitly places model copies and batches on both devices.
- Do not share one mutable `model`/`BLBNoiseRLBridge` instance across two GPUs.
  Worker 0 reuses the env model/bridge on `cuda:0`; worker 1 deep-copies the
  model to `cuda:1`, builds its own handler/bridge, and moves probe batches.
- Avoid reloading the HuggingFace model for every action. `ProbeRunner` workers
  are initialized once per run and reused across action evaluations.
- Keep the probe dataset fixed across trials exactly as today. Only the
  independent noise RNG seeds differ per trial.
- Trial seeds are deterministic from the per-action base seed and trial index
  inside `probe_runner._trial_seed(...)`.
- Preserve the invalid-chain shortcut: if `Rescale_optimizer` reports
  `any_invalid`, skip model-forward reward as current code does. Do not spend
  GPU work on invalid candidates.
- Baseline/noisy preflight that calls `_eval_on_probe(k)` should use the same
  two-GPU trial runner so baseline std and candidate std have the same
  semantics.
- Keep enough diagnostics to prove both cards are used: visible devices, reward
  probe device list, trial split, per-device elapsed time, and worker lines.

User-facing config for two-GPU Stage-2 reward probing:

```bash
--blb-v3-reward-devices 0,1
--stage2-k-trials 5
```

The expected server command is still one launcher run, for example:

```bash
cd /hy-tmp/Reinforcement-For-Robustness
git pull --ff-only

export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

CUDA_VISIBLE_DEVICES=0,1 bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --stage2-k-trials 5 \
  --blb-v3-reward-devices 0,1 \
  --fresh
```

Verification checklist:

- A smoke run logs `[multi-gpu] reward probe enabled: devices=[0, 1]`,
  `[probe-runner] worker 0: cuda:0`, and
  `[probe-runner] worker 1: cuda:1`.
- `nvidia-smi` shows both GPUs active during the model-forward reward probe.
- The metrics aggregation still uses all 5 trials for each action.
- Single-GPU fallback remains valid when only one GPU is visible or
  `--blb-v3-reward-devices` is unset.

Latest server check on 2026-05-19 after fixing the Fire tuple parsing path:
two 200-episode benchmark runs completed successfully. Single GPU took `601s`;
dual GPU took `406s`, a measured `1.48x` speedup and `195s` wall-clock saving.
The dual run log contains:

```text
[multi-gpu] reward probe enabled: devices=[0, 1]
[multi-gpu] [probe-runner] worker 0: cuda:0 (primary, reusing env.bridge)
[multi-gpu] [probe-runner] worker 1: cuda:1 (deepcopy replica)
```

`nvidia-smi` sampling showed dual-run GPU 0 and GPU 1 both active, with max
utilization `99%` on each and GPU 1 max memory about `3732 MiB`. The 200-episode
benchmark is performance/plumbing evidence only, not a claim about final RL
quality. Real Stage-2 RL quality still needs long runs around 50,000+ episodes.
Report:
`experiments/server_command_runs/stage2_reward_probe_fix_benchmark_20260519_211827/stage2_reward_probe_fix_benchmark_report.html`.

The earlier pre-fix 2026-05-19 benchmark remains useful as negative evidence:
single GPU `601s`, dual GPU `601s` (`1.00x`), no multi-GPU activation log, and
GPU 1 at `0%` utilization. Report:
`experiments/server_command_runs/stage2_reward_probe_benchmark_20260519_202236/stage2_reward_probe_benchmark_report.html`.

Latest focused action-to-config chain check on 2026-05-20: server HEAD
`c24d5b8` passed 21/21 focused tests covering optimizer output write-back,
fused-away rescale handling, Block 2 Q/K sync, live cfg reads during noise
sampling, all-max optimizer validity, and action-description slot semantics.
Report:
`experiments/server_command_runs/action_config_chain_20260520_015951_c24d5b8/action_config_chain_test_report.html`.

Latest full contract-gate rerun on 2026-05-20: server HEAD `26fe463` passed
the complete command `BLB_STRICT=0 python -m unittest discover -s tests -p
"test_blb_*.py" -v`, with `101` tests run, `0` failures, and `0` errors. This
is the rerun of the older red gate that had `99` tests with `8` failures and
`1` error. Report:
`experiments/server_command_runs/full_contract_gate_20260520_021220_26fe463/full_contract_gate_report.html`.

`SERVER_COMMAND.md` was extracted and launched once on this server. It reached
real BLB Stage-2 sequential RL execution, wrote diagnostics under
`Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/`,
and was stopped at the user's request. No project source code was edited on the
server; server git changes were generated run artifacts/logs only.

After the stopped run, those generated artifacts were mirrored back locally and
pushed to `origin/jk_standard_rl` as commit `20ee2c1`. The server checkout may
still show the same generated artifacts as local modifications until it is
synced to the pushed commit; do not treat them as source changes.

Current user/agent responsibility split:

- The user edits research code locally and pushes those code changes to git.
- Codex/Claude on the server should pull from git, run the requested experiment,
  collect generated artifacts/results, and push or hand back those results.
- Do not proactively patch `.py`, launcher, config, or test source files on the
  server. If a server-only diagnostic discovers a required source fix, document
  it and let the local code-editing agent make and push the real change.

## What This Project Is

This is a research codebase for searching noise and approximation schedules for
CKKS + MPC privacy-preserving inference of BERT, and to a smaller extent GPT-2.
The searched system is a plaintext PyTorch simulation with injected noise and
fixed-point truncation at protocol-relevant points; it is not real ciphertext
execution.

There are two search stages:

- Stage 1 picks per-layer GELU and Softmax polynomial approximation degrees.
- Stage 2 picks BLB CKKS scale/truncation schedules for five fixed blocks per
  transformer layer.

Stage 1 and Stage 2 are PPO-based by default. GA and greedy alternatives exist
in `genetic_search_module.py` and `greedy_search_module.py`. The canonical Stage
2 path is `blb_v3`; `legacy_v2` in `noise_rl_module_v2.py` is kept for older
experiment reproduction.

## Critical Mental Model

1. Plaintext simulation only. The model forward is still fp32 PyTorch. CKKS
   encode/fresh/rescale/rotation and MPC truncation are simulated through
   Gaussian noise or fixed-point truncation.
2. BLB operations are fixed. RL chooses action indices for already-required
   slots. A mask is an index mask for allowed categorical values, not an
   operation mask.
3. Actions are integer indices, not scale values. SF slots decode with
   `sf_from(idx, max_sf, levels) = max_sf - 2 * (levels - 1 - idx)`. K slots
   decode through `K_LEVELS`.
4. Slot kinds matter: `F`, `W`, `M`, `S`, `R`, and `K` have different
   semantics and noise distributions. Do not collapse them into plain numbers.
5. Rotation has no independent action. Rotation scale is inherited from the
   current scale after the optimizer-set rescale state. If the optimizer fuses
   away a rescale, the trailing rotation must follow the optimizer result.
6. Reward is hard-priority: model accuracy first, model stability second, then
   cost. `Rescale_optimizer` contributes optimizer cost / feasibility
   diagnostics only; it must not skip or replace the actual model forward reward.
   Cost must never compensate for an accuracy or stability failure.
7. `Rescale_optimizer` is the source of truth for modulus-chain cost and
   optimizer feasibility diagnostics. `HeuristicStubInvoker` was deleted;
   training and promotable final evals must use real `replan_with_user_actions`
   through `InProcessInvoker` or an explicitly real subprocess path.
8. The first HE config is treated as lossless. Layer 0 Block 1 is reserved in
   the action vector but not installed; `first_input_sf` is a deprecated
   compatibility tail slot and is not installed.

## Current BLB Action Space

Do not trust stale comments at the top of `blb_stage2_rl/action_space.py`.
Current field tables are compacted:

- Block 1: 7 slots per layer.
- Block 2: 12 slots per layer.
- Block 3: 7 slots per layer.
- Block 4: 12 slots per layer.
- Block 5: 10 slots per layer.
- Total per-layer action width: 48.
- BERT-base full action vector width: `48 * 12 + 1 = 577`.
- Sequential episode horizon for 12 layers: `4 + (12 - 1) * 5 = 59`.

The old "59 required slots" wording refers to sequential `(layer, block)` steps,
not the full categorical action-vector width. Older docs/comments may still say
73/877, 94/1129, or describe a separate first-input noise point. Treat those as
stale unless `scripts/blb_export_action_registry.py` and
`describe_action_vector(...)` confirm them.

K decoding is non-monotonic by design. Default `K_LEVELS` is
`(8, 9, 11, 13, 10, 12)`. The all-max/baseline helper means max SF plus
per-block baseline K: Blocks 1/3/5 use K=13, Blocks 2/4 use K=10. Do not find a
K baseline by taking the largest index.

## Canonical Entrypoints

Training goes through the launcher. Do not call `rl_tune.py` or older
`rl_tune*.py` files directly.

```bash
bash llama_7B_LayerImportance.sh --list-presets
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl
bash llama_7B_LayerImportance.sh compare --dataset mrpc
bash llama_7B_LayerImportance.sh general train --general-rl-tasks mrpc,cola,rte,stsb --fresh
```

Use `--fresh` the first time for a parameter combination. Re-running the same
combination resumes from the persistent directory.

Standalone final eval uses the Paean wrapper:

```bash
bash Paean/run_final_eval.sh --preset mrpc-final-eval-only
bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed
bash Paean/run_final_eval.sh --preset mrpc-blb-baseline-fixed \
  --range block3.truncation=8,9,10,11,12,13 \
  --action-fixed layer2.block5.wffn1_sf=18
```

For BLB action final evals, pass a real invoker unless the preset already does:

```bash
--rescale-invoker-kind in_process \
--rescale-optimizer-root Rescale_optimizer \
--require-rescale-optimizer
```

`Paean/config.py` still defaults `--rescale-invoker-kind` to `heuristic`, but
`Paean/blb_action_eval.py` rejects heuristic at runtime. Any report that says
BLB final eval used `heuristic` is not promotable evidence.

## BLB Sidecar Tools

Sidecars ride alongside the launcher. Do not replace the launcher with them.

```bash
python scripts/blb_phase0_preflight.py
python scripts/blb_export_action_registry.py
python scripts/blb_eval_action.py ...
python scripts/blb_f0_scan_feasible_domain.py ...
python scripts/blb_orphan_slot_audit.py --profile mrpc
python scripts/blb_verify_noise_install.py --mode smoke --profile mrpc --num-layers 12
python scripts/blb_make_run_manifest.py ...
python tools/status_board.py --write-md
```

The registry exporter is the authority for action-index mapping, effective
slots, decoded values, and rotation-derived slots. The F0 tools require a real
local `Rescale_optimizer` import.

## Tests

Tests are plain `unittest` files, not pytest-configured:

```bash
python tests/test_blb_registry_artifact_consistency.py
python tests/test_blb_baseline_bootstrap.py
python tests/test_blb_optimizer_cost_consistency.py
python tests/test_blb_warmstart_resume.py
python -m unittest discover -s tests -v
```

Many BLB tests are torch-free. `test_blb_action_mask.py`,
`test_blb_stage2_rl_regressions.py`, and model-forward/final-eval paths require
torch/transformers. `test_glue_dataset_loading.py` also needs datasets plus a
populated GLUE cache or `GLUE_LOCAL_DATASET_DIR`.

## Architecture Map

```text
llama_7B_LayerImportance.sh
  -> rl_tune.py
    -> layer_importance_evaluator.LayerImportanceEvaluator
      -> Stage 1: GTrXL PPO for GELU/Softmax degrees
      -> Stage 2 blb_v3:
        -> blb_stage2_rl.runner.BLBStage2RLRunner
          -> sequential_runner.run_sequential_via_runner (default)
          -> BLBStage2Env / BLBStage2SequentialEnv
          -> BLBStage2SequentialPolicy
          -> action_space.action_vector_to_cfgs / step_schedule
          -> rescale_optimizer_bridge.RescaleOptimizerBridge
          -> blb_rl_bridge.BLBNoiseRLBridge.apply
      -> Stage 2 legacy_v2:
        -> noise_rl_module_v2.NoiseRLModuleV2
      -> final_evaluation_module.UnifiedFinalEvaluationModule
      -> Paean.blb_action_eval.BLBActionFinalEvaluationModule when a BLB action is present
```

`layer_importance_evaluator.py` and `noise_rl_module_v2.py` import each other
for legacy graceful-stop helpers and checkpoint constants. See `docs/GLOBALS.md`
before moving globals.

## Sequential BLB Stage 2 RL

Per-block sequential RL is the default path since 2026-05-15. `runner.py`
dispatches to `run_sequential_via_runner(...)` when `sequential_rl=True`, which
is the default in `BLBStage2TrainConfig`, `rl_tune.py`, and
`layer_importance_evaluator.py`.

The schedule is:

```text
L0:B2 -> L0:B3 -> L0:B4 -> L0:B5
L1:B1 -> L1:B2 -> ... -> L11:B5
```

Step 0 also carries the deprecated first-input tail slot only for vector
compatibility. Each nonterminal step calls `RescaleOptimizerBridge.evaluate` for
that one block and gives dense cost shaping. The terminal step assembles the
full 577-wide vector and calls the base env for model forward plus hard-priority
reward.

The old single-shot `BLBStage2Env`/`BLBStage2Policy` path still exists for tests,
F0 tooling, candidate-store compatibility, and explicit
`--blb-v3-no-sequential-rl` experiments.

Current safe-curriculum fix for the 2026-05-20 collapse at episode 121:

- The collapse was optimizer-valid but accuracy-catastrophic. The first
  post-anchor sampled episode reached `any_invalid=False` with `loss_mean=100`
  and P1(acc), so the optimizer-invalid blacklist alone could not protect the
  terminal model-forward reward.
- Sequential forced-baseline anchor must respect the configured
  `warmstart_anchor_episodes` unless `force_baseline_episodes` is explicitly
  set. Anchor and entropy schedules must use absolute episode indices so resume
  does not restart the anchor.
- During forced anchor, PPO must still evaluate baseline actions under
  unrestricted slot/level support; do not apply a baseline-only mask there,
  otherwise the actor receives no useful probability-mass signal.
- After the anchor, safe neighbor sampling may restrict each episode to a small
  set of mutable full-vector offsets. Non-selected slots stay baseline-only;
  selected SF-like slots can move only downward within the local radius, and K
  slots use value-order locality through non-monotonic `K_LEVELS`.
- Store the exact per-transition `action_level_mask` used during collection and
  replay it during `sequential_ppo_update`. Recomputing support during PPO
  update breaks the PPO ratio whenever the per-episode mask changes.
- Build mutable offsets from `describe_action_vector(...)` and exclude inactive
  compatibility slots, layer-0 block-1 pseudo slots, first-input compatibility,
  and single-level dimensions.
- A second 2026-05-20 finding: K=5 / probe_size=256 noisy probes made the
  all-max baseline itself occasionally fall one discrete probe sample below
  `noisy_baseline_metric1 - stage2_limit_tolerance`, producing false P1(acc)
  points with normal `loss_mean≈0.34` and `m1≈0.865-0.867`. Sequential accuracy
  threshold calibration must subtract a one-sample probe granularity guard
  (`1 / stage2_probe_size`) so baseline jitter is not reported as an error,
  while real collapses such as `m1≈0.31` still fail hard.

Important current gap: the single-shot runner and legacy v2 runner wire
`STOP_RL`/SIGINT graceful-stop handling through `noise_rl_module_v2.py`; the
current sequential runner does not expose a `STOP_RL` check in its own loop.
It does write live checkpoints and auto-resume state, but do not promise
`STOP_RL` support for sequential runs unless you add and verify it.

## Rescale_optimizer Integration

`rescale_optimizer_bridge.py` wraps the checked-in
`Rescale_optimizer/rescale_optimizer` package. `Rescale_optimizer/` is not a
git submodule.

Key wires:

- `InProcessInvoker.from_profile(...)` builds a `ReplanSession` over local graph
  configs and static baselines.
- `RescaleOptimizerBridge.evaluate(...)` strips `_L<i>` suffixes from layered
  RL names before calling the invoker. RL names look like `block1_mrpc_L3`; RO
  graph baselines are keyed like `block1_mrpc`.
- `auto_t_new_from_cfg=True` derives `t_new` from cfg SF fields using
  `DEFAULT_CFG_TO_T_NEW_MAP`.
- `apply_optimizer_output_to_cfg(...)` mirrors `new_compact_config` back into
  the action-decoded cfg, including snapped/repaired SFs, fused-away rescale
  points set to `None`, propagation deltas, and effective rotations.
- `sync_block2_qk_binding(cfg)` must run after optimizer overrides for Block 2,
  because action-space convention binds Q-side fields to K-side fields.

If supporting a new profile beyond MRPC, extend both graph-node mapping helpers
and `DEFAULT_CFG_TO_T_NEW_MAP`; do not rely on cfg-derived defaults silently.

## Static Skeleton Baseline

BLB Stage 2 baseline must come from
`Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json`.

For each layer, graph keys are selected from Stage 1 degrees:

- Block 1: `block1_<dataset>`; skipped for layer 0.
- Block 2: `block2_<dataset>`.
- Block 3: `block3_exp_n<softmax_degree[layer]>`.
- Block 4: `block4`.
- Block 5: `block5_n<gelu_degree[layer]>`.

`load_static_skeletons_baseline(...)` extracts fresh SFs, encode deltas, rescale
SFs, optimizer cost signals, and per-layer max-SF calibration.
`static_skeletons_baseline_to_action(...)` turns that into the baseline action.
Training must fail if the archive or a required graph key is missing. Do not
fallback to an estimated all-max baseline.

## Persistence

The launcher creates persistent run roots under:

```text
Parting Chapter/persistent/{algorithm}/{model}/{dataset}/{accuracy_slug}/
```

BLB v3 progress lives under the active run root:

```text
stage2_noise/progress/
```

`resolve_blb_persistence_dir(...)` is the source for this path. Common files:

- `blb_stage2_rl_checkpoint_live.pt`
- `blb_stage2_rl_checkpoint_final.pt`
- `blb_stage2_best_cfg.pkl`
- `blb_stage2_status.json`
- `blb_stage2_training_curve.npz`
- `blb_stage2_training_curve.png`
- `blb_stage2_report.md`
- `blb_stage2_best_action_full.{json,md}`
- `blb_stage2_baseline_action_full.{json,md}`
- `diagnostics/` and `details/` artifacts

Older docs that mention `blb_stage2/progress/` are stale for current code.

## Candidate Evidence and Fidelity

The active promotion ladder is:

- F0: optimizer-only. Decode action, call real `Rescale_optimizer`, collect
  validity, total bits, fusion count, and invalid-chain details. No model
  forward.
- F1: online small probe with MC trials during training. This is the PPO reward
  signal and catches obvious accuracy/stability failures.
- F4: final eval on full or near-full validation with real BLB action install.
  Only F4 evidence belongs in final "best" claims.

RL training is long-cycle. Based on prior runs, effective BLB Stage-2 RL
usually needs 50,000+ episodes/rounds. Short runs such as 200 episodes are for
plumbing, performance, and regression smoke only; do not treat their reward
quality as evidence that the RL search worked or failed.

F2/F3 may appear in old JSONL or older documentation but are no longer the
active promotion ladder.

`blb_stage2_rl/candidate_store.py` is canonical for persisted candidates. Store
raw and effective action hashes, action indices, decoded values, N,
distribution, block, operation, metrics, optimizer signals, and rank keys.
Never log only indices.

## Final Eval Routing

`LayerImportanceEvaluator.run_unified_final_eval(...)` dispatches to
`BLBActionFinalEvaluationModule` when one of these is present:

- `--action-config`
- `--range` / `--action-fixed`
- a stage2 search result containing `blb_v3_best_action_vec`

Otherwise it uses `UnifiedFinalEvaluationModule`, which can still evaluate
legacy Stage 2 noise configs or a legacy-style max config. When touching final
eval glue, verify that BLB best action is decoded, optimizer-adjusted, installed
through `BLBNoiseRLBridge.apply(...)`, and not silently replaced by legacy
all-max noise.

Standalone Paean mode does not run random/permutation/equivalent/budget
controls unless requested. The passive training-end preset `Paean/presets/default.conf`
does enable random comparison groups.

## Block Scope

- Block 1: post-FFN/GELU output, Wffn2, LayerNorm mean/variance head. Not
  installed at layer 0.
- Block 2: LayerNorm tail, Wq/Wk/Wv, QK BSGS masks and merge. Active at layer 0.
  Q-side fields are bound to K-side fields.
- Block 3: Softmax exponential approximation. Degree controls which square
  rescale slots are effective.
- Block 4: Softmax x V, Wo, post-attention LayerNorm head.
- Block 5: LayerNorm tail, Wffn1, GELU polynomial chain. GELU degree controls
  effective high-order slots.

Field-level truth lives in `action_space.py` plus registry export artifacts, not
in prose comments.

## Conventions

- Prefer launcher/preset workflows. The launcher validates skip-mode conflicts,
  builds persistent slugs, and writes `LATEST_PID`/`LATEST_RUN_DIR` markers.
- Use `--mode train|eval|stage2-only|stage1-only|search-only` instead of
  manually mixing skip flags.
- Multi-trial evaluation is required. A single noise trial is not evidence.
- Warmstart toward the static-skeleton baseline is a prior, not a restriction.
- GLUE loading is flaky over network. Prefer local caches via
  `GLUE_LOCAL_DATASET_DIR`, `GLUE_DATASET_DIR`, HF cache, or saved parquet.
- Console logs may pass through GBK on Windows. Keep console-facing text robust;
  file logs are UTF-8.
- Do not add broad artifact patterns to `.gitignore` blindly. Many reports and
  checkpoints are intentionally ignored; exceptions are explicit.
- This local checkout uses sparse-checkout. Keep `/experiments/` included
  alongside `/experiment/`; server-command run reports such as
  `experiments/server_command_runs/final_eval_llm_ist_results_2026-05-17.html`
  may be present in Git but invisible on disk if the sparse rule is missing.

## Hard Taboos

1. Do not publish or promote a result backed by heuristic or stub optimizer
   numbers.
2. Do not treat action masks as operation masks.
3. Do not add freestanding rotation SF actions.
4. Do not install layer 0 Block 1 or deprecated first-input noise.
5. Do not select a final best from raw PPO reward or one noise trial.
6. Do not let final eval fall back to legacy all-max when a BLB best action
   exists.
7. Do not remove "extra" slots just because a doc says a different count. Export
   the registry, classify required/effective/compat/inactive, then change code.
8. Do not mutate Block 2 K-side cfg fields without restoring Q/K binding.
9. Do not reintroduce `blb_stage2_rl/default_invoker.py` or heuristic training
   fallback.
10. Do not make large multi-module BLB rewrites without F0 and F1 checks between
    steps, and F4 before result claims.

## Investigation Guide

- Semantics and rationale: `project_understanding_blb_stage2_rl.md`.
- Launcher/resume logic: `llama_7B_LayerImportance.sh`; search for
  `accuracy_slug`.
- Persistent paths and globals: `docs/ARCHITECTURE.md`, `docs/GLOBALS.md`,
  `config/paths.py`.
- Action fields and decode: `blb_stage2_rl/action_space.py`,
  `scripts/blb_export_action_registry.py`.
- Sequential RL: `blb_stage2_rl/sequential_env.py`,
  `blb_stage2_rl/sequential_policy.py`, `blb_stage2_rl/sequential_runner.py`.
- Model noise install: `blb_rl_bridge.BLBNoiseRLBridge.apply(...)` and
  `function_handler.py`.
- Optimizer cost and cfg override: `rescale_optimizer_bridge.py`,
  especially `_strip_layer_suffix`, `cfg_to_t_new_from_table`,
  `apply_optimizer_output_to_cfg`, and `sync_block2_qk_binding`.
- Baseline bootstrap: `blb_stage2_rl/baseline_bootstrap.py` and
  `docs/blb_baseline_handover_protocol.md`.
- Reward: `blb_stage2_rl/reward.py`.
- Candidate persistence/ranking: `blb_stage2_rl/candidate_store.py`.
- Offline F0 eval/scan: `scripts/blb_eval_action.py`,
  `scripts/blb_f0_scan_feasible_domain.py`.
- Final eval: `Paean/run_final_eval.py`, `Paean/blb_action_eval.py`,
  `final_evaluation_module.py`.
