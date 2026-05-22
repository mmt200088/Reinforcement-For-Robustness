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
7. **Reward is hard-priority, not weighted-sum.** `invalid/accuracy → stability → cost`. Cost reward must never offset an accuracy or stability violation. Final-best selection should use a tuple rank key `(invalid_flag, acc_violation, stab_violation, normalized_cost, …)`, not raw PPO reward. Reward v3 uses metric1 + metric2 gates and includes metric1_std/metric2_std in the stability gate, but those metric-std channels must tolerate normal small-K MRPC probe quantization; historical K=5 evidence remains useful, while current four-GPU runs use K=4. Do not use a tiny `1e-3` metric-std floor: the 2026-05-20 reward-v3 run at commit `6f3d618` failed at 345 episodes with P1=0, invalid=0, loss-cap=0 solely because normal metric-std jitter dropped 58 otherwise healthy episodes into P2 and pushed rolling300 below 35. Current behavior keeps tiny metric std jitter in P3 via a `1e-2` floor while still treating materially large metric std as P2. Current sequential Stage-2 cost reward is adaptive scalar: only P3 candidates receive cost reward; fusion gain and truncation/K gain receive clear interval-style boosts, while total bits is a weaker linear tie-breaker. `ParetoCostArchive` can still record frontier diagnostics and exploration stats, but Pareto event shaping is not the default PPO scalar reward.
8. **Invalid-chain pruning is layered, not a shape change.** Sequential rollout still keeps the full action vector and policy/critic head shapes. Exact invalid step-action tuples go into `ForbiddenActionMask`; repeated invalid evidence for a specific `(layer, block, slot, level)` goes into `EmpiricalInvalidLevelMask`, which intersects the current per-step `action_level_mask`. It always preserves the static baseline and current base/frontier proposal levels. Use the new diagnostics fields (`samples_rejected_by_mask`, `samples_rejected_by_optimizer`, `steps_fallen_back_to_baseline`, `forbidden_mask_total`, `empirical_invalid_level_disabled`, `rejection_optimizer_wall_seconds`) to judge whether this actually reduces runtime.
9. **`Rescale_optimizer` is the source of truth for modulus-chain validity and cost.** Every reward number, including per-step ones in the sequential path, must come from a real `replan_with_user_actions` (via `ReplanSession`). The legacy `HeuristicStubInvoker` was deleted on 2026-05-14; `InProcessInvoker.from_profile(...)` is hardcoded for training. If the real package fails to load, training aborts (no fallback).

The **"59 required slots"** is the user-stated target. Older `action_space.py` field tables export ~73 fields/layer and stale doc comments still say 94. **Trust `scripts/blb_export_action_registry.py` over comments** before changing slot counts; classify discrepancies as `required / effective-extra / compat-extra / inactive` rather than deleting.

## Local/Git/Server workflow

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

Current Stage-2 RL collapse goal mode:

- Treat this as a long research/debugging loop, not a one-shot bugfix. The task
  is complete only when server evidence shows BLB Stage-2 sequential RL can
  train after the anchor without reward collapse.
- Required success signal: the reward curve remains a normal RL curve,
  terminal metrics avoid collapse sentinels such as `loss_mean=100`, priority
  does not enter sustained P1(acc), and monitored quantities such as valid
  steps, PPO entropy/clip fraction, mutation count, radius, and GPU reward
  probe state do not jump pathologically.
- If a server run shows a new abnormal point, design a focused experiment,
  inspect the evidence, apply the real fix locally, push, let the server pull
  and run again, and continue until the goal is met.
- Assume this may be a long research task requiring many cycles over many
  hours. If the issue is a difficult training/search behavior rather than a
  straightforward code bug, do not stop after one or two patches. Form
  falsifiable hypotheses, run focused experiments, study the reward curve and
  diagnostics, tune or repair locally, and keep repeating the git-synced
  server-run loop until the evidence is clean.
- The active next milestone is a 60,000-episode Stage-2 RL search. This is
  not merely a longer smoke test: monitor whether reward keeps improving,
  whether entropy/clip_fraction collapse into a narrow search, whether
  safe-neighbor mutation/radius coverage expands, and whether cost signals
  improve without accuracy/stability regressions. Use online watchdog evidence
  during the run instead of waiting hours to discover a bad curve.
- For the long 60k curve, judge health by rolling reward averages and
  frequency/sustain of abnormal episodes, not by requiring every single episode
  to be clean. Occasional negative reward spikes or isolated P1(acc) episodes
  are acceptable if they do not become frequent or consecutive and the rolling
  reward windows do not collapse. The online watchdog should treat sparse
  `loss_mean=100` loss-cap spikes as warnings, not hard failures; hard-stop only
  on bursts such as consecutive loss caps or at least 5 loss caps in the latest
  100 post-anchor episodes.
- Decision boundary for this goal: make small corrective changes autonomously
  when the evidence supports them, including hyperparameter tuning, watchdog
  threshold changes, narrow diagnostic instrumentation, and focused bug fixes
  that preserve the current architecture and artifacts. Ask the user before
  major architectural/rewrite decisions, especially changes that invalidate the
  current Stage-2 setup, replace the reward/search formulation, rewrite large
  modules, or make earlier artifacts/checkpoints no longer interpretable.
- First 10k attempt evidence, 2026-05-20: the `3000/16/radius3` neighbor
  schedule reached reward around 42 but produced a P1 cluster at episodes
  1699-1757 once safe-neighbor reached `radius=2` with 8-9 mutated offsets.
  Through `radius=1`, P1 was 0 and reward improved. The current
  guarded-radius2 follow-up keeps raw safe-neighbor at
  `NEIGHBOR_MAX_RADIUS=1` with `ANCHOR_EPISODES=60`, `NEIGHBOR_RAMP=1800`,
  `NEIGHBOR_MAX_MUTATIONS=12`, `ENT_COEF=0.06`, `ENT_RAMP=600`, and
  `WARMSTART_BIAS_GAIN=1.2` with a decaying baseline prior. It enables radius2 only after frontier stall plus
  clean recent health, using defaults `GUARDED_RADIUS2_ENABLED=1`,
  `GUARDED_RADIUS2_MIN_EPISODE=1060`,
  `GUARDED_RADIUS2_STALL_WINDOW=600`,
  `GUARDED_RADIUS2_MAX_MUTATIONS=4`,
  `GUARDED_RADIUS2_EPISODE_FRACTION=0.15`, and
  `GUARDED_RADIUS2_COOLDOWN_EPISODES=300`. Do not replace this with raw
  default radius2.

### Server command bridge

Use `SERVER_COMMAND.md` as the normal bridge for server-side command execution.
The server-side agent watches that file, extracts the first fenced `bash` code
block under the active command section, and runs it from the repository root.

When a server run is needed, edit `SERVER_COMMAND.md` locally, put the exact
command in the first fenced `bash` code block, update the human-readable
metadata/checklist if useful, then commit and push. Let the server agent
pull/sync and run it. Do not SSH in just to launch routine training/evaluation
commands, and never use the server bridge to edit source code.

### GPUShare server state

As of 2026-05-19, the prepared GPUShare server is reachable at
`ssh -p 46587 root@i-1.gpushare.com`. Do not store the password in repo files,
shell profiles, or SSH config.

Verified server facts:

- OS/container: Ubuntu 22.04.5 style container environment, no systemd.
- GPUs: 2x NVIDIA GeForce RTX 5090, driver 580.159.03.
- Work directory: `/hy-tmp/Reinforcement-For-Robustness`.
- Checkout: sparse `jk_standard_rl` clone from
  `https://github.com/mmt200088/Reinforcement-For-Robustness.git`; the original
  server clone was at commit `a28d837`.
- Runtime/cache env used for successful runs:
  `HF_HOME=/hy-tmp/hf_cache`, `HF_ENDPOINT=https://hf-mirror.com`,
  `HF_HUB_DISABLE_XET=1`, `GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data`.
- Old GPUShare server Python environment was system Python 3.11.12 with
  PyTorch 2.9.1+cu128 and CUDA available on 2 GPUs.
- New GPUShare server at `ssh -p 30054 root@i-2.gpushare.com` was verified on
  2026-05-21 with 4x NVIDIA GeForce RTX 4090, about 48 GiB each, and system
  PyTorch 2.9.1+cu128 seeing all 4 GPUs. Do not downgrade PyTorch there unless
  the runtime breaks; install missing Python deps with `pip install -r
  requirements.txt`.
- `requirements-torch-cu124.txt` and `scripts/setup_cuda124_env.sh` remain an
  optional CUDA 12.4 fallback path. The normal new-server path should preserve
  the working `torch==2.9.1+cu128` runtime.
- Keep `transformers==4.44.2` unless the code is updated, because 4.57.x
  rejects `TrainingArguments(evaluation_strategy=...)`.
- GitHub HTTPS transport on this server needs repo-local Git settings:
  `git config --local http.version HTTP/1.1` and
  `git config --local protocol.version 0`. Without them, `git pull` can fail
  with `RPC failed; curl 16 Error in the HTTP2 framing layer` and
  `fatal: expected flush after ref listing`.

### N-GPU / four-GPU reward-probe parallelism

The old GPUShare server had two visible GPUs. The new GPUShare server at
`ssh -p 30054 root@i-2.gpushare.com` has four visible GPUs:

- GPU 0: NVIDIA GeForce RTX 4090, about 48 GiB.
- GPU 1: NVIDIA GeForce RTX 4090, about 48 GiB.
- GPU 2: NVIDIA GeForce RTX 4090, about 48 GiB.
- GPU 3: NVIDIA GeForce RTX 4090, about 48 GiB.

The multi-GPU optimization is not independent RL jobs. The target is still one
RL job where, after the policy selects one BLB action, the model-forward reward
probe trials for that same action run concurrently across GPUs.

Current implementation facts:

- `--stage2-k-trials` controls the number of Stage-2 reward noise trials and
  maps into `BLBStage2TrainConfig.num_trials_per_step`. On the four-GPU server,
  use `--stage2-k-trials 4` so each GPU runs one independent trial.
- Enable four-GPU reward probe with `CUDA_VISIBLE_DEVICES=0,1,2,3` plus
  `--blb-v3-reward-devices 0,1,2,3`. Leaving `--blb-v3-reward-devices` unset
  preserves the original single-GPU code path.
- `parse_device_ids(...)` accepts strings, Fire tuple/list forms, and int
  single-device forms. Invalid non-empty specs raise instead of silently falling
  back to single GPU.
- `ProbeRunner` attaches when `reward_devices` has at least two devices. Worker
  0 reuses the env model/bridge on `cuda:0`; workers 1+ deep-copy the model to
  their own devices, build their own handler/bridge, and move probe batches.
- For multi-GPU sequential runs, `BLBStage2EnvConfig.persistent_probe_install`
  is enabled after noisy baseline preflight. BLB wrappers/hooks stay installed
  across episodes and `BLBNoiseRLBridge.apply(...)` updates cfgs in place; this
  avoids the old per-episode clear/reinstall churn on four model replicas.
- RL action to `Rescale_optimizer` training interaction is in-process, not
  per-action JSON-file IPC. `InProcessInvoker` preloads `ReplanSession`; the
  hot path calls `replan_variables(...)` with Python `t_new` and
  `delta_overrides`. `SubprocessInvoker` remains the JSON-file debug path.
  Keep equivalence tests between the direct variable API and the compatibility
  payload path before changing this bridge.
- `ProbeRunner.install_action(...)` and `ProbeRunner.clear(...)` fan setup work
  across workers through threads. `episodes.jsonl` now includes timing fields
  for `policy_rollout_wall_seconds`, `per_step_optimizer_wall_seconds`,
  `terminal_cost_eval_wall_seconds`, `terminal_probe_install_wall_seconds`, and
  `terminal_probe_clear_wall_seconds` so throughput bottlenecks can be
  diagnosed from artifacts instead of guessed from GPU utilization alone.
- `build_probe_runner(...)` enables CUDA TF32 fast matmul/cudnn for reward
  probes on Ampere/Ada GPUs. This keeps FP32 tensors and changes only matmul
  kernel precision/performance, not the BLB action mapping or optimizer path.
- During rollout collection, `BLBStage2SequentialPolicy` uses a causal-prefix
  fast path (`truncate_to_current=True`) for single-step sampling/evaluation:
  because the GTrXL mask prevents the current step from attending to future
  tokens, the rollout path only parses and computes tokens `0..current_step`.
  It also caches fixed step/layer/block index tensors as module buffers instead
  of rebuilding them every forward. The per-slot warmstart-prior one-hot
  template and level-index mask are cached too, so the online loop avoids a
  Python slot loop and tiny tensor construction on every decision. PPO update
  batches still use the full-horizon path, and reward/action/probe semantics
  are unchanged.
- The GTrXL policy keeps per-slot actor heads and slot-specific previous-action
  embeddings, but both are vectorized as parameter/embedding tables rather than
  Python `ModuleList` fan-out. This is a throughput requirement for four-GPU
  runs: many tiny per-slot kernels were a measurable `policy_rollout` and PPO
  update bottleneck. Sequential PPO now defaults to `minibatch_size=2048`
  so each 60-episode update processes the same rollout with far fewer GTrXL
  forward/backward passes than the old 128-sample minibatches.
- Sequential PPO keeps the actor-critic module in eval mode during training.
  Exploration is the explicit categorical policy distribution; dropout masks
  are not part of the recorded log-prob distribution and should not add hidden
  randomness or extra tiny kernels to online rollout/PPO replay.
- GTrXL sequential PPO uses conservative KL-adaptive LR. The default adaptive
  max ratio is capped at `1.25` because the 2026-05-22 four-GPU smoke run
  reached `lr_scale=2.5` (`5e-4` effective LR) and produced a non-finite PPO
  update at episode 660. `sequential_ppo_update` now skips non-finite
  minibatches before backward/step and backs off LR instead of contaminating
  policy weights.
- With `K=4` and four GPUs, the split is `[1, 1, 1, 1]`: GPU 0 runs trial 0,
  GPU 1 trial 1, GPU 2 trial 2, and GPU 3 trial 3. Results are returned in
  trial order for existing aggregation.
- Trial seeds are independent per trial via `probe_runner._trial_seed(...)`.
  Worker threads seed only their current CUDA device; do not call
  `torch.cuda.manual_seed_all(...)` inside concurrent reward-probe workers.
- The probe subset remains the Stage-2 stability probe: validation split,
  default size 256, clipped to the full validation set if the validation set is
  smaller than 256. The same probe subset is reused across all K trials; only
  noise seeds differ.

Implementation constraints to preserve:

- Preserve one PPO learner, one action stream, one persistent run directory,
  and one reward per selected action.
- Do not solve this by running four separate launcher processes; that tests
  different actions/seeds and does not accelerate one action's reward.
- Do not assume `CUDA_VISIBLE_DEVICES=0,1,2,3` alone is enough. The reward probe
  must explicitly place model copies and probe batches on all devices.
- Preserve the invalid-chain shortcut: if `Rescale_optimizer` reports
  `any_invalid`, skip model-forward reward as current code does.
- Baseline/noisy preflight that calls `_eval_on_probe(k)` should use the same
  multi-GPU runner so baseline std and candidate std have the same semantics.
- Keep enough diagnostics to prove all requested cards are used: visible
  devices, worker lines, trial split, per-device elapsed time, and
  `terminal_probe_*` fields in `episodes.jsonl`.

User-facing config for four-GPU Stage-2 reward probing:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --stage2-k-trials 4 \
  --stage2-probe-size 256 \
  --batch-size 512 \
  --blb-v3-reward-devices 0,1,2,3 \
  --fresh
```

Run `scripts/stage2_reward_probe_scaling_benchmark.sh` on the new server before
a long run. It tests 1/2/3/4 GPUs and batch sizes 128/256/512 on the real
Stage-2 reward probe path, then writes an HTML scaling report.

Latest four-GPU scaling check on 2026-05-22, artifact
`experiments/server_command_runs/stage2_reward_probe_scaling_20260522_003406/`,
used the real Stage-2 reward probe path with `K=4`, probe size 256, and batch
sizes 128/256/512. Best observed was `batch=512,gpu=4`: mean terminal probe
wall time `1.059s`, mean speedup `3.99x` over single GPU, devices
`cuda:0..cuda:3`, trial split `[1,1,1,1]`, and max sampled utilization `100%`
on all four GPUs. Because the probe subset is 256 examples, `batch=512` does
not increase the reward-probe sample count beyond 256; it is simply the fastest
safe launcher setting observed on the new server.

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

`SERVER_COMMAND.md` was launched once on this server and reached real BLB
Stage-2 sequential RL execution. The stopped run wrote diagnostics under
`Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.005_s2t0.005_s2st0.005/`.
Those generated artifacts were mirrored locally and pushed to `origin/jk_standard_rl`
as commit `20ee2c1`. The server checkout may still show the same generated
artifacts as local modifications until it is synced to the pushed commit; do
not confuse those artifacts with source edits.

Current responsibility split: the user/local coding agent edits research code
locally and pushes it to git. Server-side agents should pull from git, run the
requested experiment, and push or return generated artifacts/results. Do not
proactively patch `.py`, launcher, config, or test source files on the server.
If a server diagnostic discovers a required source fix, document it for the
local code-editing agent instead of making the canonical change on the server.

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

**Long-term research workflow** (added 2026-05-16; see HTML guide §11 for prose):

```bash
# Multi-seed sweep: 5 seeds, isolated persistent dirs, auto-aggregated summary
bash tools/run_multi_seed.sh mrpc-blb-stage2-rl 1,2,3,4,5 trial1 --fresh
# Writes experiments/multi_seed/trial1/seed_summary.{md,json}

# Single seed override (manual): --blb-v3-seed N + --run-tag SUFFIX
bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl \
    --blb-v3-seed 7 --run-tag pilot_s7 --fresh

# Experiments tracking (auto-registered at training end; can also query manually)
python tools/experiments_log.py rebuild                 # regen experiments/index.md
python tools/experiments_log.py query --dataset mrpc --min-reward 0.4
python tools/experiments_log.py register --run-id ... --dataset ... ...  # manual

# Paper-style figures from one or more run dirs
python tools/paper_figures.py \
    --runs "Parting Chapter/persistent/.../trial1_s1" \
            "Parting Chapter/persistent/.../trial1_s2" \
            --group-label "RL (5 seeds)" \
    --out figures/trial1 --formats png pdf
# Supported figs: training_curves / invalid_heatmap / best_vs_baseline /
#                 action_histogram / ppo_dynamics / cost_vs_accuracy
```

**Environment setup** (Docker or venv; full instructions in `docs/SETUP.md`):

```bash
# Docker (CUDA 12.4 + PyTorch 2.5.1 cu124 base, deps frozen at build time)
docker build -t blb-rl:latest .
docker run --gpus all -it --rm -v "$PWD":/workspace blb-rl:latest \
    bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh

# venv: create .venv, install PyTorch first (CUDA wheel), then the rest
bash scripts/setup_cuda124_env.sh
# or manually:
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-torch-cu124.txt
pip install -r requirements.txt
pip freeze --exclude-editable > requirements-frozen.txt   # capture exact resolved set
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
# Recommended entry point: torch-free unit + artifact smoke (parity with CI)
make test                       # or: bash -c 'BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v'

# End-to-end artifact smoke (diagnostics / experiments / strict / preset validator)
make test-smoke                 # or: python tests/test_sequential_smoke.py

# All (best-effort, includes torch-requiring tests)
make test-all                   # or: python -m unittest discover -s tests -v

# Single test file
python tests/test_blb_registry_artifact_consistency.py
```

Most BLB tests are torch-free — they exercise action-space / registry / bridge / cost / mask / threshold / candidate-store / warmstart / diagnostics / strict / preset-validator / action-io logic in isolation. `test_sequential_smoke.py` exercises the SF/K-first artifact pipeline end-to-end (recorder → experiments_log → aggregator) with synthetic data. Only `test_blb_action_mask.py` and `test_blb_stage2_rl_regressions.py` pull in torch + transformers; `test_glue_dataset_loading.py` needs `datasets` + a populated GLUE cache (`GLUE_LOCAL_DATASET_DIR`).

**CI** (`.github/workflows/ci.yml`) runs on every push / PR: torch-free unit tests across py3.10 + py3.11, `ruff check` + `ruff format --check`, `pip-audit` (advisory), and a docs-sanity job that verifies ADR index coverage and HTML guide tag balance.

**Engineering shortcuts** (`make help` lists everything):

```bash
make lint              # ruff check
make lint-fix          # ruff check --fix
make format            # ruff format
make audit             # pip-audit
make docker            # build CUDA Docker image
make train             # bash launcher --fresh
make train-multi-seed  # bash tools/run_multi_seed.sh (SEEDS=... RUN_TAG=... overrides)
make index             # rebuild experiments/index.md
make figures RUN=...   # render paper figures
make preset-check      # tools/validate_preset.py against the launcher flag list
make changelog         # tail CHANGELOG.md
make clean             # nuke __pycache__ / .ruff_cache
```

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

In each BLB run dir you'll find (all SF/K-first since 2026-05-16):

* `blb_stage2_status.json` — atomic-write live status; `best.slots` / `best.slots_by_layer` carry the human view of the current best.
* `blb_stage2_rl_checkpoint_live.pt` — policy + optimizer + episode + best_reward + best_action + rl_variant.
* `blb_stage2_training_curve.{png,npz}` — episode_returns / best curve / PPO loss curve.
* `blb_stage2_best_cfg.pkl` — legacy pickle (kept for back-compat; humans should read the MD/JSON below).
* `blb_stage2_best_action_full.{json,md}` — full slot description of training best. JSON has top-level `slots`/`slots_by_layer` (Paean-compatible) and the legacy `records`; MD leads with a per-layer/per-block selection table.
* `blb_stage2_baseline_action_full.{json,md}` — same shape, for the static_skeletons baseline (the reference frame).
* `blb_stage2_report.md` — final training report; §5 has per-layer/per-block selection + best-vs-baseline diff tables (raw int vec is hidden inside a `<details>` block).
* `diagnostics/` — long-term diagnostics dashboard (see "Diagnostics dashboard" below).
* `details/noise_ppo_step_info_<start>-<end>.txt` — per-episode rollover diagnostics (one file per `details_batch_size` episodes, default 360). Wired into the sequential path on 2026-05-17 for parity with legacy v2; each record carries return / priority / cost signals / first_invalid location.
* `warning.txt` — reward-crash log; appended whenever a new PPO rollout's mean return drops by more than `drop_threshold=0.3` vs the previous one. Points at the current `details/` batch file so root-cause is one `grep` away.
* `blb_stage2_error.txt` — traceback, only on crash.

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

`first_input fresh` is folded into step 0. Do not hard-code old per-step slot counts here; use `step_schedule_max_dim(...)` or the live `seq_env.max_step_dim`. The current MRPC/BERT-base GTrXL server run reports `max_step_dim=24` and `per-slot heads=[24 x 6]`. Layer 0 has no block 1.

Three additive modules implement this:

- `blb_stage2_rl/sequential_env.py` — `BLBStage2SequentialEnv` wraps an existing `BLBStage2Env`. Each `step()` splices the per-step action into a `_pending_full_vec`, calls `ReplanSession` on **just that one block's graph** for an immediate dense cost signal, applies `apply_optimizer_output_to_cfg` + `sync_block2_qk_binding` (block 2) to that block's cfg, then either returns shaped per-step reward or, on the terminal step, hands the assembled full vec to `BLBStage2Env.step()` for the existing model forward + full hard-priority reward. Per-step reward shaping is configurable via `SequentialEnvConfig` (invalid_penalty, cost_shaping_coeff, fusion_shaping_coeff, early_terminate_on_invalid).
- `blb_stage2_rl/sequential_policy.py` — `BLBStage2SequentialPolicy` is now a v2-scale causal GTrXL token model, not the old MLP. Default shape: `d_model=256`, `n_heads=8`, `n_layers=4`, `d_ff=512`, `dropout=0.1`. Token input includes step/layer/block embeddings, previous action embeddings, previous optimizer signals, static features, and a current-step marker. Actor output uses one per-slot head per padded `max_step_dim` slot, each with up to 6 level logits; the current MRPC/BERT-base run uses `max_step_dim=24`, not a hard-coded 13. Per-step `(slot_mask, num_levels, action_level_mask)` still defines legal support. Critic is a single `Linear(256,64) -> Tanh -> Linear(64,1)` value head. Includes `SequentialRolloutBuffer` with GAE-λ over horizon-N episodes and `sequential_ppo_update`.
- `blb_stage2_rl/sequential_runner.py` — `train_sequential(...)` drives the env+policy+buffer end-to-end. Deliberately a thin standalone driver (does NOT touch persistence / status board / candidate store) so it can run alongside the existing `BLBStage2RLRunner`.

Helpers in `action_space.py`: `step_schedule(num_layers, profile, attn_degree_per_layer, gelu_degree_per_layer)` returns the ordered list of `BlockStepSpec` with each step's slot dims, kinds, full-vec offsets, and the graph-key suffix (e.g. `block3_exp_n4`) that ReplanSession should be called against. `splice_step_action_into_full_vec(...)` writes per-step actions into the legacy 577-dim vector. `step_schedule_max_dim(num_layers)` returns the padded max slots per step and must be treated as the source of truth for policy head count.

Runner integration: `BLBStage2RLRunner.run(...)` dispatches to `run_sequential_via_runner` whenever `train_cfg.sequential_rl=True` (now the default). The integration owns: persistent-dir resolution, baseline bootstrap, status board updates, checkpoint save/resume (with `rl_variant` compatibility check), diagnostics recorder wiring, SF/K-first action description files, final report markdown, and auto-registration into `experiments/registry.jsonl` at end of training. Current sequential checkpoint variant is `blb_v3_sequential_gtrxl_v2scale`; old MLP sequential checkpoints are intentionally not compatible with fresh GTrXL runs.

### Current Stage-2 GTrXL / exploration contract (2026-05-21)

- Warmstart is an external decaying logit prior, not a permanent learned actor-head bias. The actor heads are orthogonal-initialized with gain `0.01`; `apply_preferred_per_step_bias(...)` stores preferred per-slot indices and a default prior scale. During collection, the runner records `baseline_prior_scale` on each transition so `sequential_ppo_update` can replay the exact log-prob distribution.
- Baseline prior schedule: forced anchor uses `1.2`; episode 60 starts at `1.0`; episode 60-600 decays to `0.45`; episode 600-2000 decays to `0.15`; after episode 2000 the weak prior stays at `0.15`. Default forced-baseline anchor is exactly 60 episodes unless explicitly overridden by `force_baseline_episodes` or `warmstart_anchor_episodes`.
- PPO stabilization now includes running return normalization, clipped Huber value loss on normalized returns, MAD-clipped advantage normalization, approximate KL stats and early stop, adaptive LR scale, and per-slot entropy recovery. Checkpoints must persist the policy state plus PPO auxiliary state.
- Exploration is non-monotonic cost-boundary search. Do not encode "lower SF is closer to the constraint boundary" as an invariant. SF/K moves are proposals only; whether a move is useful is decided by F1 model-forward metrics, stability, Rescale_optimizer cost signals, and Pareto archive events.
- Safe-neighbor masks are bidirectional around the chosen base action for SF-like slots. K locality is by truncation-bit distance through non-monotonic `K_LEVELS`, not by categorical index and not by assuming smaller K/SF is always better. Non-selected slots remain fixed at the chosen base action.
- The chosen base action can be the static baseline or a recent Pareto-frontier seed. `GuardedRadius2Controller` keeps empirical per-offset stats: P3 success, P1/P2, loss-cap, stability violation, invalid, frontier/member/dominated/duplicate events, and average cost-vector changes. Radius2 can sample only offsets with at least three P3 successes and zero failures; any radius2 P1/P2, invalid, loss cap, or stability violation triggers cooldown.
- Store and replay both `action_level_mask` and `baseline_prior_scale`. Recomputing either during PPO update invalidates PPO ratios.

### Diagnostics dashboard (added 2026-05-16)

`blb_stage2_rl/diagnostics.py::RLDiagnosticsRecorder` writes to `<progress_dir>/diagnostics/`:

* `diagnostics_summary.md` — human-readable Chinese summary regenerated every `save_interval` episodes. Contains progress / Top-20 candidates / first-invalid heatmap / PPO dynamics / action-distribution overview / **auto-flag warnings** (learning regression / training stall / first-invalid concentration / clip_fraction high / policy collapse). **First place to look when debugging.**
* `episodes.jsonl` — per-episode JSONL (append-only, tail-able).
* `ppo_updates.jsonl` — per-PPO-update JSONL.
* `top_candidates.jsonl` — top-K best episodes with full `slots` view + `diff_vs_baseline`.
* `first_invalid_counts.json` — `{"L08-B3": 312, ...}` heatmap.
* `action_histogram.npz` — per-slot action-index counts (`int64[num_slots, max_levels]`).
* `baseline_action_vec.json` / `best_action_vec.json` — SF/K-first views (Paean-compatible — feed to `--action-config`).

`action_io.py` provides the `action_vec ↔ slots_list` bidirectional converter used by the recorder, the persistence layer, and Paean's `load_action_grid_config`. The slot label format `L<i>.B<n>.<kind>[.<short>]` is the canonical naming convention for cross-tool slot identity.

## Verification: F0 → F1 → F4 fidelity ladder

The active ladder is **three tiers** (F2 / F3 were deprecated and removed
2026-05-16 — old candidate-store JSONL records still carry those strings but
they rank as legacy and aren't promotable):

- **F0** — optimizer-only: decode action, call `Rescale_optimizer`, collect `valid / total_bits / fusion_count`. No model forward. For registry checks, sensitivity scans, cheap candidate filtering.
- **F1** — small probe + few MC trials, online during training: catches obvious accuracy collapses cheaply. This is where the per-episode reward signal comes from.
- **F4** — final eval: full/near-full validation set, real BLB install, frozen report. **Only F4 numbers belong in "best" claims.**

RL training is long-cycle. Based on prior runs, effective BLB Stage-2 RL
usually needs 50,000+ episodes/rounds. The user expects a healthy run to enter
a rapid reward-growth phase sometime after roughly 20,000 episodes; if a 60k
run is still flat well past that point, treat it as a training/search pathology
to diagnose instead of blindly spending the remaining budget. Short runs such
as 200 episodes are for plumbing, performance, and regression smoke only; do
not treat their reward quality as evidence that the RL search worked or failed.

The runner's "final eval" path must install the actual BLB best action (decode → `bridge.apply` → real `Rescale_optimizer`), not silently fall back to a legacy all-max baseline. If you change runner glue, verify this path explicitly.

## Conventions worth knowing

- **Don't directly call `rl_tune*.py`**. The launcher does conflict checks (e.g. `legacy_v2` rejects BLB-only flags), generates the persistent slug, and creates `LATEST_PID` / `LATEST_RUN_DIR` markers under `Parting Chapter/persistent/`.
- **First time for a parameter combo always needs `--fresh`**. The launcher refuses to start otherwise to prevent accidental overwrites.
- **MC repeated evaluation, not single trial.** Multi-trial probe (sampling RNG independent of `torch.manual_seed`) + per-slot entropy logging beat single-shot rewards. A single noise trial is not evidence.
- **Warmstart toward all-max baseline.** Action space is huge; uniform-random rollouts produce mostly invalid candidates. Bias the actor toward each slot's all-max index at init — this constrains the *prior*, not the search.
- **GLUE network instability.** `rl_tune.py` honors `GLUE_LOCAL_DATASET_DIR` / `GLUE_DATASET_DIR` / `DatasetDict.save_to_disk` dirs / local parquet / HF cache `local_files_only=True` fallback. Pre-stage data and `export GLUE_LOCAL_DATASET_DIR=...` before remote long runs.
- Logs/curves/checkpoints under `rl_results/` are mostly gitignored; un-ignored exceptions are explicit in `.gitignore` (e.g. `pruning_search_log.txt`, `persistent/**/*.csv`). Don't add new untracked artifact patterns blindly.
- **All user-visible artifacts are SF/K-first** (since 2026-05-16). Never persist an action as a flat int vec without also writing the slot-form view (label + decoded SF / truncation_bits). `action_io.action_vec_to_slots_list` + `describe_action_vector` are the conversion entry points.
- **Major architectural decisions live in `docs/adr/`** (added 2026-05-16). Before changing reward shape / baseline source / action space layout / fidelity ladder, read the relevant ADR. If you reverse a decision, write a new ADR that supersedes the old one (don't just edit it).
- **Every run gets auto-registered** to `experiments/registry.jsonl` at training end (via `sequential_runner.py` subprocess hook). If you bypass the launcher to debug, register manually with `python tools/experiments_log.py register ...` so cross-run comparisons stay complete.
- **Unified logging entry point**: prefer `from blb_stage2_rl.logging_helpers import get_logger; log = get_logger(__name__)` in new code. `BLB_LOG_LEVEL=DEBUG`, `BLB_LOG_FILE=path.log`, `BLB_LOG_JSON=1` switch verbosity / file sink / structured output without code edits. Legacy `evaluator.log` / `print` still work but should be migrated when touching nearby code.
- **Strict mode**: `BLB_STRICT=1` makes `blb_stage2_rl.strict.swallow` / `strict_guard` re-raise instead of swallowing. Use it when chasing a silent-failure bug. New best-effort code paths (writing optional artifacts) should use these helpers instead of bare `try/except Exception: pass`.
- **Preset typos are caught** by `tools/validate_preset.py` (also `make preset-check`). Run it before committing a preset change; it parses the launcher's flag list and reports unknown flags / duplicates / bad values.
- **Console / log hygiene (2026-05-17 rewrite).** The sequential RL console output is plain key-value bullets (no rounded box borders) because the old `╭─╮│╰╯` borders broke at narrow widths and CJK mixed content. `pruning_search_log.txt` is truncated on `--fresh` and never re-headers itself within a single run (a long-standing operator-precedence bug at `layer_importance_evaluator.py:3470` used to write the init banner **80×** per call — see `tests/test_sequential_smoke.py::OutputHygieneRegressionTest`). The new `_format_best_action_slots` lays out each slot on its own line under a `[L<i>.B<n>]` block header, column-aligned by field-name width.
- **Sequential reward calibration (2026-05-17 fix).** Sequential `BLBStage2RLRunner` was missing the **noisy baseline preflight** the legacy single-shot path has had since day one. Without it, `baseline.loss_std` came from a clean (noiseless) model → 0 → `stab_threshold = 0 * 1.5 + 1e-3 = 0.001`, well below the noise floor of any real candidate. Every episode tripped priority-2 stability and fell into the `loss_std=inf` fallback branch `r = -priority2_penalty - 1.0 * priority2_scale = -150` — terminal reward collapsed to a constant. Fix: `sequential_runner.py` now runs `base_env.step(baseline_action_vec)` once after baseline cost calibration, takes the noisy `loss_std` / `metric1_mean`, and derives `acc_threshold = noisy_acc - stage2_limit_tolerance`, `stab_threshold = noisy_std * (1 + stage2_stability_tolerance) + 1e-3` with a `≥ 0.05` floor. `env._eval_on_probe` also clamps per-trial losses to `[0, 100]` (replacing nan/inf) so a single overflowing trial can't poison the whole episode std. Regression in `tests/test_sequential_smoke.py::OutputHygieneRegressionTest::test_sequential_runner_has_noisy_baseline_preflight`.
- **Invalid-chain attribution: per-block, not just first (2026-05-17).** Previously each episode's `details/noise_ppo_step_info_*.txt` recorded only `first_invalid=step4 (L1-B1)`; when 8 sub-steps failed, the other 7 were silent — operators had to re-run the optimizer with an ad-hoc script to recover the list. Now `EpisodeRecord.invalid_block_details` collects every `(step, layer, block, graph_key, reason)` and the details writer emits them under `invalid_blocks (N):` with `_format_invalid_chain_reason` summarising the optimizer's invalid_chain dict (`primes_over_q_max`, `primes_under_q_min`, `reason`, `stage`, `message`). For post-hoc forensics on any saved action JSON, `scripts/blb_diagnose_invalid_blocks.py --action-config <path> --output-dir <dir>` runs the same pipeline offline and writes `report.{md,json}` enumerating every (layer, block) status + slot SF/K configuration. **Caveat (fixed 2026-05-17 second pass):** the diagnostic script's `_stage1_degrees_from_meta` originally read from the wrong JSON path (`cfg[dataset]` instead of `cfg[model_type][dataset]['stage1']`) and silently fell back to `[4]*L`, producing reports where every invalid block looked like `block5_n4` / `block3_exp_n4` regardless of the real per-layer stage-1 degree. Now reads `cfg[model_type][dataset]['stage1']['gelu' | 'softmax']` correctly and warns loudly on miss.
- **Sequential invalid-action mask + skip-forward (2026-05-17).** Three coordinated changes so the RL policy "only sees valid actions" the way the user spec'd: (1) `blb_stage2_rl/sequential_env.py` splits `step()` into `evaluate_step(action) → eval_info` (calls optimizer, no state mutation) and `commit_step(eval_info) → (obs, reward, done, info)` (splices accumulator, advances step, runs terminal forward); old single-call `step()` survives as a backward-compat wrapper. (2) `ForbiddenActionMask` in `blb_stage2_rl/action_mask.py` is a per-`(layer, block)` blacklist of action tuples that previously failed; `train_sequential` rejection-samples around it (up to `max_rejection_retries=32`) before calling `evaluate_step`, then adds new failures back to the mask. On exhausted retries we fall back to the baseline action slice for that step (guaranteed valid via static_skeletons). Mask survives across episodes and is round-tripped in the checkpoint as `forbidden_mask_records`. (3) `blb_stage2_rl/env.py:BLBStage2Env.step` now short-circuits the model forward when the optimizer already reported `any_invalid` — emits a priority-3 cost reward with the invalid_penalty docked and `forward_skipped_reason="any_invalid_chain"` in info, skipping `bridge.apply` + `_eval_on_probe` entirely. Combined effect: the policy explores the valid sub-space of actions only; invalid tuples that ever appear are blacklisted forever; if a committed action does turn out invalid (defensive fallback), the wasted model forward is skipped.
- **GTrXL warmstart prior (2026-05-21).** The old permanent `+3.5` actor-head bias has been replaced by an external, decaying baseline logit prior. Current default `warmstart_bias_gain` is `1.2`; the prior decays to `0.45` by episode 600 and `0.15` by episode 2000. `baseline_prior_scale` is stored per transition and replayed during PPO update.
- **Non-monotonic safe curriculum (2026-05-21).** The episode-121 collapse was not an optimizer invalid-chain failure: `any_invalid=False` but the terminal probe hit `loss_mean=100` and P1(acc), so unrestricted post-anchor sampling produced accuracy-catastrophic full actions. The safe path is: 60-episode baseline anchor by default; absolute episode indices for anchor/entropy/resume; forced-anchor PPO evaluation under unrestricted support; after anchor, small episode-level mutable offset sets; selected SF/K slots can move bidirectionally; K locality uses truncation-bit distance through non-monotonic `K_LEVELS`; `action_level_mask` and `baseline_prior_scale` are both replayed during PPO update; empirical per-offset stats and cooldown gate guarded radius2.
- **Noisy baseline accuracy guard (2026-05-20 follow-up).** In the 600-episode
  smoke, the catastrophic `loss_mean=100` collapse disappeared, but baseline
  anchor episodes still had occasional P1(acc) with normal loss (`loss_mean≈0.34`)
  and `m1≈0.865-0.867`. Root cause: with K=5 and `stage2_probe_size=256`, the
  online probe is discrete enough that the all-max baseline can land one sample
  below `noisy_baseline_metric1 - stage2_limit_tolerance`. Sequential threshold
  calibration should subtract a one-sample guard (`1 / stage2_probe_size`) from
  the noisy-baseline accuracy gate. This prevents false P1 baseline jitter while
  leaving true accuracy collapses such as `m1≈0.31` far below threshold.
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
8. **Writing a new artifact in action-index form only**. Always pair an `action_vec` with its slot-form view (via `action_io.action_vec_to_slots_list`). User-facing files (markdown / JSON the user opens) must lead with `scaling_factor` / `truncation_bits` columns; `action_index` is at most a sanity-check sidekick column.
9. **Hardcoding a seed inside the runner** instead of letting it flow through `BLBStage2TrainConfig.seed` (default 42, overridable via `--blb-v3-seed`). Multi-seed framework relies on this single seed knob.
10. **Modifying `experiments/registry.jsonl` schema without bumping the relevant ADR / experiments doc**. The registry is the cross-run index; downstream tools (`aggregate_seeds.py`, `paper_figures.py`, `experiments_log.py query`) all assume the documented fields.

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
- "Why did we make decision X" → `docs/adr/` (Architecture Decision Records). Currently: ADR-001 sequential PPO, ADR-002 hard-priority reward, ADR-003 per-block K baseline, ADR-004 static_skeletons baseline, ADR-005 SF/K-first outputs, ADR-006 F0/F1/F4 ladder. `docs/adr/README.md` has the index.
- "Find / compare across runs" → `experiments/index.md` (auto-generated from `experiments/registry.jsonl`); CLI filters via `python tools/experiments_log.py query --dataset ... --min-reward ...`.
- "Multi-seed sweep / aggregate stats across seeds" → `tools/run_multi_seed.sh <preset> <seeds_csv> <run_name>`; aggregation via `tools/aggregate_seeds.py` (also called automatically at end of sweep).
- "Generate paper-style figures" → `python tools/paper_figures.py --runs ... --out figures/...`. 6 figure types: training_curves (mean±std band on multi-seed), invalid_heatmap, best_vs_baseline, action_histogram, ppo_dynamics, cost_vs_accuracy.
- "Project user view of the whole system" → `reports/session_summary/blb_stage2_rl_guide.html` (work-product HTML; refreshed each major change; current at 2026-05-16).
- "Set up the environment from scratch" → `docs/SETUP.md` + `Dockerfile` + `requirements.txt`.
