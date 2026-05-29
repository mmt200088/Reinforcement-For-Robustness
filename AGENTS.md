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
- Treat the Stage-2 RL collapse task as long-running research work. It may
  require many experiment cycles over many hours, not one or two edits. When the
  failure is not a simple code bug, act like a researcher: form a falsifiable
  hypothesis, design the next focused experiment, observe the curve/logs, adjust
  code or hyperparameters locally, and keep iterating until the goal evidence is
  clean.
- Current extension of that goal: run and monitor a 60,000-episode Stage-2 RL
  search, not just 600/1000-episode smoke or the earlier first-10k milestone.
  Success now means the long reward curve stays healthy and makes search
  progress on rolling averages. Occasional negative reward spikes or isolated
  P1(acc) episodes are acceptable research noise if they do not become frequent
  or sustained and the rolling reward windows do not collapse. Hard failures
  remain: sustained or high-frequency collapse sentinels such as `loss_mean=100`,
  sustained/high-frequency P1(acc), invalid-step resurgence, a dead GPU
  reward-probe path, or a long plateau caused by entropy/clip collapse or an
  overly narrow safe-neighbor curriculum. For the online watchdog, sparse
  loss-cap spikes should warn, not kill the run; kill only on bursts such as
  consecutive loss caps or at least 5 loss caps in the latest 100 post-anchor
  episodes. Treat 60k runs as research evidence, with online watchdog checks and
  follow-up experiments when the curve stalls.
- Current 60k completion protocol, added 2026-05-24: when the active
  `stage2_rl_60000_curve_20260523_082630` run finishes, first copy the full
  artifact set back locally, extract the best BLB action, run a real BLB final
  eval for that action, and generate a comprehensive HTML report before
  launching the next run. That report must include the complete best
  configuration details, final-eval metrics, full learning curves, throughput,
  reward/P1/P2/stability/invalid summaries, four-GPU evidence, PPO diagnostics,
  and cost/frontier progress. Only after that report and final eval are
  captured should the next 60,000-episode run start from the latest local source
  commit, currently the unbounded P3 cost-rank selection change.
- Stage-1 post-run queue, added 2026-05-24: after the active Stage-2 60k run
  finishes and its final eval/report are captured, pull the latest server code
  that contains the Claude Code Stage-1 changes for BERT-base SST-2/RTE and
  BERT-large MRPC/SST-2/RTE. Validate Stage-1 only, without changing the Stage-1
  architecture or RL algorithm unless a concrete runtime bug requires a narrow
  fix. First run focused smoke/scaling checks for all five Stage-1 tasks
  (`bert-base` SST-2/RTE and `bert-large` MRPC/SST-2/RTE), verify that each can
  run correctly, verify real four-GPU parallelism, and compare speed against
  smaller GPU counts so the four-GPU path is demonstrated rather than assumed.
  Fix any runtime bugs locally, commit/push, then have the server pull or use
  the verified-head fallback before retrying. Once the five Stage-1 tasks are
  proven runnable and four-GPU speed is credible, launch all five full Stage-1
  RL trainings with PPO update window 120 episodes, learning rate `2e-5`, and
  50,000 episodes per task. For Stage-1, the baseline is the pure original
  plaintext model using original GELU and Softmax functions, not the polynomial
  `gelu=4, softmax=6` baseline. Stage-1 constraints are loss, metric1, and
  metric2, each at 0.5% relative tolerance: candidate loss must be at most
  `baseline_loss * 1.005`, and metric1/metric2 must be at least
  `baseline_metric * 0.995`. After each full Stage-1 task finishes, produce an
  HTML report with the best GELU/Softmax configuration, full reward curve,
  loss/metric1/metric2 curves, entropy curve, full-validation final eval for the
  best configuration, baseline loss/metric1/metric2, and absolute plus
  percentage deltas versus baseline.
- Stage-1 baseline implementation note, added 2026-05-25: the pure original
  GELU/Softmax baseline is represented in evaluator arrays with degree `-1`,
  which restores the original functions instead of installing polynomial
  replacements. Stage-1 candidate scoring and final evaluation should also be
  pure plaintext by default, without the historical max-scaling noise
  environment. The Stage-1 reward cost denominator still uses the old high-degree
  cost reference `gelu=4, softmax=6` so cost savings remain well-defined; do not
  interpret that cost reference as the metric baseline.
- Stage-1 mental model and speed note, added 2026-05-25: Stage-1 is a
  plaintext-only search over GELU and Softmax polynomial degrees. It should only
  replace those two functions, using the replacement logic in
  `function_handler.py`; it should not inject BLB/noise. GELU choices are
  degrees `1`, `2`, and `4`; Softmax choices are degrees `2`, `3`, `4`, `5`,
  and `6`. The Stage-1 RL code is in `layer_importance_evaluator.py`, and a
  BERT-base episode has 12 per-layer decisions. Stage-1 inference tests must use
  the full validation set (`validation_full`) during both RL reward evaluation
  and final evaluation; do not switch Stage-1 online reward or final eval to the
  training set or a validation proxy to improve speed unless the user explicitly
  changes this protocol. Do not judge Stage-1 throughput from the 12 decisions
  alone: the required terminal full-validation model-forward pass can dominate
  runtime. Four-GPU Stage-1 rollout is window-style data parallelism across
  complete episodes; worker logs and sampled GPU utilization are better evidence
  than a single instantaneous `nvidia-smi` snapshot.
- Stage-1 reward boundary-search update, added 2026-05-28: future Stage-1 RL
  launches after the active `large_mrpc` run must use the latest commit with
  the revised Stage-1 reward. Differential metric reward is behind
  `STAGE1_ENABLE_DIFFERENTIAL_REWARD` and defaults off; do not enable it unless
  the user explicitly asks. Dense per-step reward is now monotonic cost saving
  only, with no expected-cost-track bonus around GELU2/Softmax4 (`4.5`
  cost/layer), so the policy is free to search below that soft point. Keep the
  terminal log-barrier reward after constraints are satisfied because the user
  wants the safety-margin effect retained. The intended objective is constrained
  boundary search: satisfy full-validation loss/metric limits, then push cost
  as low as the constraints allow.
- Stage-1 entropy-stop budget update, added 2026-05-29: entropy-convergence
  runs must not treat a finite episode count as a success cap. Use
  `--stage1-search-episodes 0` together with
  `--stage1-entropy-stop-threshold 0.1` to run Stage-1 unbounded until the PPO
  policy entropy drops below the threshold. If an older finite-cap Stage-1 run
  such as the `base_sst2` run from `7352cd3` reaches 50,000 episodes without
  the `Stage-1 entropy convergence reached` marker, do not accept that as
  completion; resume from the existing Stage-1 checkpoint on the newer
  unbounded code without `--fresh`. Before launching any new Stage-1 training
  process, validate the Claude Code Stage-1 inference acceleration work and
  confirm deterministic pure Stage-1 plaintext inference with a fixed
  GELU/Softmax configuration. Stage-1 training and final Stage-1 checks must
  not add Stage-2/BLB noise; they should only replace GELU and Softmax in
  plaintext inference unless the user explicitly changes that protocol.
- Validation-only protocol, clarified 2026-05-25: Stage-1 baseline is built on
  the full validation set, and the entire Stage-1 process must not use the
  training set for baseline, RL reward evaluation, candidate checks, or final
  evaluation. Stage-2 follows the same rule: baseline, RL reward/probe
  evaluation, candidate validation, and final evaluation must use the full
  validation set rather than the training set. Do not switch either stage to
  train data, train anchors, sampled train proxies, or validation proxies for
  speed unless the user explicitly changes this protocol.
- Stage-1 RL algorithm correction, added 2026-05-25: the user clarified that
  the previously supplied LSTM `PPO_10.py` file was sent by mistake and must
  not be used as the target Stage-1 algorithm. Do not replace the current
  Stage-1 main path with that LSTM PPO. Until the user provides the correct
  target file/commit, keep the current Stage-1 RL algorithm direction as GTrXL
  PPO while preserving the newer engineering shell: four-GPU data-parallel
  rollout collection, validation_full-only evaluation, exact original
  GELU/Softmax metric baseline via degree `-1`, cost reference `gelu=4,
  softmax=6`, current output/checkpoint/report paths, and the command-line PPO
  update window override. Stage-1 full runs still use 120 episodes per PPO
  update unless the user changes that parameter.
- Stage-1 queue restart, added 2026-05-25: the first `ab9adbb` full Stage-1
  queue died stale around base_sst2 episode 4800 after the wrapper/training
  processes disappeared while `status.json` still said `running`. The user
  requested rerunning all five Stage-1 tasks from scratch and then producing
  the previously requested per-task HTML reports. Archive stale server state/log
  directories before relaunching the queue; do not mark report-done markers
  until each task's final eval/report is complete.
- Stage-1 base_sst2 completion, added 2026-05-27: the clean `base_sst2` full
  run from `e0cbedd` completed 50,000 episodes and reached queue
  `waiting_report`. The final local report is
  `experiments/server_command_runs/stage1_full_50000_base_sst2_20260525_220047/stage1_base_sst2_final_report.html`.
  The logged final selected config is the confirmed global/search best
  (`GELU=[1,1,1,1,1,1,1,4,1,1,1,1]`,
  `Softmax=[2,2,2,2,2,2,2,2,3,2,2,2]`, cost `26.50`, reward `1.7948`).
  Under the old post-selection policy, the checkpoint `best_config` field
  recorded the raw PPO reward-best before final post-selection
  (`Softmax=[2,2,2,3,3,2,2,3,2,3,2,2]`, cost `28.00`, reward `1.8694`) and
  the report showed it as an audit row. The newer Stage-1 selection protocol
  below supersedes that old reporting preference.
- Stage-1 base_rte completion, added 2026-05-27: the clean `base_rte` full run
  from server HEAD `6cd198a` completed 50,000 episodes and reached queue
  `waiting_report`. The final local report is
  `experiments/server_command_runs/stage1_full_50000_base_rte_20260527_015842/stage1_base_rte_final_report.html`.
  The final selected global/search best is
  `GELU=[1,1,1,4,4,1,1,1,1,1,1,1]`,
  `Softmax=[4,3,2,2,2,3,3,2,3,3,4,3]`, cost `33.00`, reward `1.8529`,
  confirmed at episode `38040`. Full-validation final eval on
  `validation_full` size `277` gave baseline loss/accuracy
  `0.7333006263`/`0.7256317690` and selected loss/accuracy
  `0.7247349620`/`0.7472924188`, passing the 0.5% loss/metric constraints.
  The checkpoint raw reward-best is
  `Softmax=[4,4,2,2,2,3,3,3,3,3,4,4]`, cost `34.50`, reward `1.9017`;
  it was included as an audit row under the old post-selection policy. The
  newer Stage-1 selection protocol below supersedes that old reporting
  preference.
- Stage-1 selection protocol correction, added 2026-05-27: Stage-1 GELU/Softmax
  replacement is deterministic, unlike Stage-2 stochastic noise evaluation.
  Do not repeatedly re-confirm Stage-1 window candidates on validation_full for
  final selection. The Stage-1 final selected config is now the raw PPO
  reward-best (`checkpoint["best_config"]`) with no global/search candidate
  post-selection override. If a deterministic Stage-1 tie-breaker is needed
  outside raw reward selection, the priority is `metric1 + metric2` first, then
  lower loss, then lower cost. The earlier `base_sst2` and `base_rte` reports
  used the old global/search post-selection policy and should be interpreted as
  pre-correction artifacts.
- Stage-1 unbounded base_sst2 completion, added 2026-05-29: after the old
  50,000-episode capped `7352cd3` run failed to reach entropy convergence, the
  queue was source-synced to `73e6a8f` and resumed from the existing checkpoint
  with `--stage1-search-episodes 0` and `--stage1-entropy-stop-threshold 0.1`.
  The resumed run reached entropy convergence at episode `65280` with final
  entropy `0.0959`. The final local report is
  `experiments/server_command_runs/stage1_unbounded_base_sst2_20260529_173035/stage1_base_sst2_unbounded_final_report.html`.
  The final selected config is the raw PPO reward-best:
  `GELU=[1,1,1,1,1,1,1,4,1,1,1,1]`,
  `Softmax=[2,2,2,2,2,2,2,2,2,2,2,2]`, cost `26.00`, reward
  `1.2666790039`. Full-validation final eval on `validation_full` size `872`
  gave baseline original-plaintext loss/accuracy
  `0.2818579718`/`0.9243119266` and selected loss/accuracy
  `0.2803423208`/`0.9231651376`, passing the 0.5% loss/metric constraints.
  The final eval recorded zero Stage-2/BLB noise hooks for both baseline and
  selected config, confirming Stage-1 plaintext-only semantics.
- Stage-1 large_mrpc speed/parallelism note, added 2026-05-27: the
  `large_mrpc` full run speed around 1.6k episodes/hour is broadly consistent
  with the earlier 4-GPU smoke result of about 2.1 seconds/episode. It is slower
  than BERT-base primarily because BERT-large has 24 transformer layers and a
  much heavier validation_full model-forward pass. The Stage-1 parallel rollout
  worker path now initializes previous GELU/Softmax actions with the same SOS
  tokens as the serial path, so BERT-large and BERT-base use the same PPO
  rollout semantics except for model size/adaptation. Parallel rollout windows
  should log per-worker wall times and an estimated speedup line for future
  speed checks. The pre-fix active `large_mrpc` partial run should be treated as
  superseded once the SOS-fix queue is relaunched.
- Stage-1 large_mrpc completion, added 2026-05-29: the corrected `large_mrpc`
  full run from server HEAD `cdcc42b` completed 50,000 episodes and reached
  queue `waiting_report`. The final local report is
  `experiments/server_command_runs/stage1_full_50000_large_mrpc_20260527_194810/stage1_large_mrpc_final_report.html`.
  The final selected config is the raw PPO reward-best from
  `checkpoint["best_config"]`: `GELU=[1,1,2,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1]`,
  `Softmax=[2,3,3,2,3,3,6,5,2,5,3,6,3,3,2,2,3,2,3,5,3,2,3,3]`,
  cost `67.00`, reward `3.3509`. Full-validation final eval on
  `validation_full` size `408` gave baseline loss/Accuracy/F1
  `1.4342708588`/`0.8799019608`/`0.8756547374` and selected
  loss/Accuracy/F1 `1.2522128820`/`0.8970588235`/`0.8950905297`, passing the
  0.5% loss/metric constraints. Four-worker rollout evidence covered `417`
  windows with mean speedup about `3.92x`; the last partial window ran
  `[20,20,20,20]` episodes across `cuda:0..cuda:3` at `3.89x`.
- Stage-1 queue change, added 2026-05-28: after the active corrected
  `large_mrpc` run finishes and its final eval/report are captured, do not launch
  `large_sst2` or `large_rte`. Because Stage-1 final selection changed to raw
  PPO reward-best with no candidate-window or repeated full-validation
  post-selection, rerun the earlier BERT-base `base_sst2` and `base_rte` tasks
  fresh from the corrected code. These reruns should use Stage-1 only, the same
  validation_full protocol and four-GPU rollout settings, and entropy convergence
  stopping: stop cleanly at a PPO update once policy entropy is below `0.1`
  rather than treating a fixed episode count as the success criterion.
- Decision boundary for this goal: make small corrective changes autonomously
  when the evidence supports them, including hyperparameter tuning, watchdog
  threshold changes, narrow diagnostic instrumentation, and focused bug fixes
  that preserve the current architecture and artifacts. Ask the user before
  major architectural/rewrite decisions, especially changes that invalidate the
  current Stage-2 setup, replace the reward/search formulation, rewrite large
  modules, or make earlier artifacts/checkpoints no longer interpretable.
- First 10k attempt evidence, 2026-05-20: `NEIGHBOR_RAMP=3000`,
  `NEIGHBOR_MAX_MUTATIONS=16`, `NEIGHBOR_MAX_RADIUS=3` improved reward into the
  low 42s but hit a P1 cluster around episodes 1699-1757. P1 was 0 through
  radius=1 and appeared when safe-neighbor reached `radius=2` with 8-9 mutated
  offsets. Current guarded-radius2 follow-up still keeps raw safe-neighbor at
  `NEIGHBOR_MAX_RADIUS=1`, with `ANCHOR_EPISODES=60`,
  `NEIGHBOR_RAMP=1800`, `NEIGHBOR_MAX_MUTATIONS=12`, `ENT_COEF=0.06`,
  `ENT_RAMP=600`, and `WARMSTART_BIAS_GAIN=1.2` with a decaying baseline
  prior. It enables radius2 only when
  the frontier has stalled and recent health is clean: default server settings
  are `GUARDED_RADIUS2_ENABLED=1`, `GUARDED_RADIUS2_MIN_EPISODE=1060`,
  `GUARDED_RADIUS2_STALL_WINDOW=600`, `GUARDED_RADIUS2_MAX_MUTATIONS=4`,
  `GUARDED_RADIUS2_EPISODE_FRACTION=0.15`, and
  `GUARDED_RADIUS2_COOLDOWN_EPISODES=300`. Do not replace this with raw
  default radius2.
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
- Old GPUShare server Python environment was system Python 3.11.12 with
  PyTorch 2.9.1+cu128.
- New GPUShare server at `ssh -p 30054 root@i-2.gpushare.com` was verified on
  2026-05-21 with 4x NVIDIA GeForce RTX 4090, about 48 GiB each, and system
  PyTorch 2.9.1+cu128 seeing all 4 GPUs. Do not downgrade PyTorch there unless
  the runtime breaks; install missing Python deps with `pip install -r
  requirements.txt`.
- `requirements-torch-cu124.txt` and `scripts/setup_cuda124_env.sh` remain an
  optional CUDA 12.4 fallback path. The normal new-server path should preserve
  the working `torch==2.9.1+cu128` runtime.
- The project also pins `transformers==4.44.2` because newer 4.57.x rejects
  `TrainingArguments(evaluation_strategy=...)`.
- GitHub HTTPS transport on this server needs repo-local Git settings:
  `git config --local http.version HTTP/1.1` and
  `git config --local protocol.version 0`. Without them, `git pull` can fail
  with `RPC failed; curl 16 Error in the HTTP2 framing layer` and
  `fatal: expected flush after ref listing`.
- If GitHub HTTPS is temporarily unreachable but the exact pushed commit has
  already been transferred as a git bundle and fast-forwarded on the server,
  `scripts/stage2_first10k_server_run.sh` may be launched with
  `EXPECTED_SOURCE_COMMIT=<commit>` and
  `ALLOW_VERIFIED_HEAD_WITHOUT_PULL=1`. This fallback is only valid when
  `git rev-parse HEAD` exactly matches the expected commit; otherwise the
  script must still abort instead of running a stale checkout.

### N-GPU / Four-GPU Reward-Probe Parallelism

The old GPUShare server had two visible GPUs. The new GPUShare server has four
visible GPUs:

- GPU 0: NVIDIA GeForce RTX 4090, about 48 GiB.
- GPU 1: NVIDIA GeForce RTX 4090, about 48 GiB.
- GPU 2: NVIDIA GeForce RTX 4090, about 48 GiB.
- GPU 3: NVIDIA GeForce RTX 4090, about 48 GiB.

The multi-GPU optimization is not independent RL jobs. The target is still
one RL job where, after the policy selects one BLB action, the model-forward
reward probe trials for that same action run concurrently across GPUs.
This should accelerate the repeated inference tests used to compute the PPO
reward for one action.

Current implementation facts:

- `--stage2-k-trials` controls the number of Stage-2 reward noise trials. It
  maps into `BLBStage2TrainConfig.num_trials_per_step`. On the four-GPU server,
  use `--stage2-k-trials 4` so each GPU runs one independent trial.
- Enable four-GPU reward probe with `CUDA_VISIBLE_DEVICES=0,1,2,3` plus
  `--blb-v3-reward-devices 0,1,2,3`. Leaving `--blb-v3-reward-devices` unset
  preserves the original single-GPU code path.
- `blb_stage2_rl/probe_runner.py::parse_device_ids(...)` accepts all launcher
  forms observed in practice: `"0,1,2,3"`, Python Fire tuple `(0, 1, 2, 3)`,
  list `[0, 1, 2, 3]`, int `0`, and stringified tuple/list forms. Invalid
  non-empty specs raise instead of silently falling back to single GPU.
- `BLBStage2RLRunner._build_train_config_from_evaluator(...)` fills
  `BLBStage2TrainConfig.reward_devices`. `sequential_runner.py` attaches a
  `ProbeRunner` when that list has at least two devices and logs
  `[multi-gpu] reward probe enabled: devices=[0, 1, 2, 3]`.
- `BLBStage2Env.step(...)` applies the selected BLB config, installs that same
  decoded action on every `ProbeRunner` worker, then calls
  `self._eval_on_probe(self.env_cfg.num_trials_per_step)`.
- `BLBStage2Env._eval_on_probe(k_trials)` delegates to `ProbeRunner.run_trials`
  when a runner is attached. The runner splits trials round-robin. With
  `K=4` and four GPUs, the split is `[1, 1, 1, 1]`: GPU 0 runs trial 0, GPU 1
  trial 1, GPU 2 trial 2, and GPU 3 trial 3, then returns results in trial
  order for the existing aggregation.
- RL action to `Rescale_optimizer` training interaction is in-process, not
  per-action JSON-file IPC. `InProcessInvoker` preloads `ReplanSession`; the
  hot path calls `replan_variables(...)` with Python `t_new` and
  `delta_overrides`. `SubprocessInvoker` remains the JSON-file debug path.
  Keep equivalence tests between the direct variable API and the compatibility
  payload path before changing this bridge.
- Trial seeds are independent per trial via `probe_runner._trial_seed(...)`.
  Workers seed only their current CUDA device; they must not call
  `torch.cuda.manual_seed_all(...)` inside concurrent worker threads.
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
- Do not assume `CUDA_VISIBLE_DEVICES=0,1,2,3` alone is enough. PyTorch
  `torch.device("cuda")` means the first visible GPU unless the reward probe
  explicitly places model copies and batches on all devices.
- Do not share one mutable `model`/`BLBNoiseRLBridge` instance across GPUs.
  Worker 0 reuses the env model/bridge on `cuda:0`; workers 1+ deep-copy the
  model to their own devices, build their own handler/bridge, and move probe
  batches.
- Avoid reloading the HuggingFace model for every action. `ProbeRunner` workers
  are initialized once per run and reused across action evaluations.
- For multi-GPU sequential runs, `BLBStage2EnvConfig.persistent_probe_install`
  is enabled after noisy baseline preflight. BLB wrappers/hooks stay installed
  across episodes and `BLBNoiseRLBridge.apply(...)` updates cfgs in place; this
  avoids the old per-episode clear/reinstall churn on four model replicas.
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
- Sequential rollout has three invalid-action filters, all layered on the
  per-step `action_level_mask` without shortening the full action vector or
  changing policy/critic shapes. `StaticInvalidLevelMask` runs once before RL:
  it performs a baseline-prefix, one-slot-at-a-time `Rescale_optimizer`
  feasibility scan and hides any `(layer, block, slot, level)` that is locally
  invalid. This follows the COINN-style idea of shrinking invalid configuration
  space before global optimization, and is intentionally more aggressive than
  runtime masking: it may discard combinations that could have become valid
  under another prefix, which the user accepts to reduce invalid-chain retries.
  The scan only calls `evaluate_step`; it commits baseline actions only for
  non-terminal prefix advancement and never commits the terminal step, so it
  does not trigger the terminal model-forward reward probe. `ForbiddenActionMask`
  still blacklists exact `(layer, block, step-action tuple)` samples after the
  optimizer reports `invalid_chain`. `EmpiricalInvalidLevelMask` then projects
  repeated runtime invalid evidence back onto per-slot levels. Static,
  empirical, and exact-tuple masks always preserve the static baseline and
  current base/frontier proposal levels. `episodes.jsonl` records
  `samples_rejected_by_mask`, `samples_rejected_by_optimizer`,
  `steps_fallen_back_to_baseline`, `forbidden_mask_total`,
  `static_invalid_level_disabled`, `static_invalid_level_applied`,
  `empirical_invalid_level_disabled`, and `rejection_optimizer_wall_seconds`;
  use these before claiming invalid-chain pruning improved speed.
- Current 60k watchdog policy after the 2026-05-22 user update: do not hard-stop
  just because a few P1/P2 episodes appear. Post-anchor P1+P2 is a hard failure
  only when the rate exceeds 30% after at least 100 post-anchor samples. Sparse
  P1/P2 should be warnings. Keep other hard stops: invalid-step resurgence,
  loss-cap bursts, non-finite PPO, dead/no-progress PID, and broken four-GPU
  reward-probe evidence.
- GTrXL sequential PPO uses conservative KL-adaptive LR. The default adaptive
  max ratio is capped at `1.25` because the 2026-05-22 four-GPU smoke run
  reached `lr_scale=2.5` (`5e-4` effective LR) and produced a non-finite PPO
  update at episode 660. `sequential_ppo_update` now skips non-finite
  minibatches before backward/step and backs off LR instead of contaminating
  policy weights.
- Keep the probe dataset fixed across trials exactly as today. Only the
  independent noise RNG seeds differ per trial.
- Preserve the invalid-chain shortcut: if `Rescale_optimizer` reports
  `any_invalid`, skip model-forward reward as current code does. Do not spend
  GPU work on invalid candidates.
- Baseline/noisy preflight that calls `_eval_on_probe(k)` should use the same
  multi-GPU trial runner so baseline std and candidate std have the same
  semantics.
- Optional fast online reward mode changes only the online training probe, not
  baseline calibration or promotable final validation. Enable it with
  `--blb-v3-fast-reward-mode-enabled 1`, `--blb-v3-online-k-trials 1`,
  `--blb-v3-terminal-eval-batch-size 4`, and
  `--blb-v3-promotion-validation-trials 4`. In this mode the sequential runner
  defers terminal model-forward rewards, accumulates up to four completed
  actions, and calls `ProbeRunner.run_action_trials_once(...)` so each GPU runs
  one distinct action/trial. Exact repeated action hashes may reuse cached
  terminal probe metrics; `compute_reward` still runs again so duplicate
  frontier/cost bookkeeping remains consistent. Promotion validation reruns
  selected P3 boundary/high-reward actions with the repeated-trial path and can
  replace the online reward if validation exposes a lower priority.
- Keep enough diagnostics to prove all requested cards are used: visible
  devices, reward probe device list, trial split, per-device elapsed time,
  worker lines, and `terminal_probe_*` fields in `episodes.jsonl`.
- Run `scripts/stage2_reward_probe_scaling_benchmark.sh` on the new server
  before a long run. It tests 1/2/3/4 GPUs and batch sizes 128/256/512 on the
  real Stage-2 reward probe path, then writes an HTML scaling report.

User-facing config for four-GPU Stage-2 reward probing:

```bash
--blb-v3-reward-devices 0,1,2,3
--stage2-k-trials 4
--stage2-probe-size 256
--batch-size 512
```

The expected server command is still one launcher run, for example:

```bash
cd /hy-tmp/Reinforcement-For-Robustness
git pull --ff-only

export HF_HOME=/hy-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

CUDA_VISIBLE_DEVICES=0,1,2,3 bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --stage2-k-trials 4 \
  --stage2-probe-size 256 \
  --batch-size 512 \
  --blb-v3-reward-devices 0,1,2,3 \
  --fresh
```

Verification checklist:

- A smoke run logs `[multi-gpu] reward probe enabled: devices=[0, 1, 2, 3]`,
  `[probe-runner] worker 0: cuda:0`, and workers 1/2/3 on cuda:1/2/3.
- `nvidia-smi` shows all four GPUs active during the model-forward reward probe.
- The metrics aggregation still uses all 4 trials for each action.
- Single-GPU fallback remains valid when only one GPU is visible or
  `--blb-v3-reward-devices` is unset.

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
6. Reward is hard-priority: invalid/accuracy failure first, model stability
   second, then cost. `Rescale_optimizer` contributes optimizer cost /
   feasibility diagnostics only; it must not skip or replace the actual model
   forward reward. Cost must never compensate for an accuracy or stability
   failure.
   Reward v3 uses metric1 + metric2 gates and includes metric1_std/metric2_std in
   the stability gate, but those metric std channels must tolerate normal
   small-K MRPC probe quantization. Historical K=5 evidence remains useful, but
   current four-GPU runs use K=4. Do not use a tiny `1e-3` metric-std floor:
   the 2026-05-20 reward-v3 run at commit `6f3d618` failed at 345 episodes with
   P1=0, invalid=0, loss-cap=0 solely because normal metric-std jitter dropped
   58 otherwise healthy episodes into P2 and pushed rolling300 below 35. Current
   behavior keeps tiny metric std jitter in P3 via a `1e-2` floor while still
   treating materially large metric std as P2.
   Current cost reward is budgeted adaptive scalar in the sequential Stage-2
   path. Only P3 candidates (accuracy and stability pass) receive cost reward.
   P3 shaping is split into a small metric-margin budget and a cost-led budget
   so extra accuracy margin cannot crowd out cost ranking. Fusion gain and
   truncation/K gain are interval-style boosts: each +1 fusion or each coarse
   layer-equivalent K tier (derived from average-K gain with default step size
   `1/12`) gives a clear scalar jump inside the P3 tier. The `1/12` K tier was
   chosen after an offline sweep over real 2026-05-23 fast-reward episodes:
   the older `1/59` single-slot K tier made roughly 27.5% of P3 candidates hit
   the P3 cost clip too early, while the `1/12` layer-equivalent tier kept
   saturation near 9% and preserved visible fusion/K ordering. Total bits is a
   separately clipped weak linear tie-breaker and must stay smaller than a
   fusion/K tier step. The bounded `terminal_cost_score` remains the PPO
   shaping signal for stability, but it is no longer the only ordering signal:
   `terminal_cost_rank_score` is P3-only and unbounded, with component fields
   for fusion, truncation/K, and bits. Best-action selection, top-candidate
   diagnostics, candidate-store ranking, and promotion/frontier seeds should
   use hard priority first and then this unbounded rank inside P3. P1/P2 keep
   `terminal_cost_rank_score=0`, so cost still cannot compensate for accuracy
   or stability failure.
   `ParetoCostArchive` may still record P3 frontier rows for
   diagnostics/exploration statistics, but Pareto events are not the default PPO
   scalar reward.
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

Current sequential policy/search design as of 2026-05-21:

- `BLBStage2SequentialPolicy` is a v2-scale causal GTrXL token model, not the
  old two-layer MLP. The default shape is `d_model=256`, `n_heads=8`,
  `n_layers=4`, `d_ff=512`, `dropout=0.1`. Inputs include step/layer/block
  embeddings, previous action embeddings, previous optimizer signals, static
  features, and a current-step token marker.
- Actor output uses per-slot heads sized from the live sequential environment:
  one head per padded `max_step_dim` slot, each producing up to 6 level logits.
  On the current MRPC/BERT-base server run this is `max_step_dim=24`
  (`per-slot heads=[24 x 6]` in the startup log). Do not hard-code the older
  "13 heads" wording; use `step_schedule_max_dim(...)`/`seq_env.max_step_dim`
  and let the existing slot mask plus per-level `action_level_mask` define the
  legal categorical support for each step. The critic is a single value head
  `Linear(256,64) -> Tanh -> Linear(64,1)`.
- Action heads are orthogonal-initialized with gain `0.01`. Warmstart is no
  longer a permanent learned bias inside the actor head; it is an external
  decaying baseline logit prior, and every transition stores
  `baseline_prior_scale` so PPO can replay the exact collection distribution.
- Baseline prior schedule for fresh sequential runs: anchor episodes use
  `1.2`; episode 60 starts at `1.0`; episode 60-600 decays to `0.45`; episode
  600-2000 decays to `0.15`; after episode 2000 it stays at `0.15` as a weak
  safety prior. Default forced-baseline anchor is exactly 60 episodes unless
  `force_baseline_episodes` or `warmstart_anchor_episodes` overrides it.
- PPO update now includes running return normalization, clipped Huber value
  loss on normalized returns, MAD-clipped advantage normalization, approximate
  KL stats, KL early stop, adaptive LR scaling, and per-slot entropy recovery.
  Checkpoint/resume stores policy state plus PPO auxiliary state.
- Exploration is non-monotonic cost-boundary search. Do not assume lower SF is
  closer to the metric/stability boundary. SF/K moves are proposal directions
  only; the true boundary direction comes from F1 model-forward metrics,
  stability, Rescale_optimizer cost signals, and Pareto archive events.
- Safe neighbor masks are bidirectional around the selected base action for SF
  slots; K locality is by truncation-bit distance through non-monotonic
  `K_LEVELS`, not by categorical index or "lower is better". Non-selected
  slots stay fixed at the selected base action.
- Each episode may seed its local mask from the static baseline or a recent
  Pareto-frontier action. `GuardedRadius2Controller` maintains empirical
  per-offset stats: P3 successes, P1/P2/loss-cap/stability failures, Pareto
  event counts, and mean cost-vector changes. Radius2 may sample only offsets
  with at least three P3 successes and zero failures; any radius2 P1/P2,
  invalid, loss-cap, or stability violation triggers cooldown.
- Store the exact per-transition `action_level_mask` and
  `baseline_prior_scale` used during collection and replay both during
  `sequential_ppo_update`. Recomputing support or prior scale during PPO update
  breaks the PPO ratio.
- Build mutable offsets from `describe_action_vector(...)` and exclude inactive
  compatibility slots, layer-0 block-1 pseudo slots, first-input compatibility,
  and single-level dimensions.
- The 2026-05-20 collapse at episode 121 was optimizer-valid but
  accuracy-catastrophic (`any_invalid=False`, `loss_mean=100`, P1(acc)), so the
  optimizer-invalid blacklist alone cannot protect terminal model-forward
  reward. Keep the forced anchor, blacklist, fallback baseline, cooldown, and
  health gates.
- K=5 / probe_size=256 noisy probes can make the all-max baseline fall one
  discrete probe sample below `noisy_baseline_metric1 - stage2_limit_tolerance`.
  Sequential accuracy threshold calibration must subtract a one-sample guard
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
usually needs 50,000+ episodes/rounds. The user expects a healthy run to enter
a rapid reward-growth phase sometime after roughly 20,000 episodes; if a 60k
run is still flat well past that point, treat it as a training/search pathology
to diagnose instead of blindly spending the remaining budget. Short runs such
as 200 episodes are for plumbing, performance, and regression smoke only; do
not treat their reward quality as evidence that the RL search worked or failed.

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
