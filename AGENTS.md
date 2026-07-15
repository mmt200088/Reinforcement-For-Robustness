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
- After finishing any user task, include a concise audit summary of what was
  done and how it matches the user's requested direction. Keep it high-level
  unless the user asks for detailed evidence; the purpose is to let the user
  quickly check that the work stayed aligned with the requested goal.
- Server GPU time is expensive. When the user has approved server-side work,
  keep the rented hardware busy and use available CPUs/GPUs efficiently rather
  than leaving them idle. Do this within the active objective only: do not start
  unapproved experiments, do not interfere with running jobs, and report back
  when the next useful server action needs user direction.
- Current Codex role in this project, added 2026-07-02: focus on runtime
  efficiency optimization across code produced by the other agents. This means
  profiling, reducing wall-clock time, improving CPU/GPU parallelism, reducing
  avoidable I/O or serialization overhead, and keeping approved server hardware
  well utilized. Do not change research semantics, reward/evaluation validity,
  or reported scientific conclusions merely to make runs faster; preserve
  outputs unless the user explicitly asks for a behavior change.
- Stage-2 prerequisite GELU source, updated 2026-07-11: Stage-2 currently
  defaults to `--stage2-fixed-config-source all4`, resolving GELU to degree 4
  and Softmax to degree 6 in every layer. This is a reversible experiment
  setting, not a permanent restriction: pass `stage1_result`, `json`, or
  `manual` explicitly to use another prerequisite configuration. The resolved
  pair is the single source for Stage-2 training, action mapping, model
  inference, and saved/final evaluation. Under `all4`, every layer's Block 5
  graph must resolve to `block5_n4`; keep `block5_n1/n2` maps for switch-back
  and historical reproduction.
- GPU utilization clarification, added 2026-07-02: when a task is semantically
  suitable for GPU or multi-GPU execution, prefer moving it off CPU and onto
  the server GPUs instead of leaving expensive hardware idle. Treat this as one
  optimization angle, not the only one: also consider batching, async overlap,
  data movement, caching, serialization, process/thread scheduling, and
  algorithmic hot-path reduction. Do not force GPU use for workloads already
  shown to be CPU-bound by design, such as pure-Python Rescale_optimizer replan
  enumeration, unless profiling identifies a correct GPU-equivalent path.
- Concurrent-agent note, added 2026-07-02: another agent is actively modifying
  the Stage-2 RL algorithm code. Efficiency work should avoid overlapping those
  core Stage-2 RL algorithm files unless the user explicitly coordinates the
  handoff; prefer low-conflict launcher, gate, profiling, reporting, or
  server-orchestration changes for now.
- Holistic optimization scope, added 2026-07-02: do not repeatedly optimize one
  visible hotspot while ignoring the rest of the project. Before substantial
  efficiency work, build or refresh a whole-flow map covering launcher/presets,
  Stage-1 plaintext RL and evaluation, Stage-2 BLB RL/reward probes, fusion-map
  and Rescale_optimizer replan paths, Paean/final evaluation, structured data
  capture, reports, server bridge, and artifact sync. Reason about ordering,
  dependencies, shared artifacts, and handoff contracts, then choose
  optimizations by end-to-end wall-clock impact and hardware utilization rather
  than by isolated local speedups.
- Runtime-optimization completion audit, updated 2026-07-14: source `8308bbd`
  was verified in the clean five-RTX-5090 checkout. The six-stage audit found
  all 30 expected flow files and every required artifact-evidence class. The
  complete CPU/no-GPU server gates are now green: `unittest` passed 1,509 tests
  with 6 condition skips, and `pytest` passed 1,599 tests with the same 6
  explicit CUDA/Python-3.9 skips. Source `3de5244` restored production Stage-2
  rollout efficiency after the `24e919c` merge: cached immutable probe
  assignments, preallocated result slots, cached static schedule/fusion-mask
  tensors, causal-prefix lengths that avoid current-step CUDA scalar sync, and
  batched PPO tensor materialization. Source `66a4895` restored the shared list
  parser in the runner; `8308bbd` only aligned integration fixtures and
  floating-point test tolerance. Reward, constraints, trial counts, and
  validation semantics were not changed. Evidence is under
  `experiments/server_command_runs/project_completion_audit_8308bbd_20260714/`.
  This closes the repository-wide software audit, not the whole optimization
  goal. The first strict production 1GPU-versus-5GPU gate subsequently ran at
  source `ba8bb14` and exposed both a deterministic-result defect and a
  throughput shortfall; see the red-result entry below.
- Runtime-optimization server status, updated 2026-07-14: the current server is
  `root@i-1.gpushare.com:5782` with five RTX 5090 GPUs (32,607 MiB each), 256
  logical CPUs, 629 GiB RAM, and a 50 GiB `/hy-tmp` volume. PyTorch
  `2.9.1+cu128` sees all five `sm_120` devices. The clean optimization checkout
  is `/hy-tmp/rfr_runtime_optimization` and must continue to receive source via
  Git only. The former formal process and its post-run GPU jobs are no longer
  active; do not infer from process exit that its scientific completion
  protocol succeeded. At the latest check all five GPUs were idle at `0 MiB`,
  and the first isolated gate had already completed.
- Stage-2 strict 1GPU-versus-5GPU red result, added 2026-07-14: the first
  isolated 600-episode production A/B ran at source `ba8bb14` after the idle
  guard verified that all five RTX 5090s were free. The 1GPU case completed in
  `2251s` (`959.574` episodes/hour); the 5GPU case completed in `1171s`
  (`1844.577` episodes/hour), only `1.922x`, below the required `3.4x`.
  Episode quality/effect equality and PPO-update equality both failed from the
  first episode/update. Physical coverage passed: every requested GPU was
  sampled active, but each 5GPU mean utilization was only about 31% versus
  92.73% on the single GPU. The equal intended trial-seed records but unequal
  baseline samples trace the correctness defect to `ProbeWorker.run_trial()`:
  it seeds global PyTorch/NumPy RNGs but does not reseed the independent
  per-device generator actually consumed by `function_handler` BLB noise. Fix
  that boundary with TDD before measuring the performance bottleneck. Then use
  direct install/probe timing to optimize the proven host/setup hotspot before
  rerunning the strict 600-episode gate. Evidence is under
  `experiments/server_command_runs/stage2_ngpu_speed_ab_ba8bb14_20260714_223042/`,
  with both raw mirrors under `rl_training_data_points/stage2/bert-base/mrpc/`.
- Stage-2 seeded 1GPU-versus-5GPU smoke, added 2026-07-15: source `14187ee`
  fixed `ProbeWorker.run_trial()` so each `(base_seed, trial_idx)` reseeds the
  dedicated per-device BLB-noise generator without mutating global PyTorch or
  NumPy RNG state. The server passed 13 deterministic-lock tests, a 37-test
  focused probe/seed suite, and a direct `cuda:0`/`cuda:1` replay with exact
  loss/accuracy/F1 equality. The 170-episode full-path smoke then took `642s`
  on 1 GPU and `331s` on 5 GPUs (`1.940x`); all five cards were active, but
  their mean utilization was only 28.38%-30.33%. Full episode/PPO equality
  still failed because only `build_probe_runner()` enables TF32/high matmul
  precision: actions, trial seeds, accuracy, and F1 were identical at episode
  0, while per-trial loss differed by about `3e-5` to `8e-5`. Enable the same
  fast-matmul mode before any sequential Stage-2 baseline/probe on both the
  single- and multi-device paths, prove exact short-run equality, then expose
  the environment's existing install/probe/clear timing through layerwise
  records before optimizing the host/setup hotspot. Evidence is under
  `experiments/server_command_runs/stage2_ngpu_seed_equality_170ep_14187ee_20260715_001107/`,
  with both structured raw mirrors under
  `rl_training_data_points/stage2/bert-base/mrpc/`.
- GPU activity-report rule, added 2026-07-13: do not infer that a visible GPU is
  idle solely because layerwise episode rows omit `terminal_probe_devices`.
  `scripts/gpu_utilization_report.py` must preserve probe attribution as a
  separate field and use sampled `nvidia-smi` utilization when classifying
  actual idle devices. Evidence at source `19c5a10` is under
  `experiments/server_command_runs/gpu_activity_classification_19c5a10_20260713/`.
- BLB install-log rule, added 2026-07-13: `_print_blb_install()` is quiet when
  `BLB_NOISE_INSTALL_LOGS` is unset; set `BLB_NOISE_INSTALL_LOGS=1` only for an
  explicit diagnostic run. A streamed profile of the active source `24e919c`
  log at 8,520 episodes found 1,543,515 install lines occupying 344,739,386
  bytes, or 99.9865% of the file, projecting to about 2.43 GB at 60,000
  episodes. Source `cb9a34c` passed RED/GREEN, compile, and explicit opt-in
  behavior gates. It affects future processes only; do not restart or mutate
  the formal run that started from the old source. Evidence is under
  `experiments/server_command_runs/blb_install_log_quiet_cb9a34c_20260713/`.
- Stage-2 final A/B handoff rule, updated 2026-07-13: source `28ab0c0` aligns
  `scripts/stage2_ngpu_speed_ab.sh` with the formal layerwise command. The gate
  now uses `--blb-v3-reward-devices` K-trial splitting, batch 64, rollout 120,
  all-4 Stage-1 fixed config, and `robust_constrained` reward; it removes the
  incompatible `--stage2-rl-devices` path and legacy neighbor/warmstart/forced-
  fusion overrides. `PRINT_EFFECTIVE_COMMANDS=1` records the exact shared
  command arrays without querying GPUs. Command-contract evidence is under
  `experiments/server_command_runs/stage2_layerwise_ab_contract_28ab0c0_20260713/`.
  This established command readiness only. The first strict run is now the red
  `ba8bb14` evidence above; it must not be treated as parity or speed proof.
- Stage-2 A/B scheduling-command integrity, updated 2026-07-14: source
  `a61300d` makes the A/B preflight and actual launch share one environment
  array and explicitly forwards each case's worker count. RED source `9fe4d0c`
  proved that non-default worker/policy/dynamic values were previously logged
  but absent from the effective command. The server passed all 18 related
  launcher/comparator tests and the full 1,599-test pytest gate. A live idle
  check correctly exited `20` before launch while formal PID `10089` owned the
  GPUs; no second `rl_tune.py` process appeared. Do not misinterpret these
  legacy scheduling knobs as the production layerwise speed mechanism:
  `stage2_workers_per_device`, `BLB_STAGE2_POLICY_DEVICE`, and
  `BLB_STAGE2_DYNAMIC_ASSIGNMENT` are consumed by the mutually exclusive
  `stage2_rl_devices` episode-parallel branch. The pending production gate uses
  `reward_devices` deterministic K-trial splitting and must remain fixed at the
  declared `1:worker:1` defaults. Reuse the formal run's existing read-only
  cache and dataset paths explicitly because `/hy-tmp/hf_cache` and
  `/hy-tmp/glue_data` do not exist on this host. Evidence is under
  `experiments/server_command_runs/stage2_ab_scheduling_overrides_a61300d_20260714/`.
  At capture time the formal run was healthy at 15,720/60,000 episodes (26.2%)
  with about 20.0 hours estimated remaining.
- Stage-2 A/B physical-GPU coverage gate, added 2026-07-14: RED source
  `04f2b2e` proved that `stage2_ngpu_ab_compare.py` counted only the first
  device in a multi-device `terminal_probe_devices` row and that the A/B runner
  collected `nvidia-smi` CSVs without making actual GPU activity a pass/fail
  condition. Source `1b8fa08` counts every participating device and trial, runs
  `gpu_utilization_report.py` after each case, and fails unless every requested
  physical GPU has sampled activity above the declared threshold. Reused 1GPU
  baselines must carry and revalidate their original GPU samples. Server gates
  passed all 38 related tests and the complete thread-capped, CUDA-hidden pytest
  suite (`1,602` passed, `6` environment skips). Evidence is under
  `experiments/server_command_runs/stage2_ab_gpu_coverage_1b8fa08_20260714/`.
  A follow-up passive 60-second real-hardware check at source `1dd466f` passed
  the strict gate with empty episode device attribution: sampled mean GPU
  utilization was 33.13%-34.95%, every `cuda:0..4` maximum exceeded 42%, and
  formal PID `10089` remained the sole compute process. Evidence is under
  `experiments/server_command_runs/stage2_ab_gpu_coverage_passive_1dd466f_20260714/`.
  This hardens the pending benchmark but does not replace it. At capture time
  formal PID `10089` was still active at 16,832/60,000 episodes; its latest 100
  rows had 42 P1, 9 P2, and 49 P3 outcomes, zero invalid episodes, and zero
  loss-cap sentinels. Treat the high P1 frequency as an algorithm-health
  warning for the Stage-2 owner, and do not change or stop that run from the
  efficiency workstream.
- Stage-2 production integration status, updated 2026-07-14: merge commit
  `cd6fb55` integrated published source `24e919c` without touching its active
  formal run directory. Source `87a57a1` then restored the canonical optimizer
  write-back, installed probe inference, and ProbeRunner diagnostics paths that
  historical integration commit `16b68e3` had expanded into duplicate code.
  Deterministic noise scope, device locks, trial seeds, and CUDA streams remain;
  only model forward is lock-scoped, while metric kernels and the batched host
  synchronization execute after lock release. The final server gate passed
  Bash syntax, Python compilation, exact A/B command preflight, and 117 focused
  `unittest` cases (one CUDA-only skip). The image lacks pytest, recorded as an
  environment dependency rather than a passing pytest claim. Evidence is under
  `experiments/server_command_runs/stage2_shared_path_integration_20260714/`.
  At evidence packaging the formal run was at 12,360/60,000 episodes; do not
  start the strict GPU A/B until its process exits and the idle check passes.
- Stage-2 shared evaluation-chain rule, added 2026-07-02: do not reimplement
  the BLB action-to-model pipeline in ad hoc callers. Optimizer/replan cfg
  write-back must go through
  `blb_stage2_rl.optimizer_cost.apply_optimizer_outputs_to_cfgs()`. The actual
  already-installed model inference loop must go through
  `blb_stage2_rl.inference_eval`: use
  `run_installed_model_on_dataloader()` for full-validation/final-eval style
  dataloaders and `run_installed_probe_trial()` for online Stage-2 reward probe
  trials. When deterministic probe execution needs a noise scope, shared-device
  lock, or CUDA stream, pass a forward-only context to
  `run_installed_probe_trial()` so metric aggregation and device-to-host sync do
  not extend the lock's critical section. Repeat-evaluation JSON payloads and
  mean/std trial summaries must use
  `blb_stage2_rl.eval_metrics.pack_repeat_evaluation()` /
  `summarize_eval_trials()`; do not hand-roll per-caller `trial: i + 1`,
  `loss_mean`, `p_std`, or `time_mean_ms` structures. If new special handling is
  needed, extend those shared modules and their tests instead of adding another
  local forward/metric/repeat implementation in RL, Paean, or experiment
  scripts.
- Stage-2 shared action-identity rule, added 2026-07-02: all raw BLB action
  vector identity hashes must use `blb_stage2_rl.candidate_store.action_hash()`
  (or the explicit raw/effective hash helpers in that same module). Do not
  hand-roll JSON normalization plus SHA256 in diagnostics, reports, candidate
  stores, or experiment scripts; extending `candidate_store.py` is the shared
  seam if action identity semantics need to change.
- Stage-2 shared probe-diagnostics rule, added 2026-07-02: ProbeRunner timing
  and worker split snapshots must be serialized with
  `blb_stage2_rl.probe_runner.diagnostics_payload()`. Do not hand-roll
  `per_worker_seconds`, `per_worker_trial_counts`, `speedup_vs_sequential`, or
  formatted probe-runner lines in env/runner/report code; extend
  `ProbeRunnerDiagnostics`, `format_diagnostics_line()`, and
  `diagnostics_payload()` together.
- Shared JSON-normalization rule, added 2026-07-03: reports, final-eval
  payloads, RL data-point writers, diagnostics, and experiment artifacts must
  use `json_utils.to_jsonable()` for numpy scalars/arrays, dataclasses, paths,
  and optional torch tensors. For `json.dump(..., default=...)` / `json.dumps`
  adapters, use `json_utils.json_default()`. For deterministic JSON identity
  strings or SHA256 hashes, use `json_utils.stable_json_key()` and
  `json_utils.stable_json_hash()`. For JSON artifact files that need parent
  directory creation plus normalized `indent=2` output, use
  `json_utils.write_json_file()` instead of script-local `_write_json` helpers.
  For JSON artifact reads, use `json_utils.read_json_file()` instead of
  script-local `json.loads(path.read_text(...))` loops in core
  RL/Paean/final-eval/report tools, fusion-count map loaders, Stage-2
  max-SF/action-mask/skeleton config loaders, and BLB diagnostic scripts.
  When a report reads optional JSON sidecars, pass `default={}` (or the needed
  default value) to `read_json_file()` rather than adding a local `_read_json`
  helper that swallows missing/invalid artifacts.
  Do not add local `_json_ready`, `_jsonable`, `_json_safe`, `_json_default`,
  `json_default`, `_stable_json`, `_sha256_json`, `_read_json`, `_load_json`,
  or `_write_json` helpers in core RL/Paean/final-eval code or standalone
  experiment scripts; keep legacy-named helpers only as thin wrappers around
  `json_utils` when compatibility requires them. Extend `json_utils.py` and its
  tests when a new serializable type or JSON artifact convention is needed.
- Shared report-format helper rule, added 2026-07-03: lightweight HTML/metrics
  reports must use `report_format_utils.py` for common table and number
  rendering. Use `html_table()` for small escaped tables, enabling
  `allow_html_cells=True` only for cells that are already intentionally
  rendered HTML; use `format_float()` for compact numeric cells and
  `metric_float()` for tolerant metric extraction from JSON dictionaries. Use
  `progress_bar()` and `format_elapsed()` for training/search progress logs
  rather than local `_progress_bar`, `_seq_progress_bar`, `_fmt_elapsed`, or
  `_seq_fmt_elapsed` implementations. Do not add new script-local
  `_html_table`, `_fmt`, or `_metric` helpers in report/experiment scripts;
  extend `report_format_utils.py` and `tests/test_report_format_utils.py` when
  a report or log needs a new shared formatting convention.
- Shared report-stat helper rule, added 2026-07-03: lightweight report,
  monitor, and benchmark scripts must use `stats_utils.py` for simple
  means, sorted medians, fractions, count-based ratios, and safe division.
  Do not add new script-local `_mean`, `_mean_or_none`, `_mean_seconds`,
  `_median_sorted`, `_frac`, `_frac_counts`, `_mean_counts`, or `_safe_div`
  helpers; extend `stats_utils.py` and `tests/test_stats_utils.py` when a
  report needs another small reusable statistic. Keep core model/reward
  semantics in their owning modules rather than moving domain-specific
  metrics into this generic helper.
- Shared CSV-field helper rule, added 2026-07-03: report and monitor scripts
  that parse CSV headers must use `csv_field_utils.py` for tolerant field-name
  normalization and first-present lookups. Use `normalized_field_index()` for
  `csv.reader` rows, `normalized_field_lookup()` for `csv.DictReader`-style
  rows, `first_present_index()` when the caller needs the matched column
  position rather than the row value, and set `keep_first=True` only when
  preserving the first duplicate normalized header is required. Do not add
  script-local `_normalize_header`, `_first_header_index`, `_normalized_row`,
  `_normalized_field_lookup`, `_normalized_field_index`,
  `_first_present_by_lookup`, or `_first_present_by_index`; extend
  `csv_field_utils.py` and `tests/test_csv_field_utils.py` instead. For simple
  finite CSV artifact output that writes a header and projects mappings onto a
  fixed field list, use `csv_field_utils.write_csv_rows()` instead of
  script-local `_write_csv` / `write_csv` helpers. For finite experiment CSV
  artifacts that intentionally infer the header from the first row and no-op on
  empty row lists, use `csv_field_utils.write_csv_rows_with_inferred_fields()`
  instead of local `write_csv(path, rows)` wrappers. Keep specialized streaming
  or append-only CSV writers local when they intentionally manage migration,
  rolling-window state, or trace buffering.
- Shared JSONL reader rule, added 2026-07-03: report, monitor, verifier,
  registry, and diagnostics scripts that consume JSONL artifacts must use
  `jsonl_utils.py` for common blank-line handling, malformed-line policy,
  dict-only filtering, and missing file behavior. Use
  `iter_jsonl(..., errors="skip")` for live logs that may contain partial/bad
  rows, `iter_jsonl(..., errors="raise")` for verifier scripts that should
  report `path:line`, and `read_jsonl(..., missing_ok=True)` for optional
  artifacts. Use `gzip_fallback=True` when a canonical `*.jsonl` artifact may
  also be stored as `*.jsonl.gz`; do not write script-local gzip/open fallback
  logic. Use `read_jsonl_fields()` for report tools that need only a small
  projection from large rows, `read_jsonl_xy()` for direct scatter/curve point
  projections, and `count_jsonl_with_required_fields()` for verifier row counts
  with required-field diagnostics. Do not add new one-call script-local
  `_iter_jsonl`, `_read_jsonl`, `_count_jsonl`,
  `_count_jsonl_with_required_fields`, or raw
  `for line in handle: json.loads(line)` loops in report, registry, or
  diagnostics scripts; extend `jsonl_utils.py` and `tests/test_jsonl_utils.py`
  if another JSONL convention is needed. A local wrapper is acceptable only
  when it adds real path resolution semantics, such as converting a run
  directory into its `episodes.jsonl` path before delegating to `iter_jsonl()`.
  For finite report/diagnostic JSONL artifacts, use
  `jsonl_utils.write_jsonl_rows()` instead of script-local `_write_jsonl`
  helpers; keep high-throughput append-only RL training writers local when they
  intentionally manage buffering, flush cadence, and reusable open handles.
- Shared numeric parser rule, added 2026-07-03: report scripts that need to
  pull a number out of a metric string must use
  `numeric_parse_utils.parse_first_float()`. Do not add local `FLOAT_RE`
  constants or `_float_value` helpers in report scripts; extend
  `numeric_parse_utils.py` and `tests/test_numeric_parse_utils.py` if another
  tolerant numeric parsing convention is needed.
- Shared text-line iterator rule, added 2026-07-03: report and monitor scripts
  that need to feed an in-memory command-output string into an existing
  line-oriented parser must use `text_utils.iter_text_lines()`. It preserves
  file-handle-style trailing `\n` and intentionally splits only on `\n`, not
  all Unicode line separators. Do not add script-local `_iter_text_lines`
  helpers; extend `text_utils.py` and `tests/test_text_utils.py` if another
  in-memory text iteration convention is needed.
- Shared static-test source inspection rule, added 2026-07-03: tests that
  statically inspect project source for helper reuse must use
  `tests/source_inspection_utils.py`. Use `source_text()` for source reads and
  `function_names()` for AST function-name sets; do not copy local
  `_function_names` helpers across `test_*_utils.py` files.
- Fusion-count fixed-action experiment helper rule, added 2026-07-03:
  Paean-path and RL-path fixed-action evaluation scripts must share action
  config directory scanning, JSON-list parsing, stable JSON hashes/keys, and
  duplicate-action folding through
  `scripts/fusion_count_action_eval_common.py`. Use
  `resolve_repo_path()` for repo-relative CLI paths,
  `load_paean_action_configs()` / `unique_paean_action_configs()` for Paean
  final-eval drivers, and `load_rlpath_action_configs()` /
  `rlpath_config_group_key()` / `unique_rlpath_action_configs()` for RL-path
  drivers. Do not add script-local `_resolve`, `_load_action_configs`,
  `_json_int_list`, `_group_key`, `_config_group_key`, or `_unique_configs`
  wrappers when adding a new fusion-count experiment; extend the common helper
  and its tests if the shared contract needs a new variant.
- Shared CLI numeric parser rule, added 2026-07-03: small scripts that parse
  command-line integer/float vectors and Paean/final-eval action vector text
  fields, Stage-2 environment/config integer lists, and Paean/final-eval action
  vector text fields must use `cli_parse_utils.py`. Use
  `parse_json_int_list()` for JSON-list flags with defaults,
  `parse_exact_json_int_list()` for exact-length JSON vectors,
  `parse_optional_int_list()` / `parse_int_list_text()` for comma/semicolon
  integer lists, `parse_float_list_text()` for comma/semicolon float lists,
  `parse_broadcast_int_vector()` for one-value-or-per-layer degree flags, and
  the legacy RL entrypoint helpers `parse_degree_config()`,
  `parse_noise_config()`, `parse_bool_flag()`, `parse_positive_int()`,
  `parse_optional_positive_int()`, `parse_stage1_episode_limit()`, and
  `parse_optional_positive_float()` for `rl_tune*.py` Fire arguments. Keep
  script-local `_json_int_list`, `_parse_degree_vector`, `parse_bool_flag`, or
  `parse_positive_int` only as compatibility wrappers around this seam; private
  one-call `_parse_int_list` / `_json_list` wrappers are not compatibility and
  should be replaced by direct calls to `cli_parse_utils`. Do not reimplement
  parser bodies in new launchers or RL entrypoints, including local
  `split(",")` loops for comma-separated numeric lists.
- Shared command-format helper rule, added 2026-07-03: command logging in
  launchers, final-eval wrappers, and error summaries must use
  `runtime_error_reporter.format_command()` for shell-escaped argv rendering.
  Do not add new local `" ".join(shlex.quote(...))` / `format_command`
  implementations; re-export or import the shared helper when a module needs to
  preserve a historical function name.
- Shared device-list parsing rule, added 2026-07-03: Stage-1 and Stage-2 GPU
  device flags must use `device_utils.parse_device_ids()` for int/list/tuple,
  comma-separated string, and Python Fire tuple-string forms. `stage1_rl` and
  `blb_stage2_rl.probe_runner` may re-export the helper for compatibility, but
  do not clone new parser variants in launchers, runners, or reports. When a
  script needs raw CUDA_VISIBLE_DEVICES/UUID tokens rather than integer ids,
  split with `device_utils.split_device_spec_tokens()` and preserve the raw
  tokens. When a report or monitor needs logical diagnostic names such as
  `cuda:0`, use `device_utils.normalize_logical_device_token()` or
  `device_utils.parse_logical_device_spec()` instead of script-local
  `cuda:` normalization or `str(spec).replace(";", ",").split(",")` parsing.
- All Stage-1 and Stage-2 RL runs must mirror raw training data points to the
  project-root `rl_training_data_points/` tree, classified by stage, model,
  dataset, and run id. Persist enough structured JSON/JSONL to redraw paper
  figures without rerunning training: manifest/config/baselines/constraints,
  per-step data when available, per-episode rewards/metrics/cost/action
  choices, PPO update diagnostics, throughput/parallelism fields, best-so-far
  state, and final summary. PNG/NPZ outputs are inspection artifacts only; do
  not launch new RL training with this structured writer disabled unless the
  user explicitly waives the requirement.
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
- Stage-2 natural-convergence rule, added 2026-07-15: formal layerwise
  robust-constrained PPO runs use `--stage2-search-episodes 0`, meaning no
  fixed episode budget. The active PPO objective has entropy regularization
  disabled (`ent_coef=0.0`) for the entire run; Block4 and K entropy are
  diagnostics and stop signals only. Stop after a complete PPO update only
  when both normalized entropies are below `0.1`, a promotion-qualified robust
  feasible candidate exists, and its cost frontier has not improved for 100
  complete finite PPO updates; skipped/non-finite updates do not advance this
  patience. Positive episode limits are smoke/debug budgets and do not imply
  convergence. Do not restore entropy schedules, floors, recovery, or forced
  concentration to make the policy stop.
- Stage-2 fusion-count map build status, updated 2026-06-03: commit `ea27408`
  fixed the builder-side enum domain so rescale slots enumerate only SF choices
  (`index 1..levels-1`) and never `index 0`/`None`, preserving the CKKS rule
  that RL should not decide whether a must-exist rescale operation occurs.
  Commit `41ce2f3` then fixed two generated-map artifact issues: option
  `slots` now contain SF/K-first values, and `FusionCountMap.load()` skips
  sidecars such as `_summary.json`. The clean server rerun at
  `experiments/server_command_runs/fusion_map_build_20260603_142923/` had
  `fusion_unit_exit=0` and `build_exit=0`, with non-empty active-rescale sets,
  `K-indep=True`, option 0 equal to the all-max baseline for all seven MRPC
  block maps, and non-empty `slots` for every option. `block1_mrpc` and
  `block4` are accepted true degeneracies with one option and `fusion=[0]`;
  the other five block maps have two options with `fusion=[0,1]`. Codex left
  these generated maps in the run's `fusion_maps_snapshot/` rather than
  promoting them to canonical `blb_stage2_rl/fusion_maps/mrpc/`; promote only
  after the follow-up artifact commit/review step.
- Stage-2 fusion-count runtime smoke status, added 2026-06-03: commit
  `18b2975` was packaged from the local verified source into a server temporary
  run directory and executed with the active `SERVER_COMMAND.md` fusion-count F1
  smoke. Because the server main worktree had no `Parting Chapter/stage1/record`
  artifact, Codex added a minimal temporary Stage-1 record in that temp run
  directory only (`gelu=[4]*12`, `softmax=[6]*12`) to satisfy `stage2-only`
  prerequisite loading; this was not a source edit. The clean rerun artifacts
  are in
  `experiments/server_command_runs/fusion_smoke_20260603_190037/`. The server
  passed all 20 `tests/test_blb_fusion_count_map.py` tests, including the
  torch-backed fusion-schedule tests that are skipped locally. The launcher
  started background PID `1108319`; Codex monitored that PID after the launcher
  returned because `smoke_exit=0` only means the launcher succeeded. The real
  background run exited before reaching `Fusion-count action ENABLED`, before
  four-GPU reward-probe engagement, and before writing `episodes.jsonl`. The
  blocker is a pre-fusion baseline handoff error:
  `Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json` lacks
  `block3_exp_n6@layer=0..11`, while Stage-2 now fixes Softmax degree to `6`.
  Treat this as a static skeleton/archive compatibility issue to fix locally
  and rerun; do not treat the end-to-end fusion-count runtime path as verified
  or failed past that precondition yet.
- Stage-2 fusion-count manual-degree rerun status, added 2026-06-03: commit
  `48ff2c8` changed the F1 smoke command to pass
  `--stage2-manual-gelu [4]*12` and `--stage2-manual-softmax [2]*12`, intending
  to bypass Stage-1 record lookup and the independent `block3_exp_n6` archive
  issue. Codex reran that command from a clean local `git archive` package on
  the server; artifacts are in
  `experiments/server_command_runs/fusion_smoke_20260603_191937/`. Server-side
  `tests/test_blb_fusion_count_map.py` again passed all 20 tests. The launcher
  command clearly passed `stage2_fixed_config_source=manual`, GELU all 4, and
  Softmax all 2, but the real background run still failed before reaching
  `Fusion-count action ENABLED`: `_resolve_stage2_fixed_stage1_config()` built
  its resolver from `final_eval_config_source=search`, entered
  `_resolve_stage1_degrees_from_record()`, and raised `FileNotFoundError` for
  missing `Parting Chapter/stage1/record`. Treat this as a manual fixed-config
  plumbing bug, not a fusion-count runtime failure. Fix locally so
  `stage2_fixed_config_source=manual/json` actually bypasses Stage-1 record
  resolution for Stage-2, then rerun the same F1 smoke.
- Stage-2 block3-removal rerun status, added 2026-06-03: commit `385c4a6`
  removed Block 3 from Stage-2 baseline bootstrap and optimizer requests. Codex
  reran the active `SERVER_COMMAND.md` from a clean local `git archive`
  package on the server, with a temporary Stage-1 record only inside the server
  temp package to satisfy `stage2-only` prerequisite lookup. Artifacts are in
  `experiments/server_command_runs/fusion_smoke_20260603_221900/`.
  `contract_gate_exit=1`: `test_blb_*.py` still has three failing assertions
  after block3 removal (`valid_block_count` expected `59` but actual is `47` in
  two tests, and all-max `bits_drop` expected `0.0` but actual is `100.0` in
  one test). Do not call the contract gate clean until those expectations or
  underlying semantics are fixed locally and rerun. The fusion-count runtime
  smoke itself completed 300 episodes with `smoke_exit=0`, printed
  `Fusion-count action ENABLED`, engaged four reward-probe workers on
  `cuda:0..3`, wrote 300 `episodes.jsonl` rows, had zero invalid steps, zero
  `loss_mean=100` collapse sentinels, all terminal priority `P3`, fusion count
  in `[0,14]`, and last-20 probe speedup about `3.99x`. This verifies the
  runtime path passed the previous `block3_exp_n6` and fixed-config handoff
  blockers, but the source line still needs contract-gate cleanup before it is
  considered fully verified.
- Stage-2 fusion full-build rerun, added 2026-06-04: Claude Code pushed
  `5ed03df`, which streams per-shard min-noise reduction so the full `block4`
  729M-combination fusion-map build should not retain hundreds of GB of valid
  configs in memory. The same commit made reward tests map-version agnostic
  because the deeper 10-level enumeration can activate real fusion for
  `block1_mrpc` and `block4`; dynamic reward normalization should now use the
  map-derived max fusion choices instead of assuming the old `4630` maximum or
  block1/block4 degeneracy. Codex launched the active `SERVER_COMMAND.md` from
  verified temp source snapshot `5ed03df` at
  `/hy-tmp/server_command_stage2_fusion_fullbuild_5ed03df_20260604_233625`.
  The server main worktree was still dirty/stale and GitHub fetch hung, so the
  run uses a local-source archive fallback. Because the temp snapshot has no
  decoupled MRPC Stage-1 record, Codex seeded a minimal temp-only
  `Parting Chapter/stage1/record/bert base mrpc 1 20260604/final_config.json`
  from `glue_final_configs_best_ppo.json` so the final `stage2-only` smoke can
  resolve GELU degrees; this is a run prerequisite artifact, not a source edit.
  The first launch mis-extracted the fenced bash block from the prose header and
  exited immediately; the relaunch extracted the first bash block after
  `## ▶ active command`. Contract gate then passed (`214` tests OK,
  `contract_gate_rc=0`), and Phase 2 began building the six cheaper maps with
  `--workers 96`. Monitor
  `experiments/server_command_runs/stage2_fusion_fullbuild_20260604_233648/`
  for `build_feasible_rc`, `build_block4_rc`, `map_summary.txt`,
  `soundness_audit.txt` with `superset_pass=True`, and smoke markers including
  `Fusion-count action ENABLED`.
- Stage-2 fusion full-build result, added 2026-06-05: the `5ed03df`
  full-build server run completed from the verified temp snapshot. Contract
  gate passed (`214` tests, `contract_gate_rc=0`). The six cheaper maps built
  successfully (`build_feasible_rc=0`): block1/block2/block5_n0/block5_n1 have
  `fusion_counts=[0,1]`, block5_n2/block5_n4 have `fusion_counts=[0,1,2]`.
  The full block4 build also passed (`build_block4_rc=0`) after enumerating
  `453400421/729000000` valid configs in `45846.83s`, producing
  `fusion_counts=[0,1]`. The 600-episode Stage-2 smoke did run the fusion-count
  branch (`Fusion-count action ENABLED: map graphs=7, max_options=3`), completed
  all `600/600` episodes, had `valid steps=47.00/47`, `invalid=0.00`,
  `recent invalid rate=0.0%`, and final training rank-best reward `+39.7372`.
  Do not treat the packaging as fully clean: the server-side push failed and
  the wrapper's `map_summary.txt` / `soundness_audit.txt` generation failed
  with `KeyError: 'options'` on sidecar/map parsing, so `map_summary.txt` is
  empty and `soundness_audit.txt` is absent. Codex pulled the generated
  canonical maps into `blb_stage2_rl/fusion_maps/mrpc/` and compact logs into
  `experiments/server_command_runs/stage2_fusion_fullbuild_20260604_233648/`,
  including a local robust `map_summary_local.txt` and smoke evidence log. The
  remaining work is a local fix to the wrapper/audit parser, not a rerun of the
  expensive map build unless the parser fix reveals a real map issue.
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
- Stage-1 mental model and speed note, updated 2026-06-04: Stage-1 is a
  plaintext-only search over per-layer GELU replacement choices. It should only
  replace GELU and the fixed Softmax approximation through
  `function_handler.py`; it should not inject BLB/noise. Current Stage-1 RL only
  decides GELU. Softmax is no longer an action and is fixed to degree `6` for
  every layer. GELU training choices are degrees `4`, `2`, and `1`; degree `0`
  / ReLU is now disabled for Stage-1 RL sampling and retained only for
  historical configs or manual evaluation compatibility. The Stage-1 RL code is
  in `layer_importance_evaluator.py`, and a BERT-base episode has 12 per-layer
  GELU decisions. Stage-1 inference tests must use the full validation set
  (`validation_full`) during both RL reward evaluation and final evaluation; do
  not switch Stage-1 online reward or final eval to the training set or a
  validation proxy to improve speed unless the user explicitly changes this
  protocol. Do not judge Stage-1 throughput from the 12 decisions alone: the
  required terminal full-validation model-forward pass can dominate runtime.
  Four-GPU Stage-1 rollout is window-style data parallelism across complete
  episodes; worker logs and sampled GPU utilization are better evidence than a
  single instantaneous `nvidia-smi` snapshot.
- Stage-1 PPO vs GRPO MRPC comparison, added 2026-05-31; superseded
  2026-06-02: run BERT-base MRPC
  Stage-1 twice, first PPO then GRPO, both to entropy convergence with
  `--stage1-search-episodes 0`, `--stage1-entropy-stop-threshold 0.1`,
  `--stage1-accuracy-tolerance 0.001`, `--stage1-search-lr 2e-5`,
  `--ppo-update-interval 120`, and `--stage1-rl-devices 0,1,2,3`. PPO output
  belongs under
  `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.001_s2t0.05_s2st0.05`;
  GRPO output belongs under
  `GRPO Chapter/persistent/rl/bert-base/mrpc/s1t0.001_s2t0.05_s2st0.05`.
  Current server monitor log root:
  `/hy-tmp/stage1_mrpc_ppo_then_grpo_entropy0p1_tol0p001_20260531_161526`.
  PPO PID `638501` is running first; wrapper PID `639985` waits for PPO exit
  before launching fresh GRPO. After both finish, build an HTML report comparing
  wall-clock speed, reward/loss/metric/entropy curves, final GELU configs, and
  full-validation inference metrics for baseline/PPO/GRPO with percentage
  deltas versus the original plaintext baseline.
- GRPO retirement, added 2026-06-02: project experiments and follow-up analysis
  showed that GRPO is not suitable for this task, especially because the MRPC
  validation proxy looked slightly better while official/test generalization was
  materially worse than PPO/baseline. GRPO must not be used for new Stage-1 or
  Stage-2 RL runs. Normal entrypoints must reject `--rl-algo grpo`; the
  launcher must not route to `GRPO Chapter`; Python evaluator/runner configs
  must accept only `ppo`. Historical GRPO reports, checkpoints, helper math, and
  analysis artifacts may remain for auditability, but they are not active
  training choices.
- Stage-1 PPO convergence queue, added 2026-06-03: by user request, keep the
  server busy with a fresh Stage-1 PPO queue for BERT-base RTE, BERT-base
  SST-2, BERT-large MRPC, BERT-large RTE, and BERT-large SST-2. Use the existing
  stage1-only presets, `--rl-algo ppo`, `--stage1-search-episodes 0`,
  `--stage1-entropy-stop-threshold 0.1`, `--stage1-rl-devices 0,1,2,3`,
  `--stage1-search-lr 2e-5`, `--ppo-update-interval 120`, and the preset
  `--stage1-accuracy-tolerance 0.005` unless the user changes it. Updated
  2026-06-05: the user changed the standing Stage-1 RL launch rule; all future
  Stage-1 RL launches must use `--stage1-accuracy-tolerance 0.0` unless the
  user explicitly changes it again. Run the tasks serially so each training job
  can use all four GPUs. The
  `SERVER_COMMAND.md` bridge now launches a background queue wrapper under
  `/hy-tmp/stage1_ppo_queue_entropy0p1_<ts>/`; monitor `status.json`,
  `events.log`, `logs/*_launch.log`, each parsed training log, the actual
  training PID, and the `Stage-1 entropy convergence reached` marker before
  advancing or declaring completion. The first launch from `adcc507` exposed a
  launcher bug: `S_BLB_V3_FUSION_COUNT_ACTION` was parsed/forwarded but not
  initialized, so `set -u` aborted before Stage-1 training. This was fixed
  locally by initializing the fusion-count flag pair. The next launch from the
  fixed source started `base_rte`, but the queue wrapper exited immediately
  after the launcher because `SERVER_COMMAND.md` used `{ ...; exit "$rc"; }`
  inside `run_one`, which exits the wrapper shell rather than only the
  redirected launcher block. Use a subshell `( ...; exit "$rc" )` and keep
  `LATEST_PID`/`LATEST_RUN_DIR` fallback parsing so the queue can monitor and
  advance. Updated 2026-06-04: the user wants each Stage-1 task summarized
  promptly before launching the next task, so do not let the queue auto-advance
  past a completed task without report generation. The current server state is
  `base_rte` running from verified temp source
  `/hy-tmp/RFR_stage1_queue_8d1bc9b_mLConE/src`, state directory
  `/hy-tmp/stage1_ppo_queue_entropy0p1_20260603_225253`, training PID
  `1116097`; its queue wrapper was intentionally paused so completion can be
  handled as `waiting_report` before launching `base_sst2`. Do not use GRPO for
  any queue item.
- Stage-1 no-degree0 PPO queue update, added 2026-06-04: after the user decided
  degree `0` / ReLU likely caused RTE overfitting, Stage-1 RL sampling was
  changed in local commit `f85c77e` so `STAGE1_GELU_ACTION_MASK` is
  `[True, True, True, False]`. Server source snapshot
  `/hy-tmp/stage1_no_degree0_queue_f85c77e_20260604_201520/src` verified that
  mask and is running a fresh serial PPO queue. Only `base_mrpc` PID `1248929`
  should be running; the accidental wrapper that launched the remaining tasks
  was stopped, and extra `base_rte`, `base_sst2`, `large_mrpc`, `large_rte`,
  and `large_sst2` processes were killed. Remaining order after `base_mrpc` is
  `base_rte`, `base_sst2`, `large_mrpc`, `large_rte`, `large_sst2`. Continue
  one task at a time, build and commit/push each compact HTML report before
  launching the next task, and do not use GRPO. Explicit user reminder,
  2026-06-04: when the active BERT-base MRPC Stage-1 run completes, build its
  HTML report first and return the local report path before moving on.
- Stage-1 no-degree0 base_mrpc completion, added 2026-06-05: the first fresh
  no-degree0 PPO task reached entropy convergence at episode `33000` with final
  entropy `0.099288`, `275` PPO updates, and raw reward-best
  `GELU=[1,1,1,1,1,1,1,1,2,1,1,1]`, Softmax fixed to all `6`, cost `49.50`,
  reward `0.537386`, first seen at episode `19812`. The compact report and
  artifacts were committed/pushed in `d47bc80`; the HTML report is
  `reports/html_reports/20260605_stage1_base_mrpc_no_degree0_ppo_final.html`.
  Deterministic `validation_full` eval on 408 MRPC examples gave original
  plaintext baseline loss/accuracy/weighted-F1
  `0.3462543786`/`0.8774509804`/`0.8744220872`, PPO-best
  `0.3459055424`/`0.8750000000`/`0.8737330273`, and all4/all6 reference
  `0.3434863985`/`0.8799019608`/`0.8774415001`. After this report was pushed,
  Codex launched fresh `base_rte` from the same verified `f85c77e` server
  source; current PID is `1354001`, output directory is
  `/hy-tmp/stage1_no_degree0_queue_f85c77e_20260604_201520/src/Parting Chapter/stage1/bert base rte`,
  and remaining order is `base_sst2`, `large_mrpc`, `large_rte`,
  `large_sst2`.
- Stage-1 no-degree0 strict MRPC rerun, added 2026-06-05: after reviewing the
  first no-degree0 BERT-base MRPC report, the user requested changing the
  Stage-1 metric constraint to `0%` and rerunning. Codex stopped the temporary
  `base_rte` run before using the GPUs for the strict MRPC rerun. The first
  strict launch attempt showed that `llama_7B_LayerImportance.sh` still
  rejected `--stage1-accuracy-tolerance 0.0` as non-positive, even though the
  Python evaluator accepts a zero tolerance. The local launcher validation now
  treats Stage-1 accuracy tolerance as non-negative while still requiring it to
  be `< 1`. Run the strict rerun fresh from a verified source snapshot with
  `--stage1-accuracy-tolerance 0.0`, `--stage1-search-episodes 0`,
  `--stage1-entropy-stop-threshold 0.1`, `--stage1-rl-devices 0,1,2,3`, and
  `--rl-algo ppo`. Commit `41a0b32` was packaged into verified temp source
  `/hy-tmp/stage1_no_degree0_mrpc_strict0_41a0b32_20260605_123543/src` and
  launched fresh as PID `1358171`. The launch log is
  `/hy-tmp/stage1_no_degree0_mrpc_strict0_41a0b32_20260605_123543/logs/base_mrpc_strict0_launch.log`,
  the training log is
  `/hy-tmp/stage1_no_degree0_mrpc_strict0_41a0b32_20260605_123543/src/Parting Chapter/stage1/bert base mrpc/logs/stage1_rl.log`,
  and status is tracked at
  `/hy-tmp/stage1_no_degree0_mrpc_strict0_41a0b32_20260605_123543/state/status.json`.
  Initial log evidence confirms validation_full baseline
  loss/accuracy/F1 `0.346254`/`0.877451`/`0.874422`, strict constraints
  `loss <= 0.3463`, `Accuracy >= 0.8775`, `F1 >= 0.8744`, and four
  Stage-1 rollout workers on `cuda:0..3`. After this strict MRPC report is
  complete, resume `base_rte`, `base_sst2`, `large_mrpc`, `large_rte`, and
  `large_sst2` one at a time with the same `--stage1-accuracy-tolerance 0.0`
  rule.
- Stage-1 GRPO MRPC current snapshot, added 2026-06-01: while the GRPO run was
  still active, a snapshot report was generated at
  `experiments/server_command_runs/stage1_mrpc_ppo_then_grpo_entropy0p1_tol0p001_20260531_161526/grpo_current_snapshot_20260601_164215/stage1_mrpc_grpo_current_result_report.html`.
  The snapshot checkpoint had `112320` completed episodes, `936` GRPO updates,
  entropy `0.6420237422` above the `0.1` stop threshold, and best reward
  `0.7910256242` first reached at episode `27271`. Current reward-best config
  is `GELU=[0,1,0,1,1,1,1,1,1,1,1,0]`, Softmax fixed to degree `6` in every
  layer, cost `42.00`. Deterministic validation_full inference gave original
  baseline loss/accuracy/F1 `0.3462543786`/`0.8774509804`/`0.8744220872` and
  GRPO current-best loss/accuracy/F1
  `0.3330092728`/`0.8799019608`/`0.8797757292`; this is a running snapshot, not
  final entropy-converged GRPO evidence.
- Stage-1 PPO vs GRPO MRPC comparison completion, added 2026-06-01: by user
  request, the active GRPO process `705245` was stopped after visual convergence
  rather than waiting for the configured `entropy < 0.1` stop. The code's
  official stop threshold remains `0.1`; for the comparison report, GRPO runtime
  is measured at the first `entropy < 0.6` point, which occurred at update
  `337`, episode `40440`, entropy `0.5758088231`, with interpolated elapsed
  time `6.9782` hours. The final stopped checkpoint had `116520` episodes and
  `971` updates. GRPO reward-best was first selected at episode `27271`:
  `GELU=[0,1,0,1,1,1,1,1,1,1,1,0]`, Softmax all `6`, cost `42.00`, reward
  `0.7910256242`. The final comparison report is
  `experiments/server_command_runs/stage1_mrpc_ppo_then_grpo_entropy0p1_tol0p001_20260531_161526/stage1_mrpc_ppo_vs_grpo_comparison_report_20260601_172333.html`.
- Stage-1 grouped plaintext inference check, added 2026-05-30: BERT-base MRPC
  validation_full was evaluated with Softmax degree 6 for every layer and four
  GELU variants. The local report is
  `experiments/server_command_runs/stage1_group_inference_mrpc_20260530_165907/stage1_group_inference_mrpc_report.html`.
  Results on 408 validation examples were: GELU degree 1 loss `0.3512777090`,
  accuracy `0.8799019608`, weighted F1 `0.8777671950`; GELU degree 2 loss
  `0.3460941315`, accuracy `0.8725490196`, weighted F1 `0.8693989707`; GELU
  degree 4 loss `0.3434863985`, accuracy `0.8799019608`, weighted F1
  `0.8774415001`; GELU replaced by ReLU loss `0.8100994229`, accuracy
  `0.3161764706`, weighted F1 `0.1519060138`. This was a pure Stage-1
  plaintext inference test: no BLB bridge, no Stage-2 noise, and no Stage-2
  configuration was used.
- Stage-1 layerwise ReLU plaintext inference check, added 2026-05-30:
  BERT-base MRPC validation_full was evaluated with Softmax degree 6 in every
  layer, GELU degree 4 in all non-target layers, and exactly one layer's GELU
  replaced by ReLU per group. The local report is
  `experiments/server_command_runs/stage1_layerwise_relu_mrpc_20260530_170734/stage1_layerwise_relu_mrpc_report.html`.
  Results for target ReLU layer 0..11 were: L0 loss `0.3352130055`, accuracy
  `0.8774509804`, weighted F1 `0.8760590106`; L1 loss `0.3441292346`,
  accuracy `0.8872549020`, weighted F1 `0.8859742898`; L2 loss
  `0.3209853172`, accuracy `0.8897058824`, weighted F1 `0.8885879653`; L3
  loss `0.3437007368`, accuracy `0.8676470588`, weighted F1 `0.8670759166`;
  L4 loss `0.3642531335`, accuracy `0.8627450980`, weighted F1
  `0.8557755617`; L5 loss `0.3494089246`, accuracy `0.8578431373`, weighted
  F1 `0.8500977999`; L6 loss `0.3281662464`, accuracy `0.8676470588`,
  weighted F1 `0.8661437315`; L7 loss `0.3290436566`, accuracy `0.8602941176`,
  weighted F1 `0.8617458367`; L8 loss `0.4215985239`, accuracy `0.8014705882`,
  weighted F1 `0.8080592745`; L9 loss `0.3363590539`, accuracy `0.8627450980`,
  weighted F1 `0.8593527376`; L10 loss `0.3407069147`, accuracy
  `0.8651960784`, weighted F1 `0.8608544155`; L11 loss `0.3356979489`,
  accuracy `0.8799019608`, weighted F1 `0.8774415001`. This was also a pure
  Stage-1 plaintext inference test with no BLB bridge, no Stage-2 noise, and no
  Stage-2 configuration.
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
- Stage-1 unbounded base_rte completion, added 2026-05-30: the fresh
  `73e6a8f` base_rte run used `--stage1-search-episodes 0`,
  `--stage1-entropy-stop-threshold 0.1`, raw PPO reward-best selection,
  validation_full-only reward/final evaluation, four-GPU Stage-1 rollout, and
  pure plaintext Stage-1 semantics with no Stage-2/BLB noise. It reached
  entropy convergence at episode `88320` with final entropy `0.0935`. The
  final local report is
  `experiments/server_command_runs/stage1_unbounded_base_rte_20260529_220653/stage1_base_rte_unbounded_final_report.html`.
  The final selected config is the raw PPO reward-best:
  `GELU=[1,1,1,1,1,1,1,1,1,1,1,1]`,
  `Softmax=[4,4,3,2,2,3,3,3,3,3,4,5]`, cost `31.50`, reward
  `1.1100880189`. Full-validation final eval on `validation_full` size `277`
  gave baseline original-plaintext loss/accuracy
  `0.7333006263`/`0.7256317690` and selected loss/accuracy
  `0.7335297465`/`0.7328519856`, passing the 0.5% loss/metric constraints.
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
- Stage-1 MRPC PPO-best metric check, added 2026-06-01: the diagnostic
  validation_full report at
  `experiments/server_command_runs/stage1_original_vs_ppo_best_vs_all4_mrpc_20260601_141846/stage1_original_vs_ppo_best_vs_all4_mrpc_report.html`
  compared original GELU/Softmax, PPO best
  `GELU=[0,1,1,1,1,1,1,1,1,0,1,0]` with Softmax fixed at `6`, and
  all-`GELU=4`/all-`Softmax=6`. Original and PPO best have identical aggregate
  `m1=0.8774509804`, `m2=0.8744220872`, prediction counts, and confusion
  matrix `[[94,35],[15,264]]`, but not identical per-sample predictions
  (`26/408` predictions differ). The metric computation is consistent with the
  confusion matrix; do not interpret equal m1/m2 as proof that the two configs
  predict the same examples. All4/All6 scored `m1=0.8799019608`,
  `m2=0.8774415001`, loss `0.3434863985`, while PPO best had lower loss
  `0.3370108604`.
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
- If Claude Code modifies local source and the handoff/memory update is missing,
  Codex must detect it with `git status`, `git diff`, `git log`, and remote
  comparison before launching server work. Treat unexpected local commits or
  dirty source files as user/Claude edits, review them rather than overwriting
  them, and only let the server run after the local tree and git history make
  the intended source state explicit.
- Before any server run after local or Claude Code edits, verify the sync
  boundary explicitly: local source changes are committed and pushed; server
  source is a pull/fast-forward or verified bundle of that commit; generated
  server artifacts/results are pushed or copied back; local then pulls or
  imports those results. Do not accept "server has the fix" as canonical unless
  that exact source is represented locally and in git.

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
- Current user-provided GPUShare endpoint, added 2026-07-02:
  `ssh -p 37498 root@i-2.gpushare.com`. The password was provided in chat for
  ephemeral use only; do not store it in repo files, shell profiles, SSH config,
  scripts, or logs.

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
9. Stage-2 config is strictly bound to one Stage-1 config. The Stage-1 GELU
   degree and Softmax degree chosen per layer fully determine the shape of
   Stage-2 Block 3 (softmax exp approximation, graph key `block3_exp_n<softmax>`)
   and Block 5 (GELU polynomial chain, graph key `block5_n<gelu>`). One Stage-1
   config maps to many possible Stage-2 configs; one Stage-2 config maps to
   exactly one Stage-1 config, never multiple. Stage-2 is "stage 2" because it
   takes the plaintext model selected by Stage-1, converts it into a CKKS+MPC
   ciphertext model, then optimizes that ciphertext model. A Stage-2 result is
   therefore meaningless without its prerequisite Stage-1 config: every Stage-2
   record / final-eval must carry the Stage-1 config it was built on, and any
   cost-matched Stage-2 sampling holds that Stage-1 config fixed (only Stage-2
   cost varies).

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
- Repeated `ReplanSession` calls reuse precomputed node-reference paths for the
  fixed baseline skeleton. Do not replace those references with copied node
  values: per-action propagation-delta mutations must remain visible during
  `propagate_scale()` while topology traversal stays cached.
- `ReplanSession` also owns a cached multiplication-node `name -> node`
  reference mapping per graph. Reuse it for repeated delta application; retain
  the generic on-demand lookup for standalone replan callers and keep all
  per-action CTPT/CTCT type validation intact.
- Exact built-in `dict[str, int | "x2"]` delta overrides are already normalized
  and may be reused without a copy. Do not broaden that fast path to bools,
  numpy scalars, subclasses, or custom mappings; those must keep the generic
  coercion and validation behavior.
- A `ReplanSession` tracks whether each graph's delta state is clean. Normal
  calls restore once before returning and let the next call skip a redundant
  entry restore; abnormal dirty state must still trigger entry recovery. Keep
  that recovery invariant when changing session return or exception paths.
- Compact session replans skip materializing the unused
  `applied_delta_overrides` echo dictionary. Full replan, CLI, and diagnostic
  callers retain it by default; do not remove their record or bypass delta
  lookup, mutation, and validation when maintaining this fast path.
- Compact session replans also omit `baseline_q_bits` because their result does
  not expose `delta_q_vs_baseline`. Full replan, CLI, and diagnostic callers
  must continue receiving the baseline and emitting that diagnostic; do not
  remove it globally to optimize fusion-map enumeration.
- Replan initial-drop bounds share one scan. A non-positive drop must retain
  precedence and return immediately; otherwise every 1-indexed stage above
  `q_max` must remain in the diagnostic. Do not restore separate `any(...)`
  and over-limit list-comprehension scans in the combination hot path.
- `ReplanSession` prepares each graph's default fusion policy once as both an
  ordered diagnostic list and an internal immutable normalized set. Reuse the
  prepared set for default-policy math, preserve ordered JSON output, and keep
  explicit custom policies on the normal parser/validator path.
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
- `blb_stage2_live_summary.md`
- `blb_stage2_training_curve.npz`
- `blb_stage2_training_curve.png`
- `blb_stage2_report.md`
- `blb_stage2_best_action_full.{json,md}`
- `blb_stage2_baseline_action_full.{json,md}`
- `diagnostics/` and `details/` artifacts

Older docs that mention `blb_stage2/progress/` are stale for current code.

**Formal Stage-2 RL persistence rule, updated 2026-06-25.** Future Stage-2 RL
training and monitoring must use the old constraint-slug persistent layout, not
agent-created temp training directories and not the historical flattened
`Parting Chapter/stage2/{combo}/` output. For the current MRPC 60k spec the
canonical root is:

```text
Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.001_s2t0.001_s2st3.0/
```

The live artifacts to return to the user are under
`stage2_noise/progress/` inside that root, especially
`blb_stage2_status.json`, `blb_stage2_live_summary.md`,
`blb_stage2_training_curve.png`, diagnostics JSONL, details batches,
checkpoint files, and reports. `LATEST_PID` and
`LATEST_RUN_DIR` are written one level above the slug:
`Parting Chapter/persistent/rl/bert-base/mrpc/`.
The formal Stage-2 path currently defaults to
`--stage2-fixed-config-source all4`; use `stage1_result`, `json`, or `manual`
explicitly to switch away from it. Do not rely on a same-directory Stage-1
checkpoint unless `stage1_result` was selected. Short diagnostic RL runs (A/B,
1-GPU-vs-N-GPU gates, probe sweeps) must still use this persistent group, but
with `--run-tag` so they create sibling slugs such as
`s1t0.001_s2t0.001_s2st3.0__gate_gN_<timestamp>` and cannot overwrite the
formal 60k canonical slug.
The `mrpc-blb-stage2-rl` preset defaults to this current formal constraint
slug (`stage1_accuracy_tolerance=0.001`, `stage2_limit_tolerance=0.001`,
`stage2_stability_tolerance=3.0`); command-line overrides remain allowed but
must be deliberate because they create a different persistent slug.
Use `python3 scripts/verify_stage2_persistent_outputs.py --run-dir <slug>`
after any Stage-2 RL smoke, gate, or long run to prove the live status,
human-readable summary, diagnostics JSONL, details batches, and Stage-1-style
curves are present before reporting results.

**Stage-1 decoupling IMPLEMENTED 2026-06-01 (local, py_compile/bash-n/torch-free
verified; behavioral verification pending on the server).** Stage-1 RL remains a
separate, never-chained run with one **flattened** working dir per combo (no
constraint-slug): `Parting Chapter/stage1/{combo}/`, where
`combo = {model_type with '-'→' '} {dataset}` (e.g. `bert base mrpc`, spaces
intended). Historical Stage-2 flattened directories may exist for older runs, but
they are not the formal Stage-2 RL output target after 2026-06-25. Constraint
tolerances live in `metadata.json` (with a resume-time mismatch guard →
`--fresh`). On completion each stage best-effort snapshots `final_config.json` +
basic-single-eval `final_eval.json` + curves + `report.md` + `metadata.json` into
`Parting Chapter/stage{1,2}/record/{combo} {N} {YYYYMMDD}/` (`N` = existing combo
records + 1; run-id e.g. `bert base rte 1 20260530`) and writes a `COMPLETED`
marker in the working dir. Stage-2 records must store BOTH the action and that
Stage-1 config (Critical Mental Model #9). **`run rl` REQUIRES an explicit
`--mode stage1-only`/`stage2-only`**; chained `train`/`eval`/`search-only` error
with guidance; the `eval` SUBcommand (→ Paean standalone final-eval) is untouched.
Final-eval auto-trigger is removed — completion writes only a basic snapshot; the
heavy same-cost comparison is a separate standalone tool (its own spec). Root
remains `Parting Chapter`; the old `GRPO Chapter` routing is retired with the
GRPO entrypoints. SSOT for the layout: `config/run_layout.py`.
GA/greedy/general/compare and legacy v2 keep the old
`persistent/{algorithm}/{model}/{dataset}/{accuracy_slug}/` layout (no migration).
Spec (§12 = locked grilled refinements):
`docs/superpowers/specs/2026-05-30-decouple-stage1-stage2-persistence-design.md`.
Key code: launcher `llama_7B_LayerImportance.sh` (rl branch + `DECOUPLED_LAYOUT` +
`--stage1-run-id` + constraint guard), `rl_tune.py` (threads `decoupled_layout`/
`stage1_run_id`), `layer_importance_evaluator.py`
(`resolve_run_output_layout(flattened=…)`, `_resolve_stage1_degrees_from_record`,
`_maybe_snapshot_decoupled_stage1_record`), `blb_stage2_rl/runner.py`
(`resolve_blb_persistence_dir` flatten + Stage-2 snapshot).

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

**In-progress final-eval decoupling (design APPROVED 2026-05-30, NOT yet
implemented).** Final-eval is becoming a standalone full-validation tool,
decoupled from training (the training-end auto-trigger
`Paean/embedded.run_embedded_final_eval` is removed; training completion only
snapshots config + curves + a basic single-eval metric into the `record`). Two
separate tools write under `Paean/stage1/{combo} {N} {YYYYMMDD}/` and
`Paean/stage2/{combo} {N} {YYYYMMDD}/` (flat numbered dirs, independent
"final-eval 序号"; replaces `Paean/outputs/.../final_eval/`). Input:
`--stage stage1|stage2 --record-dir <path>` (or `--run-id`) pointing at a Parting
Chapter `record` run. Stage-1 FE: 50 same-(stage1)-cost RL-domain peer configs +
the selected, 1 eval each on validation_full, no noise / no stage2, 3 sorted-bar
plots (loss/m1/m2, selected highlighted). Stage-2 FE (adapts
`BLBActionFinalEvaluationModule`): Stage-1 held FIXED to the record's prerequisite
Stage-1, 50 same-(stage2)-cost VALID peers + selected, `--repeat` (default 50)
trials each, 6 sorted-bar plots (loss/m1/m2 + their std). Spec:
`docs/superpowers/specs/2026-05-30-decoupled-standalone-final-eval-design.md`.

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

- Stage-2 degree-0 / skeleton-SSOT cleanup note, added 2026-06-02: Claude Code
  noted two residual display/cleanup items after the `00871c3` validation
  command was prepared. `_is_action_field_effective` / `_COMPAT_EXTRA_FIELDS`
  were originally only used by `describe_action_vector` display/logging and
  could show old Block-2/Block-4 active-slot assumptions; three deprecated
  baseline tables were also defined but unreferenced. Claude's cleanup commit
  `f6b91ba` made report/effectiveness display skeleton-driven and dropped those
  dead baseline tables. The correctness path should follow
  `blb_stage2_rl/skeleton_stage_map.py`, `baseline_bootstrap.py`,
  `action_space.py`, and `rescale_optimizer_bridge.py` derived from the current
  static skeletons archive.
- Stage-2 degree-0 server verification, added 2026-06-02: the real passing
  server validation is commit `8c63fe1`, not the earlier `70c561a` attempt.
  `70c561a` passed the shell command but exposed that `make_config_name()` used
  `getattr(... ) or 4`, so valid GELU degree `0` was silently named as
  `block5_n4` and the full HTML did not actually exercise `block5_n0`.
  `8c63fe1` preserves degree zero in config names and adds a regression test
  proving `action_vector_to_cfgs(..., gelu_degree=[0])` builds a
  `block5_n0_L0` optimizer request. Server verification at
  `experiments/server_command_runs/stage2_degree0_verify_20260602_184748/`
  passed: `contract_gate_exit=0` across `166` BLB tests,
  `degree0_tests_exit=0`, and all three full noise-install commands exited
  `0`. The user-facing HTML copies are under `reports/html_reports/` as
  `20260602_stage2_degree0_noise_install_mixed.html`,
  `20260602_stage2_degree0_noise_install_allrelu.html`, and
  `20260602_stage2_degree0_noise_install_normal.html`. In the final HTML,
  all-ReLU maps all 12 layers to `block5_n0` with `valid=True`; the normal
  all-GELU4 run has block2/block4/block5_n4 valid across all layers. The mixed
  probe correctly maps ReLU layers 0/4/8 to valid `block5_n0`, but its degree-1
  layers 1/5/9 are still invalid under all-max action settings; treat that as a
  separate degree-1/search-space issue, not a failure of the degree-0 path.
  Follow-up server verification at commit `f1c8ebc` added `[1c]` and passed at
  `experiments/server_command_runs/stage2_degree0_verify_20260602_192644/`:
  `contract_gate_exit=0` across `168` BLB tests, `degree0_tests_exit=0`,
  `bridge_derivation_exit=0`, and all three full noise-install commands exited
  `0`. This confirms the bridge actively auto-derives `t_new` from the live
  skeleton (`BridgeDerivesT_newFromSkeletonTest` passed), rather than relying on
  the static fallback table as the source of truth. The same three user-facing
  HTML paths under `reports/html_reports/` were refreshed from this run.
  Final cleanup verification at commit `f6b91ba` passed at
  `experiments/server_command_runs/stage2_degree0_verify_20260602_220145/`:
  `contract_gate_exit=0` across `168` BLB tests, `degree0_tests_exit=0`,
  `bridge_derivation_exit=0`, and all three full noise-install commands exited
  `0`. This validates the report/effective-field cleanup under the torch-backed
  contract gate.

## Conventions

- Prefer launcher/preset workflows. The launcher validates skip-mode conflicts,
  builds persistent slugs, and writes `LATEST_PID`/`LATEST_RUN_DIR` markers.
- Put user-facing HTML reports created by Codex under `reports/html_reports/`
  with clear date-prefixed names such as
  `YYYYMMDD_stage1_mrpc_ppo_vs_grpo_comparison.html`. Keep helper scripts,
  notebooks, and temporary code out of that folder; the folder should contain
  HTML deliverables only. When an HTML report is first produced elsewhere for
  experiment provenance, also copy or move the final HTML into this central
  folder before reporting it to the user.
- Use `--mode train|eval|stage2-only|stage1-only|search-only` instead of
  manually mixing skip flags.
- Multi-trial evaluation is required. A single noise trial is not evidence.
- Warmstart toward the static-skeleton baseline is a prior, not a restriction.
- GLUE loading is flaky over network. Prefer local caches via
  `GLUE_LOCAL_DATASET_DIR`, `GLUE_DATASET_DIR`, HF cache, or saved parquet.
- Stage-1-only GLUE submission packages for the 2026-06-01 BERT-base MRPC
  PPO/GRPO comparison were generated under
  `experiments/server_command_runs/glue_stage1_ppo_grpo_mrpc_submission_20260601_180334/`.
  The generated zips are
  `ppo_stage1_submission/submission.zip` and
  `grpo_stage1_submission/submission.zip`. MRPC is real test-set inference;
  the other GLUE TSVs are official-format placeholders because the selected
  Stage-1 configs are MRPC-specific. The run used only Stage-1 GELU/Softmax
  replacement configs, no `--noise_config`, no `--blb_action_config`, and no
  Stage-2/BLB noise. The current `generate_glue_submission.py` Stage-1 helper
  must include GELU degree `0` when applying configs; degree `0` is ReLU via
  `ReversibleLayerHandler.replace_layer_gelu`.
- Stage-1-only GLUE submission packages for the 2026-06-04 BERT-base MRPC/RTE
  best-vs-baseline official test submission check were generated under
  `experiments/server_command_runs/glue_stage1_best_vs_baseline_mrpc_rte_20260604_111155/`.
  This run used source commit `064500c`, where `generate_glue_submission.py`
  now natively applies GELU degree `0` as ReLU instead of silently ignoring it.
  The best package is `best_stage1_ppo_submission/submission.zip` and uses
  MRPC GELU `[0,1,1,1,1,1,1,1,1,0,1,0]`, RTE GELU
  `[0,0,1,1,0,1,2,1,0,0,1,0]`, and Softmax degree `6` for every layer, with
  no Stage-2/BLB noise. The baseline package is
  `baseline_stage1_original_submission/submission.zip` and uses
  `--no_approx --no_noise`, i.e. original GELU/Softmax functions. MRPC/RTE zip
  contents were locally re-verified: MRPC has `1726` lines and labels `0/1`;
  RTE has `3001` lines and labels `entailment/not_entailment`; both packages
  include MRPC/RTE in the zip and are not placeholder-only for those tasks.
- Stage-1 PPO vs GRPO advantage report for the 2026-06-01 BERT-base MRPC
  comparison is at
  `experiments/server_command_runs/stage1_mrpc_ppo_grpo_advantages_20260601_200615/stage1_ppo_vs_grpo_advantages_report.html`.
  It includes same-script deterministic `validation_full` eval for original
  baseline, PPO best, and GRPO best. The key interpretation is: PPO is safer
  for the current GLUE official submission candidate because it converged
  faster and had lower test-distribution drift; GRPO found a stronger
  validation reward/loss/weighted-F1 point, but current MRPC `metric2` is
  weighted F1 rather than GLUE positive-class F1, which explains why GRPO can
  look good on the Stage-1 reward metric while being riskier for official GLUE.
- Stage-1 PPO/GRPO MRPC generalization analysis report, added 2026-06-01, is at
  `experiments/server_command_runs/stage1_mrpc_generalization_analysis_20260601_232710/stage1_ppo_grpo_generalization_analysis.html`.
  It argues that the GRPO validation/test mismatch is primarily metric-proxy
  mismatch plus validation selection bias: GRPO improves validation
  loss/accuracy/weighted F1 but lowers validation positive-class F1 and shifts
  MRPC test predictions toward class 0. PPO's better generalization is not
  proven luck-free, but current evidence favors a conservative near-baseline
  behavior explanation because PPO lowers validation loss while preserving
  aggregate validation classification metrics and keeps test prediction priors
  close to the original baseline.
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
