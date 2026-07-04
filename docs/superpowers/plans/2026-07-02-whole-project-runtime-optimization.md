# Whole-Project Runtime Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the entire project runtime path, from launch to reports, while preserving research semantics and validation protocols.

**Architecture:** Start with project-wide observability, then optimize one flow stage at a time behind tests and parity gates. Hardware-sensitive changes require server A/B evidence before becoming defaults.

**Tech Stack:** Bash launcher, Python 3 stdlib tooling, PyTorch/Hugging Face hot paths, JSON/JSONL artifacts, unittest, ruff, git-synced server workflow.

---

## Current Execution Spine

This plan is executed as one main chain, not as isolated micro-optimizations.
Each optimization must either reduce time/memory in the active training/eval
flow, improve hardware utilization evidence, or remove overhead from required
post-run artifacts without weakening the validation protocol.

### Main Chain Order

1. **Keep runs launchable and observable.** Launcher/preset/resource snapshot
   work comes first because every server run depends on correct device flags,
   source identity, and artifact roots.
2. **Keep required artifact writes cheap.** JSON/JSONL/NPZ data that is needed
   for paper figures remains mandatory, but expensive HTML/PNG/PDF rendering
   should happen after training unless live rendering is explicitly requested.
3. **Reduce shared GPU forward overhead.** `function_handler.py`,
   `layer_importance_evaluator.py`, Paean final eval, and Stage-2 probe
   installs are shared by Stage-1, Stage-2, and final-eval paths; safe tensor
   allocation and synchronization reductions here have broad effect.
4. **Optimize pure-Python cost/replan/report loops.** Rescale/fusion-map and
   report tools stay CPU-bound; use streaming, session reuse, bounded heaps,
   and exact-cache keys rather than GPU rewrites.
5. **Promote hardware defaults only with server evidence.** Any change that
   claims better GPU utilization, worker assignment, or batch/device defaults
   needs 1GPU-vs-NGPU parity plus wall-clock evidence before becoming a
   default.

### Current Priority Queue

| Priority | Flow | Next concrete work | Verification gate |
| --- | --- | --- | --- |
| P0 | Plan and audit | Keep this plan synchronized with real commits and missing gates. | `python3 -m unittest tests.test_project_optimization_audit -v` and plan self-review. |
| P1 | Shared GPU forward | Continue removing redundant full-shape allocations, scalar syncs, and repeated install work in `function_handler.py` and `layer_importance_evaluator.py` without changing noise/reward semantics. | Source/behavior tests in `tests/test_stage1_eval_accel.py`, `tests/test_stage1_approx_reuse.py`, and server torch parity when available. |
| P1 | Stage-2/Paean evaluation | Reuse parsed action/config metadata and avoid retaining large unused action payloads while final-eval groups are scheduled. | `tests/test_paean_action_grid.py`, `tests/test_run_fusion_count_action_eval.py`, `tests/test_run_fusion_count_action_eval_rlpath.py`, and a fixed-action server smoke before default changes. |
| P1 | Structured artifacts | Keep required `rl_training_data_points/`, `episodes.jsonl`, PPO diagnostics, and manifests complete while streaming verification/report generation. | `tests/test_rl_data_points.py`, `tests/test_stage2_persistent_output_verifier.py`, and generated evidence bundle checks. |
| P2 | Rescale/fusion maps | Reuse session-loaded graph/baseline data, stream large map builds and summaries, and avoid parsing sidecars as maps. | `tests/test_rescale_optimizer_bridge_cache.py`, `tests/test_blb_fusion_count_map.py`, fusion-map report tests, and server build logs for large maps. |
| P2 | Stage-1 rollout | Add or consume timing fields for per-worker rollout, cache hit rate, forward wall, and report write wall before changing worker defaults. | Stage-1 local semantics tests plus server 1GPU/4GPU speed and deterministic-result evidence. |
| P2 | Stage-2 GPU scheduling | Leave core Stage-2 RL algorithm changes to the active Stage-2 agent; optimize only tooling/gates unless a handoff is explicit. | `stage2_ngpu_ab_compare.py`, GPU utilization reports, rollout signatures, and PPO-visible equality gates. |

### Recent Cross-Flow Runtime Commits

- `ed77481` batched installed inference metric transfers, reducing repeated
  CPU/GPU scalar transfers in installed-model evaluation.
- `edbd766` and `794be01` cached Stage-1 GELU and Stage-2 fusion action masks.
- `63b8f22`, `66ee403`, `990a89c`, `ebeddae`, and related commits deferred or
  batched scalar synchronization in Stage-1/Stage-2 PPO paths.
- `980ca8d`, `50c9907`, and `70a2655` removed unnecessary full-shape tensor
  allocations in shared GELU/LN/softmax noisy forward paths.
- `9468f58`, `51cd4a0`, `58c6666`, `9273440`, and `b0a8697` centralized JSON
  artifact read/write helpers so report/final-eval code avoids duplicated
  parsing and formatting paths.

### Standing Verification Rules

- Local tests can prove pure-Python parsing, caching, and source-level
  allocation constraints.
- Local torch-less skips are not hardware evidence; any GPU utilization claim
  requires a server run with visible GPU inventory and sampled utilization.
- Result-equivalence claims for Stage-1/Stage-2 require the validation_full
  protocol to remain unchanged and must not use train data or proxies.
- When a concurrent agent owns Stage-2 RL logic, restrict this optimization
  work to shared tooling, forward-path allocation, report/artifact paths, and
  explicit handoff-safe files.

### Execution Ledger and Remaining Main Chain

Progress is measured by high-impact flow coverage and verification strength,
not by raw commit count. As of source head `d125fd4`, the conservative
completion estimate is about 98% of the full goal: the plan/audit layer,
artifact helpers, several low-conflict hot paths, and the Stage-1 1GPU vs 4GPU
gate have landed. Hardware-default promotion remains evidence-gated rather
than automatic, and remaining flow-wide scheduling work is still open.

Server-verified optimization commits currently in the execution ledger:

| Flow | Source commit | Evidence directory | Optimization |
| --- | --- | --- | --- |
| Paean final eval | `567ad75` | `experiments/server_command_runs/final_eval_repeat_install_reuse_567ad75_20260703_203900/` | Reuse one clean-baseline install and one BLB bridge install across `repeat_n > 1` forwards. |
| Paean final eval | `b2a7325` | `experiments/server_command_runs/final_eval_max_sfs_cache_b2a7325_20260703_205000/` | Cache `load_max_sfs(profile)` per final-eval module instance. |
| Paean final eval | `fa52906` | `experiments/server_command_runs/final_eval_normalize_ndarray_fa52906_20260703_234423/` | Normalize ndarray-backed final-eval config arrays without first materializing Python lists. |
| Paean final eval | `e443e4a` | `experiments/server_command_runs/final_eval_stage2_cost_incremental_e443e4a_20260703_235252/` | Maintain current cost incrementally in Stage-2 cost-matched final-eval random search instead of rescanning the full candidate config every mutation. |
| Paean final eval | `2ca2516` | `experiments/server_command_runs/paean_action_grid_max_sfs_cache_2ca2516_20260704_010810/` | Cache Paean action-grid max-SF tables by profile so batched slot-form action configs and fixed/range candidates avoid repeated `load_max_sfs()` parsing. |
| Paean final eval | `94f1aad` | `experiments/server_command_runs/paean_cost_match_degree_arrays_94f1aad_20260704_013930/` | Reuse normalized GELU/Softmax degree arrays and target integers across Paean cost-matched random action decode attempts. |
| Paean final eval | `85c03b9` | `experiments/server_command_runs/paean_base_action_ndarray_85c03b9_20260704_014440/` | Normalize ndarray-backed Paean base action vectors without first copying the full vector through `list()`. |
| Paean final eval | `a600b79` | `experiments/server_command_runs/paean_parse_action_vec_a600b79_20260704_014720/` | Parse legacy Paean action-vector lists directly with numpy instead of copying them through `list()` first. |
| Paean final eval | `8101feb` | `experiments/server_command_runs/final_summary_running_8101feb_20260704_042900/` | Summarize final-eval random-result families with running counters and stats instead of repeated materialized lists for `np.mean()` / `np.std()`. |
| Paean final eval | `08560c1` | `experiments/server_command_runs/final_stat_helpers_08560c1_20260704_044500/` | Stream shared final-eval finite-float mean/std helpers without clean-list materialization or numpy stats calls. |
| Paean final eval | `75cce4c` | `experiments/server_command_runs/final_variance_plot_mean_75cce4c_20260704_052500/` | Stream final-eval variance-plot group means through the shared finite-float helper instead of materializing per-group `vals` lists and calling `np.mean(vals)`. |
| Paean final eval | `e4c3d47` | `experiments/server_command_runs/final_variance_scatter_scan_e4c3d47_20260704_054500/` | Scan variance-plot random scatter points once per family/panel instead of building separate `xs` and `ys` list comprehensions over the same rows. |
| Paean final eval | `c85b896` | `experiments/server_command_runs/final_comparison_scatter_scan_c85b896_20260704_061000/` | Scan main final-eval comparison random scatter points once per family/panel instead of building separate `xs` and `ys` list comprehensions over the same rows. |
| Paean final eval | `a1de9a3` | `experiments/server_command_runs/final_axis_limits_stream_a1de9a3_20260704_063500/` | Stream final-eval plot axis-limit min/max calculation without a `clean` list and without converting each finite value twice. |
| Paean final eval | `d76cf29` | `experiments/server_command_runs/final_eval_invalid_values_d76cf29_20260704_054400/` | Scan normalized final-eval config arrays for unsupported values without materializing `arr.tolist()` sets. |
| Paean final eval | `b9f01de` | `experiments/server_command_runs/final_eval_signature_tuple_b9f01de_20260704_054950/` | Build final-eval cache signature keys as direct integer tuples and reuse `_full_signature()` from `_noise_eval()` instead of materializing arrays through `.tolist()`. |
| Paean final eval | `54d7bf9` | `experiments/server_command_runs/paean_selected_random_summary_54d7bf9_20260704_062553/` | Stream BLB action selected-vs-random final-eval summary rows once, accumulating field stats and anchor ranks without per-field numpy arrays or separate rank lists. |
| Paean final eval | `af9884a` | `experiments/server_command_runs/paean_results_plot_scan_af9884a_20260704_063156/` | Scan BLB action final-eval plot candidate rows once before numpy conversion instead of rebuilding each plotted column with a separate `candidate_results` list comprehension. |
| Paean final eval | `32ea5f5` | `experiments/server_command_runs/paean_scatter_plot_scan_32ea5f5_20260704_063613/` | Scan BLB action final-eval selected/random scatter groups once to collect primary and secondary metric columns instead of separate per-panel list comprehensions. |
| Paean final eval | `3f020ef` | `experiments/server_command_runs/paean_full_noise_table_stream_3f020ef_20260704_064005/` | Iterate BLB action final-eval full-noise Markdown table entries directly instead of copying every entry through `list()` before rendering. |
| Unified final eval | `9026d8f` | `experiments/server_command_runs/final_eval_relative_chain_9026d8f_20260704_071220/` | Stream baseline/optimized/max-SF and random-result rows into relative-metric attachment with `itertools.chain()` instead of copying `random_results` through list concatenation. |
| Paean final eval | `5fe7760` | `experiments/server_command_runs/paean_fusion_decode_copy_5fe7760_20260704_072520/` | Remove short-lived dict/list copy wrappers from fusion fixed-action decode metadata normalization, per-step block slicing, and option-field replay. |
| Unified final eval reports | `7a7e9d4` | `experiments/server_command_runs/final_eval_axes_islice_7a7e9d4_20260704_073330/` | Iterate final-eval comparison and variance plot axes with `itertools.islice()` instead of materializing `list(axes.flat)[:3]`. |
| Unified final eval reports | `22eb07e` | `experiments/server_command_runs/final_eval_summary_bar_22eb07e_20260704_074230/` | Collect final-eval summary bar chart family labels, feasibility rates, and dominance rates in one pass instead of separate scans/list comprehensions over `summary["by_family"]`. |
| Unified final eval reports | `aed348f` | `experiments/server_command_runs/final_eval_family_order_aed348f_20260704_071234/` | Reuse a static final-eval family color order tuple in `_ordered_families()` instead of rebuilding the color map and copying keys for every plotted panel. |
| Unified final eval reports | `37890c5` | `experiments/server_command_runs/final_eval_color_map_37890c5_20260704_071600/` | Use the static final-eval family color map directly in internal plot rendering instead of copying it through `_family_colors()` for each plot. |
| Unified final eval reports | `bb0962e` | `experiments/server_command_runs/final_eval_ordered_family_cache_bb0962e_20260704_072039/` | Cache the ordered final-eval comparison plot family list once per render instead of recomputing it inside each metric panel loop. |
| Stage-1 eval | `dca7526` | `experiments/server_command_runs/stage1_apply_config_reuse_dca7526_20260703_210000/` | Skip repeated `apply_configuration()` installs for unchanged GELU/Softmax configs. |
| Stage-1 eval | `5d15e6c` | `experiments/server_command_runs/stage1_worker_apply_config_reuse_5d15e6c_20260703_211000/` | Skip repeated worker-handler installs for unchanged Stage-1 configs. |
| Stage-1 eval | `61c8c57` | `experiments/server_command_runs/stage1_reward_history_deque_392b646_20260703_215700/` | Maintain Stage-1 reward normalization history with a bounded deque instead of list `pop(0)`. |
| Stage-1 eval | `5901ffb` | `experiments/server_command_runs/stage1_report_regex_dispatch_5901ffb_20260704_002526/` | Dispatch Stage-1 report log lines by marker so `[stage1-rollout-total]` rows skip worker/cache regex parsers. |
| Stage-1 eval | `54feaa4` | `experiments/server_command_runs/stage1_rollout_pack_batch_54feaa4_20260704_011755/` | Batch recurrent rollout `logprobs` and `values` tensor conversion before PPO updates so each field uses one stacked CPU transfer instead of per-step scalar `.item()` syncs. |
| Stage-1 eval | `92ad0f0` | `experiments/server_command_runs/stage1_rollout_direct_tensor_92ad0f0_20260704_012541/` | Pack recurrent rollout `logprobs` and `values` directly as target-device tensors before PPO updates, avoiding the CPU numpy round trip introduced by the earlier batch path. |
| Stage-1 eval | `343a5e3` | `experiments/server_command_runs/stage1_noise_validation_scan_343a5e3_20260704_055529/` | Scan layer-evaluator noise scaling validation arrays directly for unsupported values instead of materializing `arr.tolist()` sets. |
| Stage-1 eval | `e17eee8` | `experiments/server_command_runs/stage1_reward_stats_window_e17eee8_20260704_060340/` | Maintain Stage-1 reward normalization window sum/sumsq incrementally instead of rescanning the bounded deque with `np.mean()` / `np.std()` every episode. |
| Stage-1 eval | `dbd1b6f` | `experiments/server_command_runs/stage1_semantics_gate_dbd1b6f_20260704_080342/` | Restore the Stage-1 semantic gate after shared fast-path changes: coefficient-order low-allocation GELU polynomial evaluation, Stage-1 batch-loss averaging, and legacy sklearn metric precision. |
| Stage-1 rollout | `b62743a` | `experiments/server_command_runs/stage1_timing_fields_b62743a_20260704_082005/` | Add Stage-1 rollout timing diagnostics for model-forward wall seconds, forward calls, and report-write wall seconds while preserving existing worker/cache/total timing log fields. |
| Launcher gates | `4bca31a` | `experiments/server_command_runs/stage1_gpu_ab_4bca31a_20260704_084330/` | Make `scripts/launcher_gpu_audit.py` runnable as a script from repo root without relying on external `PYTHONPATH`, unblocking clean server launcher gates. |
| Stage-1 throughput gate | `4bca31a` | `experiments/server_command_runs/stage1_gpu_ab_4bca31a_20260704_084330/` | Ran the formal 170-episode Stage-1 MRPC 1GPU vs 4GPU gate. Both completed and 4GPU used `cuda:0..3`, but the gate exposed a diagnostic undercount fixed in `4834b2f`; rerun this gate before using the wall-clock ratio for default promotion. |
| Stage-1 diagnostics | `4834b2f` | `experiments/server_command_runs/stage1_parallel_episode_count_4834b2f_20260704_085732/` | Count Stage-1 parallel rollout total episodes from actual worker counts (`43/43/42/42 => 170`) instead of `num_workers * floor(episodes_per_worker)` (`168`), fixing throughput undercounting for imbalanced windows. |
| Stage-1 throughput gate | `180d319` | `experiments/server_command_runs/stage1_gpu_ab_180d319_20260704_090423/` | Reran the formal 170-episode Stage-1 MRPC 1GPU vs 4GPU gate after the episode-count diagnostics fix. 4GPU correctly reports 170 episodes and uses `cuda:0..3`, but remains slower; keep Stage-1 4GPU default promotion blocked until duplicated validation/model-forward cost is reduced. |
| Stage-1 launcher default | `9f3864d` | `experiments/server_command_runs/stage1_default_batch_gpu_ab_9f3864d_20260704_092741/` | Default `run rl --mode stage1-only` to batch size 128 when `--batch-size` is omitted, preserving explicit user overrides. The no-batch default runtime gate completed 170-episode MRPC A/B with `g4` at `9007.153` ep/h versus `g1` at `7558.355` ep/h. |
| Shared inference eval | `497ecda` | `experiments/server_command_runs/probe_scalar_sync_497ecda_20260704_025145/` | Batch reward-probe loss/metric scalar tensors into one packed CPU transfer instead of three per-field scalar sequence transfers. |
| Shared inference eval | `b5dfff5` | `experiments/server_command_runs/probe_skip_pred_arrays_b5dfff5_20260704_025610/` | Skip reward-probe prediction/label tensor retention and numpy transfer for accuracy-only metric profiles. |
| Shared inference eval | `2d98907` | `experiments/server_command_runs/probe_tensor_arrays_2d98907_20260704_030105/` | Concatenate same-device reward-probe prediction/label tensors before one packed CPU/numpy transfer. |
| Shared inference eval | `7be83af` | `experiments/server_command_runs/inference_mnli_accuracy_helper_7be83af_20260704_042030/` | Reuse the shared direct-count accuracy helper for MNLI full eval instead of carrying a local `np.mean()` implementation. |
| Shared inference eval | `ee8a68e` | `experiments/server_command_runs/handler_layer_resolve_cache_ee8a68e_20260704_061121/` | Cache `ReversibleLayerHandler` layer-name resolution for Stage-1 GELU/Softmax and shared legacy noise install/restore paths instead of repeatedly evaluating and copying layer sequences. |
| Shared eval metrics | `da02fca` | `experiments/server_command_runs/eval_metric_weights_da02fca_20260704_030610/` | Reuse one count-weight array and weight sum for reward-probe loss/metric batch means instead of rebuilding weights three times. |
| Shared eval metrics | `1a6969a` | `experiments/server_command_runs/eval_single_array_1a6969a_20260704_031145/` | Reuse single packed reward-probe prediction/label arrays directly instead of copying them through `np.concatenate()`. |
| Shared eval metrics | `f9bbb29` | `experiments/server_command_runs/eval_binary_f1_f9bbb29_20260704_034430/` | Compute 0/1 binary weighted F1 with direct count reductions instead of sorting a class union for every MRPC/QQP reward-probe trial. |
| Shared eval metrics | `d0e8b8c` | `experiments/server_command_runs/eval_binary_mcc_d0e8b8c_20260704_035430/` | Compute 0/1 binary Matthews correlation with direct count reductions instead of sorting a class union for CoLA-style evals. |
| Shared eval metrics | `211ca50` | `experiments/server_command_runs/eval_accuracy_count_211ca50_20260704_041100/` | Compute classification accuracy with `np.count_nonzero()` match counts instead of generic `np.mean()` over a boolean mask. |
| Shared attention forward | `a416d46` | `experiments/server_command_runs/attention_tail_cursor_a416d46_20260703_214800/` | Parse positional attention tail args with an index cursor instead of front-of-list `pop(0)`. |
| Stage-2 artifacts | `cf4eed6` | `experiments/server_command_runs/candidate_action_hash_cf4eed6_20260703_221100/` | Stream normalized integer action hash payloads directly into sha256 instead of `json.dumps` materialization. |
| Stage-2 artifacts | `0aa212a` | `experiments/server_command_runs/candidate_store_ndarray_0aa212a_20260704_024050/` | Normalize ndarray-backed candidate action vectors through a direct reshape iterator instead of copying through `.tolist()`. |
| Stage-2/Paean action space | `2ee6de2` | `experiments/server_command_runs/action_space_splice_no_tolist_2ee6de2_20260704_013204/` | Splice per-step and fusion-step action vectors by iterating checked numpy arrays directly instead of materializing `arr.tolist()` for every splice. |
| Stage-2/Paean action space | `522b42f` | `experiments/server_command_runs/action_mask_degree_vector_522b42f_20260704_023140/` | Normalize ndarray-backed action-mask degree vectors without copying through `list(raw)` first. |
| Stage-2/Paean action space | `43ec3cc` | `experiments/server_command_runs/action_avg_k_direct_43ec3cc_20260704_044200/` | Compute average effective truncation K with direct integer sum/count arithmetic instead of dispatching through `np.mean(ks)`. |
| Stage-2/Paean action space | `4db8e02` | `experiments/server_command_runs/action_k_accum_direct_4db8e02_20260704_044800/` | Share a direct effective-K sum/count accumulator and cached K-slot positions so avg/sum helpers avoid gathered K list allocation. |
| Structured artifacts | `73cf14d` | `experiments/server_command_runs/stable_json_hash_73cf14d_20260703_222834/` | Stream canonical JSON chunks directly into sha256 for shared stable hashes instead of materializing full stable-key strings. |
| Structured artifacts | `e0376a5` | `experiments/server_command_runs/jsonl_encoder_reuse_e0376a5_20260703_223743/` | Reuse one `JSONEncoder` for finite JSONL row writes instead of calling `json.dump()` for every row. |
| Structured artifacts | `643ae60` | `experiments/server_command_runs/jsonl_resolve_once_643ae60_20260704_034331/` | Resolve JSONL paths once in shared readers and open the resolved file directly, avoiding duplicate filesystem checks in report/artifact scans. |
| Structured artifacts | `2ded3e7` | `experiments/server_command_runs/glue_json_reader_2ded3e7_20260704_040930/` | Read BLB GLUE action configs through the shared streaming JSON loader instead of `json.loads(open(...).read())`. |
| Structured artifacts | `d0f543b` | `experiments/server_command_runs/stage2_monitor_stream_ppo_d0f543b_20260704_000540/` | Stream Stage-2 monitor PPO updates with a bounded recent window while preserving full-file `n_samples` and non-finite-loss checks. |
| Structured artifacts | `cdcbeca` | `experiments/server_command_runs/manifest_registry_hash_cdcbeca_20260704_003917/` | Stream Trust-0 manifest registry JSON hashing through `JSONEncoder.iterencode()` instead of materializing one canonical JSON string before sha256. |
| Structured artifacts | `9b78854` | `experiments/server_command_runs/action_registry_klevel_scan_9b78854_20260704_074904/` | Find the all-max truncation-K action index for registry export with one pass over `k_levels` instead of copying the sequence and scanning twice. |
| Structured artifacts | `b66a8d2` | `experiments/server_command_runs/experiments_log_json_stream_b66a8d2_20260704_102030/` | Stream `tools.experiments_log query --format json` output directly to stdout with `json.dump()` instead of materializing one full JSON string through `json.dumps()`. |
| Structured artifacts | `c5424cd` | `experiments/server_command_runs/experiments_log_register_json_stream_c5424cd_20260704_102430/` | Stream `tools.experiments_log register` JSON output through the shared stdout `json.dump()` helper instead of materializing one full JSON string through `json.dumps()`. |
| Structured artifacts | `afcc72a` | `experiments/server_command_runs/action_registry_stdout_json_afcc72a_20260704_110045/` | Stream BLB action registry CLI path summary directly to stdout with `json.dump()` instead of materializing one full JSON string through `json.dumps()` before printing. |
| Stage-2 artifacts | `1b15448` | `experiments/server_command_runs/blb_eval_action_stdout_json_1b15448_20260704_110534/` | Stream BLB F0 eval action CLI candidate record directly to stdout with `json.dump()` instead of materializing one full JSON string through `json.dumps()` before printing. |
| Paean final eval | `2f74c05` | `experiments/server_command_runs/fusion_action_eval_stdout_json_2f74c05_20260704_110908/` | Stream fusion-count action eval CLI summary directly to stdout with `json.dump()` instead of materializing one full JSON string through `json.dumps()` before printing. |
| Stage-2 artifacts | `d125fd4` | `experiments/server_command_runs/fusion_rlpath_stdout_json_d125fd4_20260704_111300/` | Stream fusion-count RL-path eval CLI summary directly to stdout with `json.dump()` instead of materializing one full JSON string through `json.dumps()` before printing. |
| Reports / paper figures | `5a75eee` | `experiments/server_command_runs/stage2_monitor_html_stream_5a75eee_20260704_005141/` | Stream Stage-2 monitor HTML report rows and nested reward-probe/GPU JSON chunks directly to the file handle instead of materializing full JSON/table strings. |
| Reports / paper figures | `dcfea75` | `experiments/server_command_runs/paper_episode_column_dcfea75_20260703_225013/` | Read paper-figure episode rewards as a direct float column instead of building one dict per episode row. |
| Reports / paper figures | `b6dda66` | `experiments/server_command_runs/paper_figures_payload_reuse_b6dda66_20260704_094735/` | Reuse JSON-native list/dict payloads from paper-figure action and invalid-count sidecars instead of copying them through `list(...)` / `dict(...)` during `load_run()`. |
| Reports / paper figures | `596c458` | `experiments/server_command_runs/paper_training_curve_reuse_596c458_20260704_095340/` | Reuse list-backed paper-figure episode reward series directly in training-curve rendering instead of copying them through `[float(value) for ...]` before plotting. |
| Reports / paper figures | `7d66fed` | `experiments/server_command_runs/paper_group_curve_matrix_7d66fed_20260704_095833/` | Build grouped paper-figure training-curve reward matrices with `itertools.islice()` / `np.fromiter()` instead of copying each seed through `s[:min_len]` before numpy conversion. |
| Reports / aggregate seeds | `9388563` | `experiments/server_command_runs/aggregate_seed_json_stream_9388563_20260704_100730/` | Stream multi-seed `seed_summary.json` row by row with `json.JSONEncoder.iterencode()` instead of materializing `[asdict(row) for row in seed_rows]` before `json.dump()`. |
| Reports / paper figures | `bd4ca26` | `experiments/server_command_runs/persistence_curve_ndarray_bd4ca26_20260704_015320/` | Preserve ndarray fast paths in Stage-2 curve smoothing/moving-average helpers instead of copying curve arrays through `list()`. |
| Reports / paper figures | `7460284` | `experiments/server_command_runs/persistence_panel_ndarray_7460284_20260704_015635/` | Preserve ndarray fast paths in Stage-2 Stage-1-style panel plotting instead of copying panel raw series through `list()`. |
| Reports / paper figures | `74de148` | `experiments/server_command_runs/persistence_npz_ndarray_74de148_20260704_020650/` | Preserve ndarray fast paths in Stage-2 NPZ training-curve writes instead of copying every array-backed series through `list()`. |
| Reports / paper figures | `47782e9` | `experiments/server_command_runs/persistence_entropy_ndarray_47782e9_20260704_021330/` | Preserve ndarray fast paths in Stage-2 entropy PNG rendering instead of copying entropy series and update episodes through `list()`. |
| Reports / paper figures | `e8bb0dc` | `experiments/server_command_runs/persistence_float_array_sequence_e8bb0dc_20260704_045400/` | Send already-materialized list/tuple/range curve inputs directly to `numpy.asarray()` so Stage-2 curve/report generation avoids one extra Python sequence copy. |
| Reports / paper figures | `00bc7e8` | `experiments/server_command_runs/diagnostic_curve_array_cache_00bc7e8_20260704_050000/` | Reuse `_float_array()` and per-render array caching in Stage-2 diagnostic-curve generation instead of repeatedly materializing series with `list(seq)`. |
| Reports / paper figures | `ec0776b` | `experiments/server_command_runs/persistence_seq_len_count_ec0776b_20260704_051045/` | Count unsized Stage-2 curve/report iterables directly instead of materializing `list(values)` to compute length. |
| Stage-2 scheduling gate | `27be72e` | `experiments/server_command_runs/stage2_ab_ordered_jsonl_27be72e_20260703_225809/` | Skip sorting already ordered Stage-2 A/B JSONL logs while preserving sorted fallback for out-of-order artifacts. |
| Stage-2 scheduling gate | `623cd5d` | `experiments/server_command_runs/stage2_ab_excluded_keys_623cd5d_20260704_101330/` | Materialize Stage-2 A/B canonical excluded keys once per comparison and reuse the set across all per-row canonicalization calls. |
| Rescale/fusion maps | `0f12311` | `experiments/server_command_runs/rescale_adjacency_0f12311_20260703_230927/` | Reuse per-source stage-edge adjacency in reachability and backward DP instead of rescanning all stage edges per cut point. |
| Rescale/fusion maps | `0812807` | `experiments/server_command_runs/feasibility_incremental_0812807_20260703_232545/` | Accumulate feasibility-DAG stage nodes, scale propagation, and edge costs incrementally instead of rebuilding lists and rescanning path nodes for every candidate edge. |
| Rescale/fusion maps | `c48e63d` | `experiments/server_command_runs/feasibility_cutpoint_index_c48e63d_20260703_233525/` | Precompute cut-point node identity indices once during feasibility-DAG construction instead of linearly scanning all cut points for every graph node. |
| Rescale/fusion maps | `5760c6d` | `experiments/server_command_runs/fusion_report_option_scan_5760c6d_20260704_001718/` | Build fusion-map report graph payloads with one ordered-options summary loop after the order check, instead of separately scanning for base option, available fusion counts, and option summaries. |
| Rescale/fusion maps | `1410ba0` | `experiments/server_command_runs/fusion_slots_option_index_1410ba0_20260704_010024/` | Cache fusion-count slots eval report option lookups by options-list identity so repeated selected-option and boost-audit sections avoid rescanning graph options. |
| Rescale/fusion maps | `c15cb03` | `experiments/server_command_runs/fusion_k_independence_count_c15cb03_20260704_022055/` | Count fusion-map K-independence sample configs during the existing scan instead of materializing `sample_configs` a second time. |
| Rescale/fusion maps | `ccbfc5f` | `experiments/server_command_runs/fusion_report_option_index_ccbfc5f_20260704_051827/` | Prebuild fusion report option-id indices so action-config generation reuses option lookups across action-vector and slot splicing. |
| Rescale/fusion maps | `71fbfc6` | `experiments/server_command_runs/fusion_report_field_kinds_71fbfc6_20260704_052234/` | Reuse per-block fusion report field-kind lookups during slot-form action-config splicing instead of rebuilding them per schedule step. |
| Rescale/fusion maps | `82b83ca` | `experiments/server_command_runs/fusion_report_block_actions_82b83ca_20260704_052649/` | Cache adjusted per-graph/per-option block actions during fusion report action-config splicing instead of rebuilding action-index lists per schedule step. |
| Rescale/fusion maps | `f8d649e` | `experiments/server_command_runs/fusion_report_slot_entries_f8d649e_20260704_053236/` | Cache bound per-graph/per-option slot entries during fusion report slot-form action-config splicing instead of rebinding and resorting slots per schedule step. |
| Rescale/fusion maps | `476a230` | `experiments/server_command_runs/fusion_report_bound_slot_items_476a230_20260704_053630/` | Iterate fusion report slot mappings directly in bound-slot compatibility expansion instead of copying through `dict(slots)`. |
| Rescale/fusion maps | `269ba69` | `experiments/server_command_runs/fusion_report_slot_mapping_269ba69_20260704_072557/` | Normalize fusion report option/base slot mappings by iterating mapping `.items()` directly instead of copying through `dict(...).items()` for each option summary. |
| Rescale/fusion maps | `74d5d28` | `experiments/server_command_runs/fusion_report_occurrences_74d5d28_20260704_073030/` | Accumulate fusion report graph occurrence layers as sets during the schedule scan instead of building lists and then deduplicating with `sorted(set(v))`. |
| Rescale/fusion maps | `c3db582` | `experiments/server_command_runs/fusion_report_action_sequence_c3db582_20260704_073447/` | Index fusion report option/base action-index sequences directly in `_option_slot_summary()` instead of copying the full sequences through integer list comprehensions. |
| Rescale/fusion maps | `b0a1928` | `experiments/server_command_runs/fusion_report_base_action_b0a1928_20260704_073904/` | Pass the fusion report base option action-index sequence directly into option summaries instead of copying it through an integer list comprehension once per graph. |
| Rescale/fusion maps | `248a0ec` | `experiments/server_command_runs/fusion_map_gate_filter_248a0ec_20260704_094103/` | Reuse the fusion-report canonical map-path filter in the active `SERVER_COMMAND.md` phase2 map gate, so sidecars such as `map_summary.json` are skipped before JSON parsing instead of being opened and failing on missing `options`. |
| Rescale bridge | `dab3b8b` | `experiments/server_command_runs/baseline_archive_cache_dab3b8b_20260703_212500/` | Cache static-skeleton archive parses by path, mtime, and size while returning fresh caller lists. |
| Skeleton map discovery | `cb215bd` | `experiments/server_command_runs/skeleton_profile_config_discovery_cb215bd_20260703_213500/` | Discover profile config JSON files with `os.scandir()` and skip `.json` directories before parsing. |

Remaining main-chain gates before this goal can be complete:

1. **Evidence loop:** every new source optimization must have a red/green or
   parity evidence directory committed back from the server.
2. **Stage-1 throughput default promotion:** `9f3864d` promotes the Stage-1-only
   RL launcher's omitted `--batch-size` default from the generic 16 to 128. The
   no-batch default runtime gate shows `g4` at `9007.153` ep/h versus `g1` at
   `7558.355` ep/h. Keep `--stage1-rl-devices` explicit for now; do not auto-
   select visible GPUs until a separate 4GPU/5GPU and queue-safety gate proves
   that default is safe.
3. **Stage-2 scheduling:** keep core RL files out of scope while the Stage-2
   RL agent is active; after handoff, run 1GPU vs NGPU parity and wall-clock
   gates before promoting GPU defaults.
4. **Rescale/fusion maps:** the active `SERVER_COMMAND.md` sidecar parser gap
   is closed by `248a0ec`. Continue only with profiled session/graph/DAG reuse
   and large-build server evidence when the claim is about build wall time or
   memory.
5. **Paean final eval:** finish model/tokenizer reuse and independent-config
   GPU scheduling only after fixed-action metric parity evidence.
6. **Artifacts/reports:** keep required JSON/JSONL/NPZ data complete while
   moving remaining PNG/HTML-heavy rendering to post-run commands.
7. **Workflow:** keep local source, git, and server evidence synchronized after
   every optimization; server temp packages may run code but must not become
   the canonical source.

## File Structure

- Create `scripts/project_optimization_audit.py`: dependency-free whole-flow inventory and artifact summary.
- Create `tests/test_project_optimization_audit.py`: unit tests for inventory, artifact discovery, and CLI output.
- Modify `llama_7B_LayerImportance.sh`: only for launcher/server resource checks after tests prove behavior.
- Modify `stage1_rl/*` and `layer_importance_evaluator.py`: only for Stage-1 evaluation/rollout improvements with Stage-1 parity evidence.
- Modify `blb_stage2_rl/*`: only after coordinating with concurrent Stage-2 RL work; require 1GPU vs NGPU gates.
- Modify `Rescale_optimizer/rescale_optimizer/*`: only for profiled CPU hot paths or safe memoization.
- Modify `Paean/*` and `final_evaluation_module.py`: only for final-eval batching, reuse, or independent-config scheduling.
- Modify `rl_data_points.py`, reports, and verifier scripts: only for hot-path/report decoupling and artifact integrity.

## Task 1: Whole-Flow Optimization Audit Tool

**Files:**

- Create: `scripts/project_optimization_audit.py`
- Create: `tests/test_project_optimization_audit.py`
- Verify: `docs/superpowers/specs/2026-07-02-whole-project-runtime-optimization-design.md`

- [x] **Step 1: Write failing tests**

Create tests that build a temporary mini-repo with representative files:

```python
root = Path(td)
(root / "llama_7B_LayerImportance.sh").write_text("#!/usr/bin/env bash\n")
(root / "presets").mkdir()
(root / "presets" / "mrpc-blb-stage2-rl.conf").write_text("--stage2-rl-variant\nblb_v3\n")
(root / "blb_stage2_rl").mkdir()
(root / "blb_stage2_rl" / "parallel_runner.py").write_text("")
(root / "run" / "diagnostics").mkdir(parents=True)
(root / "run" / "diagnostics" / "episodes.jsonl").write_text('{"episode":0}\n')
```

Assert that `build_project_audit(root)` reports:

- stage names include `launcher`, `stage1`, `stage2`, `rescale`, `paean`, `artifacts`.
- existing files are marked `present=True`.
- artifact summary counts one `episodes.jsonl`.
- CLI writes JSON and Markdown.

Run:

```bash
python3 -m unittest tests.test_project_optimization_audit -v
```

Expected before implementation: FAIL because the script does not exist.

- [x] **Step 2: Implement the audit tool**

Implement these functions:

```python
def build_project_audit(root: str | Path, artifact_roots: Sequence[str | Path] = ()) -> dict:
    ...

def render_markdown(report: Mapping[str, object]) -> str:
    ...

def main(argv: Sequence[str] | None = None) -> int:
    ...
```

The tool must be stdlib-only, deterministic, and safe on a dirty worktree. It
must not import torch or training modules.

- [x] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_project_optimization_audit -v
python3 -m py_compile scripts/project_optimization_audit.py
python3 -m ruff check scripts/project_optimization_audit.py tests/test_project_optimization_audit.py
```

Expected: all pass.

Progress 2026-07-02: `scripts/project_optimization_audit.py` now classifies
artifact evidence filenames with direct deterministic rules instead of running
every file through every glob pattern. A local 200k-name parity benchmark kept
the same counts and reduced classification time from `0.622s` to `0.093s`
(`6.69x`).

Progress 2026-07-03: `scripts/project_optimization_audit.py` CLI JSON output
now writes through `json.dump(..., handle)` instead of materializing the whole
audit report with `json.dumps()` before `Path.write_text()`. The audit tool
remains stdlib-only and keeps deterministic indentation, sorted keys, and a
trailing newline while avoiding one extra full-report string allocation.

- [x] **Step 4: Commit**

```bash
git add scripts/project_optimization_audit.py tests/test_project_optimization_audit.py docs/superpowers/specs/2026-07-02-whole-project-runtime-optimization-design.md docs/superpowers/plans/2026-07-02-whole-project-runtime-optimization.md
git commit -m "Add whole-project optimization plan and audit"
```

## Task 2: Launcher and Server Resource Gates

**Files:**

- Modify: `scripts/launcher_gpu_audit.py`
- Create: `scripts/server_resource_snapshot.py`
- Modify: `llama_7B_LayerImportance.sh`
- Test: `tests/test_launcher_gpu_audit.py`
- Test: `tests/test_server_resource_snapshot.py`
- Test: `tests/test_stage2_persistent_launcher.py`

- [x] **Step 1: Extend tests**

Add tests for these cases:

- Stage-1 `--stage1-rl-devices` set to fewer devices than `CUDA_VISIBLE_DEVICES`.
- Stage-2 `--stage2-rl-devices` and `--blb-v3-reward-devices` disagree.
- `RFR_GPU_AUDIT_STRICT=1` fails only when warnings exist.

- [x] **Step 2: Implement warnings only**

Keep default behavior non-fatal. Add strict failure only through
`RFR_GPU_AUDIT_STRICT=1`.

Progress 2026-07-02: added torch-free `scripts/server_resource_snapshot.py`
for pre-run GPU inventory/utilization, CPU/load, and git dirty-state evidence.
`scripts/optimization_evidence_bundle.py` now includes this snapshot in every
bundle.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now parses offline
`nvidia-smi` CSV input from a line iterator instead of `Path.read_text()`,
keeping server evidence capture memory-bounded if a sampled GPU log is supplied
instead of a tiny one-shot inventory.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now skips blank
offline `nvidia-smi` CSV rows with `isspace()` before handing lines to
`csv.reader`, instead of allocating stripped copies during row filtering. A
local 100k valid-row / 100k blank-row GPU CSV benchmark preserved the collapsed
GPU summary and reduced parsing from `2.666752s` to `2.537524s`, with traced
peak allocation dropping from `27,575B` to `23,068B`.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now detects sampled
`nvidia-smi` CSV headers and collapses repeated samples into one max-util/max
memory row per GPU. This keeps snapshot `gpu_count` tied to unique devices and
keeps evidence bundles small when they reuse a long sampled GPU log. A local
100k-sample CSV benchmark reduced parsed rows from `100001` to `4`, traced
peak memory from `28.24MB` to `0.05MB`, and serialized GPU JSON from about
`10.5MB` to `504B`.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now bounds its
best-effort `nvidia-smi` and git subprocess calls with a 5-second timeout.
This keeps pre-run resource evidence collection from stalling a server launch
when the GPU driver CLI or git status is slow, while preserving the existing
empty-output fallback behavior.

Progress 2026-07-02: `scripts/server_resource_snapshot.py` now summarizes
`git status --porcelain` output by counting newline-delimited records and
retaining only the first 20 examples, instead of materializing every dirty line
with `splitlines()`. A regression test guards against calling `splitlines()` on
the dirty-status payload. A local 200k-row synthetic dirty-status benchmark
preserved count/examples and improved the summary from `0.087380s` / `17.21MB`
to `0.002384s` / near-zero traced allocations (`36.66x`).

Progress 2026-07-02: `scripts/launcher_gpu_audit.py` now bounds fallback
`nvidia-smi` GPU discovery with a 5-second timeout. This prevents the
non-fatal GPU audit gate from stalling expensive RL launchers when the server
driver/CLI is slow or wedged and `CUDA_VISIBLE_DEVICES` was not set.

Progress 2026-07-02: `scripts/launcher_gpu_audit.py` now parses comma device
lists and fallback `nvidia-smi` output through one shared single-strip helper,
so each nonblank token is trimmed once instead of twice. A regression test
counts `strip()` calls on synthetic `nvidia-smi` output. A local 200k-line
synthetic parser benchmark preserved device IDs and improved parsing from
`0.013890s` to `0.010055s` (`1.38x`).

Progress 2026-07-02: startup/preset parsing now streams configuration files
instead of materializing full `read().splitlines()` lists. `tools/validate_preset.py`
parses preset files with a one-line lookahead iterator, and `Paean/config.py`
streams final-eval preset lines through `shlex`. Local synthetic preset
benchmarks preserved parsed arguments/flags; `validate_preset` improved from
`0.2321s` / `25.81MB` to `0.2060s` / `21.16MB`, and Paean preset parsing
reduced peak memory from `21.50MB` to `10.62MB`.

Progress 2026-07-02: `scripts/server_resource_snapshot.py`
`parse_nvidia_smi_csv()` now streams text through a small line iterator instead
of materializing `splitlines()`. This keeps resource snapshot parsing
memory-bounded when callers pass long sampled `nvidia-smi` text. A local 100k
sample benchmark preserved collapsed GPU rows while reducing traced peak memory
from `9.53MB` to `0.02MB` and wall time from `1.0655s` to `0.9515s`.

- [x] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_launcher_gpu_audit tests.test_stage2_persistent_launcher -v
bash -n llama_7B_LayerImportance.sh
python3 -m ruff check scripts/launcher_gpu_audit.py tests/test_launcher_gpu_audit.py tests/test_stage2_persistent_launcher.py
```

## Task 3: Stage-1 Evaluation and Rollout Throughput

**Files:**

- Modify: `stage1_rl/eval_cache.py`
- Modify: `stage1_rl/parallel_runner.py`
- Modify: `layer_importance_evaluator.py`
- Create: `scripts/stage1_parallel_report.py`
- Test: `tests/test_stage1_eval_accel.py`
- Test: `tests/test_stage1_parallel_semantics.py`
- Test: `tests/test_stage1_parallel_report.py`

- [x] **Step 1: Baseline current behavior**

Run focused local tests:

```bash
python3 -m unittest tests.test_stage1_eval_accel tests.test_stage1_parallel_semantics -v
```

Server evidence 2026-07-04: because this workflow forbids local execution, the
focused Stage-1 gate was run on the server from temporary source packages. The
initial run at source `8faf478` failed two semantic locks; final source
`dbd1b6f` passes `tests.test_stage1_eval_accel`,
`tests.test_stage1_parallel_semantics`, and `tests.test_blb_inference_eval_shared`
under
`experiments/server_command_runs/stage1_semantics_gate_dbd1b6f_20260704_080342/`.
This validates the Stage-1 focused semantic gate only; it is not 1GPU vs 4GPU
throughput evidence.

- [x] **Step 2: Add timing fields**

Add Stage-1 window diagnostics for cache hit rate, worker wall seconds,
model-forward wall seconds, and report-write wall seconds. Write them to the
existing Stage-1 log/status path, not to a new hot-path report.

Server evidence 2026-07-04: source commit `b62743a` adds model-forward wall
seconds / forward-call counts to Stage-1 worker-window diagnostics, emits
`report_write` alongside the existing `detail` field in
`[stage1-rollout-total]`, and teaches `scripts/stage1_parallel_report.py` to
summarize these fields while staying backward compatible with old logs. The
final committed-source server gate at
`experiments/server_command_runs/stage1_timing_fields_b62743a_20260704_082005/`
passed `py_compile` plus `tests.test_stage1_eval_accel`,
`tests.test_stage1_parallel_semantics`, and `tests.test_stage1_parallel_report`
(`50` tests). This completes the timing-field prerequisite only; it is still
not 1GPU vs 4GPU throughput evidence.

Progress 2026-07-02: added torch-free `scripts/stage1_parallel_report.py` to
summarize existing Stage-1 rollout/cache/component timing logs into JSON and
Markdown for server 1GPU vs 4GPU evidence. Existing
`tests.test_stage1_parallel_semantics` currently fails against the dirty
`layer_importance_evaluator.py` worktree because `_stage1_collect_episode_in_worker`
no longer contains `SOS_TOKEN_SOFTMAX`; this optimization pass did not modify
that core file.

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now parses Stage-1
training logs from a single-pass line iterator, and
`scripts/optimization_evidence_bundle.py` streams Stage-1 log files into that
parser instead of concatenating all logs into one large in-memory string.

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now aggregates
Stage-1 rollout windows, total timing windows, cache status, and worker speedup
with running counters instead of retaining every parsed window row before
rendering the report.

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now skips regex
matching for unrelated training-log lines that do not contain
`[stage1-rollout`, and `parse_log_text()` streams text through a small line
iterator instead of materializing `splitlines()`. A local noisy 250k-line log
benchmark preserved the full summary; the file/line parser path improved from
`1.1329s` to `0.8107s` (`1.40x`), while the text-entry path reduced
`splitlines()` peak memory from `38.77MB` to `0.01MB`.

Progress 2026-07-02: `scripts/stage1_approx_reuse_benchmark.py` now summarizes
install/forward/total timing means and speedups in one helper using
`math.fsum()`, avoiding repeated `statistics.mean()` calls and intermediate
millisecond/total timing lists after a server benchmark completes. A local
200k-row timing-summary benchmark produced matching rounded results, reduced
summary time from `2.133355s` to `0.124024s` (`17.20x`), and eliminated the
`18.39MB` traced peak from list materialization.

Progress 2026-07-02: `scripts/stage1_parallel_report.py` now checks Stage-1
worker episode-count imbalance without materializing `worker_episode_counts`
into a list and without scanning the minimum twice. For real `dict_values`
inputs it uses built-in `min/max` over the existing view; single-pass iterables
fall back to one-pass aggregation. A local 200k-worker synthetic count
benchmark preserved the warning result, reduced time from `2.868017s` to
`1.976543s` (`1.45x`), and cut traced peak allocation from `1,600,128B` to
`72B`.

Progress 2026-07-03: `_select_piecewise_gelu_output()` in `function_handler.py`
now uses scalar `0.0` for the low/NaN branch instead of allocating
`torch.zeros_like(x)`. The helper is shared by regular Stage-1
`PolynomialGELU` and Block-5 noisy GELU, so this removes one full-shape zero
tensor allocation from both installed forward paths while preserving the same
piecewise boundaries.

Progress 2026-07-03: Block-2 QK-merge, Block-2 BSGS, and Block-4 input /
softmax-V ones-mask encode hooks in `function_handler.py` now sample the
same-shape encode noise against the current tensor and `add_(1.0)` instead of
allocating `torch.ones_like(...)` before adding noise. This removes five
full-shape ones tensor allocations from shared installed forward paths while
preserving the CKKS `ones + encode-noise` mask semantics.

Progress 2026-07-03: BERT and GPT-2 approximation-softmax lower-bound masks in
`function_handler.py` now use scalar `0.0` in `torch.where(...)` instead of
allocating `torch.zeros_like(x)` before discarding below-band values. This
removes two full-shape zero tensor allocations from shared approximate-softmax
forward paths while preserving the same lower-bound zeroing semantics.

Progress 2026-07-03: single-GPU `LayerImportanceEvaluator.evaluate_model()` now
uses the same `Stage1EvalCache` helper as the worker path and keys cache entries
by the resolved split name. This avoids duplicate validation_full forwards when
callers reach the same Stage-1 plaintext configuration through equivalent
`use_train=False` / `split="validation_full"` entrypoints, while preserving the
existing install/forward semantics.

Progress 2026-07-03: Stage-1 parallel rollout replay now keeps prefetched
rollouts in a `deque` and consumes them with `popleft()` instead of repeatedly
calling `list.pop(0)`. This preserves global rollout order while avoiding
per-episode list shifting in every PPO update window.

Progress 2026-07-04: Stage-1 recurrent rollout packing first batched tensor
`logprobs` and `values` before PPO updates, replacing per-step `.item()`
scalar syncs with a stacked conversion. The current source then goes further:
`_stage1_scalar_episode_values_to_tensor()` returns target-device tensors for
those scalar fields, so tensor-backed rollouts avoid the intermediate CPU numpy
array and the later `torch.from_numpy(...).to(device)` transfer. Plain float
and mixed-value fallback paths preserve scalar conversion semantics.

Server evidence 2026-07-04: source commit `54feaa4` has red/green verification
under
`experiments/server_command_runs/stage1_rollout_pack_batch_54feaa4_20260704_011755/`.
The green gate compiles `layer_importance_evaluator.py`, verifies the new
source guard, and runs a functional tensor-pack script. Two unrelated numeric
tests in `tests.test_stage1_eval_accel` still fail on clean `8336eef`; the
evidence directory includes that baseline reproduction so this change does not
claim a clean full-module gate.

Server evidence 2026-07-04: source commit `92ad0f0` has red/green verification
under
`experiments/server_command_runs/stage1_rollout_direct_tensor_92ad0f0_20260704_012541/`.
The green gate compiles `layer_importance_evaluator.py`, verifies the
direct-to-device source guards, runs all `tests.test_stage1_parallel_semantics`
source tests, and runs a functional `RecurrentRolloutBuffer.get_batch()` script
that returns `logprobs` and `values` on `cuda:0`.

Progress 2026-07-04: `blb_stage2_rl/inference_eval.py` now converts reward-probe
`loss`, `metric1`, and `metric2` scalar tensor sequences with one packed helper
instead of calling the per-sequence scalar conversion helper three times. Probe
trial metric semantics, sample weighting, prediction/label aggregation, and
model train/eval restoration stay unchanged while the shared reward-probe path
does one packed CPU transfer for those scalar metric fields.

Server evidence 2026-07-04: source commit `497ecda` has red/green verification
under
`experiments/server_command_runs/probe_scalar_sync_497ecda_20260704_025145/`.
The red test failed because `run_installed_probe_trial()` still used three
per-field scalar conversions. The green gate passed `py_compile`, all six
`tests.test_blb_inference_eval_shared` tests, and a source guard proving the
batched helper replaces those three old calls.

Progress 2026-07-04: `run_installed_probe_trial()` now gates prediction/label
tensor retention and numpy transfer behind the metric profiles that actually
need full arrays: regression profiles and MRPC/QQP weighted-F1 profiles.
Accuracy-only profiles such as SST-2 and RTE keep using sample-weighted
per-batch accuracy metrics and avoid storing every batch's prediction/label
tensors or transferring them back to CPU.

Server evidence 2026-07-04: source commit `b5dfff5` has red/green verification
under
`experiments/server_command_runs/probe_skip_pred_arrays_b5dfff5_20260704_025610/`.
The red test proved the old accuracy-only path still called
`tensor_values_to_numpy_arrays()`. The green gate passed `py_compile`, all
seven `tests.test_blb_inference_eval_shared` tests, and a source guard
confirming the `need_prediction_arrays` predicate controls tensor retention and
numpy transfer.

Progress 2026-07-04: `tensor_values_to_numpy_arrays()` now has a same-device
tensor fast path for reward-probe prediction/label arrays. When MRPC/QQP or
regression probes actually need full predictions and labels, tensors are
flattened and concatenated on-device first, then transferred to CPU/numpy as one
array. Mixed inputs and mixed-device fallbacks keep the previous per-value
conversion behavior.

Server evidence 2026-07-04: source commit `2d98907` has red/green verification
under
`experiments/server_command_runs/probe_tensor_arrays_2d98907_20260704_030105/`.
The red test proved the old helper returned one numpy array per tensor. The
green gate passed `py_compile`, all eight `tests.test_blb_inference_eval_shared`
tests, and a source guard confirming the `torch.cat(...).cpu().numpy()` packed
transfer path.

Progress 2026-07-04: `blb_stage2_rl/eval_metrics.py`
`weighted_probe_batch_means()` now converts probe batch `counts` into one
non-negative weight array and reuses the same weight sum for loss, metric1, and
metric2 weighted means. This keeps `sample_weighted_mean()` compatibility for
other callers while avoiding three repeated count-list conversions in every
reward-probe trial summary.

Server evidence 2026-07-04: source commit `da02fca` has red/green verification
under
`experiments/server_command_runs/eval_metric_weights_da02fca_20260704_030610/`.
The red test failed because the old `weighted_probe_batch_means()` iterated
`counts` once per metric. The green gate passed `py_compile`, all six
`tests.test_blb_eval_metrics_shared` tests, and a source guard confirming the
shared weight helper replaces the three old `sample_weighted_mean()` calls.

Progress 2026-07-04: `finalize_probe_trial_metrics()` now routes prediction
and label arrays through `_flatten_probe_arrays()`. Single packed arrays
produced by the shared inference fast path are reshaped directly instead of
copied through `np.concatenate()`, while the multi-array fallback keeps the
previous concatenation behavior.

Server evidence 2026-07-04: source commit `1a6969a` has red/green verification
under
`experiments/server_command_runs/eval_single_array_1a6969a_20260704_031145/`.
The red test proved the old finalizer called `np.concatenate()` even for one
packed array. The green gate passed `py_compile`, all seven
`tests.test_blb_eval_metrics_shared` tests, and a source guard confirming the
single-array fast path and helper usage in both finalizer branches.

Progress 2026-07-04: `weighted_f1_from_labels()` now has a direct 0/1 binary
fast path for MRPC/QQP-style label and prediction arrays. The function computes
positive and negative F1 from count reductions and only falls back to
`np.union1d()` sorting for non-binary class sets.

Server evidence 2026-07-04: source commit `f9bbb29` has red/green verification
under
`experiments/server_command_runs/eval_binary_f1_f9bbb29_20260704_034430/`.
The red test proved the old binary weighted-F1 path called `np.union1d()`. The
green gate passed `py_compile`, all eight
`tests.test_blb_eval_metrics_shared` tests, and a source guard confirming the
binary fast path precedes the class-union fallback.

Progress 2026-07-04: `matthews_corrcoef_from_labels()` now has a direct 0/1
binary fast path for CoLA-style label and prediction arrays. It computes the
MCC numerator and denominator from count reductions and preserves the existing
`np.union1d()` fallback for non-binary or multi-class inputs.

Server evidence 2026-07-04: source commit `d0e8b8c` has red/green verification
under
`experiments/server_command_runs/eval_binary_mcc_d0e8b8c_20260704_035430/`.
The red test proved the old binary MCC path called `np.union1d()`. The green
gate passed `py_compile`, all nine `tests.test_blb_eval_metrics_shared` tests,
and a source guard confirming the binary MCC fast path precedes the
class-union fallback.

Progress 2026-07-04: `accuracy_from_labels()` now computes classification
accuracy with a direct `np.count_nonzero(preds == labels) / n` match count
instead of calling generic `np.mean()` on the boolean match mask. This trims a
shared metric hot path used by MRPC/QQP/SST-2/RTE-style probes.

Server evidence 2026-07-04: source commit `211ca50` has red/green verification
under
`experiments/server_command_runs/eval_accuracy_count_211ca50_20260704_041100/`.
The red test proved the old accuracy path called `np.mean()`. The green gate
passed `py_compile`, all ten `tests.test_blb_eval_metrics_shared` tests, and a
source guard confirming the direct count path without `np.mean()`.

Progress 2026-07-04: `run_installed_model_on_dataloader()` now reuses the
shared `accuracy_from_labels()` helper for its MNLI full-eval branch. This
removes the remaining local `np.mean(pred_classes == all_labels)` accuracy
implementation from the installed inference path, so future accuracy hot-path
improvements stay centralized.

Server evidence 2026-07-04: source commit `7be83af` has red/green verification
under
`experiments/server_command_runs/inference_mnli_accuracy_helper_7be83af_20260704_042030/`.
The red test proved the old MNLI full-eval branch called the local `np.mean()`
path. The green gate passed `py_compile`, all nine
`tests.test_blb_inference_eval_shared` tests, and a source guard confirming the
MNLI branch uses the shared accuracy helper.

Progress 2026-07-03: Stage-1 plaintext repeat evaluation and the MRPC
layer-output noise experiment now use pinned DataLoader memory with
`non_blocking=True` tensor transfers when CUDA is available. Their GPU eval
loops also defer loss/correct or label/pred CPU synchronization until after the
batch loop, reducing per-batch device synchronization while preserving the same
metrics and deterministic evaluation protocol.

Progress 2026-07-04: `scripts/stage1_plaintext_repeat_eval.py` now writes its
stdout JSON summary with `json.dump(..., sys.stdout)` plus a trailing newline
instead of building a second complete JSON string through `json.dumps()` before
`print()`. The required `--output-json` artifact still uses `write_json_file()`,
and the evaluation protocol/metrics are unchanged.

Server evidence 2026-07-04: source commit `5d248a0` has focused RED/GREEN
verification under
`experiments/server_command_runs/stage1_stdout_json_stream_5d248a0_20260704_103647/`.
The RED test failed on the old `print(json.dumps(summary, ...))` stdout path.
The GREEN gate passed `py_compile` and all three
`Stage1GpuEvalScriptSourceTest` tests.

Progress 2026-07-03: `LayerImportanceEvaluator.apply_configuration()` now
short-circuits repeated GELU/Softmax installs for an unchanged configuration
while still forcing the model into eval mode on every call. This extends the
existing `evaluate_model()` local skip to final-eval and noise-eval call sites
that invoke `apply_configuration()` directly.

Server evidence 2026-07-03: source commit `dca7526` has red/green verification
under
`experiments/server_command_runs/stage1_apply_config_reuse_dca7526_20260703_210000/`.
The red test proved repeated installs were not skipped; the green run verifies
that unchanged-config installs are skipped while `model.eval()` is still called.

Progress 2026-07-03: `_stage1_evaluate_on_model()` now applies the same
unchanged-configuration install short-circuit to each Stage-1 worker handler.
The worker still runs a validation forward for uncached evaluations, but a
worker no longer repeats GELU/Softmax restore/replace calls when its replica is
already on the requested configuration.

Server evidence 2026-07-03: source commit `5d15e6c` has red/green verification
under
`experiments/server_command_runs/stage1_worker_apply_config_reuse_5d15e6c_20260703_211000/`.
The scope is the worker-side `_stage1_evaluate_on_model()` install path, not a
claim about end-to-end 4GPU speedup.

Progress 2026-07-03: `BertSelfAttentionWithAproximation.forward()` now consumes
legacy positional tail arguments with a head index cursor instead of
`list.pop(0)`. Tail-end `pop()` handling for `cache_position` and
`output_attentions` stays unchanged, while encoder masks and past-key values no
longer shift the remaining argument list during attention forward parsing.

Server evidence 2026-07-03: source commit `a416d46` has red/green verification
under
`experiments/server_command_runs/attention_tail_cursor_a416d46_20260703_214800/`.
The green gate compiles `function_handler.py` and verifies the source-level
performance guard that this parsing region contains `tail_pos = 0` and no
`pop(0)`.

Progress 2026-07-03: Stage-1 runtime reward normalization history now uses
`deque(maxlen=RUNNING_REWARD_HISTORY_SIZE)` for initialization, runtime reset,
and checkpoint resume. `update_reward_statistics()` still appends every new
episode reward and computes the same mean/std once enough samples exist, but
the window no longer trims overflow with list `pop(0)` on every post-window
episode.

Server evidence 2026-07-03: source commit `61c8c57`, with test-head
`392b646`, has red/green verification under
`experiments/server_command_runs/stage1_reward_history_deque_392b646_20260703_215700/`.
The first green attempt compiled but failed a too-narrow source assertion; the
final green run verifies `py_compile=0` and the bounded-deque source guard.

- [x] **Step 3: Optimize only proven redundant work**

Allowed changes:

- Share deterministic cache for worker evals.
- Avoid rebuilding identical Stage-1 GELU/Softmax installs when the config
  hash is unchanged.
- Keep worker seeding and validation_full split unchanged.

Server evidence 2026-07-04: Stage-1 redundant-work optimizations now have
focused server evidence for unchanged-config install reuse, worker-handler
install reuse, reward-history bounded windows, tensor-backed rollout packing,
direct noise-scaling validation scans, and timing diagnostics. The Stage-1
validation protocol remains `validation_full`; this pass did not change
worker seeding or the Stage-1 PPO objective.

- [x] **Step 4: Verify**

Run the local tests above, then run a server 1GPU vs 4GPU smoke before changing
defaults.

Server evidence 2026-07-04: source commit `4bca31a` is verified under
`experiments/server_command_runs/stage1_gpu_ab_4bca31a_20260704_084330/`.
The new launcher audit regression test first failed against `877eefa` because
`scripts/launcher_gpu_audit.py` could not import `device_utils` when executed
as a script from repo root without `PYTHONPATH`; after the import-path fix,
`tests.test_launcher_gpu_audit` passed all `11` tests plus a direct CLI run.
The same evidence directory contains the formal 170-episode Stage-1 MRPC
1GPU vs 4GPU gate. Both runs completed with `launcher_rc=0`, `wait_rc=0`, and
`COMPLETED`; 4GPU used `cuda:0..3` with worker counts `43/43/42/42`. This gate
also exposed that the total timing line counted `168` episodes by multiplying
`num_workers * floor(episodes_per_worker)` instead of using actual worker
counts. Source commit `4834b2f`, verified under
`experiments/server_command_runs/stage1_parallel_episode_count_4834b2f_20260704_085732/`,
fixes that diagnostics undercount. Do not change Stage-1 defaults to 4GPU
until the 170-episode A/B gate is rerun on `4834b2f` or newer.

Server evidence 2026-07-04: the gate was rerun from source commit `180d319`
under
`experiments/server_command_runs/stage1_gpu_ab_180d319_20260704_090423/`.
The diagnostics fix held: the 4GPU run reports `total_episodes=170` with worker
counts `43/43/42/42`. Both runs completed cleanly, and 4GPU used `cuda:0..3`,
but 4GPU remained slower (`wall_clock_speedup_g4_over_g1=0.543`,
parser throughput speedup `0.501`). Keep Stage-1 4GPU default promotion
blocked until duplicated validation/model-forward cost is reduced.

Server evidence 2026-07-04: an explicit batch-size probe from source
`ab9ed62` under
`experiments/server_command_runs/stage1_batch128_gpu_ab_ab9ed62_20260704_091809/`
showed the duplicated validation/model-forward cost was primarily a small-batch
problem: `--batch-size 128` kept single-GPU throughput effectively flat
(`7548.287` ep/h versus `7469.518` ep/h at batch 16) while improving 4GPU
throughput to `8810.700` ep/h and wall speedup to `1.121x`.

Source commit `9f3864d` then promoted the launcher default only for
`run rl --mode stage1-only` when `--batch-size` is omitted. RED
`tests.test_stage1_launcher_defaults` failed against the old source because the
Stage-1-specific default was absent; GREEN passed
`tests.test_stage1_launcher_defaults` plus `tests.test_launcher_gpu_audit` on
the server. The runtime gate from `9f3864d`, stored under
`experiments/server_command_runs/stage1_default_batch_gpu_ab_9f3864d_20260704_092741/`,
launched without `--batch-size`, showed Python receiving
`--batch_size 128 --micro_batch_size 128`, completed both g1/g4 runs, and
reported `g4` throughput `9007.153` ep/h versus `g1` `7558.355` ep/h.

## Task 4: Stage-2 BLB RL GPU Scheduling and Diagnostics

**Files:**

- Modify: `scripts/stage2_ngpu_ab_compare.py`
- Modify: `scripts/gpu_utilization_report.py`
- Modify: `scripts/stage2_reward_probe_scaling_report.py`
- Modify only after coordination: `blb_stage2_rl/parallel_runner.py`,
  `blb_stage2_rl/probe_runner.py`, `blb_stage2_rl/sequential_runner.py`
- Test: `tests/test_stage2_ngpu_ab_compare.py`
- Test: `tests/test_gpu_utilization_report.py`
- Test: `tests/test_stage2_reward_probe_scaling_report.py`
- Test: `tests/test_stage2_parallel_runner.py`

- [x] **Step 1: Strengthen evidence tools**

Ensure reports include:

- per-device episode counts.
- per-device terminal probe wall means.
- policy rollout wall mean.
- replan wall mean.
- JSONL write/report render wall time when present.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now reports
per-device probe episode counts, per-device terminal probe wall statistics,
global policy rollout wall statistics, replan/optimizer wall statistics, and
optional JSONL/report hot-path wall fields when they are present in
`episodes.jsonl`.

Progress 2026-07-02: the same report now streams `episodes.jsonl` through a
single-pass row summarizer instead of loading every episode into memory first.
This keeps report generation and evidence bundling lightweight for 60k+ episode
Stage-2 runs while preserving the existing CLI output shape.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now summarizes
optional `nvidia-smi` CSV samples with running per-device aggregates instead of
keeping every utilization sample in memory before computing means and maxima.
This keeps GPU evidence reports memory-bounded for long server sampling logs.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now normalizes
`nvidia-smi` CSV headers once per file and uses a raw-field lookup for every
sample row, instead of regex-normalizing the header keys again for each row.
A local 200k-row sampled GPU CSV benchmark preserved the per-device summary
and reduced CSV post-processing from `4.274366s` to `2.521548s` (`1.70x`),
with traced peak allocation dropping from `62,023B` to `51,313B`.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now parses sampled
`nvidia-smi` CSV files with `csv.reader` and precomputed column indices instead
of `csv.DictReader`, avoiding one per-row dictionary allocation in long server
GPU sampling logs. A local 100k-row / 4-GPU sampled CSV benchmark preserved the
per-device utilization summary and reduced post-processing from `1.216738s` /
`0.05MiB` peak to `1.002973s` / `0.04MiB`.

Progress 2026-07-02: the same report now aggregates Stage-2 episode timing
fields (`terminal_probe_wall_seconds`, policy rollout, replan/optimizer,
per-device probe walls, and optional hot-path write/render timings) with
running count/sum/min/max stats instead of retaining every float in lists. A
local 100k-row parity benchmark kept the same stats and reduced timing-summary
peak memory from `15.28MB` to `0.02MB`.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now uses a streaming
sorted `os.walk()` fallback to locate nested `episodes.jsonl` files instead of
`Path.rglob("episodes.jsonl")`. A local 241-directory/2401-file synthetic run
tree preserved the same discovery result and reduced fallback lookup from
`0.0084s`/`0.05MB` to `0.0052s`/`0.02MB`.

Progress 2026-07-02: `_find_episodes_path()` in
`scripts/gpu_utilization_report.py` now avoids sorting each directory's
`filenames` list during fallback search because it only needs an
`"episodes.jsonl"` membership check. Directory traversal order remains stable
via `dirnames.sort()`. A local 120k-filename synthetic directory preserved the
same discovered path and reduced fallback lookup from `0.021896s` to
`0.000526s` (`41.66x`), with traced peak memory dropping from `1.37MB` to
`0.92MB`.

Progress 2026-07-02: extracted the reward-probe scaling benchmark postprocessor
from `scripts/stage2_reward_probe_scaling_benchmark.sh` into
`scripts/stage2_reward_probe_scaling_report.py`. The new report script is
unit-tested and streams `runs.jsonl`, per-run `episodes.jsonl`, and sampled
`nvidia-smi` CSV files instead of reading them into large strings.

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now
computes probe wall/speedup means with running totals while retaining the wall
list only for exact median. A local 100k-row parity benchmark reduced report
post-processing from `1.282s`/`7.21MB` to `1.074s`/`4.14MB` (`1.74x` lower
peak memory).

Progress 2026-07-02: the same reward-probe scaling report now sorts the retained
probe-wall list in place and computes the median directly instead of calling
`statistics.median()`, which copies and sorts another list. A local 80k-row
episode JSONL benchmark produced identical summaries, reduced total time from
`0.798515s` to `0.767511s` (`1.04x`), and cut traced peak memory from `3.42MB`
to `2.81MB` (`1.22x`).

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now
normalizes sampled `nvidia-smi` CSV headers once per file and uses raw-field
lookups for every sample row, instead of regex-normalizing field names on each
row. A local 200k-row sampled GPU CSV benchmark preserved max util/memory maps
and reduced GPU-sample post-processing from `3.853528s` to `2.066302s`
(`1.86x`), with traced peak allocation dropping from `60,023B` to `49,098B`.

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now parses
sampled `nvidia-smi` CSV files with `csv.reader` and precomputed column indices
instead of `csv.DictReader`, avoiding one dictionary allocation per sample row.
A local 120k-row / 4-GPU sampled CSV benchmark preserved max util/memory maps
and reduced post-processing from `1.039389s` / `0.05MB` peak to `0.910350s` /
`0.04MB` (`1.14x`).

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py`
`render_html()` now iterates the summary `runs` collection directly instead of
wrapping it with `list()` before rendering rows. This preserves HTML output and
lets future streaming callers avoid one full run-list materialization. A local
100k-row render benchmark had comparable wall time (`1.530112s` vs
`1.540720s`) while reducing traced peak memory from `29.26MB` to `28.50MB`.

Progress 2026-07-02: `scripts/gpu_utilization_report.py` and
`scripts/stage2_reward_probe_scaling_report.py` now parse JSONL episode lines
directly instead of allocating `line.strip()` copies before `json.loads()`,
while still skipping whitespace-only lines. Local 80k-row long-line benchmarks
preserved report summaries and reduced `gpu_utilization_report` from
`3.0698s` to `2.8886s` (`1.06x`) and reward-probe scaling episode summary
from `1.3052s` to `1.2424s` (`1.05x`).

Progress 2026-07-02: `scripts/gpu_utilization_report.py` now uses a
precompiled hot-path float regex for episode timing fields and sampled
`nvidia-smi` values, instead of dispatching through `re.search()` for every
numeric parse. A local 200k-value mixed GPU/timing benchmark preserved parsed
values and reduced `_float_value()` from `0.103639s` to `0.072524s` (`1.43x`).

Progress 2026-07-04: `scripts/gpu_utilization_report.py` Markdown rendering now
streams visible/used/idle device iterables through `_join_or_none()` instead of
wrapping each collection in `list()` before joining. Empty iterables still
render as `none`, and existing Markdown device order is preserved.

Server evidence 2026-07-04: source commit `9412e3b` has focused RED/GREEN
verification under
`experiments/server_command_runs/gpu_markdown_device_stream_9412e3b_20260704_104315/`.
The RED test failed on the old `list(summary.get(...))` path. The GREEN gate
passed `py_compile`, the new no-materialization test, representative CLI
Markdown output, and core probe-device summary output.

Progress 2026-07-02: `scripts/stage2_reward_probe_scaling_report.py` now uses
the same precompiled float regex for probe wall/speedup values, trial counts,
and sampled `nvidia-smi` utilization/memory fields. A local 200k-value mixed
scaling-report benchmark preserved parsed values and reduced `_float_value()`
from `0.108688s` to `0.072969s` (`1.49x`).

Progress 2026-07-02: `blb_stage2_rl/fusion_cost.py`
`compute_fusion_cost_saving()` now reuses the per-call fusion/truncation maxima
it already accumulates to derive the default `max_actual`, instead of rescanning
the same block choices through `max_actual_for_choices()`. A local 47-block
hot-path benchmark preserved all reward-cost outputs and reduced the full
calculation from `1.867492s` to `1.701235s` over 10k iterations (`1.10x`).

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now analyzes ordered
Stage-2 `episodes.jsonl` files with a streaming two-pass path for summary and
bounded-window rows, avoiding materializing full 60k+ episode lists during A/B
HTML generation. If an input file is out of episode order, it falls back to the
legacy load-and-sort path to preserve report semantics.

Progress 2026-07-02: the ordered Stage-2 A/B path now combines episode-order
scanning and bounded-window aggregation in the first pass, then performs only
one second pass for tail-aware summary statistics. This reduces ordered
`episodes.jsonl` reads from three passes to two while preserving the legacy
list-based summary/window output.

Progress 2026-07-03: `scripts/stage2_ngpu_ab_compare.py` now loads
`episodes.jsonl` and optional `ppo_updates.jsonl` through the shared
`jsonl_utils.iter_jsonl(errors="raise")` path instead of carrying a local
`open()` / `strip()` / `json.loads()` loop. The A/B report still sorts by
episode/update before equality comparison, but long-run evidence parsing now
shares the same low-copy JSONL implementation as the other Stage-2 GPU
diagnostic reports.

Progress 2026-07-03: `scripts/stage2_ngpu_ab_compare.py` now scans each
Stage-2 rollout log once for both `[stage2-rollout-timing]` summaries and
N-GPU marker flags such as worker-local probe noise scopes, worker-local CUDA
streams, and CPU policy mode. This avoids rereading the same long launch log
three extra times while preserving the existing report fields.

Progress 2026-07-03: `scripts/stage2_ngpu_ab_compare.py` now computes episode
timestamp spans with a single min/max pass instead of collecting every
timestamp into a temporary list before calling `min()` and `max()`. Long
1GPU-vs-NGPU evidence reports keep the same wall-source fallback while avoiding
one extra 60k-row float list.

Progress 2026-07-04: `scripts/stage2_ngpu_ab_compare.py` now materializes the
canonical excluded-key set once at the start of `compare_rows()` and reuses it
for both 1GPU and N-GPU rows. This removes two repeated `set(excluded_keys)`
allocations per compared episode/PPO-update row while preserving the same
timing/device and diagnostic-bookkeeping exclusions.

Server evidence 2026-07-04: source commit `623cd5d` has focused red/green
verification under
`experiments/server_command_runs/stage2_ab_excluded_keys_623cd5d_20260704_101330/`.
The red package failed the new regression test at the old repeated
`set(excluded_keys)` path. The green package passed that test, `py_compile`,
and the complete `tests.test_stage2_ngpu_ab_compare` suite (`11` tests).

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now checks common
`blb_stage2_best_action_full.json` locations directly before falling back to a
recursive directory walk. A local synthetic run tree with 402 directories and
2001 files preserved the same best-action payload and reduced common-path
lookup from `0.0064s`/`0.03MB` to `0.0001s`/`0.01MB`.

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now skips blank JSONL
lines with `isspace()` and passes nonblank episode lines directly to
`json.loads()` instead of allocating a stripped copy for every row. A local
80k-row long-line benchmark preserved row count and reduced episode parse time
from `0.434637s` to `0.415472s` (`1.05x`).

Progress 2026-07-02: `scripts/blb_fusion_ab_compare.py` now computes each
per-window stats row with a single chunk scan and running counters instead of
building temporary value and boolean lists for every metric. A local 200k-row,
400-window benchmark preserved identical stats and reduced window aggregation
from `0.152045s` to `0.142315s` (`1.07x`), with traced peak allocation dropping
from `381,192B` to `359,168B`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` and
`scripts/run_fusion_count_action_eval_rlpath.py` now keep the first unique
fusion-count action config by reference during deduplication instead of
eagerly shallow-copying every input config via `dict(cfg)` and then copying
the unique values into a list. The returned `dict_values` view still supports
`len()` and ordered iteration for the launch loops. Local 80k-config duplicate
benchmarks preserved canonical selection; standalone Paean action eval
deduplication improved from `1.340131s` to `0.819613s` (`1.64x`, peak
`15,592B` to `6,888B`), while RL-path deduplication improved from `19.053310s`
to `18.179581s` (`1.05x`, peak `108,156B` to `50,443B`).

- [x] **Step 2: Do not change core RL during concurrent edits**

Until the Stage-2 RL agent handoff is clear, restrict work to tools and gates.

- [ ] **Step 3: Server A/B before defaults**

Use `SERVER_COMMAND.md` to run 1GPU vs NGPU parity and speed checks. Promote a
new default only when effect equality passes and wall-clock evidence improves.

## Task 5: Rescale Optimizer and Fusion Map Runtime

**Files:**

- Modify: `Rescale_optimizer/rescale_optimizer/replan_interface.py`
- Modify: `Rescale_optimizer/rescale_optimizer/replan.py`
- Modify: `scripts/blb_build_fusion_count_map.py`
- Modify: `scripts/blb_f0_scan_feasible_domain.py`
- Modify: `scripts/report_fusion_count_map.py`
- Test: `tests/test_rescale_optimizer_bridge_cache.py`
- Test: `tests/test_blb_fusion_count_map.py`

- [ ] **Step 1: Profile before editing**

Use existing fusion-map build logs and local unit tests to identify whether
time is in graph loading, feasibility DAG build, replan calls, or summary
parsing.

- [ ] **Step 2: Apply safe reuse**

Allowed changes:

- Cache loaded profile graph data inside `ReplanSession`.
- Cache feasibility DAGs keyed by graph/config hash.
- Stream map summaries without loading sidecars as maps.

Progress 2026-07-02: `ReplanSession.from_profile()` now passes the static
skeleton baselines it already loaded into `ReplanSession.__init__()`, avoiding a
second read/parse of the same `static_skeletons_<profile>.json` archive during
session construction. This reduces repeated worker/session setup overhead in
Rescale/fusion-map paths without changing replan semantics.

Progress 2026-07-03: `rescale_optimizer_bridge.load_baseline_archive()` now
caches parsed static-skeleton archives by absolute path, mtime, and size. The
cache stores immutable tuples and returns fresh lists to callers, so repeated
fallback loads avoid JSON parsing without exposing shared mutable state.

Server evidence 2026-07-03: source commit `dab3b8b` has red/green verification
under
`experiments/server_command_runs/baseline_archive_cache_dab3b8b_20260703_212500/`.
The green gate compiled `rescale_optimizer_bridge.py` and passed the targeted
archive-cache unittest.

Progress 2026-07-03: `blb_stage2_rl/skeleton_stage_map.py`
`load_profile_configs()` now discovers profile config JSON files with
`os.scandir()`, filters `static_skeletons*.json`, and skips non-file entries
before JSON parsing. This avoids parsing `.json`-named directories and reduces
unnecessary filesystem wrapper allocation while preserving deterministic
filename ordering.

Server evidence 2026-07-03: source commit `cb215bd` has red/green verification
under
`experiments/server_command_runs/skeleton_profile_config_discovery_cb215bd_20260703_213500/`.
The red run failed with `IsADirectoryError` on a `.json` directory; the green
run passed `py_compile` and the targeted discovery unittest.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now filters fusion
map candidates by block-map filename before opening JSON files, so post-build
sidecars such as `map_summary.json` are not parsed as maps. This keeps fusion
map reporting focused on real `block*.json` maps and avoids unnecessary
sidecar reads after large server builds.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now discovers map
files with `os.scandir()` and filename filtering before constructing `Path`
objects, instead of `Path.glob("*.json")` followed by sidecar filtering. A
regression test patches `Path.glob()` out of `_load_maps()`. A local directory
benchmark with 3000 map files plus 3000 sidecar/non-map files preserved map
ordering and improved discovery from `0.059813s` / `3.15MB` to `0.027619s` /
`0.86MB` (`2.17x`).

Progress 2026-07-03: `scripts/render_fusion_count_slots_eval_report.py` now
uses the same canonical block-map filename filter and `os.scandir()` discovery
before loading fusion maps for detailed slot-eval reports. The report keeps
deterministic map ordering while skipping post-build sidecars such as
`map_summary.json` before JSON parsing, and a regression test patches
`Path.glob()` out of the loader.

Progress 2026-07-04: `scripts/report_fusion_count_map.py` now exposes the
canonical fusion-map iterator as `iter_fusion_map_paths()`, and the active
`SERVER_COMMAND.md` phase2 map gate imports that iterator instead of using
`glob("*.json")` plus ad hoc `_summary.json` filtering. This closes the
remaining post-build parser/audit gap where sidecars such as `map_summary.json`
could be opened as maps and fail on missing `options`.

Server evidence 2026-07-04: source commit `248a0ec` has red/green verification
under
`experiments/server_command_runs/fusion_map_gate_filter_248a0ec_20260704_094103/`.
The red package failed the two new regression tests against the old source
(`iter_fusion_map_paths` missing, and `SERVER_COMMAND.md` still using the old
glob path). The green package passed those tests, `py_compile`, and the full
`tests.test_report_fusion_count_map` suite (`21` tests).

Progress 2026-07-04: `scripts/report_fusion_count_map.py` now streams its CLI
stdout JSON summary with `json.dump(..., sys.stdout)` plus a trailing newline
instead of materializing the summary through `json.dumps()` before `print()`.
The report JSON artifact still uses `write_json_file()`, and generated HTML /
action-config semantics are unchanged.

Server evidence 2026-07-04: source commit `0980322` has focused RED/GREEN
verification under
`experiments/server_command_runs/fusion_report_stdout_json_0980322_20260704_104732/`.
The RED test failed on the old `print(json.dumps(...))` stdout path. The GREEN
gate passed `py_compile` and the full `tests.test_report_fusion_count_map`
suite (`22` tests).

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now precomputes the
static all-max baseline action, layer width, and block offsets once per
action-config report instead of recomputing them for every generated group. A
local 12-layer / 5000-group splice benchmark preserved generated actions and
reduced action splice time from `1.613343s` / `15.57MiB` peak to `0.853562s` /
`14.08MiB`.

Progress 2026-07-04: `blb_stage2_rl/action_space.py` now splices per-step and
fusion-step action values by iterating the checked numpy array directly instead
of materializing `arr.tolist()` before every offset write. This keeps shape and
offset validation unchanged while removing a short Python-list allocation from
high-frequency Stage-2/Paean action construction.

Server evidence 2026-07-04: source commit `2ee6de2` has red/green verification
under
`experiments/server_command_runs/action_space_splice_no_tolist_2ee6de2_20260704_013204/`.
The red source guard failed on the old `arr.tolist()` implementation. The
focused green gate passed `py_compile`, the source guard, and an artifact-free
functional splice test for both step and fusion helpers. One broader green
attempt compiled and passed the functional splice check but could not run a
fusion-map fixture test because the temp source package intentionally did not
include canonical fusion-map JSON artifacts.

Progress 2026-07-04: `blb_stage2_rl/action_mask.py` now normalizes
ndarray-backed GELU/attention degree vectors in `_degree_vector()` through a
direct numpy reshape/length check instead of first copying them with
`list(raw)`. Non-ndarray iterables keep the existing list-materialization path,
so CLI/list compatibility is unchanged.

Server evidence 2026-07-04: source commit `522b42f` has red/green verification
under
`experiments/server_command_runs/action_mask_degree_vector_522b42f_20260704_023140/`.
The valid red test failed on `_degree_vector()` calling `list(raw)` for ndarray
input. The green gate passed `py_compile`, the ndarray regression test, the
existing minimal import-shim action-mask roundtrip test, and a source guard
confirming the ndarray branch does not call `list(raw)`.

Progress 2026-07-02: `scripts/blb_f0_scan_feasible_domain.py` now lazily
imports its torch/optimizer-heavy execution dependencies and uses
`heapq.nsmallest()` for the masked-random and multi-random best-cost summaries
instead of sorting every valid sampled row. On a 300k-row local top-20
benchmark, the summary selection stayed identical while median time dropped
from `0.280s` to `0.069s` and traced peak memory dropped from `22.76MB` to
near zero.

Progress 2026-07-02: `scripts/blb_f0_scan_feasible_domain.py` now builds the
per-slot summary rows with one pass over `per_slot_scan` rows instead of
rescanning all rows once per slot. A local 16k-row / 800-slot synthetic
benchmark preserved the summary output and reduced this step from `1.8017s` to
`0.0195s`.

Progress 2026-07-02: `scripts/blb_build_fusion_count_map.py` now consumes golden
enumeration shard results with `Pool.imap_unordered()` and merges the result
stream directly, instead of `Pool.map()` batching every shard result before
reduction. A synthetic 80-shard/120k-row local benchmark kept identical merged
counts while reducing traced peak memory from `69.60MB` to `59.74MB` and median
merge wall time from `0.718s` to `0.644s`; the real benefit is avoiding a second
parent-process copy of large golden fallback shard payloads.

Progress 2026-07-02: `blb_stage2_rl/fusion_enum.py` golden
`enumerate_shard()` now decodes only the product ranks assigned to the shard,
instead of iterating the entire Cartesian product in every worker and skipping
`num_shards - 1` out of every `num_shards` combos. The shard partition is
unchanged (`rank % num_shards == shard_idx`) and a 1,000,000-combo / 64-shard
local iterator benchmark produced identical assigned combos while reducing one
worker's pure iteration time from `0.0646s` to `0.0253s` (`2.55x`).

Progress 2026-07-02: `blb_stage2_rl/fusion_enum.py` degeneracy probes now reuse
the already-computed all-min corner result and stream random probes, avoiding one
duplicate real replan call and a temporary probe list on every over-budget
degeneracy check. A local call-count check with `num_random=5` now performs the
expected `7` evals (`baseline + corner + 5 random`) instead of the previous `8`.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now selects the best
option for a requested fusion count with targeted streaming scans instead of
building per-count candidate buckets and sorting the selected bucket. A local
250k-option benchmark preserved the selected option for exact, clamped, and
max targets; exact target selection improved from `0.5034s` / `6.17MB` to
`0.3138s` / near-zero traced allocations.

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now detects
already-ordered fusion map option lists and reuses the list directly instead of
unconditionally sorting by `option_id` while building the HTML/JSON report.
Unordered inputs still fall back to the old sort semantics. A local
300k-option ordered-list benchmark preserved rows and reduced extra allocation
from `4.58MB` to near zero with comparable wall time (`0.0327s` -> `0.0325s`).

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now caches
`_choose_option()` results inside `_group_specs()` by `(graph_key, target)`.
The report still emits the same fixed-action groups, but repeated global,
one-hot, combined, and partial-block4 specs no longer rescan the same graph
options for `0`, `1`, and `max` targets. A local seven-graph / 25k-options-per
graph benchmark preserved the exact group specs and reduced `_group_specs()`
from `0.463743s` to `0.103732s` (`4.47x`).

Progress 2026-07-02: `scripts/report_fusion_count_map.py` now decodes each
graph's fusion0/base option action indices and real slots once while building
the report payload, then reuses that baseline for every option summary. A local
8000-option / 24-slot payload benchmark preserved the exact graph payload and
reduced payload construction from `0.536779s` to `0.498236s` (`1.08x`).

Progress 2026-07-04: `blb_stage2_rl/fusion_enum.py`
`check_k_independence()` now counts sample configs while scanning them instead
of calling `len(list(sample_configs))` after the scan. This removes a second
materialization pass from the fusion-map build audit path and preserves correct
counts for streamed sample config iterators.

Server evidence 2026-07-04: source commit `c15cb03` has red/green verification
under
`experiments/server_command_runs/fusion_k_independence_count_c15cb03_20260704_022055/`.
The valid red test failed with `samples_checked=0` for a streamed generator.
The green gate passed `py_compile`, `CheckKIndependenceTest`,
`GroupMinNoiseOptionsTest`, and a source guard confirming the function counts
during iteration and no longer contains `len(list(sample_configs))`.

Progress 2026-07-02: `scripts/blb_verify_boosted_install.py` now lazily imports
the torch/rescale install-path dependencies only after it finds a non-skipped map
with boosted fusion options, and the map loop passes the already-loaded JSON
payload into `verify_map()`. This removes one duplicate full JSON parse per
verified fusion map and lets degenerate/no-boost maps skip without loading torch.
A local 7-map synthetic JSON benchmark for the eliminated parse path preserved
the checked option count and reduced median time from `6.84s` to `3.44s`, with
traced peak memory down from `60.37MB` to `42.10MB`.

Progress 2026-07-02: `scripts/blb_verify_boosted_install.py` now discovers
fusion maps with `os.scandir()` and canonical block-map filename filtering
before opening JSON payloads. Post-build sidecars such as `map_summary.json`
and hidden summaries are no longer read or passed to the boosted-install
verifier. A local 7-map / 3000-sidecar benchmark reduced the discover+load
phase from `0.159301s` / `3.18MB` to `0.015133s` / `0.01MB` (`10.53x`, with
candidate payloads dropping from `3006` to `6` after degenerate skips).

Progress 2026-07-03: `scripts/blb_verify_boosted_install.py` now skips
known degenerate/dormant fusion maps (`block1_<profile>.json` and
`block5_n0.json`) by filename before opening JSON. The boosted-install gate
still verifies all non-degenerate maps, but avoids parsing maps that cannot
contain boosted fusion options; a regression test guards that corrupt skipped
maps are not read.

Progress 2026-07-02: `scripts/blb_orphan_slot_audit.py` now caches parsed ASTs
for `function_handler.py` and `rescale_optimizer_bridge.py` across all block
loaders in one audit process. The static slot/cfg/t_new extraction output is
unchanged, but the audit no longer rereads and reparses the same bridge source
for every block. A real-source local benchmark over blocks 1..5 reduced the
static extraction phase from `0.471s` to `0.231s` (`2.04x`).

Progress 2026-07-02: `scripts/blb_orphan_slot_audit.py` now discovers
Rescale graph JSON files with one cached `os.scandir()` result per config
directory, then filters that sorted filename tuple for each block. This avoids
five repeated directory scans in one audit run and keeps graph discovery
independent of `Path.glob()`. A local 603-graph / 3000-sidecar benchmark
preserved graph ordering while improving discovery from `0.026675s` /
`1.95MB` to `0.014171s` / `0.41MB` (`1.88x`, `4.76x` lower peak memory).

Progress 2026-07-02: `scripts/blb_f0_scan_feasible_domain.py` now sorts each
slot's allowed action indices once while building the suggested action mask and
reuses that list for both `allowed_indices` and `allowed_values`. This removes
one duplicate filter/sort pass per slot in F0 feasible-domain scans. A local
20k-slot synthetic mask benchmark preserved the full mask payload and reduced
runtime from `9.584897s` to `8.428960s` (`1.14x`) over 80 builds, with traced
peak memory unchanged at `13.36MiB`.

Progress 2026-07-02: `scripts/blb_apply_precision_boost.py` no longer
materializes an unused pre-boost snapshot of every fusion-map option before
calling the deterministic precision-boost pass. A 300k-option synthetic
benchmark preserved the effective option count while eliminating `0.089972s`
of tuple-copy time and `25.22MB` of traced peak allocation.

Progress 2026-07-02: `Rescale_optimizer/scripts/batch_run_configs.py` and
`Rescale_optimizer/scripts/check_compress_headroom.py` now discover config JSON
files with `os.scandir()` and filename filtering instead of materializing
`Path.glob("*.json")` results. The new path preserves sorted real-file config
discovery and skips `.json` directories before optimizer work starts. A local
4000-config / 4000-sidecar / 200-json-directory benchmark reduced batch
discovery from `0.052645s` / `4.31MB` to `0.031833s` / `1.18MB`, and headroom
discovery from `0.071282s` / `5.20MB` to `0.046929s` / `2.41MB`.

Progress 2026-07-02: `Rescale_optimizer/scripts/update_noise_tables_from_csv.py`
now uses the same `os.scandir()` real-file config discovery for noise-table
maintenance runs, avoiding full `Path.glob("*.json")` materialization and
skipping `.json` directories before any file rewrite attempt. A local
4000-config / 4000-sidecar / 200-json-directory benchmark preserved real config
ordering and reduced discovery from `0.073012s` / `5.20MB` to `0.030925s` /
`1.18MB`.

Progress 2026-07-02: `Rescale_optimizer/scripts/update_noise_tables_from_csv.py`
now parses measured-noise CSV rows with `csv.reader` and precomputed header
indices instead of `csv.DictReader`, avoiding one dictionary allocation per
measured `(N, scale_bits)` row. A local 200k-row synthetic noise CSV benchmark
preserved the loaded table and reduced parsing from `0.996201s` / `0.07MB` peak
to `0.723280s` / `0.06MB` (`1.38x`).

Progress 2026-07-02: `blb_stage2_rl/fusion_count_map.py` runtime
`FusionCountMap.load()` now discovers canonical block-map JSON files with
`os.scandir()` and filename filtering before parsing payloads. Runtime map
loading no longer opens sidecars such as `map_summary.json`, and `.json`
directories are ignored before any read attempt. A local 1000-map /
3000-sidecar benchmark preserved loaded graph count and reduced map loading
from `0.208784s` / `3.57MB` to `0.068689s` / `2.54MB` (`3.04x`).

Progress 2026-07-03: `scripts/blb_build_fusion_count_map.py` now reduces
golden shard results with the same `_MinNoiseReducer` used inside shard workers
while streaming parent-process merge input. The builder still counts every valid
config, but dominated cross-shard candidates are discarded before the final
grouping pass, reducing parent memory and follow-up work during golden fallback
builds.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_rescale_optimizer_bridge_cache tests.test_blb_fusion_count_map -v
```

Use server only for large fusion-map wall-clock evidence.

## Task 6: Paean Final Evaluation Throughput

**Files:**

- Modify: `Paean/run_final_eval.py`
- Modify: `Paean/action_grid.py`
- Modify: `Paean/blb_action_eval.py`
- Modify: `final_evaluation_module.py`
- Test: `tests/test_final_eval_layout.py`
- Test: `tests/test_blb_final_eval_feasibility.py`

- [x] **Step 1: Add final-eval plan diagnostics**

Expose how many configs, repeats, random controls, and expected model loads a
Paean run will perform before launch.

- [ ] **Step 2: Optimize shared work**

Allowed changes:

- Group action-grid configs by shared Stage-1 install.
- Reuse model/tokenizer initialization inside one final-eval process.
- Schedule independent configs across visible GPUs only after local tests and
  server smoke show no metric drift.

Progress 2026-07-02: `UnifiedFinalEvaluationModule` now caches the loaded
final-eval JSON config map per `config_path` inside one module instance. This
prevents Stage-1 and Stage-2 JSON resolution from opening/parsing the same
config file twice in a single final-eval flow while preserving config-source
semantics.

Progress 2026-07-03: BLB action final-eval repeat handling now reuses one
clean-baseline Stage-1 install and one BLB bridge install per candidate when
`repeat_n > 1`, while still running the validation forward pass once per
repeat. This removes repeated configuration/noise installation work from
repeat measurements without adding an evaluation-result cache.

Server evidence 2026-07-03: source commit `567ad75` has red/green verification
under
`experiments/server_command_runs/final_eval_repeat_install_reuse_567ad75_20260703_203900/`.
The evidence validates repeated-install reuse for both clean-baseline and BLB
action repeat paths without caching the model-forward result.

Progress 2026-07-03: `BLBActionFinalEvaluationModule` now caches
`load_max_sfs(profile)` results per module instance. Candidate decoding,
cost-matched sampling, and GLUE action export reuse the same max-SF table for
the same profile instead of reopening/parsing it at each call site.

Server evidence 2026-07-03: source commit `b2a7325` has red/green verification
under
`experiments/server_command_runs/final_eval_max_sfs_cache_b2a7325_20260703_205000/`.
The green gate confirms one module instance reuses its profile max-SF table.

Progress 2026-07-02: `UnifiedFinalEvaluationModule` now caches the Stage-2
total-cost count-solution maps used by random final-eval groups. Repeated
Stage2Budget/Budget sampling in one final-eval run reuses the same per-noise
domain enumerations instead of rebuilding all seven maps for every sampled
config.

Progress 2026-07-02: Stage-2 Equiv random final-eval sampling now uses the
same cached exact count-solution maps instead of running iterative random
cost-matching searches for every noise type and trial. It still falls back to
the old matcher if an exact target key is unavailable.

Progress 2026-07-02: Stage2Budget/Budget random final-eval sampling now caches
the Stage-2 total-cost combination plan for a reused solution-map set. Repeated
samples no longer rebuild the same suffix reachability table or rescan cost
keys for every noise domain.

Progress 2026-07-02: Stage1Budget/Budget random final-eval sampling now caches
the feasible GELU/Softmax total-cost pair list for a reused solution-map pair
and target. Repeated Stage-1 budget samples no longer rescan the GELU cost-key
domain for every random control.

Progress 2026-07-02: final-eval random comparison generation now builds
Stage-1 GELU/Softmax solution maps lazily. Stage2Budget-only runs and other
paths that do not need Stage-1 budget/equivalence controls skip the Stage-1
cost-solution enumeration entirely.

Progress 2026-07-02: `UnifiedFinalEvaluationModule` Stage2Budget/Budget random
sampling now caches feasible Stage-2 count-combo keys by `(noise-domain index,
remaining target cost)` inside the reused combo plan. Repeated random controls
with the same target no longer rescan every cost key for every domain on every
trial. A local 5000-sample synthetic benchmark kept valid samples while reducing
key scans from `25000` to `478` and sampler wall time from `0.168s` to
`0.047s`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` now folds
duplicate/no-op fusion-count final-eval groups with a shallow top-level
candidate copy instead of `copy.deepcopy()` for every alias. The combined JSON
still carries the same nested candidate diagnostics, but repeated groups no
longer clone large result payloads. A local 1200-alias synthetic combined-report
benchmark reduced merge cost from `2.0259s` / `68.41MB` to `0.0048s` /
`0.86MB`.

Progress 2026-07-02: `Paean/final_eval_layout.py` now precomputes quantized
GELU choice costs for Stage-1 same-cost peer sampling and checks sampled index
costs before materializing a GELU vector. The accepted peer sequence is
unchanged for the same seed. A local same-process benchmark over 12-layer
sampler cases kept outputs identical and improved runtime by `1.48x` to
`2.26x`; the common all-degree-1 case reduced traced peak memory from
`0.782MB` to `0.024MB`.

Progress 2026-07-02: the same sampler now short-circuits unique min/max GELU
cost extremes, where the selected vector is the only possible same-cost vector
and peers must be empty. This avoids burning `max_attempts` on impossible
final-eval controls. Same-process benchmarks kept outputs identical and reduced
all-degree-4 from `0.2044s` to `0.00014s` and all-degree-0 from `0.1829s` to
`0.00011s`.

Progress 2026-07-02: `Paean/run_final_eval.py --list-presets` now reads only
the first line of each preset with `readline()` instead of materializing the
whole preset through `read_text().splitlines()[0]`. A local 200k-line synthetic
preset benchmark preserved the displayed summary while reducing the lookup
from `0.0400s` / `14.92MB` to `0.0002s` / `0.02MB`.

Progress 2026-07-02: `Paean/config.py` now lists final-eval preset names with
a single `os.scandir()` pass instead of `Path.glob("*.conf")` plus per-entry
`Path` objects. The CLI behavior is unchanged, but startup/list-presets paths
avoid extra filesystem wrapper allocation. A local 4000-preset /
4000-nonpreset benchmark preserved sorted preset names while improving
discovery from `0.058477s` / `2.48MB` to `0.010787s` / `0.29MB` (`5.42x`,
`8.41x` lower peak memory).

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` now drops the
full parsed action-config payload after deriving the fields it actually uses
(`name`, `path`, `group`, and `action_hash`). Final-eval launch/report behavior
is unchanged, but large `action_vec`/`slots` diagnostics no longer stay resident
for the whole combined-report flow. A local 120-config synthetic benchmark kept
the same action hashes and reduced retained loader memory from `16.90MB` to
`0.17MB` (`101.86x`), with comparable wall time (`0.3065s` -> `0.3008s`).

Progress 2026-07-02: `scripts/run_fusion_count_action_eval_rlpath.py` now keeps
its action-config records dependency-light and payload-light: module import no
longer pulls torch/HF/RL dependencies, and `_load_action_configs()` retains only
the group metadata, config path/name, and `baseline_k_index` needed by
deduplication and `_run_group()`. A local 120-config synthetic benchmark with
large unused payload fields preserved config count while reducing retained
current memory from `254.69MB` to `0.65MB` and traced peak memory from
`255.42MB` to `5.60MB`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval.py` and
`scripts/run_fusion_count_action_eval_rlpath.py` now discover action config
JSON files with `os.scandir()` and pre-filter `_`/`._` sidecars before building
`Path` objects, instead of using `Path.glob("*.json")` and then filtering.
Regression tests patch `Path.glob()` out of both loaders. A local directory
benchmark with 3000 valid configs plus 2000 sidecar/non-JSON files preserved
config ordering and improved discovery from `0.036433s` / `2.17MB` to
`0.020808s` / `0.86MB` (`1.75x`).

Progress 2026-07-02: `scripts/run_fusion_count_action_eval_rlpath.py` now
computes each action config's canonical group key once at load time and reuses
it during deduplication, result indexing, and requested-group result backfill.
This removes repeated JSON key serialization from the RL-path fusion-count
comparison driver. A local 80k-config duplicate benchmark covering those three
keyed phases preserved canonical selection and reduced runtime from
`25.343264s` to `1.572615s` (`16.12x`), with traced peak allocation dropping
from `0.78MiB` to `0.70MiB`.

Progress 2026-07-02: `Paean/action_grid.py` now precomputes the truncation
`K_LEVELS` value-to-action-index map and the invalid-value choices string at
module import. Action-range and slot decoding no longer allocates
`list(K_LEVELS)` for every K value conversion. A local same-signature 1M-call
benchmark preserved indexes and reduced K lookup time from `0.293003s` to
`0.213070s` (`1.38x`), with traced peak allocation dropping from `180B` to
`156B` over 10k calls.

Progress 2026-07-02: `Paean/action_grid.py` now caches non-K scaling-factor
value-to-action-index tables by `(kind, max_sf)`. Repeated action-range and
slot decoding no longer rebuilds the same SF choices list or scans it twice for
membership plus index lookup. A local same-signature 1M-call benchmark
preserved indexes and reduced lookup time from `1.802358s` to `0.447846s`
(`4.02x`), with traced peak allocation dropping from `404B` to `124B` over
10k calls.

Progress 2026-07-02: `Paean/action_grid.py` now caches selector-to-slot
resolution by `(num_layers, selector)` during action-grid expansion. Repeated
final-eval action-range candidates no longer rescan every layer's action-space
fields for the same selector before setting values. A local 200k-call repeated
selector benchmark preserved vector writes and reduced setter time from
`4.215663s` to `1.279825s` (`3.29x`), with traced peak allocation dropping
from `1,576B` to `124B` over 10k calls.

Progress 2026-07-03: the current source now has these Paean action-grid cache
guards green again: K lookup uses a precomputed value map, non-K scaling-factor
choice tables are reused across repeated values, and selector-slot resolution
is cached across repeated `_set_selector_value()` calls. This keeps repeated
final-eval action config expansion from rebuilding the same small lookup
structures.

Progress 2026-07-03: `Paean/action_grid.py` cost-matched random final-eval
sampling now parses `fixed_specs` once before the sampling loop and reuses the
selector/value pairs for every random attempt. This preserves fixed override
application order and cost-match filtering semantics while avoiding up to
`max_attempts * len(fixed_specs)` duplicate CLI-spec parses in same-cost peer
generation.

Progress 2026-07-04: `Paean/action_grid.py` cost-matched random final-eval
sampling now normalizes GELU/Softmax degree arrays once before the reject-sample
loop and reuses those arrays for every `action_vector_to_cfgs()` decode attempt.
The loop also reuses integer target totals for sum-K, total bits, and fusion
count comparisons. This preserves candidate sampling and optimizer semantics
while removing repeated array construction and target casts from accepted
prefilter attempts.

Server evidence 2026-07-04: source commit `94f1aad` has red/green verification
under
`experiments/server_command_runs/paean_cost_match_degree_arrays_94f1aad_20260704_013930/`.
The red test showed repeated decode attempts received different degree array
objects. The green gate passed `py_compile`, all six `tests.test_paean_action_grid`
tests, and a source guard confirming the `np.asarray()` calls moved out of the
decode loop.

Progress 2026-07-04: `Paean/action_grid.py` now normalizes non-string
`base_action_vec` inputs with `np.asarray(base_action_vec, dtype=int)` directly
instead of first materializing `list(base_action_vec)`. This preserves the
existing validation and copy-on-return behavior while avoiding one full Python
list copy when final-eval callers already hold an ndarray-backed action vector.

Server evidence 2026-07-04: source commit `85c03b9` has red/green verification
under
`experiments/server_command_runs/paean_base_action_ndarray_85c03b9_20260704_014440/`.
The red test failed on `list(base_action_vec)` for ndarray input. The green gate
passed `py_compile`, all seven `tests.test_paean_action_grid` tests, and a
source guard confirming the list-copy path was removed.

Progress 2026-07-04: `Paean/action_grid.py` now parses legacy list-form
`action_vec` / `base_action_vec` config payloads with `np.asarray(base_raw,
dtype=int)` directly instead of first copying the complete list through
`list(base_raw)`. This preserves legacy action-config behavior while removing
one full-vector Python-list copy during final-eval action-config loading.

Server evidence 2026-07-04: source commit `a600b79` has red/green verification
under
`experiments/server_command_runs/paean_parse_action_vec_a600b79_20260704_014720/`.
The red test failed on the legacy list-copy path. The green gate passed
`py_compile`, all eight `tests.test_paean_action_grid` tests, and a source
guard confirming the direct `np.asarray()` parse path.

Progress 2026-07-04: `UnifiedFinalEvaluationModule._summarize_random_results()`
now streams final-eval random-result summaries through per-family and overall
running counters/stats. It no longer builds separate `feasible`, win-rate,
dominance, metric, delta, total-cost, stage-cost, and variance lists before
calling `np.mean()` / `np.std()`. Summary keys and population-std semantics are
preserved while long random-search comparison reports avoid repeated list
materialization over the same result rows.

Server evidence 2026-07-04: source commit `8101feb` has red/green verification
under
`experiments/server_command_runs/final_summary_running_8101feb_20260704_042900/`.
The red test patched `np.mean` / `np.std` and proved the old summary path still
depended on materialized list statistics. The green gate passed `py_compile`,
all `tests.test_final_evaluation_config_cache` tests, and a source guard
confirming `_summarize_random_results()` uses `_RunningStats` without `np.mean`
or `np.std` calls.

Progress 2026-07-04: `_mean_float_or_none()` and `_std_float_or_none()` now
share `_finite_float_stats()`, streaming finite floats once and computing
population mean/std from running totals instead of building a clean list and
calling `np.mean(clean)` / `np.std(clean)`. Optional and non-finite filtering
semantics stay unchanged.

Server evidence 2026-07-04: source commit `08560c1` has red/green verification
under
`experiments/server_command_runs/final_stat_helpers_08560c1_20260704_044500/`.
The red test patched `np.mean` / `np.std` and proved the old helpers still used
numpy stats on materialized `clean` lists. The green gate passed `py_compile`,
all `tests.test_final_evaluation_config_cache` tests, and a source guard
confirming the helpers stream through `_finite_float_stats()`.

Progress 2026-07-04: `_plot_variance_results()` now computes the variance bar
chart's per-group means with the shared streaming finite-float helper instead
of building a `vals` list and calling `np.mean(vals)` for every group/metric
pair. The plotted row order, non-finite filtering, and fallback `0.0` behavior
for empty groups are preserved.

Server evidence 2026-07-04: source commit `75cce4c` has red/green verification
under
`experiments/server_command_runs/final_variance_plot_mean_75cce4c_20260704_052500/`.
The red test patched `np.mean` and proved the old variance plot aggregation
still depended on materialized `vals` means. The green gate passed
`py_compile`, all `tests.test_final_evaluation_config_cache` tests, and a
source guard confirming `_plot_variance_results()` now uses
`_mean_float_or_none(item.get(key) for item in items)`.

Progress 2026-07-04: `_plot_variance_results()` now builds variance scatter
panel `xs` and `ys` in one loop per family/panel. Each random-result row reads
`total_cost` once for that metric panel instead of scanning the same `items`
twice through separate `xs` and `ys` comprehensions. Scatter values and
non-finite variance filtering remain unchanged.

Server evidence 2026-07-04: source commit `e4c3d47` has red/green verification
under
`experiments/server_command_runs/final_variance_scatter_scan_e4c3d47_20260704_054500/`.
The red test used a guarded random-result row to prove the old scatter path
read `total_cost` more than once per panel. The green gate passed `py_compile`,
all `tests.test_final_evaluation_config_cache` tests, and a source guard
confirming the old paired list-comprehension scan is gone.

Progress 2026-07-04: `_plot_results()` now builds the main final-eval
comparison scatter panel `xs` and `ys` in one loop per family/panel. Each
random-result row reads `total_cost` once for that metric panel instead of
scanning the same `items` twice through separate `xs` and `ys` comprehensions.
The plotted comparison points and summary bar chart semantics remain unchanged.

Server evidence 2026-07-04: source commit `c85b896` has red/green verification
under
`experiments/server_command_runs/final_comparison_scatter_scan_c85b896_20260704_061000/`.
The red test used a guarded random-result row to prove the old comparison plot
path read `total_cost` more than once per panel. The green gate passed
`py_compile`, all `tests.test_final_evaluation_config_cache` tests, and a
source guard confirming the old paired list-comprehension scan is gone.

Progress 2026-07-04: `_set_numeric_axis_limits()` now streams finite values and
keeps running `lo`/`hi` instead of building a `clean` list, then calling
`min(clean)` and `max(clean)`. Each finite value is converted to float once,
which removes repeated conversion and a short-lived list allocation from every
final-eval comparison and variance plot panel.

Server evidence 2026-07-04: source commit `a1de9a3` has red/green verification
under
`experiments/server_command_runs/final_axis_limits_stream_a1de9a3_20260704_063500/`.
The red test used guarded float values to prove the old axis-limit helper
converted finite values more than once. The green gate passed `py_compile`, all
`tests.test_final_evaluation_config_cache` tests, and a source guard confirming
the `clean` list/min/max pattern is gone.

Progress 2026-07-04: `BLBActionFinalEvaluationModule._summarize_selected_vs_random()`
now scans final-eval random comparison rows once and accumulates loss/metric
means, standard deviations, min/max values, and selected-anchor ranks in the
same pass. This removes six per-field numpy-array materializations plus the
separate metric/loss rank list builds from the BLB action final-eval report
summary path.

Server evidence 2026-07-04: source commit `54d7bf9` has focused red/green
verification under
`experiments/server_command_runs/paean_selected_random_summary_54d7bf9_20260704_062553/`.
The RED target test failed on the old `np.asarray([float(...) ...])` and rank
list patterns. The GREEN gate passed `py_compile` and the two focused
selected-vs-random summary tests, including statistic/rank semantic parity.

Progress 2026-07-04: `BLBActionFinalEvaluationModule._save_results_plot()`
now scans BLB action final-eval `candidate_results` once to collect labels,
loss, metric, cost, and timing columns before converting those columns to numpy
arrays for matplotlib. This removes six separate candidate-result list
comprehensions from the plot-generation path while preserving the plotted
series and output path.

Server evidence 2026-07-04: source commit `af9884a` has focused red/green
verification under
`experiments/server_command_runs/paean_results_plot_scan_af9884a_20260704_063156/`.
The RED source guard failed on the old per-column `np.asarray([... for r in
candidate_results])` patterns. The GREEN gate passed `py_compile` and
`test_results_plot_scans_candidate_rows_once`.

Progress 2026-07-04: `BLBActionFinalEvaluationModule._save_scatter_plot()`
now scans selected and random final-eval result groups once each to collect
primary and secondary metric scatter columns. This removes the `_xs_ys()` helper
list comprehensions and the second-panel `selected_results` / `random_results`
list comprehensions while preserving the plotted points and output path.

Server evidence 2026-07-04: source commit `32ea5f5` has focused red/green
verification under
`experiments/server_command_runs/paean_scatter_plot_scan_32ea5f5_20260704_063613/`.
The RED source guard failed on the old `_xs_ys` and secondary-metric list
comprehension patterns. The GREEN gate passed `py_compile` and
`test_scatter_plot_scans_result_rows_once_per_group`.

Progress 2026-07-04: `BLBActionFinalEvaluationModule._full_noise_config_markdown_table()`
now iterates the full-noise `entries` sequence directly instead of first copying
it through `list()`. This trims per-candidate Markdown detail rendering memory
when final-eval reports include large full noise/truncation tables.

Server evidence 2026-07-04: source commit `3f020ef` has focused red/green
verification under
`experiments/server_command_runs/paean_full_noise_table_stream_3f020ef_20260704_064005/`.
The RED source guard failed on the old `entries = list(...)` materialization.
The GREEN gate passed `py_compile` and
`test_full_noise_markdown_table_streams_entries_without_copy`.

Progress 2026-07-04: `UnifiedFinalEvaluationModule.run()` now passes the
baseline, optimized, Stage1Fixed+MaxSF, and random final-eval result rows to
`_attach_relative_metrics()` through `itertools.chain()` instead of building an
`all_results` list with `+ list(random_results)`. This preserves the same
relative metric mutations and downstream random-result ordering while removing
one list copy proportional to the number of random final-eval candidates.

Server evidence 2026-07-04: source commit `9026d8f` has focused red/green
verification under
`experiments/server_command_runs/final_eval_relative_chain_9026d8f_20260704_071220/`.
The RED source guard failed on the old `+ list(random_results)` concatenation.
The GREEN gate passed `py_compile` and
`test_relative_metric_attach_does_not_copy_random_results`.

Progress 2026-07-04: `BLBActionFinalEvaluationModule._decode_fusion_count_fixed_action()`
now avoids short-lived copy wrappers while replaying fusion fixed-action
configs: option metadata mappings are iterated directly, per-step block slices
use `np.take(base_arr, block_offsets)` instead of `base_arr[list(...)]`, and
selected option fields are iterated directly instead of copying through
`dict(option_fields).items()`. The normalized option maps and selected-K
restoration path remain unchanged.

Server evidence 2026-07-04: source commit `5fe7760` has focused red/green
verification under
`experiments/server_command_runs/paean_fusion_decode_copy_5fe7760_20260704_072520/`.
The RED source guard failed on the old `base_arr[list(block_offsets)]` pattern.
The GREEN gate passed `py_compile` and
`test_fusion_fixed_action_decode_avoids_step_copy_wrappers`. A separate
pre-change run showed the existing boosted replay functional test already
fails on clean `62cae98` (`13 != 14`), so it is recorded as baseline evidence
and not used as this optimization's gate.

Progress 2026-07-04: `UnifiedFinalEvaluationModule._plot_results()` and
`_plot_variance_results()` now iterate the first three matplotlib axes through
`itertools.islice(axes.flat, 3)` instead of materializing `list(axes.flat)[:3]`.
The comparison and variance report panel order is unchanged, but PNG generation
no longer allocates a temporary axes list just to take the first three panels.

Server evidence 2026-07-04: source commit `7a7e9d4` has focused red/green
verification under
`experiments/server_command_runs/final_eval_axes_islice_7a7e9d4_20260704_073330/`.
The RED source guard failed on the old `list(axes.flat)[:3]` pattern. The GREEN
gate passed `py_compile` and
`test_final_eval_plots_iterate_axes_without_flat_list_copy`.

Progress 2026-07-04: `UnifiedFinalEvaluationModule._plot_results()` now
collects the summary bar chart family labels, feasible rates, and dominance
rates in one pass over `summary["by_family"].items()` instead of taking keys and
then rescanning the mapping with two list comprehensions. Bar order and plotted
values are unchanged.

Server evidence 2026-07-04: source commit `22eb07e` has focused red/green
verification under
`experiments/server_command_runs/final_eval_summary_bar_22eb07e_20260704_074230/`.
The RED source guard failed on the old feasible-rate list comprehension. The
GREEN gate passed `py_compile` and
`test_final_eval_summary_bar_chart_collects_series_once`.

Progress 2026-07-04: `_ordered_families()` now reuses a module-level
`_FAMILY_COLOR_ORDER` tuple when ordering final-eval plot families, rather than
rebuilding the color mapping and copying `keys()` through `list()` for every
metric panel. `_family_colors()` still returns a fresh dictionary, preserving
caller isolation.

Server evidence 2026-07-04: source commit `aed348f` has focused red/green
verification under
`experiments/server_command_runs/final_eval_family_order_aed348f_20260704_071234/`.
The RED source guard failed on the old `self._family_colors().keys()` pattern.
The GREEN gate passed `py_compile` and
`test_ordered_families_reuses_static_preferred_order`.

Progress 2026-07-04: `_plot_results()` and `_plot_variance_results()` now use
the module-level `_FAMILY_COLOR_MAP` directly for internal read-only final-eval
plot color lookups instead of calling `_family_colors()` and copying a fresh
dictionary for each render. The public `_family_colors()` helper remains
unchanged for external callers.

Server evidence 2026-07-04: source commit `37890c5` has focused red/green
verification under
`experiments/server_command_runs/final_eval_color_map_37890c5_20260704_071600/`.
The RED source guard failed on the old internal `self._family_colors()` call.
The GREEN gate passed `py_compile` and
`test_final_eval_plots_reuse_static_family_color_map`.

Progress 2026-07-04: `_plot_results()` now computes
`ordered_families = self._ordered_families(grouped)` once after random-result
grouping and reuses it across the three comparison metric panels. This matches
the existing `_plot_variance_results()` pattern and avoids repeating preferred
family ordering work per panel.

Server evidence 2026-07-04: source commit `bb0962e` has focused red/green
verification under
`experiments/server_command_runs/final_eval_ordered_family_cache_bb0962e_20260704_072039/`.
The RED source guard failed on the old per-panel
`for fam in self._ordered_families(grouped):` loop. The GREEN gate passed
`py_compile` and
`test_final_eval_comparison_plot_reuses_ordered_families`.

- [ ] **Step 3: Verify**

Run final-eval unit tests locally and a server repeated final-eval smoke for
the same fixed action before/after.

## Task 7: Structured Data and Report Decoupling

**Files:**

- Modify: `rl_data_points.py`
- Modify: `scripts/verify_stage2_persistent_outputs.py`
- Create: `scripts/optimization_evidence_bundle.py`
- Modify report generators under `reports/` or `tools/`
- Test: `tests/test_rl_data_points.py`
- Test: `tests/test_stage2_persistent_output_verifier.py`
- Test: `tests/test_optimization_evidence_bundle.py`

- [x] **Step 1: Protect data completeness**

Add tests that fail if required structured fields are dropped from Stage-1 or
Stage-2 mirrored data.

Progress 2026-07-02: strengthened
`scripts/verify_stage2_persistent_outputs.py` so Stage-2 persistent verification
fails when `episodes.jsonl` or `ppo_updates.jsonl` drops required reward,
metric, cost, action-summary, PPO, or timing fields. This did not modify the
dirty `rl_data_points.py` worktree file.

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now reads JSONL
diagnostics line by line instead of building a full `read_text().splitlines()`
string first, and it treats a missing/empty `nvidia_log` path as no GPU samples
instead of trying to open the current directory. This keeps long-run monitoring
memory-bounded and avoids a default-path crash during post-run summaries.

Progress 2026-07-02: the same monitor now skips blank JSONL lines with
`isspace()` instead of allocating `strip()` copies while loading episode and PPO
diagnostics. A local 80k-row long-line benchmark preserved parsed row count and
reduced `_read_jsonl()` from `0.497555s` to `0.475890s` (`1.05x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now writes
`reward_windows.csv` with one-pass rolling accumulators and monotonic queues
for mean/min/max, instead of repeatedly slicing and sorting each prefix window.
This reduces final monitor CSV rendering CPU work for 10k/60k episode runs.
Local 20k-row mixed-reward microbenchmark produced identical CSV output and
reduced wall time from `9.580s` to `0.186s` (`51.50x`).

Progress 2026-07-02: `_window()` in `scripts/stage2_first10k_monitor.py` now
reuses the sorted tail for min/max bounds and uses `math.fsum()` for the mean,
avoiding separate `statistics.mean()`, `min()`, and `max()` scans while
preserving the summary values. A local 1000-point window benchmark over 3000
calls produced identical output and reduced time from `3.776341s` to
`0.181783s` (`20.77x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now computes
full-run and recent PPO means in `build_summary()` with `math.fsum()/len`
instead of Python's slower `statistics.mean()` exact-rational path. A local
60k-value benchmark over 400 repeats produced the same mean and reduced time
from `27.821296s` to `0.499612s` (`55.69x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now aggregates
nvidia-smi GPU samples while reading the CSV instead of retaining every parsed
row and grouping in a second pass. A local 30k-row CSV comparison produced
identical GPU summaries and reduced traced peak memory from `9.15MB` to
`1.08MB` (`8.48x`).

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` final mode now loads
episode and PPO JSONL rows once and reuses those rows for the summary,
`reward_windows.csv`, and `episode_health_windows.csv`. This avoids rereading
large Stage-2 episode logs during final monitor report generation.

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now streams
`reward_windows.csv` rows directly to `csv.DictWriter.writerow()` after each
rolling-window update instead of buffering the full row list before
`writerows()`. A local 100k-row parity benchmark kept the CSV output identical
and reduced write peak memory from `141.65MB` to `0.25MB` while slightly
improving wall time from `2.460s` to `2.335s`.

Progress 2026-07-02: `scripts/stage2_first10k_monitor.py` now parses sampled
`nvidia-smi` CSV logs with `csv.reader` and header indices instead of
`csv.DictReader`, avoiding one per-row dictionary allocation in final monitor
summaries. A local 120k-row / 4-GPU CSV benchmark preserved the GPU summary and
reduced `_gpu_stats()` from `0.562882s` / `4.22MB` to `0.333455s` / `4.24MB`
(`1.69x`).

Progress 2026-07-02: `scripts/verify_stage2_persistent_outputs.py` now counts
Stage-2 detail batch files without materializing and sorting the full file list.
The verifier only needs the count for its gate output, so long runs with many
detail batches avoid unnecessary path-list allocation. A local 8000-detail-file
benchmark preserved the count and reduced detail discovery from `0.1079s` /
`3.51MB` to `0.0805s` / `0.56MB`.

Progress 2026-07-02: `scripts/verify_stage2_persistent_outputs.py` now skips
blank `episodes.jsonl` and `ppo_updates.jsonl` rows with `isspace()` instead of
allocating a stripped copy of every diagnostics line while checking required
fields. A local 80k-row long-line benchmark preserved row/failure counts and
reduced required-field scanning from `0.658321s` to `0.642286s` (`1.02x`).

Progress 2026-07-02: `scripts/verify_stage2_persistent_outputs.py` now checks
required JSONL fields with a complete-row fast path that returns `None` and only
constructs the missing-field tuple for rows that actually fail the schema gate.
This keeps the persistent-output verifier's hot loop allocation-free on healthy
60k+ episode logs. A local 200k complete-row field-check benchmark reduced the
field-scan loop from `0.173905s` to `0.118869s` (`1.46x`); full JSONL parse is
still dominated by `json.loads`, so end-to-end 100k-row verification improved
modestly from `1.253700s` to `1.236978s`.

Progress 2026-07-02: `tools/paper_figures.py` now lazily loads run artifacts
based on the requested `--figs`. For example, `--figs cost_vs_accuracy` no
longer reads `episodes.jsonl`, `ppo_updates.jsonl`, best-action JSON, baseline
JSON, first-invalid counts, or action histograms before rendering the
top-candidate scatter. A local 100k-episode synthetic run comparison reduced
run loading for that path from `0.464s`/`39.39MB` to effectively `0.000s`/
`0.00MB`.

Progress 2026-07-02: `tools/paper_figures.py` no longer rereads each
`top_candidates.jsonl` only to decide whether the cost-vs-accuracy scatter
needs a legend; it reuses a loop-local `has_points` flag. A local 100k-row
top-candidate benchmark reduced that figure's candidate-read path from
`0.9476s`/`78.58MB` to `0.4444s`/`39.28MB`.

Progress 2026-07-02: `tools/paper_figures.py` now projects JSONL rows to the
fields each figure needs while loading run data: `episodes.jsonl` keeps only
`total_reward`, `ppo_updates.jsonl` keeps the four plotted PPO metrics, and
`top_candidates.jsonl` keeps only `total_bits`/`total_reward` for the scatter.
A local 100k-row synthetic episode log with 20 unused debug fields preserved
row count and reduced traced peak memory from `368.41MB` to `25.19MB`.

Progress 2026-07-02: `tools/paper_figures.py` now streams
`top_candidates.jsonl` directly into the cost-vs-accuracy scatter's `xs`/`ys`
lists instead of first building a list of projected row dictionaries and then
splitting it into columns. A local 120k-row top-candidate benchmark preserved
the plotted points and reduced this read path from `0.8249s` / `39.78MB` to
`0.6445s` / `7.81MB`.

Progress 2026-07-02: `tools/paper_figures.py` now skips whitespace-only JSONL
lines with `isspace()` before calling `json.loads()` in both projected-row and
XY readers. Nonblank lines are still passed through unstripped, preserving the
existing low-copy parse path. A local 100k valid-row / 50k blank-row episode
benchmark preserved projected rows and reduced read time from `0.716423s` to
`0.468636s` (`1.53x`).

Progress 2026-07-04: `tools/paper_figures.py` now reuses JSON-native
`list`/`dict` payloads returned by `read_json_file()` for best action vectors,
best/baseline slot metadata, diff-vs-baseline rows, and first-invalid counts.
Malformed/non-native truthy containers keep the old compatibility path through
`list(...)` / `dict(...)`, but normal paper-figure sidecars avoid a second
full-container copy during `load_run()`.

Server evidence 2026-07-04: source commit `b6dda66` has red/green verification
under
`experiments/server_command_runs/paper_figures_payload_reuse_b6dda66_20260704_094735/`.
The red package failed the new regression test at the old
`best_action_vec=list(...)` copy. The green package passed that test,
`py_compile`, and the complete `tests.test_paper_figures` suite (`5` tests).

Progress 2026-07-04: `tools/paper_figures.py` now reuses list-backed
`RunData.episodes` directly in `fig_training_curves()` instead of converting
the already-loaded reward series through `[float(value) for ...]` before
plotting. Non-list iterables keep the old compatibility path and are still
converted to float lists.

Server evidence 2026-07-04: source commit `596c458` has red/green verification
under
`experiments/server_command_runs/paper_training_curve_reuse_596c458_20260704_095340/`.
The red package failed the new regression test at the old reward-list copy.
The green package passed that test, `py_compile`, and the complete
`tests.test_paper_figures` suite (`6` tests).

Progress 2026-07-04: `tools/paper_figures.py` now builds grouped
training-curve reward matrices by streaming each seed's first `min_len` rewards
through `itertools.islice()` and `np.fromiter()` instead of materializing one
sliced list per seed. The grouped mean/std band keeps the same numpy matrix
semantics while avoiding the `s[:min_len]` intermediate copies on long
multi-seed reward curves.

Server evidence 2026-07-04: source commit `7d66fed` has red/green verification
under
`experiments/server_command_runs/paper_group_curve_matrix_7d66fed_20260704_095833/`.
The red package failed the new regression test at the old `s[:min_len]` copy.
The green package passed that test, `py_compile`, and the complete
`tests.test_paper_figures` suite (`7` tests).

Progress 2026-07-02: `blb_stage2_rl/candidate_store.py` now skips blank
candidate JSONL rows with `isspace()` and passes nonblank rows directly to
`json.loads()` instead of allocating stripped copies while loading append-only
candidate stores. A local 100k-row synthetic `top_candidates.jsonl` benchmark
preserved record count and reduced `read_all()` from `0.720805s` to
`0.691242s` (`1.04x`).

Progress 2026-07-02: `tools/aggregate_seeds.py` now finds the latest Paean
`blb_action_final_eval_results_*.json` by streaming `os.walk()` and retaining
only the newest path, instead of recursive `glob` plus sorting a materialized
candidate list. A local 6000-candidate synthetic final-eval tree kept the same
latest result and reduced lookup peak memory from `1.28MB` to `0.02MB`, with
wall time improving from `0.0838s` to `0.0613s`.

Progress 2026-07-02: `tools/aggregate_seeds.py` now builds one filtered
legacy persistent-dir index for the seed list instead of running a persistent
tree glob once per seed. A local synthetic tree with 5000 persistent dirs and
100 queried seed tags returned the same 100 matches while reducing directory
discovery wall time from `4.5251s` to `0.0379s` and traced peak memory from
`0.11MB` to `0.05MB`.

Progress 2026-07-02: `tools/aggregate_seeds.py` now scans
`diagnostics_summary.md` for the last-50 invalid-rate line with bounded 1MiB
chunks instead of reading the whole Markdown file into memory. A local
200k-line synthetic summary preserved the parsed rate and improved the lookup
from `0.0015s` / `8.40MB` to `0.0011s` / `4.01MB`.

Progress 2026-07-02: `tools/aggregate_seeds.py` now parses the multi-seed
seed-list with `line.split()` directly instead of allocating a stripped copy of
every line before splitting. A regression test guards that the parser does not
call `strip()` per line. A local 200k-line synthetic seed-list benchmark
preserved parsed `(seed, run_tag)` rows and improved parsing from `0.284336s`
to `0.239681s` (`1.19x`).

Progress 2026-07-02: `tools/aggregate_seeds.py` now formats aggregate
`mean ± std` values with `math.fsum()` plus a direct sample-standard-deviation
calculation instead of `statistics.mean()` / `statistics.stdev()`. A local
200k-value benchmark produced identical formatted output and reduced formatting
time from `0.909304s` to `0.036743s` (`24.75x`).

Progress 2026-07-03: `tools/aggregate_seeds.py` now streams
`seed_summary.md` rows directly to the output handle from the main aggregation
path instead of building the complete Markdown report string before writing.
The existing `_build_summary_md()` compatibility helper still returns the same
joined text for callers that need it, while multi-seed report generation avoids
one full-report string allocation.

Progress 2026-07-04: `tools/aggregate_seeds.py` now streams
`seed_summary.json` as a top-level JSON array row by row through
`json.JSONEncoder.iterencode()` instead of first constructing
`[asdict(row) for row in seed_rows]`. The JSON field schema stays equivalent,
while large multi-seed sweeps avoid holding a second full table of row dicts
during report finalization.

Server evidence 2026-07-04: source commit `9388563` has focused red/green
verification under
`experiments/server_command_runs/aggregate_seed_json_stream_9388563_20260704_100730/`.
The red package failed the new regression test at the old full-list
`json.dump(...)` path. The green package passed that test, `py_compile`, and
the complete `tests.test_aggregate_seeds` suite (`8` tests).

Progress 2026-07-02: `tools/experiments_log.py` now bounds both best-effort
git provenance subprocesses in `_git_info()` with a 5-second timeout. This
prevents run registration/index rebuild from hanging indefinitely on a slow or
wedged git command while preserving the existing fallback behavior.

Progress 2026-07-02: `tools/experiments_log.py` now streams
`experiments/registry.jsonl` rows through `_iter_records()` for query and index
rebuild instead of first materializing every append-only record in a list.
A local 100k-row registry with 1000 unique run IDs preserved the query result
and reduced peak memory from `98.00MB` to `1.08MB`, with wall time improving
from `0.7549s` to `0.6560s`.

Progress 2026-07-02: `_iter_records()` in `tools/experiments_log.py` now skips
blank registry JSONL rows with `isspace()` and passes nonblank rows directly to
`json.loads()` instead of allocating stripped line copies. A local 100k-row
registry benchmark preserved row count and reduced raw registry iteration from
`0.688479s` to `0.666293s` (`1.03x`).

Progress 2026-07-02: `tools/experiments_log.py` now computes the
best-by-dataset table for `experiments/index.md` with one streaming max pass
over latest records, instead of building per-dataset buckets and sorting each
bucket. A local 250k-record synthetic registry preserved the selected run IDs
while reducing that index substep from `0.132s`/`1.44MB` to `0.113s`/`0.01MB`.

Progress 2026-07-02: `tools/experiments_log.py` query now applies filters
before ordering and uses a bounded heap for `--last-n`, instead of sorting
every latest run record before filtering. A local 120k-record registry
preserved `last_n=20` results while reducing query wall time from `0.9792s` to
`0.8729s`; peak memory is still dominated by latest-run de-duplication.

Progress 2026-07-02: `tools/experiments_log.py` now delays materializing
registry rows while selecting the latest record per `run_id` and best record
per dataset. Overwritten records are kept as mapping references during the scan
and only final winners are copied. A local 100k-row synthetic registry kept the
same selected run IDs while improving latest-run de-duplication from `0.1253s`
to `0.0454s` (`2.76x`) and best-by-dataset selection from `0.1291s` to
`0.0613s` (`2.11x`).

Progress 2026-07-02: `tools/experiments_log.py query` now keeps latest
registry records as mapping references through filtering and materializes dicts
only for rows returned to the caller. A local 100k-record synthetic registry
with 1k matching `mrpc` rows preserved query output while reducing filtered
query time from `0.056000s` to `0.030052s` (`1.86x`) and traced peak
allocation from `42.06MB` to `7.88MB`.

Progress 2026-07-02: `tools/experiments_log.py rebuild` now prepares and sorts
latest registry records as mapping references instead of copying every latest
record into a fresh dict before rendering `experiments/index.md`. A local
100k-record synthetic index-prep benchmark preserved selected rows while
reducing prep/sort time from `0.092343s` to `0.050433s` (`1.83x`) and traced
peak allocation from `42.04MB` to `7.86MB`.

Progress 2026-07-02: `tools/experiments_log.py rebuild` now writes
`experiments/index.md` incrementally instead of accumulating every Markdown row
in a list and joining the full document before one final write. A local
50k-record synthetic rebuild produced byte-identical index output with
comparable wall time (`0.472218s` to `0.473366s`) while reducing traced peak
allocation from `166.66MB` to `131.88MB`.

Progress 2026-07-04: `tools/experiments_log.py query --format json` now writes
the result rows directly to stdout with `json.dump(..., sys.stdout)` and a
trailing newline instead of materializing one full JSON string through
`json.dumps()` before `print()`. Query filtering and JSON indentation stay
unchanged while large registry query output avoids an extra complete string
copy.

Server evidence 2026-07-04: source commit `b66a8d2` has focused red/green
verification under
`experiments/server_command_runs/experiments_log_json_stream_b66a8d2_20260704_102030/`.
The red package failed the new regression test at the old
`print(json.dumps(rows, ...))` path. The green package passed that test,
`py_compile`, and the complete `tests.test_experiments_log` suite (`15` tests).

Progress 2026-07-04: `tools/experiments_log.py register` now reuses the same
stdout JSON streaming helper as `query --format json`, writing the registered
run record with `json.dump(..., sys.stdout)` plus a trailing newline instead of
building one full JSON string through `json.dumps()` before `print()`.

Server evidence 2026-07-04: source commit `c5424cd` has focused red/green
verification under
`experiments/server_command_runs/experiments_log_register_json_stream_c5424cd_20260704_102430/`.
The red package failed the new regression test at the old
`print(json.dumps(rec, ...))` path. The green package passed that test,
`py_compile`, and the complete `tests.test_experiments_log` suite (`16` tests).

Progress 2026-07-02: `tools/experiments_log.py rebuild` now computes status
counts, dataset counts, and best-by-dataset rows in one pass over the sorted
latest registry records, instead of scanning the same latest list again through
`_best_by_dataset()`. A local 120k-row synthetic rebuild preserved index output
and reduced the aggregation/render-prep path from `2.113363s` to `2.021970s`
(`1.05x`).

Progress 2026-07-02: `_git_info()` in `tools/experiments_log.py` now treats
raw `git status --porcelain` emptiness as the dirty-state signal instead of
calling `strip()` on the whole status payload. Git porcelain output is empty
for clean trees and non-empty for dirty trees, so run registration keeps the
same semantics while avoiding a full dirty-status copy. A local 200k-row
synthetic dirty-status benchmark preserved the result and improved the check
from `0.010460s` / `11.89MB` to `0.000009s` / `520B` traced peak allocation.

Progress 2026-07-03: `tools/experiments_log.py` now appends
`experiments/registry.jsonl` records with `json.dump(..., handle)` plus a
newline instead of materializing each registry row with `json.dumps()` before
writing. The append-only registry schema, non-ASCII handling, and newline
delimiting stay unchanged while each run registration avoids one full-row
string allocation.

Progress 2026-07-02: `scripts/blb_export_action_registry.py` now lazily imports
`blb_stage2_rl.action_space` only when registry generation actually needs it,
so dependency-light imports and help/static tooling no longer pull the torch
stack. `build_registry_payload()` also computes `per_layer_field_offsets()`
once and reuses the result for expected slot count, block counts, and summary
slot count. A synthetic 120k-offset benchmark preserved the registry summary
and reduced the offset-counting path from `0.447s` to `0.226s` (`1.97x`).

Progress 2026-07-03: `scripts/blb_export_action_registry.py` now writes the
full registry, full slot registry, and effective slot registry JSON artifacts
through shared `write_json_file()` streaming instead of materializing each
large JSON document with `json.dumps()` before `Path.write_text()`. This keeps
the exported artifact schema unchanged while avoiding three extra full-size
JSON string copies during registry export.

Progress 2026-07-04: `_all_max_action_index()` in
`scripts/blb_export_action_registry.py` now finds the all-max truncation-K
action index with one pass over `k_levels` instead of `max(k_levels)` followed
by `list(k_levels).index(...)`. Server RED/GREEN evidence is committed under
`experiments/server_command_runs/action_registry_klevel_scan_9b78854_20260704_074904/`.

- [ ] **Step 2: Move expensive rendering out of hot paths**

Keep JSON/JSONL writes in training; move PNG/HTML/NPZ rendering to post-run
commands unless the user explicitly requests live rendering.

Progress 2026-07-02: `blb_stage2_rl/persistence.py` now supports
`render_plots=False` and the `RFR_STAGE2_RENDER_PLOTS=0` environment switch
for Stage-2 curve generation. Training can keep writing the required NPZ data
while skipping synchronous PNG/PDF matplotlib rendering; the offline
`scripts/blb_regen_stage2_outputs.py` path forces `render_plots=True` so
post-run reports can regenerate the inspection artifacts later. A local
5000-episode benchmark reduced `write_training_curves()` from `1.999s` /
`44.88MB` with PNG/PDF rendering to `0.002s` / `0.30MB` when plot rendering is
disabled.

Progress 2026-07-02: the Stage-2 curve persistence path now checks sequence
length with `len()` where available and converts each NPZ series with one
`list()` pass, instead of using `len(list(seq))` before converting the same
series again. A regression test now enforces that `render_plots=False` iterates
each supplied NPZ sequence at most once. A warmed 60k-episode local parity
benchmark kept the NPZ byte size identical (`3374656` bytes) with comparable
median write time (`0.0123s` -> `0.0116s`), while removing avoidable temporary
copies from the training hot path.

Progress 2026-07-04: `blb_stage2_rl/persistence.py` curve helpers now preserve
ndarray-backed input when smoothing reward curves and computing moving
averages. `_ema_smooth()` and `_moving_average()` share `_float_array()`, which
keeps iterator inputs on the existing one-materialization path but avoids a
full Python-list copy when callers already pass numpy arrays from NPZ/report
pipelines.

Server evidence 2026-07-04: source commit `bd4ca26` has red/green verification
under
`experiments/server_command_runs/persistence_curve_ndarray_bd4ca26_20260704_015320/`.
The red test failed on `_ema_smooth()` calling `list(values)` for ndarray
input. The green gate passed `py_compile`, all nine `UpgradedCurvesTest` tests,
and a source guard for the ndarray fast-path helper.

Progress 2026-07-04: Stage-2 Stage-1-style curve panels now reuse the same
`_float_array()` helper for raw panel series. This avoids another full
Python-list copy in `_stage1_style_panel()` when report regeneration or
training-curve rendering passes ndarray-backed reward/loss/metric arrays, while
leaving iterator inputs on the existing one-materialization path.

Server evidence 2026-07-04: source commit `7460284` has red/green verification
under
`experiments/server_command_runs/persistence_panel_ndarray_7460284_20260704_015635/`.
The red test failed on `_stage1_style_panel()` calling `list(raw)` for ndarray
input. The green gate passed `py_compile`, all ten `UpgradedCurvesTest` tests,
and a source guard confirming the panel path uses `_float_array(raw)`.

Progress 2026-07-04: Stage-2 NPZ training-curve writes now reuse
`_float_array()` inside `write_training_curves()` instead of first converting
each supplied series with `list(seq)`. Iterator inputs still materialize once,
while ndarray-backed episode/reward/loss/metric arrays keep the fast numpy path
through the mandatory paper-figure NPZ artifact write.

Server evidence 2026-07-04: source commit `74de148` has red/green verification
under
`experiments/server_command_runs/persistence_npz_ndarray_74de148_20260704_020650/`.
The red test failed because the old NPZ path swallowed the blocked
`list(seq)` copy and did not write `out["npz"]`. The green gate passed
`py_compile`, all eleven `UpgradedCurvesTest` tests, and a source guard
confirming the NPZ writer uses `_float_array(seq)` without
`values = list(seq)`.

Progress 2026-07-04: Stage-2 entropy PNG rendering now reuses
`_float_array()` for both `entropy_series` and matching `entropy_episodes`.
This removes two remaining Python-list copies from the report/figure path when
training or offline regeneration already has ndarray-backed entropy curves,
while preserving the existing iterator materialization behavior.

Server evidence 2026-07-04: source commit `47782e9` has red/green verification
under
`experiments/server_command_runs/persistence_entropy_ndarray_47782e9_20260704_021330/`.
The valid red test failed because the old entropy branch caught the blocked
`list(entropy_series)` copy and did not write `entropy_png`. The green gate
passed `py_compile`, all twelve `UpgradedCurvesTest` tests, and a source guard
confirming the entropy branch uses `_float_array()` for both series and update
episodes with no `list(entropy_series)` or `list(entropy_episodes)`.

Progress 2026-07-02: Stage-2 live trace CSV schema migration now streams old
rows directly into the migrated file instead of materializing all rows in a
list and calling `writerows()`. A regression test patches `writerows()` out of
the migration path, and a local 50k-row old-schema trace benchmark preserved
the migrated row count while reducing migration peak memory from `137.08MB` to
`0.19MB` and wall time from `0.980s` to `0.877s`.

Progress 2026-07-02: `append_blb_episode_trace_row()` now caches trace CSV
paths whose schema has already been confirmed or freshly written in the current
process, so repeated PPO-update trace appends skip the redundant header
inspection. A regression test verifies the second append does not call schema
migration again after the first append created a current-schema trace. A local
3000-row append benchmark preserved CSV byte size while reducing append wall
time from `0.5079s` to `0.3204s`.

Progress 2026-07-02: `scripts/blb_regen_stage2_outputs.py` now lazily
materializes ADR-014 optional diagnostic series while reading `episodes.jsonl`.
Older runs that do not contain those fields no longer allocate nine extra
all-zero lists before plotting. A local 100k-row legacy-log benchmark preserved
the base reward/loss/metric series and reduced parsing from `1.068s` /
`25.97MB` to `0.973s` / `19.10MB`.

Progress 2026-07-02: the same offline regenerator now skips blank
`episodes.jsonl` and `ppo_updates.jsonl` lines with `isspace()` and passes
nonblank rows directly to `json.loads()` instead of allocating stripped copies.
Local 80k-row long-line benchmarks preserved parsed row counts and reduced
episode parsing from `0.776086s` to `0.755639s` (`1.03x`) and entropy parsing
from `0.300351s` to `0.283807s` (`1.06x`).

Progress 2026-07-02: `scripts/blb_regen_stage2_outputs.py` now parses
baseline reference values from `blb_stage2_report.md` and
`diagnostics_summary.md` by scanning lines and stopping once the needed values
are found, instead of reading the whole Markdown files. A local synthetic
report with the baseline table followed by 200k tail lines preserved parsed
baselines and reduced parsing from `0.0006s` / `4.20MB` to `0.0001s` /
`0.02MB`.

Progress 2026-07-03: `scripts/blb_regen_stage2_outputs.py` now reuses the
shared `jsonl_utils.iter_jsonl(..., gzip_fallback=True)` path for offline
episode and PPO diagnostics instead of maintaining a local JSONL iterator.
`scripts/verify_stage2_persistent_outputs.py` also aliases the shared required
field counter directly instead of carrying a second JSONL scan loop. This keeps
Stage-2 offline artifact regeneration and persistent-output verification on
the same low-copy JSONL implementation used by the rest of the reporting
toolchain. `Paean/action_grid.py` now reads action-config JSON artifacts through
the shared `read_json_file(..., encoding="utf-8-sig")` helper as well, keeping
Paean final-eval config loading on the same artifact-read path while preserving
the existing JSON spec parsing for command-line strings.

Progress 2026-07-03: `rl_local_optimum.py` now materializes episode returns,
optional entropy/best-score series, and collapse-attribution fusion/margin
series at most once before computing local-optimum and HOT/COLD reports. This
keeps Stage-1/Stage-2 shared health-report generation compatible with
one-shot iterators from offline regenerators and removes repeated full-list
copies from long-run report paths.

Progress 2026-07-03: `scripts/optimization_evidence_bundle.py` now resolves the
bundle output directory and optional tarball path once before archive traversal,
then skips an in-bundle tarball by relative path. Large evidence bundles no
longer call `Path.resolve()` for every payload file while adding reports to the
tar archive, preserving deterministic archive names and self-skip behavior.

Progress 2026-07-03: shared `json_utils.read_json_file()` now reads artifacts
through `Path.open()` and `json.load()` instead of `Path.read_text()` plus
`json.loads()`. Paean action configs, fusion maps, report inputs, manifests,
and optional sidecars keep the same strict/default error semantics while large
JSON artifacts avoid one extra full-file text materialization.

Progress 2026-07-03: shared `json_utils.write_json_file()` now writes artifacts
through `Path.open()` and `json.dump()` instead of materializing the whole JSON
document with `json.dumps()` before `write_text()`. Required structured
artifacts keep the same normalization, sorting, indentation, and trailing
newline behavior while large manifests/reports/evidence summaries avoid one
extra full-document string allocation.

Progress 2026-07-04: `generate_blb_glue_submission()` now reads BLB action
config JSON through shared `read_json_file(..., encoding="utf-8-sig")` instead
of `json.loads(open(...).read())`. The GLUE submission handoff keeps BOM
compatibility while avoiding one full-file string materialization for action
configs and keeping the path on the same artifact reader as Paean/final-eval
tools.

Server evidence 2026-07-04: source commit `2ded3e7` has red/green verification
under `experiments/server_command_runs/glue_json_reader_2ded3e7_20260704_040930/`.
The red static guard proved the GLUE action-config path still used
`json.loads(open(...).read())`. The green gate passed `py_compile`, the shared
reader static guard, and a source guard confirming
`payload = read_json_file(action_config_path, encoding="utf-8-sig")`.

Progress 2026-07-03: shared `jsonl_utils.write_jsonl_rows()` now streams each
bounded report/diagnostic row through `json.dump(..., handle)` and then writes
the newline, instead of building a full per-row JSON string with
`json.dumps(...)+ "\n"`. Required JSONL artifacts keep the same normalization,
sorting, and line-delimited output while large finite report rows avoid one
extra full-row string allocation.

Progress 2026-07-04: `jsonl_utils.iter_jsonl_records()` now resolves
`.jsonl`/`.jsonl.gz` paths once and opens the resolved path through
`_open_resolved_jsonl()`. `open_jsonl()` keeps the public behavior for callers,
while shared readers avoid the duplicate `resolve_jsonl_path()` call on every
file scan.

Server evidence 2026-07-04: source commit `643ae60` has red/green verification
under
`experiments/server_command_runs/jsonl_resolve_once_643ae60_20260704_034331/`.
The red test proved the old iterator resolved the path twice. The green gate
passed `py_compile`, all `tests.test_jsonl_utils` tests, and a source guard
confirming `iter_jsonl_records()` opens the already resolved path directly.

Progress 2026-07-03: `scripts/stage2_first10k_monitor.py` now writes live and
final monitor summary JSON files through shared `write_json_file()` streaming
and appends `monitor_events.jsonl` rows with `json.dump()` directly into the
open append handle. The online watchdog keeps the same summary/event schema and
exit-code behavior while avoiding full-document string copies on every monitor
poll.

Progress 2026-07-03: `rl_data_points.py` now uses shared streaming JSON
helpers for manifest merge/write and summary write, and appends Stage-1/Stage-2
training JSONL rows with a reused `JSONEncoder.iterencode()` into the existing
buffered file handle. Required `rl_training_data_points/` schemas and flush
cadence stay unchanged while long RL runs avoid one full JSON string allocation
per manifest, summary, step, episode, and PPO diagnostic row.

Progress 2026-07-03: `blb_stage2_rl/diagnostics.py` now reuses one
`JSONEncoder` for primary Stage-2 diagnostic JSONL rows and streams encoder
chunks directly into the buffered `episodes.jsonl` / `ppo_updates.jsonl`
handles. Periodic `top_candidates.jsonl` and `pareto_frontier.jsonl` rewrites
use the same row writer. The append schema, default string fallback, buffering,
and flush cadence stay unchanged while training diagnostics avoid per-row
`json.dumps(...)+ "\n"` allocations.

Progress 2026-07-03: `blb_stage2_rl/candidate_store.py` now appends candidate
store JSONL records with a reused `JSONEncoder.iterencode()` writer instead of
building a complete `json.dumps(...)+ "\n"` string for every candidate. The
append-only store keeps the same stable sorting, ASCII escaping, action hashes,
candidate identity fields, and read-back behavior while reducing allocation in
Stage-2 search candidate persistence.

Progress 2026-07-03: `blb_stage2_rl/candidate_store.py`
`_action_hash_from_tuple()` now streams the compact integer-array payload
directly into `hashlib.sha256()` instead of building `list(action_indices)`,
serializing it through `json.dumps()`, and encoding the resulting string. The
hash bytes remain compatible with the previous compact JSON form, so candidate
identity and persisted records stay stable.

Server evidence 2026-07-03: source commit `cf4eed6` has red/green verification
under
`experiments/server_command_runs/candidate_action_hash_cf4eed6_20260703_221100/`.
The red run proved the old helper still used `json.dumps`; the green run
verified `py_compile=0`, hash compatibility for `[4,3,2,-1]`, and the
no-`json.dumps` source guard.

Progress 2026-07-04: `blb_stage2_rl/candidate_store.py` now normalizes
ndarray-backed candidate action vectors by flattening with `reshape(-1)` before
the legacy `.tolist()` compatibility branch. This keeps list, string, and
generic iterable compatibility while avoiding an eager nested-list copy on
candidate-store and action-hash paths that already hold numpy arrays.

Server evidence 2026-07-04: source commit `0aa212a` has red/green verification
under
`experiments/server_command_runs/candidate_store_ndarray_0aa212a_20260704_024050/`.
The red test failed on the old `.tolist()` path for ndarray input. The green
gate passed `py_compile`, all ten `BLBCandidateStoreIdentityTests`, and a source
guard confirming the reshape fast path precedes the `.tolist()` compatibility
path.

Progress 2026-07-03: `blb_stage2_rl/diagnostics.py` now streams generated
`diagnostics_summary.md` and `pareto_frontier.html` lines into their temporary
files instead of materializing a single `"\n".join(lines)` document before
atomic replace. Report contents, file names, and refresh cadence stay unchanged
while periodic diagnostics flushes avoid a second full Markdown/HTML string
copy during long Stage-2 runs.

Progress 2026-07-03: `blb_stage2_rl/persistence.py` now streams action
description Markdown, final Stage-2 report Markdown, and crash-report text line
writes instead of materializing one full joined string before writing. These
paths keep the same report sections and output file names while reducing peak
allocation in Stage-2 persistence/report flushes.

- [ ] **Step 3: Verify**

Run:

```bash
python3 -m unittest tests.test_rl_data_points tests.test_stage2_persistent_output_verifier -v
```

## Task 8: Server Evidence and Promotion Loop

**Files:**

- Modify: `SERVER_COMMAND.md`
- Use: `scripts/project_optimization_audit.py`
- Use: `scripts/server_resource_snapshot.py`
- Use: `scripts/gpu_utilization_report.py`
- Use: `scripts/stage1_parallel_report.py`
- Use: `scripts/optimization_evidence_bundle.py`
- Use: `scripts/stage2_ngpu_ab_compare.py`

- [ ] **Step 1: Write one server command per promoted optimization**

Each command must record:

- source commit.
- GPU inventory.
- exact command.
- output artifact directory.
- timing summary.
- semantic parity/eval evidence.

- [ ] **Step 2: Pull artifacts back locally**

Import compact summaries into `experiments/server_command_runs/` or
`reports/html_reports/` as appropriate.

Progress 2026-07-02: added torch-free
`scripts/optimization_evidence_bundle.py`, which packages project audit,
Stage-1 parallel timing, Stage-2 GPU utilization, and Stage-2 persistent
verification outputs into one manifest/index, with optional `--tar-gz` archive
output. This reduces server post-run manual stitching before evidence
promotion. The project audit artifact scan now walks each artifact root once
instead of once per pattern, reducing post-run evidence packaging overhead on
large `experiments/` trees. Stage-1 logs and Stage-2 episode diagnostics are
now streamed by the evidence tools, reducing peak memory during long-run bundle
generation. Actual server A/B artifact pullback is still pending.

Progress 2026-07-02: `scripts/optimization_evidence_bundle.py` now writes
optional tar.gz archives with a streaming `os.walk` traversal instead of
materializing and sorting every path under the bundle directory. A local 2k-file
archive comparison produced identical entries and reduced traced peak memory
from `2.76MB` to `1.70MB` (`1.63x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now hashes the
canonical Rescale_optimizer source/config subset by streaming file chunks
instead of calling `Path.read_bytes()` for every `.py`/`.json` file. This keeps
Trust-0 manifest generation memory-bounded for larger optimizer/config trees
while preserving the manifest hash format.

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now also streams the
canonical Rescale_optimizer file traversal with a heap-ordered directory walk
instead of collecting all `.py`/`.json` paths before hashing. A local 2k-file
Rescale tree comparison produced the same canonical hash and reduced traced
peak memory from `2.05MB` to `1.08MB` (`1.90x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now uses the same
heap-ordered traversal for generic directory hashes, replacing
`sorted(path.rglob("*"))` in `_dir_sha256()`. A local 2k-file directory
comparison preserved the full-tree hash, kept cache-directory skip semantics,
and reduced traced peak memory from `2.21MB` to `1.10MB` (`2.02x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now bounds all
best-effort git subprocess calls in `_run_git()` with a 5-second timeout. This
keeps Trust-0 manifest generation from stalling a server command indefinitely
if git status/diff/upstream resolution hangs, while preserving the existing
`None` fallback on failure.

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now reuses the
already-cleaned `git status --short --branch` string for manifest dirty-state
detection instead of calling `strip()` a second time in `build_manifest()`.
This preserves `status_short` and dirty semantics while avoiding another full
status edge scan in evidence packaging. A local 200k-row synthetic status
benchmark preserved the dirty result and improved the second-check path from
`0.078152s` to `0.060709s` over 200k repeats (`1.29x`).

Progress 2026-07-02: `scripts/blb_make_run_manifest.py` now builds
`per_layer_field_offsets()` once per manifest and reuses that tuple for both
per-block slot counts and total per-layer slot count. A local 120k-offset
synthetic benchmark preserved the manifest-derived counts, improved best time
from `0.349566s` to `0.222708s` (`1.57x`), and cut traced peak memory from
`30.30MB` to `15.18MB` (`2.00x`).

Progress 2026-07-04: `scripts/blb_make_run_manifest.py` now streams the CLI
stdout JSON path summary with `json.dump(..., sys.stdout)` plus a trailing
newline instead of materializing it through `json.dumps()` before `print()`.
The manifest JSON artifact still uses `write_json_file()`, and Trust-0 manifest
contents are unchanged.

Server evidence 2026-07-04: source commit `362ea22` has focused RED/GREEN
verification under
`experiments/server_command_runs/manifest_stdout_json_362ea22_20260704_105117/`.
The RED test failed on the old `print(json.dumps(paths, ...))` stdout path.
The GREEN gate passed `py_compile` and the full
`tests.test_blb_make_run_manifest` suite (`11` tests).

Progress 2026-07-02: `_dir_sha256()` in `scripts/blb_make_run_manifest.py` now
prunes skip directories such as `.git`, `__pycache__`, `.pytest_cache`, and
`.mypy_cache` before calling `iterdir()` on them, while preserving the existing
full-tree hash for included files. A local synthetic tree with 30 kept files and
3600 skipped `.git` files preserved the digest and reduced hash time from
`0.3584s` to `0.0105s`.

Progress 2026-07-02: `scripts/blb_phase0_preflight.py` now scans source/config
files line by line when building `blb_entrypoints_grep.txt` instead of
materializing each file with `read_text().splitlines()`. A local 100k-line
source-file benchmark preserved the same 10 entrypoint matches and reduced
traced peak memory from `8.36MB` to `0.02MB`.

Progress 2026-07-02: `scripts/blb_phase0_preflight.py` now reuses the same
repository walk to generate `repo_file_list.txt`, `repo_code_config_files.txt`,
and `blb_entrypoints_grep.txt`, instead of walking the tree once for file lists
and again for grep. A local synthetic 2400-file repo preserved all three report
outputs and reduced Phase-0 report generation from `0.1505s` / `0.78MB` to
`0.1234s` / `0.30MB`.

Progress 2026-07-02: `scripts/blb_verify_noise_install.py` now caches the
torch-free noise-variance table extracted from `function_handler.py` after the
first AST parse in a process. Repeated verifier/test calls no longer reread and
reparse the same `_NOISE_STD_RAW` literal while the variance lookup table shape
and values remain unchanged. A real-source local benchmark over 80 repeated
loads reduced table extraction from `3.228s` to `0.040s` (`80.43x`).

Progress 2026-07-02: `scripts/project_optimization_audit.py` now walks
artifact roots with a streaming sorted `os.walk()` iterator instead of
`Path.rglob("*")`. A local synthetic 3400-file artifact tree preserved the same
evidence counts and reduced scan cost from `0.1523s`/`1.89MB` to
`0.1270s`/`0.58MB`.

Progress 2026-07-02: `_iter_files()` in `scripts/project_optimization_audit.py`
now keeps deterministic directory traversal with `dirnames.sort()` but avoids
sorting each directory's `filenames` list before artifact classification. A
local 120k-filename synthetic artifact directory preserved evidence counts,
reduced scan time from `2.215341s` to `2.059308s` (`1.08x`), and cut traced
peak memory from `8.43MB` to `0.92MB`.

Progress 2026-07-02: `scripts/run_fusion_count_action_eval_rlpath.py` now
uses copy-on-convert behavior in `_jsonable()`, so already JSON-native
RL-path report trees such as `step_records`, `fusion_action_steps`, and
terminal diagnostics are reused instead of being recursively cloned before
`json.dumps()`. A local 800-group synthetic RL-path report benchmark preserved
the converted payload semantics and reduced conversion from `2.391359s` /
`37.62MiB` peak to `0.485470s` / near-zero traced allocation.

Progress 2026-07-03: the low-conflict source/evidence loop has been exercised
for six targeted optimizations. Each followed the local-source, git-push,
server-temp-run, artifact-pullback, evidence-commit workflow:

- `567ad75` Paean repeat-install reuse, evidence committed in the
  `final_eval_repeat_install_reuse_567ad75_20260703_203900` run directory.
- `b2a7325` Paean max-SF table cache, evidence committed in the
  `final_eval_max_sfs_cache_b2a7325_20260703_205000` run directory.
- `dca7526` Stage-1 `apply_configuration()` install reuse, evidence committed
  in the `stage1_apply_config_reuse_dca7526_20260703_210000` run directory.
- `5d15e6c` Stage-1 worker install reuse, evidence committed in the
  `stage1_worker_apply_config_reuse_5d15e6c_20260703_211000` run directory.
- `61c8c57` Stage-1 reward history bounded deque, evidence committed in the
  `stage1_reward_history_deque_392b646_20260703_215700` run directory.
- `497ecda` shared reward-probe scalar sync batching, evidence committed in the
  `probe_scalar_sync_497ecda_20260704_025145` run directory.
- `b5dfff5` shared reward-probe accuracy-only prediction-array skip, evidence
  committed in the `probe_skip_pred_arrays_b5dfff5_20260704_025610` run
  directory.
- `2d98907` shared reward-probe tensor prediction-array packed transfer,
  evidence committed in the `probe_tensor_arrays_2d98907_20260704_030105` run
  directory.
- `7be83af` shared installed inference MNLI accuracy helper reuse, evidence
  committed in the `inference_mnli_accuracy_helper_7be83af_20260704_042030`
  run directory.
- `08560c1` final-eval finite-float helper streaming stats, evidence committed
  in the `final_stat_helpers_08560c1_20260704_044500` run directory.
- `75cce4c` final-eval variance plot mean streaming, evidence committed in the
  `final_variance_plot_mean_75cce4c_20260704_052500` run directory.
- `e4c3d47` final-eval variance scatter single-scan path, evidence committed
  in the `final_variance_scatter_scan_e4c3d47_20260704_054500` run directory.
- `c85b896` final-eval comparison scatter single-scan path, evidence committed
  in the `final_comparison_scatter_scan_c85b896_20260704_061000` run
  directory.
- `a1de9a3` final-eval axis-limit streaming min/max helper, evidence committed
  in the `final_axis_limits_stream_a1de9a3_20260704_063500` run directory.
- `643ae60` shared JSONL single path resolution, evidence committed in the
  `jsonl_resolve_once_643ae60_20260704_034331` run directory.
- `2ded3e7` BLB GLUE action-config shared JSON reader, evidence committed in
  the `glue_json_reader_2ded3e7_20260704_040930` run directory.
- `8101feb` final-eval random summary running stats, evidence committed in the
  `final_summary_running_8101feb_20260704_042900` run directory.
- `da02fca` shared reward-probe count-weight reuse, evidence committed in the
  `eval_metric_weights_da02fca_20260704_030610` run directory.
- `1a6969a` shared reward-probe single-array flatten fast path, evidence
  committed in the `eval_single_array_1a6969a_20260704_031145` run directory.
- `f9bbb29` shared reward-probe binary weighted-F1 fast path, evidence
  committed in the `eval_binary_f1_f9bbb29_20260704_034430` run directory.
- `d0e8b8c` shared reward-probe binary MCC fast path, evidence committed in the
  `eval_binary_mcc_d0e8b8c_20260704_035430` run directory.
- `211ca50` shared reward-probe direct accuracy count path, evidence committed
  in the `eval_accuracy_count_211ca50_20260704_041100` run directory.
- `a416d46` shared attention tail cursor parsing, evidence committed in the
  `attention_tail_cursor_a416d46_20260703_214800` run directory.
- `43ec3cc` Stage-2/Paean action-space average K direct arithmetic helper,
  evidence committed in the `action_avg_k_direct_43ec3cc_20260704_044200`
  run directory.
- `4db8e02` Stage-2/Paean action-space direct effective-K accumulator and
  cached K-slot positions, evidence committed in the
  `action_k_accum_direct_4db8e02_20260704_044800` run directory.
- `e8bb0dc` Stage-2 persistence curve sequence fast path, evidence committed
  in the `persistence_float_array_sequence_e8bb0dc_20260704_045400` run
  directory.
- `00bc7e8` Stage-2 diagnostic-curve array cache, evidence committed in the
  `diagnostic_curve_array_cache_00bc7e8_20260704_050000` run directory.
- `ec0776b` Stage-2 persistence iterable-length streaming count, evidence
  committed in the `persistence_seq_len_count_ec0776b_20260704_051045` run
  directory.
- `b6dda66` paper-figure JSON-native sidecar payload reuse, evidence committed
  in the `paper_figures_payload_reuse_b6dda66_20260704_094735` run directory.
- `596c458` paper-figure training reward series reuse, evidence committed in
  the `paper_training_curve_reuse_596c458_20260704_095340` run directory.
- `7d66fed` paper-figure grouped reward matrix streaming, evidence committed
  in the `paper_group_curve_matrix_7d66fed_20260704_095833` run directory.
- `9388563` aggregate-seed JSON summary streaming, evidence committed in the
  `aggregate_seed_json_stream_9388563_20260704_100730` run directory.
- `623cd5d` Stage-2 A/B excluded-key set reuse, evidence committed in the
  `stage2_ab_excluded_keys_623cd5d_20260704_101330` run directory.
- `b66a8d2` experiments-log query JSON stdout streaming, evidence committed in
  the `experiments_log_json_stream_b66a8d2_20260704_102030` run directory.
- `c5424cd` experiments-log register JSON stdout streaming, evidence committed
  in the `experiments_log_register_json_stream_c5424cd_20260704_102430` run
  directory.
- `5d248a0` Stage-1 plaintext repeat-eval stdout JSON streaming, evidence
  committed in the `stage1_stdout_json_stream_5d248a0_20260704_103647` run
  directory.
- `9412e3b` GPU utilization Markdown device-list streaming, evidence committed
  in the `gpu_markdown_device_stream_9412e3b_20260704_104315` run directory.
- `0980322` fusion-count map report stdout JSON streaming, evidence committed
  in the `fusion_report_stdout_json_0980322_20260704_104732` run directory.
- `362ea22` BLB Trust-0 manifest stdout JSON streaming, evidence committed in
  the `manifest_stdout_json_362ea22_20260704_105117` run directory.
- `cf4eed6` Stage-2 candidate action hash streaming, evidence committed in the
  `candidate_action_hash_cf4eed6_20260703_221100` run directory.
- `0aa212a` Stage-2 candidate ndarray normalization, evidence committed in the
  `candidate_store_ndarray_0aa212a_20260704_024050` run directory.
- `ccbfc5f` fusion report action-config option-index reuse, evidence committed
  in the `fusion_report_option_index_ccbfc5f_20260704_051827` run directory.
- `71fbfc6` fusion report field-kind lookup reuse, evidence committed in the
  `fusion_report_field_kinds_71fbfc6_20260704_052234` run directory.
- `82b83ca` fusion report adjusted block-action cache, evidence committed in
  the `fusion_report_block_actions_82b83ca_20260704_052649` run directory.
- `f8d649e` fusion report bound slot-entry cache, evidence committed in the
  `fusion_report_slot_entries_f8d649e_20260704_053236` run directory.
- `476a230` fusion report bound slot mapping direct iteration, evidence
  committed in the `fusion_report_bound_slot_items_476a230_20260704_053630`
  run directory.
- `269ba69` fusion report option/base slot mapping direct iteration, evidence
  committed in the `fusion_report_slot_mapping_269ba69_20260704_072557` run
  directory.
- `74d5d28` fusion report graph occurrence set accumulation, evidence
  committed in the `fusion_report_occurrences_74d5d28_20260704_073030` run
  directory.
- `c3db582` fusion report action sequence direct indexing, evidence committed
  in the `fusion_report_action_sequence_c3db582_20260704_073447` run
  directory.
- `b0a1928` fusion report base action direct pass-through, evidence committed
  in the `fusion_report_base_action_b0a1928_20260704_073904` run directory.
- `248a0ec` active SERVER_COMMAND fusion-map gate sidecar filtering, evidence
  committed in the `fusion_map_gate_filter_248a0ec_20260704_094103` run
  directory.
- `dab3b8b` static-skeleton archive cache, evidence committed in the
  `baseline_archive_cache_dab3b8b_20260703_212500` run directory.
- `cb215bd` skeleton profile config discovery, evidence committed in the
  `skeleton_profile_config_discovery_cb215bd_20260703_213500` run directory.

- [ ] **Step 3: Commit/push source and evidence**

Never leave canonical source changes only on the server.

## Completion Audit

Before marking the objective complete, verify:

- This plan has no unchecked high-impact phases.
- Every implemented optimization has tests and timing evidence.
- Server-side hardware-utilization claims have server artifacts.
- Stage-1 and Stage-2 validation protocols remain intact.
- `git status` shows no uncommitted source changes made by this optimization
  work.
