# Full Test Summary

- Source: `9ab82f04375689aeefdb89dac22cca5cdd0ceb7a`
- Tests: 1229
- Result: `FAILED (failures=18, errors=1, skipped=2)`
- Same failure set as `991329a`: True
- Added failures: 0
- Removed failures: 0

## Remaining Failures

| Kind | Test | Module |
| --- | --- | --- |
| ERROR | `test_action_candidate_applies_replan_cfg_before_model_forward` | `test_blb_stage2_rl_regressions.BLBActionFinalEvalRegressionTests` |
| FAIL | `test_policy_sample_and_evaluate_share_masked_logprob` | `test_blb_action_mask.BLBActionMaskTests` |
| FAIL | `test_adaptive_scalar_cost_has_fusion_and_truncation_step_boosts` | `test_blb_reward_archive.ParetoCostArchiveTests` |
| FAIL | `test_compute_reward_uses_adaptive_scalar_for_p3_cost_while_recording_pareto` | `test_blb_reward_archive.ParetoCostArchiveTests` |
| FAIL | `test_p3_cost_rank_remains_unbounded_after_ppo_cost_score_clips` | `test_blb_reward_archive.ParetoCostArchiveTests` |
| FAIL | `test_executable_eval_paths_use_shared_optimizer_writeback_helper` | `test_blb_stage2_eval_single_path_static.Stage2EvalSinglePathStaticTest` |
| FAIL | `test_installed_model_forward_paths_use_shared_inference_eval` | `test_blb_stage2_eval_single_path_static.Stage2EvalSinglePathStaticTest` |
| FAIL | `test_probe_runner_diagnostics_payload_uses_shared_helper` | `test_blb_stage2_eval_single_path_static.Stage2EvalSinglePathStaticTest` |
| FAIL | `test_action_candidate_still_runs_model_forward_when_optimizer_invalid` | `test_blb_stage2_rl_regressions.BLBActionFinalEvalRegressionTests` |
| FAIL | `test_baseline_derived_threshold_uses_all_max_blb_metric` | `test_blb_threshold_semantics.BLBThresholdSemanticsTests` |
| FAIL | `test_existing_script_wrappers_use_shared_helper` | `test_cli_parse_utils.CliParseUtilsTest` |
| FAIL | `test_legacy_stage2_rollout_defers_logprob_value_sync_until_buffer_pack` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
| FAIL | `test_probe_runner_aggregates_trial_results_in_preallocated_lists` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
| FAIL | `test_probe_runner_caches_round_robin_trial_assignments` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
| FAIL | `test_rollout_policy_uses_causal_prefix_fast_path` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
| FAIL | `test_sequential_stage2_rollout_defers_logprob_value_sync_until_buffer_pack` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
| FAIL | `test_stage2_fusion_action_level_masks_are_cached_per_device` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
| FAIL | `test_stage2_parallel_rollout_batches_worker_scalar_sync_at_episode_end` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
| FAIL | `test_stage2_step_static_tensors_are_cached_per_schedule_device` | `test_sequential_smoke.MultiGpuProbeThroughputRegressionTest` |
