from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAYER_EVALUATOR = ROOT / "src/rfr/search/common/evaluator.py"
PARALLEL_RUNNER = ROOT / "src/rfr/search/rl/stage1/parallel_runner.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _method_region(source: str, method_name: str) -> str:
    start = source.index(f"    def {method_name}")
    next_method = source.find("\n    def ", start + 1)
    if next_method == -1:
        next_method = len(source)
    return source[start:next_method]


def _single_gpu_rollout_region(source: str) -> str:
    start = source.index("                if not _handled_via_parallel:")
    end = source.index("                    buffer.end_episode()", start)
    return source[start:end]


class Stage1ParallelSemanticsTest(unittest.TestCase):
    def test_parallel_worker_uses_serial_sos_token_for_initial_previous_action(self):
        region = _method_region(_source(LAYER_EVALUATOR), "_stage1_collect_episode_in_worker")

        self.assertIn("SOS_TOKEN_GELU", region)
        self.assertNotIn("prev_g = torch.zeros", region)
        self.assertNotIn("seq_prev_s", region)
        self.assertNotIn("prev_s = torch.zeros", region)

    def test_parallel_rollout_logs_per_window_timing_diagnostics(self):
        source = _source(LAYER_EVALUATOR)

        self.assertIn("format_diagnostics_line", source)
        self.assertIn("last_diagnostics", source)

    def test_parallel_runner_exposes_worker_timing_and_speedup_diagnostics(self):
        source = _source(PARALLEL_RUNNER)

        self.assertIn("per_worker_seconds", source)
        self.assertIn("speedup_vs_sequential", source)

    def test_parallel_rollout_logs_forward_and_report_write_timing(self):
        evaluator_source = _source(LAYER_EVALUATOR)
        runner_source = _source(PARALLEL_RUNNER)

        self.assertIn("_stage1_parallel_model_forward_seconds", evaluator_source)
        self.assertIn("_stage1_worker_timing_snapshot", evaluator_source)
        self.assertIn('"model_forward_seconds"', evaluator_source)
        self.assertIn('"report_write_seconds"', evaluator_source)
        self.assertIn("model_forward=", evaluator_source)
        self.assertIn("report_write=", evaluator_source)
        self.assertIn("model_forward_seconds", runner_source)
        self.assertIn("model_forward_calls", runner_source)

    def test_stage1_parallel_total_uses_actual_worker_episode_counts(self):
        source = _source(LAYER_EVALUATOR)
        marker = "_stage1_parallel_window_episodes = ("
        start = source.index(marker)
        region = source[start:source.index("_stage1_parallel_known_seconds", start)]

        self.assertIn("sum(_diag.per_worker_episode_counts)", region)
        self.assertNotIn(
            "len(_stage1_parallel_runner.workers)\n"
            "                            * int(_stage1_parallel_runner.last_diagnostics.episodes_per_worker)",
            region,
        )

    def test_policy_replica_sync_reuses_one_state_dict_per_window(self):
        region = _method_region(_source(PARALLEL_RUNNER), "_sync_policy_replicas")

        self.assertIn("state_dict = None", region)
        self.assertIn("state_dict = gtrxl_net.state_dict()", region)
        self.assertIn("load_state_dict(state_dict)", region)
        self.assertNotIn("load_state_dict(gtrxl_net.state_dict())", region)

    def test_recurrent_rollout_buffer_packs_scalar_arrays_before_tensor_transfer(self):
        source = _source(LAYER_EVALUATOR)
        region = _method_region(source, "get_batch")

        self.assertIn("def _pack_recurrent_rollout_arrays", source)
        self.assertIn("np.asarray([ep['layer_indices'] for ep in episodes], dtype=np.int64)", source)
        self.assertIn("np.asarray([ep['gelu_masks'] for ep in episodes], dtype=bool)", source)
        self.assertIn("layer_indices_np,", region)
        self.assertIn("prev_g_actions_np,", region)
        self.assertIn("actions_g_np,", region)
        self.assertIn("torch.from_numpy(layer_indices_np).to(device)", region)
        self.assertNotIn("torch.tensor(ep['layer_indices'], dtype=torch.long)", region)
        self.assertNotIn("torch.tensor(ep['prev_g_actions'], dtype=torch.long)", region)
        self.assertNotIn("torch.tensor(ep['actions_g'], dtype=torch.long)", region)
        self.assertNotIn("torch.tensor([", region)

    def test_recurrent_rollout_buffer_packs_scalar_tensor_fields_directly_to_device(self):
        source = _source(LAYER_EVALUATOR)
        region = _method_region(source, "get_batch")

        self.assertIn("def _pack_recurrent_rollout_tensor_arrays", source)
        if "def _stage1_scalar_episode_values_to_tensor" not in source:
            self.fail("Stage-1 rollout scalar tensor fields must pack directly to target device")
        self.assertIn("cont_features_np, logprobs, values", region)
        self.assertIn("torch.from_numpy(cont_features_np).to(device)", region)
        self.assertIn("_stage1_scalar_episode_values_to_tensor(episodes, 'logprobs', device)", source)
        self.assertIn("_stage1_scalar_episode_values_to_tensor(episodes, 'values', device)", source)
        self.assertNotIn("torch.from_numpy(logprobs_np).to(device)", region)
        self.assertNotIn("torch.from_numpy(values_np).to(device)", region)
        self.assertNotIn("cont_features = torch.stack([", region)
        self.assertNotIn("logprobs = torch.stack([", region)
        self.assertNotIn("values = torch.stack([", region)

    def test_stage1_rollout_reuses_action_scalar_and_builds_device_tensors_directly(self):
        source = _source(LAYER_EVALUATOR)
        worker_region = _method_region(source, "_stage1_collect_episode_in_worker")
        fallback_region = _single_gpu_rollout_region(source)

        self.assertIn("prev_g_idx = SOS_TOKEN_GELU", worker_region)
        self.assertIn("gelu_action_idx = int(gelu_action.item())", worker_region)
        self.assertIn("env.step(gelu_action_idx)", worker_region)
        self.assertIn("rollout.actions_g.append(gelu_action_idx)", worker_region)
        self.assertIn("cont_feat_record, dtype=torch.float32, device=device", worker_region)
        self.assertIn("seq_prev_g[0, step] = int(prev_g_idx)", worker_region)

        self.assertIn("prev_g_idx = SOS_TOKEN_GELU", source)
        self.assertIn("gelu_action_idx,", fallback_region)
        self.assertIn("prev_g_idx,", fallback_region)
        self.assertIn("action_g=action_g_record", fallback_region)
        self.assertIn("prev_g=prev_g_record", fallback_region)
        self.assertNotIn("env.step(gelu_action.item())", source)
        self.assertNotIn("GELU_MAP[int(gelu_action.item())]", source)
        self.assertNotIn("prev_g=prev_g.squeeze().item()", source)
        self.assertNotIn("action_g=gelu_action.item()", source)

    def test_stage1_rollout_reuses_static_gelu_mask_template_per_device(self):
        source = _source(LAYER_EVALUATOR)
        worker_region = _method_region(source, "_stage1_collect_episode_in_worker")
        fallback_region = _single_gpu_rollout_region(source)

        self.assertIn("def _get_stage1_gelu_mask_templates(", source)
        self.assertIn("_stage1_gelu_mask_template_cache", source)
        for region in (worker_region, fallback_region):
            self.assertIn(
                "gelu_mask_np, gelu_mask_t = _get_stage1_gelu_mask_templates(",
                region,
            )
            self.assertIn(
                "seq_gelu_masks.copy_(gelu_mask_t.view(1, 1, -1).expand_as(seq_gelu_masks))",
                region,
            )
            self.assertNotIn("env.get_gelu_action_mask(layer_idx)", region)
            self.assertNotIn("torch.as_tensor(\n                gelu_mask_np, dtype=torch.bool", region)

    def test_stage1_rollout_reuses_preallocated_sequence_tensors(self):
        source = _source(LAYER_EVALUATOR)
        worker_region = _method_region(source, "_stage1_collect_episode_in_worker")

        self.assertIn("seq_cont_feats = torch.empty(", worker_region)
        self.assertIn("seq_cont_feats[0, step].copy_(cont_feat_t)", worker_region)
        self.assertIn("full_cont = seq_cont_feats[:, : step + 1, :]", worker_region)
        self.assertIn("full_prev_g = seq_prev_g[:, : step + 1]", worker_region)

        self.assertIn("seq_cont_feats = torch.empty(", source)
        self.assertNotIn("torch.cat(seq_cont_feats", source)
        self.assertNotIn("torch.cat(seq_layer_indices", source)
        self.assertNotIn("torch.cat(seq_prev_g", source)
        self.assertNotIn("torch.cat(seq_gelu_masks", source)

    def test_stage1_rollout_records_cont_features_without_cpu_tensor_roundtrip(self):
        source = _source(LAYER_EVALUATOR)
        worker_region = _method_region(source, "_stage1_collect_episode_in_worker")
        fallback_region = _single_gpu_rollout_region(source)
        runner_source = _source(PARALLEL_RUNNER)

        self.assertIn("cont_feat_record = np.asarray(cont_feat_np, dtype=np.float32)", worker_region)
        self.assertIn("rollout.cont_features.append(cont_feat_record)", worker_region)
        self.assertIn("transition_records.append(", fallback_region)
        self.assertIn("cont_feat_record,", fallback_region)
        self.assertIn("cont_feat=cont_feat_record", fallback_region)
        self.assertIn("cont_features: List[np.ndarray]", runner_source)
        self.assertNotIn("torch.tensor(cont_feat_np", source)

    def test_stage1_rollout_defers_worker_logprob_value_sync_until_episode_end(self):
        source = _source(LAYER_EVALUATOR)
        worker_region = _method_region(source, "_stage1_collect_episode_in_worker")
        runner_source = _source(PARALLEL_RUNNER)

        self.assertIn("logprob_tensors.append(logprob.detach()", worker_region)
        self.assertIn("value_tensors.append(value.detach()", worker_region)
        self.assertIn("gelu_prob_tensors.append(gelu_probs.detach()", worker_region)
        self.assertIn("_stage1_scalar_tensors_to_float_list(logprob_tensors)", worker_region)
        self.assertIn("_stage1_prob_tensors_to_nested_lists(gelu_prob_tensors)", worker_region)
        self.assertNotIn("logprob_value = float(logprob.detach().cpu().item())", worker_region)
        self.assertNotIn("critic_value = float(value.item())", worker_region)
        self.assertNotIn("gelu_probs.detach().cpu().numpy().tolist()", worker_region)

        self.assertIn("logprobs: List[float]", runner_source)
        self.assertIn("values: List[float]", runner_source)
        self.assertNotIn("rollout.logprobs.append(logprob.detach().cpu())", source)
        self.assertNotIn("rollout.values.append(value.detach().cpu())", source)
        self.assertNotIn("logprob=logprob.cpu()", source)
        self.assertNotIn("value=value.cpu()", source)

    def test_stage1_single_gpu_rollout_defers_logprob_value_sync_until_episode_end(self):
        source = _source(LAYER_EVALUATOR)
        fallback_region = _single_gpu_rollout_region(source)

        self.assertIn("logprob_tensors.append(logprob.detach()", fallback_region)
        self.assertIn("value_tensors.append(value.detach()", fallback_region)
        self.assertIn("gelu_prob_tensors.append(gelu_probs.detach()", fallback_region)
        self.assertIn("transition_records.append(", fallback_region)
        self.assertIn("_stage1_scalar_tensors_to_float_list(logprob_tensors)", fallback_region)
        self.assertIn("_stage1_prob_tensors_to_nested_lists(gelu_prob_tensors)", fallback_region)
        self.assertIn("for idx, step_info in enumerate(step_infos):", fallback_region)
        self.assertIn("buffer.add_step(", fallback_region)
        self.assertNotIn("logprob_value = float(logprob.detach().cpu().item())", fallback_region)
        self.assertNotIn("critic_value = float(value.item())", fallback_region)
        self.assertNotIn("gelu_probs.cpu().numpy().tolist()", fallback_region)

    def test_stage1_parallel_replay_stash_uses_deque(self):
        source = _source(LAYER_EVALUATOR)
        start = source.index("_stage1_parallel_stash = deque()")
        end = source.index("                if not _handled_via_parallel:", start)
        parallel_region = source[start:end]

        self.assertIn("from collections import deque", source)
        self.assertIn("_stage1_parallel_stash = deque()", source)
        self.assertIn("_stage1_parallel_stash.extend(_rollouts)", parallel_region)
        self.assertIn("rollout = _stage1_parallel_stash.popleft()", parallel_region)
        self.assertNotIn("_stage1_parallel_stash.pop(0)", source)

    def test_noise_scaling_validation_scans_arrays_without_tolist_materialization(self):
        source = _source(LAYER_EVALUATOR)

        if "def _unsupported_int_values" not in source:
            self.fail("layer evaluator is missing shared _unsupported_int_values helper")
        for method_name in (
            "validate_input_noise_scaling_factors",
            "validate_weight_noise_scaling_factors",
            "validate_softmax_value_noise_scaling_factors",
        ):
            region = _method_region(source, method_name)
            if "_unsupported_int_values(" not in region:
                self.fail(f"{method_name} does not use _unsupported_int_values")
            if "arr.tolist()" in region:
                self.fail(f"{method_name} still materializes arr.tolist()")


if __name__ == "__main__":
    unittest.main()
