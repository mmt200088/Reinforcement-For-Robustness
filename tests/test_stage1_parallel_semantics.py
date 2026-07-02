from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAYER_EVALUATOR = ROOT / "layer_importance_evaluator.py"
PARALLEL_RUNNER = ROOT / "stage1_rl" / "parallel_runner.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _method_region(source: str, method_name: str) -> str:
    start = source.index(f"    def {method_name}")
    next_method = source.find("\n    def ", start + 1)
    if next_method == -1:
        next_method = len(source)
    return source[start:next_method]


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

    def test_recurrent_rollout_buffer_packs_tensor_fields_before_transfer(self):
        source = _source(LAYER_EVALUATOR)
        region = _method_region(source, "get_batch")

        self.assertIn("def _pack_recurrent_rollout_tensor_arrays", source)
        self.assertIn("cont_features_np, logprobs_np, values_np", region)
        self.assertIn("torch.from_numpy(cont_features_np).to(device)", region)
        self.assertIn("torch.from_numpy(logprobs_np).to(device)", region)
        self.assertIn("torch.from_numpy(values_np).to(device)", region)
        self.assertNotIn("cont_features = torch.stack([", region)
        self.assertNotIn("logprobs = torch.stack([", region)
        self.assertNotIn("values = torch.stack([", region)

    def test_stage1_rollout_reuses_action_scalar_and_builds_device_tensors_directly(self):
        source = _source(LAYER_EVALUATOR)
        worker_region = _method_region(source, "_stage1_collect_episode_in_worker")
        marker = "if not _handled_via_parallel:\n                    # GTrXL Rollout"
        fallback_region = source.split(marker, 1)[1].split("buffer.end_episode()", 1)[0]

        self.assertIn("prev_g_idx = SOS_TOKEN_GELU", worker_region)
        self.assertIn("gelu_action_idx = int(gelu_action.item())", worker_region)
        self.assertIn("env.step(gelu_action_idx)", worker_region)
        self.assertIn("rollout.actions_g.append(gelu_action_idx)", worker_region)
        self.assertIn("cont_feat_record, dtype=torch.float32, device=device", worker_region)
        self.assertIn("gelu_mask_np, dtype=torch.bool, device=device", worker_region)
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
        marker = "if not _handled_via_parallel:\n                    # GTrXL Rollout"
        fallback_region = source.split(marker, 1)[1].split("buffer.end_episode()", 1)[0]
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
        marker = "if not _handled_via_parallel:\n                    # GTrXL Rollout"
        fallback_region = source.split(marker, 1)[1].split("buffer.end_episode()", 1)[0]

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


if __name__ == "__main__":
    unittest.main()
