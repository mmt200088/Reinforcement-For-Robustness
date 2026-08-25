"""Static guard for the Stage-2 action-to-model cfg write-back path."""
from __future__ import annotations

import pathlib
import unittest


class Stage2EvalSinglePathStaticTest(unittest.TestCase):
    def test_model_eval_routes_use_canonical_action_materialization(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = (repo / "src/rfr/search/rl/stage2/env.py").read_text(encoding="utf-8")
        sequential = (repo / "src/rfr/preparation/rescale/block_materialization.py").read_text(
            encoding="utf-8"
        )
        paean = (repo / "src/rfr/evaluation/action_eval.py").read_text(encoding="utf-8")

        self.assertIn("materialize_action_for_model", env)
        self.assertNotIn("evaluate_action_for_cost(", env)
        self.assertNotIn("apply_optimizer_outputs_to_cfgs(", env)
        self.assertIn("materialize_decoded_action", sequential)
        self.assertNotIn("apply_optimizer_outputs_to_cfgs(", sequential)
        self.assertIn("materialize_decoded_action", paean)
        self.assertNotIn("apply_optimizer_outputs_to_cfgs(", paean)

    def test_layer0_block1_k_is_materialized_installed_and_verified(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        action_space = (
            repo / "src/rfr/search/common/action_space.py"
        ).read_text(encoding="utf-8")
        bridge = (repo / "src/rfr/search/runtime/blb_bridge.py").read_text(encoding="utf-8")
        paean = (
            repo / "src/rfr/evaluation/action_eval.py"
        ).read_text(encoding="utf-8")

        self.assertIn("noise_enabled=(li != 0)", action_space)
        self.assertIn('"output_truncation_k":', action_space)
        self.assertNotIn(
            "li: cfg for li, cfg in block1_cfgs.items() if int(li) != 0",
            bridge,
        )
        self.assertIn('"block1": expected_all', paean)

    def test_install_cache_uses_final_config_fingerprint_not_flat_action_hash(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = (repo / "src/rfr/search/rl/stage2/env.py").read_text(encoding="utf-8")

        self.assertIn("_installed_config_fingerprint", env)
        self.assertNotIn("_installed_action_hash", env)
        self.assertIn("final_config_fingerprint", env)

    def test_training_diagnostics_persist_post_replan_install_identity(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        diagnostics = (repo / "src/rfr/search/common/diagnostics.py").read_text(
            encoding="utf-8"
        )
        sequential = (repo / "src/rfr/search/rl/stage2/sequential_runner.py").read_text(
            encoding="utf-8"
        )
        layerwise = (repo / "src/rfr/search/rl/stage2/layerwise_runner.py").read_text(
            encoding="utf-8"
        )
        for field in (
            "terminal_final_config_fingerprint",
            "terminal_materialization_failure_reason",
            "terminal_model_uses_replan_config",
        ):
            self.assertIn(field, diagnostics)
            self.assertIn(field, sequential)
        self.assertIn("final_config_fingerprint", layerwise)
        self.assertIn("materialization_failure_reason", layerwise)
        self.assertIn("model_uses_replan_config", layerwise)

    def test_all_five_blocks_use_the_shared_configured_truncation_executor(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        handler = (repo / "src/rfr/search/runtime/model_handler.py").read_text(encoding="utf-8")

        self.assertIn("def _apply_configured_truncation(", handler)
        self.assertEqual(
            handler.count("= _apply_configured_truncation("),
            5,
            "Blocks 1-5 must each execute K through the same backend dispatcher",
        )

    def test_block2_hook_receives_the_materialized_config_it_executes(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        handler = (repo / "src/rfr/search/runtime/model_handler.py").read_text(encoding="utf-8")
        start = handler.index("def _make_block2_qkt_merge_hook(")
        end = handler.index("):", start)
        signature = handler[start:end]

        self.assertIn("truncation_cfg", signature)
        self.assertNotIn("output_truncation_k", signature)
        self.assertNotIn("output_truncation_mode", signature)

    def test_block2_config_builder_exposes_k_and_backend_fields(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        handler = (repo / "src/rfr/search/runtime/model_handler.py").read_text(encoding="utf-8")
        start = handler.index("def make_block2_default_config(")
        end = handler.index(") -> \"Block2NoiseConfig\":", start)
        signature = handler[start:end]

        self.assertIn("output_truncation_k", signature)
        self.assertIn("output_truncation_mode", signature)
        self.assertNotIn("truncation_cfg", signature)

    def test_truncation_backend_is_explicitly_wired_and_defaults_to_binary(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        launcher = (repo / "llama_7B_LayerImportance.sh").read_text(encoding="utf-8")
        evaluator = (repo / "src/rfr/search/common/evaluator.py").read_text(encoding="utf-8")
        training = (repo / "src/rfr/search/rl/stage2/training.py").read_text(encoding="utf-8")
        sequential = (repo / "src/rfr/search/rl/stage2/sequential_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("--blb_v3_truncation_backend", launcher)
        self.assertNotIn("blb_v3_truncation_backend", evaluator)
        self.assertIn('truncation_backend: str = "binary"', training)
        self.assertIn("truncation_backend=train_cfg.truncation_backend", sequential)
        self.assertIn("truncation_ring_bits=train_cfg.truncation_ring_bits", sequential)
        self.assertIn("truncation_source_fractional_bits=(\n", sequential)

    def test_executable_eval_paths_use_shared_optimizer_writeback_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        checked = [
            repo / "src/rfr/search/rl/stage2/env.py",
            repo / "src/rfr/evaluation/action_eval.py",
        ]
        forbidden = [
            "apply_optimizer_output_to_cfg(",
            "sync_block2_qk_binding(",
            "sync_block2_aux_fresh_binding(",
            "sync_block4_v_mask_binding(",
            "sync_block5_aux_fresh_binding(",
            "_strip_layer_suffix(",
        ]
        offenders = []
        for path in checked:
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    offenders.append(f"{path.relative_to(repo)} contains {token}")
            expected_helper = (
                "materialize_action_for_model"
                if path.name == "env.py"
                else "materialize_decoded_action"
            )
            self.assertIn(expected_helper, text)
        self.assertEqual(offenders, [])

    def test_paean_final_eval_does_not_forward_unapplied_replan_cfgs(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        text = (repo / "src/rfr/evaluation/action_eval.py").read_text(encoding="utf-8")
        self.assertIn("optimizer_invalid_chain", text)
        self.assertIn("replan_config_not_fully_applied", text)
        self.assertIn("skipped_forward:{skip_reason}", text)

    def test_installed_model_forward_paths_use_shared_inference_eval(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        layer_eval = (repo / "src/rfr/search/common/evaluator.py").read_text(encoding="utf-8")
        env = (repo / "src/rfr/search/rl/stage2/env.py").read_text(encoding="utf-8")
        probe = (repo / "src/rfr/search/runtime/probe_runner.py").read_text(encoding="utf-8")

        self.assertIn("run_installed_model_on_dataloader", layer_eval)
        self.assertIn("run_installed_probe_trial", env)
        self.assertIn("run_installed_probe_trial", probe)
        self.assertNotIn("def _compute_metrics_on_batch", env)
        self.assertNotIn("def _compute_metrics_on_batch_local", probe)
        self.assertNotIn("model(**kwargs)", env)
        self.assertNotIn("model(**kwargs)", probe)

    def test_repeat_evaluation_payloads_use_shared_pack_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        paean = (repo / "src/rfr/evaluation/action_eval.py").read_text(encoding="utf-8")
        final_eval = (repo / "src/rfr/evaluation/final_evaluation.py").read_text(encoding="utf-8")
        eval_metrics = (repo / "src/rfr/search/common/eval_metrics.py").read_text(encoding="utf-8")

        self.assertIn("def pack_repeat_evaluation(", eval_metrics)
        self.assertIn("pack_repeat_evaluation", paean)
        self.assertIn("pack_repeat_evaluation", final_eval)
        for text in (paean, final_eval):
            self.assertNotIn('"trial": i + 1', text)
            self.assertNotIn("for i, t in enumerate(trials)", text)

    def test_probe_runner_diagnostics_payload_uses_shared_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = (repo / "src/rfr/search/rl/stage2/env.py").read_text(encoding="utf-8")
        probe = (repo / "src/rfr/search/runtime/probe_runner.py").read_text(encoding="utf-8")

        self.assertIn("def diagnostics_payload(", probe)
        self.assertIn("diagnostics_payload", env)
        self.assertNotIn('"per_worker_trial_counts": [int(x) for x in diag.', env)
        self.assertNotIn('"per_worker_trial_counts": [int(x) for x in diag_obj.', env)

    def test_stable_json_hash_callers_use_shared_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        candidate_store = (repo / "src/rfr/search/common/candidate_store.py").read_text(encoding="utf-8")
        registry = (repo / "src/rfr/preparation/fusion/export_action_registry.py").read_text(encoding="utf-8")

        self.assertRegex(candidate_store, r"from rfr.common.json_utils import .*\bstable_json_hash\b")
        self.assertRegex(registry, r"from rfr.common.json_utils import .*\bstable_json_hash\b")
        self.assertNotIn("def _stable_json", candidate_store)


if __name__ == "__main__":
    unittest.main()
