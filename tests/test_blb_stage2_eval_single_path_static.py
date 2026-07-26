"""Static guard for the Stage-2 action-to-model cfg write-back path."""
from __future__ import annotations

import pathlib
import unittest


class Stage2EvalSinglePathStaticTest(unittest.TestCase):
    def test_model_eval_routes_use_canonical_action_materialization(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = (repo / "blb_stage2_rl" / "env.py").read_text(encoding="utf-8")
        sequential = (repo / "blb_stage2_rl" / "sequential_env.py").read_text(
            encoding="utf-8"
        )
        paean = (repo / "Paean" / "blb_action_eval.py").read_text(encoding="utf-8")
        glue = (repo / "generate_glue_submission.py").read_text(encoding="utf-8")

        self.assertIn("materialize_action_for_model", env)
        self.assertNotIn("evaluate_action_for_cost(", env)
        self.assertNotIn("apply_optimizer_outputs_to_cfgs(", env)
        self.assertIn("materialize_decoded_action", sequential)
        self.assertNotIn("apply_optimizer_outputs_to_cfgs(", sequential)
        self.assertIn("materialize_decoded_action", paean)
        self.assertNotIn("apply_optimizer_outputs_to_cfgs(", paean)
        self.assertIn("materialize_decoded_action", glue)
        self.assertNotIn("_apply_optimizer_outputs_to_decoded(", glue)

    def test_layer0_block1_k_is_materialized_installed_and_verified(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        action_space = (
            repo / "blb_stage2_rl" / "action_space.py"
        ).read_text(encoding="utf-8")
        bridge = (repo / "blb_rl_bridge.py").read_text(encoding="utf-8")
        paean = (
            repo / "Paean" / "blb_action_eval.py"
        ).read_text(encoding="utf-8")

        self.assertIn("noise_enabled=(li != 0)", action_space)
        self.assertIn('"output_truncation_k":', action_space)
        self.assertNotIn(
            "li: cfg for li, cfg in block1_cfgs.items() if int(li) != 0",
            bridge,
        )
        self.assertIn('"block1": expected_all', paean)

    def test_install_auditors_consume_the_canonical_materialized_config(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        for relative in (
            "scripts/blb_verify_noise_install.py",
            "scripts/blb_verify_boosted_install.py",
        ):
            text = (repo / relative).read_text(encoding="utf-8")
            self.assertIn("materialize_decoded_action", text, relative)
            self.assertNotIn("apply_optimizer_output_to_cfg(", text, relative)

    def test_install_cache_uses_final_config_fingerprint_not_flat_action_hash(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = (repo / "blb_stage2_rl" / "env.py").read_text(encoding="utf-8")

        self.assertIn("_installed_config_fingerprint", env)
        self.assertNotIn("_installed_action_hash", env)
        self.assertIn("final_config_fingerprint", env)

    def test_training_diagnostics_persist_post_replan_install_identity(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        diagnostics = (repo / "blb_stage2_rl" / "diagnostics.py").read_text(
            encoding="utf-8"
        )
        sequential = (repo / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8"
        )
        layerwise = (repo / "blb_stage2_rl" / "layerwise_runner.py").read_text(
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
        handler = (repo / "function_handler.py").read_text(encoding="utf-8")

        self.assertIn("def _apply_configured_truncation(", handler)
        self.assertEqual(
            handler.count("= _apply_configured_truncation("),
            5,
            "Blocks 1-5 must each execute K through the same backend dispatcher",
        )

    def test_block2_hook_receives_the_materialized_config_it_executes(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        handler = (repo / "function_handler.py").read_text(encoding="utf-8")
        start = handler.index("def _make_block2_qkt_merge_hook(")
        end = handler.index("):", start)
        signature = handler[start:end]

        self.assertIn("truncation_cfg", signature)
        self.assertNotIn("output_truncation_k", signature)
        self.assertNotIn("output_truncation_mode", signature)

    def test_block2_config_builder_exposes_k_and_backend_fields(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        handler = (repo / "function_handler.py").read_text(encoding="utf-8")
        start = handler.index("def make_block2_default_config(")
        end = handler.index(") -> \"Block2NoiseConfig\":", start)
        signature = handler[start:end]

        self.assertIn("output_truncation_k", signature)
        self.assertIn("output_truncation_mode", signature)
        self.assertNotIn("truncation_cfg", signature)

    def test_truncation_backend_is_explicitly_wired_and_defaults_to_binary(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        launcher = (repo / "llama_7B_LayerImportance.sh").read_text(encoding="utf-8")
        evaluator = (repo / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        runner = (repo / "blb_stage2_rl" / "runner.py").read_text(encoding="utf-8")
        sequential = (repo / "blb_stage2_rl" / "sequential_runner.py").read_text(
            encoding="utf-8"
        )
        substage = (repo / "blb_stage2_rl" / "substage_runner.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('BLB_V3_TRUNCATION_BACKEND="binary"', launcher)
        self.assertIn("--blb-v3-truncation-backend)", launcher)
        self.assertIn("--blb_v3_truncation_backend", launcher)
        self.assertIn("blb_v3_truncation_backend='binary'", evaluator)
        self.assertIn("self.blb_v3_truncation_backend", evaluator)
        self.assertIn('truncation_backend: str = "binary"', runner)
        for text in (runner, sequential, substage):
            self.assertIn("truncation_backend=train_cfg.truncation_backend", text)
            self.assertIn("truncation_ring_bits=train_cfg.truncation_ring_bits", text)
            self.assertIn(
                "truncation_source_fractional_bits=(\n",
                text,
            )

        glue = (repo / "generate_glue_submission.py").read_text(encoding="utf-8")
        self.assertIn('--blb_truncation_backend', glue)
        self.assertIn('truncation_backend: str = "binary"', glue)
        self.assertIn("truncation_backend=truncation_backend", glue)

    def test_executable_eval_paths_use_shared_optimizer_writeback_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        checked = [
            repo / "blb_stage2_rl" / "env.py",
            repo / "blb_stage2_rl" / "sequential_env.py",
            repo / "Paean" / "blb_action_eval.py",
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
            self.assertIn(
                "apply_optimizer_outputs_to_cfgs",
                text,
                f"{path.relative_to(repo)} must delegate optimizer write-back "
                "through the shared Stage-2 helper",
            )
        self.assertEqual(offenders, [])

    def test_paean_final_eval_does_not_forward_unapplied_replan_cfgs(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        text = (repo / "Paean" / "blb_action_eval.py").read_text(encoding="utf-8")
        self.assertIn("optimizer_invalid_chain", text)
        self.assertIn("replan_config_not_fully_applied", text)
        self.assertIn("skipped_forward:{skip_reason}", text)

    def test_installed_model_forward_paths_use_shared_inference_eval(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        layer_eval = (repo / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        env = (repo / "blb_stage2_rl" / "env.py").read_text(encoding="utf-8")
        probe = (repo / "blb_stage2_rl" / "probe_runner.py").read_text(encoding="utf-8")

        self.assertIn("run_installed_model_on_dataloader", layer_eval)
        self.assertIn("run_installed_probe_trial", env)
        self.assertIn("run_installed_probe_trial", probe)
        self.assertNotIn("def _compute_metrics_on_batch", env)
        self.assertNotIn("def _compute_metrics_on_batch_local", probe)
        self.assertNotIn("model(**kwargs)", env)
        self.assertNotIn("model(**kwargs)", probe)

    def test_repeat_evaluation_payloads_use_shared_pack_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        paean = (repo / "Paean" / "blb_action_eval.py").read_text(encoding="utf-8")
        final_eval = (repo / "final_evaluation_module.py").read_text(encoding="utf-8")
        eval_metrics = (repo / "blb_stage2_rl" / "eval_metrics.py").read_text(encoding="utf-8")

        self.assertIn("def pack_repeat_evaluation(", eval_metrics)
        self.assertIn("pack_repeat_evaluation", paean)
        self.assertIn("pack_repeat_evaluation", final_eval)
        for text in (paean, final_eval):
            self.assertNotIn('"trial": i + 1', text)
            self.assertNotIn("for i, t in enumerate(trials)", text)

    def test_probe_runner_diagnostics_payload_uses_shared_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        env = (repo / "blb_stage2_rl" / "env.py").read_text(encoding="utf-8")
        probe = (repo / "blb_stage2_rl" / "probe_runner.py").read_text(encoding="utf-8")

        self.assertIn("def diagnostics_payload(", probe)
        self.assertIn("diagnostics_payload", env)
        self.assertNotIn('"per_worker_trial_counts": [int(x) for x in diag.', env)
        self.assertNotIn('"per_worker_trial_counts": [int(x) for x in diag_obj.', env)

    def test_report_json_normalization_uses_shared_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        paean = (repo / "Paean" / "blb_action_eval.py").read_text(encoding="utf-8")
        final_eval = (repo / "final_evaluation_module.py").read_text(encoding="utf-8")
        persistence = (repo / "blb_stage2_rl" / "persistence.py").read_text(encoding="utf-8")
        noise_install = (repo / "scripts" / "blb_verify_noise_install.py").read_text(encoding="utf-8")
        layer_noise = (repo / "scripts" / "bert_mrpc_layer_noise_experiment.py").read_text(encoding="utf-8")
        rl_ga = (repo / "rl_ga_compare_runner.py").read_text(encoding="utf-8")

        for text in (paean, final_eval):
            self.assertRegex(text, r"from json_utils import .*\bto_jsonable\b")
            self.assertIn("to_jsonable(", text)
            self.assertNotIn("def _json_ready", text)
        self.assertIn("from json_utils import to_jsonable as _to_jsonable", persistence)
        self.assertNotIn("def _to_jsonable", persistence)
        for text in (noise_install, layer_noise):
            self.assertIn("from json_utils import to_jsonable", text)
            self.assertIn("to_jsonable(", text)
            self.assertNotIn("def _json_safe", text)
            self.assertNotIn("return {str(k): _json_safe", text)
            self.assertNotIn("return {str(key): _json_safe", text)
        self.assertIn("from json_utils import to_jsonable", rl_ga)
        self.assertNotIn("def to_jsonable(value)", rl_ga)
        rlpath = (repo / "scripts" / "run_fusion_count_action_eval_rlpath.py").read_text(encoding="utf-8")
        self.assertRegex(rlpath, r"from json_utils import .*\bto_jsonable\b")
        self.assertNotIn("def _jsonable", rlpath)
        self.assertIn("stage2_rl_episodes=0", rlpath)

        genetic = (repo / "rl_tune_genetic.py").read_text(encoding="utf-8")
        skip_branch = genetic.split("if skip_noise_rl:", 1)[1].split("return {", 1)[0]
        self.assertIn("stage2_rl_episodes = 0", skip_branch)

    def test_json_default_scripts_use_shared_adapter(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        stage2_probe = (
            repo / "experiment" / "scripts" / "noise" / "stage2_probe_subset_size_experiment.py"
        ).read_text(encoding="utf-8")
        softmax_sweep = (
            repo / "experiment" / "scripts" / "noise" / "softmax_v_noise_sweep.py"
        ).read_text(encoding="utf-8")

        self.assertIn("from json_utils import json_default as _json_default", stage2_probe)
        self.assertNotIn("def _json_default", stage2_probe)
        self.assertIn("from json_utils import json_default", softmax_sweep)
        self.assertNotIn("def json_default", softmax_sweep)

    def test_stable_json_hash_callers_use_shared_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        candidate_store = (repo / "blb_stage2_rl" / "candidate_store.py").read_text(encoding="utf-8")
        action_mask = (repo / "blb_stage2_rl" / "action_mask.py").read_text(encoding="utf-8")
        fusion_common = (repo / "scripts" / "fusion_count_action_eval_common.py").read_text(encoding="utf-8")
        f0_scan = (repo / "scripts" / "blb_f0_scan_feasible_domain.py").read_text(encoding="utf-8")
        registry = (repo / "scripts" / "blb_export_action_registry.py").read_text(encoding="utf-8")

        self.assertRegex(candidate_store, r"from json_utils import .*\bstable_json_hash\b")
        self.assertRegex(action_mask, r"from json_utils import .*\bstable_json_hash\b")
        self.assertRegex(
            fusion_common,
            r"from json_utils import .*\bstable_json_hash\b.*\bstable_json_key\b",
        )
        self.assertRegex(f0_scan, r"from json_utils import .*\bstable_json_hash\b")
        self.assertRegex(registry, r"from json_utils import .*\bstable_json_hash\b")
        self.assertNotIn("def _stable_json", candidate_store)
        self.assertNotIn("def _stable_mask_payload", action_mask)
        self.assertNotIn("def stable_json_hash", fusion_common)
        self.assertNotIn("def _sha256_json", f0_scan)


if __name__ == "__main__":
    unittest.main()
