"""Static guard for the Stage-2 action-to-model cfg write-back path."""
from __future__ import annotations

import pathlib
import unittest


class Stage2EvalSinglePathStaticTest(unittest.TestCase):
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

        self.assertIn("from json_utils import stable_json_hash", candidate_store)
        self.assertIn("from json_utils import stable_json_hash", action_mask)
        self.assertRegex(
            fusion_common,
            r"from json_utils import .*\bstable_json_hash\b.*\bstable_json_key\b",
        )
        self.assertIn("from json_utils import stable_json_hash", f0_scan)
        self.assertIn("from json_utils import stable_json_hash", registry)
        self.assertNotIn("def _stable_json", candidate_store)
        self.assertNotIn("def _stable_mask_payload", action_mask)
        self.assertNotIn("def stable_json_hash", fusion_common)
        self.assertNotIn("def _sha256_json", f0_scan)


if __name__ == "__main__":
    unittest.main()
