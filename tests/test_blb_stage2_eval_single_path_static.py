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


if __name__ == "__main__":
    unittest.main()
