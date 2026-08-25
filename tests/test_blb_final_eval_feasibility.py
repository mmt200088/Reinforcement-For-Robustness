import tempfile
import unittest
from pathlib import Path


class BLBFinalEvalFeasibilityTests(unittest.TestCase):
    def test_loss_is_report_only_for_feasible_result(self):
        from blb_stage2_rl.feasibility import build_final_eval_feasibility

        report = build_final_eval_feasibility(
            optimizer_valid=True,
            decode_ok=True,
            apply_ok=True,
            eval_ok=True,
            acc_mean=0.86,
            f1_mean=0.88,
            acc_std=0.01,
            f1_std=0.01,
            acc_limit=0.85,
            f1_limit=0.87,
            acc_std_limit=0.02,
            f1_std_limit=0.02,
            loss_mean=10.0,
            threshold_source="explicit",
            strict_z=1.0,
        )

        self.assertTrue(report["feasible"])
        self.assertFalse(report["loss_is_hard_constraint"])
        self.assertTrue(report["strict_feasible"])

    def test_unknown_threshold_source_never_claims_feasibility(self):
        from blb_stage2_rl.feasibility import build_final_eval_feasibility

        report = build_final_eval_feasibility(
            optimizer_valid=True,
            decode_ok=True,
            apply_ok=True,
            eval_ok=True,
            acc_mean=0.86,
            f1_mean=0.88,
            acc_std=0.01,
            f1_std=0.01,
            acc_limit=0.85,
            f1_limit=0.87,
            acc_std_limit=0.02,
            f1_std_limit=0.02,
            threshold_source="unknown",
        )

        self.assertIsNone(report["feasible"])
        self.assertTrue(report["diagnostic_feasible"])
        self.assertIn(
            "threshold source unknown",
            report["feasible_unavailable_reason"],
        )

    def test_feasibility_report_writes_json_and_markdown(self):
        from blb_stage2_rl.feasibility import (
            build_final_eval_feasibility,
            write_final_eval_feasibility_report,
        )

        report = build_final_eval_feasibility(
            optimizer_valid=False,
            decode_ok=True,
            apply_ok=False,
            eval_ok=False,
            acc_mean=0.0,
            f1_mean=0.0,
            acc_std=0.0,
            f1_std=0.0,
            acc_limit=0.85,
            f1_limit=0.87,
            acc_std_limit=0.02,
            f1_std_limit=0.02,
            threshold_source="explicit",
        )
        with tempfile.TemporaryDirectory() as td:
            paths = write_final_eval_feasibility_report(report, Path(td))
            self.assertTrue(Path(paths["json"]).exists())
            text = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("loss 非硬约束", text)
            self.assertIn("strict_z", text)

    def test_feasibility_report_uses_shared_json_writer(self):
        source = (Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "feasibility.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("from rfr.common.json_utils import write_json_file", source)
        self.assertIn("write_json_file(json_path", source)
        self.assertNotIn("json.dumps(dict(report)", source)
