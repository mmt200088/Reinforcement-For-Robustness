import json
from pathlib import Path
import tempfile
import unittest


def _group(name, *, loss, acc, f1, treatment):
    steps = []
    for step_idx, block_idx in enumerate((2, 4, 5)):
        fused = int(treatment and block_idx in (2, 5))
        steps.append({
            "step_idx": step_idx,
            "layer_idx": 0,
            "block_idx": block_idx,
            "graph_key": f"block{block_idx}",
            "option_id": fused,
            "map_option_id": fused,
            "policy_option_index": 0,
            "k_index": 3,
            "k_value": 13,
            "valid": True,
            "fusion_count_replan": fused,
            "boosted": bool(fused),
        })
    return {
        "name": name,
        "metrics": {
            "loss_mean": loss,
            "loss_std": 0.01,
            "metric1_mean": acc,
            "metric1_std": 0.02,
            "metric2_mean": f1,
            "metric2_std": 0.03,
        },
        "fusion_total": 2 if treatment else 0,
        "fusion_by_block": {"2": 1, "4": 0, "5": 1} if treatment else {"2": 0, "4": 0, "5": 0},
        "k_distribution": {"13": 3},
        "step_records": steps,
    }


def _payload(gelu, *, treatment_loss):
    return {
        "stage1_gelu": gelu,
        "stage1_softmax": [6] * len(gelu),
        "repeat": 5,
        "probe_size": 408,
        "install_path": "evaluate_step -> commit_step -> step",
        "group_results": [
            _group("all_fusion0", loss=0.40, acc=0.80, f1=0.79, treatment=False),
            _group(
                "block2_block5_all_layers_fusionmax",
                loss=treatment_loss,
                acc=0.82,
                f1=0.81,
                treatment=True,
            ),
        ],
    }


class FixedB2B5ComparisonReportTest(unittest.TestCase):
    def test_summary_computes_treatment_minus_control_and_gates(self):
        from scripts.render_fixed_b2b5_comparison_report import build_summary

        summary = build_summary(
            stage1_best=_payload([1] * 12, treatment_loss=0.38),
            gelu4=_payload([4] * 12, treatment_loss=0.42),
            source_commit="abc123",
        )

        best = summary["pairs"][0]
        self.assertAlmostEqual(best["deltas"]["loss"], -0.02)
        self.assertAlmostEqual(best["deltas"]["accuracy"], 0.02)
        self.assertEqual(best["treatment"]["fusion_by_block"], {"2": 1, "4": 0, "5": 1})
        self.assertTrue(best["gates"]["all_steps_valid"])
        self.assertTrue(best["gates"]["k_is_13_everywhere"])
        self.assertTrue(best["gates"]["treatment_is_b2_b5_one_b4_zero"])

    def test_cli_writes_html_with_human_readable_layer_block_decisions(self):
        from scripts.render_fixed_b2b5_comparison_report import main

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            best = root / "best.json"
            gelu4 = root / "gelu4.json"
            output_json = root / "summary.json"
            output_html = root / "report.html"
            best.write_text(json.dumps(_payload([1] * 12, treatment_loss=0.38)), encoding="utf-8")
            gelu4.write_text(json.dumps(_payload([4] * 12, treatment_loss=0.42)), encoding="utf-8")

            rc = main([
                "--stage1-best-json", str(best),
                "--gelu4-json", str(gelu4),
                "--source-commit", "abc123",
                "--output-json", str(output_json),
                "--output-html", str(output_html),
            ])
            html = output_html.read_text(encoding="utf-8")

        self.assertEqual(rc, 0)
        self.assertIn("Stage-1 best GELU", html)
        self.assertIn("GELU degree 4", html)
        self.assertIn("Layer 0", html)
        self.assertIn("Block 2", html)
        self.assertIn("map option 1", html)
        self.assertIn("K=13", html)
        self.assertIn("boosted", html)


if __name__ == "__main__":
    unittest.main()
