from pathlib import Path
import unittest

from Paean.config import FinalEvalSettings
from Paean.run_final_eval import configuration_lines, estimate_workload


class FinalEvalWorkloadEstimateTest(unittest.TestCase):
    def test_counts_action_range_product_and_random_controls(self):
        settings = FinalEvalSettings(
            repeat=3,
            random_enabled=True,
            perm_trials=2,
            cost_trials=1,
            budget_trials=0,
            stage1_budget_trials=1,
            stage2_budget_trials=1,
            cost_match_count=4,
            action_ranges=(
                "block3.truncation=8,9,10",
                "layer2.block5.wffn1_sf=18,20",
            ),
        )

        workload = estimate_workload(settings)

        self.assertEqual(workload["action_range_dimensions"], 2)
        self.assertEqual(workload["selected_config_count"], 6)
        self.assertEqual(workload["legacy_random_control_count"], 5)
        self.assertEqual(workload["cost_matched_random_count"], 4)
        self.assertEqual(workload["total_config_count"], 15)
        self.assertEqual(workload["total_repeated_evaluations"], 45)
        self.assertTrue(workload["gpu_parallelism_candidate"])

    def test_default_cost_match_count_is_counted_even_without_legacy_random(self):
        settings = FinalEvalSettings(random_enabled=False, cost_match_count=50)

        workload = estimate_workload(settings)

        self.assertEqual(workload["selected_config_count"], 1)
        self.assertEqual(workload["legacy_random_control_count"], 0)
        self.assertEqual(workload["cost_matched_random_count"], 50)
        self.assertEqual(workload["total_config_count"], 51)
        self.assertTrue(workload["gpu_parallelism_candidate"])

    def test_configuration_lines_include_workload_summary(self):
        settings = FinalEvalSettings(
            repeat=5,
            random_enabled=False,
            cost_match_count=0,
            action_ranges=("block3.truncation=8,9",),
        )

        lines = configuration_lines(
            settings,
            Path("/tmp/paean-final-eval"),
            ["python", "rl_tune.py"],
            include_command=False,
        )
        text = "\n".join(lines)

        self.assertIn("  workload:", text)
        self.assertIn("    selected_configs: 2", text)
        self.assertIn("    total_configs: 2", text)
        self.assertIn("    repeat: 5", text)
        self.assertIn("    total_repeated_evaluations: 10", text)
        self.assertIn("    gpu_parallelism_candidate: true", text)


if __name__ == "__main__":
    unittest.main()
