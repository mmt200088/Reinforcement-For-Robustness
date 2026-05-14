import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.bert_mrpc_layer_noise_experiment import (
    aggregate_metric_trials,
    build_sigma_grid,
    inject_noise_into_layer_output,
    select_mild_drop_sigma,
)


class TransformerLayerNoiseExperimentTests(unittest.TestCase):
    def test_build_sigma_grid_is_sorted_unique_and_includes_endpoints(self):
        grid = build_sigma_grid(1e-10, 1e-1)
        self.assertEqual(grid[0], 1e-10)
        self.assertEqual(grid[-1], 1e-1)
        self.assertEqual(grid, sorted(set(grid)))
        self.assertIn(1e-4, grid)
        self.assertIn(2e-4, grid)
        self.assertIn(9e-4, grid)

    def test_select_mild_drop_sigma_targets_small_f1_drop(self):
        rows = [
            {"sigma": 1e-5, "f1_mean": 0.910, "acc_mean": 0.880},
            {"sigma": 1e-4, "f1_mean": 0.905, "acc_mean": 0.879},
            {"sigma": 1e-3, "f1_mean": 0.890, "acc_mean": 0.868},
            {"sigma": 1e-2, "f1_mean": 0.810, "acc_mean": 0.790},
        ]
        chosen = select_mild_drop_sigma(
            rows,
            baseline_f1=0.910,
            baseline_acc=0.880,
            target_drop=0.02,
        )
        self.assertEqual(chosen, 1e-3)

    def test_aggregate_metric_trials_reports_mean_and_sample_std(self):
        summary = aggregate_metric_trials([
            {"acc": 0.80, "f1": 0.90},
            {"acc": 0.84, "f1": 0.86},
            {"acc": 0.82, "f1": 0.88},
        ])
        self.assertAlmostEqual(summary["acc_mean"], 0.82)
        self.assertAlmostEqual(summary["f1_mean"], 0.88)
        self.assertAlmostEqual(summary["acc_std"], 0.02)
        self.assertAlmostEqual(summary["f1_std"], 0.02)

    def test_inject_noise_preserves_bert_tuple_structure(self):
        output = ("hidden", "attention")
        result = inject_noise_into_layer_output(output, lambda x: f"noisy:{x}")
        self.assertEqual(result, ("noisy:hidden", "attention"))


if __name__ == "__main__":
    unittest.main()
