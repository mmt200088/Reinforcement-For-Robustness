import math
import importlib.util
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from blb_stage2_rl import eval_metrics


class SharedEvalMetricsTest(unittest.TestCase):
    def test_sample_weighted_mean_uses_example_counts(self):
        self.assertAlmostEqual(
            eval_metrics.sample_weighted_mean([0.0, 10.0], [4, 1]),
            2.0,
        )

    def test_mrpc_metric2_is_weighted_f1_from_complete_trial(self):
        labels = np.asarray([0, 1, 1, 1, 1], dtype=int)
        preds = np.asarray([0, 1, 0, 0, 1], dtype=int)

        metric1, metric2 = eval_metrics.metric_pair_for_dataset(
            "mrpc",
            labels,
            preds,
            predictions_are_classes=True,
        )
        self.assertAlmostEqual(metric1, 0.6)
        self.assertAlmostEqual(metric2, 0.6333333333333333)

        trial = eval_metrics.finalize_probe_trial_metrics(
            losses=[0.2, 0.8],
            m1s=[0.5, 1.0],
            m2s=[0.5, 1.0],
            counts=[2, 3],
            metric_profile="mrpc",
            is_regression=False,
            preds=[preds[:2], preds[2:]],
            labels=[labels[:2], labels[2:]],
        )
        self.assertIsNotNone(trial)
        loss, acc, f1 = trial
        self.assertAlmostEqual(loss, 0.56)
        self.assertAlmostEqual(acc, 0.8)
        self.assertAlmostEqual(f1, 0.6333333333333333)

    def test_repeat_summary_uses_population_stats(self):
        summary = eval_metrics.summarize_eval_trials([
            {"loss": 1.0, "p": 2.0, "s": 3.0, "time_ms": 10.0},
            {"loss": 3.0, "p": 4.0, "s": 7.0, "time_ms": 20.0},
        ])
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["loss_mean"], 2.0)
        self.assertAlmostEqual(summary["loss_std"], 1.0)
        self.assertAlmostEqual(summary["p_mean"], 3.0)
        self.assertAlmostEqual(summary["s_std"], 2.0)
        self.assertAlmostEqual(summary["time_mean_ms"], 15.0)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch unavailable")
    def test_env_and_probe_runner_use_the_shared_probe_finalizer(self):
        from blb_stage2_rl.env import _finalize_probe_trial_metrics
        from blb_stage2_rl.probe_runner import _finalize_probe_trial_metrics_local

        self.assertIs(
            _finalize_probe_trial_metrics,
            eval_metrics.finalize_probe_trial_metrics,
        )
        self.assertIs(
            _finalize_probe_trial_metrics_local,
            eval_metrics.finalize_probe_trial_metrics,
        )


if __name__ == "__main__":
    unittest.main()
