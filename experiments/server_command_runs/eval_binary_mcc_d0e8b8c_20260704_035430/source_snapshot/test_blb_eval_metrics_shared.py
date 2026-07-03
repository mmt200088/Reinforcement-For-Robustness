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

    def test_weighted_probe_batch_means_reuses_count_weights(self):
        class SingleUseCounts:
            def __init__(self, values):
                self.values = list(values)
                self.used = False

            def __iter__(self):
                if self.used:
                    raise AssertionError("counts should be converted to weights once")
                self.used = True
                return iter(self.values)

        loss, metric1, metric2 = eval_metrics.weighted_probe_batch_means(
            losses=[0.0, 10.0],
            m1s=[1.0, 0.0],
            m2s=[0.5, 0.0],
            counts=SingleUseCounts([4, 1]),
        )

        self.assertAlmostEqual(loss, 2.0)
        self.assertAlmostEqual(metric1, 0.8)
        self.assertAlmostEqual(metric2, 0.4)

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

    def test_binary_weighted_f1_skips_union_sort(self):
        labels = np.asarray([0, 1, 1, 1, 1], dtype=int)
        preds = np.asarray([0, 1, 0, 0, 1], dtype=int)
        original_union1d = eval_metrics.np.union1d

        def fail_if_called(_left, _right, *args, **kwargs):
            raise AssertionError("binary weighted F1 should not sort class union")

        eval_metrics.np.union1d = fail_if_called
        try:
            metric = eval_metrics.weighted_f1_from_labels(labels, preds)
        finally:
            eval_metrics.np.union1d = original_union1d

        self.assertAlmostEqual(metric, 0.6333333333333333)

    def test_binary_matthews_corrcoef_skips_union_sort(self):
        labels = np.asarray([0, 0, 1, 1], dtype=int)
        preds = np.asarray([0, 1, 1, 1], dtype=int)
        original_union1d = eval_metrics.np.union1d

        def fail_if_called(_left, _right, *args, **kwargs):
            raise AssertionError("binary MCC should not sort class union")

        eval_metrics.np.union1d = fail_if_called
        try:
            metric = eval_metrics.matthews_corrcoef_from_labels(labels, preds)
        finally:
            eval_metrics.np.union1d = original_union1d

        self.assertAlmostEqual(metric, 2.0 / math.sqrt(12.0))

    def test_finalize_probe_trial_metrics_reuses_single_packed_arrays(self):
        labels = np.asarray([0, 1, 1, 1, 1], dtype=int)
        preds = np.asarray([0, 1, 0, 0, 1], dtype=int)
        original_concatenate = eval_metrics.np.concatenate
        original_weighted_f1 = eval_metrics.weighted_f1_from_labels

        def fail_if_called(_arrays, *args, **kwargs):
            raise AssertionError("single packed arrays should not be concatenated")

        def weighted_f1_stub(labels_arg, preds_arg):
            np.testing.assert_array_equal(labels_arg, labels)
            np.testing.assert_array_equal(preds_arg, preds)
            return 0.6333333333333333

        eval_metrics.np.concatenate = fail_if_called
        eval_metrics.weighted_f1_from_labels = weighted_f1_stub
        try:
            trial = eval_metrics.finalize_probe_trial_metrics(
                losses=[0.0],
                m1s=[0.6],
                m2s=[0.6],
                counts=[5],
                metric_profile="mrpc",
                is_regression=False,
                preds=[preds],
                labels=[labels],
            )
        finally:
            eval_metrics.np.concatenate = original_concatenate
            eval_metrics.weighted_f1_from_labels = original_weighted_f1

        self.assertIsNotNone(trial)
        _loss, metric1, metric2 = trial
        self.assertAlmostEqual(metric1, 0.6)
        self.assertAlmostEqual(metric2, 0.6333333333333333)

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

    def test_pack_repeat_evaluation_numbers_trials_and_stats(self):
        packed = eval_metrics.pack_repeat_evaluation(
            [
                {"loss": "1.0", "p": 2, "s": 3, "time_ms": 10},
                {"loss": 3, "p": "4.0", "s": 7, "time_ms": 20},
            ],
            evaluation_mode="unit_repeat",
        )

        self.assertEqual(packed["trials"][0]["trial"], 1)
        self.assertEqual(packed["trials"][1]["trial"], 2)
        self.assertIsInstance(packed["trials"][0]["loss"], float)
        self.assertEqual(packed["stats"]["evaluation_mode"], "unit_repeat")
        self.assertEqual(packed["stats"]["n"], 2)
        self.assertAlmostEqual(packed["stats"]["loss_mean"], 2.0)
        self.assertAlmostEqual(packed["stats"]["time_mean_ms"], 15.0)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "torch unavailable")
    def test_probe_eval_uses_the_shared_installed_inference_module(self):
        from blb_stage2_rl.inference_eval import run_installed_probe_trial

        self.assertTrue(callable(run_installed_probe_trial))


if __name__ == "__main__":
    unittest.main()
