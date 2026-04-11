import math
import unittest

import numpy as np


class MetricShapeAndPenaltyRegressionTests(unittest.TestCase):
    def test_ga_std_penalty_remains_finite_for_large_gap(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        penalty = ga_module._penalty_std_upper_bound(
            value=1e12,
            reference=1e-6,
            cap=1e-6,
        )
        self.assertTrue(math.isfinite(penalty))
        self.assertGreater(penalty, 0.0)

    def test_metric_logit_normalization_handles_single_item_classification_batch(self):
        try:
            import layer_importance_evaluator as evaluator_module
        except ImportError as exc:
            self.skipTest(f"layer_importance_evaluator import unavailable: {exc}")

        batch_logits = evaluator_module.LayerImportanceEvaluator._normalize_logits_for_metrics(
            np.asarray([[0.2, 0.8], [0.7, 0.3]], dtype=float),
            expected_batch_size=2,
        )
        tail_logits = evaluator_module.LayerImportanceEvaluator._normalize_logits_for_metrics(
            np.asarray([0.4, 0.6], dtype=float),
            expected_batch_size=1,
        )

        all_preds = []
        all_preds.extend(batch_logits.tolist())
        all_preds.extend(tail_logits.tolist())

        pred_classes = evaluator_module.LayerImportanceEvaluator._logits_to_classes(all_preds)
        self.assertEqual(pred_classes.tolist(), [1, 0, 1])


if __name__ == "__main__":
    unittest.main()
