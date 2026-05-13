import unittest

from blb_stage2_rl.runner import _baseline_derived_metric_threshold


class BLBThresholdSemanticsTests(unittest.TestCase):
    def test_baseline_derived_threshold_uses_all_max_blb_metric(self):
        result = _baseline_derived_metric_threshold(
            current_threshold=0.0,
            raw_baseline_metric=0.9375,
            all_max_blb_metric=0.8750,
            allowed_drop=0.00492,
        )

        self.assertAlmostEqual(result.threshold, 0.87008)
        self.assertEqual(result.source, "baseline_derived_all_max_blb")
        self.assertAlmostEqual(result.allowed_drop, 0.00492)

    def test_explicit_threshold_is_preserved(self):
        result = _baseline_derived_metric_threshold(
            current_threshold=0.91,
            raw_baseline_metric=0.9375,
            all_max_blb_metric=0.8750,
            allowed_drop=0.00492,
        )

        self.assertAlmostEqual(result.threshold, 0.91)
        self.assertEqual(result.source, "explicit")


if __name__ == "__main__":
    unittest.main()
