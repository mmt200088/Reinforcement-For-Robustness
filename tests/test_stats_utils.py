import math
import unittest

from tests.source_inspection_utils import function_names, source_text
from stats_utils import (
    fraction_true,
    mean_from_total,
    mean_or_default,
    mean_or_none,
    median_sorted,
    ratio_or_default,
    safe_div_or_none,
)


class StatsUtilsTest(unittest.TestCase):
    def test_mean_variants(self):
        self.assertIsNone(mean_or_none([]))
        self.assertEqual(mean_or_none([1.0, 2.0, 3.0]), 2.0)
        self.assertEqual(mean_or_none(value for value in [1.0, 2.0, 3.0]), 2.0)
        self.assertEqual(mean_or_default([], default=7.5), 7.5)
        self.assertTrue(math.isnan(mean_or_default([], default=float("nan"))))

    def test_mean_or_none_streams_without_materializing_float_list(self):
        text = source_text("stats_utils.py")
        self.assertNotIn("vals = [float(value) for value in values]", text)
        self.assertNotIn("len(vals)", text)

    def test_count_based_ratios(self):
        self.assertEqual(mean_from_total(9.0, 3), 3.0)
        self.assertEqual(mean_from_total(9.0, 0), 0.0)
        self.assertEqual(ratio_or_default(2, 4), 0.5)
        self.assertEqual(ratio_or_default(2, 0, default=-1.0), -1.0)

    def test_safe_div_or_none_requires_positive_denominator(self):
        self.assertEqual(safe_div_or_none(6.0, 3.0), 2.0)
        self.assertIsNone(safe_div_or_none(6.0, 0.0))
        self.assertIsNone(safe_div_or_none(6.0, -1.0))

    def test_fraction_true(self):
        self.assertEqual(fraction_true([True, False, True]), 2.0 / 3.0)
        self.assertEqual(fraction_true([], default=0.25), 0.25)

    def test_median_sorted(self):
        self.assertEqual(median_sorted([], default=-1.0), -1.0)
        self.assertEqual(median_sorted([1.0]), 1.0)
        self.assertEqual(median_sorted([1.0, 3.0]), 2.0)
        self.assertEqual(median_sorted([1.0, 3.0, 5.0]), 3.0)


class StatsUtilsStaticGuardTest(unittest.TestCase):
    def test_known_report_scripts_use_shared_stats_helpers(self):
        expected_imports = {
            "scripts/stage2_reward_probe_scaling_report.py": "from stats_utils import median_sorted",
            "scripts/stage1_parallel_report.py": "from stats_utils import safe_div_or_none",
            "scripts/stage1_approx_reuse_benchmark.py": "from stats_utils import mean_or_default",
            "scripts/blb_f0_scan_feasible_domain.py": "from stats_utils import mean_or_none, ratio_or_default",
        }
        forbidden = {
            "_mean",
            "_mean_or_none",
            "_mean_seconds",
            "_mean_counts",
            "_median_sorted",
            "_safe_div",
            "_frac",
            "_frac_counts",
        }
        for rel_path, import_line in expected_imports.items():
            with self.subTest(path=rel_path):
                text = source_text(rel_path)
                self.assertIn(import_line, text)
                self.assertFalse(forbidden & function_names(rel_path))


if __name__ == "__main__":
    unittest.main()
