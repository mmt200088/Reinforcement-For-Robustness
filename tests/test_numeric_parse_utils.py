import unittest

from numeric_parse_utils import parse_first_float
from tests.source_inspection_utils import function_names, source_text


class NumericParseUtilsTest(unittest.TestCase):
    def test_parse_first_float(self):
        self.assertIsNone(parse_first_float(None))
        self.assertEqual(parse_first_float(3), 3.0)
        self.assertEqual(parse_first_float(2.5), 2.5)
        self.assertEqual(parse_first_float(" util=91.5 % "), 91.5)
        self.assertEqual(parse_first_float("memory -12.25 MiB"), -12.25)
        self.assertIsNone(parse_first_float("no number"))


class NumericParseStaticGuardTest(unittest.TestCase):
    def test_gpu_report_scripts_use_shared_float_parser(self):
        for rel_path in (
            "scripts/gpu_utilization_report.py",
            "scripts/stage2_reward_probe_scaling_report.py",
        ):
            with self.subTest(path=rel_path):
                text = source_text(rel_path)
                self.assertIn("from numeric_parse_utils import parse_first_float", text)
                self.assertNotIn("FLOAT_RE = re.compile", text)
                self.assertNotIn("_float_value", function_names(rel_path))


if __name__ == "__main__":
    unittest.main()
