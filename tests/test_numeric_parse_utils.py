import unittest

from rfr.common.numeric_parse_utils import parse_first_float
from tests.source_inspection_utils import function_names, source_text


class NumericParseUtilsTest(unittest.TestCase):
    def test_parse_first_float(self):
        self.assertIsNone(parse_first_float(None))
        self.assertEqual(parse_first_float(3), 3.0)
        self.assertEqual(parse_first_float(2.5), 2.5)
        self.assertEqual(parse_first_float(" util=91.5 % "), 91.5)
        self.assertEqual(parse_first_float("memory -12.25 MiB"), -12.25)
        self.assertIsNone(parse_first_float("no number"))


if __name__ == "__main__":
    unittest.main()
