import pathlib
import unittest

from cli_parse_utils import (
    parse_broadcast_int_vector,
    parse_exact_json_int_list,
    parse_int_list_text,
    parse_json_int_list,
    parse_optional_int_list,
    split_int_tokens,
)


class CliParseUtilsTest(unittest.TestCase):
    def test_split_and_parse_int_list_text(self):
        self.assertEqual(split_int_tokens("1, 2;3"), ["1", "2", "3"])
        self.assertEqual(parse_int_list_text("1, 2;3"), [1, 2, 3])
        self.assertEqual(split_int_tokens("1, 2;3", allow_semicolon=False), ["1", "2;3"])
        with self.assertRaises(ValueError):
            parse_int_list_text("1, 2;3", allow_semicolon=False)

    def test_optional_int_list(self):
        self.assertIsNone(parse_optional_int_list(None))
        self.assertIsNone(parse_optional_int_list(""))
        self.assertEqual(parse_optional_int_list("4; 5"), [4, 5])

    def test_json_int_list_with_default_and_errors(self):
        self.assertEqual(parse_json_int_list("", default=[1, 2], name="x"), [1, 2])
        self.assertEqual(parse_json_int_list("[3, \"4\"]", default=[], name="x"), [3, 4])
        with self.assertRaises(SystemExit):
            parse_json_int_list("3", default=[], name="x")

    def test_exact_json_int_list(self):
        self.assertEqual(parse_exact_json_int_list("[1, \"2\"]", name="gelu", length=2), [1, 2])
        with self.assertRaisesRegex(ValueError, "gelu must be a JSON list with 3 entries"):
            parse_exact_json_int_list("[1, 2]", name="gelu", length=3)

    def test_broadcast_int_vector(self):
        self.assertEqual(parse_broadcast_int_vector(None, num_layers=3, default=4), [4, 4, 4])
        self.assertEqual(parse_broadcast_int_vector("2", num_layers=3, default=4), [2, 2, 2])
        self.assertEqual(parse_broadcast_int_vector("[1, 2, 3]", num_layers=3, default=4), [1, 2, 3])
        self.assertEqual(parse_broadcast_int_vector("1;2;3", num_layers=3, default=4), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "degree vector length 2 must be 1 or num_layers=3"):
            parse_broadcast_int_vector([1, 2], num_layers=3, default=4)

    def test_existing_script_wrappers_use_shared_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        checks = {
            "scripts/fusion_count_action_eval_common.py": "from cli_parse_utils import parse_json_int_list",
            "scripts/report_fusion_count_map.py": "from cli_parse_utils import parse_json_int_list",
            "scripts/blb_export_action_registry.py": "from cli_parse_utils import parse_broadcast_int_vector",
            "scripts/blb_make_fusion_fixed_action_config.py": "from cli_parse_utils import parse_exact_json_int_list",
            "scripts/blb_f0_scan_feasible_domain.py": "from cli_parse_utils import parse_optional_int_list",
            "scripts/stage1_parallel_report.py": "from cli_parse_utils import parse_int_list_text, split_int_tokens",
        }
        for rel, needle in checks.items():
            text = (repo / rel).read_text(encoding="utf-8")
            self.assertIn(needle, text)
        self.assertNotIn("text.replace(\";\", \",\").split(\",\")", (repo / "scripts/blb_f0_scan_feasible_domain.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
