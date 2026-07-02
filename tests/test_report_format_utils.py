import ast
import math
import pathlib
import unittest

from report_format_utils import format_float, html_table, metric_float


class ReportFormatUtilsTest(unittest.TestCase):
    def test_html_table_escapes_cells_by_default(self):
        out = html_table(["<h>"], [["<b>x</b>", "plain"]])

        self.assertIn("&lt;h&gt;", out)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", out)
        self.assertNotIn("<td><b>x</b></td>", out)

    def test_html_table_allows_intentional_html_cells(self):
        out = html_table(["name"], [["<span class='changed'>ok</span>"]], allow_html_cells=True)

        self.assertIn("<td><span class='changed'>ok</span></td>", out)

    def test_format_float_preserves_report_conventions(self):
        self.assertEqual(format_float(None), "")
        self.assertEqual(format_float(1.2345678), "1.234568")
        self.assertEqual(format_float(1.2345678, digits=4), "1.2346")
        self.assertEqual(format_float(float("nan")), "nan")
        self.assertEqual(format_float("text"), "text")

    def test_metric_float_returns_default_on_missing_or_bad_value(self):
        self.assertEqual(metric_float({"loss": "0.25"}, "loss"), 0.25)
        self.assertEqual(metric_float({}, "loss", default=3.0), 3.0)
        self.assertTrue(math.isnan(metric_float({"loss": object()}, "loss", default=math.nan)))


class ReportFormatStaticGuardTest(unittest.TestCase):
    def _function_names(self, rel_path: str) -> set[str]:
        repo = pathlib.Path(__file__).resolve().parents[1]
        tree = ast.parse((repo / rel_path).read_text(encoding="utf-8"))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_shared_report_helpers_are_used_by_known_report_scripts(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        expected_imports = {
            "scripts/run_fusion_count_action_eval.py": "from report_format_utils import html_table, metric_float",
            "scripts/run_fusion_count_action_eval_rlpath.py": (
                "from report_format_utils import format_float, html_table, metric_float"
            ),
            "scripts/report_fusion_count_map.py": "from report_format_utils import html_table",
            "scripts/stage2_reward_probe_scaling_report.py": "from report_format_utils import format_float",
        }
        for rel_path, import_line in expected_imports.items():
            with self.subTest(path=rel_path):
                text = (repo / rel_path).read_text(encoding="utf-8")
                self.assertIn(import_line, text)
                function_names = self._function_names(rel_path)
                self.assertNotIn("_html_table", function_names)
                self.assertNotIn("_fmt", function_names)
                if not rel_path.endswith("stage2_reward_probe_scaling_report.py"):
                    self.assertNotIn("_metric", function_names)


if __name__ == "__main__":
    unittest.main()
