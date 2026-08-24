import math
import unittest

from report_format_utils import format_elapsed, format_float, html_table, metric_float, progress_bar
from tests.source_inspection_utils import function_names, source_text


class ReportFormatUtilsTest(unittest.TestCase):
    def test_html_table_escapes_cells_by_default(self):
        out = html_table(["<h>"], [["<b>x</b>", "plain"]])

        self.assertIn("&lt;h&gt;", out)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", out)
        self.assertNotIn("<td><b>x</b></td>", out)

    def test_html_table_allows_intentional_html_cells(self):
        out = html_table(["name"], [["<span class='changed'>ok</span>"]], allow_html_cells=True)

        self.assertIn("<td><span class='changed'>ok</span></td>", out)

    def test_html_table_can_render_row_classes(self):
        out = html_table(["name"], [["a"], ["b"]], row_classes=["ok", "bad"])

        self.assertIn('<tr class="ok">', out)
        self.assertIn('<tr class="bad">', out)

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

    def test_training_log_format_helpers(self):
        self.assertEqual(format_elapsed(65), "1m05s")
        self.assertEqual(format_elapsed(3661), "1h01m01s")
        self.assertEqual(progress_bar(1, 4, width=4), "[█░░░]  25.0%")
        self.assertEqual(progress_bar(5, 4, width=4), "[████] 100.0%")


if __name__ == "__main__":
    unittest.main()
