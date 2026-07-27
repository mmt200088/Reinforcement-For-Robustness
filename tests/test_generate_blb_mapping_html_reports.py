import importlib.util
from pathlib import Path
import unittest

from blb_stage2_rl.truncation_levels import K_LEVELS


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "reports" / "generate_blb_mapping_html_reports.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location(
        "generate_blb_mapping_html_reports_for_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerateBlbMappingHtmlReportsTest(unittest.TestCase):
    def test_report_uses_canonical_eight_level_truncation_domain(self):
        report = _load_report_module()

        self.assertEqual(tuple(report.K_LEVELS), tuple(K_LEVELS))
        self.assertEqual(report.LEVELS_BY_KIND["K"], 8)
        self.assertEqual(tuple(report.K_LEVELS).index(13), 3)


if __name__ == "__main__":
    unittest.main()
