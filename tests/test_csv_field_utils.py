import pathlib
import tempfile
import unittest

from tests.source_inspection_utils import function_names, source_text
from csv_field_utils import (
    first_present,
    first_present_by_index,
    first_present_by_lookup,
    normalize_field_name,
    normalized_field_index,
    normalized_field_lookup,
    normalized_row,
    write_csv_rows,
)


class CsvFieldUtilsTest(unittest.TestCase):
    def test_normalize_field_name(self):
        self.assertEqual(normalize_field_name(" utilization.gpu [%] "), "utilization_gpu")
        self.assertEqual(normalize_field_name("memory used MiB"), "memory_used_mib")

    def test_normalized_row_and_first_present(self):
        row = normalized_row({"GPU Index": "0", "Memory Used MiB": "1024"})
        self.assertEqual(row["gpu_index"], "0")
        self.assertEqual(first_present(row, ["index", "gpu_index"]), "0")
        self.assertIsNone(first_present(row, ["missing"]))

    def test_lookup_preserves_original_field_names(self):
        lookup = normalized_field_lookup(["GPU Index", None, "Util %"])
        row = {"GPU Index": "1", "Util %": "90"}
        self.assertEqual(first_present_by_lookup(row, lookup, ["gpu_index"]), "1")
        self.assertEqual(first_present_by_lookup(row, lookup, ["util"]), "90")

    def test_index_lookup_duplicate_policy_is_explicit(self):
        header = ["GPU Index", "gpu_index", "Other"]
        self.assertEqual(normalized_field_index(header)["gpu_index"], 1)
        self.assertEqual(normalized_field_index(header, keep_first=True)["gpu_index"], 0)

    def test_first_present_by_index(self):
        row = ["0", "4096"]
        lookup = normalized_field_index(["index", "memory.used"])
        self.assertEqual(first_present_by_index(row, lookup, ["gpu_index", "index"]), "0")
        self.assertEqual(first_present_by_index(row, lookup, ["memory_used"]), "4096")
        self.assertIsNone(first_present_by_index(row, lookup, ["missing"]))

    def test_write_csv_rows_projects_fields_and_creates_parent(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "nested" / "rows.csv"

            written = write_csv_rows(
                path,
                [{"b": 2, "a": 1, "extra": 3}, {"a": 4}],
                ["a", "b"],
            )
            text = path.read_text(encoding="utf-8")

        self.assertEqual(written, path)
        self.assertEqual(text, "a,b\n1,2\n4,\n")


class CsvFieldUtilsStaticGuardTest(unittest.TestCase):
    def test_known_gpu_report_scripts_use_shared_csv_helpers(self):
        expected = {
            "scripts/gpu_utilization_report.py": (
                "from csv_field_utils import first_present_by_index, normalized_field_index"
            ),
            "scripts/stage2_reward_probe_scaling_report.py": (
                "from csv_field_utils import first_present_by_index, normalized_field_index"
            ),
        }
        forbidden = {
            "_normalized_fieldnames",
            "_normalized_row",
            "_normalized_field_lookup",
            "_normalized_index_lookup",
            "_normalized_field_index",
            "_first_present",
            "_first_present_by_lookup",
            "_first_present_by_index",
        }
        for rel_path, needle in expected.items():
            with self.subTest(path=rel_path):
                text = source_text(rel_path)
                self.assertIn(needle, text)
                self.assertFalse(forbidden & function_names(rel_path))

    def test_simple_csv_artifact_scripts_use_shared_writer(self):
        expected = {
            "scripts/blb_f0_scan_feasible_domain.py": "from csv_field_utils import write_csv_rows",
            "scripts/bert_mrpc_layer_noise_experiment.py": "from csv_field_utils import write_csv_rows",
        }
        for rel_path, needle in expected.items():
            with self.subTest(path=rel_path):
                text = source_text(rel_path)
                self.assertIn(needle, text)
                self.assertNotIn("def _write_csv(", text)
                self.assertNotIn("def write_csv(", text)


if __name__ == "__main__":
    unittest.main()
