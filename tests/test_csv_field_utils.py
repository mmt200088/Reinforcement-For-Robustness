import ast
import pathlib
import unittest

from csv_field_utils import (
    first_present,
    first_present_by_index,
    first_present_by_lookup,
    normalize_field_name,
    normalized_field_index,
    normalized_field_lookup,
    normalized_row,
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


class CsvFieldUtilsStaticGuardTest(unittest.TestCase):
    def _function_names(self, rel_path: str) -> set[str]:
        repo = pathlib.Path(__file__).resolve().parents[1]
        tree = ast.parse((repo / rel_path).read_text(encoding="utf-8"))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_known_gpu_report_scripts_use_shared_csv_helpers(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
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
                text = (repo / rel_path).read_text(encoding="utf-8")
                self.assertIn(needle, text)
                self.assertFalse(forbidden & self._function_names(rel_path))


if __name__ == "__main__":
    unittest.main()
