import ast
import pathlib
import tempfile
import unittest

from jsonl_utils import iter_jsonl, read_jsonl


class JsonlUtilsTest(unittest.TestCase):
    def test_iter_jsonl_skips_blank_bad_and_non_dict_rows_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1}\n\nbad\n[1, 2]\n{"b": 2}\n', encoding="utf-8")

            rows = list(iter_jsonl(path))

        self.assertEqual(rows, [{"a": 1}, {"b": 2}])

    def test_iter_jsonl_can_raise_with_path_and_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1}\nbad\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"rows\.jsonl:2: invalid JSON"):
                list(iter_jsonl(path, errors="raise"))

    def test_iter_jsonl_can_yield_non_dict_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "rows.jsonl"
            path.write_text('{"a": 1}\n[1, 2]\n', encoding="utf-8")

            rows = list(iter_jsonl(path, dict_only=False))

        self.assertEqual(rows, [{"a": 1}, [1, 2]])

    def test_read_jsonl_missing_ok(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "missing.jsonl"

            self.assertEqual(read_jsonl(path, missing_ok=True), [])
            with self.assertRaises(FileNotFoundError):
                read_jsonl(path)


class JsonlUtilsStaticGuardTest(unittest.TestCase):
    def _function_names(self, rel_path: str) -> set[str]:
        repo = pathlib.Path(__file__).resolve().parents[1]
        tree = ast.parse((repo / rel_path).read_text(encoding="utf-8"))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_known_report_scripts_use_shared_jsonl_reader(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        expected = {
            "scripts/stage2_first10k_monitor.py": "from jsonl_utils import read_jsonl",
            "scripts/stage2_reward_probe_scaling_report.py": "from jsonl_utils import iter_jsonl",
            "scripts/gpu_utilization_report.py": "from jsonl_utils import iter_jsonl",
            "scripts/blb_fusion_ab_compare.py": "from jsonl_utils import iter_jsonl",
        }
        forbidden = {"_read_jsonl"}
        for rel_path, needle in expected.items():
            with self.subTest(path=rel_path):
                text = (repo / rel_path).read_text(encoding="utf-8")
                self.assertIn(needle, text)
                self.assertFalse(forbidden & self._function_names(rel_path))


if __name__ == "__main__":
    unittest.main()
