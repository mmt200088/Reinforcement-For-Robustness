import ast
import pathlib
import unittest

from numeric_parse_utils import parse_first_float


class NumericParseUtilsTest(unittest.TestCase):
    def test_parse_first_float(self):
        self.assertIsNone(parse_first_float(None))
        self.assertEqual(parse_first_float(3), 3.0)
        self.assertEqual(parse_first_float(2.5), 2.5)
        self.assertEqual(parse_first_float(" util=91.5 % "), 91.5)
        self.assertEqual(parse_first_float("memory -12.25 MiB"), -12.25)
        self.assertIsNone(parse_first_float("no number"))


class NumericParseStaticGuardTest(unittest.TestCase):
    def _function_names(self, rel_path: str) -> set[str]:
        repo = pathlib.Path(__file__).resolve().parents[1]
        tree = ast.parse((repo / rel_path).read_text(encoding="utf-8"))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    def test_gpu_report_scripts_use_shared_float_parser(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        for rel_path in (
            "scripts/gpu_utilization_report.py",
            "scripts/stage2_reward_probe_scaling_report.py",
        ):
            with self.subTest(path=rel_path):
                text = (repo / rel_path).read_text(encoding="utf-8")
                self.assertIn("from numeric_parse_utils import parse_first_float", text)
                self.assertNotIn("FLOAT_RE = re.compile", text)
                self.assertNotIn("_float_value", self._function_names(rel_path))


if __name__ == "__main__":
    unittest.main()
