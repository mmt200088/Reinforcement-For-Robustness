import ast
import pathlib
import unittest

from text_utils import iter_text_lines


class TextUtilsTest(unittest.TestCase):
    def test_iter_text_lines_preserves_newline_only_split_semantics(self):
        self.assertEqual(list(iter_text_lines("")), [])
        self.assertEqual(list(iter_text_lines("a")), ["a"])
        self.assertEqual(list(iter_text_lines("a\nb\n")), ["a\n", "b\n"])
        self.assertEqual(list(iter_text_lines("a\rb\n")), ["a\rb\n"])


class TextUtilsStaticGuardTest(unittest.TestCase):
    def test_known_in_memory_line_parsers_use_shared_helper(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        for rel_path in (
            "scripts/server_resource_snapshot.py",
            "scripts/stage1_parallel_report.py",
        ):
            with self.subTest(path=rel_path):
                source = (repo / rel_path).read_text(encoding="utf-8")
                tree = ast.parse(source)
                function_names = {
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                self.assertIn("from text_utils import iter_text_lines", source)
                self.assertNotIn("_iter_text_lines", function_names)


if __name__ == "__main__":
    unittest.main()
