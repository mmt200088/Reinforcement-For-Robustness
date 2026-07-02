import ast
import unittest
from pathlib import Path


class ApplyPrecisionBoostTest(unittest.TestCase):
    def test_main_does_not_materialize_unused_preboost_snapshot(self):
        source = Path("scripts/blb_apply_precision_boost.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        before_assignments = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "before":
                        before_assignments.append(node)

        self.assertEqual(before_assignments, [])


if __name__ == "__main__":
    unittest.main()

