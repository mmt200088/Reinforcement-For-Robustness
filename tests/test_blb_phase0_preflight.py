import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import blb_phase0_preflight


class Phase0PreflightTest(unittest.TestCase):
    def test_write_phase0_reports_reuses_single_repo_walk_for_grep(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "runner.py"
            src.write_text("class BLBStage2RLRunner:\n    pass\n", encoding="utf-8")
            calls = 0
            original_iter = blb_phase0_preflight.iter_repo_files

            def counted_iter(repo_root):
                nonlocal calls
                calls += 1
                yield from original_iter(repo_root)

            with mock.patch.object(blb_phase0_preflight, "iter_repo_files", counted_iter):
                paths = blb_phase0_preflight.write_phase0_reports(root, reports_dir="reports")

            self.assertEqual(calls, 1)
            grep_text = Path(paths["blb_entrypoints_grep"]).read_text(encoding="utf-8")
            self.assertIn("runner.py:1:class BLBStage2RLRunner:", grep_text)

    def test_grep_entrypoints_streams_files_without_read_text(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "runner.py"
            src.write_text("class BLBStage2RLRunner:\n    pass\n", encoding="utf-8")

            with mock.patch.object(
                Path,
                "read_text",
                side_effect=AssertionError("entrypoint grep should stream files"),
            ):
                matches = blb_phase0_preflight._grep_entrypoints(root.resolve())

        self.assertEqual(matches, ["runner.py:1:class BLBStage2RLRunner:"])


if __name__ == "__main__":
    unittest.main()
