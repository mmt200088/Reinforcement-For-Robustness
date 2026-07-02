import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import blb_phase0_preflight


class Phase0PreflightTest(unittest.TestCase):
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
