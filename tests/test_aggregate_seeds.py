import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import aggregate_seeds


def _write_json(path: Path, payload: dict, mtime: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(path, (mtime, mtime))


class AggregateSeedsFinalEvalTest(unittest.TestCase):
    def test_read_final_eval_results_streams_latest_json_without_recursive_glob(self):
        with tempfile.TemporaryDirectory() as td:
            persistent_dir = Path(td)
            _write_json(
                persistent_dir / "final_eval" / "old" / "blb_action_final_eval_results_old.json",
                {"candidate_results": [{"loss": 2.0}]},
                mtime=100,
            )
            _write_json(
                persistent_dir / "nested" / "new" / "blb_action_final_eval_results_new.json",
                {"candidate_results": [{"loss": 1.0}]},
                mtime=200,
            )

            with mock.patch.object(
                aggregate_seeds.glob,
                "glob",
                side_effect=AssertionError("recursive glob should not be used"),
            ):
                payload = aggregate_seeds._read_final_eval_results(str(persistent_dir))

        self.assertEqual(payload["candidate_results"][0]["loss"], 1.0)


if __name__ == "__main__":
    unittest.main()
