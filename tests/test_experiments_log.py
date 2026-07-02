import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import experiments_log


class ExperimentsLogTest(unittest.TestCase):
    def test_query_streams_registry_without_load_records(self):
        with tempfile.TemporaryDirectory() as td:
            registry = Path(td) / "registry.jsonl"
            registry.write_text(
                json.dumps({
                    "run_id": "run-a",
                    "registered_at": "2026-07-02T00:00:00",
                    "dataset": "mrpc",
                    "algorithm": "rl",
                    "status": "complete",
                    "best_reward": 0.5,
                }) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                experiments_log,
                "_load_records",
                side_effect=AssertionError("query should stream registry rows"),
            ):
                rows = experiments_log._query(dataset="mrpc", registry_path=str(registry))

        self.assertEqual([row["run_id"] for row in rows], ["run-a"])

    def test_rebuild_index_streams_registry_without_load_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry.jsonl"
            index = root / "index.md"
            registry.write_text(
                json.dumps({
                    "run_id": "run-a",
                    "registered_at": "2026-07-02T00:00:00",
                    "dataset": "mrpc",
                    "algorithm": "rl",
                    "status": "complete",
                    "best_reward": 0.5,
                }) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                experiments_log,
                "_load_records",
                side_effect=AssertionError("rebuild should stream registry rows"),
            ):
                experiments_log._rebuild_index(str(registry), str(index))

            text = index.read_text(encoding="utf-8")

        self.assertIn("run-a", text)

    def test_git_info_bounds_git_commands_with_timeout(self):
        calls = []

        def fake_check_output(cmd, **kwargs):
            calls.append((cmd, kwargs))
            if cmd[:2] == ["git", "rev-parse"]:
                return b"abc123\n"
            if cmd[:2] == ["git", "status"]:
                return b""
            raise AssertionError(f"unexpected command: {cmd}")

        with mock.patch.object(experiments_log.subprocess, "check_output", fake_check_output):
            info = experiments_log._git_info()

        self.assertEqual(info["git_commit"], "abc123")
        self.assertFalse(info["git_dirty"])
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(kwargs.get("timeout") == 5 for _, kwargs in calls))


if __name__ == "__main__":
    unittest.main()
