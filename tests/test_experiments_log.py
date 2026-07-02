import builtins
from collections.abc import Mapping
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import experiments_log


class LazyMaterializationRecord(Mapping):
    def __init__(self, payload, *, materialize_ok, message):
        self.payload = dict(payload)
        self.materialize_ok = bool(materialize_ok)
        self.message = str(message)

    def get(self, key, default=None):
        return self.payload.get(key, default)

    def __getitem__(self, key):
        if not self.materialize_ok:
            raise AssertionError(self.message)
        return self.payload[key]

    def __iter__(self):
        if not self.materialize_ok:
            raise AssertionError(self.message)
        return iter(self.payload)

    def __len__(self):
        return len(self.payload)


class ExperimentsLogTest(unittest.TestCase):
    def test_iter_records_skips_blank_lines_without_strip_copy(self):
        class NoStripLine(str):
            def strip(self, *_args, **_kwargs):
                raise AssertionError("registry reader should not allocate strip() copies")

        class FakeHandle:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([
                    NoStripLine(json.dumps({"run_id": "run-a"}) + "\n"),
                    NoStripLine("   \n"),
                    NoStripLine(json.dumps({"run_id": "run-b"}) + "\n"),
                ])

        def fake_open(*_args, **_kwargs):
            return FakeHandle()

        with mock.patch.object(experiments_log.os.path, "isfile", return_value=True):
            with mock.patch.object(builtins, "open", fake_open):
                rows = list(experiments_log._iter_records("registry.jsonl"))

        self.assertEqual([row["run_id"] for row in rows], ["run-a", "run-b"])

    def test_latest_per_run_id_does_not_materialize_overwritten_records(self):
        rows = experiments_log._latest_per_run_id([
            LazyMaterializationRecord(
                {
                    "run_id": "same-run",
                    "registered_at": "2026-07-02T00:00:00",
                    "notes": "old",
                },
                materialize_ok=False,
                message="overwritten records should not be materialized",
            ),
            LazyMaterializationRecord(
                {
                    "run_id": "same-run",
                    "registered_at": "2026-07-02T00:01:00",
                    "notes": "new",
                },
                materialize_ok=True,
                message="overwritten records should not be materialized",
            ),
        ])

        self.assertEqual(rows, [{
            "run_id": "same-run",
            "registered_at": "2026-07-02T00:01:00",
            "notes": "new",
        }])

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

    def test_query_last_n_does_not_sort_all_latest_records(self):
        class NoFullSortList(list):
            def sort(self, *_args, **_kwargs):
                raise AssertionError("last_n query should not sort every latest record")

        latest = NoFullSortList(
            {
                "run_id": f"run-{idx}",
                "registered_at": f"2026-07-02T00:{idx:02d}:00",
                "dataset": "mrpc",
                "algorithm": "rl",
                "status": "complete",
                "best_reward": float(idx),
            }
            for idx in range(12)
        )

        with mock.patch.object(experiments_log, "_latest_per_run_id", return_value=latest):
            rows = experiments_log._query(dataset="mrpc", last_n=3, registry_path="unused.jsonl")

        self.assertEqual([row["run_id"] for row in rows], ["run-11", "run-10", "run-9"])

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

    def test_git_info_marks_dirty_without_status_strip_copy(self):
        class NoStripStatus(str):
            def strip(self, *_args, **_kwargs):
                raise AssertionError("dirty status check should not allocate strip() copies")

        class FakeOutput:
            def __init__(self, text):
                self.text = text

            def decode(self):
                return self.text

        def fake_check_output(cmd, **_kwargs):
            if cmd[:2] == ["git", "rev-parse"]:
                return b"abc123\n"
            if cmd[:2] == ["git", "status"]:
                return FakeOutput(NoStripStatus(" M experiments/registry.jsonl\n"))
            raise AssertionError(f"unexpected command: {cmd}")

        with mock.patch.object(experiments_log.subprocess, "check_output", fake_check_output):
            info = experiments_log._git_info()

        self.assertEqual(info["git_commit"], "abc123")
        self.assertTrue(info["git_dirty"])

    def test_best_by_dataset_selects_max_reward_without_bucket_sort(self):
        records = [
            {"run_id": "a", "dataset": "mrpc", "status": "complete", "best_reward": 0.1},
            {"run_id": "b", "dataset": "mrpc", "status": "complete", "best_reward": 0.9},
            {"run_id": "c", "dataset": "mrpc", "status": "crashed", "best_reward": 5.0},
            {"run_id": "d", "dataset": "rte", "status": "training_only", "best_reward": 0.4},
            {"run_id": "e", "dataset": "rte", "status": "complete", "best_reward": 0.3},
        ]

        best = experiments_log._best_by_dataset(records)

        self.assertEqual(set(best), {"mrpc", "rte"})
        self.assertEqual(best["mrpc"]["run_id"], "b")
        self.assertEqual(best["rte"]["run_id"], "d")

    def test_best_by_dataset_does_not_materialize_overwritten_best(self):
        best = experiments_log._best_by_dataset([
            LazyMaterializationRecord(
                {
                    "run_id": "old-best",
                    "dataset": "mrpc",
                    "status": "complete",
                    "best_reward": 0.1,
                },
                materialize_ok=False,
                message="overwritten best records should not be materialized",
            ),
            LazyMaterializationRecord(
                {
                    "run_id": "new-best",
                    "dataset": "mrpc",
                    "status": "complete",
                    "best_reward": 0.9,
                },
                materialize_ok=True,
                message="overwritten best records should not be materialized",
            ),
        ])

        self.assertEqual(best["mrpc"]["run_id"], "new-best")


if __name__ == "__main__":
    unittest.main()
