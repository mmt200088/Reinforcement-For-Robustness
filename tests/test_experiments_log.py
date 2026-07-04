import builtins
from collections.abc import Mapping
import contextlib
import io
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
    def test_iter_records_delegates_to_shared_jsonl_reader(self):
        with tempfile.TemporaryDirectory() as td:
            registry = Path(td) / "registry.jsonl"
            registry.write_text(
                json.dumps({"run_id": "run-a"}) + "\n\n{bad-json\n"
                + json.dumps({"run_id": "run-b"}) + "\n",
                encoding="utf-8",
            )

            rows = list(experiments_log._iter_records(str(registry)))

        self.assertEqual([row["run_id"] for row in rows], ["run-a", "run-b"])

    def test_append_record_streams_jsonl_without_json_dumps(self):
        with tempfile.TemporaryDirectory() as td:
            registry = Path(td) / "registry.jsonl"
            record = {
                "run_id": "run-stream",
                "dataset": "mrpc",
                "artifact_paths": {"large": "x" * 1000},
            }

            with mock.patch.object(
                experiments_log.json,
                "dumps",
                side_effect=AssertionError("registry appends should stream JSONL rows"),
            ):
                experiments_log._append_record(str(registry), record)

            rows = list(experiments_log._iter_records(str(registry)))

        self.assertEqual(rows, [record])

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

    def test_query_json_output_streams_without_json_dumps(self):
        with tempfile.TemporaryDirectory() as td:
            registry = Path(td) / "registry.jsonl"
            registry.write_text(
                json.dumps({
                    "run_id": "run-json",
                    "registered_at": "2026-07-02T00:00:00",
                    "dataset": "mrpc",
                    "algorithm": "rl",
                    "status": "complete",
                    "best_reward": 0.5,
                }) + "\n",
                encoding="utf-8",
            )
            out = io.StringIO()

            with (
                mock.patch.object(
                    experiments_log.json,
                    "dumps",
                    side_effect=AssertionError("query JSON output should stream to stdout"),
                ),
                contextlib.redirect_stdout(out),
            ):
                rc = experiments_log.main([
                    "query",
                    "--dataset",
                    "mrpc",
                    "--format",
                    "json",
                    "--registry-path",
                    str(registry),
                ])

        self.assertEqual(rc, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual([row["run_id"] for row in payload], ["run-json"])

    def test_query_filters_before_materializing_latest_records(self):
        rows = [
            LazyMaterializationRecord(
                {
                    "run_id": "run-rte",
                    "registered_at": "2026-07-02T00:00:00",
                    "dataset": "rte",
                    "algorithm": "rl",
                    "status": "complete",
                    "best_reward": 0.1,
                    "artifact_paths": {"large": "x" * 1000},
                },
                materialize_ok=False,
                message="non-matching query rows should not be copied into dicts",
            ),
            LazyMaterializationRecord(
                {
                    "run_id": "run-mrpc",
                    "registered_at": "2026-07-02T00:01:00",
                    "dataset": "mrpc",
                    "algorithm": "rl",
                    "status": "complete",
                    "best_reward": 0.9,
                },
                materialize_ok=True,
                message="matching query rows may be materialized for the return value",
            ),
        ]

        with mock.patch.object(experiments_log, "_iter_records", return_value=rows):
            result = experiments_log._query(dataset="mrpc", registry_path="unused.jsonl")

        self.assertEqual([row["run_id"] for row in result], ["run-mrpc"])

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

        with mock.patch.object(experiments_log, "_iter_records", return_value=latest):
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

    def test_rebuild_index_avoids_copying_all_latest_records(self):
        rows = [
            LazyMaterializationRecord(
                {
                    "run_id": "crashed-large-run",
                    "registered_at": "2026-07-02T00:00:00",
                    "dataset": "rte",
                    "algorithm": "rl",
                    "status": "crashed",
                    "best_reward": 100.0,
                    "artifact_paths": {"large": "x" * 1000},
                },
                materialize_ok=False,
                message="index rebuild should not copy every latest record into dicts",
            ),
            LazyMaterializationRecord(
                {
                    "run_id": "complete-mrpc",
                    "registered_at": "2026-07-02T00:01:00",
                    "dataset": "mrpc",
                    "algorithm": "rl",
                    "status": "complete",
                    "best_reward": 0.9,
                },
                materialize_ok=True,
                message="best records may be materialized for summary output",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            index = Path(td) / "index.md"
            with mock.patch.object(experiments_log, "_iter_records", return_value=rows):
                experiments_log._rebuild_index("unused.jsonl", str(index))

            text = index.read_text(encoding="utf-8")

        self.assertIn("crashed-large-run", text)
        self.assertIn("complete-mrpc", text)

    def test_rebuild_index_writes_incrementally_without_joining_full_index(self):
        class CapturingHandle:
            def __init__(self):
                self.parts = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def write(self, text):
                self.parts.append(str(text))
                return len(str(text))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry.jsonl"
            index = root / "index.md"
            registry.write_text(
                "\n".join(
                    json.dumps({
                        "run_id": f"run-{idx}",
                        "registered_at": f"2026-07-02T00:0{idx}:00",
                        "dataset": "mrpc",
                        "algorithm": "rl",
                        "status": "complete",
                        "best_reward": float(idx),
                    })
                    for idx in range(3)
                )
                + "\n",
                encoding="utf-8",
            )
            captured = CapturingHandle()
            original_open = builtins.open

            def fake_open(path, mode="r", *args, **kwargs):
                if Path(path) == index and "w" in mode:
                    return captured
                return original_open(path, mode, *args, **kwargs)

            with mock.patch.object(builtins, "open", fake_open):
                experiments_log._rebuild_index(str(registry), str(index))

        text = "".join(captured.parts)
        self.assertIn("# Experiments index", text)
        self.assertIn("run-2", text)
        self.assertGreater(len(captured.parts), 5)

    def test_rebuild_index_computes_best_during_summary_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            registry = root / "registry.jsonl"
            index = root / "index.md"
            registry.write_text(
                "\n".join([
                    json.dumps({
                        "run_id": "run-a",
                        "registered_at": "2026-07-02T00:00:00",
                        "dataset": "mrpc",
                        "algorithm": "rl",
                        "status": "complete",
                        "best_reward": 0.5,
                        "final_eval": {"loss": 0.3, "metric1": 0.8},
                    }),
                    json.dumps({
                        "run_id": "run-b",
                        "registered_at": "2026-07-02T00:01:00",
                        "dataset": "mrpc",
                        "algorithm": "rl",
                        "status": "complete",
                        "best_reward": 0.9,
                        "final_eval": {"loss": 0.2, "metric1": 0.9},
                    }),
                ])
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                experiments_log,
                "_best_by_dataset",
                side_effect=AssertionError("rebuild should not rescan latest records for best rows"),
            ):
                experiments_log._rebuild_index(str(registry), str(index))

            text = index.read_text(encoding="utf-8")

        self.assertIn("| mrpc | +0.9000 | 0.2000 | 0.9000 | `run-b` |", text)

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
