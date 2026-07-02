import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "scripts" / "server_resource_snapshot.py"


def _load_snapshot_module():
    spec = importlib.util.spec_from_file_location("server_resource_snapshot", SNAPSHOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ServerResourceSnapshotTest(unittest.TestCase):
    def test_parse_nvidia_smi_rows_normalizes_numeric_fields(self):
        snap = _load_snapshot_module()

        rows = snap.parse_nvidia_smi_csv(
            "0, NVIDIA A100-SXM4-40GB, 40960, 1024, 57\n"
            "1, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n"
        )

        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["name"], "NVIDIA A100-SXM4-40GB")
        self.assertEqual(rows[0]["memory_total_mib"], 40960)
        self.assertEqual(rows[0]["memory_used_mib"], 1024)
        self.assertEqual(rows[0]["utilization_gpu_pct"], 57)
        self.assertEqual(rows[1]["utilization_gpu_pct"], 0)

    def test_parse_nvidia_smi_sample_csv_collapses_rows_by_gpu(self):
        snap = _load_snapshot_module()

        rows = snap.parse_nvidia_smi_csv(
            "timestamp,index,name,memory_total_mib,memory_used_mib,utilization_gpu_pct\n"
            "2026/07/02 00:00:00.000,0,NVIDIA A100-SXM4-40GB,40960,1024,57\n"
            "2026/07/02 00:00:00.000,1,NVIDIA A100-SXM4-40GB,40960,0,0\n"
            "2026/07/02 00:00:15.000,0,NVIDIA A100-SXM4-40GB,40960,2048,81\n"
            "2026/07/02 00:00:15.000,1,NVIDIA A100-SXM4-40GB,40960,512,9\n"
        )
        summary = snap._gpu_summary(rows)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["index"], 0)
        self.assertEqual(rows[0]["memory_used_mib"], 2048)
        self.assertEqual(rows[0]["utilization_gpu_pct"], 81)
        self.assertEqual(rows[1]["index"], 1)
        self.assertEqual(rows[1]["memory_used_mib"], 512)
        self.assertEqual(rows[1]["utilization_gpu_pct"], 9)
        self.assertEqual(summary["gpu_count"], 2)
        self.assertEqual(summary["active_gpu_count"], 2)
        self.assertEqual(summary["memory_total_mib"], 81920)
        self.assertEqual(summary["max_utilization_gpu_pct"], 81)

    def test_parse_nvidia_smi_csv_streams_text_lines_without_list(self):
        snap = _load_snapshot_module()
        captured = {}

        def fake_parse_lines(lines):
            captured["is_list"] = isinstance(lines, list)
            captured["rows"] = list(lines)
            return [{"index": 0}]

        with mock.patch.object(snap, "parse_nvidia_smi_lines", fake_parse_lines):
            rows = snap.parse_nvidia_smi_csv("0, GPU, 10, 1, 2\n")

        self.assertEqual(rows, [{"index": 0}])
        self.assertFalse(captured["is_list"])
        self.assertEqual(captured["rows"], ["0, GPU, 10, 1, 2\n"])

    def test_external_commands_are_bounded(self):
        snap = _load_snapshot_module()

        with mock.patch.object(
            snap.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stdout="ok\n"),
        ) as run:
            self.assertEqual(snap._run_command(["git", "status"]), "ok")

        self.assertEqual(run.call_args.kwargs["timeout"], 5.0)

    def test_git_summary_counts_dirty_rows_without_splitlines_list(self):
        snap = _load_snapshot_module()

        class NoSplitlinesStatus:
            def __str__(self):
                return "".join(f" M file_{idx}.py\n" for idx in range(25))

            def splitlines(self):
                raise AssertionError("git status summary should not materialize splitlines()")

        def fake_run_command(cmd, **_kwargs):
            if cmd == ["git", "rev-parse", "HEAD"]:
                return "abc123"
            if cmd == ["git", "rev-parse", "--abbrev-ref", "HEAD"]:
                return "jk_standard_rl"
            if cmd[:2] == ["git", "status"]:
                return NoSplitlinesStatus()
            raise AssertionError(f"unexpected command: {cmd}")

        with mock.patch.object(snap, "_run_command", side_effect=fake_run_command):
            summary = snap._git_summary(Path("/tmp"))

        self.assertEqual(summary["commit"], "abc123")
        self.assertEqual(summary["branch"], "jk_standard_rl")
        self.assertEqual(summary["dirty_file_count"], 25)
        self.assertEqual(len(summary["dirty_examples"]), 20)
        self.assertEqual(summary["dirty_examples"][0], " M file_0.py")
        self.assertEqual(summary["dirty_examples"][-1], " M file_19.py")

    def test_cli_writes_json_and_markdown_from_offline_gpu_csv(self):
        snap = _load_snapshot_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smi = root / "nvidia_smi.csv"
            smi.write_text(
                "0, NVIDIA A100-SXM4-40GB, 40960, 1024, 57\n"
                "1, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n",
                encoding="utf-8",
            )
            out_json = root / "snapshot.json"
            out_md = root / "snapshot.md"

            rc = snap.main(
                [
                    "--root",
                    str(root),
                    "--nvidia-smi-csv",
                    str(smi),
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["gpu_summary"]["gpu_count"], 2)
            self.assertEqual(payload["gpu_summary"]["idle_gpu_count"], 1)
            markdown = out_md.read_text(encoding="utf-8")
            self.assertIn("# Server Resource Snapshot", markdown)
            self.assertIn("GPU count: 2", markdown)
            self.assertIn("idle GPUs: 1", markdown)

    def test_collect_snapshot_streams_offline_gpu_csv(self):
        snap = _load_snapshot_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smi = root / "nvidia_smi.csv"
            smi.write_text(
                "0, NVIDIA A100-SXM4-40GB, 40960, 1024, 57\n"
                "1, NVIDIA A100-SXM4-40GB, 40960, 0, 0\n",
                encoding="utf-8",
            )

            original_read_text = Path.read_text

            def guarded_read_text(path, *args, **kwargs):
                if Path(path) == smi:
                    raise AssertionError("offline GPU CSV should be streamed")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", guarded_read_text):
                snapshot = snap.collect_snapshot(root, nvidia_smi_csv=smi)

            self.assertEqual(snapshot["gpu_summary"]["gpu_count"], 2)
            self.assertEqual(snapshot["gpu_summary"]["idle_gpu_count"], 1)


if __name__ == "__main__":
    unittest.main()
