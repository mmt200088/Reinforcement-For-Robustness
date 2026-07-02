from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "gpu_utilization_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("gpu_utilization_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class GpuUtilizationReportTest(unittest.TestCase):
    def test_summarizes_probe_devices_trials_and_idle_visible_devices(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            episodes = Path(td) / "episodes.jsonl"
            _write_jsonl(
                episodes,
                [
                    {
                        "episode": 0,
                        "terminal_probe_devices": ["cuda:0", "cuda:1"],
                        "terminal_probe_trial_counts": [2, 2],
                        "terminal_probe_wall_seconds": 1.5,
                        "terminal_probe_device_wall_seconds": [1.0, 2.0],
                        "policy_rollout_wall_seconds": 0.5,
                        "per_step_optimizer_wall_seconds": 0.2,
                        "jsonl_write_wall_seconds": 0.1,
                    },
                    {
                        "episode": 1,
                        "terminal_probe_devices": ["cuda:0"],
                        "terminal_probe_trial_counts": [4],
                        "terminal_probe_wall_seconds": 2.5,
                        "policy_rollout_wall_seconds": 0.25,
                        "replan_wall_seconds": 0.4,
                        "report_render_wall_seconds": 0.3,
                    },
                ],
            )

            summary = report.summarize_run(
                episodes,
                visible_devices=["0", "1", "2", "3"],
            )

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["visible_devices"], ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        self.assertEqual(summary["used_probe_devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(summary["idle_visible_devices"], ["cuda:2", "cuda:3"])
        self.assertEqual(summary["probe_trial_counts_by_device"], {"cuda:0": 6, "cuda:1": 2})
        self.assertEqual(summary["probe_episode_counts_by_device"], {"cuda:0": 2, "cuda:1": 1})
        self.assertEqual(summary["probe_wall_seconds_by_device"]["cuda:0"]["mean"], 1.75)
        self.assertEqual(summary["probe_wall_seconds_by_device"]["cuda:1"]["mean"], 2.0)
        self.assertEqual(summary["probe_device_sets"], [["cuda:0"], ["cuda:0", "cuda:1"]])
        self.assertEqual(summary["probe_trial_splits"], [[2, 2], [4]])
        self.assertEqual(summary["terminal_probe_wall_seconds"]["mean"], 2.0)
        self.assertEqual(summary["policy_rollout_wall_seconds"]["mean"], 0.375)
        self.assertAlmostEqual(summary["replan_wall_seconds"]["mean"], 0.3)
        self.assertEqual(summary["hot_path_wall_seconds"]["jsonl_write_wall_seconds"]["mean"], 0.1)
        self.assertEqual(summary["hot_path_wall_seconds"]["report_render_wall_seconds"]["mean"], 0.3)
        self.assertTrue(
            any("visible GPUs were not used by terminal probes" in item for item in summary["warnings"])
        )

    def test_reads_nvidia_smi_csv_and_warns_on_low_utilization(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            episodes = root / "episodes.jsonl"
            _write_jsonl(
                episodes,
                [
                    {
                        "episode": 0,
                        "terminal_probe_devices": ["cuda:0"],
                        "terminal_probe_trial_counts": [4],
                    }
                ],
            )
            smi = root / "nvidia_smi.csv"
            smi.write_text(
                "timestamp,index,utilization.gpu,memory.used\n"
                "2026/07/02 00:00:00.000,0,95 %,12000 MiB\n"
                "2026/07/02 00:00:00.000,1,0 %,500 MiB\n"
                "2026/07/02 00:00:15.000,0,80 %,12200 MiB\n"
                "2026/07/02 00:00:15.000,1,3 %,550 MiB\n",
                encoding="utf-8",
            )

            summary = report.summarize_run(
                episodes,
                nvidia_smi_csv=smi,
                visible_devices=["0", "1"],
            )

        self.assertEqual(summary["gpu_utilization"]["cuda:0"]["max_util_pct"], 95.0)
        self.assertEqual(summary["gpu_utilization"]["cuda:0"]["mean_util_pct"], 87.5)
        self.assertEqual(summary["gpu_utilization"]["cuda:0"]["max_memory_mib"], 12200.0)
        self.assertEqual(summary["gpu_utilization"]["cuda:1"]["max_util_pct"], 3.0)
        self.assertEqual(summary["gpu_utilization"]["cuda:1"]["active_sample_rate"], 0.5)
        self.assertTrue(any("cuda:1 max utilization 3.0%" in item for item in summary["warnings"]))

    def test_nvidia_smi_csv_summary_uses_running_aggregates(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smi = root / "nvidia_smi.csv"
            smi.write_text(
                "timestamp,index,utilization.gpu,memory.used\n"
                "2026/07/02 00:00:00.000,0,20 %,1000 MiB\n"
                "2026/07/02 00:00:01.000,0,40 %,1500 MiB\n",
                encoding="utf-8",
            )

            def fail_mean(_values):
                raise AssertionError("nvidia-smi summary should not materialize samples for statistics.mean")

            with mock.patch.object(statistics, "mean", fail_mean):
                summary = report._load_nvidia_smi_csv(smi)

        self.assertEqual(summary["cuda:0"]["samples"], 2)
        self.assertEqual(summary["cuda:0"]["mean_util_pct"], 30.0)
        self.assertEqual(summary["cuda:0"]["max_util_pct"], 40.0)
        self.assertEqual(summary["cuda:0"]["active_sample_rate"], 1.0)
        self.assertEqual(summary["cuda:0"]["max_memory_mib"], 1500.0)

    def test_summarizes_rows_uses_running_timing_stats(self):
        report = _load_report_module()

        rows = [
            {
                "terminal_probe_devices": ["cuda:0"],
                "terminal_probe_trial_counts": [4],
                "terminal_probe_wall_seconds": 1.0,
                "policy_rollout_wall_seconds": 0.25,
                "replan_wall_seconds": 0.5,
                "jsonl_write_wall_seconds": 0.05,
            },
            {
                "terminal_probe_devices": ["cuda:0"],
                "terminal_probe_trial_counts": [4],
                "terminal_probe_wall_seconds": 3.0,
                "policy_rollout_wall_seconds": 0.75,
                "replan_wall_seconds": 1.5,
                "jsonl_write_wall_seconds": 0.15,
            },
        ]

        def fail_mean(_values):
            raise AssertionError("episode timing summary should use running aggregates")

        with mock.patch.object(statistics, "mean", fail_mean):
            summary = report.summarize_rows(rows, visible_devices=["0"])

        self.assertEqual(summary["terminal_probe_wall_seconds"]["count"], 2)
        self.assertEqual(summary["terminal_probe_wall_seconds"]["mean"], 2.0)
        self.assertEqual(summary["policy_rollout_wall_seconds"]["mean"], 0.5)
        self.assertEqual(summary["replan_wall_seconds"]["mean"], 1.0)
        self.assertEqual(summary["probe_wall_seconds_by_device"]["cuda:0"]["mean"], 2.0)
        self.assertEqual(summary["hot_path_wall_seconds"]["jsonl_write_wall_seconds"]["mean"], 0.1)

    def test_summarizes_rows_from_single_pass_iterable(self):
        report = _load_report_module()

        class SinglePassRows:
            def __init__(self):
                self.iterated = False

            def __iter__(self):
                if self.iterated:
                    raise AssertionError("rows were iterated more than once")
                self.iterated = True
                yield {
                    "episode": 0,
                    "terminal_probe_devices": ["cuda:0", "cuda:1"],
                    "terminal_probe_trial_counts": [1, 3],
                    "terminal_probe_wall_seconds": 2.0,
                    "terminal_probe_wall_seconds_by_device": {"cuda:0": 1.5, "cuda:1": 2.5},
                    "policy_rollout_wall_seconds": 0.25,
                }
                yield {
                    "episode": 1,
                    "terminal_probe_devices": ["cuda:1"],
                    "terminal_probe_trial_counts": [4],
                    "terminal_probe_wall_seconds": 1.0,
                    "terminal_probe_wall_seconds_by_device": {"cuda:1": 1.0},
                    "policy_rollout_wall_seconds": 0.75,
                }

        summary = report.summarize_rows(SinglePassRows(), visible_devices=["0", "1", "2"])

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["used_probe_devices"], ["cuda:0", "cuda:1"])
        self.assertEqual(summary["idle_visible_devices"], ["cuda:2"])
        self.assertEqual(summary["probe_trial_counts_by_device"], {"cuda:0": 1, "cuda:1": 7})
        self.assertEqual(summary["probe_episode_counts_by_device"], {"cuda:0": 1, "cuda:1": 2})
        self.assertEqual(summary["terminal_probe_wall_seconds"]["mean"], 1.5)
        self.assertEqual(summary["policy_rollout_wall_seconds"]["mean"], 0.5)
        self.assertEqual(summary["probe_wall_seconds_by_device"]["cuda:1"]["mean"], 1.75)

    def test_find_episodes_path_fallback_streams_without_path_rglob(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            episodes = root / "nested" / "run" / "diagnostics" / "episodes.jsonl"
            episodes.parent.mkdir(parents=True)
            episodes.write_text("{}\n", encoding="utf-8")

            with mock.patch.object(
                Path,
                "rglob",
                side_effect=AssertionError("episode fallback search should stream with os.walk"),
            ):
                found = report._find_episodes_path(root)

        self.assertEqual(found, episodes)

    def test_iter_jsonl_passes_unstripped_lines_to_json_loader(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            episodes = Path(td) / "episodes.jsonl"
            episodes.write_text('{"episode": 0}\n', encoding="utf-8")
            seen = []
            original_loads = report.json.loads

            def recording_loads(value):
                seen.append(value)
                return original_loads(value)

            with mock.patch.object(report.json, "loads", recording_loads):
                rows = list(report._iter_jsonl(episodes))

        self.assertEqual(rows, [{"episode": 0}])
        self.assertTrue(seen[0].endswith("\n"))

    def test_iter_jsonl_skips_whitespace_only_lines(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            episodes = Path(td) / "episodes.jsonl"
            episodes.write_text('\n  \n{"episode": 0}\n', encoding="utf-8")

            rows = list(report._iter_jsonl(episodes))

        self.assertEqual(rows, [{"episode": 0}])

    def test_cli_writes_json_and_markdown_reports(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            episodes = root / "episodes.jsonl"
            _write_jsonl(
                episodes,
                [
                    {
                        "episode": 0,
                        "terminal_probe_devices": ["cuda:0", "cuda:1"],
                        "terminal_probe_trial_counts": [1, 1],
                        "terminal_probe_wall_seconds": 1.0,
                    }
                ],
            )
            out_json = root / "gpu_report.json"
            out_md = root / "gpu_report.md"

            rc = report.main(
                [
                    "--episodes",
                    str(episodes),
                    "--visible-devices",
                    "0,1",
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                ]
            )

            self.assertEqual(rc, 0)
            written = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(written["used_probe_devices"], ["cuda:0", "cuda:1"])
            markdown = out_md.read_text(encoding="utf-8")
            self.assertIn("# GPU Utilization Report", markdown)
            self.assertIn("Used probe devices: cuda:0, cuda:1", markdown)
            self.assertIn("## Probe Wall By Device", markdown)
            self.assertIn("- cuda:0: episodes=1", markdown)
            self.assertIn("Replan/optimizer mean seconds:", markdown)


if __name__ == "__main__":
    unittest.main()
