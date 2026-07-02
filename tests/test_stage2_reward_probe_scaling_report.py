import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "stage2_reward_probe_scaling_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("stage2_reward_probe_scaling_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


class Stage2RewardProbeScalingReportTest(unittest.TestCase):
    def test_iter_jsonl_passes_unstripped_lines_to_json_loader(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            path.write_text('{"label": "bs128_g1"}\n', encoding="utf-8")
            seen = []
            original_loads = report.json.loads

            def recording_loads(value):
                seen.append(value)
                return original_loads(value)

            with mock.patch.object(report.json, "loads", recording_loads):
                rows = list(report._iter_jsonl(path))

        self.assertEqual(rows, [{"label": "bs128_g1"}])
        self.assertTrue(seen[0].endswith("\n"))

    def test_iter_jsonl_skips_whitespace_only_lines(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rows.jsonl"
            path.write_text('\n  \n{"label": "bs128_g1"}\n', encoding="utf-8")

            rows = list(report._iter_jsonl(path))

        self.assertEqual(rows, [{"label": "bs128_g1"}])

    def test_episode_summary_uses_running_means(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            episodes = Path(td) / "episodes.jsonl"
            _write_jsonl(
                episodes,
                [
                    {
                        "terminal_probe_wall_seconds": 1.0,
                        "terminal_probe_speedup": 2.0,
                    },
                    {
                        "terminal_probe_wall_seconds": 3.0,
                        "terminal_probe_speedup": 4.0,
                    },
                ],
            )

            def fail_mean(_values):
                raise AssertionError("episode summary means should use running totals")

            with mock.patch.object(statistics, "mean", fail_mean):
                summary = report._summarize_episodes(episodes)

        self.assertEqual(summary["probe_calls"], 2)
        self.assertEqual(summary["mean_wall"], 2.0)
        self.assertEqual(summary["median_wall"], 2.0)
        self.assertEqual(summary["mean_speedup"], 3.0)

    def test_build_summary_streams_benchmark_artifacts_and_selects_best_run(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(
                root / "runs.jsonl",
                [
                    {
                        "label": "bs128_g1",
                        "batch_size": 128,
                        "gpu_count": 1,
                        "device_spec": "0",
                        "launch_rc": 0,
                        "rc": 0,
                    },
                    {
                        "label": "bs256_g4",
                        "batch_size": 256,
                        "gpu_count": 4,
                        "device_spec": "0,1,2,3",
                        "launch_rc": 0,
                        "rc": 0,
                    },
                ],
            )
            _write_jsonl(
                root / "bs128_g1_episodes.jsonl",
                [
                    {
                        "terminal_probe_wall_seconds": 4.0,
                        "terminal_probe_speedup": 1.0,
                        "terminal_probe_devices": ["cuda:0"],
                        "terminal_probe_trial_counts": [4],
                    }
                ],
            )
            _write_jsonl(
                root / "bs256_g4_episodes.jsonl",
                [
                    {
                        "terminal_probe_wall_seconds": 1.0,
                        "terminal_probe_speedup": 3.8,
                        "terminal_probe_devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                        "terminal_probe_trial_counts": [1, 1, 1, 1],
                    },
                    {
                        "terminal_probe_wall_seconds": 1.2,
                        "terminal_probe_speedup": 3.6,
                        "terminal_probe_devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                        "terminal_probe_trial_counts": [1, 1, 1, 1],
                    },
                ],
            )
            (root / "bs256_g4_nvidia_smi.csv").write_text(
                "timestamp,index,name,memory_used_mib,utilization_gpu_pct\n"
                "2026/07/02 00:00:00.000,0,A100,12000,91\n"
                "2026/07/02 00:00:00.000,1,A100,11800,88\n",
                encoding="utf-8",
            )

            summary = report.build_summary(root)

        self.assertEqual(summary["best"]["label"], "bs256_g4")
        rows = {row["label"]: row for row in summary["runs"]}
        self.assertEqual(rows["bs128_g1"]["probe_calls"], 1)
        self.assertEqual(rows["bs256_g4"]["probe_calls"], 2)
        self.assertAlmostEqual(rows["bs256_g4"]["mean_wall"], 1.1)
        self.assertAlmostEqual(rows["bs256_g4"]["mean_speedup"], 3.7)
        self.assertEqual(rows["bs256_g4"]["devices_seen"], ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        self.assertEqual(rows["bs256_g4"]["trial_splits"], [[1, 1, 1, 1]])
        self.assertEqual(rows["bs256_g4"]["max_gpu_util_pct"], {"0": 91.0, "1": 88.0})
        self.assertEqual(rows["bs256_g4"]["max_gpu_mem_mib"], {"0": 12000.0, "1": 11800.0})

    def test_gpu_sample_summary_normalizes_header_once_not_per_row(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            smi = root / "samples.csv"
            smi.write_text(
                "gpu index, utilization.gpu [%], memory.used [MiB]\n"
                "0,20 %,1000 MiB\n"
                "0,40 %,1500 MiB\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                report,
                "_normalized_row",
                side_effect=AssertionError("CSV headers should be normalized once"),
            ):
                gpu_util, gpu_mem = report._summarize_gpu_samples(smi)

        self.assertEqual(gpu_util, {"0": 40.0})
        self.assertEqual(gpu_mem, {"0": 1500.0})

    def test_render_html_iterates_runs_without_materializing_list(self):
        report = _load_report_module()

        class StreamingRuns:
            def __iter__(self):
                return iter([
                    {
                        "label": "bs128_g1",
                        "batch_size": 128,
                        "gpu_count": 1,
                        "rc": 0,
                        "mean_wall": 2.0,
                    }
                ])

            def __length_hint__(self):
                raise AssertionError("render_html should not materialize runs with list()")

        html = report.render_html({
            "runs": StreamingRuns(),
            "best": {"label": "bs128_g1", "mean_wall": 2.0},
        })

        self.assertIn("bs128_g1", html)

    def test_main_writes_summary_html_and_best_batch_size(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_jsonl(
                root / "runs.jsonl",
                [
                    {
                        "label": "bs128_g1",
                        "batch_size": 128,
                        "gpu_count": 1,
                        "device_spec": "0",
                        "launch_rc": 0,
                        "rc": 0,
                    }
                ],
            )
            _write_jsonl(
                root / "bs128_g1_episodes.jsonl",
                [{"terminal_probe_wall_seconds": 2.0}],
            )

            rc = report.main([str(root)])

            self.assertEqual(rc, 0)
            self.assertTrue((root / "benchmark_summary.json").is_file())
            self.assertTrue((root / "stage2_reward_probe_scaling_report.html").is_file())
            self.assertEqual((root / "best_batch_size.txt").read_text(encoding="utf-8"), "128\n")


if __name__ == "__main__":
    unittest.main()
