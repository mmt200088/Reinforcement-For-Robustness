import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = REPO_ROOT / "scripts" / "stage2_first10k_monitor.py"


def _load_monitor_module():
    spec = importlib.util.spec_from_file_location("stage2_first10k_monitor", MONITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Stage2First10kMonitorTest(unittest.TestCase):
    def test_read_jsonl_streams_lines_without_read_text(self):
        monitor = _load_monitor_module()
        original_read_text = Path.read_text

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "episodes.jsonl"
            path.write_text('{"episode": 0, "total_reward": 1.0}\n\nnot-json\n{"episode": 1}\n', encoding="utf-8")

            def fail_read_text(_path, *args, **kwargs):
                raise AssertionError("_read_jsonl should stream file lines")

            try:
                Path.read_text = fail_read_text
                rows = monitor._read_jsonl(path)
            finally:
                Path.read_text = original_read_text

        self.assertEqual([row["episode"] for row in rows], [0, 1])

    def test_read_jsonl_skips_blank_lines_without_strip_copy(self):
        monitor = _load_monitor_module()

        class NoStripLine(str):
            def strip(self, *_args, **_kwargs):
                raise AssertionError("_read_jsonl should not allocate strip() copies")

        class FakeHandle:
            def __init__(self):
                self.lines = [
                    NoStripLine('{"episode": 0}\n'),
                    NoStripLine("   \n"),
                    NoStripLine('{"episode": 1}\n'),
                ]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter(self.lines)

        class FakePath:
            def exists(self):
                return True

            def open(self, **_kwargs):
                return FakeHandle()

        rows = monitor._read_jsonl(FakePath())

        self.assertEqual([row["episode"] for row in rows], [0, 1])

    def test_window_reuses_sorted_tail_for_bounds_and_mean(self):
        monitor = _load_monitor_module()

        with (
            mock.patch.object(
                monitor.statistics,
                "mean",
                side_effect=AssertionError("_window should use a single sum pass"),
            ),
            mock.patch(
                "builtins.min",
                side_effect=AssertionError("_window should reuse sorted bounds"),
            ),
            mock.patch(
                "builtins.max",
                side_effect=AssertionError("_window should reuse sorted bounds"),
            ),
        ):
            summary = monitor._window([9.0, 1.0, 5.0, 3.0, 7.0], 5)

        self.assertEqual(
            summary,
            {
                "size": 5,
                "mean": 5.0,
                "min": 1.0,
                "p05": 1.0,
                "p50": 5.0,
                "p95": 9.0,
                "max": 9.0,
                "slope": -0.5,
            },
        )

    def test_gpu_stats_ignores_directory_path(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            summary = monitor._gpu_stats(Path(td))

        self.assertEqual(summary, {"samples": 0, "by_gpu": {}})

    def test_gpu_stats_aggregates_large_csv_without_retaining_rows(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nvidia.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["gpu_idx", "util_pct", "mem_used_mib"])
                for idx in range(30000):
                    writer.writerow([idx % 4, (idx * 7) % 100, 1000 + (idx % 17)])

            tracemalloc.start()
            try:
                summary = monitor._gpu_stats(path)
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()

        self.assertEqual(summary["samples"], 30000)
        self.assertEqual(sorted(summary["by_gpu"]), ["0", "1", "2", "3"])
        self.assertLess(peak, 6 * 1024 * 1024)

    def test_gpu_stats_avoids_per_row_dict_reader(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nvidia.csv"
            path.write_text(
                "gpu_idx,util_pct,mem_used_mib\n"
                "0,10,100\n"
                "0,30,120\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                monitor.csv,
                "DictReader",
                side_effect=AssertionError("_gpu_stats should avoid per-row CSV dicts"),
            ):
                summary = monitor._gpu_stats(path)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["by_gpu"]["0"]["max_util"], 30.0)
        self.assertEqual(summary["by_gpu"]["0"]["p50_util"], 20.0)
        self.assertEqual(summary["by_gpu"]["0"]["max_mem_mib"], 120.0)

    def test_write_window_csv_uses_linear_rolling_stats(self):
        monitor = _load_monitor_module()
        episodes = [
            {"episode": idx, "total_reward": float(idx)}
            for idx in range(1001)
        ]

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "reward_windows.csv"
            with mock.patch.object(
                monitor,
                "_window",
                side_effect=AssertionError("write_window_csv should not reslice windows"),
            ):
                monitor.write_window_csv(path, episodes)

            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1001)
        self.assertEqual(rows[0]["episode"], "0")
        self.assertEqual(rows[58]["rolling60_mean"], "")
        self.assertEqual(rows[59]["rolling60_mean"], "29.50000000")
        self.assertEqual(rows[59]["rolling60_min"], "0.00000000")
        self.assertEqual(rows[59]["rolling60_max"], "59.00000000")
        self.assertEqual(rows[1000]["rolling1000_mean"], "500.50000000")
        self.assertEqual(rows[1000]["rolling1000_min"], "1.00000000")
        self.assertEqual(rows[1000]["rolling1000_max"], "1000.00000000")

    def test_write_window_csv_streams_rows_to_writer(self):
        monitor = _load_monitor_module()
        episodes = [
            {"episode": idx, "total_reward": float(idx)}
            for idx in range(65)
        ]

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "reward_windows.csv"
            with mock.patch.object(
                csv.DictWriter,
                "writerows",
                side_effect=AssertionError("write_window_csv should not buffer all rows"),
            ):
                monitor.write_window_csv(path, episodes)

            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 65)
        self.assertEqual(rows[64]["rolling60_mean"], "34.50000000")

    def test_final_main_reuses_loaded_episode_rows_for_csv_outputs(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / "artifact"
            artifact.mkdir()
            stage2_noise = root / "stage2_noise"
            (stage2_noise / "progress" / "diagnostics").mkdir(parents=True)
            episodes_path = artifact / "episodes.jsonl"
            episodes_path.write_text(
                "\n".join(
                    [
                        (
                            '{"episode": 0, "total_reward": 1.0, "terminal_reward": 1.0, '
                            '"terminal_priority": 3, "terminal_loss_mean": 0.2, '
                            '"terminal_metric1_mean": 0.9, "terminal_metric2_mean": 0.8, '
                            '"valid_steps": 47, "invalid_steps": 0, "total_bits": 100}'
                        ),
                        (
                            '{"episode": 1, "total_reward": 2.0, "terminal_reward": 2.0, '
                            '"terminal_priority": 3, "terminal_loss_mean": 0.2, '
                            '"terminal_metric1_mean": 0.9, "terminal_metric2_mean": 0.8, '
                            '"valid_steps": 47, "invalid_steps": 0, "total_bits": 99}'
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (artifact / "ppo_updates.jsonl").write_text(
                (
                    '{"update": 1, "n_samples": 1, "policy_loss": 0.1, '
                    '"value_loss": 0.2, "entropy": 1.0, "clip_fraction": 0.0}'
                    "\n"
                ),
                encoding="utf-8",
            )
            nvidia_log = root / "nvidia.csv"
            nvidia_log.write_text(
                "gpu_idx,util_pct,mem_used_mib\n0,10,100\n1,12,110\n",
                encoding="utf-8",
            )
            original_read_jsonl = monitor._read_jsonl
            read_counts = {}

            def counting_read_jsonl(path):
                path = Path(path)
                read_counts[path] = read_counts.get(path, 0) + 1
                return original_read_jsonl(path)

            argv = [
                "stage2_first10k_monitor.py",
                "--phase",
                "final",
                "--artifact-dir",
                str(artifact),
                "--stage2-noise",
                str(stage2_noise),
                "--nvidia-log",
                str(nvidia_log),
                "--planned",
                "2",
                "--anchor",
                "0",
                "--rollout",
                "1",
                "--horizon",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                monitor,
                "_read_jsonl",
                side_effect=counting_read_jsonl,
            ):
                rc = monitor.main()

            self.assertEqual(rc, 0)
            self.assertEqual(read_counts.get(episodes_path), 1)
            self.assertTrue((artifact / "reward_windows.csv").is_file())
            self.assertTrue((artifact / "episode_health_windows.csv").is_file())


if __name__ == "__main__":
    unittest.main()
