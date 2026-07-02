import csv
import importlib.util
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
