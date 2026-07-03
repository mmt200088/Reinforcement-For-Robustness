import argparse
import csv
import importlib.util
import json
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
    def test_monitor_uses_shared_jsonl_reader(self):
        source = MONITOR_PATH.read_text(encoding="utf-8")

        self.assertIn("from jsonl_utils import read_jsonl", source)
        self.assertNotIn("def _read_jsonl(", source)

    def test_window_reuses_sorted_tail_for_bounds_and_mean(self):
        monitor = _load_monitor_module()

        with (
            mock.patch(
                "statistics.mean",
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

    def test_build_summary_uses_fast_mean_for_large_series(self):
        monitor = _load_monitor_module()
        episodes = [
            {
                "episode": idx,
                "total_reward": float(idx + 1),
                "terminal_reward": float(idx) + 0.5,
                "terminal_priority": 3,
                "valid_steps": 47,
                "invalid_steps": 0,
                "total_bits": 100 - idx,
            }
            for idx in range(3)
        ]
        ppo = [
            {
                "update": 1,
                "n_samples": 1,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 0.2,
                "clip_fraction": 0.4,
                "approx_kl": 0.01,
                "lr_scale": 0.5,
                "entropy_recovery_delta": 0.03,
            }
        ]

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "statistics.mean",
            side_effect=AssertionError("build_summary should avoid statistics.mean"),
        ):
            args = argparse.Namespace(
                artifact_dir=str(Path(td) / "artifact"),
                stage2_noise=str(Path(td) / "noise"),
                nvidia_log=str(Path(td) / "nvidia.csv"),
                phase="live",
                planned=3,
                anchor=1,
                rollout=1,
                horizon=1,
                k_trials=5,
                probe_size=256,
                expected_reward_devices="",
                max_post_anchor_p12_rate=0.30,
                min_post_anchor_p12_rate_samples=100,
            )
            summary = monitor.build_summary(args, episodes=episodes, ppo=ppo)

        self.assertEqual(summary["reward"]["mean"], 2.0)
        self.assertEqual(summary["reward"]["post_anchor_mean"], 2.5)
        self.assertEqual(summary["ppo"]["recent_entropy_mean"], 0.2)
        self.assertEqual(summary["ppo"]["recent_clip_fraction_mean"], 0.4)
        self.assertEqual(summary["ppo"]["recent_approx_kl_mean"], 0.01)
        self.assertEqual(summary["ppo"]["recent_lr_scale_mean"], 0.5)
        self.assertEqual(summary["ppo"]["recent_entropy_recovery_mean"], 0.03)

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

    def test_gpu_stats_computes_median_without_statistics_copy(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nvidia.csv"
            path.write_text(
                "gpu_idx,util_pct,mem_used_mib\n"
                "0,40,100\n"
                "0,10,110\n"
                "0,30,120\n"
                "0,20,130\n",
                encoding="utf-8",
            )

            with mock.patch(
                "statistics.median",
                side_effect=AssertionError("_gpu_stats should avoid statistics.median copies"),
            ):
                summary = monitor._gpu_stats(path)

        self.assertEqual(summary["by_gpu"]["0"]["max_util"], 40.0)
        self.assertEqual(summary["by_gpu"]["0"]["p50_util"], 25.0)

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
            original_read_jsonl = monitor.read_jsonl
            read_counts = {}

            def counting_read_jsonl(path, **kwargs):
                path = Path(path)
                read_counts[path] = read_counts.get(path, 0) + 1
                return original_read_jsonl(path, **kwargs)

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
                "read_jsonl",
                side_effect=counting_read_jsonl,
            ):
                rc = monitor.main()

            self.assertEqual(rc, 0)
            self.assertEqual(read_counts.get(episodes_path), 1)
            self.assertTrue((artifact / "reward_windows.csv").is_file())
            self.assertTrue((artifact / "episode_health_windows.csv").is_file())

    def test_build_summary_streams_ppo_updates_without_full_read_jsonl(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / "artifact"
            artifact.mkdir()
            stage2_noise = root / "stage2_noise"
            (artifact / "episodes.jsonl").write_text(
                '{"episode": 0, "total_reward": 1.0, "terminal_reward": 1.0, '
                '"terminal_priority": 3, "valid_steps": 47, "invalid_steps": 0, '
                '"total_bits": 100}\n',
                encoding="utf-8",
            )
            ppo_path = artifact / "ppo_updates.jsonl"
            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "update": idx,
                        "n_samples": 999 if idx == 0 else 1,
                        "policy_loss": 0.1,
                        "value_loss": 0.2,
                        "entropy": float(idx),
                        "clip_fraction": 0.0,
                    }
                )
            ppo_path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )

            original_read_jsonl = monitor.read_jsonl

            def guarded_read_jsonl(path, **kwargs):
                path = Path(path)
                if path.name == "ppo_updates.jsonl":
                    raise AssertionError("PPO updates should be streamed, not fully materialized")
                return original_read_jsonl(path, **kwargs)

            args = argparse.Namespace(
                artifact_dir=str(artifact),
                stage2_noise=str(stage2_noise),
                nvidia_log=str(root / "nvidia.csv"),
                phase="live",
                planned=1,
                anchor=0,
                rollout=1,
                horizon=1,
                k_trials=5,
                probe_size=256,
                expected_reward_devices="",
                max_post_anchor_p12_rate=0.30,
                min_post_anchor_p12_rate_samples=100,
            )
            with mock.patch.object(monitor, "read_jsonl", side_effect=guarded_read_jsonl):
                summary = monitor.build_summary(args)

        self.assertEqual(summary["ppo"]["updates_seen"], 10)
        self.assertEqual(summary["ppo"]["last_update"]["update"], 9)
        self.assertEqual(summary["ppo"]["recent_entropy_mean"], 7.0)
        self.assertTrue(
            any("PPO update 0 n_samples != 1" in item for item in summary["hard_failures"])
        )

    def test_live_main_streams_monitor_json_without_json_dumps(self):
        monitor = _load_monitor_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / "artifact"
            artifact.mkdir()
            stage2_noise = root / "stage2_noise"
            nvidia_log = root / "nvidia.csv"
            (artifact / "episodes.jsonl").write_text(
                (
                    '{"episode": 0, "total_reward": 1.0, "terminal_reward": 1.0, '
                    '"terminal_priority": 3, "terminal_loss_mean": 0.2, '
                    '"terminal_metric1_mean": 0.9, "terminal_metric2_mean": 0.8, '
                    '"valid_steps": 47, "invalid_steps": 0, "total_bits": 100}'
                    "\n"
                ),
                encoding="utf-8",
            )
            nvidia_log.write_text("gpu_idx,util_pct,mem_used_mib\n", encoding="utf-8")
            argv = [
                "stage2_first10k_monitor.py",
                "--phase",
                "live",
                "--artifact-dir",
                str(artifact),
                "--stage2-noise",
                str(stage2_noise),
                "--nvidia-log",
                str(nvidia_log),
                "--planned",
                "1",
                "--anchor",
                "0",
                "--rollout",
                "1",
                "--horizon",
                "1",
            ]

            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                monitor.json,
                "dumps",
                side_effect=AssertionError("live monitor JSON should stream through file handles"),
            ):
                rc = monitor.main()

            live = json.loads((artifact / "monitor_live.json").read_text(encoding="utf-8"))
            events = [
                json.loads(line)
                for line in (artifact / "monitor_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(rc, 0)
        self.assertEqual(live["completed_episodes"], 1)
        self.assertEqual(events[-1]["phase"], "live")
        self.assertEqual(events[-1]["completed_episodes"], 1)

    def test_write_report_streams_nested_json_without_json_dumps(self):
        monitor = _load_monitor_module()

        summary = {
            "status": "ok",
            "completed_episodes": 1,
            "reward": {"best_reward": 1.0},
            "reward_probe": {
                "devices": ["cuda:0", "cuda:1"],
                "large_debug": [{"idx": idx, "reward": float(idx)} for idx in range(8)],
            },
            "gpu": {
                "by_gpu": {
                    "cuda:0": {"samples": 2, "max_util": 80.0},
                    "cuda:1": {"samples": 2, "max_util": 82.0},
                }
            },
            "hard_failures": [],
            "warnings": [],
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.html"
            with mock.patch.object(
                monitor.json,
                "dumps",
                side_effect=AssertionError("write_report should stream nested JSON chunks"),
            ):
                monitor.write_report(path, summary)

            html_text = path.read_text(encoding="utf-8")

        self.assertIn("reward_probe", html_text)
        self.assertIn("cuda:0", html_text)
        self.assertIn("max_util", html_text)


if __name__ == "__main__":
    unittest.main()
