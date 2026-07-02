import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "scripts" / "stage1_parallel_report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("stage1_parallel_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


SAMPLE_LOG = """
  [stage1-rollout] window=0 eps_per_worker=30  devices=[cuda:0, cuda:1] counts=[30, 30]  wall=100.000s worker_seconds=[99.000, 98.000]  speedup=1.97x
  [stage1-rollout] window=0 eval_cache hits=12 misses=48 distinct=48 hit_rate=20.0%
  [stage1-rollout-total] window=0 episodes=60 total=120.000s collect=100.000s replay=3.000s detail=2.000s ppo_update=10.000s other=5.000s throughput=1800.0ep/h
  [stage1-rollout] window=1 eps_per_worker=30  devices=[cuda:0, cuda:1] counts=[30, 30]  wall=90.000s worker_seconds=[89.000, 87.000]  speedup=1.96x
  [stage1-rollout] window=1 eval_cache hits=30 misses=90 distinct=90 hit_rate=25.0%
  [stage1-rollout-total] window=1 episodes=60 total=100.000s collect=82.000s replay=4.000s detail=1.000s ppo_update=8.000s other=5.000s throughput=2160.0ep/h
"""


class Stage1ParallelReportTest(unittest.TestCase):
    def test_parse_log_text_summarizes_parallel_windows(self):
        report = _load_report_module()

        summary = report.parse_log_text(SAMPLE_LOG)

        self.assertEqual(summary["windows"], 2)
        self.assertEqual(summary["total_episodes"], 120)
        self.assertEqual(summary["device_count"], 2)
        self.assertEqual(summary["worker_episode_counts_by_device"], {"cuda:0": 60, "cuda:1": 60})
        self.assertAlmostEqual(summary["throughput_ep_per_hour"], 1963.6363636)
        self.assertAlmostEqual(summary["mean_worker_speedup"], 1.965)
        self.assertEqual(summary["eval_cache"]["hits"], 30)
        self.assertEqual(summary["eval_cache"]["misses"], 90)
        self.assertEqual(summary["eval_cache"]["hit_rate"], 0.25)
        self.assertEqual(summary["component_seconds"]["collect"], 182.0)
        self.assertAlmostEqual(summary["component_share"]["collect"], 182.0 / 220.0)

    def test_parse_log_lines_accepts_single_pass_iterable(self):
        report = _load_report_module()

        class SinglePassLines:
            def __init__(self):
                self.iterated = False

            def __iter__(self):
                if self.iterated:
                    raise AssertionError("log lines were iterated more than once")
                self.iterated = True
                yield from SAMPLE_LOG.splitlines()

        summary = report.parse_log_lines(SinglePassLines())

        self.assertEqual(summary["windows"], 2)
        self.assertEqual(summary["total_episodes"], 120)
        self.assertEqual(summary["worker_episode_counts_by_device"], {"cuda:0": 60, "cuda:1": 60})
        self.assertAlmostEqual(summary["throughput_ep_per_hour"], 1963.6363636)
        self.assertEqual(summary["eval_cache"]["hit_rate"], 0.25)

    def test_parse_log_text_streams_text_lines_without_list(self):
        report = _load_report_module()
        captured = {}

        def fake_parse_lines(lines):
            captured["is_list"] = isinstance(lines, list)
            captured["rows"] = list(lines)
            return {"windows": 0}

        with mock.patch.object(report, "parse_log_lines", fake_parse_lines):
            summary = report.parse_log_text("first\nsecond\n")

        self.assertEqual(summary, {"windows": 0})
        self.assertFalse(captured["is_list"])
        self.assertEqual(captured["rows"], ["first\n", "second\n"])

    def test_parse_log_lines_uses_running_speedup_aggregate(self):
        report = _load_report_module()

        with mock.patch.object(report, "statistics", object(), create=True):
            summary = report.parse_log_lines(SAMPLE_LOG.splitlines())

        self.assertAlmostEqual(summary["mean_worker_speedup"], 1.965)
        self.assertAlmostEqual(summary["max_worker_speedup"], 1.97)

    def test_worker_balance_check_streams_counts_without_list(self):
        report = _load_report_module()

        class SinglePassCounts:
            def __init__(self):
                self.iterated = False

            def __iter__(self):
                if self.iterated:
                    raise AssertionError("worker counts were iterated more than once")
                self.iterated = True
                yield 10
                yield 13

        with mock.patch("builtins.list", side_effect=AssertionError("worker counts were materialized")):
            self.assertTrue(report._worker_counts_imbalanced(SinglePassCounts()))

        self.assertFalse(report._worker_counts_imbalanced([10, 12]))
        self.assertFalse(report._worker_counts_imbalanced([0, 100]))

    def test_parse_log_lines_skips_regex_for_unrelated_lines(self):
        report = _load_report_module()

        class BombPattern:
            def search(self, _line):
                raise AssertionError("unrelated lines should not hit regex")

        with (
            mock.patch.object(report, "ROLLOUT_RE", BombPattern()),
            mock.patch.object(report, "CACHE_RE", BombPattern()),
            mock.patch.object(report, "TOTAL_RE", BombPattern()),
        ):
            summary = report.parse_log_lines(["epoch 1 loss=0.1\n", "saving checkpoint\n"])

        self.assertEqual(summary["windows"], 0)
        self.assertIn("No [stage1-rollout] worker timing lines found.", summary["warnings"])

    def test_cli_writes_json_and_markdown(self):
        report = _load_report_module()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            log_path = root / "stage1.log"
            log_path.write_text(SAMPLE_LOG, encoding="utf-8")
            out_json = root / "stage1_report.json"
            out_md = root / "stage1_report.md"

            rc = report.main([
                "--log",
                str(log_path),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ])

            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["total_episodes"], 120)
            markdown = out_md.read_text(encoding="utf-8")
            self.assertIn("# Stage-1 Parallel Report", markdown)
            self.assertIn("Throughput: 1963.636", markdown)
            self.assertIn("- cuda:0: 60", markdown)


if __name__ == "__main__":
    unittest.main()
