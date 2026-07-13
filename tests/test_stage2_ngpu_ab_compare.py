import argparse
import builtins
import io
import importlib.util
import inspect
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


_REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "stage2_ngpu_ab_compare", str(_REPO / "scripts" / "stage2_ngpu_ab_compare.py")
)
ngpu_mod = importlib.util.module_from_spec(_spec)
sys.modules["stage2_ngpu_ab_compare"] = ngpu_mod
_spec.loader.exec_module(ngpu_mod)


def _row(ep, *, timestamp, device, pareto_kind):
    return {
        "episode": ep,
        "timestamp": timestamp,
        "total_reward": 12.5 + ep,
        "terminal_reward": 10.0 + ep,
        "terminal_priority": 3,
        "fusion_count": 7,
        "terminal_loss_mean": 0.3,
        "terminal_metric1_mean": 0.88,
        "terminal_metric2_mean": 0.87,
        "action_hash": f"h{ep}",
        "terminal_probe_devices": [device],
        "terminal_probe_wall_seconds": 1.0,
        "policy_rollout_wall_seconds": 0.2,
        "terminal_pareto_event_kind": pareto_kind,
    }


def _ppo_row(update, *, timestamp, elapsed_sec, entropy=0.1):
    return {
        "update": update,
        "completed_episodes": update * 120,
        "policy_loss": -0.01,
        "value_loss": 1.25,
        "entropy": entropy,
        "clip_fraction": 0.02,
        "n_samples": 120,
        "window_mean_return": 35.0,
        "best_reward_so_far": 40.0,
        "elapsed_sec": elapsed_sec,
        "approx_kl": 0.001,
        "timestamp": timestamp,
    }


class Stage2NgpuCompareTests(unittest.TestCase):
    def test_device_breakdown_counts_every_probe_device_and_trial(self):
        rows = [
            {
                "episode": 0,
                "terminal_probe_devices": ["cuda:0", "cuda:1"],
                "terminal_probe_trial_counts": [2, 3],
                "terminal_probe_wall_seconds": 1.5,
                "policy_rollout_wall_seconds": 0.2,
                "per_step_optimizer_wall_seconds": 0.1,
            }
        ]

        breakdown = ngpu_mod._device_breakdown(rows)

        self.assertEqual(set(breakdown), {"cuda:0", "cuda:1"})
        self.assertEqual(breakdown["cuda:0"]["episodes"], 1.0)
        self.assertEqual(breakdown["cuda:1"]["episodes"], 1.0)
        self.assertEqual(breakdown["cuda:0"]["trials"], 2.0)
        self.assertEqual(breakdown["cuda:1"]["trials"], 3.0)

    def test_load_jsonl_uses_shared_iter_jsonl(self):
        calls = []

        def fake_iter_jsonl(path, **kwargs):
            calls.append((path, kwargs))
            yield {"episode": 2}
            yield {"episode": 1}

        with mock.patch.object(ngpu_mod, "_find_jsonl", return_value="/tmp/run/episodes.jsonl"):
            with mock.patch.object(ngpu_mod, "iter_jsonl", fake_iter_jsonl):
                rows = ngpu_mod._load_jsonl(
                    "/tmp/run",
                    filename="episodes.jsonl",
                    sort_key="episode",
                )

        self.assertEqual([row["episode"] for row in rows], [1, 2])
        self.assertEqual(calls, [("/tmp/run/episodes.jsonl", {"errors": "raise"})])

    def test_load_jsonl_does_not_unconditionally_sort_ordered_logs(self):
        source = inspect.getsource(ngpu_mod._load_jsonl)
        self.assertNotIn(".sort(", source)

    def test_timestamp_span_streams_without_collecting_values(self):
        rows = [
            {"timestamp": 10.0},
            {"timestamp": None},
            {"timestamp": 16.5},
            {"timestamp": 12.0},
        ]

        self.assertEqual(ngpu_mod._timestamp_span(rows), 6.5)
        self.assertIsNone(ngpu_mod._timestamp_span([{"timestamp": 10.0}]))
        source = inspect.getsource(ngpu_mod._timestamp_span)
        self.assertNotIn("values = [", source)
        self.assertNotIn("max(values)", source)

    def test_effect_equality_ignores_timing_device_and_bookkeeping(self):
        one = [_row(0, timestamp=1.0, device="cuda:0", pareto_kind="dominated")]
        many = [_row(0, timestamp=2.0, device="cuda:4", pareto_kind="")]

        ok, diffs = ngpu_mod.compare_rows(one, many, atol=0.0, limit=10)
        self.assertTrue(ok, diffs)

        strict_ok, strict_diffs = ngpu_mod.compare_rows(
            one, many, atol=0.0, limit=10, strict_diagnostics=True
        )
        self.assertFalse(strict_ok)
        self.assertIn("terminal_pareto_event_kind", strict_diffs[0])

    def test_effect_equality_fails_on_metric_drift(self):
        one = [_row(0, timestamp=1.0, device="cuda:0", pareto_kind="")]
        many = [_row(0, timestamp=1.0, device="cuda:0", pareto_kind="")]
        many[0]["terminal_metric1_mean"] = 0.89

        ok, diffs = ngpu_mod.compare_rows(one, many, atol=0.0, limit=10)
        self.assertFalse(ok)
        self.assertTrue(any("terminal_metric1_mean" in diff for diff in diffs))

    def test_compare_rows_materializes_excluded_keys_once(self):
        class SinglePassExcluded:
            def __init__(self):
                self.used = False

            def __iter__(self):
                if self.used:
                    raise AssertionError("excluded keys should be materialized once per compare")
                self.used = True
                return iter(["timestamp"])

        one = [
            {"episode": 0, "timestamp": 1.0, "total_reward": 12.5},
            {"episode": 1, "timestamp": 2.0, "total_reward": 13.5},
        ]
        many = [
            {"episode": 0, "timestamp": 100.0, "total_reward": 12.5},
            {"episode": 1, "timestamp": 200.0, "total_reward": 13.5},
        ]

        ok, diffs = ngpu_mod.compare_rows(
            one,
            many,
            atol=0.0,
            limit=10,
            strict_diagnostics=True,
            excluded_keys=SinglePassExcluded(),
        )

        self.assertTrue(ok, diffs)

    def test_require_speedup_adds_fatal_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            one_path = root / "one.jsonl"
            many_path = root / "many.jsonl"
            one_path.write_text(
                "\n".join(ngpu_mod.json.dumps(_row(i, timestamp=i, device="cuda:0", pareto_kind="")) for i in range(2)) + "\n",
                encoding="utf-8",
            )
            many_path.write_text(
                "\n".join(ngpu_mod.json.dumps(_row(i, timestamp=i, device="cuda:1", pareto_kind="")) for i in range(2)) + "\n",
                encoding="utf-8",
            )
            one_wall = root / "one_wall.txt"
            many_wall = root / "many_wall.txt"
            one_wall.write_text("100\n", encoding="utf-8")
            many_wall.write_text("60\n", encoding="utf-8")

            report = ngpu_mod.build_report(
                argparse.Namespace(
                    one=str(one_path),
                    many=str(many_path),
                    one_ppo=None,
                    many_ppo=None,
                    one_wall=str(one_wall),
                    many_wall=str(many_wall),
                    atol=0.0,
                    max_diffs=10,
                    strict_diagnostics=False,
                    require_equal=True,
                    min_speedup=4.5,
                    require_speedup=True,
                    one_log=None,
                    many_log=None,
                )
            )
        self.assertIn("quality/effect equality: PASS", report)
        self.assertIn("PPO update equality: n/a", report)
        self.assertIn("[FATAL] speedup requirement failed", report)
        self.assertIn("NGPU distinct probe devices: 1", report)
        self.assertIn("speedup/device_count: 1.667", report)
        self.assertIn("NGPU device episode balance min/max: 2/2", report)
        self.assertIn("NGPU probe-bound ceiling episodes/hour: 3600.000", report)
        self.assertIn("NGPU wall/probe_bound ratio: 30.000", report)
        self.assertIn("NGPU probe ceiling utilization: 0.033", report)
        self.assertIn("NGPU component critical-path lower bound_s: 2.400", report)
        self.assertIn("NGPU component-bound ceiling episodes/hour: 3000.000", report)
        self.assertIn("NGPU wall/component_bound ratio: 25.000", report)
        self.assertIn("NGPU component ceiling utilization: 0.040", report)
        self.assertIn("NGPU/1GPU terminal probe mean ratio: 1.000", report)
        self.assertIn("NGPU/1GPU policy rollout mean ratio: 1.000", report)
        self.assertIn("NGPU worker-local probe noise scopes detected: False", report)
        self.assertIn("NGPU worker-local CUDA probe streams detected: False", report)

    def test_many_log_is_scanned_once_for_timing_and_marker_flags(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            one_path = root / "one.jsonl"
            many_path = root / "many.jsonl"
            one_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=1.0, device="cuda:0", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            many_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=2.0, device="cuda:1", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            one_log = root / "one.log"
            many_log = root / "many.log"
            one_log.write_text(
                "[stage2-rollout-timing] window_start=0 episodes=1 collect_s=1.000\n",
                encoding="utf-8",
            )
            many_log.write_text(
                "[stage2-parallel] workers-per-device=2 -> 10 workers "
                "(worker-local probe noise scopes active; "
                "worker-local CUDA probe streams active)\n"
                "[stage2-parallel] policy_device=cpu "
                "(GTrXL rollout kept off reward-probe GPUs)\n"
                "[stage2-rollout-timing] window_start=0 episodes=1 collect_s=2.000\n",
                encoding="utf-8",
            )
            many_log_opens = 0
            original_open = builtins.open

            def counting_open(file, *args, **kwargs):
                nonlocal many_log_opens
                if str(file) == str(many_log):
                    many_log_opens += 1
                return original_open(file, *args, **kwargs)

            with mock.patch("builtins.open", counting_open):
                report = ngpu_mod.build_report(
                    argparse.Namespace(
                        one=str(one_path),
                        many=str(many_path),
                        one_ppo=None,
                        many_ppo=None,
                        one_wall=None,
                        many_wall=None,
                        atol=0.0,
                        max_diffs=10,
                        strict_diagnostics=False,
                        require_equal=True,
                        min_speedup=None,
                        require_speedup=False,
                        one_log=str(one_log),
                        many_log=str(many_log),
                    )
                )

        self.assertEqual(many_log_opens, 1)
        self.assertIn("collect_s: total_s=2.000", report)
        self.assertIn("NGPU worker-local probe noise scopes detected: True", report)
        self.assertIn("NGPU worker-local CUDA probe streams detected: True", report)
        self.assertIn("NGPU cpu policy mode detected: True", report)

    def test_rollout_timing_logs_are_summarized(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            one_path = root / "one.jsonl"
            many_path = root / "many.jsonl"
            one_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=1.0, device="cuda:0", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            many_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=2.0, device="cuda:1", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            one_log = root / "one.log"
            many_log = root / "many.log"
            one_log.write_text(
                "[stage2-rollout-timing] window_start=0 episodes=120 "
                "sync_s=1.000 collect_s=20.000\n"
                "  [stage2-rollout-timing] window_start=0 episodes=120 "
                "collect_total_s=21.000 assembly_update_s=4.000 "
                "buffer_add_s=0.500 finalize_update_s=3.500 "
                "ppo_update_s=2.000 episode_callback_s=1.000 "
                "finalize_other_s=0.500 window_total_s=25.000\n",
                encoding="utf-8",
            )
            many_log.write_text(
                "[stage2-parallel] workers-per-device=2 -> 10 workers "
                "(worker-local probe noise scopes active; "
                "worker-local CUDA probe streams active)\n"
                "[stage2-parallel] policy_device=cpu "
                "(GTrXL rollout kept off reward-probe GPUs)\n"
                "[stage2-rollout-timing] window_start=0 episodes=120 "
                "sync_s=2.000 collect_s=8.000\n"
                "  [stage2-rollout-timing] window_start=0 episodes=120 "
                "collect_total_s=10.000 assembly_update_s=6.000 "
                "buffer_add_s=0.700 finalize_update_s=5.300 "
                "ppo_update_s=4.000 episode_callback_s=1.500 "
                "finalize_other_s=0.200 window_total_s=16.000\n",
                encoding="utf-8",
            )

            report = ngpu_mod.build_report(
                argparse.Namespace(
                    one=str(one_path),
                    many=str(many_path),
                    one_ppo=None,
                    many_ppo=None,
                    one_wall=None,
                    many_wall=None,
                    atol=0.0,
                    max_diffs=10,
                    strict_diagnostics=False,
                    require_equal=True,
                    min_speedup=None,
                    require_speedup=False,
                    one_log=str(one_log),
                    many_log=str(many_log),
                )
            )
        self.assertIn("1GPU rollout timing log: windows=2 episodes=240", report)
        self.assertIn("NGPU rollout timing log: windows=2 episodes=240", report)
        self.assertIn("finalize_update_s: total_s=5.300", report)
        self.assertIn("ppo_update_s: total_s=4.000", report)
        self.assertIn("episode_callback_s: total_s=1.500", report)
        self.assertIn("finalize_other_s: total_s=0.200", report)
        self.assertIn("window_total_s: total_s=16.000", report)
        self.assertIn("NGPU worker-local probe noise scopes detected: True", report)
        self.assertIn("NGPU worker-local CUDA probe streams detected: True", report)
        self.assertIn("NGPU cpu policy mode detected: True", report)

    def test_ppo_update_equality_ignores_timing_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            one_path = root / "one.jsonl"
            many_path = root / "many.jsonl"
            one_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=1.0, device="cuda:0", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            many_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=2.0, device="cuda:1", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            one_ppo = root / "one_ppo.jsonl"
            many_ppo = root / "many_ppo.jsonl"
            one_ppo.write_text(
                ngpu_mod.json.dumps(_ppo_row(1, timestamp=10.0, elapsed_sec=30.0)) + "\n",
                encoding="utf-8",
            )
            many_ppo.write_text(
                ngpu_mod.json.dumps(_ppo_row(1, timestamp=50.0, elapsed_sec=9.0)) + "\n",
                encoding="utf-8",
            )

            report = ngpu_mod.build_report(
                argparse.Namespace(
                    one=str(one_path),
                    many=str(many_path),
                    one_ppo=str(one_ppo),
                    many_ppo=str(many_ppo),
                    one_wall=None,
                    many_wall=None,
                    atol=0.0,
                    max_diffs=10,
                    strict_diagnostics=False,
                    require_equal=True,
                    min_speedup=None,
                    require_speedup=False,
                    one_log=None,
                    many_log=None,
                )
            )
        self.assertIn("PPO update equality: PASS", report)
        self.assertNotIn("[FATAL] equality requirement failed", report)

    def test_ppo_update_equality_fails_on_metric_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            one_path = root / "one.jsonl"
            many_path = root / "many.jsonl"
            one_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=1.0, device="cuda:0", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            many_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=2.0, device="cuda:1", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            one_ppo = root / "one_ppo.jsonl"
            many_ppo = root / "many_ppo.jsonl"
            one_ppo.write_text(
                ngpu_mod.json.dumps(_ppo_row(1, timestamp=10.0, elapsed_sec=30.0, entropy=0.1)) + "\n",
                encoding="utf-8",
            )
            many_ppo.write_text(
                ngpu_mod.json.dumps(_ppo_row(1, timestamp=10.0, elapsed_sec=30.0, entropy=0.2)) + "\n",
                encoding="utf-8",
            )

            report = ngpu_mod.build_report(
                argparse.Namespace(
                    one=str(one_path),
                    many=str(many_path),
                    one_ppo=str(one_ppo),
                    many_ppo=str(many_ppo),
                    one_wall=None,
                    many_wall=None,
                    atol=0.0,
                    max_diffs=10,
                    strict_diagnostics=False,
                    require_equal=True,
                    min_speedup=None,
                    require_speedup=False,
                    one_log=None,
                    many_log=None,
                )
            )
        self.assertIn("PPO update equality: FAIL", report)
        self.assertIn("ppo_update[1].entropy", report)
        self.assertIn("[FATAL] equality requirement failed", report)

    def test_main_streams_output_without_build_report_string(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            one_path = root / "one.jsonl"
            many_path = root / "many.jsonl"
            one_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=1.0, device="cuda:0", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            many_path.write_text(
                ngpu_mod.json.dumps(_row(0, timestamp=2.0, device="cuda:1", pareto_kind="")) + "\n",
                encoding="utf-8",
            )
            out_path = root / "report.txt"
            argv = [
                "stage2_ngpu_ab_compare.py",
                "--one", str(one_path),
                "--many", str(many_path),
                "--out", str(out_path),
                "--require-equal",
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ngpu_mod,
                    "build_report",
                    side_effect=AssertionError("main should stream report lines"),
                ),
                mock.patch("sys.stdout", new_callable=io.StringIO) as stdout,
            ):
                rc = ngpu_mod.main()

            out_text = out_path.read_text(encoding="utf-8")
            stdout_text = stdout.getvalue()

        self.assertEqual(rc, 0)
        self.assertIn("quality/effect equality: PASS", out_text)
        self.assertEqual(out_text, stdout_text)


if __name__ == "__main__":
    unittest.main()
