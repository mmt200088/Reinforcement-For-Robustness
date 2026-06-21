"""Torch-free locks for the Stage-2 rollout A/B comparator."""

import importlib.util
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "blb_kvcache_ab_compare", str(_REPO / "scripts" / "blb_kvcache_ab_compare.py")
)
ab_mod = importlib.util.module_from_spec(_spec)
sys.modules["blb_kvcache_ab_compare"] = ab_mod
_spec.loader.exec_module(ab_mod)


def _row(ep, ts, priority, fusion, reward=40.0, probe=2.0, policy=0.1, replan=0.2):
    return {
        "episode": ep,
        "timestamp": ts,
        "terminal_priority": priority,
        "fusion_count": fusion,
        "terminal_reward": reward,
        "terminal_probe_wall_seconds": probe,
        "terminal_probe_devices": ["cuda:0", "cuda:1"],
        "terminal_probe_trial_counts": [3, 2],
        "policy_rollout_wall_seconds": policy,
        "per_step_optimizer_wall_seconds": replan,
    }


class RolloutABComparatorTest(unittest.TestCase):
    def test_summary_reads_current_priority_and_fusion_fields(self):
        rows = [
            _row(0, 1000.0, 3, 10),
            _row(1, 1010.0, 3, 12),
            _row(2, 1020.0, 1, 14),
        ]

        summary = ab_mod.summarize(rows, "OFF")

        self.assertEqual(summary["episodes"], 3)
        self.assertAlmostEqual(summary["priority_frac"][3], 2 / 3)
        self.assertAlmostEqual(summary["priority_frac"][1], 1 / 3)
        self.assertAlmostEqual(summary["fusion"]["mean"], 12.0)
        self.assertFalse(ab_mod.is_nan(summary["fusion"]["mean"]))

    def test_speed_verdict_uses_end_to_end_throughput_not_policy_timer(self):
        off = ab_mod.summarize(
            [
                _row(0, 1000.0, 3, 10, policy=0.4),
                _row(1, 1010.0, 3, 12, policy=0.4),
                _row(2, 1020.0, 3, 11, policy=0.4),
            ],
            "OFF",
        )
        on = ab_mod.summarize(
            [
                _row(0, 2000.0, 3, 10, policy=0.1),
                _row(1, 2020.0, 3, 12, policy=0.1),
                _row(2, 2040.0, 3, 11, policy=0.1),
            ],
            "ON",
        )

        verdict = ab_mod.compare_summaries(off, on)

        self.assertEqual(verdict["quality"], "MATCHED")
        self.assertLess(verdict["end_to_end_speedup"], 1.0)
        self.assertEqual(verdict["speed"], "NOT EFFECTIVE")
        self.assertGreater(verdict["policy_rollout_speedup"], 1.0)

    def test_terminal_probe_bound_uses_busiest_device_allocation(self):
        rows = [
            _row(0, 1000.0, 3, 10, probe=10.0),
            _row(1, 1010.0, 3, 10, probe=5.0),
        ]

        summary = ab_mod.summarize(rows, "OFF")

        # trial split is [3, 2], so cuda:0 gets 60% of each probe wall.
        self.assertAlmostEqual(summary["terminal_probe_critical_path_seconds"], 9.0)

    def test_current_batched_artifact_priority_and_fusion_are_not_nan(self):
        artifact = (
            _REPO
            / "experiments"
            / "server_command_runs"
            / "stage2_batched_rollout_validate_20260621_204023"
        )
        off_path = artifact / "off_episodes.jsonl"
        on_path = artifact / "on_episodes.jsonl"
        if not off_path.exists() or not on_path.exists():
            self.skipTest("local batched rollout A/B artifact is not present")

        off = ab_mod.summarize(ab_mod._load(str(off_path)), "OFF")
        on = ab_mod.summarize(ab_mod._load(str(on_path)), "ON")

        for summary in (off, on):
            self.assertFalse(ab_mod.is_nan(summary["priority_frac"][3]))
            self.assertFalse(ab_mod.is_nan(summary["fusion"]["mean"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
