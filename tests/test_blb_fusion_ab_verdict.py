"""Verdict-logic lock for scripts/blb_fusion_ab_compare.py (torch-free).

Locks the 2026-06-10 lesson from the real 6k-episode curriculum A/B: when both
arms avoid accuracy collapse, the verdict must judge SEARCH PROGRESS (best P3
reward + when it was found), not tail mean reward — the OFF arm "won" the tail
mean only by parking at baseline (tail fusion=0, all best candidates before
ep 1000), which is exploration collapse, not a better search. The comparator
must also report P2 (the missing 11% that made P1+P3 look like 89%).
"""
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

_REPO = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "blb_fusion_ab_compare", str(_REPO / "scripts" / "blb_fusion_ab_compare.py")
)
abc_mod = importlib.util.module_from_spec(_spec)
sys.modules["blb_fusion_ab_compare"] = abc_mod
_spec.loader.exec_module(abc_mod)


def _episode(ep, priority, reward, fusion, loss=0.3):
    return {
        "episode": ep,
        "terminal_priority": priority,
        "total_reward": reward,
        "fusion_count": fusion,
        "terminal_loss_mean": loss,
        "terminal_metric1_mean": 0.866,
        "invalid_steps": 0,
        "safe_neighbor_active": False,
        "safe_neighbor_radius": 0,
    }


def _make_on_arm(n=6000, anchor=80):
    """ON-like: keeps exploring; best P3 found late; small P1/P2 tax."""
    eps = []
    for i in range(n):
        if i < anchor:
            eps.append(_episode(i, 3, 45.0, 0))
        elif i % 37 == 0:
            eps.append(_episode(i, 1, -5.0, 18, loss=2.5))
        elif i % 11 == 0:
            eps.append(_episode(i, 2, 20.0, 12))
        else:
            # rewards keep improving with episode index → best is late
            eps.append(_episode(i, 3, 38.0 + 2.6 * (i / n), 15))
    return eps


def _make_off_arm(n=6000, anchor=80):
    """OFF-like: early burst then parks at baseline (fusion 0) forever."""
    eps = []
    for i in range(n):
        if i < anchor:
            eps.append(_episode(i, 3, 45.0, 0))
        elif i < 1000:
            if i % 9 == 0:
                eps.append(_episode(i, 2, 19.5, 10))
            else:
                eps.append(_episode(i, 3, 40.1, 13))  # early best, never beaten
        else:
            eps.append(_episode(i, 3, 38.9, 0))  # parked at baseline: high mean!
    return eps


class SummarizeP2Test(unittest.TestCase):
    def test_p2_reported_and_priorities_sum_to_one(self):
        s = abc_mod.summarize(_make_on_arm(), anchor=80)
        self.assertIn("post_p2", s)
        self.assertIn("tail_p2", s)
        self.assertGreater(s["post_p2"], 0.0)
        self.assertAlmostEqual(s["post_p1"] + s["post_p2"] + s["post_p3"], 1.0, places=6)

    def test_best_p3_progress_fields(self):
        s_on = abc_mod.summarize(_make_on_arm(), anchor=80)
        s_off = abc_mod.summarize(_make_off_arm(), anchor=80)
        # ON's best P3 is found late; OFF's in the first third.
        self.assertGreater(s_on["best_p3_episode"], 4000)
        self.assertLess(s_off["best_p3_episode"], 2000)
        self.assertGreater(s_on["best_p3_reward"], s_off["best_p3_reward"])


class VerdictSearchProgressTest(unittest.TestCase):
    def test_off_must_not_win_on_tail_mean(self):
        s_on = abc_mod.summarize(_make_on_arm(), anchor=80)
        s_off = abc_mod.summarize(_make_off_arm(), anchor=80)
        # Precondition replicating the real A/B trap: OFF has the better tail
        # mean reward and zero tail P1 — the metric the old verdict used.
        self.assertGreater(s_off["tail_mean_reward"], s_on["tail_mean_reward"])
        self.assertEqual(s_off["tail_p1"], 0.0)
        verdict = abc_mod._verdict(s_on, s_off, "curriculum ON", "curriculum OFF")
        self.assertIn("curriculum ON</b> wins", verdict)
        self.assertIn("exploration collapse", verdict)

    def test_collapse_branch_still_fires(self):
        # The original "OFF collapses into sustained P1" detection must survive.
        s_on = abc_mod.summarize(_make_on_arm(), anchor=80)
        collapsed = [_episode(i, 3, 45.0, 0) for i in range(80)] + [
            _episode(i, 1, -5.0, 20, loss=100.0) for i in range(80, 6000)
        ]
        s_bad = abc_mod.summarize(collapsed, anchor=80)
        verdict = abc_mod._verdict(s_on, s_bad, "curriculum ON", "curriculum OFF")
        self.assertIn("Curriculum helps", verdict)


class StreamingMainTest(unittest.TestCase):
    def _write_run(self, root, label, rows):
        run_dir = root / label / "diagnostics"
        run_dir.mkdir(parents=True)
        with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return run_dir.parent

    def test_main_streams_ordered_episode_files_without_load_episodes(self):
        rows_a = _make_on_arm(n=260, anchor=80)
        rows_b = _make_off_arm(n=260, anchor=80)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            run_a = self._write_run(root, "a", rows_a)
            run_b = self._write_run(root, "b", rows_b)
            out = root / "ab.html"

            with mock.patch.object(abc_mod, "load_episodes", side_effect=AssertionError("should stream")):
                with mock.patch.object(abc_mod, "_try_plots", return_value=[]):
                    abc_mod.main([
                        "--run-a",
                        str(run_a),
                        "--run-b",
                        str(run_b),
                        "--out",
                        str(out),
                        "--window",
                        "50",
                    ])

            self.assertTrue(out.is_file())
            payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(payload["summary_a"]["n_total"], len(rows_a))
            self.assertEqual(payload["summary_b"]["n_total"], len(rows_b))

    def test_streaming_analysis_matches_legacy_list_analysis_for_ordered_jsonl(self):
        rows = _make_on_arm(n=260, anchor=80)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            run = self._write_run(root, "a", rows)

            summary, windows = abc_mod.analyze_episodes(str(run), anchor=80, window=50)

        self.assertEqual(summary, abc_mod.summarize(rows, anchor=80))
        self.assertEqual(windows, abc_mod.window_stats(rows, window=50))

    def test_ordered_analysis_reads_episode_file_twice(self):
        rows = _make_on_arm(n=260, anchor=80)
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            run = self._write_run(root, "a", rows)
            original_iter = abc_mod._iter_episode_rows
            pass_count = 0

            def counting_iter(path):
                nonlocal pass_count
                pass_count += 1
                yield from original_iter(path)

            with mock.patch.object(abc_mod, "_iter_episode_rows", counting_iter):
                abc_mod.analyze_episodes(str(run), anchor=80, window=50)

        self.assertEqual(pass_count, 2)

    def test_iter_episode_rows_passes_original_line_to_json_loads(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "episodes.jsonl"
            path.write_text('{"episode": 0}\n   \n{"episode": 1}\n', encoding="utf-8")
            original_loads = abc_mod.json.loads
            seen_inputs = []

            def tracking_loads(raw):
                seen_inputs.append(raw)
                return original_loads(raw)

            with mock.patch.object(abc_mod.json, "loads", tracking_loads):
                rows = list(abc_mod._iter_episode_rows(str(path)))

        self.assertEqual([row["episode"] for row in rows], [0, 1])
        self.assertEqual(seen_inputs, ['{"episode": 0}\n', '{"episode": 1}\n'])

    def test_load_best_action_uses_common_path_without_os_walk(self):
        with tempfile.TemporaryDirectory() as td:
            run = pathlib.Path(td)
            path = run / "blb_stage2" / "progress" / "blb_stage2_best_action_full.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"slots": [{"slot": 1}]}), encoding="utf-8")

            with mock.patch.object(
                abc_mod.os,
                "walk",
                side_effect=AssertionError("common best-action path should avoid recursive walk"),
            ):
                payload = abc_mod._load_best_action(str(run))

        self.assertEqual(payload, {"slots": [{"slot": 1}]})

    def test_window_stats_chunk_scans_once_without_temp_list_helpers(self):
        chunk = [
            {
                "episode": 10,
                "terminal_priority": 1,
                "total_reward": 3.0,
                "fusion_count": 0,
                "terminal_loss_mean": 100.0,
                "terminal_metric1_mean": 0.7,
                "safe_neighbor_active": True,
                "safe_neighbor_radius": 1,
                "invalid_steps": 0,
            },
            {
                "episode": 11,
                "terminal_priority": 2,
                "total_reward": 6.0,
                "fusion_count": 2,
                "terminal_loss_mean": 0.3,
                "terminal_metric1_mean": 0.8,
                "safe_neighbor_active": False,
                "safe_neighbor_radius": 2,
                "invalid_steps": 1,
            },
            {
                "episode": 12,
                "terminal_priority": 3,
                "total_reward": 9.0,
                "fusion_count": 4,
                "terminal_loss_mean": 0.4,
                "terminal_metric1_mean": 0.9,
                "safe_neighbor_active": True,
                "safe_neighbor_radius": 3,
                "invalid_steps": 2,
            },
        ]

        with mock.patch.object(abc_mod, "_mean", side_effect=AssertionError("no temp mean list")):
            with mock.patch.object(abc_mod, "_frac", side_effect=AssertionError("no temp frac list")):
                stats = abc_mod._window_stats_chunk(chunk, fallback_offset=0)

        self.assertEqual(stats["ep_lo"], 10)
        self.assertEqual(stats["ep_hi"], 12)
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["reward"], 6.0)
        self.assertAlmostEqual(stats["p1"], 1 / 3)
        self.assertAlmostEqual(stats["p2"], 1 / 3)
        self.assertAlmostEqual(stats["p3"], 1 / 3)
        self.assertAlmostEqual(stats["fusion"], 2.0)
        self.assertAlmostEqual(stats["loss_cap"], 1 / 3)
        self.assertAlmostEqual(stats["m1"], 0.8)
        self.assertAlmostEqual(stats["sn_active"], 2 / 3)
        self.assertAlmostEqual(stats["sn_radius"], 2.0)
        self.assertAlmostEqual(stats["invalid"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
