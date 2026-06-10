"""Verdict-logic lock for scripts/blb_fusion_ab_compare.py (torch-free).

Locks the 2026-06-10 lesson from the real 6k-episode curriculum A/B: when both
arms avoid accuracy collapse, the verdict must judge SEARCH PROGRESS (best P3
reward + when it was found), not tail mean reward — the OFF arm "won" the tail
mean only by parking at baseline (tail fusion=0, all best candidates before
ep 1000), which is exploration collapse, not a better search. The comparator
must also report P2 (the missing 11% that made P1+P3 look like 89%).
"""
import importlib.util
import pathlib
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
