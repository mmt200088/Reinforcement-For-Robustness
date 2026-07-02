"""Torch-free tests for the decoupled standalone final-eval layout + samplers.

Covers (spec 2026-05-30-decoupled-standalone-final-eval-design.md §10):
  - build_cost_matched_stage1_configs: exact-cost equality, RL-domain only
    (incl. ReLU=0), dedup, softmax held fixed, shortfall near a cost extreme,
    deterministic under a seed.
  - next_final_eval_number: scan over spaced Paean/stage{1,2}/ dirs.
  - paean_stage_run_dir: run-dir path format.
  - sorted_bar_highlight: sort order + selected-index tracking + 1-based rank.
"""
import os
import tempfile
import unittest

from Paean.final_eval_layout import (
    _gelu_choice_costs_q,
    build_cost_matched_stage1_configs,
    next_final_eval_number,
    paean_stage_run_dir,
    sorted_bar_highlight,
)
from config.constants import GELU_COST, SOFTMAX_COST


def _total_cost(gelu, softmax):
    return sum(GELU_COST[int(g)] for g in gelu) + sum(SOFTMAX_COST[int(s)] for s in softmax)


class Stage1CostMatchTest(unittest.TestCase):
    def test_gelu_choice_costs_are_precomputed_in_choice_order(self):
        costs = _gelu_choice_costs_q((4, 0, 2, 1))
        self.assertEqual(costs.tolist(), [6, -2, 5, 2])

    def test_peers_match_total_cost_exactly_and_exclude_selected(self):
        gelu, softmax = [2, 1, 4], [6, 6, 6]
        target = _total_cost(gelu, softmax)  # 2.5+1.0+3.0 + 9.0 = 15.5
        peers, shortfall = build_cost_matched_stage1_configs(
            gelu, softmax, num_layers=3, count=50, seed=1
        )
        for pg, ps in peers:
            self.assertAlmostEqual(_total_cost(pg, ps), target, places=6)
            self.assertNotEqual(list(pg), list(gelu))  # selected excluded

    def test_only_rl_domain_gelu_degrees_including_relu(self):
        peers, _ = build_cost_matched_stage1_configs([1] * 4, [6] * 4, num_layers=4, count=50, seed=2)
        allowed = {0, 1, 2, 4}
        for pg, _ps in peers:
            self.assertTrue(set(int(g) for g in pg) <= allowed)

    def test_softmax_held_fixed(self):
        peers, _ = build_cost_matched_stage1_configs([2, 1, 4], [6, 6, 6], num_layers=3, count=50, seed=3)
        for _pg, ps in peers:
            self.assertEqual(list(ps), [6, 6, 6])

    def test_no_duplicate_peers(self):
        peers, _ = build_cost_matched_stage1_configs([1] * 5, [6] * 5, num_layers=5, count=50, seed=4)
        seen = {tuple(pg) for pg, _ in peers}
        self.assertEqual(len(seen), len(peers))

    def test_shortfall_at_cost_extreme(self):
        # all-degree-4 is the unique max-gelu-cost vector -> zero peers, full shortfall.
        peers, shortfall = build_cost_matched_stage1_configs([4] * 4, [6] * 4, num_layers=4, count=50, seed=5)
        self.assertEqual(len(peers), 0)
        self.assertEqual(shortfall, 50)

    def test_tiny_domain_finds_all_permutation_peers(self):
        # {4,2,1} (costs 3.0,2.5,1.0) is the only multiset summing to 6.5 over 3 layers
        # -> 3! = 6 ordered vectors, minus the selected = 5 peers.
        peers, shortfall = build_cost_matched_stage1_configs([2, 1, 4], [6, 6, 6], num_layers=3, count=50, seed=6)
        self.assertEqual(len(peers), 5)
        self.assertEqual(shortfall, 45)

    def test_deterministic_under_seed(self):
        a, _ = build_cost_matched_stage1_configs([1] * 6, [6] * 6, num_layers=6, count=20, seed=123)
        b, _ = build_cost_matched_stage1_configs([1] * 6, [6] * 6, num_layers=6, count=20, seed=123)
        self.assertEqual([tuple(g) for g, _ in a], [tuple(g) for g, _ in b])


class NextFinalEvalNumberTest(unittest.TestCase):
    def test_empty_root_returns_1(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(next_final_eval_number(1, "bert-base", "mrpc", d), 1)

    def test_increments_per_combo_independently(self):
        with tempfile.TemporaryDirectory() as d:
            s1 = os.path.join(d, "stage1")
            os.makedirs(os.path.join(s1, "bert base mrpc 1 20260601"))
            os.makedirs(os.path.join(s1, "bert base rte 1 20260601"))
            self.assertEqual(next_final_eval_number(1, "bert-base", "mrpc", d), 2)
            self.assertEqual(next_final_eval_number(1, "bert-base", "rte", d), 2)
            # stage2 is a separate tree
            self.assertEqual(next_final_eval_number(2, "bert-base", "mrpc", d), 1)

    def test_sst2_digit_in_combo_parses(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "stage1", "bert base sst2 3 20260601"))
            self.assertEqual(next_final_eval_number(1, "bert-base", "sst2", d), 4)


class PaeanStageRunDirTest(unittest.TestCase):
    def test_path_format_with_spaces(self):
        path = paean_stage_run_dir(1, "bert-base", "mrpc", "/tmp/paean", n=2, timestamp="20260601")
        self.assertEqual(path, os.path.join("/tmp/paean", "stage1", "bert base mrpc 2 20260601"))

    def test_stage2_subdir(self):
        path = paean_stage_run_dir(2, "bert-large", "rte", "/tmp/paean", n=1, timestamp="20260601_1200")
        self.assertEqual(path, os.path.join("/tmp/paean", "stage2", "bert large rte 1 20260601"))


class SortedBarHighlightTest(unittest.TestCase):
    def test_ascending_lower_is_rank_one(self):
        out = sorted_bar_highlight([0.5, 0.3, 0.9, 0.4], ["a", "b", "c", "d"], selected_idx=1, ascending=True)
        self.assertEqual(out.sorted_values, [0.3, 0.4, 0.5, 0.9])
        self.assertEqual(out.sorted_labels, ["b", "d", "a", "c"])
        self.assertEqual(out.selected_position, 0)  # 'b'=0.3 lands first
        self.assertEqual(out.rank, 1)
        self.assertEqual(out.total, 4)

    def test_descending_higher_is_rank_one(self):
        out = sorted_bar_highlight([0.5, 0.3, 0.9, 0.4], ["a", "b", "c", "d"], selected_idx=2, ascending=False)
        self.assertEqual(out.sorted_values, [0.9, 0.5, 0.4, 0.3])
        self.assertEqual(out.selected_position, 0)  # 'c'=0.9 highest
        self.assertEqual(out.rank, 1)

    def test_selected_worst_rank(self):
        out = sorted_bar_highlight([0.5, 0.3, 0.9, 0.4], ["a", "b", "c", "d"], selected_idx=2, ascending=True)
        self.assertEqual(out.rank, 4)  # 0.9 is worst when lower-is-better


if __name__ == "__main__":
    unittest.main()
