import unittest
import importlib.util
import sys
from pathlib import Path
from unittest import mock


def load_candidate_store_module():
    path = Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "candidate_store.py"
    spec = importlib.util.spec_from_file_location("candidate_store_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BLBCostSemanticsTests(unittest.TestCase):
    def test_avg_truncation_k_uses_direct_sum_without_numpy_mean(self):
        from blb_stage2_rl import action_space

        num_layers = 2
        action = action_space.make_all_max_action_vector(num_layers=num_layers)
        expected_count = num_layers * 5 - 1
        expected = action_space.sum_truncation_k_in_action(action, num_layers) / expected_count

        with mock.patch.object(action_space.np, "mean", side_effect=AssertionError("avg K should not call np.mean")):
            actual = action_space.avg_truncation_k_in_action(action, num_layers)

        self.assertAlmostEqual(actual, expected)

    def test_truncation_k_helpers_accumulate_without_gather_list(self):
        from blb_stage2_rl import action_space

        num_layers = 3
        action = action_space.make_all_min_action_vector(num_layers=num_layers)
        expected_count = num_layers * 5 - 1
        expected_sum = action_space.sum_truncation_k_in_action(action, num_layers)
        expected_avg = expected_sum / expected_count

        with mock.patch.object(
                action_space,
                "_gather_effective_k_values_in_action",
                side_effect=AssertionError("hot K helpers should not allocate a gathered list"),
        ):
            self.assertEqual(action_space.sum_truncation_k_in_action(action, num_layers), expected_sum)
            self.assertAlmostEqual(action_space.avg_truncation_k_in_action(action, num_layers), expected_avg)

    def test_rescale_rank_key_uses_only_total_bits_and_fusion(self):
        rescale_cost_rank_key = load_candidate_store_module().rescale_cost_rank_key

        base = {
            "rescale_cost": {
                "optimizer_cost_terms": {
                    "total_bits_sum": 1200,
                    "fusion_count": 7,
                }
            },
            "rescale_debug": {
                "optimizer_diagnostic_terms": {
                    "q_bits": [60, 50, 40],
                    "q_head_bits": 60,
                    "q_tail_bits": 40,
                }
            },
        }
        changed_debug = {
            **base,
            "rescale_debug": {
                "optimizer_diagnostic_terms": {
                    "q_bits": [99, 1],
                    "q_head_bits": 99,
                    "q_tail_bits": 1,
                }
            },
        }

        self.assertEqual(rescale_cost_rank_key(base), (1200.0, 7.0))
        self.assertEqual(rescale_cost_rank_key(base), rescale_cost_rank_key(changed_debug))

    def test_invalid_chain_is_validity_gate_not_numeric_cost(self):
        store = load_candidate_store_module()
        candidate_rank_key = store.candidate_rank_key
        f0_sort_key = store.f0_sort_key

        valid_high_cost = {
            "valid": True,
            "rescale_cost": {
                "optimizer_cost_terms": {
                    "total_bits_sum": 9999,
                    "fusion_count": 99,
                }
            },
        }
        invalid_low_cost = {
            "valid": False,
            "rescale_cost": {
                "optimizer_cost_terms": {
                    "total_bits_sum": 1,
                    "fusion_count": 0,
                }
            },
            "rescale_debug": {
                "optimizer_validity_terms": {
                    "invalid_chain": {"reason": "unit"},
                    "optimizer_valid": False,
                    "any_invalid": True,
                }
            },
        }

        self.assertLess(candidate_rank_key(valid_high_cost), candidate_rank_key(invalid_low_cost))
        self.assertEqual(f0_sort_key(valid_high_cost), (0.0, 9999.0, 99.0))
        self.assertEqual(f0_sort_key(invalid_low_cost), (1.0, 1.0, 0.0))
        self.assertGreater(f0_sort_key(invalid_low_cost), f0_sort_key(valid_high_cost))

    def test_candidate_rank_key_uses_stage1_reward_before_cost_rank_without_breaking_priority(self):
        candidate_rank_key = load_candidate_store_module().candidate_rank_key

        p3_capped_low = {
            "valid": True,
            "terminal_priority": 3,
            "acc_violation": 0.0,
            "stability_violation": 0.0,
            "terminal_cost_score": 4.5,
            "terminal_cost_rank_score": 6.0,
            "terminal_fusion_gain": 8.0,
            "terminal_k_gain": 0.5,
            "terminal_bits_gain": 300.0,
            "terminal_reward": 45.0,
            "total_reward": 42.2,
        }
        p3_capped_high = {
            "valid": True,
            "terminal_priority": 3,
            "acc_violation": 0.0,
            "stability_violation": 0.0,
            "terminal_cost_score": 4.5,
            "terminal_cost_rank_score": 10.0,
            "terminal_fusion_gain": 14.0,
            "terminal_k_gain": 1.2,
            "terminal_bits_gain": 500.0,
            "terminal_reward": 45.0,
            "total_reward": 42.0,
        }
        p2_huge_cost = {
            "valid": True,
            "terminal_priority": 2,
            "acc_violation": 0.0,
            "stability_violation": 0.1,
            "terminal_cost_score": 4.5,
            "terminal_cost_rank_score": 999.0,
        }

        self.assertLess(candidate_rank_key(p3_capped_low), candidate_rank_key(p3_capped_high))
        p3_same_reward_higher_cost = dict(p3_capped_high)
        p3_same_reward_higher_cost["total_reward"] = p3_capped_low["total_reward"]
        self.assertLess(candidate_rank_key(p3_same_reward_higher_cost), candidate_rank_key(p3_capped_low))
        self.assertLess(candidate_rank_key(p3_capped_low), candidate_rank_key(p2_huge_cost))

    def test_f0_record_splits_rescale_cost_debug_and_mpc_truncation(self):
        from scripts.blb_eval_action import build_f0_candidate_record

        signals = type(
            "Signals",
            (),
            {
                "any_invalid": False,
                "total_bits_sum": 1200,
                "total_fusion_count": 7,
                "invalid_chains": {},
            },
        )()
        record = build_f0_candidate_record(
            [4, 4, 3, 2],
            source="unit",
            signals=signals,
            baseline_total_bits=2400,
            optimizer_debug={"q_bits": [60, 50], "q_head_bits": 60, "q_tail_bits": 50},
            action_avg_k=12.5,
        )

        self.assertEqual(record["rescale_cost"]["rank_key"], [1200, 7])
        self.assertEqual(record["rescale_cost"]["optimizer_cost_terms"]["total_bits_sum"], 1200)
        self.assertEqual(record["rescale_cost"]["optimizer_cost_terms"]["fusion_count"], 7)
        self.assertEqual(record["rescale_debug"]["optimizer_diagnostic_terms"]["q_bits"], [60, 50])
        self.assertTrue(record["mpc_truncation_cost_enabled"])
        self.assertEqual(record["mpc_truncation_term"]["avg_k"], 12.5)
