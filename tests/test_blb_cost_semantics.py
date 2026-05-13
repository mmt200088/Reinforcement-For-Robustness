import unittest


class BLBCostSemanticsTests(unittest.TestCase):
    def test_rescale_rank_key_uses_only_total_bits_and_fusion(self):
        from blb_stage2_rl.candidate_store import rescale_cost_rank_key

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
        from blb_stage2_rl.candidate_store import candidate_rank_key, f0_sort_key

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
