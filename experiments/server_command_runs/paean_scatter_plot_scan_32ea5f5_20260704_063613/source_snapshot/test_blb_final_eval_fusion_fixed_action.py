import importlib.util
import inspect
import pathlib
import sys
import unittest

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if p not in sys.path:
        sys.path.insert(0, p)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch required for Paean.blb_action_eval import",
)
class FusionCountFixedActionDecodeTest(unittest.TestCase):
    def test_per_step_fusion_option_replay_preserves_rl_selected_k(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule
        from blb_stage2_rl.action_space import (
            K_LEVELS,
            load_max_sfs,
            make_all_max_action_vector,
            step_schedule,
        )
        from blb_stage2_rl.fusion_count_map import FusionCountMap

        num_layers = 12
        gelu = [4] * num_layers
        softmax = [6] * num_layers
        fusion_map = FusionCountMap.load("mrpc")
        schedule = step_schedule(
            num_layers,
            profile="mrpc",
            attn_degree_per_layer=softmax,
            gelu_degree_per_layer=gelu,
        )
        step = next(s for s in schedule if s.layer_idx == 0 and s.block_idx == 4)
        option_id = 1
        k_index = 2  # K_LEVELS[2] == 11 under the legacy-compatible table.

        action_vec = make_all_max_action_vector(num_layers)
        block_vec = fusion_map.expand(step.graph_key_suffix, option_id, k_index)
        for offset, value in zip(step.full_vec_offsets, block_vec.tolist()):
            action_vec[int(offset)] = int(value)

        metadata = {
            "schema_version": "fusion_count_fixed_action_v1",
            "group": {
                "option_by_step": {str(step.step_idx): option_id},
            },
        }

        module = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        decoded = module._decode_fusion_count_fixed_action(
            action_vec=action_vec,
            metadata=metadata,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=num_layers,
            gelu=gelu,
            softmax=softmax,
            profile="mrpc",
        )

        cfg = decoded.block4_cfgs[0]
        self.assertEqual(cfg.output_truncation_k, int(K_LEVELS[k_index]))
        # block4 option 1 is a BOOSTED option (2026-06-25 precision-boost rebuild):
        # the decode must use explicit_field_values, so these are the boosted SFs
        # (softmax_out_mask 13→14 after the boost). Asserting the boosted values
        # locks that the boost is actually replayed, not the pre-boost grid SFs.
        self.assertEqual(cfg.softmax_out_fresh.scaling_factor, 21)
        self.assertEqual(cfg.softmax_out_mask_encode.scaling_factor, 14)

    def test_selected_vs_random_summary_keeps_existing_statistics(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        module = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        selected = [{
            "name": "selected",
            "loss": 1.1,
            "loss_std": 0.01,
            "p": 0.80,
            "p_std": 0.02,
            "s": 0.70,
            "s_std": 0.03,
            "total_bits_sum": 44,
            "total_fusion_count": 3,
            "avg_truncation_k": 12.0,
        }]
        random_results = [
            {"loss": 1.2, "loss_std": 0.04, "p": 0.70, "p_std": 0.05, "s": 0.65, "s_std": 0.06},
            {"loss": 1.0, "loss_std": 0.02, "p": 0.85, "p_std": 0.01, "s": 0.72, "s_std": 0.04},
            {"loss": 1.3, "loss_std": 0.03, "p": 0.78, "p_std": 0.03, "s": 0.75, "s_std": 0.02},
        ]

        summary = module._summarize_selected_vs_random(
            selected_results=selected,
            random_results=random_results,
            num_metrics=2,
        )

        self.assertEqual(summary["random_count"], 3)
        self.assertEqual(summary["random_stats"]["loss_mean"]["n"], 3)
        self.assertAlmostEqual(summary["random_stats"]["loss_mean"]["mean"], np.mean([1.2, 1.0, 1.3]))
        self.assertAlmostEqual(summary["random_stats"]["loss_mean"]["std"], np.std([1.2, 1.0, 1.3]))
        self.assertAlmostEqual(summary["random_stats"]["metric1_mean"]["max"], 0.85)
        ranks = summary["anchor_rank_vs_random"]
        self.assertEqual(ranks["metric1_higher_better"]["rank_better_than_selected"], 2)
        self.assertEqual(ranks["loss_lower_better"]["rank_better_than_selected"], 2)
        self.assertEqual(ranks["metric2_higher_better"]["rank_better_than_selected"], 1)

    def test_selected_vs_random_summary_streams_random_rows_once(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._summarize_selected_vs_random)

        self.assertNotIn("np.asarray([float(r.get(key, 0.0)) for r in rows]", source)
        self.assertNotIn("metric_rows = [", source)
        self.assertNotIn("loss_rows = [", source)
        self.assertNotIn("metric2_rows = [", source)

    def test_results_plot_scans_candidate_rows_once(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._save_results_plot)

        self.assertNotIn('np.asarray([float(r["loss"]) for r in candidate_results]', source)
        self.assertNotIn("np.asarray([float(r.get(\"loss_std\", 0.0)) for r in candidate_results]", source)
        self.assertNotIn('np.asarray([float(r["p"]) for r in candidate_results]', source)
        self.assertNotIn("np.asarray([float(r.get(\"p_std\", 0.0)) for r in candidate_results]", source)
        self.assertNotIn('np.asarray([float(r["total_bits_sum"]) for r in candidate_results]', source)
        self.assertNotIn('np.asarray([float(r["time_ms"]) for r in candidate_results]', source)

    def test_scatter_plot_scans_result_rows_once_per_group(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        source = inspect.getsource(BLBActionFinalEvaluationModule._save_scatter_plot)

        self.assertNotIn("def _xs_ys", source)
        self.assertNotIn('[float(r.get("p", 0.0)) for r in rows]', source)
        self.assertNotIn('[float(r.get("p_std", 0.0)) for r in rows]', source)
        self.assertNotIn('[float(r.get("s", 0.0)) for r in random_results]', source)
        self.assertNotIn('[float(r.get("s_std", 0.0)) for r in random_results]', source)
        self.assertNotIn('[float(r.get("s", 0.0)) for r in selected_results]', source)
        self.assertNotIn('[float(r.get("s_std", 0.0)) for r in selected_results]', source)


if __name__ == "__main__":
    unittest.main()
