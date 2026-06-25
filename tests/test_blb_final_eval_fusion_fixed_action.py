import importlib.util
import pathlib
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
