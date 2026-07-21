"""GLUE submission must install the SAME boosted config the RL search selected.

The persisted best is a flat grid-index vector that cannot carry the precision
boost. The GLUE generator therefore decodes a ``fusion_count_fixed_action_v1``
config (boost replayed from the option's ``explicit_field_values``) AND applies
the Rescale_optimizer override (fused rescales → None), exactly like the
validation-set final eval and the training terminal probe.

This is torch-gated: the decode helper imports ``action_space`` (pulls torch) and
``Paean.blb_action_eval`` + the real Rescale_optimizer. Runs on the server.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl"), str(_REPO / "Rescale_optimizer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch required for the GLUE BLB decode helper",
)
class GlueBoostInstallTest(unittest.TestCase):
    def test_calibrated_baseline_slots_round_trip_ineffective_block1_k(self):
        import numpy as np

        from blb_stage2_rl.action_io import (
            action_vec_to_slots_list,
            slots_list_to_action_vec,
        )
        from blb_stage2_rl.baseline_bootstrap import (
            load_calibrated_stage2_action_context,
        )

        num_layers = 12
        gelu = [4] * num_layers
        softmax = [6] * num_layers
        context = load_calibrated_stage2_action_context(
            rescale_optimizer_root="Rescale_optimizer",
            dataset="mrpc",
            num_layers=num_layers,
            gelu_per_layer=gelu,
            softmax_per_layer=softmax,
            snap_sf_to_noise_table=False,
        )
        action = np.asarray(context.baseline_action_vec, dtype=int)
        slots = action_vec_to_slots_list(
            action,
            max_sfs=context.max_sfs,
            num_layers=num_layers,
            gelu_degree=gelu,
            attn_degree=softmax,
            profile="mrpc",
        )

        round_trip, notes = slots_list_to_action_vec(
            slots,
            max_sfs=context.max_sfs,
            num_layers=num_layers,
            gelu_degree=gelu,
            attn_degree=softmax,
            base_action_vec=action,
        )

        np.testing.assert_array_equal(round_trip, action)
        self.assertEqual(notes, [])

    def test_decode_helper_installs_boosted_block2_output(self):
        import numpy as np

        from blb_stage2_rl.action_space import make_all_max_action_vector
        from blb_stage2_rl.baseline_bootstrap import (
            load_calibrated_stage2_action_context,
        )
        from blb_stage2_rl.fusion_count_map import FusionCountMap
        from blb_stage2_rl.fusion_fixed_action import build_fusion_fixed_config
        from generate_glue_submission import _decode_blb_action_for_glue

        num_layers = 12
        gelu = [4] * num_layers
        softmax = [6] * num_layers
        fusion_map = FusionCountMap.load("mrpc")
        action_context = load_calibrated_stage2_action_context(
            rescale_optimizer_root="Rescale_optimizer",
            dataset="mrpc",
            num_layers=num_layers,
            gelu_per_layer=gelu,
            softmax_per_layer=softmax,
            snap_sf_to_noise_table=False,
        )

        # Pick block2's boosted fusion option (option 1) and splice it into a
        # baseline action vector so the flat vec carries its (pre-boost) indices.
        from blb_stage2_rl.action_space import step_schedule

        schedule = step_schedule(
            num_layers, profile="mrpc",
            attn_degree_per_layer=softmax, gelu_degree_per_layer=gelu,
        )
        step = next(s for s in schedule if s.layer_idx == 1 and s.block_idx == 2)
        graph = fusion_map.graphs[step.graph_key_suffix]
        boosted = next(o for o in graph.options if o.boosted and o.explicit_field_values)

        action_vec = make_all_max_action_vector(num_layers)
        block_vec = fusion_map.expand(step.graph_key_suffix, int(boosted.option_id), 0)
        for off, val in zip(step.full_vec_offsets, block_vec.tolist()):
            action_vec[int(off)] = int(val)

        cfg = build_fusion_fixed_config(
            action_vec, profile="mrpc", num_layers=num_layers,
            gelu=gelu, softmax=softmax, fusion_map=fusion_map,
        )
        fusion_metadata = {"schema_version": "fusion_count_fixed_action_v1", "group": cfg["group"]}

        decoded = _decode_blb_action_for_glue(
            action_vec=np.asarray(action_vec, dtype=int),
            fusion_metadata=fusion_metadata,
            profile="mrpc",
            gelu_degrees=gelu,
            softmax_degrees=softmax,
            max_sfs=action_context.max_sfs,
        )
        self.assertRegex(decoded.final_config_fingerprint, r"^[0-9a-f]{64}$")
        self.assertTrue(
            decoded.replan_application["model_uses_replan_config"],
            decoded.replan_application,
        )

        # The installed block2 layer-1 cfg must carry a BOOSTED SF (above the grid
        # baseline). The boost raises an encode/output SF beyond what the flat
        # index decode (action_vector_to_cfgs) would produce.
        from blb_stage2_rl.action_space import action_vector_to_cfgs

        preboost = action_vector_to_cfgs(
            action_vec=np.asarray(action_vec, dtype=int),
            max_sfs=action_context.max_sfs,
            num_layers=num_layers,
            gelu_degree=np.asarray(gelu, dtype=int),
            attn_degree=np.asarray(softmax, dtype=int),
        )

        decoded_block2 = decoded.block2_cfgs[1]
        preboost_block2 = preboost.block2_cfgs[1]

        # Do not compare the sum of all SFs: fusion intentionally removes some
        # rescale points (None), which can lower the total even while the
        # effective boosted points are installed at higher precision.
        boosted_fields = [
            "kt_mask1_result_rescale",
            "q_mask1_result_rescale",
            "qkt_matmul_result_rescale",
            "qkt_merge_mask_encode",
        ]
        higher_fields = []
        for field in boosted_fields:
            boosted_sf = getattr(getattr(decoded_block2, field), "scaling_factor", None)
            preboost_sf = getattr(getattr(preboost_block2, field), "scaling_factor", None)
            if boosted_sf is not None and preboost_sf is not None and int(boosted_sf) > int(preboost_sf):
                higher_fields.append(field)
        self.assertTrue(
            higher_fields,
            "GLUE decode did not install any effective boosted block2 SF above pre-boost",
        )
        self.assertIsNone(
            getattr(decoded_block2.gamma_result_rescale, "scaling_factor", None),
            "GLUE decode did not preserve fused-away block2 gamma rescale",
        )
        block3 = decoded.block3_cfgs[0]
        self.assertEqual(block3.x_fresh.scaling_factor, 31)
        self.assertEqual(block3.inv_2n_encode.scaling_factor, 15)
        self.assertEqual(
            [entry.scaling_factor for entry in block3.square_rescales],
            [35] * 6,
        )


if __name__ == "__main__":
    unittest.main()
