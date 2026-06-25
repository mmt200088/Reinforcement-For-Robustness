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
    def test_decode_helper_installs_boosted_block2_output(self):
        import numpy as np

        from blb_stage2_rl.action_space import load_max_sfs, make_all_max_action_vector
        from blb_stage2_rl.fusion_count_map import FusionCountMap
        from blb_stage2_rl.fusion_fixed_action import build_fusion_fixed_config
        from generate_glue_submission import _decode_blb_action_for_glue

        num_layers = 12
        gelu = [4] * num_layers
        softmax = [6] * num_layers
        fusion_map = FusionCountMap.load("mrpc")

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
            max_sfs=load_max_sfs("mrpc"),
        )

        # The installed block2 layer-1 cfg must carry a BOOSTED SF (above the grid
        # baseline). The boost raises an encode/output SF beyond what the flat
        # index decode (action_vector_to_cfgs) would produce.
        from blb_stage2_rl.action_space import action_vector_to_cfgs

        preboost = action_vector_to_cfgs(
            action_vec=np.asarray(action_vec, dtype=int),
            max_sfs=load_max_sfs("mrpc"),
            num_layers=num_layers,
            gelu_degree=np.asarray(gelu, dtype=int),
            attn_degree=np.asarray(softmax, dtype=int),
        )

        def _sf_sum(cfg_obj):
            total = 0
            for attr in vars(cfg_obj).values():
                sf = getattr(attr, "scaling_factor", None)
                if isinstance(sf, (int, float)):
                    total += int(sf)
            return total

        boosted_sf_sum = _sf_sum(decoded.block2_cfgs[1])
        preboost_sf_sum = _sf_sum(preboost.block2_cfgs[1])
        self.assertGreater(
            boosted_sf_sum, preboost_sf_sum,
            "GLUE decode did not install the boosted (higher-SF) block2 config",
        )


if __name__ == "__main__":
    unittest.main()
