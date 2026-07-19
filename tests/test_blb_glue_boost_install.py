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

import ast
import importlib.util
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl"), str(_REPO / "Rescale_optimizer")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class GlueNoiseSeedWiringStaticTest(unittest.TestCase):
    def test_blb_seed_reseeds_the_independent_noise_rng(self):
        tree = ast.parse((_REPO / "generate_glue_submission.py").read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_seed_all_for_reproducibility"
        )
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "reseed_noise_rng"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, "seed")


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch required for the GLUE BLB decode helper",
)
class GlueBoostInstallTest(unittest.TestCase):
    def test_blb_seed_controls_the_independent_noise_stream(self):
        import torch

        from function_handler import _sample_independent_gaussian, reseed_noise_rng
        from generate_glue_submission import _seed_all_for_reproducibility

        reference = torch.zeros(32)
        try:
            _seed_all_for_reproducibility(2026071901)
            first = _sample_independent_gaussian(reference, 1.0)
            _seed_all_for_reproducibility(2026071901)
            replay = _sample_independent_gaussian(reference, 1.0)
            _seed_all_for_reproducibility(2026071902)
            second = _sample_independent_gaussian(reference, 1.0)
        finally:
            reseed_noise_rng(None)

        self.assertTrue(torch.equal(first, replay))
        self.assertFalse(torch.equal(first, second))

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


if __name__ == "__main__":
    unittest.main()
