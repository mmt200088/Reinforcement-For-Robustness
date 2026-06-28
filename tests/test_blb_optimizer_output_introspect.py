"""fused_skeleton_positions identifies which baseline-skeleton rescale positions
the optimizer FUSED AWAY, from the replan 'new_compact_config'.

Needed to fix the block2 runtime-install Q2 check. Q2 asserted "a fusing option
nulls >=1 rescale" by counting `rescale_fused_away` overrides — but that override
is only recorded when the fused rescale was an INSTALLED noise point to begin
with. block2's fused rescale (gama1 / gamma_result_rescale) is structurally None
at runtime (active-set filter — it is not a noise-install point; only the V-side
kt_mask2/q_mask2 rescales install), so the optimizer fuses it but there is nothing
to null → no override → Q2 false-failed. The correct invariant is "every FUSED
rescale installs no noise" — verified by checking the cfg field at each fused
skeleton position is None, which this pure helper locates.

Pure (torch-free): compact is the replan dict, baseline_skeleton the node-id
sequence, skel_field_specs the (cfg_field, tuple_index) per skeleton position.
Mirrors apply_optimizer_output_to_cfg's fused detection (cut_point absent OR a
passthrough with no sf_post).
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import optimizer_output_introspect as ooi


class FusedSkeletonPositionsTest(unittest.TestCase):
    # block2 fc=1 skeleton: r0 source (i=0), r1 gama1 (i=2), r2 kt_mask1 (i=4),
    # r3 qkt_matmul (i=6). Boosted-fc1 fusion makes gama1 a passthrough.
    SKEL = [("inv_std_fresh", None), ("gamma_result_rescale", None),
            ("kt_mask1_result_rescale", None), ("qkt_matmul_result_rescale", None)]
    BASELINE = [0, 2, 4, 6]

    @staticmethod
    def _compact(i_to_entry):
        return {"cut_point_sf": [dict(i=i, **e) for i, e in i_to_entry.items()]}

    def test_block2_fc1_gama1_passthrough_is_fused(self):
        # gama1 (i=2) passthrough: sf set, NO sf_post -> fused. kt_mask1/qkt survive.
        compact = self._compact(
            {0: {"sf": 21}, 2: {"sf": 57}, 4: {"sf_post": 29}, 6: {"sf_post": 30}}
        )
        fused = ooi.fused_skeleton_positions(compact, self.BASELINE, self.SKEL)
        self.assertEqual(fused, [("gamma_result_rescale", None)])

    def test_absent_cut_point_is_fused(self):
        # i=4 absent entirely -> fused.
        compact = self._compact({0: {"sf": 21}, 2: {"sf_post": 28}, 6: {"sf_post": 30}})
        fused = ooi.fused_skeleton_positions(compact, self.BASELINE, self.SKEL)
        self.assertIn(("kt_mask1_result_rescale", None), fused)

    def test_all_surviving_none_fused(self):
        compact = self._compact(
            {0: {"sf": 21}, 2: {"sf_post": 28}, 4: {"sf_post": 29}, 6: {"sf_post": 30}}
        )
        self.assertEqual(ooi.fused_skeleton_positions(compact, self.BASELINE, self.SKEL), [])

    def test_source_position_never_fused(self):
        # r0 (source) is never treated as a rescale even if absent.
        compact = self._compact({2: {"sf_post": 28}, 4: {"sf_post": 29}, 6: {"sf_post": 30}})
        fused = ooi.fused_skeleton_positions(compact, self.BASELINE, self.SKEL)
        self.assertNotIn(("inv_std_fresh", None), fused)

    def test_tuple_index_preserved(self):
        skel = [("x_fresh", None), ("square_rescales", 0), ("square_rescales", 1)]
        baseline = [0, 2, 3]
        compact = self._compact({0: {"sf": 21}, 2: {"sf": 50}, 3: {"sf_post": 40}})
        fused = ooi.fused_skeleton_positions(compact, baseline, skel)
        self.assertEqual(fused, [("square_rescales", 0)])

    def test_empty_compact_returns_empty(self):
        self.assertEqual(ooi.fused_skeleton_positions({}, self.BASELINE, self.SKEL), [])


if __name__ == "__main__":
    unittest.main()
