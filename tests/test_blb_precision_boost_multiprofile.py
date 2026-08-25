"""Precision-boost topology resolution must generalize to all fine-tuned profiles.

block2's Rescale_optimizer graph key is profile-suffixed (``block2_<profile>``),
but its modulus-chain STRUCTURE is profile-independent — verified torch-free: the
``cut_point_sf`` / ``propagation_deltas`` node lists of ``block2_<profile>`` are
identical across mrpc / rte / sst2 and their ``_large`` variants (only the SF
VALUES differ, calibrated per model). So a single block2 topology must resolve for
every profile; otherwise the precision boost silently skips block2 on every model
except mrpc.

block4 / block5_n* keys are NOT profile-suffixed (shared graph names), so they
match exactly. block1 is profile-suffixed but is never boosted (no topology).

Torch-free: ``precision_boost`` imports only stdlib + ``rescale_optimizer`` lazily.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl"), str(_REPO / "configs/preparation/rescale")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rfr.search.common import precision_boost as pb

PROFILES = ["mrpc", "rte", "sst2", "mrpc_large", "rte_large", "sst2_large"]


class TopologyForGraphKeyTest(unittest.TestCase):
    def test_block2_resolves_for_every_profile(self):
        for pf in PROFILES:
            topo = pb.topology_for_graph_key(f"block2_{pf}")
            self.assertIsNotNone(topo, f"block2_{pf} did not resolve to a topology")

            self.assertIs(topo, pb.BLOCK2_MRPC_TOPOLOGY)

    def test_profile_agnostic_keys_match_exactly(self):
        self.assertIs(pb.topology_for_graph_key("block4"), pb.BLOCK4_MRPC_TOPOLOGY)
        self.assertIs(pb.topology_for_graph_key("block5_n1"), pb.BLOCK5_N1_MRPC_TOPOLOGY)
        self.assertIs(pb.topology_for_graph_key("block5_n2"), pb.BLOCK5_N2_MRPC_TOPOLOGY)
        self.assertIs(pb.topology_for_graph_key("block5_n4"), pb.BLOCK5_N4_MRPC_TOPOLOGY)

    def test_block1_never_boosted(self):


        for pf in PROFILES:
            self.assertIsNone(pb.topology_for_graph_key(f"block1_{pf}"))

    def test_unknown_key_is_none(self):
        self.assertIsNone(pb.topology_for_graph_key("block3_exp_n4"))


if __name__ == "__main__":
    unittest.main()
