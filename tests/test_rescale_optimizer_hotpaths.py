import inspect
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RESCALE_ROOT = REPO_ROOT / "Rescale_optimizer"
if str(RESCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(RESCALE_ROOT))


class RescaleOptimizerHotPathTests(unittest.TestCase):
    def test_delta_state_restore_does_not_deepcopy_scalar_fields(self):
        from rescale_optimizer import replan_interface

        graph = SimpleNamespace(nodes=[
            SimpleNamespace(scale_delta_bits=7, other_ct_scale_bits=31),
            SimpleNamespace(scale_delta_bits=0, other_ct_scale_bits=None),
        ])

        with mock.patch("copy.deepcopy", side_effect=AssertionError("scalar deepcopy")):
            state = replan_interface._snapshot_graph_delta_state(graph)
            graph.nodes[0].scale_delta_bits = 99
            graph.nodes[0].other_ct_scale_bits = 42
            replan_interface._restore_graph_delta_state(graph, state)

        self.assertEqual(state, [(7, 31), (0, None)])
        self.assertEqual(graph.nodes[0].scale_delta_bits, 7)
        self.assertEqual(graph.nodes[0].other_ct_scale_bits, 31)

    def test_reachability_uses_stage_successor_adjacency(self):
        from rescale_optimizer import reachability

        source = inspect.getsource(reachability.compute_reachability)

        self.assertNotIn("for (ii, v) in graph.stage_edges.keys()", source)

    def test_backward_dp_uses_stage_successor_adjacency(self):
        from rescale_optimizer import backward_level_dp

        source = inspect.getsource(backward_level_dp.build_dp_table)
        dp_loop = source.split("# Process cut points", 1)[1]

        self.assertIn("stage_successor_edges", source)
        self.assertNotIn("graph.stage_edges.items()", dp_loop)

    def test_feasibility_dag_uses_incremental_stage_accumulation(self):
        from rescale_optimizer import feasibility

        source = inspect.getsource(feasibility.build_feasibility_dag)

        self.assertNotIn("cumulative_nodes = cumulative_nodes +", source)
        self.assertNotIn("cum = cum +", source)
        self.assertNotIn("tail_cum = tail_cum +", source)
        self.assertNotIn("propagate_scale(t_i, cumulative_nodes)", source)
        self.assertNotIn("propagate_scale(t_i, cum)", source)
        self.assertNotIn("for n in cumulative_nodes)", source)

    def test_feasibility_dag_precomputes_cut_point_indices(self):
        from rescale_optimizer import feasibility

        source = inspect.getsource(feasibility.build_feasibility_dag)

        self.assertIn("cut_point_index_by_node_id", source)
        self.assertNotIn("_cut_point_index(graph, node)", source)


if __name__ == "__main__":
    unittest.main()
