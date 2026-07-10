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
    def test_compact_replan_matches_full_result_without_building_full_output(self):
        from rescale_optimizer import CompactReplanResult, ReplanSession
        from rescale_optimizer import replan_interface

        session = ReplanSession.from_profile(
            profile="mrpc",
            root=RESCALE_ROOT,
            include=["block4"],
        )
        full = session.replan("block4")

        with mock.patch.object(
            replan_interface,
            "build_replan_output_dict",
            side_effect=AssertionError("built full replan output"),
        ):
            compact = session.replan_compact("block4")

        self.assertIsInstance(compact, CompactReplanResult)
        self.assertEqual(compact.valid, full["valid"])
        self.assertEqual(compact.fusion_count, full["fusion_count"])
        self.assertEqual(compact.total_bits, full["result"]["chain"]["total_bits"])
        self.assertEqual(compact.compact_config, full["new_compact_config"])

    def test_compact_config_propagates_each_stage_without_nodes_between(self):
        from rescale_optimizer import ReplanSession
        from rescale_optimizer.graph import propagate_scale
        from rescale_optimizer.replan_interface import build_new_compact_config

        session = ReplanSession.from_profile(
            profile="mrpc",
            root=RESCALE_ROOT,
            include=["block4"],
        )
        result = session.replan("block4", return_dict=False)
        self.assertTrue(result.valid)
        graph = session._graphs["block4"]

        skeleton = [int(value) for value in result.skeleton]
        t_vec = [int(value) for value in result.t_final]
        rescale_index_at = {
            skeleton[index]: index
            for index in range(1, result.chain.R + 1)
        }
        expected = []
        for cut_point_index in range(graph.M + 1):
            cut_point = graph.cut_points[cut_point_index]
            row = {
                "i": cut_point_index,
                "name": cut_point.node.name,
                "type": cut_point.node.node_type.name,
            }
            if cut_point_index in rescale_index_at:
                stage_index = rescale_index_at[cut_point_index]
                row.update({
                    "sf_pre": int(propagate_scale(
                        t_vec[stage_index - 1],
                        graph.nodes_between(
                            skeleton[stage_index - 1], cut_point_index,
                        ),
                    )),
                    "sf_post": int(t_vec[stage_index]),
                    "drop": int(result.chain.q_bits[stage_index - 1]),
                })
            elif cut_point_index == skeleton[0]:
                row["sf"] = int(t_vec[0])
            else:
                stage_index = max(
                    index
                    for index in range(result.chain.R + 1)
                    if skeleton[index] <= cut_point_index
                )
                row["sf"] = int(propagate_scale(
                    t_vec[stage_index],
                    graph.nodes_between(skeleton[stage_index], cut_point_index),
                ))
            expected.append(row)

        with mock.patch.object(
            graph,
            "nodes_between",
            side_effect=AssertionError("replayed cut-point path"),
        ):
            compact = build_new_compact_config(graph, "block4", result)

        self.assertEqual(compact["cut_point_sf"], expected)

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
