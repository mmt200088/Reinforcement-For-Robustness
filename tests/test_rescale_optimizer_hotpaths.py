import inspect
from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESCALE_ROOT = REPO_ROOT / "Rescale_optimizer"
if str(RESCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(RESCALE_ROOT))


class RescaleOptimizerHotPathTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
