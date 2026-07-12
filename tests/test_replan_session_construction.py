from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
RESCALE_ROOT = REPO_ROOT / "Rescale_optimizer"
if str(RESCALE_ROOT) not in sys.path:
    sys.path.insert(0, str(RESCALE_ROOT))

import rescale_optimizer.replan_interface as replan_interface


class ReplanSessionConstructionTest(unittest.TestCase):
    def test_from_profile_reuses_loaded_baselines_for_session_construction(self):
        calls = []
        original_loader = replan_interface.load_static_skeleton_baselines
        original_load_graph = replan_interface.load_graph_from_json
        original_build_dag = replan_interface.build_feasibility_dag

        def fake_loader(path):
            calls.append(Path(path).name)
            return {
                "graph_a": SimpleNamespace(
                    config_name="graph_a",
                    skeleton=[],
                )
            }

        def fake_load_graph(_path):
            return SimpleNamespace(nodes=[]), None, None

        try:
            replan_interface.load_static_skeleton_baselines = fake_loader
            replan_interface.load_graph_from_json = fake_load_graph
            replan_interface.build_feasibility_dag = lambda graph: graph

            with tempfile.TemporaryDirectory() as td:
                cfg_dir = Path(td) / "configs" / "toy"
                cfg_dir.mkdir(parents=True)
                (cfg_dir / "graph_a.json").write_text("{}", encoding="utf-8")

                session = replan_interface.ReplanSession.from_profile(
                    profile="toy",
                    root=td,
                )
        finally:
            replan_interface.load_static_skeleton_baselines = original_loader
            replan_interface.load_graph_from_json = original_load_graph
            replan_interface.build_feasibility_dag = original_build_dag

        self.assertEqual(session.graph_keys, ["graph_a"])
        self.assertEqual(calls, ["static_skeletons_toy.json"])


if __name__ == "__main__":
    unittest.main()
