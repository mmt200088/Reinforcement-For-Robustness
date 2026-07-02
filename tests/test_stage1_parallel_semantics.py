from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LAYER_EVALUATOR = ROOT / "layer_importance_evaluator.py"
PARALLEL_RUNNER = ROOT / "stage1_rl" / "parallel_runner.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _method_region(source: str, method_name: str) -> str:
    start = source.index(f"    def {method_name}")
    next_method = source.find("\n    def ", start + 1)
    if next_method == -1:
        next_method = len(source)
    return source[start:next_method]


class Stage1ParallelSemanticsTest(unittest.TestCase):
    def test_parallel_worker_uses_serial_sos_tokens_for_initial_previous_actions(self):
        region = _method_region(_source(LAYER_EVALUATOR), "_stage1_collect_episode_in_worker")

        self.assertIn("SOS_TOKEN_GELU", region)
        self.assertIn("SOS_TOKEN_SOFTMAX", region)
        self.assertNotIn("prev_g = torch.zeros", region)
        self.assertNotIn("prev_s = torch.zeros", region)

    def test_parallel_rollout_logs_per_window_timing_diagnostics(self):
        source = _source(LAYER_EVALUATOR)

        self.assertIn("format_diagnostics_line", source)
        self.assertIn("last_diagnostics", source)

    def test_parallel_runner_exposes_worker_timing_and_speedup_diagnostics(self):
        source = _source(PARALLEL_RUNNER)

        self.assertIn("per_worker_seconds", source)
        self.assertIn("speedup_vs_sequential", source)

    def test_policy_replica_sync_reuses_one_state_dict_per_window(self):
        region = _method_region(_source(PARALLEL_RUNNER), "_sync_policy_replicas")

        self.assertIn("state_dict = None", region)
        self.assertIn("state_dict = gtrxl_net.state_dict()", region)
        self.assertIn("load_state_dict(state_dict)", region)
        self.assertNotIn("load_state_dict(gtrxl_net.state_dict())", region)


if __name__ == "__main__":
    unittest.main()
