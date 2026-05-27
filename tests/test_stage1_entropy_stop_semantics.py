import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAYER_EVALUATOR = ROOT / "layer_importance_evaluator.py"
RL_TUNE = ROOT / "rl_tune.py"
LAUNCHER = ROOT / "llama_7B_LayerImportance.sh"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class Stage1EntropyStopSemanticsTest(unittest.TestCase):
    def test_rl_tune_exposes_and_passes_stage1_entropy_stop_threshold(self):
        source = _source(RL_TUNE)

        self.assertIn("stage1_entropy_stop_threshold", source)
        self.assertIn("parse_optional_positive_float", source)
        self.assertIn(
            "stage1_entropy_stop_threshold=stage1_entropy_stop_threshold",
            source,
        )

    def test_launcher_exposes_stage1_entropy_stop_threshold_flag(self):
        source = _source(LAUNCHER)

        self.assertIn("--stage1-entropy-stop-threshold", source)
        self.assertIn("STAGE1_ENTROPY_STOP_THRESHOLD", source)
        self.assertIn("--stage1_entropy_stop_threshold", source)

    def test_stage1_loop_stops_cleanly_when_entropy_drops_below_threshold(self):
        source = _source(LAYER_EVALUATOR)

        self.assertIn("stage1_entropy_stop_threshold", source)
        self.assertIn("stage1_stop_reason = \"entropy_converged\"", source)
        self.assertIn("completed_episodes", source)
        self.assertIn("stop_reason", source)
        self.assertIn("Stage-1 entropy convergence reached", source)


if __name__ == "__main__":
    unittest.main()
