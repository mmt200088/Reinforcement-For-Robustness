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

    def test_stage1_can_run_unbounded_until_entropy_convergence(self):
        evaluator_source = _source(LAYER_EVALUATOR)
        rl_tune_source = _source(RL_TUNE)
        launcher_source = _source(LAUNCHER)

        self.assertIn("parse_stage1_episode_limit", rl_tune_source)
        self.assertIn("stage1_rl_unbounded_until_entropy", evaluator_source)
        self.assertIn("itertools.count(stage1_resume_start_episode)", evaluator_source)
        self.assertIn("Stage-1 RL 进度 · 回合 {episode + 1} / entropy<", evaluator_source)
        self.assertIn('STAGE1_EPISODES="0"', launcher_source)
        self.assertIn('STAGE1_ENTROPY_STOP_THRESHOLD="0.1"', launcher_source)

    def test_stage1_evaluation_protocol_is_plaintext_only(self):
        source = _source(LAYER_EVALUATOR)
        stage1_eval = source.split("    def stage1_evaluate(", 1)[1].split(
            "    def _stage1_evaluate_on_model(", 1
        )[0]
        worker_eval = source.split("    def _stage1_evaluate_on_model(", 1)[1].split(
            "    def stage1_final_evaluate(", 1
        )[0]
        stage1_final_eval = source.split("    def stage1_final_evaluate(", 1)[1].split(
            "    def build_constraint_limits_from_metrics(", 1
        )[0]

        self.assertNotIn("stage1_use_max_scaling_noise_env", stage1_eval)
        self.assertNotIn("evaluate_model_with_attention_noise", stage1_eval)
        self.assertNotIn("noise_env_enabled", worker_eval)
        self.assertNotIn("replace_layer_input_noise", worker_eval)
        self.assertNotIn("replace_layer_query_noise", worker_eval)
        self.assertNotIn("replace_layer_softmax_value_noise", worker_eval)
        self.assertNotIn("final_eval_use_max_scaling_noise_env", stage1_final_eval)
        self.assertNotIn("evaluate_model_with_attention_noise", stage1_final_eval)


if __name__ == "__main__":
    unittest.main()
