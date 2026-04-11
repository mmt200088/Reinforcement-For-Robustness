import unittest
from pathlib import Path


class GABudgetAndSplitTests(unittest.TestCase):
    def test_build_stage1_context_uses_validation_full(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        class DummyEvaluator:
            total_layers = 4

            def __init__(self):
                self.last_split = None
                self.last_use_train = None

            def has_dataset_split(self, split_name):
                return split_name == "validation_full"

            def get_reward_reference_split_name(self):
                return "validation_proxy"

            def get_simulated_cost(self, gelu, softmax):
                del gelu, softmax
                return 12.0, 6.0, 6.0

            def stage1_evaluate(self, gelu, softmax, split=None, use_train=False):
                del gelu, softmax
                self.last_split = split
                self.last_use_train = use_train
                return 0.2, 0.8, 0.7, 0.0

        evaluator = DummyEvaluator()
        context = ga_module.build_stage1_context(evaluator, log_fn=None)

        self.assertEqual(context.reward_reference_split, "validation_full")
        self.assertEqual(evaluator.last_split, "validation_full")
        self.assertFalse(evaluator.last_use_train)

    def test_stage1_searcher_prefers_explicit_generation_budget(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        class DummyEvaluator:
            total_layers = 4
            stage1_ga_generations = 7
            stage1_rl_episodes = 999
            step_info_file = str(Path("tmp") / "stage1" / "pruning_search_log.txt")

            def log(self, message):
                del message

        searcher = ga_module.Stage1GeneticSearcher(DummyEvaluator())
        self.assertEqual(searcher.population_size, 32)
        self.assertEqual(searcher.max_generations, 7)

    def test_compare_runner_uses_generations_for_ga_child(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        rl_cmd = compare_runner.build_child_command(
            python_exe="python",
            algorithm="rl",
            side_config=compare_runner.CompareSideConfig(),
            base_model="dummy-model",
            data_path="mrpc",
            run_dir=Path("rl_run"),
            batch_size=16,
            stage1_search_episodes=170,
            stage2_search_episodes=340,
            stage1_search_generations=11,
            stage2_search_generations=13,
            stage1_search_lr="1e-4",
            stage2_search_lr="1e-4",
            random_seed=42,
            perm_trials=10,
            cost_trials=10,
            budget_trials=10,
            noise_eval_repeat=1,
        )
        ga_cmd = compare_runner.build_child_command(
            python_exe="python",
            algorithm="ga",
            side_config=compare_runner.CompareSideConfig(),
            base_model="dummy-model",
            data_path="mrpc",
            run_dir=Path("ga_run"),
            batch_size=16,
            stage1_search_episodes=170,
            stage2_search_episodes=340,
            stage1_search_generations=11,
            stage2_search_generations=13,
            stage1_search_lr="1e-4",
            stage2_search_lr="1e-4",
            random_seed=42,
            perm_trials=10,
            cost_trials=10,
            budget_trials=10,
            noise_eval_repeat=1,
        )

        self.assertIn("--stage1_rl_episodes", rl_cmd)
        self.assertNotIn("--stage1_ga_generations", rl_cmd)
        self.assertIn("--stage1_ga_generations", ga_cmd)
        self.assertIn("--stage2_ga_generations", ga_cmd)
        self.assertNotIn("--stage1_rl_episodes", ga_cmd)

    def test_compare_runner_omits_skipped_search_budget_flags(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        side_config = compare_runner.CompareSideConfig(
            skip_stage1_search=True,
            final_eval_config_source="json",
            final_eval_config_path="glue_configs_best_ppo.json",
            skip_noise_search=True,
            noise_eval_config_source="json",
            noise_eval_config_path="glue_noise_configs_best_genetic.json",
        )
        rl_cmd = compare_runner.build_child_command(
            python_exe="python",
            algorithm="rl",
            side_config=side_config,
            base_model="dummy-model",
            data_path="mrpc",
            run_dir=Path("rl_run"),
            batch_size=16,
            stage1_search_episodes=170,
            stage2_search_episodes=340,
            stage1_search_generations=11,
            stage2_search_generations=13,
            stage1_search_lr="1e-4",
            stage2_search_lr="1e-4",
            random_seed=42,
            perm_trials=10,
            cost_trials=10,
            budget_trials=10,
            noise_eval_repeat=1,
        )
        ga_cmd = compare_runner.build_child_command(
            python_exe="python",
            algorithm="ga",
            side_config=side_config,
            base_model="dummy-model",
            data_path="mrpc",
            run_dir=Path("ga_run"),
            batch_size=16,
            stage1_search_episodes=170,
            stage2_search_episodes=340,
            stage1_search_generations=11,
            stage2_search_generations=13,
            stage1_search_lr="1e-4",
            stage2_search_lr="1e-4",
            random_seed=42,
            perm_trials=10,
            cost_trials=10,
            budget_trials=10,
            noise_eval_repeat=1,
        )

        self.assertIn("--skip_stage1_rl", rl_cmd)
        self.assertIn("true", rl_cmd)
        self.assertIn("--final_eval_config_source", rl_cmd)
        self.assertIn("json", rl_cmd)
        self.assertNotIn("--stage1_rl_episodes", rl_cmd)
        self.assertNotIn("--stage2_rl_episodes", rl_cmd)
        self.assertNotIn("--stage1_rl_episodes_specified", rl_cmd)
        self.assertNotIn("--stage2_rl_episodes_specified", rl_cmd)
        self.assertNotIn("--stage1_ga_generations", ga_cmd)
        self.assertNotIn("--stage2_ga_generations", ga_cmd)


if __name__ == "__main__":
    unittest.main()
