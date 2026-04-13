import unittest
from unittest.mock import patch


class Stage2FixedConfigAliasTests(unittest.TestCase):
    def test_ga_stage2_fixed_config_accepts_stage1_result_alias(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        captured = {}

        class DummyModule:
            def __init__(self, **kwargs):
                captured["config_source"] = kwargs["config_source"]

            def _resolve_selected_config(self, search_best_config, total_layers):
                del search_best_config
                return [1] * total_layers, [2] * total_layers, "Selected", captured["config_source"]

        class DummyEvaluator:
            final_eval_random_seed = 42
            final_eval_permutation_trials = 10
            final_eval_cost_equivalent_trials = 10
            final_eval_budget_equivalent_trials = 10
            stage1_final_eval_dir = "unused"
            total_layers = 4

        with patch.object(ga_module, "GeneticFinalEvaluationModule", DummyModule):
            gelu, softmax, label, source = ga_module.resolve_stage1_selected_config(
                evaluator=DummyEvaluator(),
                search_best_config={"gelu": [1, 1, 1, 1], "softmax": [2, 2, 2, 2]},
                config_source="stage1_result",
            )

        self.assertEqual(captured["config_source"], "search")
        self.assertEqual(source, "stage1_result")
        self.assertEqual(label, "Selected")
        self.assertEqual(gelu, [1, 1, 1, 1])
        self.assertEqual(softmax, [2, 2, 2, 2])

    def test_ga_stage2_fixed_config_keeps_old_search_alias_compatible(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        captured = {}

        class DummyModule:
            def __init__(self, **kwargs):
                captured["config_source"] = kwargs["config_source"]

            def _resolve_selected_config(self, search_best_config, total_layers):
                del search_best_config
                return [1] * total_layers, [2] * total_layers, "Selected", captured["config_source"]

        class DummyEvaluator:
            final_eval_random_seed = 42
            final_eval_permutation_trials = 10
            final_eval_cost_equivalent_trials = 10
            final_eval_budget_equivalent_trials = 10
            stage1_final_eval_dir = "unused"
            total_layers = 4

        with patch.object(ga_module, "GeneticFinalEvaluationModule", DummyModule):
            _, _, _, source = ga_module.resolve_stage1_selected_config(
                evaluator=DummyEvaluator(),
                search_best_config={"gelu": [1, 1, 1, 1], "softmax": [2, 2, 2, 2]},
                config_source="search",
            )

        self.assertEqual(captured["config_source"], "search")
        self.assertEqual(source, "stage1_result")


if __name__ == "__main__":
    unittest.main()
