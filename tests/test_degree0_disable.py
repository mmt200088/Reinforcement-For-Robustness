import unittest

import numpy as np


class DegreeZeroDisableTests(unittest.TestCase):
    def test_ga_stage1_allowed_degrees_excludes_zero(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        searcher = object.__new__(ga_module.Stage1GeneticSearcher)
        self.assertEqual(
            ga_module.Stage1GeneticSearcher._allowed_gelu_degrees(searcher, 0),
            (4, 2, 1),
        )

        gelu_arr, softmax_arr, had_degree0 = (
            ga_module.Stage1GeneticSearcher._sanitize_stage1_candidate(
                [4, 0, 2, 1],
                [6, 6, 6, 6],
            )
        )
        self.assertTrue(had_degree0)
        self.assertTrue(np.array_equal(gelu_arr, np.array([4, 1, 2, 1])))
        self.assertTrue(np.array_equal(softmax_arr, np.array([6, 6, 6, 6])))

    def test_greedy_stage1_never_steps_to_degree0(self):
        try:
            import greedy_search_module as greedy_module
        except ImportError as exc:
            self.skipTest(f"greedy_search_module import unavailable: {exc}")

        searcher = object.__new__(greedy_module.Stage1GreedySearcher)
        self.assertEqual(searcher._next_gelu_degree(4), 2)
        self.assertEqual(searcher._next_gelu_degree(2), 1)
        self.assertIsNone(searcher._next_gelu_degree(1))

        gelu_arr, _, had_degree0 = greedy_module.Stage1GreedySearcher._sanitize_stage1_candidate(
            [4, 0, 2, 1],
            [6, 6, 6, 6],
        )
        self.assertTrue(had_degree0)
        self.assertTrue(np.array_equal(gelu_arr, np.array([4, 1, 2, 1])))

    def test_rl_stage1_env_masks_degree0_even_when_legacy_flag_is_true(self):
        try:
            import layer_importance_evaluator as li_module
        except ImportError as exc:
            self.skipTest(f"layer_importance_evaluator import unavailable: {exc}")

        env = object.__new__(li_module.TransformerOptEnv)
        env.total_layers = 3
        env.current_layer = 0
        env.gelu_degree0_eligible = np.ones(env.total_layers, dtype=bool)

        self.assertEqual(li_module.GELU_MAP[3], 1)
        self.assertTrue(
            np.array_equal(
                env.get_gelu_action_mask(0),
                np.array([True, True, True, False], dtype=bool),
            )
        )

    def test_general_stage1_prepare_task_keeps_degree0_disabled(self):
        try:
            import general_policy_module as gp_module
        except ImportError as exc:
            self.skipTest(f"general_policy_module import unavailable: {exc}")

        class DummyEvaluator:
            total_layers = 4
            error_threshold = 0.005
            correlation_drop_ratio = 0.005

            def get_simulated_cost(self, gelu, softmax):
                del gelu, softmax
                return 36.0, 18.0, 18.0

            def stage1_evaluate(self, gelu, softmax, use_train=True):
                del gelu, softmax, use_train
                return 0.2, 0.8, 0.7, 0.0

            def get_num_metrics(self):
                return 2

        task = gp_module.prepare_stage1_task(DummyEvaluator())
        self.assertTrue(
            np.array_equal(
                task["gelu0_eligible"],
                np.zeros(DummyEvaluator.total_layers, dtype=bool),
            )
        )

    def test_general_stage1_candidate_validator_rejects_degree0(self):
        try:
            import general_policy_module as gp_module
        except ImportError as exc:
            self.skipTest(f"general_policy_module import unavailable: {exc}")

        with self.assertRaises(ValueError):
            gp_module._validate_general_stage1_candidate_configs(
                [
                    {
                        "gelu": [4, 0, 2, 1],
                        "softmax": [6, 6, 6, 6],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
