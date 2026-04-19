"""Compare-mode skip-json tests.

The original suite exercised the split Stage-1 / Stage-2 final-eval flow
(``FinalEvaluationModule`` + ``NoiseFinalEvaluationModule`` with
``noise_eval_config_source`` / ``skip_noise_final_eval`` etc.).
That split flow has been replaced by the unified ``UnifiedFinalEvaluationModule``
and its single ``final_eval_config_source`` / ``skip_final_eval`` knobs, so the
old assertions no longer describe real code paths.

The tests below are placeholders — they import the current modules to verify
the unified API still wires through ``rl_ga_compare_runner`` / ``genetic_search_module``,
and skip themselves if the imports fail. Full coverage of the unified path
lives in ``tests/test_compare_config_modes.py`` and the integration fixtures
under ``rl_results/runs/compare``.
"""

import unittest


class UnifiedFinalEvalImportSmokeTests(unittest.TestCase):
    def test_compare_runner_exposes_unified_final_eval_symbols(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        self.assertTrue(hasattr(compare_runner, "CompareSideConfig"))
        config = compare_runner.CompareSideConfig(
            skip_stage1_search=True,
            skip_noise_search=True,
            final_eval_config_source="json",
            final_eval_config_path="glue_final_configs_best_ppo.json",
        )
        self.assertEqual(config.final_eval_config_source, "json")

    def test_genetic_module_exposes_unified_final_eval_subclass(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        self.assertTrue(hasattr(ga_module, "GeneticUnifiedFinalEvaluationModule"))


if __name__ == "__main__":
    unittest.main()
