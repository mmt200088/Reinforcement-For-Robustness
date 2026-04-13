import unittest


class GASkipBudgetCompatTests(unittest.TestCase):
    def test_skip_stage1_resets_legacy_eval_budget_guard(self):
        try:
            import rl_tune_genetic as ga_tune
        except ImportError as exc:
            self.skipTest(f"rl_tune_genetic import unavailable: {exc}")

        normalized = ga_tune.normalize_ga_skip_budget_flags(
            skip_stage1_rl=True,
            skip_noise_rl=False,
            stage1_ga_generations=1594,
            stage2_ga_generations=2500,
            stage1_budget_specified=False,
            stage2_budget_specified=True,
            stage1_rl_episodes=1594 * 32,
            stage2_rl_episodes=2500 * 32,
        )

        self.assertEqual(normalized["stage1_rl_episodes"], 51000)
        self.assertFalse(normalized["stage1_budget_specified"])
        self.assertEqual(normalized["stage1_budget_source"], "skipped")
        self.assertEqual(normalized["stage2_rl_episodes"], 2500 * 32)

    def test_skip_stage2_resets_legacy_eval_budget_guard(self):
        try:
            import rl_tune_genetic as ga_tune
        except ImportError as exc:
            self.skipTest(f"rl_tune_genetic import unavailable: {exc}")

        normalized = ga_tune.normalize_ga_skip_budget_flags(
            skip_stage1_rl=False,
            skip_noise_rl=True,
            stage1_ga_generations=100,
            stage2_ga_generations=1250,
            stage1_budget_specified=True,
            stage2_budget_specified=False,
            stage1_rl_episodes=100 * 32,
            stage2_rl_episodes=1250 * 32,
        )

        self.assertEqual(normalized["stage2_rl_episodes"], 40000)
        self.assertFalse(normalized["stage2_budget_specified"])
        self.assertEqual(normalized["stage2_budget_source"], "skipped")
        self.assertEqual(normalized["stage1_rl_episodes"], 100 * 32)

    def test_skip_stage1_rejects_explicit_generation_budget(self):
        try:
            import rl_tune_genetic as ga_tune
        except ImportError as exc:
            self.skipTest(f"rl_tune_genetic import unavailable: {exc}")

        with self.assertRaisesRegex(ValueError, "skip_stage1_rl=True"):
            ga_tune.normalize_ga_skip_budget_flags(
                skip_stage1_rl=True,
                skip_noise_rl=False,
                stage1_ga_generations=10,
                stage2_ga_generations=20,
                stage1_budget_specified=True,
                stage2_budget_specified=False,
                stage1_rl_episodes=320,
                stage2_rl_episodes=640,
            )

    def test_skip_stage2_rejects_explicit_generation_budget(self):
        try:
            import rl_tune_genetic as ga_tune
        except ImportError as exc:
            self.skipTest(f"rl_tune_genetic import unavailable: {exc}")

        with self.assertRaisesRegex(ValueError, "skip_noise_rl=True"):
            ga_tune.normalize_ga_skip_budget_flags(
                skip_stage1_rl=False,
                skip_noise_rl=True,
                stage1_ga_generations=10,
                stage2_ga_generations=20,
                stage1_budget_specified=False,
                stage2_budget_specified=True,
                stage1_rl_episodes=320,
                stage2_rl_episodes=640,
            )


if __name__ == "__main__":
    unittest.main()
