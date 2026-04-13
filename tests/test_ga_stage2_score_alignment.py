import unittest


class GAStage2ScoreAlignmentTests(unittest.TestCase):
    def test_rl_aligned_score_prefers_better_metrics_over_small_cost_gain(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        baseline_stats = {
            "loss_mean": 0.300,
            "p_mean": 0.880,
            "s_mean": 0.870,
        }
        search_limits = {
            "loss": 0.350,
            "metric1": 0.860,
            "metric2": 0.850,
        }

        accurate_candidate = ga_module._compute_rl_aligned_noise_search_score(
            stats={
                "loss_mean": 0.305,
                "loss_std": 0.0020,
                "p_mean": 0.879,
                "p_std": 0.0020,
                "s_mean": 0.869,
                "s_std": 0.0020,
            },
            cost=38.0,
            baseline_reference_stats=baseline_stats,
            search_limits=search_limits,
            dynamic_loss_std_cap=0.0040,
            dynamic_m1_std_cap=0.0040,
            dynamic_m2_std_cap=0.0040,
            cost_lower_bound=30.0,
            cost_upper_bound=40.0,
            num_metrics=2,
        )
        cheaper_but_weaker_candidate = ga_module._compute_rl_aligned_noise_search_score(
            stats={
                "loss_mean": 0.333,
                "loss_std": 0.0035,
                "p_mean": 0.865,
                "p_std": 0.0035,
                "s_mean": 0.855,
                "s_std": 0.0035,
            },
            cost=34.0,
            baseline_reference_stats=baseline_stats,
            search_limits=search_limits,
            dynamic_loss_std_cap=0.0040,
            dynamic_m1_std_cap=0.0040,
            dynamic_m2_std_cap=0.0040,
            cost_lower_bound=30.0,
            cost_upper_bound=40.0,
            num_metrics=2,
        )

        self.assertGreater(
            cheaper_but_weaker_candidate["cost_score"],
            accurate_candidate["cost_score"],
        )
        self.assertGreater(
            accurate_candidate["score"],
            cheaper_but_weaker_candidate["score"],
        )

    def test_rl_aligned_score_penalizes_std_excess(self):
        try:
            import genetic_search_module as ga_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        baseline_stats = {
            "loss_mean": 0.300,
            "p_mean": 0.880,
            "s_mean": 0.870,
        }
        search_limits = {
            "loss": 0.350,
            "metric1": 0.860,
            "metric2": 0.850,
        }

        stable_candidate = ga_module._compute_rl_aligned_noise_search_score(
            stats={
                "loss_mean": 0.312,
                "loss_std": 0.0030,
                "p_mean": 0.874,
                "p_std": 0.0030,
                "s_mean": 0.864,
                "s_std": 0.0030,
            },
            cost=36.0,
            baseline_reference_stats=baseline_stats,
            search_limits=search_limits,
            dynamic_loss_std_cap=0.0040,
            dynamic_m1_std_cap=0.0040,
            dynamic_m2_std_cap=0.0040,
            cost_lower_bound=30.0,
            cost_upper_bound=40.0,
            num_metrics=2,
        )
        unstable_candidate = ga_module._compute_rl_aligned_noise_search_score(
            stats={
                "loss_mean": 0.312,
                "loss_std": 0.0065,
                "p_mean": 0.874,
                "p_std": 0.0060,
                "s_mean": 0.864,
                "s_std": 0.0055,
            },
            cost=36.0,
            baseline_reference_stats=baseline_stats,
            search_limits=search_limits,
            dynamic_loss_std_cap=0.0040,
            dynamic_m1_std_cap=0.0040,
            dynamic_m2_std_cap=0.0040,
            cost_lower_bound=30.0,
            cost_upper_bound=40.0,
            num_metrics=2,
        )

        self.assertLess(
            unstable_candidate["stability_reward_penalty"],
            stable_candidate["stability_reward_penalty"],
        )
        self.assertGreater(
            stable_candidate["score"],
            unstable_candidate["score"],
        )


if __name__ == "__main__":
    unittest.main()
