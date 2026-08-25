from __future__ import annotations

from dataclasses import fields
import unittest


class Stage2ProductionConfigTest(unittest.TestCase):
    def test_config_contains_only_the_layerwise_production_surface(self):
        from rfr.search.rl.stage2.training import BLBStage2TrainConfig

        names = {field.name for field in fields(BLBStage2TrainConfig)}
        removed = {
            "sequential_rl",
            "stage2_rl_devices",
            "stage2_workers_per_device",
            "substage_mode",
            "osr_results_path",
            "fusion_neighbor_curriculum_enabled",
            "protected_k1_enabled",
            "action_mask_enabled",
            "warmstart_neighbor_sampling",
            "guarded_radius2_enabled",
            "reward_design",
            "decision_granularity",
        }
        self.assertFalse(names & removed)
        self.assertTrue(
            {
                "total_episodes",
                "rollout_size",
                "seed",
                "ppo",
                "reward_devices",
                "search_backend",
                "stage2_stability_multiplier",
                "communication_importance_ratio",
            }.issubset(names)
        )

    def test_runner_is_exposed_only_from_training_module(self):
        from rfr.search.rl.stage2.training import BLBStage2RLRunner

        self.assertEqual(BLBStage2RLRunner.__module__, "rfr.search.rl.stage2.training")


if __name__ == "__main__":
    unittest.main()
