import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class BLBActionFinalEvalRegressionTests(unittest.TestCase):
    def test_resolve_base_action_accepts_numpy_arrays_without_truthiness(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        action = np.arange(12, dtype=int)

        resolved = runner._resolve_base_action(
            {
                "blb_v3_best_action_vec": action,
                "best_action_vec": np.ones(12, dtype=int),
                "best_action": [2] * 12,
            }
        )

        self.assertTrue(np.array_equal(resolved, action))


class BLBPolicyWarmstartRegressionTests(unittest.TestCase):
    def test_preferred_action_bias_drives_deterministic_sample_to_baseline(self):
        from blb_stage2_rl.action_space import layer_dims, make_all_max_action_vector
        from blb_stage2_rl.policy import BLBStage2Policy

        preferred = make_all_max_action_vector(num_layers=2)
        policy = BLBStage2Policy(
            state_dim=7,
            num_layers=2,
            per_layer_dims=layer_dims(),
            first_input_levels=5,
            d_hidden=16,
            d_layer_emb=4,
        )

        policy.apply_preferred_action_bias(preferred, gain=50.0)
        state = torch.zeros(1, 7)
        action, _log_prob, _value = policy.sample_action(state, deterministic=True)

        self.assertEqual(action.squeeze(0).tolist(), preferred.tolist())


class BLBTraceWriterRegressionTests(unittest.TestCase):
    def test_trace_writer_appends_structured_rollout_rows(self):
        from blb_stage2_rl.persistence import append_blb_episode_trace_row

        with tempfile.TemporaryDirectory() as td:
            trace_path = append_blb_episode_trace_row(
                td,
                {
                    "episode": 120,
                    "total_episodes": 240,
                    "ppo_update_count": 1,
                    "rollout_reward_mean": -273.0,
                    "rollout_reward_max": -273.0,
                    "best_reward": -273.0,
                    "priority1_count": 120,
                    "priority2_count": 0,
                    "priority3_count": 0,
                    "invalid_count": 0,
                    "entropy": 0.5,
                },
            )
            append_blb_episode_trace_row(
                td,
                {
                    "episode": 240,
                    "total_episodes": 240,
                    "ppo_update_count": 2,
                    "rollout_reward_mean": -10.0,
                    "rollout_reward_max": 0.0,
                    "best_reward": 0.0,
                    "priority1_count": 10,
                    "priority2_count": 0,
                    "priority3_count": 110,
                    "invalid_count": 1,
                    "entropy": 0.4,
                },
            )

            with Path(trace_path).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["episode"], "120")
        self.assertEqual(rows[0]["priority1_count"], "120")
        self.assertEqual(rows[1]["best_reward"], "0.0")


class BLBPersistencePathRegressionTests(unittest.TestCase):
    def test_blb_progress_stays_under_stage2_noise_progress(self):
        from blb_stage2_rl.runner import resolve_blb_persistence_dir

        class DummyEvaluator:
            pass

        with tempfile.TemporaryDirectory() as td:
            ev = DummyEvaluator()
            ev.run_output_dir = td
            path = Path(resolve_blb_persistence_dir(ev))

        self.assertEqual(path.name, "progress")
        self.assertEqual(path.parent.name, "stage2_noise")


if __name__ == "__main__":
    unittest.main()
