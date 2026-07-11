from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - exercised on dependency-light clients
    torch = None

from blb_stage2_rl.layerwise_action import LayerwiseStepSpec


@unittest.skipIf(torch is None, "torch is required for layerwise policy tests")
class LayerwisePolicyTest(unittest.TestCase):
    def _config(self, **overrides):
        from blb_stage2_rl.sequential_policy import SequentialPolicyConfig

        values = {
            "state_dim": 4 + 12 + 12 * 6 + 12 * 4,
            "max_step_dim": 6,
            "max_num_levels": 6,
            "horizon": 12,
            "num_layers": 12,
            "block_count": 5,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 64,
            "dropout": 0.0,
            "metadata_width": 0,
            "signal_width": 4,
        }
        values.update(overrides)
        return SequentialPolicyConfig(**values)

    def _policy(self, **overrides):
        from blb_stage2_rl.sequential_policy import BLBStage2SequentialPolicy

        return BLBStage2SequentialPolicy(self._config(**overrides))

    def test_explicit_schedule_indices_are_used_verbatim(self):
        layer_indices = tuple(range(12))
        block_indices = (4, 2, 0, 3, 1, 4, 2, 0, 3, 1, 4, 2)
        policy = self._policy(
            step_layer_indices=layer_indices,
            step_block_indices=block_indices,
        )

        steps, layers, blocks = policy._step_layer_block_indices()

        self.assertEqual(steps.tolist(), list(range(12)))
        self.assertEqual(layers.tolist(), list(layer_indices))
        self.assertEqual(blocks.tolist(), list(block_indices))
        self.assertEqual(len(set(layers.tolist())), 12)
        self.assertIsInstance(policy.cfg.step_layer_indices, tuple)
        self.assertIsInstance(policy.cfg.step_block_indices, tuple)

    def test_malformed_explicit_schedule_indices_fail(self):
        valid_layers = tuple(range(12))
        valid_blocks = (0,) * 12
        cases = (
            {"step_layer_indices": valid_layers},
            {"step_block_indices": valid_blocks},
            {"step_layer_indices": valid_layers[:-1], "step_block_indices": valid_blocks},
            {"step_layer_indices": valid_layers, "step_block_indices": valid_blocks[:-1]},
            {"step_layer_indices": valid_layers[:-1] + (12,), "step_block_indices": valid_blocks},
            {"step_layer_indices": valid_layers, "step_block_indices": valid_blocks[:-1] + (5,)},
        )
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self._config(**values)

    def test_observation_layout_validation_fails_early(self):
        with self.assertRaises(ValueError):
            self._config(metadata_width=-1)
        with self.assertRaises(ValueError):
            self._config(signal_width=0)

    def test_layerwise_parser_reads_exact_action_and_signal_rows(self):
        policy = self._policy()
        static = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        current = np.zeros(12, dtype=np.float32)
        current[7] = 1.0
        action_indices = np.asarray([
            [row % 2] + [(row + col) % 6 for col in range(1, 6)]
            for row in range(12)
        ], dtype=np.float32)
        action_history = action_indices / np.asarray([1, 5, 5, 5, 5, 5], dtype=np.float32)
        signal_history = np.asarray(
            [[100.0 + row, 200.0 + row, 300.0 + row, 400.0 + row] for row in range(12)],
            dtype=np.float32,
        )
        state = torch.from_numpy(np.concatenate((
            static,
            current,
            action_history.reshape(-1),
            signal_history.reshape(-1),
        ))).unsqueeze(0)

        parsed_static, current_step, actions, signals = policy._parse_state(state)

        torch.testing.assert_close(parsed_static[0], torch.from_numpy(static))
        self.assertEqual(current_step.tolist(), [7])
        torch.testing.assert_close(actions[0], torch.from_numpy(action_indices).long())
        torch.testing.assert_close(signals[0], torch.from_numpy(signal_history))
        self.assertEqual(policy.fc_continuous[0].in_features, 9)

        _, truncated_step, truncated_actions, truncated_signals = policy._parse_state(
            state,
            truncate_to_current=True,
        )
        self.assertEqual(truncated_step.tolist(), [7])
        torch.testing.assert_close(truncated_actions[0], torch.from_numpy(action_indices[:8]).long())
        torch.testing.assert_close(truncated_signals[0], torch.from_numpy(signal_history[:8]))

    def test_legacy_parser_layout_defaults_remain_unchanged(self):
        policy = self._policy(
            state_dim=4 + 12 + 6 + 12 * 6 + 12 * 3,
            metadata_width=6,
            signal_width=3,
        )
        state = np.zeros(policy.cfg.state_dim, dtype=np.float32)
        state[4 + 3] = 1.0
        action_start = 4 + 12 + 6
        state[action_start:action_start + 12 * 6] = 2.0 / 8.0
        signal_start = action_start + 12 * 6
        state[signal_start:signal_start + 12 * 3] = 7.0

        _, current_step, actions, signals = policy._parse_state(torch.from_numpy(state).unsqueeze(0))

        self.assertEqual(current_step.tolist(), [3])
        self.assertTrue(torch.all(actions == 2))
        self.assertTrue(torch.all(signals == 7.0))
        self.assertEqual(policy.fc_continuous[0].in_features, 8)

    def test_layerwise_step_mask_preserves_inactive_block1_slot(self):
        from blb_stage2_rl.sequential_policy import step_to_mask_and_levels

        layer0 = LayerwiseStepSpec(
            step_idx=0,
            layer_idx=0,
            slot_dims=(2, 6, 6, 6, 6, 6),
            slot_mask=(True, False, True, True, True, True),
            terminal=False,
            num_layers=12,
            graph_keys_by_block=(),
        )
        layer1 = LayerwiseStepSpec(
            step_idx=1,
            layer_idx=1,
            slot_dims=(2, 6, 6, 6, 6, 6),
            slot_mask=(True, True, True, True, True, True),
            terminal=False,
            num_layers=12,
            graph_keys_by_block=(),
        )

        mask0, levels0 = step_to_mask_and_levels(layer0, 6, 6)
        mask1, levels1 = step_to_mask_and_levels(layer1, 6, 6)

        self.assertEqual(mask0.tolist(), [True, False, True, True, True, True])
        self.assertEqual(mask1.tolist(), [True, True, True, True, True, True])
        self.assertEqual(levels0.tolist(), [2, 6, 6, 6, 6, 6])
        self.assertEqual(levels1.tolist(), [2, 6, 6, 6, 6, 6])

    def test_initial_probabilities_map_by_decoded_value_in_reordered_support(self):
        policy = self._policy(
            step_layer_indices=tuple(range(12)),
            step_block_indices=(0,) * 12,
        )
        k_values = (13, 8, 12, 9, 11, 10)
        fusion_probs = {0: 0.60, 1: 0.40}
        k_probs = {13: 0.50, 12: 0.20, 11: 0.12, 10: 0.08, 9: 0.06, 8: 0.04}

        policy.apply_preferred_per_step_bias([1] * 6, gain=50.0)
        policy.set_initial_slot_probabilities(
            [fusion_probs] + [k_probs] * 5,
            [(0, 1)] + [k_values] * 5,
        )
        policy.eval()
        state = torch.zeros(1, policy.cfg.state_dim)
        state[0, 4] = 1.0
        logits, _ = policy(state)

        fusion_actual = torch.softmax(logits[0, 0, :2], dim=-1)
        self.assertTrue(torch.all(fusion_actual > 0.0))
        torch.testing.assert_close(
            fusion_actual,
            torch.tensor([0.60, 0.40]),
            rtol=1e-6,
            atol=1e-7,
        )
        expected_k = torch.tensor([k_probs[value] for value in k_values])
        for slot_idx in range(1, 6):
            self.assertTrue(torch.all(torch.softmax(logits[0, slot_idx, :6], dim=-1) > 0.0))
            torch.testing.assert_close(
                torch.softmax(logits[0, slot_idx, :6], dim=-1),
                expected_k,
                rtol=1e-6,
                atol=1e-7,
            )
        self.assertTrue(torch.all(policy._preferred_per_slot_idx < 0))
        self.assertEqual(float(policy._preferred_prior_template.abs().sum()), 0.0)
        self.assertFalse(policy._slot_exploration_enabled)

    def test_initial_probability_validation_rejects_bad_support_or_mass(self):
        policy = self._policy()
        valid_fusion = {0: 0.6, 1: 0.4}
        valid_k = {13: 0.5, 12: 0.2, 11: 0.12, 10: 0.08, 9: 0.06, 8: 0.04}
        supports = [(0, 1)] + [(8, 9, 10, 11, 12, 13)] * 5
        cases = (
            ([None] + [valid_k] * 5, supports),
            ([{0: 1.0}] + [valid_k] * 5, supports),
            ([{0: 0.6, 1: 0.4}] + [dict(valid_k, **{})] * 4, supports),
            ([valid_fusion] + [{**valid_k, 8: 0.0}] * 5, supports),
            ([valid_fusion] + [{**valid_k, 8: float("nan")}] * 5, supports),
            ([valid_fusion] + [{**valid_k, 8: 0.05}] * 5, supports),
            ([valid_fusion] + [valid_k] * 5, [(0, 1)] + [(8, 9, 10, 11, 12, 14)] * 5),
        )
        for probabilities, action_values in cases:
            with self.subTest(probabilities=probabilities, action_values=action_values):
                with self.assertRaises(ValueError):
                    policy.set_initial_slot_probabilities(probabilities, action_values)

    def test_masked_block1_slot_contributes_no_log_probability_or_entropy(self):
        policy = self._policy()
        fusion_probs = {0: 0.6, 1: 0.4}
        k_probs = {13: 0.5, 12: 0.2, 11: 0.12, 10: 0.08, 9: 0.06, 8: 0.04}
        k_values = (8, 9, 10, 11, 12, 13)
        policy.set_initial_slot_probabilities(
            [fusion_probs] + [k_probs] * 5,
            [(0, 1)] + [k_values] * 5,
        )
        policy.eval()
        state = torch.zeros(1, policy.cfg.state_dim)
        state[0, 4] = 1.0
        actions = torch.zeros((1, 6), dtype=torch.long)
        levels = torch.tensor([[2, 6, 6, 6, 6, 6]])
        full_mask = torch.ones((1, 6), dtype=torch.bool)
        layer0_mask = full_mask.clone()
        layer0_mask[0, 1] = False

        full_log_prob, full_entropy, _ = policy.evaluate_action(
            state, actions, full_mask, levels,
        )
        masked_log_prob, masked_entropy, _ = policy.evaluate_action(
            state, actions, layer0_mask, levels,
        )
        sampled_actions, sampled_log_prob, _ = policy.sample_action(
            state,
            layer0_mask,
            levels,
            deterministic=True,
        )
        replay_log_prob, _, _ = policy.evaluate_action(
            state,
            sampled_actions,
            layer0_mask,
            levels,
        )
        changed_masked_action = sampled_actions.clone()
        changed_masked_action[0, 1] = 5
        changed_log_prob, changed_entropy, _ = policy.evaluate_action(
            state,
            changed_masked_action,
            layer0_mask,
            levels,
        )

        block1_prob = k_probs[k_values[0]]
        block1_entropy = -sum(prob * np.log(prob) for prob in k_probs.values())
        self.assertAlmostEqual(
            float(full_log_prob - masked_log_prob),
            float(np.log(block1_prob)),
            places=6,
        )
        self.assertAlmostEqual(
            float(full_entropy - masked_entropy),
            block1_entropy,
            places=6,
        )
        torch.testing.assert_close(sampled_log_prob, replay_log_prob)
        torch.testing.assert_close(changed_log_prob, replay_log_prob)
        torch.testing.assert_close(changed_entropy, masked_entropy)

    def test_terminal_reward_has_undiscounted_credit_at_every_layer(self):
        from blb_stage2_rl.sequential_policy import SequentialRolloutBuffer

        buffer = SequentialRolloutBuffer()
        for step_idx in range(12):
            buffer.add(
                state=np.zeros(4, dtype=np.float32),
                action=np.zeros(6, dtype=np.int64),
                slot_mask=np.ones(6, dtype=bool),
                per_slot_num_levels=np.full(6, 2, dtype=np.int64),
                log_prob=0.0,
                value=0.0,
                reward=2.0 if step_idx == 11 else 0.0,
                done=step_idx == 11,
            )

        returns_np, advantages_np = buffer.compute_gae(gamma=1.0, lam=1.0)
        tensors = buffer.to_tensors(
            torch.device("cpu"),
            gamma=1.0,
            lam=1.0,
            advantage_normalize=False,
        )
        returns_t, advantages_t = tensors[7], tensors[8]

        np.testing.assert_allclose(returns_np, np.full(12, 2.0, dtype=np.float32))
        np.testing.assert_allclose(advantages_np, np.full(12, 2.0, dtype=np.float32))
        torch.testing.assert_close(returns_t, torch.full((12,), 2.0))
        torch.testing.assert_close(advantages_t, torch.full((12,), 2.0))


if __name__ == "__main__":
    unittest.main()
