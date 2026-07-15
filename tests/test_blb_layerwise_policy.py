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

    def test_initial_probabilities_disable_existing_epsilon_exploration(self):
        policy = self._policy()
        fusion_probs = {0: 0.60, 1: 0.40}
        k_probs = {13: 0.50, 12: 0.20, 11: 0.12, 10: 0.08, 9: 0.06, 8: 0.04}
        k_values = (13, 8, 12, 9, 11, 10)
        action_values = [(0, 1)] + [k_values] * 5

        policy.set_slot_exploration_epsilon([0.20] * 6)
        self.assertTrue(policy._slot_exploration_enabled)
        self.assertTrue(torch.all(policy._slot_exploration_epsilon > 0.0))

        policy.set_initial_slot_probabilities(
            [fusion_probs] + [k_probs] * 5,
            action_values,
        )
        policy.eval()

        self.assertFalse(policy._slot_exploration_enabled)
        self.assertTrue(torch.all(policy._slot_exploration_epsilon == 0.0))

        state = torch.zeros(1, policy.cfg.state_dim)
        state[0, 4] = 1.0
        logits, _ = policy(state)
        levels = torch.tensor([[2, 6, 6, 6, 6, 6]])
        full_mask = torch.ones((1, 6), dtype=torch.bool)
        logit_mask = policy._build_logit_mask(full_mask, levels, 6)
        masked_logits = logits + logit_mask
        dist = policy._action_dist(masked_logits, masked_logits)
        expected = torch.stack((
            torch.tensor([0.60, 0.40, 0.0, 0.0, 0.0, 0.0]),
            *(
                torch.tensor([k_probs[value] for value in k_values])
                for _ in range(5)
            ),
        )).unsqueeze(0)
        torch.testing.assert_close(dist.probs, expected, rtol=1e-6, atol=1e-7)

        sampled_actions, sampled_log_prob, _ = policy.sample_action(
            state,
            full_mask,
            levels,
            deterministic=True,
        )
        replay_log_prob, _, _ = policy.evaluate_action(
            state,
            sampled_actions,
            full_mask,
            levels,
        )
        self.assertEqual(sampled_actions.tolist(), [[0, 0, 0, 0, 0, 0]])
        expected_selected_log_prob = np.log(0.60) + 5.0 * np.log(0.50)
        self.assertAlmostEqual(float(sampled_log_prob), expected_selected_log_prob, places=6)
        torch.testing.assert_close(sampled_log_prob, replay_log_prob)

        for slot_idx, support in enumerate(action_values):
            isolated_mask = torch.zeros((1, 6), dtype=torch.bool)
            isolated_mask[0, slot_idx] = True
            for action_idx, decoded_value in enumerate(support):
                actions = torch.zeros((1, 6), dtype=torch.long)
                actions[0, slot_idx] = action_idx
                log_prob, _, _ = policy.evaluate_action(
                    state,
                    actions,
                    isolated_mask,
                    levels,
                )
                desired = fusion_probs if slot_idx == 0 else k_probs
                self.assertAlmostEqual(
                    float(torch.exp(log_prob)),
                    desired[decoded_value],
                    places=6,
                )

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

    def test_evaluate_action_exposes_factorized_log_probabilities(self):
        policy = self._policy()
        state = torch.zeros(3, policy.cfg.state_dim)
        state[:, 4] = 1.0
        actions = torch.tensor([
            [0, 0, 1, 2, 3, 4],
            [1, 5, 4, 3, 2, 1],
            [0, 2, 2, 2, 2, 2],
        ])
        levels = torch.tensor([[2, 6, 6, 6, 6, 6]]).expand(3, -1)
        masks = torch.ones((3, 6), dtype=torch.bool)
        masks[0, 1] = False

        joint_log_prob, entropy, value, entropy_per_slot, log_prob_per_slot = (
            policy.evaluate_action(
                state,
                actions,
                masks,
                levels,
                return_per_slot_entropy=True,
                return_per_slot_log_prob=True,
            )
        )

        self.assertEqual(log_prob_per_slot.shape, (3, 6))
        torch.testing.assert_close(log_prob_per_slot.sum(dim=-1), joint_log_prob)
        torch.testing.assert_close(entropy_per_slot.sum(dim=-1), entropy)
        self.assertEqual(value.shape, (3,))
        self.assertEqual(float(log_prob_per_slot[0, 1].detach()), 0.0)

    def test_factorized_clipping_does_not_cross_clip_sibling_slots(self):
        from blb_stage2_rl.sequential_policy import _factorized_clipped_policy_loss

        old_log_prob = torch.zeros((1, 2))
        new_log_prob = torch.log(torch.tensor([[1.50, 0.95]]))
        advantages = torch.ones(1)
        active = torch.ones((1, 2), dtype=torch.bool)

        loss, ratio = _factorized_clipped_policy_loss(
            new_log_prob,
            old_log_prob,
            advantages,
            active,
            clip_range=0.2,
        )

        torch.testing.assert_close(ratio, torch.tensor([[1.50, 0.95]]))
        self.assertAlmostEqual(float(loss), -(1.20 + 0.95) / 2.0, places=6)

        active[0, 1] = False
        masked_loss, _ = _factorized_clipped_policy_loss(
            new_log_prob,
            old_log_prob,
            advantages,
            active,
            clip_range=0.2,
        )
        self.assertAlmostEqual(float(masked_loss), -1.20, places=6)

    def test_factorized_kl_uses_active_slot_mean_not_joint_sum(self):
        from blb_stage2_rl.sequential_policy import _factorized_approx_kl

        old_log_prob = torch.zeros((1, 2))
        new_log_prob = torch.tensor([[-0.10, -0.30]])
        active = torch.ones((1, 2), dtype=torch.bool)

        mean_kl = _factorized_approx_kl(new_log_prob, old_log_prob, active)
        self.assertAlmostEqual(float(mean_kl), 0.20, places=6)

        active[0, 1] = False
        masked_kl = _factorized_approx_kl(new_log_prob, old_log_prob, active)
        self.assertAlmostEqual(float(masked_kl), 0.10, places=6)

    def test_normalized_entropy_objective_equalizes_binary_and_six_way_slots(self):
        from blb_stage2_rl.sequential_policy import _active_slot_entropy_objective

        entropy = torch.log(torch.tensor([[2.0, 6.0]]))
        levels = torch.tensor([[2, 6]])
        active = torch.ones((1, 2), dtype=torch.bool)

        normalized = _active_slot_entropy_objective(
            entropy,
            levels,
            active,
            normalize_by_levels=True,
        )
        raw = _active_slot_entropy_objective(
            entropy,
            levels,
            active,
            normalize_by_levels=False,
        )

        self.assertAlmostEqual(float(normalized), 1.0, places=6)
        self.assertAlmostEqual(float(raw), float(entropy.mean()), places=6)

    def test_factorized_actor_cost_removes_same_step_sibling_noise(self):
        from blb_stage2_rl.sequential_policy import SequentialRolloutBuffer

        buffer = SequentialRolloutBuffer()
        transition_index = buffer.add(
            state=np.zeros(1, dtype=np.float32),
            action=np.zeros(2, dtype=np.int64),
            slot_mask=np.ones(2, dtype=bool),
            per_slot_num_levels=np.full(2, 2, dtype=np.int64),
            log_prob=0.0,
            value=0.0,
            reward=0.3,
            done=True,
        )
        buffer.set_actor_cost_at(transition_index, np.asarray([0.1, 0.2]))

        factorized = buffer.factorized_actor_advantages(
            torch.tensor([0.7]),
            torch.device("cpu"),
        )

        torch.testing.assert_close(factorized, torch.tensor([[0.5, 0.6]]))

    def test_explicit_shared_constraint_return_replaces_scalar_critic_noise(self):
        from blb_stage2_rl.sequential_policy import SequentialRolloutBuffer

        buffer = SequentialRolloutBuffer()
        transition_index = buffer.add(
            state=np.zeros(1, dtype=np.float32),
            action=np.zeros(2, dtype=np.int64),
            slot_mask=np.ones(2, dtype=bool),
            per_slot_num_levels=np.full(2, 2, dtype=np.int64),
            log_prob=0.0,
            value=0.0,
            reward=99.0,
            done=True,
        )
        buffer.set_actor_cost_at(transition_index, np.asarray([0.1, 0.2]))
        buffer.set_actor_shared_return_at(transition_index, -0.25)

        factorized = buffer.factorized_actor_advantages(
            torch.tensor([50.0]),
            torch.device("cpu"),
        )

        torch.testing.assert_close(factorized, torch.tensor([[-0.15, -0.05]]))

    def test_factorized_policy_loss_accepts_one_advantage_per_slot(self):
        from blb_stage2_rl.sequential_policy import _factorized_clipped_policy_loss

        loss, _ = _factorized_clipped_policy_loss(
            torch.log(torch.tensor([[1.50, 0.95]])),
            torch.zeros((1, 2)),
            torch.tensor([[1.0, -1.0]]),
            torch.ones((1, 2), dtype=torch.bool),
            clip_range=0.2,
        )

        self.assertAlmostEqual(float(loss), -0.125, places=6)

    def test_stochastic_sampling_zeros_masked_slot_without_changing_active_semantics(self):
        policy = self._policy()
        fusion_probs = {0: 0.60, 1: 0.40}
        k_probs = {13: 0.50, 12: 0.20, 11: 0.12, 10: 0.08, 9: 0.06, 8: 0.04}
        k_values = (13, 8, 12, 9, 11, 10)
        policy.set_initial_slot_probabilities(
            [fusion_probs] + [k_probs] * 5,
            [(0, 1)] + [k_values] * 5,
        )
        policy.eval()

        batch_size = 64
        states = torch.zeros(batch_size, policy.cfg.state_dim)
        states[:, 4] = 1.0
        levels = torch.tensor([[2, 6, 6, 6, 6, 6]]).expand(batch_size, -1)
        slot_mask = torch.ones((batch_size, 6), dtype=torch.bool)
        slot_mask[:, 1] = False
        generator = torch.Generator().manual_seed(20260712)

        actions, sampled_log_prob, _ = policy.sample_action(
            states,
            slot_mask,
            levels,
            generator=generator,
        )
        replay_log_prob, replay_entropy, _ = policy.evaluate_action(
            states,
            actions,
            slot_mask,
            levels,
        )

        self.assertTrue(torch.all(actions[:, 1] == 0))
        self.assertTrue(torch.all((actions[:, 0] >= 0) & (actions[:, 0] < 2)))
        self.assertTrue(torch.all((actions[:, 2:] >= 0) & (actions[:, 2:] < 6)))
        self.assertGreater(torch.unique(actions[:, 0]).numel(), 1)
        self.assertGreater(torch.unique(actions[:, 2:]).numel(), 1)

        fusion_by_index = torch.tensor([fusion_probs[0], fusion_probs[1]])
        k_by_index = torch.tensor([k_probs[value] for value in k_values])
        expected_log_prob = torch.log(fusion_by_index[actions[:, 0]])
        expected_log_prob += torch.log(k_by_index[actions[:, 2:]]).sum(dim=-1)
        torch.testing.assert_close(sampled_log_prob, expected_log_prob)
        torch.testing.assert_close(replay_log_prob, expected_log_prob)

        fusion_entropy = -sum(prob * np.log(prob) for prob in fusion_probs.values())
        k_entropy = -sum(prob * np.log(prob) for prob in k_probs.values())
        torch.testing.assert_close(
            replay_entropy,
            torch.full((batch_size,), fusion_entropy + 4.0 * k_entropy),
        )

    def test_sampling_can_return_behavior_log_probability_per_slot(self):
        policy = self._policy()
        policy.eval()
        states = torch.zeros(2, policy.cfg.state_dim)
        slot_mask = torch.tensor([
            [True, False, True, True, True, True],
            [True, True, True, True, True, True],
        ])
        levels = torch.tensor([[2, 6, 6, 6, 6, 6]]).expand(2, -1)

        actions, joint_log_prob, _, per_slot_log_prob = policy.sample_action(
            states,
            slot_mask,
            levels,
            generator=torch.Generator().manual_seed(20260715),
            return_per_slot_log_prob=True,
        )

        self.assertEqual(tuple(per_slot_log_prob.shape), (2, 6))
        self.assertTrue(torch.all(per_slot_log_prob[~slot_mask] == 0.0))
        torch.testing.assert_close(per_slot_log_prob.sum(dim=-1), joint_log_prob)
        replay = policy.evaluate_action(
            states,
            actions,
            slot_mask,
            levels,
            return_per_slot_log_prob=True,
        )
        torch.testing.assert_close(per_slot_log_prob, replay[3])

    def test_factorized_ppo_uses_stored_behavior_log_probability_after_policy_changes(self):
        from blb_stage2_rl.sequential_policy import (
            SequentialPPOConfig,
            SequentialRolloutBuffer,
            sequential_ppo_update,
        )

        class DriftedPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self._ppo_lr_scale = 1.0
                self._ppo_last_avg_kl = 0.0

            def evaluate_action(
                    self,
                    state,
                    actions,
                    slot_mask,
                    per_slot_num_levels,
                    action_level_mask=None,
                    baseline_prior_scale=None,
                    return_per_slot_entropy=False,
                    return_per_slot_log_prob=False,
                    ):
                del actions, per_slot_num_levels, action_level_mask, baseline_prior_scale
                batch = state.shape[0]
                per_slot = (self.weight - 1.0).expand_as(slot_mask.float())
                per_slot = per_slot * slot_mask.float()
                entropy_per_slot = torch.full_like(per_slot, 0.5) * slot_mask.float()
                result = (
                    per_slot.sum(dim=-1),
                    entropy_per_slot.sum(dim=-1),
                    self.weight.expand(batch),
                )
                if return_per_slot_entropy and return_per_slot_log_prob:
                    return result + (entropy_per_slot, per_slot)
                if return_per_slot_entropy:
                    return result + (entropy_per_slot,)
                if return_per_slot_log_prob:
                    return result + (per_slot,)
                return result

        buffer = SequentialRolloutBuffer()
        behavior_per_slot = np.asarray([-0.2, -0.3], dtype=np.float32)
        buffer.add(
            state=np.zeros(1, dtype=np.float32),
            action=np.zeros(2, dtype=np.int64),
            slot_mask=np.ones(2, dtype=bool),
            per_slot_num_levels=np.full(2, 2, dtype=np.int64),
            log_prob=float(behavior_per_slot.sum()),
            log_prob_per_slot=behavior_per_slot,
            value=0.0,
            reward=1.0,
            done=True,
        )
        policy = DriftedPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)

        metrics = sequential_ppo_update(
            policy,
            optimizer,
            buffer,
            SequentialPPOConfig(
                lr=0.0,
                n_epochs=1,
                minibatch_size=1,
                ent_coef=0.0,
                normalize_returns=False,
                use_kl_early_stop=False,
                factorized_actor_clip=True,
            ),
            torch.device("cpu"),
        )

        self.assertEqual(metrics["actor_clip_mode"], "factorized_per_slot")
        self.assertAlmostEqual(metrics["approx_kl"], 0.75, places=6)

    def test_factorized_ppo_rejects_missing_behavior_log_probability_per_slot(self):
        from blb_stage2_rl.sequential_policy import (
            SequentialPPOConfig,
            SequentialRolloutBuffer,
            sequential_ppo_update,
        )

        policy = self._policy()
        buffer = SequentialRolloutBuffer()
        buffer.add(
            state=np.zeros(policy.cfg.state_dim, dtype=np.float32),
            action=np.zeros(6, dtype=np.int64),
            slot_mask=np.ones(6, dtype=bool),
            per_slot_num_levels=np.full(6, 2, dtype=np.int64),
            log_prob=0.0,
            value=0.0,
            reward=0.0,
            done=True,
        )

        with self.assertRaisesRegex(
                RuntimeError,
                "sampling-time per-slot behavior log probabilities",
        ):
            sequential_ppo_update(
                policy,
                torch.optim.Adam(policy.parameters(), lr=1.0e-4),
                buffer,
                SequentialPPOConfig(
                    n_epochs=1,
                    minibatch_size=1,
                    factorized_actor_clip=True,
                ),
                torch.device("cpu"),
            )

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

    def test_factorized_ppo_converges_fusion_and_k_on_contextual_cost_bandit(self):
        from blb_stage2_rl.layerwise_action import K_LEVELS
        from blb_stage2_rl.sequential_policy import (
            SequentialPPOConfig,
            SequentialRolloutBuffer,
            sequential_ppo_update,
        )

        class ContextualCostPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                initial = torch.empty((12, 6, 6), dtype=torch.float32)
                initial[:, 0, :] = -20.0
                initial[:, 0, :2] = torch.log(torch.tensor([0.90, 0.10]))
                k_probs = torch.tensor([0.50, 0.20, 0.12, 0.08, 0.06, 0.04])
                initial[:, 1:, :] = torch.log(k_probs)
                self.logits = torch.nn.Parameter(initial)
                self.values = torch.nn.Parameter(torch.zeros(12))

            def _distribution(self, state, slot_mask, per_slot_num_levels):
                layer = state[:, 0].long()
                logits = self.logits.index_select(0, layer)
                level_index = torch.arange(6).view(1, 1, 6)
                valid = level_index < per_slot_num_levels.unsqueeze(-1)
                valid = valid & slot_mask.unsqueeze(-1)
                safe_logits = logits.masked_fill(~valid, -20.0)
                safe_logits = torch.where(
                    slot_mask.unsqueeze(-1), safe_logits, torch.zeros_like(safe_logits),
                )
                return torch.distributions.Categorical(logits=safe_logits), layer

            def sample_action(
                    self,
                    state,
                    slot_mask,
                    per_slot_num_levels,
                    *,
                    return_per_slot_log_prob=False,
                    ):
                dist, layer = self._distribution(state, slot_mask, per_slot_num_levels)
                action = dist.sample()
                action = torch.where(slot_mask, action, torch.zeros_like(action))
                per_slot = dist.log_prob(action) * slot_mask.float()
                result = (
                    action,
                    per_slot.sum(dim=-1),
                    self.values.index_select(0, layer),
                )
                if return_per_slot_log_prob:
                    return result + (per_slot,)
                return result

            def evaluate_action(
                    self,
                    state,
                    actions,
                    slot_mask,
                    per_slot_num_levels,
                    action_level_mask=None,
                    baseline_prior_scale=None,
                    return_per_slot_entropy=False,
                    return_per_slot_log_prob=False,
                    ):
                del action_level_mask, baseline_prior_scale
                dist, layer = self._distribution(state, slot_mask, per_slot_num_levels)
                per_slot = dist.log_prob(actions.long()) * slot_mask.float()
                entropy_per_slot = dist.entropy() * slot_mask.float()
                result = (
                    per_slot.sum(dim=-1),
                    entropy_per_slot.sum(dim=-1),
                    self.values.index_select(0, layer),
                )
                if return_per_slot_entropy and return_per_slot_log_prob:
                    return result + (entropy_per_slot, per_slot)
                if return_per_slot_entropy:
                    return result + (entropy_per_slot,)
                if return_per_slot_log_prob:
                    return result + (per_slot,)
                return result

        torch.manual_seed(20260714)
        policy = ContextualCostPolicy()
        policy.eval()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.05)
        config = SequentialPPOConfig(
            lr=0.05,
            clip_range=0.2,
            n_epochs=4,
            minibatch_size=256,
            ent_coef=0.0,
            value_coef=0.1,
            gamma=1.0,
            gae_lambda=1.0,
            normalize_returns=False,
            use_kl_early_stop=False,
            adaptive_lr_kl=False,
            factorized_actor_clip=True,
            entropy_average_active_slots=True,
            entropy_normalize_active_slots=True,
        )
        slot_levels = np.asarray([2, 6, 6, 6, 6, 6], dtype=np.int64)
        update_count = 180
        last_metrics = None

        for update_idx in range(update_count):
            buffer = SequentialRolloutBuffer()
            for sample_idx in range(120):
                layer_idx = sample_idx % 12
                slot_mask = np.ones(6, dtype=bool)
                if layer_idx == 0:
                    slot_mask[1] = False
                state = np.asarray([layer_idx], dtype=np.float32)
                with torch.no_grad():
                    action, log_prob, value, log_prob_per_slot = policy.sample_action(
                        torch.from_numpy(state).unsqueeze(0),
                        torch.from_numpy(slot_mask).unsqueeze(0),
                        torch.from_numpy(slot_levels).unsqueeze(0),
                        return_per_slot_log_prob=True,
                    )
                action_np = action[0].numpy()
                k_cost_units = np.asarray(
                    [0.5 * (13.0 - float(K_LEVELS[int(index)])) for index in action_np[1:]],
                    dtype=np.float32,
                )
                raw_cost_units = float(action_np[0]) + float(
                    k_cost_units[slot_mask[1:]].sum()
                )
                transition_index = buffer.add(
                    state=state,
                    action=action_np,
                    slot_mask=slot_mask,
                    per_slot_num_levels=slot_levels,
                    log_prob=log_prob[0],
                    log_prob_per_slot=log_prob_per_slot[0],
                    value=value[0],
                    reward=raw_cost_units / 159.5,
                    done=True,
                )
                per_slot_cost = np.concatenate((
                    np.asarray([float(action_np[0])]),
                    k_cost_units,
                )) / 159.5
                per_slot_cost[~slot_mask] = 0.0
                buffer.set_actor_cost_at(transition_index, per_slot_cost)
                buffer.set_actor_shared_return_at(transition_index, 0.0)
            last_metrics = sequential_ppo_update(
                policy,
                optimizer,
                buffer,
                config,
                torch.device("cpu"),
                ent_coef_override=0.0,
            )

        self.assertEqual(last_metrics["actor_clip_mode"], "factorized_per_slot")
        self.assertEqual(
            last_metrics["actor_credit_mode"],
            "shared_constraint_plus_own_cost",
        )
        self.assertEqual(
            last_metrics["entropy_objective_mode"],
            "normalized_active_slot_mean",
        )

        with torch.no_grad():
            fusion_dist = torch.distributions.Categorical(logits=policy.logits[:, 0, :2])
            k_dist = torch.distributions.Categorical(logits=policy.logits[:, 1:, :])
            fusion_entropy = float((fusion_dist.entropy() / np.log(2.0)).mean())
            active_k_entropy = torch.cat((
                k_dist.entropy()[0, 1:].reshape(-1),
                k_dist.entropy()[1:].reshape(-1),
            ))
            k_entropy = float((active_k_entropy / np.log(6.0)).mean())
            fusion_modes = torch.argmax(policy.logits[:, 0, :2], dim=-1)
            k_modes = torch.argmax(policy.logits[:, 1:, :], dim=-1)
            active_k_modes = torch.cat((
                k_modes[0, 1:].reshape(-1),
                k_modes[1:].reshape(-1),
            ))
            diagnostic = (
                f"fusion_entropy={fusion_entropy:.6f}, k_entropy={k_entropy:.6f}, "
                f"fusion_modes={fusion_modes.tolist()}, k_modes={k_modes.tolist()}"
            )
            self.assertTrue(torch.all(fusion_modes == 1), diagnostic)
            selected_k_values = torch.as_tensor(K_LEVELS)[active_k_modes]
            self.assertTrue(torch.all(selected_k_values == 8), diagnostic)
        self.assertLess(fusion_entropy, 0.1, diagnostic)
        self.assertLess(k_entropy, 0.1, diagnostic)


if __name__ == "__main__":
    unittest.main()
