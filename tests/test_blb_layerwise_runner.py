from __future__ import annotations

import math
from pathlib import Path
import inspect
import tempfile
import types
import unittest

import numpy as np


class LayerwiseRunnerPureRulesTests(unittest.TestCase):
    @staticmethod
    def _candidate(*, cost, probabilities, loss=0.3, metric1=0.9, metric2=0.8):
        names = (
            "loss_precision_probability",
            "metric1_precision_probability",
            "metric2_precision_probability",
            "loss_stability_probability",
            "metric1_stability_probability",
            "metric2_stability_probability",
        )
        return {
            "variable_cost": cost,
            "assessment": dict(zip(names, probabilities)),
            "metrics": {
                "loss_mean": loss,
                "metric1_mean": metric1,
                "metric2_mean": metric2,
            },
        }

    def test_strict_rank_orders_cost_then_confidence_then_metrics(self):
        from blb_stage2_rl.layerwise_runner import strict_rank_key

        lower_cost_high_reward = self._candidate(
            cost=0.4, probabilities=[0.99] * 6, loss=0.1, metric1=0.99, metric2=0.99,
        )
        higher_cost = self._candidate(
            cost=0.5, probabilities=[0.81] * 6, loss=0.5, metric1=0.7, metric2=0.6,
        )
        self.assertLess(strict_rank_key(higher_cost), strict_rank_key(lower_cost_high_reward))

        low_confidence = self._candidate(cost=0.5, probabilities=[0.81] * 6)
        high_confidence = self._candidate(cost=0.5, probabilities=[0.91] * 6)
        self.assertLess(strict_rank_key(high_confidence), strict_rank_key(low_confidence))

        high_loss = self._candidate(cost=0.5, probabilities=[0.91] * 6, loss=0.4)
        low_loss = self._candidate(cost=0.5, probabilities=[0.91] * 6, loss=0.2)
        self.assertLess(strict_rank_key(low_loss), strict_rank_key(high_loss))

        weak_metrics = self._candidate(
            cost=0.5, probabilities=[0.91] * 6, loss=0.2, metric1=0.8, metric2=0.7,
        )
        strong_metrics = self._candidate(
            cost=0.5, probabilities=[0.91] * 6, loss=0.2, metric1=0.9, metric2=0.8,
        )
        self.assertLess(strict_rank_key(strong_metrics), strict_rank_key(weak_metrics))

    def test_normalized_entropy_excludes_masked_and_one_option_slots(self):
        from blb_stage2_rl.layerwise_runner import normalized_entropy_snapshot

        entropy = np.asarray([
            [0.05 * math.log(2), 999.0, 0.08 * math.log(6), 999.0, 0.12 * math.log(6), 0.16 * math.log(6)],
            [
                0.07 * math.log(2), 0.04 * math.log(6), 0.06 * math.log(6),
                0.10 * math.log(6), 999.0, 0.14 * math.log(6),
            ],
        ])
        masks = np.asarray([
            [True, False, True, True, True, True],
            [True, True, True, True, True, True],
        ])
        levels = np.asarray([
            [2, 6, 6, 1, 6, 6],
            [2, 6, 6, 6, 1, 6],
        ])

        snapshot = normalized_entropy_snapshot(entropy, masks, levels)

        self.assertAlmostEqual(snapshot["block4"], 0.06)
        self.assertAlmostEqual(snapshot["k"], 0.10)
        self.assertEqual(snapshot["block4_slot_count"], 2)
        self.assertEqual(snapshot["k_slot_count"], 7)

    def test_convergence_requires_feasible_stall_entropies_and_minimum_episodes(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        for _ in range(50):
            state = tracker.observe_update(
                completed_episodes=40_000,
                block4_entropy=0.05,
                k_entropy=0.05,
                robust_feasible_cost=None,
            )
        self.assertEqual(state.stall_update_windows, 0)
        self.assertFalse(state.converged)

        state = tracker.observe_update(
            completed_episodes=29_999,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
        )
        self.assertFalse(state.converged)
        self.assertEqual(state.stall_update_windows, 0)

        for _ in range(99):
            state = tracker.observe_update(
                completed_episodes=40_000,
                block4_entropy=0.05,
                k_entropy=0.05,
                robust_feasible_cost=0.4,
            )
        self.assertEqual(state.stall_update_windows, 99)
        self.assertFalse(state.converged)

        state = tracker.observe_update(
            completed_episodes=40_120,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
        )
        self.assertEqual(state.stall_update_windows, 100)
        self.assertTrue(state.converged)

        state = tracker.observe_update(
            completed_episodes=40_240,
            block4_entropy=0.1,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
        )
        self.assertFalse(state.converged)

        state = tracker.observe_update(
            completed_episodes=40_360,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.5,
        )
        self.assertEqual(state.stall_update_windows, 0)
        self.assertFalse(state.converged)


class LayerwiseDispatchRulesTests(unittest.TestCase):
    def test_dispatch_selects_layerwise_only_for_fusion_plus_layer(self):
        from blb_stage2_rl.layerwise_runner import resolve_decision_path

        self.assertEqual(
            resolve_decision_path(
                fusion_count_action=True,
                decision_granularity="layer",
                reward_design="robust_constrained",
            ),
            "layerwise",
        )
        self.assertEqual(
            resolve_decision_path(
                fusion_count_action=True,
                decision_granularity="block",
                reward_design="stage1_aligned",
            ),
            "block",
        )
        with self.assertRaisesRegex(ValueError, "robust_constrained.*layer"):
            resolve_decision_path(
                fusion_count_action=True,
                decision_granularity="block",
                reward_design="robust_constrained",
            )
        with self.assertRaisesRegex(ValueError, "fusion-count"):
            resolve_decision_path(
                fusion_count_action=False,
                decision_granularity="layer",
                reward_design="robust_constrained",
            )
        with self.assertRaisesRegex(ValueError, "layer.*robust_constrained"):
            resolve_decision_path(
                fusion_count_action=True,
                decision_granularity="layer",
                reward_design="stage1_aligned",
            )
        with self.assertRaisesRegex(ValueError, "layer.*block"):
            resolve_decision_path(
                fusion_count_action=True,
                decision_granularity="token",
                reward_design="stage1_aligned",
            )

    def test_initial_probabilities_use_decoded_k_values_and_disable_epsilon(self):
        from blb_stage2_rl.layerwise_action import K_LEVELS
        from blb_stage2_rl.layerwise_runner import initialize_layerwise_policy

        class FakePolicy:
            def set_initial_slot_probabilities(self, probabilities, values):
                self.probabilities = probabilities
                self.values = values

        policy = FakePolicy()
        initialize_layerwise_policy(policy)

        self.assertEqual(policy.probabilities[0], {0: 0.60, 1: 0.40})
        expected_k = {13: 0.50, 12: 0.20, 11: 0.12, 10: 0.08, 9: 0.06, 8: 0.04}
        self.assertEqual(policy.probabilities[1:], [expected_k] * 5)
        self.assertEqual(policy.values[0], (0, 1))
        self.assertEqual(policy.values[1:], [tuple(K_LEVELS)] * 5)

class _FakeBuffer:
    def __init__(self):
        self.transitions = []
        self.cleared = False

    def __len__(self):
        return len(self.transitions)

    def add(self, **transition):
        self.transitions.append(dict(transition))
        return len(self.transitions) - 1

    def clear(self):
        self.cleared = True


class _FakePolicy:
    def __init__(self):
        self.masks = []
        self.training = True

    def eval(self):
        self.training = False

    def sample_action(self, state, slot_mask, per_slot_num_levels, **_kwargs):
        del state, per_slot_num_levels
        mask = np.asarray(slot_mask, dtype=bool).reshape(1, 6)
        self.masks.append(mask[0].copy())
        action = np.asarray([[1, 5, 4, 3, 2, 1]], dtype=np.int64)
        return action, np.asarray([float(mask.sum())]), np.asarray([0.25])


class _FakeLayerwiseEnv:
    horizon = 12
    max_step_dim = 6
    state_dim = 4

    def __init__(self, probabilities=0.7):
        self._step = 0
        self.actions = []
        self.boosted_overrides = {(4, 3): {"v_mask_rescale_sf": 47}}
        self.base = types.SimpleNamespace(statistical_reference=object())
        self.runtime_terminal_info = None
        self._probabilities = float(probabilities)

    def reset(self, *, seed=None):
        self.seed = seed
        self._step = 0
        self.actions = []
        self.runtime_terminal_info = None
        return np.zeros(4, dtype=np.float32)

    def current_spec(self):
        return types.SimpleNamespace(
            step_idx=self._step,
            layer_idx=self._step,
            slot_dims=(2, 6, 6, 6, 6, 6),
            slot_mask=(True, self._step != 0, True, True, True, True),
        )

    def step(self, action):
        self.actions.append([int(value) for value in action])
        self._step += 1
        done = self._step == 12
        if not done:
            return np.full(4, self._step, dtype=np.float32), 123.0, False, {
                "layer_summary": {"all_valid": True},
            }
        probability_fields = {
            name: self._probabilities
            for name in (
                "loss_precision_probability",
                "metric1_precision_probability",
                "metric2_precision_probability",
                "loss_stability_probability",
                "metric1_stability_probability",
                "metric2_stability_probability",
            )
        }
        self.runtime_terminal_info = {
            "reward_breakdown": types.SimpleNamespace(priority=3),
            "statistical_trials": {
                "loss": [0.30, 0.31, 0.29, 0.30, 0.32],
                "metric1": [0.90, 0.89, 0.91, 0.90, 0.88],
                "metric2": [0.80, 0.79, 0.81, 0.80, 0.78],
                "seeds": [100, 101, 102, 103, 104],
            },
            "statistical_assessment": {**probability_fields, "bootstrap_seed": 77},
            "metrics": types.SimpleNamespace(
                loss_mean=0.304, metric1_mean=0.896, metric2_mean=0.796,
            ),
            "invalid": False,
        }
        return np.zeros(4, dtype=np.float32), 7.5, True, {
            "policy_actions": [row[:] for row in self.actions],
            "pending_full_vector": list(range(20)),
            "variable_cost": {"normalized": 0.4},
            "layer_summaries": [
                {"all_valid": True} for _ in range(12)
            ],
        }


def _assessment(probability):
    fields = {
        name: float(probability)
        for name in (
            "loss_precision_probability",
            "metric1_precision_probability",
            "metric2_precision_probability",
            "loss_stability_probability",
            "metric1_stability_probability",
            "metric2_stability_probability",
        )
    }
    return types.SimpleNamespace(
        **fields,
        precision_probability=float(probability),
        stability_probability=float(probability),
        gate_probability=0.8,
        online_precision_pass=probability >= 0.8,
        online_stability_pass=probability >= 0.8,
    )


class LayerwiseRolloutTests(unittest.TestCase):
    def test_train_collects_exactly_12_terminal_credit_transitions(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        env = _FakeLayerwiseEnv(probabilities=0.7)
        policy = _FakePolicy()
        buffer = _FakeBuffer()
        observed_ppo = []

        def fake_update(_policy, _optimizer, rollout, cfg, _device, **kwargs):
            observed_ppo.append((len(rollout), cfg.gamma, cfg.gae_lambda, kwargs))
            return {"entropy": 0.0, "n_samples": len(rollout)}

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=env,
                policy=policy,
                train_cfg=types.SimpleNamespace(
                    total_episodes=1,
                    update_every_n_episodes=1,
                    absolute_episode_start=0,
                    seed=42,
                    ent_coef_schedule="cosine",
                    ent_coef_cosine_start=0.05,
                    ent_coef_cosine_end=0.001,
                    ent_coef_cosine_plateau=0.25,
                    ent_coef_cosine_lower_bound=0.012,
                    ppo=types.SimpleNamespace(
                        lr=5e-5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                    ),
                ),
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=buffer,
                ppo_update_fn=fake_update,
                assess_candidate_fn=lambda trials, *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(len(buffer.transitions), 12)
        self.assertEqual([row["reward"] for row in buffer.transitions[:-1]], [0.0] * 11)
        self.assertEqual(buffer.transitions[-1]["reward"], 7.5)
        self.assertEqual([row["done"] for row in buffer.transitions], [False] * 11 + [True])
        self.assertEqual([float(row["log_prob"]) for row in buffer.transitions], [5.0] + [6.0] * 11)
        self.assertFalse(policy.masks[0][1])
        self.assertEqual(env.actions[0][1], 0)
        self.assertEqual(observed_ppo[0][0:3], (12, 1.0, 1.0))
        self.assertEqual(summary["episode_records"][0].action_matrix[0][1], 0)
        self.assertEqual(summary["episode_records"][0].pending_full_vector, tuple(range(20)))
        self.assertEqual(summary["episode_rewards"], [7.5])
        self.assertIn("best_action", summary)
        self.assertIsNone(summary["best_action"])

    def test_layerwise_loop_contains_no_retired_blockwise_scaffolds(self):
        from blb_stage2_rl.layerwise_runner import train_layerwise

        source = inspect.getsource(train_layerwise)
        for retired in (
            "force_baseline_episodes",
            "warmstart_neighbor_sampling",
            "fusion_probe_interval",
            "safe_neighbor",
            "invalid_mask",
            "terminal_metric_cache",
            "validation_metric_cache",
            "exploration_epsilon",
        ):
            self.assertNotIn(retired, source)


class _PromotionBase:
    def __init__(self, fresh_probability=0.9):
        self.statistical_reference = object()
        self.prepare_calls = []
        self.evaluate_calls = []
        self.fresh_probability = fresh_probability

    def prepare_action_for_terminal_probe(self, full_vec, **kwargs):
        self.prepare_calls.append((list(full_vec), dict(kwargs)))
        return {"prepared": True, "action": list(full_vec)}

    def evaluate_prepared_terminal_batch(self, prepared, **kwargs):
        self.evaluate_calls.append((prepared, dict(kwargs)))
        count = int(kwargs["num_trials_per_action"])
        fields = {
            name: self.fresh_probability
            for name in (
                "loss_precision_probability",
                "metric1_precision_probability",
                "metric2_precision_probability",
                "loss_stability_probability",
                "metric1_stability_probability",
                "metric2_stability_probability",
            )
        }
        return [(None, 0.0, True, {
            "statistical_trials": {
                "loss": [0.3 + i * 0.001 for i in range(count)],
                "metric1": [0.9 - i * 0.001 for i in range(count)],
                "metric2": [0.8 - i * 0.001 for i in range(count)],
                "seeds": list(range(1000, 1000 + count)),
            },
            "statistical_assessment": {**fields, "bootstrap_seed": 77},
            "metrics": types.SimpleNamespace(
                loss_mean=0.3, metric1_mean=0.9, metric2_mean=0.8,
            ),
        })]


class LayerwisePromotionTests(unittest.TestCase):
    def _store_with_five(self, root):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        store = CandidateStore(Path(root) / "candidates.jsonl")
        store.append_trial_group(
            list(range(20)),
            TrialSeries(
                loss=[0.3] * 5,
                metric1=[0.9] * 5,
                metric2=[0.8] * 5,
                seeds=[1, 2, 3, 4, 5],
            ),
            {"identity_context": {"action_space_version": "layerwise-v1"}},
        )
        return store

    def test_promotion_tops_up_five_to_25_through_real_chain_once(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            base = _PromotionBase()
            env = types.SimpleNamespace(base=base)
            kwargs = dict(
                env=env,
                candidate_store=store,
                action_indices=list(range(20)),
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.85),
                priority=3,
                variable_cost=0.6,
                frontier_cost=0.5,
                boosted_overrides={(4, 3): {"v_mask_rescale_sf": 47}},
                bootstrap_seed=77,
                assess_candidate_fn=lambda trials, *_args, **_kwargs: _assessment(
                    0.9 if len(trials.loss) == 25 else 0.85
                ),
            )
            promoted = promote_candidate_if_eligible(**kwargs)
            repeated = promote_candidate_if_eligible(**kwargs)

        self.assertEqual(promoted.status, "promoted")
        self.assertEqual(promoted.trial_count, 25)
        self.assertEqual(promoted.fresh_trial_count, 20)
        self.assertEqual(repeated.status, "already_promoted")
        self.assertEqual(len(base.prepare_calls), 1)
        self.assertEqual(base.prepare_calls[0][1]["external_cost_score"], 0.6)
        self.assertEqual(base.prepare_calls[0][1]["external_cost_rank"], 0.6)
        self.assertEqual(
            base.prepare_calls[0][1]["boosted_overrides"],
            {(4, 3): {"v_mask_rescale_sf": 47}},
        )
        self.assertEqual(base.evaluate_calls[0][1]["num_trials_per_action"], 20)
        self.assertTrue(base.evaluate_calls[0][1]["validation_required"])

    def test_promotion_rejects_priority_confidence_and_dominated_candidates(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        cases = (
            (1, 0.9, 0.6, 0.5, "priority_not_p3"),
            (2, 0.9, 0.6, 0.5, "priority_not_p3"),
            (3, 0.79, 0.6, 0.5, "promotion_probability_below_gate"),
            (3, 0.9, 0.5, 0.5, "not_frontier_improvement"),
            (3, 0.9, 0.4, 0.5, "not_frontier_improvement"),
        )
        for priority, probability, cost, frontier, status in cases:
            with self.subTest(status=status, priority=priority, probability=probability, cost=cost):
                with tempfile.TemporaryDirectory() as td:
                    store = self._store_with_five(td)
                    base = _PromotionBase()
                    result = promote_candidate_if_eligible(
                        env=types.SimpleNamespace(base=base),
                        candidate_store=store,
                        action_indices=list(range(20)),
                        identity_context={"action_space_version": "layerwise-v1"},
                        action_matrix=[[0] * 6 for _ in range(12)],
                        assessment=_assessment(probability),
                        priority=priority,
                        variable_cost=cost,
                        frontier_cost=frontier,
                        boosted_overrides={},
                        bootstrap_seed=77,
                        assess_candidate_fn=lambda *_args, **_kwargs: _assessment(probability),
                    )
                self.assertEqual(result.status, status)
                self.assertEqual(base.prepare_calls, [])
                self.assertEqual(base.evaluate_calls, [])

    def test_failed_required_promotion_is_not_marked_promoted(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=_PromotionBase()),
                candidate_store=store,
                action_indices=list(range(20)),
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.85),
                priority=3,
                variable_cost=0.6,
                frontier_cost=0.5,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.79),
            )

        self.assertEqual(result.status, "failed_probability_gate")
        self.assertFalse(result.evidence.promoted)
        self.assertTrue(result.evidence.promotion_attempted)

if __name__ == "__main__":
    unittest.main()
