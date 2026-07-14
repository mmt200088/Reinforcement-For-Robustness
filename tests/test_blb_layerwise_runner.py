from __future__ import annotations

import ast
import hashlib
import inspect
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


class LayerwiseSeedTests(unittest.TestCase):
    def test_adjacent_episode_trial_domains_are_disjoint_and_bounded(self):
        from blb_stage2_rl.seed_utils import (
            derive_layerwise_episode_probe_seed,
            derive_probe_trial_seed,
        )

        first_base = derive_layerwise_episode_probe_seed(42, 100, trial_count=25)
        second_base = derive_layerwise_episode_probe_seed(42, 101, trial_count=25)
        first = {derive_probe_trial_seed(first_base, i) for i in range(25)}
        second = {derive_probe_trial_seed(second_base, i) for i in range(25)}

        self.assertEqual(len(first), 25)
        self.assertEqual(len(second), 25)
        self.assertTrue(first.isdisjoint(second))
        with self.assertRaises(ValueError):
            derive_layerwise_episode_probe_seed(42, -1, trial_count=5)
        with self.assertRaises(ValueError):
            derive_layerwise_episode_probe_seed(42, 0, trial_count=10_000)


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

    def test_convergence_cannot_trigger_while_current_frontier_is_empty(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        tracker.observe_update(
            completed_episodes=30_000,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
        )
        for update_idx in range(150):
            state = tracker.observe_update(
                completed_episodes=30_120 + update_idx * 120,
                block4_entropy=0.05,
                k_entropy=0.05,
                robust_feasible_cost=None,
            )

        self.assertEqual(state.stall_update_windows, 0)
        self.assertEqual(state.best_robust_feasible_cost, 0.4)
        self.assertFalse(state.converged)

    def test_convergence_tracker_state_round_trips_across_resume(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        tracker.observe_update(
            completed_episodes=30_000,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
        )
        for update_idx in range(7):
            tracker.observe_update(
                completed_episodes=30_120 + update_idx * 120,
                block4_entropy=0.05,
                k_entropy=0.05,
                robust_feasible_cost=0.4,
            )

        restored = LayerwiseConvergenceTracker()
        restored.load_state_dict(tracker.state_dict())
        state = restored.observe_update(
            completed_episodes=31_000,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
        )

        self.assertEqual(state.stall_update_windows, 8)
        self.assertEqual(state.best_robust_feasible_cost, 0.4)

    def test_convergence_rebases_when_feasible_frontier_retracts(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        tracker.observe_update(
            completed_episodes=30_000,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.8,
        )
        tracker.observe_update(
            completed_episodes=30_120,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=None,
        )
        state = tracker.observe_update(
            completed_episodes=30_240,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.6,
        )

        self.assertEqual(state.stall_update_windows, 0)
        self.assertEqual(state.best_robust_feasible_cost, 0.6)
        self.assertFalse(state.converged)

    def test_entropy_schedule_uses_absolute_progress_across_resume(self):
        from blb_stage2_rl.layerwise_runner import _cosine_entropy_coefficient

        uninterrupted = types.SimpleNamespace(
            total_episodes=60_000,
            planned_total_episodes=60_000,
            ent_coef_cosine_start=0.05,
            ent_coef_cosine_end=0.001,
            ent_coef_cosine_plateau=0.25,
            ent_coef_cosine_lower_bound=0.0,
        )
        resumed = types.SimpleNamespace(
            total_episodes=30_000,
            planned_total_episodes=60_000,
            ent_coef_cosine_start=0.05,
            ent_coef_cosine_end=0.001,
            ent_coef_cosine_plateau=0.25,
            ent_coef_cosine_lower_bound=0.0,
        )

        self.assertEqual(
            _cosine_entropy_coefficient(resumed, 30_000),
            _cosine_entropy_coefficient(uninterrupted, 30_000),
        )


class LayerwiseDispatchRulesTests(unittest.TestCase):
    def test_sequential_train_config_carries_layerwise_resume_state(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        config_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SequentialTrainConfig"
        )
        fields = {
            node.target.id
            for node in config_class.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

        self.assertIn("planned_total_episodes", fields)
        self.assertIn("convergence_resume_state", fields)

    def test_layerwise_episode_diagnostics_map_existing_probe_timing_fields(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        for required in (
            "probe_diagnostics = _to_plain_mapping(record.probe_diagnostics)",
            "terminal_probe_wall_seconds=float(probe_diagnostics.get(",
            "terminal_probe_devices=[str(value) for value in (probe_diagnostics.get(",
            "terminal_probe_trial_counts=[",
            "terminal_probe_trial_indices=[",
            "terminal_probe_speedup=float(probe_diagnostics.get(",
            "terminal_cost_eval_wall_seconds=float(probe_diagnostics.get(",
            "terminal_probe_install_wall_seconds=float(probe_diagnostics.get(",
            "terminal_probe_clear_wall_seconds=float(probe_diagnostics.get(",
            "terminal_probe_install_skipped=bool(probe_diagnostics.get(",
            "terminal_probe_clear_skipped=bool(probe_diagnostics.get(",
        ):
            self.assertIn(required, branch_source)

    def test_layerwise_checkpoint_preserves_long_run_search_state(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        for required in (
            'checkpoint.get("convergence_state")',
            'convergence_state=metrics.get("convergence_state")',
            "planned_total_episodes=int(planned_total_episodes)",
            "ppo_update_counter = start_episode //",
            '"boosted_overrides":',
            'best_reward = summary.get("best_reward")',
            '"torch_rng_state": torch.get_rng_state()',
            "torch.set_rng_state(",
            '"numpy_rng_state": np.random.get_state()',
            "np.random.set_state(",
            '"python_rng_state": random.getstate()',
            "random.setstate(",
            "existing_diagnostic_episodes",
            "existing_diagnostic_updates",
            "weights_only=False",
            'checkpoint.get("candidate_store_size")',
            '"candidate_store_size": (',
            "candidate_store.path.stat().st_size",
            "candidate_store.recover_to_checkpoint_size",
            'checkpoint.get("diagnostics_jsonl_sizes")',
            '"diagnostics_jsonl_sizes": diag_recorder.committed_jsonl_sizes()',
            "diag_recorder.recover_to_checkpoint_sizes",
            "planned_total_episodes = int(checkpoint.get(",
            '"planned_total_episodes", planned_total_episodes',
        ):
            self.assertIn(required, branch_source)

    def test_layerwise_checkpoint_and_result_publish_reloadable_fusion_best(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn("build_fusion_fixed_config", branch_source)
        self.assertIn('"best_action": checkpoint_best_action', branch_source)
        self.assertIn('"blb_v3_best_action_vec": checkpoint_best_action', branch_source)
        self.assertIn('"blb_v3_best_action_group": checkpoint_best_group', branch_source)
        self.assertIn('"blb_v3_best_action_group": best_action_group', branch_source)
        self.assertNotIn('"blb_v3_best_action_group": None', branch_source)

        evaluator_source = Path("layer_importance_evaluator.py").read_text(
            encoding="utf-8",
        )
        self.assertIn('checkpoint.get("strict_best")', evaluator_source)
        self.assertIn("_build_stage2_final_eval_handoff", evaluator_source)
        self.assertIn('out["blb_v3_best_action_group"]', evaluator_source)

    def test_outer_layerwise_branch_does_not_resurrect_stale_checkpoint_best(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn(
            'strict_best = dict(metrics.get("strict_best") or {})',
            branch_source,
        )
        self.assertNotIn(
            'summary.get("best_full_vector") is None and strict_best.get("full_vector")',
            branch_source,
        )

    def test_layerwise_branch_rewrites_checkpoint_after_zero_episode_revalidation(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn("def save_layerwise_checkpoint(", branch_source)
        self.assertGreaterEqual(branch_source.count("save_layerwise_checkpoint("), 3)
        self.assertIn('strict_best=summary.get("strict_best")', branch_source)

    def test_layerwise_branch_persists_authoritative_run_mode_and_strict_summary(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn('"stage2_layerwise_robust_run_v1"', branch_source)
        self.assertIn('"layerwise_run_manifest.json"', branch_source)
        self.assertIn("write_strict_json_file(", branch_source)
        self.assertNotIn("write_json_file(\n        os.path.join(blb_progress_dir, \"layerwise_summary.json\")", branch_source)

    def test_layerwise_branch_finalizes_live_status_and_run_manifest(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn("status.update_after_ppo_update(", branch_source)
        self.assertIn("status.set_best(", branch_source)
        self.assertIn("run_manifest.update({", branch_source)
        self.assertIn(
            '"status": ("completed" if training_completed else "failed")',
            branch_source,
        )
        self.assertIn('"completed_episodes": int(', branch_source)
        self.assertIn('"ppo_update_count": int(ppo_update_counter)', branch_source)
        self.assertGreaterEqual(
            branch_source.count("write_strict_json_file(layerwise_manifest_path"),
            2,
        )

    def test_layerwise_branch_persists_complete_ppo_update_metrics(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn(
            'kl_early_stop=bool(metrics.get("kl_early_stop", False))',
            branch_source,
        )
        for field in (
            "lr", "lr_scale", "entropy_recovery_delta", "return_mean", "return_std",
        ):
            self.assertIn(
                f'metrics.get("{field}",',
                branch_source,
            )
        for status_field in ("ent_coef", "lr", "lr_scale", "kl_early_stop"):
            self.assertIn(
                f'"{status_field}": update_stats.{status_field}',
                branch_source,
            )

    def test_layerwise_online_trial_count_uses_authoritative_stage2_k_trials(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn(
            "online_num_trials_per_step=int(train_cfg.num_trials_per_step)",
            branch_source,
        )
        self.assertIn(
            '"stage2_k_trials": int(train_cfg.num_trials_per_step)',
            branch_source,
        )
        self.assertNotIn(
            'getattr(train_cfg, "online_num_trials_per_step", 5)',
            branch_source,
        )

    def test_layerwise_curves_rebuild_from_full_diagnostic_history(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        branch_source = ast.get_source_segment(source, branch)

        self.assertIn("for row in iter_jsonl(existing_episode_path", branch_source)
        self.assertIn("for row in iter_jsonl(existing_update_path", branch_source)
        self.assertIn("errors=\"raise\"", branch_source)
        self.assertNotIn("list(iter_jsonl(", branch_source)
        self.assertNotIn("if episode_records:\n        fusion_counts", branch_source)

    def test_public_config_builder_drives_real_dispatch_rules(self):
        from blb_stage2_rl.layerwise_runner import (
            apply_public_stage2_decision_config,
            resolve_decision_path,
        )

        layer_cfg = types.SimpleNamespace(
            decision_granularity="block",
            reward_design="stage1_aligned",
        )
        apply_public_stage2_decision_config(
            types.SimpleNamespace(
                blb_v3_decision_granularity="layer",
                blb_v3_reward_design="robust_constrained",
            ),
            layer_cfg,
        )
        self.assertEqual(layer_cfg.decision_granularity, "layer")
        self.assertEqual(layer_cfg.reward_design, "robust_constrained")
        self.assertEqual(
            resolve_decision_path(
                fusion_count_action=True,
                decision_granularity=layer_cfg.decision_granularity,
                reward_design=layer_cfg.reward_design,
            ),
            "layerwise",
        )

        block_cfg = types.SimpleNamespace(
            decision_granularity="block",
            reward_design="stage1_aligned",
        )
        apply_public_stage2_decision_config(
            types.SimpleNamespace(
                blb_v3_decision_granularity="block",
                blb_v3_reward_design="continuous",
            ),
            block_cfg,
        )
        self.assertEqual(
            resolve_decision_path(
                fusion_count_action=True,
                decision_granularity=block_cfg.decision_granularity,
                reward_design=block_cfg.reward_design,
            ),
            "block",
        )

        for field, value in (
            ("blb_v3_decision_granularity", "token"),
            ("blb_v3_reward_design", "legacy_unknown"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    apply_public_stage2_decision_config(
                        types.SimpleNamespace(**{field: value}),
                        types.SimpleNamespace(
                            decision_granularity="block",
                            reward_design="stage1_aligned",
                        ),
                    )

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

    def __init__(self, probabilities=0.7, evidence_mode="valid", invalid=False):
        self._step = 0
        self.actions = []
        self.boosted_overrides = {(4, 3): {"v_mask_rescale_sf": 47}}
        self.base = types.SimpleNamespace(
            statistical_reference=object(),
            probe_noise_seed=None,
        )
        self.runtime_terminal_info = None
        self._probabilities = float(probabilities)
        self._evidence_mode = str(evidence_mode)
        self._invalid = bool(invalid)

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
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

        trial_count = 4 if self._evidence_mode == "wrong_count" else 5
        statistical_trials = {
            "loss": [0.30 + 0.001 * i for i in range(trial_count)],
            "metric1": [0.90 - 0.001 * i for i in range(trial_count)],
            "metric2": [0.80 - 0.001 * i for i in range(trial_count)],
            "seeds": [
                derive_probe_trial_seed(self.base.probe_noise_seed, i)
                for i in range(trial_count)
            ],
        }
        if self._evidence_mode == "malformed":
            statistical_trials["loss"] = ["not-a-number"] * trial_count
        self.runtime_terminal_info = {
            "reward_breakdown": types.SimpleNamespace(priority=3),
            "statistical_assessment": {**probability_fields, "bootstrap_seed": 77},
            "metrics": types.SimpleNamespace(
                loss_mean=0.304, metric1_mean=0.896, metric2_mean=0.796,
            ),
            "probe_diagnostics": {
                "wall_seconds": 1.25,
                "devices": ["cuda:0", "cuda:1"],
                "per_worker_seconds": [1.10, 1.20],
                "per_worker_trial_counts": [3, 2],
                "per_worker_trial_indices": [[0, 2, 4], [1, 3]],
                "speedup_vs_sequential": 1.75,
                "cost_eval_wall_seconds": 0.11,
                "probe_install_wall_seconds": 0.22,
                "probe_clear_wall_seconds": 0.33,
                "probe_install_skipped": True,
                "probe_clear_skipped": False,
            },
            "invalid": self._invalid,
        }
        if self._evidence_mode != "missing":
            self.runtime_terminal_info["statistical_trials"] = statistical_trials
        return np.zeros(4, dtype=np.float32), (-5.0 if self._invalid else 7.5), True, {
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
    @staticmethod
    def _train_cfg(total_episodes=1, update_every=1):
        return types.SimpleNamespace(
            total_episodes=total_episodes,
            update_every_n_episodes=update_every,
            absolute_episode_start=0,
            seed=42,
            online_num_trials_per_step=5,
            ent_coef_schedule="cosine",
            ent_coef_cosine_start=0.05,
            ent_coef_cosine_end=0.001,
            ent_coef_cosine_plateau=0.25,
            ent_coef_cosine_lower_bound=0.012,
            ppo=types.SimpleNamespace(
                lr=5e-5, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            ),
        )

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
                train_cfg=self._train_cfg(),
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
        self.assertEqual(
            summary["episode_records"][0].probe_diagnostics,
            env.runtime_terminal_info["probe_diagnostics"],
        )
        self.assertEqual(summary["episode_rewards"], [7.5])
        self.assertIn("best_action", summary)
        self.assertIsNone(summary["best_action"])

    def test_adjacent_same_action_episodes_pool_ten_distinct_real_probe_seeds(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        identity_context = {"action_space_version": "layerwise-v1"}
        env = _FakeLayerwiseEnv(probabilities=0.7)
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            train_layerwise(
                env=env,
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=2, update_every=2),
                candidate_store=store,
                identity_context=identity_context,
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {"entropy": 0.0},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )
            evidence = store.trial_evidence_for_action(
                list(range(20)), identity_context,
            )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.trial_count, 10)
        self.assertEqual(len(set(evidence.trials.seeds)), 10)

    def test_episode_record_labels_fresh_reward_and_pooled_ranking_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=2, update_every=2),
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {"entropy": 0.0},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        second = summary["episode_records"][1]
        self.assertEqual(len(second.raw_trials.loss), 5)
        self.assertEqual(len(second.pooled_trials.loss), 10)
        self.assertEqual(second.fresh_trial_count, 5)
        self.assertEqual(second.pooled_trial_count, 10)
        self.assertEqual(second.reward_evidence, "fresh_trials")
        self.assertEqual(second.ranking_evidence, "pooled_prefix_trials")

    def test_repeated_action_keeps_bootstrap_assessment_at_fixed_25_trial_budget(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        assessed_trial_counts = []

        def assess(trials, *_args, **_kwargs):
            assessed_trial_counts.append(len(trials.loss))
            return _assessment(0.7)

        identity_context = {"action_space_version": "layerwise-v1"}
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=10, update_every=10),
                candidate_store=store,
                identity_context=identity_context,
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {"entropy": 0.0},
                assess_candidate_fn=assess,
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )
            raw = store.trial_evidence_for_action(list(range(20)), identity_context)

        self.assertEqual(raw.trial_count, 50)
        self.assertLessEqual(max(assessed_trial_counts), 25)
        self.assertEqual(assessed_trial_counts[-1], 25)

    def test_valid_terminal_requires_exact_aligned_raw_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        for evidence_mode, message in (
            ("missing", "statistical_trials"),
            ("malformed", "finite numeric sequence"),
            ("wrong_count", "expected exactly 5"),
        ):
            with self.subTest(evidence_mode=evidence_mode):
                with tempfile.TemporaryDirectory() as td:
                    with self.assertRaisesRegex((RuntimeError, ValueError), message):
                        train_layerwise(
                            env=_FakeLayerwiseEnv(evidence_mode=evidence_mode),
                            policy=_FakePolicy(),
                            train_cfg=self._train_cfg(),
                            candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                            identity_context={"action_space_version": "layerwise-v1"},
                            optimizer=object(),
                            rollout_buffer=_FakeBuffer(),
                            ppo_update_fn=lambda *_args, **_kwargs: {},
                            assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                            step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                                np.asarray(spec.slot_mask, dtype=bool),
                                np.asarray(spec.slot_dims, dtype=np.int64),
                            ),
                        )

    def test_invalid_terminal_may_omit_raw_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(evidence_mode="missing", invalid=True),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(),
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(summary["episode_rewards"], [-5.0])
        self.assertEqual(summary["episode_records"][0].promotion_status, "invalid_terminal")

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

    def test_zero_remaining_resume_preserves_frontier_and_convergence_summary(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        matrix = [[layer % 2, 0, 1, 2, 3, 4] for layer in range(12)]
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 25,
                    metric1=[0.9] * 25,
                    metric2=[0.8] * 25,
                    seeds=list(range(1, 26)),
                ),
                {
                    "identity_context": context,
                    "action_matrix": matrix,
                    "variable_cost": 0.6,
                    "episode_reward": 1.25,
                    "assessment_bootstrap_seed": 77,
                    "boosted_overrides": [],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })
            config = self._train_cfg(total_episodes=0)
            config.absolute_episode_start = 60_000
            config.planned_total_episodes = 60_000
            config.convergence_resume_state = {
                "best_robust_feasible_cost": 0.6,
                "current_robust_feasible_cost": 0.6,
                "stall_update_windows": 101,
                "block4_entropy": 0.05,
                "k_entropy": 0.06,
                "converged": True,
                "extension_required": False,
            }
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.9),
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=store,
                identity_context=context,
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(summary["completed_episodes"], 60_000)
        self.assertEqual(summary["best_full_vector"], action)
        self.assertEqual(summary["best_reward"], 1.25)
        self.assertEqual(summary["best_promotion_evidence"]["trial_count"], 25)
        self.assertEqual(
            len(summary["best_promotion_evidence"]["trials"]["seeds"]), 25,
        )
        self.assertEqual(summary["block4_entropy"], 0.05)
        self.assertEqual(summary["k_entropy"], 0.06)
        self.assertEqual(summary["stall_update_windows"], 101)
        self.assertTrue(summary["converged"])
        self.assertEqual(summary["strict_best"]["full_vector"], action)
        self.assertTrue(summary["convergence_state"]["converged"])

    def test_promoted_reward_and_install_metadata_restore_from_promotion_record(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import restore_promoted_candidates
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        matrix = [[layer % 2, 0, 1, 2, 3, 4] for layer in range(12)]
        overrides = [{
            "block_idx": 4,
            "layer_idx": 3,
            "field_values": {"v_mask_rescale_sf": 47},
        }]
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 5,
                    metric1=[0.9] * 5,
                    metric2=[0.8] * 5,
                    seeds=list(range(1, 6)),
                ),
                {
                    "identity_context": context,
                    "action_matrix": matrix,
                    "variable_cost": 0.6,
                    "episode_reward": 9.0,
                    "assessment_bootstrap_seed": 77,
                    "boosted_overrides": [],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": context,
                "promotion_status": "promoted",
                "promotion_metadata": {
                    "action_matrix": matrix,
                    "variable_cost": 0.6,
                    "episode_reward": 1.25,
                    "assessment_bootstrap_seed": 77,
                    "boosted_overrides": overrides,
                },
                "fidelity": "F4",
                "valid": True,
            })
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 20,
                    metric1=[0.9] * 20,
                    metric2=[0.8] * 20,
                    seeds=list(range(6, 26)),
                ),
                {
                    "identity_context": context,
                    "action_matrix": matrix,
                    "variable_cost": 0.6,
                    "episode_reward": 7.5,
                    "assessment_bootstrap_seed": 77,
                    "boosted_overrides": [],
                },
            )

            with mock.patch.object(
                store,
                "read_all",
                side_effect=AssertionError("frontier restore must stream active records"),
            ):
                restored = restore_promoted_candidates(
                    candidate_store=store,
                    identity_context=context,
                    statistical_reference=object(),
                    assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
                    promotion_probability=0.8,
                    assessment_trial_limit=25,
                )

        candidate = next(iter(restored.values()))
        self.assertEqual(candidate["reward"], 1.25)
        self.assertEqual(
            candidate["boosted_overrides"],
            {(4, 3): {"v_mask_rescale_sf": 47}},
        )

    def test_zero_remaining_resume_clears_convergence_after_frontier_retraction(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        matrix = [[layer % 2, 0, 1, 2, 3, 4] for layer in range(12)]
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 25,
                    metric1=[0.9] * 25,
                    metric2=[0.8] * 25,
                    seeds=list(range(1, 26)),
                ),
                {
                    "identity_context": context,
                    "action_matrix": matrix,
                    "variable_cost": 0.6,
                    "episode_reward": 1.25,
                    "assessment_bootstrap_seed": 77,
                    "boosted_overrides": [],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })
            config = self._train_cfg(total_episodes=0)
            config.absolute_episode_start = 60_000
            config.planned_total_episodes = 60_000
            config.convergence_resume_state = {
                "best_robust_feasible_cost": 0.8,
                "current_robust_feasible_cost": 0.8,
                "stall_update_windows": 101,
                "block4_entropy": 0.05,
                "k_entropy": 0.06,
                "converged": True,
                "extension_required": False,
            }
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.9),
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=store,
                identity_context=context,
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(summary["best_variable_cost"], 0.6)
        self.assertEqual(summary["stall_update_windows"], 0)
        self.assertFalse(summary["converged"])

    def test_ppo_update_exposes_current_strict_frontier_snapshot(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        matrix = [[1, 0, 1, 2, 3, 4] for _ in range(12)]
        observed_updates = []
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 25,
                    metric1=[0.9] * 25,
                    metric2=[0.8] * 25,
                    seeds=list(range(1, 26)),
                ),
                {
                    "identity_context": context,
                    "action_matrix": matrix,
                    "variable_cost": 0.6,
                    "episode_reward": 1.25,
                    "assessment_bootstrap_seed": 77,
                    "boosted_overrides": [],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.9),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=1, update_every=1),
                candidate_store=store,
                identity_context=context,
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {"entropy": 0.0},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
                on_ppo_update_end=lambda metrics, *_args: observed_updates.append(metrics),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(observed_updates[0]["strict_best"]["full_vector"], action)
        self.assertEqual(observed_updates[0]["strict_best"]["variable_cost"], 0.4)
        self.assertEqual(summary["best_reward"], 1.25)


class _PromotionBase:
    def __init__(self, fresh_probability=0.9):
        self.statistical_reference = object()
        self.prepare_calls = []
        self.evaluate_calls = []
        self.fresh_probability = fresh_probability
        self.probe_noise_seed = None

    def prepare_action_for_terminal_probe(self, full_vec, **kwargs):
        self.prepare_calls.append((list(full_vec), dict(kwargs)))
        return {"prepared": True, "action": list(full_vec)}

    def evaluate_prepared_terminal_batch(self, prepared, **kwargs):
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

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
                "seeds": [
                    derive_probe_trial_seed(self.probe_noise_seed, i)
                    for i in range(count)
                ],
            },
            "statistical_assessment": {**fields, "bootstrap_seed": 77},
            "metrics": types.SimpleNamespace(
                loss_mean=0.3, metric1_mean=0.9, metric2_mean=0.8,
            ),
        })]


class LayerwisePromotionTests(unittest.TestCase):
    def _store_with_five(self, root, seeds=None):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        store = CandidateStore(Path(root) / "candidates.jsonl")
        store.append_trial_group(
            list(range(20)),
            TrialSeries(
                loss=[0.3] * 5,
                metric1=[0.9] * 5,
                metric2=[0.8] * 5,
                seeds=[1, 2, 3, 4, 5] if seeds is None else seeds,
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

    def test_promotion_assesses_existing_evidence_above_target_without_new_probe(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            for group_idx in range(1, 6):
                seeds = list(range(group_idx * 5 + 1, group_idx * 5 + 6))
                store.append_trial_group(
                    action,
                    TrialSeries(
                        loss=[0.3] * 5,
                        metric1=[0.9] * 5,
                        metric2=[0.8] * 5,
                        seeds=seeds,
                    ),
                    {"identity_context": context, "group_index": group_idx},
                )
            base = _PromotionBase()
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.85),
                priority=3,
                variable_cost=0.6,
                frontier_cost=0.5,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
            )

        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.trial_count, 30)
        self.assertEqual(result.fresh_trial_count, 0)
        self.assertEqual(base.prepare_calls, [])
        self.assertEqual(base.evaluate_calls, [])

    def test_promotion_recovers_pending_reassessment_after_top_up_crash(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 20,
                    metric1=[0.9] * 20,
                    metric2=[0.8] * 20,
                    seeds=list(range(100, 120)),
                ),
                {
                    "identity_context": context,
                    "promotion_marker": "fresh_top_up",
                    "promotion_status": "pending_reassessment",
                },
            )
            base = _PromotionBase()
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.85),
                priority=3,
                variable_cost=0.6,
                frontier_cost=0.5,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
            )

        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.trial_count, 25)
        self.assertEqual(result.fresh_trial_count, 0)
        self.assertEqual(base.prepare_calls, [])
        self.assertEqual(base.evaluate_calls, [])

    def test_restore_promoted_candidates_reassesses_persisted_frontier(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import restore_promoted_candidates
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        matrix = [[layer % 2, 0, 1, 2, 3, 4] for layer in range(12)]
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 25,
                    metric1=[0.9] * 25,
                    metric2=[0.8] * 25,
                    seeds=list(range(1, 26)),
                ),
                {
                    "identity_context": context,
                    "action_matrix": matrix,
                    "variable_cost": 0.6,
                    "episode_reward": 1.25,
                    "assessment_bootstrap_seed": 77,
                    "boosted_overrides": [{
                        "block_idx": 4,
                        "layer_idx": 3,
                        "field_values": {"v_mask_rescale_sf": 47},
                    }],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })

            restored = restore_promoted_candidates(
                candidate_store=store,
                identity_context=context,
                statistical_reference=object(),
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
                promotion_probability=0.8,
            )

        self.assertEqual(len(restored), 1)
        candidate = next(iter(restored.values()))
        self.assertEqual(candidate["action_matrix"], tuple(tuple(row) for row in matrix))
        self.assertEqual(candidate["full_vector"], tuple(action))
        self.assertEqual(candidate["variable_cost"], 0.6)
        self.assertEqual(candidate["reward"], 1.25)
        self.assertEqual(
            candidate["boosted_overrides"],
            {(4, 3): {"v_mask_rescale_sf": 47}},
        )

    def test_promotion_selects_fresh_trial_seeds_disjoint_from_existing_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

        action = list(range(20))
        context = {"action_space_version": "layerwise-v1"}
        key = candidate_key(action, context)
        material = f"layerwise-promotion:{key}:77:5".encode("utf-8")
        colliding_base = int.from_bytes(
            hashlib.sha256(material).digest()[:8], "big"
        ) & 0x7FFFFFFFFFFFFFFF
        existing_seeds = [derive_probe_trial_seed(colliding_base, i) for i in range(5)]

        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td, seeds=existing_seeds)
            base = _PromotionBase()
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.85),
                priority=3,
                variable_cost=0.6,
                frontier_cost=0.5,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
            )
            evidence = store.trial_evidence_for_action(action, context)

        self.assertEqual(result.status, "promoted")
        self.assertEqual(evidence.trial_count, 25)
        fresh_seeds = set(evidence.trials.seeds[5:])
        self.assertTrue(fresh_seeds.isdisjoint(existing_seeds))
        self.assertEqual(len(fresh_seeds), 20)

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
