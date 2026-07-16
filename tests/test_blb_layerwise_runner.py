from __future__ import annotations

import ast
import shutil
import hashlib
import inspect
import json
import math
import os
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

    def test_convergence_depends_on_policy_and_frontier_not_episode_count(self):
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
            completed_episodes=120,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
        )
        self.assertFalse(state.converged)
        self.assertEqual(state.stall_update_windows, 0)

        for update_idx in range(99):
            state = tracker.observe_update(
                completed_episodes=240 + 120 * update_idx,
                block4_entropy=0.9,
                k_entropy=0.9,
                robust_feasible_cost=0.4,
                robust_feasible_action_identity="candidate-a",
            )
        self.assertEqual(state.stall_update_windows, 99)
        self.assertFalse(state.converged)

        state = tracker.observe_update(
            completed_episodes=12_120,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
        )
        self.assertEqual(state.stall_update_windows, 100)
        self.assertTrue(state.converged)

        state = tracker.observe_update(
            completed_episodes=40_360,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.5,
            robust_feasible_action_identity="candidate-b",
        )
        self.assertEqual(state.stall_update_windows, 0)
        self.assertEqual(state.selected_action_stable_update_windows, 0)
        self.assertFalse(state.converged)

    def test_convergence_uses_stable_selected_action_not_entropy(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        state = tracker.observe_update(
            completed_episodes=120,
            block4_entropy=0.99,
            k_entropy=0.99,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
        )
        self.assertFalse(state.converged)

        for update_idx in range(100):
            state = tracker.observe_update(
                completed_episodes=240 + update_idx * 120,
                block4_entropy=None,
                k_entropy=None,
                robust_feasible_cost=0.4,
                robust_feasible_action_identity="candidate-a",
            )

        self.assertEqual(state.stall_update_windows, 100)
        self.assertEqual(state.selected_action_stable_update_windows, 100)
        self.assertEqual(state.selected_action_identity, "candidate-a")
        self.assertTrue(state.converged)

    def test_same_cost_selected_action_change_resets_action_stability(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        tracker.observe_update(
            completed_episodes=120,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
        )
        for update_idx in range(99):
            tracker.observe_update(
                completed_episodes=240 + update_idx * 120,
                block4_entropy=0.9,
                k_entropy=0.9,
                robust_feasible_cost=0.4,
                robust_feasible_action_identity="candidate-a",
            )

        state = tracker.observe_update(
            completed_episodes=12_120,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-b",
        )

        self.assertEqual(state.stall_update_windows, 100)
        self.assertEqual(state.selected_action_stable_update_windows, 0)
        self.assertEqual(state.selected_action_identity, "candidate-b")
        self.assertFalse(state.converged)

    def test_strict_best_snapshot_uses_candidate_key_as_final_tie_break(self):
        from blb_stage2_rl.layerwise_runner import _strict_best_snapshot

        common = {
            "variable_cost": 0.4,
            "assessment": _assessment(0.9),
            "metrics": {
                "loss_mean": 0.3,
                "metric1_mean": 0.9,
                "metric2_mean": 0.8,
            },
            "action_matrix": [[0, 0, 0, 0, 0, 0] for _ in range(12)],
            "full_vector": list(range(20)),
            "boosted_overrides": {},
            "reward": 1.4,
            "promotion_trials": None,
        }
        later_key = dict(common)
        earlier_key = dict(common)
        earlier_key["full_vector"] = list(range(20, 40))

        snapshot = _strict_best_snapshot({
            "candidate-z": later_key,
            "candidate-a": earlier_key,
        })

        self.assertEqual(snapshot["candidate_key"], "candidate-a")
        self.assertEqual(snapshot["full_vector"], list(range(20, 40)))

    def test_strict_selection_key_matches_candidate_key_tie_break(self):
        from blb_stage2_rl.layerwise_runner import strict_selection_key

        candidate = self._candidate(cost=0.5, probabilities=[0.9] * 6)
        lower_cost = self._candidate(cost=0.4, probabilities=[0.99] * 6)

        self.assertLess(
            strict_selection_key("candidate-a", candidate),
            strict_selection_key("candidate-z", candidate),
        )
        self.assertLess(
            strict_selection_key("candidate-z", candidate),
            strict_selection_key("candidate-a", lower_cost),
        )

    def test_orphaned_selected_action_counter_resets_on_restore(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        tracker.load_state_dict({
            "best_robust_feasible_cost": 0.4,
            "current_robust_feasible_cost": 0.4,
            "stall_update_windows": 37,
            "selected_action_stable_update_windows": 22,
        })

        state = tracker.state_dict()
        self.assertIsNone(state["selected_action_identity"])
        self.assertEqual(state["selected_action_stable_update_windows"], 0)

    def test_same_cost_resume_reconciliation_resets_stale_action_identity(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        tracker.load_state_dict({
            "best_robust_feasible_cost": 0.4,
            "current_robust_feasible_cost": 0.4,
            "stall_update_windows": 101,
            "selected_action_identity": "candidate-old",
            "selected_action_stable_update_windows": 101,
        })
        tracker.reconcile_frontier(0.4, "candidate-current")

        state = tracker.state_dict()
        self.assertEqual(state["stall_update_windows"], 101)
        self.assertEqual(state["selected_action_identity"], "candidate-current")
        self.assertEqual(state["selected_action_stable_update_windows"], 0)

    def test_nonfinite_entropy_is_unavailable_diagnostic_not_fatal(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        state = tracker.observe_update(
            completed_episodes=120,
            block4_entropy=float("nan"),
            k_entropy=float("inf"),
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
        )

        self.assertIsNone(state.block4_entropy)
        self.assertIsNone(state.k_entropy)
        self.assertEqual(state.selected_action_identity, "candidate-a")

    def test_diagnostics_best_snapshot_can_be_reconciled_without_episode_append(self):
        from blb_stage2_rl.diagnostics import EpisodeStats, RLDiagnosticsRecorder

        with tempfile.TemporaryDirectory() as td:
            recorder = RLDiagnosticsRecorder(
                output_dir=td,
                num_layers=12,
                num_action_slots=4,
            )
            stats = EpisodeStats(
                episode=120,
                total_reward=1.5,
                terminal_reward=1.5,
                per_step_sum=0.0,
                valid_steps=12,
                invalid_steps=0,
                steps_taken=12,
                total_bits=0,
                fusion_count=24,
                first_invalid_step=None,
                first_invalid_block=None,
                first_invalid_layer=None,
                early_terminated=False,
                terminal_cost_rank_score=0.4,
            )
            recorder.write_best_action_snapshot(
                episode_stats=stats,
                full_action_vec=np.asarray([1, 2, 3, 4]),
                best_reward_so_far=1.5,
            )

            payload = json.loads(Path(recorder.best_json_path).read_text())
            self.assertEqual(payload["action_vec"], [1, 2, 3, 4])
            self.assertFalse(Path(recorder.episodes_path).exists())

            recorder.clear_best_action_snapshot()
            self.assertFalse(Path(recorder.best_json_path).exists())

    def test_convergence_cannot_trigger_while_current_frontier_is_empty(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker()
        tracker.observe_update(
            completed_episodes=30_000,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
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
            robust_feasible_action_identity="candidate-a",
        )
        for update_idx in range(7):
            tracker.observe_update(
                completed_episodes=30_120 + update_idx * 120,
                block4_entropy=0.05,
                k_entropy=0.05,
                robust_feasible_cost=0.4,
                robust_feasible_action_identity="candidate-a",
            )

        restored = LayerwiseConvergenceTracker()
        restored.load_state_dict(tracker.state_dict())
        state = restored.observe_update(
            completed_episodes=31_000,
            block4_entropy=0.05,
            k_entropy=0.05,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
        )

        self.assertEqual(state.stall_update_windows, 8)
        self.assertEqual(state.best_robust_feasible_cost, 0.4)
        self.assertEqual(state.selected_action_identity, "candidate-a")
        self.assertEqual(state.selected_action_stable_update_windows, 8)

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

    def test_layerwise_episode_budget_zero_remains_unbounded_across_resume(self):
        from blb_stage2_rl.layerwise_runner import (
            is_unbounded_layerwise_training,
            resolve_layerwise_episode_budget,
        )

        self.assertEqual(resolve_layerwise_episode_budget(0, 0), 0)
        self.assertEqual(resolve_layerwise_episode_budget(0, 72_000), 0)
        self.assertEqual(resolve_layerwise_episode_budget(60_000, 12_000), 48_000)
        self.assertTrue(is_unbounded_layerwise_training(0, 0))
        self.assertFalse(is_unbounded_layerwise_training(0, 60_000))
        self.assertFalse(is_unbounded_layerwise_training(48_000, 60_000))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            resolve_layerwise_episode_budget(-1, 0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            resolve_layerwise_episode_budget(60_000, 60_001)

    def test_p3_cost_is_redistributed_without_changing_episode_return(self):
        from blb_stage2_rl.layerwise_runner import redistribute_layerwise_rewards

        layer_costs = tuple((index + 1) / 780.0 for index in range(12))
        variable_cost = sum(layer_costs)
        rewards = redistribute_layerwise_rewards(
            terminal_reward=1.75 + variable_cost,
            priority=3,
            variable_cost=variable_cost,
            layer_cost_rewards=layer_costs,
        )

        self.assertEqual(len(rewards), 12)
        self.assertEqual(rewards[:-1], layer_costs[:-1])
        self.assertAlmostEqual(rewards[-1], layer_costs[-1] + 1.75)
        self.assertAlmostEqual(sum(rewards), 1.75 + variable_cost)

    def test_p1_and_p2_never_receive_cost_credit(self):
        from blb_stage2_rl.layerwise_runner import redistribute_layerwise_rewards

        layer_costs = (0.01,) * 12
        for priority, terminal_reward in ((1, -3.2), (2, -1.7)):
            with self.subTest(priority=priority):
                rewards = redistribute_layerwise_rewards(
                    terminal_reward=terminal_reward,
                    priority=priority,
                    variable_cost=sum(layer_costs),
                    layer_cost_rewards=layer_costs,
                )
                self.assertEqual(rewards[:-1], (0.0,) * 11)
                self.assertEqual(rewards[-1], terminal_reward)

    def test_reward_redistribution_rejects_inconsistent_cost_terms(self):
        from blb_stage2_rl.layerwise_runner import redistribute_layerwise_rewards

        with self.assertRaisesRegex(ValueError, "sum"):
            redistribute_layerwise_rewards(
                terminal_reward=1.5,
                priority=3,
                variable_cost=0.5,
                layer_cost_rewards=(0.01,) * 12,
            )


class LayerwiseDispatchRulesTests(unittest.TestCase):
    def test_layerwise_candidate_identity_binds_k_level_order(self):
        from blb_stage2_rl.candidate_store import candidate_key
        from blb_stage2_rl import layerwise_runner

        binder = getattr(
            layerwise_runner, "bind_layerwise_candidate_identity", None,
        )
        self.assertIsNotNone(binder)
        if binder is None:
            return
        base = {"action_space_version": "stage2_layerwise_12x6_v1"}
        first_order = (8, 9, 11, 13, 10, 12)
        second_order = (13, 8, 9, 10, 11, 12)
        first = binder(base, first_order, "cost-v1")
        second = binder(base, second_order, "cost-v1")

        self.assertEqual(first["k_levels"], list(first_order))
        self.assertEqual(second["k_levels"], list(second_order))
        self.assertNotEqual(candidate_key([0], first), candidate_key([0], second))

    def test_layerwise_checkpoint_metadata_rejects_foreign_run_context(self):
        from blb_stage2_rl import layerwise_runner

        validator = getattr(
            layerwise_runner, "validate_layerwise_checkpoint_metadata", None,
        )
        self.assertIsNotNone(validator)
        if validator is None:
            return
        checkpoint = {
            "rl_variant": "layerwise",
            "algorithm_revision": "v3",
            "algorithm_contract_hash": "algorithm-a",
            "run_context_hash": "run-a",
        }
        validator(
            checkpoint,
            rl_variant="layerwise",
            algorithm_revision="v3",
            algorithm_contract_hash="algorithm-a",
            run_context_hash="run-a",
        )
        with self.assertRaisesRegex(RuntimeError, "run context"):
            validator(
                checkpoint,
                rl_variant="layerwise",
                algorithm_revision="v3",
                algorithm_contract_hash="algorithm-a",
                run_context_hash="run-b",
            )
        with self.assertRaisesRegex(RuntimeError, "algorithm revision"):
            validator(
                {
                    **checkpoint,
                    "algorithm_revision": "factorized_slot_credit_natural_convergence_v5",
                },
                rl_variant="layerwise",
                algorithm_revision="factorized_slot_credit_equivalence_convergence_v6",
                algorithm_contract_hash="algorithm-a",
                run_context_hash="run-a",
            )

    def test_checkpoint_file_fingerprint_allows_suffix_only_and_rejects_prefix_change(self):
        from blb_stage2_rl import layerwise_runner

        fingerprint = getattr(
            layerwise_runner, "checkpoint_file_fingerprints", None,
        )
        validator = getattr(
            layerwise_runner, "validate_checkpoint_file_fingerprints", None,
        )
        self.assertIsNotNone(fingerprint)
        self.assertIsNotNone(validator)
        if fingerprint is None or validator is None:
            return
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "episodes.jsonl"
            path.write_bytes(b"abc\ndef\n")
            specs = {"episodes": (path, 4)}
            expected = fingerprint(specs)
            path.write_bytes(b"abc\ndef\nnew\n")
            validator(expected, specs)
            path.write_bytes(b"xbc\ndef\nnew\n")
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                validator(expected, specs)

    def test_fresh_layerwise_run_rejects_stale_marker_without_checkpoint(self):
        from blb_stage2_rl import layerwise_runner

        validator = getattr(
            layerwise_runner, "validate_fresh_layerwise_run_state", None,
        )
        self.assertIsNotNone(validator)
        if validator is None:
            return
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            marker = root / "rl_data_points_run_id.txt"
            candidate = root / "candidate_store.jsonl"
            validator(marker, (candidate,))
            marker.write_text("old-run\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "without a checkpoint"):
                validator(marker, (candidate,))

    def test_layerwise_run_lock_rejects_second_writer_and_records_context(self):
        from blb_stage2_rl import layerwise_runner

        lock_type = getattr(layerwise_runner, "LayerwiseRunLock", None)
        self.assertIsNotNone(lock_type)
        if lock_type is None:
            return
        with tempfile.TemporaryDirectory() as td:
            with lock_type(td) as first:
                first.bind_context("run-context-a")
                payload = Path(first.path).read_text(
                    encoding="utf-8",
                )
                self.assertIn('"run_context_hash": "run-context-a"', payload)
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with lock_type(td):
                        pass
            with lock_type(td):
                pass

    def test_stage2_run_lock_survives_deletion_of_fresh_run_directory(self):
        from blb_stage2_rl import layerwise_runner

        lock_type = getattr(layerwise_runner, "LayerwiseRunLock", None)
        self.assertIsNotNone(lock_type)
        if lock_type is None:
            return
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "mrpc" / "constraint"
            progress_dir = run_dir / "stage2_noise" / "progress"
            progress_dir.mkdir(parents=True)
            with lock_type(progress_dir) as first:
                self.assertFalse(str(first.path).startswith(str(run_dir) + os.sep))
                shutil.rmtree(run_dir)
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with lock_type(progress_dir):
                        pass

    def test_checkpoint_fingerprint_tracker_hashes_only_new_suffixes(self):
        from blb_stage2_rl import layerwise_runner

        tracker_type = getattr(
            layerwise_runner, "CheckpointFileFingerprintTracker", None,
        )
        self.assertIsNotNone(tracker_type)
        if tracker_type is None:
            return
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "episodes.jsonl"
            path.write_bytes(b"abc\n")
            tracker = tracker_type()
            first = tracker.fingerprints({"episodes": (path, 4)})
            self.assertEqual(tracker.bytes_hashed, 4)
            path.write_bytes(b"abc\ndef\n")
            second = tracker.fingerprints({"episodes": (path, 8)})
            self.assertEqual(tracker.bytes_hashed, 8)
            self.assertNotEqual(first, second)
            self.assertEqual(
                tracker.fingerprints({"episodes": (path, 8)}), second,
            )
            self.assertEqual(tracker.bytes_hashed, 8)

    def test_layerwise_branch_locks_and_writes_episode_zero_checkpoint(self):
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

        self.assertIn("validate_fresh_layerwise_run_state(", branch_source)
        self.assertIn("fingerprint_tracker.fingerprints(", branch_source)
        self.assertLess(
            branch_source.index("save_layerwise_checkpoint(\n            completed=0"),
            branch_source.index("summary = train_layerwise("),
        )
        public = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "run_sequential_via_runner"
        )
        public_source = ast.get_source_segment(source, public)
        self.assertIn("with LayerwiseRunLock(blb_progress_dir)", public_source)
        self.assertIn("_run_sequential_via_runner_locked(", public_source)

    def test_launcher_locks_stage2_directory_before_fresh_cleanup(self):
        source = Path("llama_7B_LayerImportance.sh").read_text(encoding="utf-8")
        stage2 = source[source.index(
            'PERSISTENT_DIR="${PERSISTENT_ROOT}/${SEARCH_ALGORITHM}/${MODEL_TYPE}/'
            '${DATASET}/${CONSTRAINT_SLUG}"'
        ):]
        cleanup = stage2.index('rm -rf "$PERSISTENT_DIR"')
        lock = stage2.index('flock -n "$BLB_STAGE2_RUN_LOCK_FD"')

        self.assertLess(lock, cleanup)
        self.assertIn("BLB_STAGE2_RUN_LOCK_PATH", stage2[:cleanup])
        self.assertIn("export BLB_STAGE2_RUN_LOCK_FD", stage2[:cleanup])
        self.assertIn("BLB_STAGE2_RUN_LOCK_FD=9", stage2[:cleanup])
        self.assertIn(
            'exec 9>>"$BLB_STAGE2_RUN_LOCK_PATH"',
            stage2[:cleanup],
        )
        self.assertNotIn("exec {BLB_STAGE2_RUN_LOCK_FD}", stage2[:cleanup])

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
        self.assertIn(
            "probe_diagnostics = _to_plain_mapping(record.probe_diagnostics)",
            branch_source,
        )
        callback = next(
            node for node in branch.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "on_layerwise_episode"
        )
        episode_stats_call = next(
            node for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "EpisodeStats"
        )
        keyword_sources = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in episode_stats_call.keywords
        }
        expected_probe_keys = {
            "terminal_probe_wall_seconds": "wall_seconds",
            "terminal_probe_devices": "devices",
            "terminal_probe_trial_counts": "per_worker_trial_counts",
            "terminal_probe_trial_indices": "per_worker_trial_indices",
            "terminal_probe_speedup": "speedup_vs_sequential",
            "terminal_cost_eval_wall_seconds": "cost_eval_wall_seconds",
            "terminal_probe_install_wall_seconds": "probe_install_wall_seconds",
            "terminal_probe_clear_wall_seconds": "probe_clear_wall_seconds",
            "terminal_probe_install_skipped": "probe_install_skipped",
            "terminal_probe_clear_skipped": "probe_clear_skipped",
        }
        for field_name, probe_key in expected_probe_keys.items():
            expression = keyword_sources[field_name]
            self.assertIn("probe_diagnostics", expression)
            self.assertIn(probe_key, expression)

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
            "ppo_update_counter = int(resumed_ppo_update_count)",
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
            '"candidate_store_size": int(candidate_store_size)',
            "candidate_store.path.stat().st_size",
            "candidate_store.recover_to_checkpoint_size",
            'checkpoint.get("store_file_fingerprints")',
            '"store_file_fingerprints": store_file_fingerprints',
            'checkpoint.get("diagnostics_jsonl_sizes")',
            '"diagnostics_jsonl_sizes": diagnostics_jsonl_sizes',
            "diag_recorder.committed_jsonl_sizes()",
            "diag_recorder.recover_to_checkpoint_sizes",
            "checkpoint_planned_total = int(checkpoint.get(",
            '"planned_total_episodes", planned_total_episodes',
            'checkpoint.get("ppo_update_count")',
            '"ppo_update_count": int(ppo_update_counter)',
        ):
            self.assertIn(required, branch_source)

    def test_layerwise_branch_uses_reward_only_natural_convergence_contract(self):
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

        self.assertIn("factorized_actor_clip=True", branch_source)
        self.assertIn("entropy_average_active_slots=True", branch_source)
        self.assertIn("entropy_normalize_active_slots=True", branch_source)
        self.assertIn("per_slot_entropy_recovery=False", branch_source)
        self.assertIn("ent_coef=0.0", branch_source)
        self.assertNotIn("ent_coef_cosine_", branch_source)
        self.assertIn('"factorized_actor_clip": True', branch_source)
        self.assertIn('"algorithm_revision": algorithm_revision', branch_source)
        self.assertIn('"factorized_slot_credit_equivalence_convergence_v6"', branch_source)
        self.assertIn('"algorithm_contract_hash": algorithm_contract_hash', branch_source)
        self.assertIn('"run_context_hash": run_context_hash', branch_source)
        self.assertIn("validate_layerwise_checkpoint_metadata(", branch_source)
        self.assertIn('"cost_model_revision": LAYERWISE_COST_MODEL_REVISION', source)
        self.assertIn('"ppo": asdict(ppo)', branch_source)
        self.assertIn('"rollout_size": int(train_cfg.rollout_size)', branch_source)
        self.assertIn(
            '"persistence_protocol": "stable_parent_lock_incremental_fingerprint_v2"',
            branch_source,
        )
        self.assertIn(
            '"actor_advantage_normalization": "per_slot_center_shared_scale_v1"',
            branch_source,
        )
        self.assertIn(
            '"behavior_log_prob_source": "sampling_time_per_slot_v1"',
            branch_source,
        )
        self.assertIn("bind_layerwise_candidate_identity(", source)
        self.assertIn('"entropy_average_active_slots": True', branch_source)
        self.assertIn('"kind": "disabled"', branch_source)
        self.assertIn('"coefficient": 0.0', branch_source)
        self.assertIn('"optimization_role": "monitor_only"', branch_source)
        self.assertIn('"mode": "natural_convergence"', branch_source)
        self.assertNotIn('"block4_entropy_below": 0.1', branch_source)
        self.assertNotIn('"k_entropy_below": 0.1', branch_source)
        self.assertIn('"frontier_stall_update_windows": 100', branch_source)
        self.assertIn('"selected_action_stable_update_windows": 100', branch_source)
        self.assertIn('"selection_tie_break": "candidate_key"', branch_source)
        self.assertIn("strict_selection_key(", branch_source)
        self.assertIn("candidate_key(record.pending_full_vector, identity_context)", branch_source)
        self.assertIn("diag_recorder.write_best_action_snapshot(", branch_source)
        self.assertIn("diag_recorder.clear_best_action_snapshot()", branch_source)
        self.assertIn('"counts_only_finite_ppo_updates": True', branch_source)
        self.assertIn("resolve_layerwise_episode_budget(", branch_source)
        self.assertIn("completion_status", branch_source)
        self.assertIn("diag_recorder.finalize(status=completion_status)", branch_source)
        self.assertIn('"status": completion_status', branch_source)
        self.assertIn(
            '"blb_v3_total_episodes": int(summary.get("completed_episodes"',
            branch_source,
        )
        self.assertGreater(
            branch_source.index("write_strict_json_file(layerwise_manifest_path"),
            branch_source.index("fingerprint_tracker.validate_and_seed("),
        )
        self.assertGreater(
            branch_source.index("candidate_store.recover_to_checkpoint_size"),
            branch_source.index("fingerprint_tracker.validate_and_seed("),
        )

    def test_factorized_ppo_uses_active_slot_kl_scale(self):
        source = Path("blb_stage2_rl/sequential_policy.py").read_text(
            encoding="utf-8",
        )

        self.assertIn("def _factorized_approx_kl(", source)
        self.assertIn(
            "approx_kl_t = _factorized_approx_kl(",
            source,
        )

    def test_factorized_ppo_uses_per_slot_cost_control_variate(self):
        source = Path("blb_stage2_rl/sequential_policy.py").read_text(
            encoding="utf-8",
        )

        self.assertIn("def set_actor_cost_at(", source)
        self.assertIn("def set_actor_shared_return_at(", source)
        self.assertIn("def factorized_actor_advantages(", source)
        self.assertIn("factorized_actor_advantages = (", source)

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

        self.assertIn('"stage2_layerwise_robust_run_v2"', branch_source)
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
        self.assertIn('"converged"', branch_source)
        self.assertIn('"budget_exhausted"', branch_source)
        self.assertIn('completion_status = "failed"', branch_source)
        self.assertIn('"status": completion_status', branch_source)
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
        for field in (
            "nonfinite_minibatches",
            "nonfinite_update_skipped",
            "convergence_update_counted",
            "selected_action_identity",
            "selected_action_stable_update_windows",
        ):
            self.assertIn(field, branch_source)

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

    def test_zero_episode_limit_is_reserved_for_layerwise_robust_training(self):
        from blb_stage2_rl.layerwise_runner import validate_stage2_episode_limit_mode

        self.assertEqual(
            validate_stage2_episode_limit_mode(
                0,
                fusion_count_action=True,
                decision_granularity="layer",
                reward_design="robust_constrained",
                sequential_rl=True,
                substage_mode=False,
                stage2_rl_variant="blb_v3",
            ),
            0,
        )
        self.assertEqual(
            validate_stage2_episode_limit_mode(
                600,
                fusion_count_action=True,
                decision_granularity="block",
                reward_design="stage1_aligned",
                sequential_rl=True,
                substage_mode=False,
                stage2_rl_variant="blb_v3",
            ),
            600,
        )
        with self.assertRaisesRegex(ValueError, "only.*layerwise robust"):
            validate_stage2_episode_limit_mode(
                0,
                fusion_count_action=True,
                decision_granularity="block",
                reward_design="stage1_aligned",
                sequential_rl=True,
                substage_mode=False,
                stage2_rl_variant="blb_v3",
            )
        for unsupported in (
            {"sequential_rl": False},
            {"substage_mode": True},
            {"stage2_rl_variant": "legacy_v2"},
        ):
            values = {
                "fusion_count_action": True,
                "decision_granularity": "layer",
                "reward_design": "robust_constrained",
                "sequential_rl": True,
                "substage_mode": False,
                "stage2_rl_variant": "blb_v3",
                **unsupported,
            }
            with self.subTest(unsupported=unsupported), self.assertRaisesRegex(
                    ValueError, "only.*layerwise robust",
            ):
                validate_stage2_episode_limit_mode(0, **values)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_stage2_episode_limit_mode(
                -1,
                fusion_count_action=True,
                decision_granularity="layer",
                reward_design="robust_constrained",
                sequential_rl=True,
                substage_mode=False,
                stage2_rl_variant="blb_v3",
            )
        runner_source = Path("blb_stage2_rl/runner.py").read_text(encoding="utf-8")
        self.assertIn("validate_stage2_episode_limit_mode(", runner_source)
        evaluator_source = Path("layer_importance_evaluator.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("validate_stage2_episode_limit_mode(", evaluator_source)
        self.assertIn('noise_stage_result.get("status", "completed")', evaluator_source)
        self.assertIn('"blb_v3_total_episodes"', evaluator_source)

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

    def add_reward_at(self, index, delta):
        self.transitions[int(index)]["reward"] += float(delta)

    def set_actor_cost_at(self, index, per_slot_cost):
        self.transitions[int(index)]["actor_cost_per_slot"] = np.asarray(
            per_slot_cost, dtype=np.float32,
        )

    def set_actor_shared_return_at(self, index, shared_return):
        self.transitions[int(index)]["actor_shared_return"] = float(shared_return)

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
        result = action, np.asarray([float(mask.sum())]), np.asarray([0.25])
        if _kwargs.get("return_per_slot_log_prob", False):
            return result + (mask.astype(np.float32),)
        return result


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
        layer_cost = 0.4 / 12.0
        slot_cost_rewards = []
        for layer_idx in range(12):
            active_count = 5 if layer_idx == 0 else 6
            row = [layer_cost / active_count] * 6
            if layer_idx == 0:
                row[1] = 0.0
            slot_cost_rewards.append(row)
        return np.zeros(4, dtype=np.float32), (-5.0 if self._invalid else 7.5), True, {
            "policy_actions": [row[:] for row in self.actions],
            "pending_full_vector": list(range(20)),
            "variable_cost": {
                "normalized": 0.4,
                "layer_cost_rewards": [0.4 / 12.0] * 12,
                "slot_cost_rewards": slot_cost_rewards,
            },
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
        expected_local = 0.4 / 12.0
        for row in buffer.transitions[:-1]:
            self.assertAlmostEqual(row["reward"], expected_local)
        self.assertAlmostEqual(
            buffer.transitions[-1]["reward"],
            7.5 - 0.4 + expected_local,
        )
        self.assertAlmostEqual(
            sum(row["reward"] for row in buffer.transitions), 7.5,
        )
        for row in buffer.transitions:
            self.assertIn("actor_cost_per_slot", row)
            self.assertEqual(row["actor_shared_return"], 7.5 - 0.4)
            self.assertAlmostEqual(
                float(row["actor_cost_per_slot"].sum()), expected_local,
            )
        self.assertEqual([row["done"] for row in buffer.transitions], [False] * 11 + [True])
        self.assertEqual([float(row["log_prob"]) for row in buffer.transitions], [5.0] + [6.0] * 11)
        self.assertFalse(policy.masks[0][1])
        self.assertEqual(env.actions[0][1], 0)
        self.assertEqual(observed_ppo[0][0:3], (12, 1.0, 1.0))
        self.assertEqual(observed_ppo[0][3]["ent_coef_override"], 0.0)
        self.assertEqual(summary["episode_records"][0].action_matrix[0][1], 0)
        self.assertEqual(summary["episode_records"][0].pending_full_vector, tuple(range(20)))
        self.assertEqual(
            summary["episode_records"][0].probe_diagnostics,
            env.runtime_terminal_info["probe_diagnostics"],
        )
        self.assertEqual(summary["episode_rewards"], [7.5])
        self.assertIn("best_action", summary)
        self.assertIsNone(summary["best_action"])

    def test_unbounded_training_stops_only_after_natural_convergence(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        action_matrix = [[1, 0, 1, 2, 3, 4] for _ in range(12)]
        frontier = {
            "variable_cost": 0.4,
            "assessment": _assessment(0.9),
            "metrics": {"loss_mean": 0.3, "metric1_mean": 0.9, "metric2_mean": 0.8},
            "action_matrix": action_matrix,
            "full_vector": list(range(20)),
            "boosted_overrides": {},
            "reward": 1.4,
            "promotion_trials": None,
        }
        config = self._train_cfg(total_episodes=0, update_every=1)
        config.convergence_resume_state = {
            "best_robust_feasible_cost": 0.4,
            "current_robust_feasible_cost": 0.4,
            "stall_update_windows": 99,
            "selected_action_identity": "frontier",
            "selected_action_stable_update_windows": 99,
            "block4_entropy": 0.99,
            "k_entropy": 0.99,
            "converged": False,
        }
        update_ent_coef = []

        def fake_update(*_args, **kwargs):
            update_ent_coef.append(kwargs["ent_coef_override"])
            return {"entropy": 0.0}

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.restore_promoted_candidates",
            return_value={"frontier": frontier},
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner._current_policy_entropy",
            return_value={
                "block4": 0.99,
                "k": 0.99,
                "block4_slot_count": 12,
                "k_slot_count": 59,
            },
        ):
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=fake_update,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(summary["completed_episodes"], 1)
        self.assertEqual(len(summary["episode_records"]), 1)
        self.assertEqual(update_ent_coef, [0.0])
        self.assertTrue(summary["converged"])
        self.assertEqual(summary["stall_update_windows"], 100)
        self.assertEqual(summary["selected_action_identity"], "frontier")
        self.assertEqual(summary["selected_action_stable_update_windows"], 100)

    def test_positive_episode_budget_remains_a_bounded_smoke_run(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=2, update_every=1),
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

        self.assertEqual(summary["completed_episodes"], 2)
        self.assertEqual(len(summary["episode_records"]), 2)
        self.assertFalse(summary["converged"])

    def test_exhausted_bounded_resume_does_not_become_unbounded(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        config = self._train_cfg(total_episodes=0, update_every=1)
        config.absolute_episode_start = 2
        config.planned_total_episodes = 2
        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                policy=_FakePolicy(),
                train_cfg=config,
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

        self.assertEqual(summary["completed_episodes"], 2)
        self.assertEqual(summary["episode_records"], [])
        self.assertFalse(summary["converged"])

    def test_nonfinite_restored_entropy_is_unavailable_not_fatal(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        config = self._train_cfg(total_episodes=0, update_every=1)
        config.absolute_episode_start = 2
        config.planned_total_episodes = 2
        config.convergence_resume_state = {
            "block4_entropy": float("nan"),
            "k_entropy": float("inf"),
            "converged": False,
        }
        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                policy=_FakePolicy(),
                train_cfg=config,
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

        self.assertIsNone(summary["block4_entropy"])
        self.assertIsNone(summary["k_entropy"])

    def test_nonfinite_skipped_update_does_not_advance_convergence_patience(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        action_matrix = [[1, 0, 1, 2, 3, 4] for _ in range(12)]
        frontier = {
            "variable_cost": 0.4,
            "assessment": _assessment(0.9),
            "metrics": {"loss_mean": 0.3, "metric1_mean": 0.9, "metric2_mean": 0.8},
            "action_matrix": action_matrix,
            "full_vector": list(range(20)),
            "boosted_overrides": {},
            "reward": 1.4,
            "promotion_trials": None,
        }
        config = self._train_cfg(total_episodes=1, update_every=1)
        config.convergence_resume_state = {
            "best_robust_feasible_cost": 0.4,
            "current_robust_feasible_cost": 0.4,
            "stall_update_windows": 99,
            "selected_action_identity": "frontier",
            "selected_action_stable_update_windows": 99,
            "block4_entropy": 0.2,
            "k_entropy": 0.2,
            "converged": False,
        }

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.restore_promoted_candidates",
            return_value={"frontier": frontier},
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner._current_policy_entropy",
            return_value={
                "block4": 0.05,
                "k": 0.05,
                "block4_slot_count": 12,
                "k_slot_count": 59,
            },
        ):
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {
                    "entropy": 0.0,
                    "nonfinite_minibatches": 1,
                    "nonfinite_update_skipped": True,
                },
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(summary["stall_update_windows"], 99)
        self.assertEqual(summary["selected_action_identity"], "frontier")
        self.assertEqual(summary["selected_action_stable_update_windows"], 99)
        self.assertFalse(summary["converged"])
        self.assertFalse(summary["ppo_metrics"][0]["convergence_update_counted"])

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

        buffer = _FakeBuffer()
        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(evidence_mode="missing", invalid=True),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(),
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=buffer,
                ppo_update_fn=lambda *_args, **_kwargs: {},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(summary["episode_rewards"], [-5.0])
        self.assertEqual(summary["episode_records"][0].promotion_status, "invalid_terminal")
        self.assertEqual([row["reward"] for row in buffer.transitions[:-1]], [0.0] * 11)
        self.assertEqual(buffer.transitions[-1]["reward"], -5.0)
        self.assertTrue(all(
            np.count_nonzero(row["actor_cost_per_slot"]) == 0
            for row in buffer.transitions
        ))
        self.assertEqual(
            [row["actor_shared_return"] for row in buffer.transitions],
            [-5.0] * 12,
        )

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
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key
        from blb_stage2_rl.layerwise_action import compute_variable_cost_from_action_matrix
        from blb_stage2_rl.layerwise_runner import train_layerwise
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        expected_identity = candidate_key(action, context)
        matrix = [[layer % 2, 0, 1, 2, 3, 4] for layer in range(12)]
        expected_cost = compute_variable_cost_from_action_matrix(matrix).normalized
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
                "best_robust_feasible_cost": expected_cost,
                "current_robust_feasible_cost": expected_cost,
                "stall_update_windows": 101,
                "selected_action_identity": expected_identity,
                "selected_action_stable_update_windows": 101,
                "block4_entropy": 0.95,
                "k_entropy": 0.96,
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
        self.assertEqual(summary["block4_entropy"], 0.95)
        self.assertEqual(summary["k_entropy"], 0.96)
        self.assertEqual(summary["stall_update_windows"], 101)
        self.assertEqual(summary["selected_action_identity"], expected_identity)
        self.assertEqual(summary["selected_action_stable_update_windows"], 101)
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

    def test_bounded_resume_clears_convergence_after_frontier_retraction(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_action import compute_variable_cost_from_action_matrix
        from blb_stage2_rl.layerwise_runner import train_layerwise
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        matrix = [[layer % 2, 0, 1, 2, 3, 4] for layer in range(12)]
        expected_cost = compute_variable_cost_from_action_matrix(matrix).normalized
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
            config = self._train_cfg(total_episodes=1)
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

        self.assertEqual(summary["completed_episodes"], 60_001)
        self.assertEqual(summary["best_variable_cost"], 0.4)
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
        matrix = [[0, 3, 3, 3, 3, 3] for _layer in range(12)]
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
        self.assertEqual(candidate["variable_cost"], 0.0)
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
