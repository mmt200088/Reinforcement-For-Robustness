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
    def _candidate(
            *, cost, probabilities, margins=None, full_vector=None,
            loss=0.3, metric1=0.9, metric2=0.8,
            ):
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
            "constraint_safety_margins": list(
                [0.1] * 6 if margins is None else margins
            ),
            "full_vector": list([0] if full_vector is None else full_vector),
        }

    def test_strict_rank_orders_cost_then_confidence_then_safety_margin(self):
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

        weak_margin = self._candidate(
            cost=0.5, probabilities=[0.91] * 6, margins=[0.01] * 6,
        )
        strong_margin = self._candidate(
            cost=0.5, probabilities=[0.91] * 6, margins=[0.03] * 6,
        )
        self.assertLess(strict_rank_key(strong_margin), strict_rank_key(weak_margin))

    def test_strict_rank_uses_all_six_confidences_and_margins(self):
        from blb_stage2_rl.layerwise_runner import strict_rank_key

        weaker_confidence = self._candidate(
            cost=0.5,
            probabilities=[0.80, 0.80, 0.90, 0.90, 1.00, 1.00],
        )
        stronger_confidence = self._candidate(
            cost=0.5,
            probabilities=[0.80, 0.85, 0.85, 0.90, 1.00, 1.00],
        )
        self.assertLess(
            strict_rank_key(stronger_confidence),
            strict_rank_key(weaker_confidence),
        )

        weaker_margin = self._candidate(
            cost=0.5,
            probabilities=[0.90] * 6,
            margins=[0.10, 0.10, 0.20, 0.20, 0.30, 0.30],
        )
        stronger_margin = self._candidate(
            cost=0.5,
            probabilities=[0.90] * 6,
            margins=[0.10, 0.15, 0.15, 0.20, 0.30, 0.30],
        )
        self.assertLess(strict_rank_key(stronger_margin), strict_rank_key(weaker_margin))

    def test_normalized_safety_margins_cover_all_six_constraints(self):
        from blb_stage2_rl.layerwise_runner import normalized_constraint_safety_margins

        metrics = {
            "loss_mean": 0.30,
            "metric1_mean": 0.81,
            "metric2_mean": 0.72,
            "loss_std": 0.03,
            "metric1_std": 0.02,
            "metric2_std": 0.01,
        }
        reference = types.SimpleNamespace(
            loss_limit=0.40,
            metric1_limit=0.80,
            metric2_limit=0.70,
            loss_std_limit=0.04,
            metric1_std_limit=0.04,
            metric2_std_limit=0.02,
        )

        margins = normalized_constraint_safety_margins(metrics, reference)

        self.assertEqual(len(margins), 6)
        self.assertAlmostEqual(margins[0], 0.25)
        self.assertAlmostEqual(margins[1], 0.0125)
        self.assertAlmostEqual(margins[2], 2.0 / 70.0)
        self.assertAlmostEqual(margins[3], 0.25)
        self.assertAlmostEqual(margins[4], 0.5)
        self.assertAlmostEqual(margins[5], 0.5)

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

    def test_evidence_identity_context_separates_probe_and_full_validation(self):
        from blb_stage2_rl.candidate_store import candidate_key
        from blb_stage2_rl.layerwise_runner import evidence_identity_context

        base = {
            "action_space_version": "stage2_layerwise_12x6_v1",
            "fidelity": None,
        }
        action = [1, 2, 3]

        probe = evidence_identity_context(base, "F1")
        full = evidence_identity_context(base, "F4")

        self.assertEqual(probe["fidelity"], "F1")
        self.assertEqual(full["fidelity"], "F4")
        self.assertIsNone(base["fidelity"])
        self.assertNotEqual(candidate_key(action, probe), candidate_key(action, full))

    def test_full_validation_loader_is_uncapped_and_does_not_use_probe_subset(self):
        source_path = Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "runner.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "_build_validation_full_batches"
        )
        method_source = ast.get_source_segment(
            source_path.read_text(encoding="utf-8"), method,
        )

        self.assertIn('dataset_splits.get("validation_full")', method_source)
        self.assertNotIn("_get_stability_probe", method_source)
        self.assertNotIn("_effective_probe_batch_count", method_source)
        self.assertNotIn("break", method_source)

    def test_layerwise_branch_routes_promotion_through_authoritative_env(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "sequential_runner.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn("def _build_authoritative_validation_env(", source)
        self.assertIn("validation_full_batches = runner._build_validation_full_batches(ev)", source)
        self.assertIn("promotion_base_env=promotion_base_env", source)
        self.assertIn("authoritative_robust_reference", source)

    def test_layerwise_online_probe_is_fixed_to_256_examples(self):
        runner_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "runner.py"
        )
        sequential_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "sequential_runner.py"
        )
        runner_source = runner_path.read_text(encoding="utf-8")
        sequential_source = sequential_path.read_text(encoding="utf-8")

        self.assertIn("probe_size_override: Optional[int] = None", runner_source)
        self.assertIn(
            "probe_size_override=(256 if decision_path == \"layerwise\" else None)",
            sequential_source,
        )
        self.assertIn("if online_probe_example_count != 256:", sequential_source)

    def test_objective_convergence_ignores_episode_count_and_requires_revalidation(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker(patience_updates=100)
        tracker.observe_update(
            completed_episodes=1,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
        )
        for update_idx in range(100):
            state = tracker.observe_update(
                completed_episodes=2 + update_idx,
                block4_entropy=0.9,
                k_entropy=0.9,
                robust_feasible_cost=0.4,
                robust_feasible_action_identity="candidate-a",
            )

        self.assertEqual(state.stall_update_windows, 100)
        self.assertTrue(state.plateau_ready)
        self.assertFalse(state.converged)
        self.assertEqual(state.termination_reason, "running")

        state = tracker.observe_update(
            completed_episodes=1_000_000,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
            count_patience=False,
            strict_revalidation_passed=True,
        )
        self.assertTrue(state.converged)
        self.assertEqual(state.termination_reason, "converged")

        huge_episode_without_plateau = LayerwiseConvergenceTracker().observe_update(
            completed_episodes=10_000_000,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-b",
        )
        self.assertFalse(huge_episode_without_plateau.plateau_ready)
        self.assertFalse(huge_episode_without_plateau.converged)

    def test_convergence_checkpoint_persists_and_validates_contract(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker(patience_updates=100)
        state = tracker.state_dict()

        self.assertEqual(state["patience_updates"], 100)
        self.assertNotIn("minimum_episodes", state)
        self.assertNotIn("maximum_episodes", state)

        incompatible = dict(state, patience_updates=99)
        with self.assertRaisesRegex(ValueError, "convergence contract mismatch"):
            tracker.load_state_dict(incompatible)

    def test_layerwise_contract_records_multifidelity_and_honest_stop_statuses(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "sequential_runner.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn("factorized_slot_credit_multifidelity_convergence_v8", source)
        self.assertIn('"F1": {', source)
        self.assertIn('"F4": {', source)
        self.assertNotIn('"minimum_episodes": convergence_min_episodes', source)
        self.assertNotIn('"maximum_episodes": convergence_max_episodes', source)
        self.assertNotIn('completion_status = "budget_cap_reached"', source)
        self.assertIn('"strict_revalidation_required": True', source)
        self.assertIn('completion_status = "bounded_budget_exhausted"', source)

    def test_convergence_depends_on_policy_and_frontier_not_episode_count(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker(patience_updates=100)
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
        self.assertTrue(state.plateau_ready)
        self.assertFalse(state.converged)

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

        tracker = LayerwiseConvergenceTracker(patience_updates=100)
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
        self.assertTrue(state.plateau_ready)
        self.assertFalse(state.converged)

        state = tracker.observe_update(
            completed_episodes=12_241,
            block4_entropy=0.99,
            k_entropy=0.99,
            robust_feasible_cost=0.4,
            robust_feasible_action_identity="candidate-a",
            count_patience=False,
            strict_revalidation_passed=True,
        )
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

    def test_strict_best_snapshot_uses_action_lexicographic_final_tie_break(self):
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
            "constraint_safety_margins": [0.1] * 6,
        }
        lex_first = dict(common)
        lex_last = dict(common)
        lex_first["full_vector"] = [0, 9]
        lex_last["full_vector"] = [1, 0]

        snapshot = _strict_best_snapshot({
            "candidate-z": lex_first,
            "candidate-a": lex_last,
        })

        self.assertEqual(snapshot["candidate_key"], "candidate-z")
        self.assertEqual(snapshot["full_vector"], [0, 9])

    def test_strict_selection_key_uses_action_vector_before_candidate_key(self):
        from blb_stage2_rl.layerwise_runner import strict_selection_key

        candidate = self._candidate(
            cost=0.5, probabilities=[0.9] * 6, full_vector=[0, 9],
        )
        lex_later = self._candidate(
            cost=0.5, probabilities=[0.9] * 6, full_vector=[1, 0],
        )
        lower_cost = self._candidate(cost=0.4, probabilities=[0.99] * 6)

        self.assertLess(
            strict_selection_key("candidate-z", candidate),
            strict_selection_key("candidate-a", lex_later),
        )
        self.assertLess(
            strict_selection_key("candidate-z", candidate),
            strict_selection_key("candidate-a", lower_cost),
        )

    def test_strict_snapshot_selection_key_matches_live_key_shape(self):
        from blb_stage2_rl.layerwise_runner import (
            strict_selection_key,
            strict_selection_key_from_snapshot,
        )

        restored = self._candidate(
            cost=0.5, probabilities=[0.9] * 6, full_vector=[0, 9],
        )
        restored["candidate_key"] = "candidate-z"
        restored["rank_key"] = [0.0]
        live = self._candidate(
            cost=0.5, probabilities=[0.9] * 6, full_vector=[1, 0],
        )

        restored_key = strict_selection_key_from_snapshot(restored)
        live_key = strict_selection_key("candidate-a", live)

        self.assertEqual(
            restored_key,
            strict_selection_key("candidate-z", restored),
        )
        self.assertLess(restored_key, live_key)

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

    def test_p3_resource_score_is_redistributed_without_changing_episode_return(self):
        from blb_stage2_rl.layerwise_runner import redistribute_layerwise_rewards

        layer_resources = tuple((index + 1) / 780.0 for index in range(12))
        resource_score = sum(layer_resources)
        rewards = redistribute_layerwise_rewards(
            terminal_reward=1.75 + resource_score,
            priority=3,
            ppo_resource_score=resource_score,
            layer_resource_rewards=layer_resources,
        )

        self.assertEqual(len(rewards), 12)
        self.assertEqual(rewards[:-1], layer_resources[:-1])
        self.assertAlmostEqual(rewards[-1], layer_resources[-1] + 1.75)
        self.assertAlmostEqual(sum(rewards), 1.75 + resource_score)

    def test_p1_and_p2_never_receive_cost_credit(self):
        from blb_stage2_rl.layerwise_runner import redistribute_layerwise_rewards

        layer_resources = (0.01,) * 12
        for priority, terminal_reward in ((1, -3.2), (2, -1.7)):
            with self.subTest(priority=priority):
                rewards = redistribute_layerwise_rewards(
                    terminal_reward=terminal_reward,
                    priority=priority,
                    ppo_resource_score=sum(layer_resources),
                    layer_resource_rewards=layer_resources,
                )
                self.assertEqual(rewards[:-1], (0.0,) * 11)
                self.assertEqual(rewards[-1], terminal_reward)

    def test_reward_redistribution_rejects_inconsistent_cost_terms(self):
        from blb_stage2_rl.layerwise_runner import redistribute_layerwise_rewards

        with self.assertRaisesRegex(ValueError, "sum"):
            redistribute_layerwise_rewards(
                terminal_reward=1.5,
                priority=3,
                ppo_resource_score=0.5,
                layer_resource_rewards=(0.01,) * 12,
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
        self.assertIn(
            '"factorized_slot_credit_multifidelity_convergence_v8"',
            branch_source,
        )
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
        self.assertIn(
            '"frontier_stall_update_windows": convergence_patience_updates',
            branch_source,
        )
        self.assertIn(
            '"selected_action_stable_update_windows": convergence_patience_updates',
            branch_source,
        )
        self.assertIn(
            '"feasible,cost,confidence_vector,safety_margin_vector,"',
            branch_source,
        )
        self.assertIn("strict_selection_key(", branch_source)
        self.assertIn("record.promotion_candidate_key", branch_source)
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

        self.assertIn('"stage2_layerwise_robust_run_v3"', branch_source)
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
        self.assertIn('"strict_revalidation_passed"', branch_source)
        self.assertIn('"plateau_ready"', branch_source)
        self.assertIn('"bounded_budget_exhausted"', branch_source)
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


def _strict_reference():
    return types.SimpleNamespace(
        loss_limit=0.40,
        metric1_limit=0.80,
        metric2_limit=0.70,
        loss_std_limit=0.02,
        metric1_std_limit=0.02,
        metric2_std_limit=0.02,
    )


class _FakeLayerwiseEnv:
    horizon = 12
    max_step_dim = 6
    state_dim = 4

    def __init__(self, probabilities=0.7, evidence_mode="valid", invalid=False):
        self._step = 0
        self.actions = []
        self.boosted_overrides = {(4, 3): {"v_mask_rescale_sf": 47}}
        self.base = types.SimpleNamespace(
            statistical_reference=_strict_reference(),
            probe_noise_seed=None,
        )
        self.runtime_terminal_info = None
        self._probabilities = float(probabilities)
        self._evidence_mode = str(evidence_mode)
        self._invalid = bool(invalid)
        self.last_resource_objective = None

    def reset(self, *, seed=None):
        self.seed = seed
        self._step = 0
        self.actions = []
        self.runtime_terminal_info = None
        self.last_resource_objective = None
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
        from blb_stage2_rl.layerwise_action import (
            RESOURCE_SECONDARY_EPSILON,
            dual_resource_score,
            resource_shapley_credits,
        )

        communication_saving = 0.8
        compute_saving = (
            0.4 * (1.0 + RESOURCE_SECONDARY_EPSILON)
            - 0.5 * RESOURCE_SECONDARY_EPSILON * communication_saving
        ) / (1.0 + 0.5 * RESOURCE_SECONDARY_EPSILON)
        robust_floor, secondary_progress, _packed = dual_resource_score(
            compute_saving, communication_saving,
        )
        compute_credit, communication_credit = resource_shapley_credits(
            compute_saving, communication_saving,
        )
        ppo_resource_score = 0.4
        communication_credit += ppo_resource_score - (
            compute_credit + communication_credit
        )
        slot_resource_rewards = []
        for layer_idx in range(12):
            row = [compute_credit / 12.0] + [communication_credit / 59.0] * 5
            if layer_idx == 0:
                row[1] = 0.0
            slot_resource_rewards.append(row)
        layer_resource_rewards = [sum(row) for row in slot_resource_rewards]
        resource_objective = {
            "compute_saving": compute_saving,
            "communication_saving": communication_saving,
            "robust_floor": robust_floor,
            "secondary_progress": secondary_progress,
            "ppo_resource_score": ppo_resource_score,
            "compute_shapley_credit": compute_credit,
            "communication_shapley_credit": communication_credit,
            "fusion_count": 0,
            "removed_k_bits": 0,
            "layer_resource_rewards": layer_resource_rewards,
            "slot_resource_rewards": slot_resource_rewards,
        }
        self.last_resource_objective = resource_objective
        return np.zeros(4, dtype=np.float32), (-5.0 if self._invalid else 7.5), True, {
            "policy_actions": [row[:] for row in self.actions],
            "pending_full_vector": list(range(20)),
            "resource_objective": resource_objective,
            "variable_cost": dict(resource_objective),
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
            convergence_patience_updates=100,
            seed=42,
            online_num_trials_per_step=5,
            promotion_validation_trials=25,
            final_selection_validation_trials=25,
            online_constraint_probability=0.50,
            promotion_constraint_probability=0.80,
            final_constraint_probability=0.95,
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
        resource = env.last_resource_objective
        self.assertIsNotNone(resource)
        for row, expected_local in zip(
                buffer.transitions[:-1], resource["layer_resource_rewards"][:-1],
        ):
            self.assertAlmostEqual(row["reward"], expected_local)
        self.assertAlmostEqual(
            buffer.transitions[-1]["reward"],
            7.5
            - resource["ppo_resource_score"]
            + resource["layer_resource_rewards"][-1],
        )
        self.assertAlmostEqual(
            sum(row["reward"] for row in buffer.transitions), 7.5,
        )
        for row in buffer.transitions:
            self.assertIn("actor_cost_per_slot", row)
            self.assertEqual(
                row["actor_shared_return"],
                7.5 - resource["ppo_resource_score"],
            )
        for row, expected_slots in zip(
                buffer.transitions, resource["slot_resource_rewards"],
        ):
            np.testing.assert_allclose(row["actor_cost_per_slot"], expected_slots)
        self.assertAlmostEqual(
            sum(float(row["actor_cost_per_slot"][0]) for row in buffer.transitions),
            resource["compute_shapley_credit"],
        )
        self.assertAlmostEqual(
            sum(float(row["actor_cost_per_slot"][1:].sum()) for row in buffer.transitions),
            resource["communication_shapley_credit"],
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
            "constraint_safety_margins": [0.1] * 6,
        }
        config = self._train_cfg(total_episodes=0, update_every=1)
        config.final_selection_validation_trials = 31
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

        revalidation_calls = []

        def successful_revalidation(**kwargs):
            if "convergence_revalidation_update" not in kwargs["identity_context"]:
                return types.SimpleNamespace(
                    status="promotion_probability_below_gate",
                    trial_count=0,
                    fresh_trial_count=0,
                    evidence=None,
                    assessment=_assessment(0.7),
                    metrics=None,
                )
            revalidation_calls.append(kwargs)
            trial_count = int(kwargs["target_trial_count"])
            return types.SimpleNamespace(
                status="promoted",
                trial_count=trial_count,
                fresh_trial_count=trial_count,
                evidence=types.SimpleNamespace(
                    promoted=True,
                    trials=types.SimpleNamespace(
                        loss=(0.30,) * trial_count,
                        metric1=(0.90,) * trial_count,
                        metric2=(0.80,) * trial_count,
                        seeds=tuple(range(trial_count)),
                    ),
                ),
                assessment=_assessment(0.99),
                metrics={
                    "loss_mean": 0.30,
                    "loss_std": 0.001,
                    "metric1_mean": 0.90,
                    "metric1_std": 0.001,
                    "metric2_mean": 0.80,
                    "metric2_std": 0.001,
                },
            )

        reference = types.SimpleNamespace(
            loss_limit=0.40,
            metric1_limit=0.80,
            metric2_limit=0.70,
            loss_std_limit=0.01,
            metric1_std_limit=0.01,
            metric2_std_limit=0.01,
        )
        promotion_env = types.SimpleNamespace(statistical_reference=reference)

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.restore_promoted_candidates",
            return_value={"frontier": frontier},
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner.promote_candidate_if_eligible",
            side_effect=successful_revalidation,
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
                promotion_base_env=promotion_env,
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
        self.assertTrue(summary["strict_revalidation_passed"])
        self.assertEqual(summary["stall_update_windows"], 100)
        self.assertEqual(summary["selected_action_identity"], "frontier")
        self.assertEqual(summary["selected_action_stable_update_windows"], 100)
        self.assertEqual(len(revalidation_calls), 1)
        self.assertEqual(revalidation_calls[0]["target_trial_count"], 31)
        self.assertEqual(revalidation_calls[0]["promotion_probability"], 0.95)
        self.assertEqual(revalidation_calls[0]["prefilter_probability"], 0.80)
        self.assertIn(
            "convergence_revalidation_update",
            revalidation_calls[0]["identity_context"],
        )

    def test_failed_strict_revalidation_removes_winner_before_continuing(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        frontier = {
            "variable_cost": 0.4,
            "assessment": _assessment(0.9),
            "metrics": {
                "loss_mean": 0.3,
                "metric1_mean": 0.9,
                "metric2_mean": 0.8,
            },
            "action_matrix": [[1, 0, 1, 2, 3, 4] for _ in range(12)],
            "full_vector": list(range(20)),
            "boosted_overrides": {},
            "reward": 1.4,
            "promotion_trials": None,
            "constraint_safety_margins": [0.1] * 6,
        }
        config = self._train_cfg(total_episodes=0, update_every=1)
        config.convergence_patience_updates = 1
        config.convergence_resume_state = {
            "best_robust_feasible_cost": 0.4,
            "current_robust_feasible_cost": 0.4,
            "stall_update_windows": 0,
            "selected_action_identity": "frontier",
            "selected_action_stable_update_windows": 0,
        }
        failed = types.SimpleNamespace(
            status="failed_probability_gate",
            trial_count=25,
            fresh_trial_count=25,
            evidence=None,
            assessment=_assessment(0.7),
            metrics=None,
        )
        promoted_trials = types.SimpleNamespace(
            loss=(0.30,) * 25,
            metric1=(0.90,) * 25,
            metric2=(0.80,) * 25,
            seeds=tuple(range(25)),
        )
        promoted_evidence = types.SimpleNamespace(
            promoted=True,
            trial_count=25,
            candidate_key="candidate-b",
            trials=promoted_trials,
        )
        promoted_metrics = {
            "loss_mean": 0.30,
            "loss_std": 0.001,
            "metric1_mean": 0.90,
            "metric1_std": 0.001,
            "metric2_mean": 0.80,
            "metric2_std": 0.001,
        }
        online_calls = 0
        revalidation_calls = 0

        def promotion_flow(**kwargs):
            nonlocal online_calls, revalidation_calls
            if "convergence_revalidation_update" in kwargs["identity_context"]:
                revalidation_calls += 1
                if revalidation_calls == 1:
                    return failed
                return types.SimpleNamespace(
                    status="promoted",
                    trial_count=25,
                    fresh_trial_count=25,
                    evidence=promoted_evidence,
                    assessment=_assessment(0.99),
                    metrics=promoted_metrics,
                )
            online_calls += 1
            if online_calls == 1:
                return types.SimpleNamespace(
                    status="promotion_probability_below_gate",
                    trial_count=0,
                    fresh_trial_count=0,
                    evidence=None,
                    assessment=_assessment(0.7),
                    metrics=None,
                )
            return types.SimpleNamespace(
                status=("promoted" if online_calls == 2 else "already_promoted"),
                trial_count=25,
                fresh_trial_count=(25 if online_calls == 2 else 0),
                evidence=promoted_evidence,
                assessment=_assessment(0.99),
                metrics=promoted_metrics,
            )
        promotion_env = types.SimpleNamespace(
            statistical_reference=_strict_reference(),
        )

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.restore_promoted_candidates",
            return_value={"frontier": frontier},
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner.promote_candidate_if_eligible",
            side_effect=promotion_flow,
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner._current_policy_entropy",
            return_value={
                "block4": 0.2,
                "k": 0.6,
                "block4_slot_count": 12,
                "k_slot_count": 59,
            },
        ):
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                promotion_base_env=promotion_env,
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

        self.assertEqual(summary["completed_episodes"], 3)
        self.assertTrue(summary["converged"])
        self.assertEqual(revalidation_calls, 2)
        first = summary["episode_records"][0]
        self.assertFalse(first.converged)
        self.assertFalse(first.strict_revalidation_passed)
        self.assertEqual(first.strict_revalidation_status, "failed_probability_gate")
        self.assertIsNone(first.selected_action_identity)
        self.assertEqual(first.stall_update_windows, 0)
        self.assertEqual(summary["selected_action_identity"], "candidate-b")

    def test_transient_strict_revalidation_error_keeps_winner_and_retries(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        frontier = {
            "variable_cost": 0.4,
            "assessment": _assessment(0.9),
            "metrics": {
                "loss_mean": 0.3,
                "metric1_mean": 0.9,
                "metric2_mean": 0.8,
            },
            "action_matrix": [[1, 0, 1, 2, 3, 4] for _ in range(12)],
            "full_vector": list(range(20)),
            "boosted_overrides": {},
            "reward": 1.4,
            "promotion_trials": None,
            "constraint_safety_margins": [0.1] * 6,
        }
        config = self._train_cfg(total_episodes=0, update_every=1)
        config.convergence_resume_state = {
            "best_robust_feasible_cost": 0.4,
            "current_robust_feasible_cost": 0.4,
            "stall_update_windows": 99,
            "selected_action_identity": "frontier",
            "selected_action_stable_update_windows": 99,
        }
        revalidation_calls = 0
        online_calls = 0

        def promoted_result(trial_count):
            evidence = types.SimpleNamespace(
                promoted=True,
                trial_count=trial_count,
                candidate_key="frontier",
                trials=types.SimpleNamespace(
                    loss=(0.30,) * trial_count,
                    metric1=(0.90,) * trial_count,
                    metric2=(0.80,) * trial_count,
                    seeds=tuple(range(trial_count)),
                ),
            )
            return types.SimpleNamespace(
                status="promoted",
                trial_count=trial_count,
                fresh_trial_count=trial_count,
                evidence=evidence,
                assessment=_assessment(0.99),
                metrics={
                    "loss_mean": 0.30,
                    "loss_std": 0.001,
                    "metric1_mean": 0.90,
                    "metric1_std": 0.001,
                    "metric2_mean": 0.80,
                    "metric2_std": 0.001,
                },
            )

        def promotion_flow(**kwargs):
            nonlocal online_calls, revalidation_calls
            if "convergence_revalidation_update" not in kwargs["identity_context"]:
                online_calls += 1
                if online_calls > 1:
                    return promoted_result(25)
                return types.SimpleNamespace(
                    status="promotion_probability_below_gate",
                    trial_count=0,
                    fresh_trial_count=0,
                    evidence=None,
                    assessment=_assessment(0.7),
                    metrics=None,
                )
            revalidation_calls += 1
            if revalidation_calls == 1:
                return types.SimpleNamespace(
                    status="failed_evaluation",
                    trial_count=0,
                    fresh_trial_count=0,
                    evidence=None,
                    assessment=_assessment(0.9),
                    metrics=None,
                )
            return promoted_result(int(kwargs["target_trial_count"]))

        promotion_env = types.SimpleNamespace(
            statistical_reference=_strict_reference(),
        )
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.restore_promoted_candidates",
            return_value={"frontier": frontier},
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner.promote_candidate_if_eligible",
            side_effect=promotion_flow,
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner._current_policy_entropy",
            return_value={
                "block4": 0.2,
                "k": 0.6,
                "block4_slot_count": 12,
                "k_slot_count": 59,
            },
        ):
            store_path = Path(td) / "candidates.jsonl"
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.7),
                promotion_base_env=promotion_env,
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=CandidateStore(store_path),
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
            persisted = store_path.read_text(encoding="utf-8")

        self.assertEqual(summary["completed_episodes"], 2)
        self.assertEqual(revalidation_calls, 2)
        self.assertTrue(summary["converged"])
        first = summary["episode_records"][0]
        self.assertEqual(first.strict_revalidation_status, "failed_evaluation")
        self.assertEqual(first.selected_action_identity, "frontier")
        self.assertEqual(first.stall_update_windows, 100)
        self.assertNotIn("final_revalidation_failed", persisted)

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
        self.assertEqual(summary["termination_reason"], "bounded_budget_exhausted")
        self.assertEqual(
            summary["episode_records"][-1].termination_reason,
            "bounded_budget_exhausted",
        )

    def test_bounded_smoke_never_runs_natural_revalidation(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        frontier = {
            "variable_cost": 0.4,
            "assessment": _assessment(0.9),
            "metrics": {
                "loss_mean": 0.3,
                "metric1_mean": 0.9,
                "metric2_mean": 0.8,
            },
            "action_matrix": [[1, 0, 1, 2, 3, 4] for _ in range(12)],
            "full_vector": list(range(20)),
            "boosted_overrides": {},
            "reward": 1.4,
            "promotion_trials": None,
            "constraint_safety_margins": [0.1] * 6,
        }
        config = self._train_cfg(total_episodes=1, update_every=1)
        config.convergence_patience_updates = 1
        config.convergence_resume_state = {
            "best_robust_feasible_cost": 0.4,
            "current_robust_feasible_cost": 0.4,
            "stall_update_windows": 1,
            "selected_action_identity": "frontier",
            "selected_action_stable_update_windows": 1,
        }
        online_result = types.SimpleNamespace(
            status="promotion_probability_below_gate",
            trial_count=0,
            fresh_trial_count=0,
            evidence=None,
            assessment=_assessment(0.7),
            metrics=None,
        )

        def promotion_only(**kwargs):
            self.assertNotIn("convergence_revalidation_update", kwargs["identity_context"])
            return online_result

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.restore_promoted_candidates",
            return_value={"frontier": frontier},
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner.promote_candidate_if_eligible",
            side_effect=promotion_only,
        ):
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

        self.assertFalse(summary["converged"])
        self.assertTrue(summary["plateau_ready"])
        self.assertFalse(summary["strict_revalidation_passed"])
        self.assertEqual(summary["termination_reason"], "bounded_budget_exhausted")

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
        self.assertEqual(summary["termination_reason"], "bounded_budget_exhausted")

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
            "constraint_safety_margins": [0.1] * 6,
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
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            train_layerwise,
        )

        identity_context = {"action_space_version": "layerwise-v1"}
        probe_context = evidence_identity_context(identity_context, "F1")
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
                list(range(20)), probe_context,
            )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.trial_count, 10)
        self.assertEqual(len(set(evidence.trials.seeds)), 10)

    def test_episode_record_labels_fresh_reward_and_f1_prefilter_evidence(self):
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
        self.assertEqual(second.ranking_evidence, "F1_prefilter_only")

    def test_repeated_action_keeps_bootstrap_assessment_at_fixed_25_trial_budget(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            train_layerwise,
        )

        assessed_trial_counts = []

        def assess(trials, *_args, **_kwargs):
            assessed_trial_counts.append(len(trials.loss))
            return _assessment(0.7)

        identity_context = {"action_space_version": "layerwise-v1"}
        probe_context = evidence_identity_context(identity_context, "F1")
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
            raw = store.trial_evidence_for_action(list(range(20)), probe_context)

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

    def test_zero_remaining_bounded_resume_preserves_frontier_without_convergence(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key
        from blb_stage2_rl.layerwise_action import compute_variable_cost_from_action_matrix
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            train_layerwise,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
        action = list(range(20))
        expected_identity = candidate_key(action, full_context)
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
                    "identity_context": full_context,
                    "fidelity": "F4",
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
                "identity_context": full_context,
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
                "strict_revalidation_passed": True,
                "strict_revalidation_status": "passed",
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
        self.assertFalse(summary["converged"])
        self.assertEqual(summary["strict_best"]["full_vector"], action)
        self.assertFalse(summary["convergence_state"]["converged"])
        self.assertEqual(summary["termination_reason"], "bounded_budget_exhausted")

    def test_promoted_reward_and_install_metadata_restore_from_promotion_record(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            restore_promoted_candidates,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
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
                    "identity_context": full_context,
                    "fidelity": "F4",
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
                "identity_context": full_context,
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
                    "identity_context": full_context,
                    "fidelity": "F4",
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
                    statistical_reference=_strict_reference(),
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
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            train_layerwise,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
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
                    "identity_context": full_context,
                    "fidelity": "F4",
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
                "identity_context": full_context,
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
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            train_layerwise,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
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
                    "identity_context": full_context,
                    "fidelity": "F4",
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
                "identity_context": full_context,
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
        self.statistical_reference = _strict_reference()
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
        from blb_stage2_rl.layerwise_runner import evidence_identity_context
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = evidence_identity_context(
            {"action_space_version": "layerwise-v1"}, "F4",
        )
        store = CandidateStore(Path(root) / "candidates.jsonl")
        store.append_trial_group(
            list(range(20)),
            TrialSeries(
                loss=[0.3] * 5,
                metric1=[0.9] * 5,
                metric2=[0.8] * 5,
                seeds=[1, 2, 3, 4, 5] if seeds is None else seeds,
            ),
            {"identity_context": context, "fidelity": "F4"},
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

    def test_already_promoted_candidate_uses_f4_evidence_not_new_f1_result(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            base = _PromotionBase()
            common = dict(
                env=types.SimpleNamespace(base=base),
                candidate_store=store,
                action_indices=list(range(20)),
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=[[0] * 6 for _ in range(12)],
                variable_cost=0.6,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
            )
            promoted = promote_candidate_if_eligible(
                **common,
                assessment=_assessment(0.9),
                priority=3,
                frontier_cost=0.5,
            )
            repeated = promote_candidate_if_eligible(
                **common,
                assessment=_assessment(0.1),
                priority=1,
                frontier_cost=0.6,
            )

        self.assertEqual(promoted.status, "promoted")
        self.assertEqual(repeated.status, "already_promoted")
        self.assertEqual(repeated.trial_count, 25)
        self.assertEqual(
            repeated.assessment.loss_precision_probability,
            0.9,
        )
        self.assertEqual(len(base.evaluate_calls), 1)

    def test_failed_f4_promotion_status_is_not_mislabeled_f1(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=_PromotionBase()),
                candidate_store=store,
                action_indices=list(range(20)),
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.9),
                priority=3,
                variable_cost=0.6,
                frontier_cost=0.5,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.79),
            )
            status_rows = [
                row for row in store.iter_active_records()
                if row.get("record_type") == "candidate_promotion_status_v1"
            ]

        self.assertEqual(result.status, "failed_probability_gate")
        self.assertEqual(status_rows[-1]["fidelity"], "F4")
        self.assertEqual(status_rows[-1]["identity_context"]["fidelity"], "F4")

    def test_promotion_collects_25_full_trials_without_pooling_five_probe_trials(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            promote_candidate_if_eligible,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1", "fidelity": None}
        action = list(range(20))
        probe_context = evidence_identity_context(context, "F1")
        full_context = evidence_identity_context(context, "F4")
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3] * 5,
                    metric1=[0.9] * 5,
                    metric2=[0.8] * 5,
                    seeds=[1, 2, 3, 4, 5],
                ),
                {"identity_context": probe_context, "fidelity": "F1"},
            )
            online_base = _PromotionBase()
            full_base = _PromotionBase()
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=online_base),
                promotion_base_env=full_base,
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
            probe_evidence = store.trial_evidence_for_action(action, probe_context)
            full_evidence = store.trial_evidence_for_action(action, full_context)

        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.trial_count, 25)
        self.assertEqual(result.fresh_trial_count, 25)
        self.assertEqual(probe_evidence.trial_count, 5)
        self.assertEqual(full_evidence.trial_count, 25)
        self.assertEqual(online_base.evaluate_calls, [])
        self.assertEqual(full_base.evaluate_calls[0][1]["num_trials_per_action"], 25)
        self.assertTrue(full_base.evaluate_calls[0][1]["validation_required"])

    def test_promotion_assesses_existing_evidence_above_target_without_new_probe(self):
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            promote_candidate_if_eligible,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
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
                    {
                        "identity_context": full_context,
                        "fidelity": "F4",
                        "group_index": group_idx,
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
        self.assertEqual(result.trial_count, 30)
        self.assertEqual(result.fresh_trial_count, 0)
        self.assertEqual(base.prepare_calls, [])
        self.assertEqual(base.evaluate_calls, [])

    def test_promotion_recovers_pending_reassessment_after_top_up_crash(self):
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            promote_candidate_if_eligible,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
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
                    "identity_context": full_context,
                    "fidelity": "F4",
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
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            restore_promoted_candidates,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
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
                    "identity_context": full_context,
                    "fidelity": "F4",
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
                "identity_context": full_context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })

            restored = restore_promoted_candidates(
                candidate_store=store,
                identity_context=context,
                statistical_reference=_strict_reference(),
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

    def test_final_revalidation_pass_restores_fresh_strict_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            _record_final_revalidation_outcome,
            evidence_identity_context,
            restore_promoted_candidates,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        base_context = evidence_identity_context(context, "F4")
        revalidation_context = {
            **context,
            "convergence_revalidation_update": 12_000,
            "convergence_revalidation_candidate": "candidate-a",
        }
        strict_context = evidence_identity_context(revalidation_context, "F4")
        action = list(range(20))
        matrix = [[0] * 6 for _layer in range(12)]
        candidate = {
            "candidate_key": "candidate-a",
            "variable_cost": 0.0,
            "action_matrix": matrix,
            "full_vector": action,
            "boosted_overrides": {},
            "reward": 1.25,
        }
        observed_assessments = []

        def assess(trials, _reference, *, gate_probability, bootstrap_seed):
            observed_assessments.append(
                (len(trials.loss), float(gate_probability), int(bootstrap_seed))
            )
            return _assessment(0.99)

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
                    "identity_context": base_context,
                    "fidelity": "F4",
                    "action_matrix": matrix,
                    "variable_cost": 0.0,
                    "boosted_overrides": [],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": base_context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.29] * 31,
                    metric1=[0.91] * 31,
                    metric2=[0.81] * 31,
                    seeds=list(range(100, 131)),
                ),
                {
                    "identity_context": strict_context,
                    "fidelity": "F4",
                    "action_matrix": matrix,
                    "variable_cost": 0.0,
                    "boosted_overrides": [],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": strict_context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })
            _record_final_revalidation_outcome(
                candidate_store=store,
                identity_context=context,
                revalidation_identity_context=revalidation_context,
                candidate=candidate,
                passed=True,
                revalidation_status="promoted",
                bootstrap_seed=123,
                final_probability=0.95,
                final_trial_count=31,
            )

            restored = restore_promoted_candidates(
                candidate_store=store,
                identity_context=context,
                statistical_reference=_strict_reference(),
                assess_candidate_fn=assess,
                promotion_probability=0.80,
                assessment_trial_limit=25,
                final_probability=0.95,
                final_assessment_trial_limit=31,
            )

        self.assertEqual(len(restored), 1)
        restored_candidate = next(iter(restored.values()))
        self.assertEqual(len(restored_candidate["promotion_trials"].loss), 31)
        self.assertAlmostEqual(restored_candidate["metrics"]["loss_mean"], 0.29)
        self.assertEqual(restored_candidate["final_revalidation_status"], "passed")
        self.assertEqual(observed_assessments, [(31, 0.95, 123)])

    def test_final_revalidation_failure_remains_excluded_after_restore(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            _record_final_revalidation_outcome,
            evidence_identity_context,
            promote_candidate_if_eligible,
            restore_promoted_candidates,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        context = {"action_space_version": "layerwise-v1"}
        base_context = evidence_identity_context(context, "F4")
        revalidation_context = {
            **context,
            "convergence_revalidation_update": 12_000,
            "convergence_revalidation_candidate": "candidate-a",
        }
        action = list(range(20))
        matrix = [[0] * 6 for _layer in range(12)]
        candidate = {
            "candidate_key": "candidate-a",
            "variable_cost": 0.0,
            "action_matrix": matrix,
            "full_vector": action,
            "boosted_overrides": {},
            "reward": 1.25,
        }

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
                    "identity_context": base_context,
                    "fidelity": "F4",
                    "action_matrix": matrix,
                    "variable_cost": 0.0,
                    "boosted_overrides": [],
                },
            )
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": base_context,
                "promotion_status": "promoted",
                "fidelity": "F4",
                "valid": True,
            })
            _record_final_revalidation_outcome(
                candidate_store=store,
                identity_context=context,
                revalidation_identity_context=revalidation_context,
                candidate=candidate,
                passed=False,
                revalidation_status="failed_probability_gate",
                bootstrap_seed=123,
                final_probability=0.95,
                final_trial_count=31,
            )

            restored = restore_promoted_candidates(
                candidate_store=store,
                identity_context=context,
                statistical_reference=_strict_reference(),
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                final_assessment_trial_limit=31,
            )
            base = _PromotionBase()
            retry = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=matrix,
                assessment=_assessment(0.99),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
            )

        self.assertEqual(restored, {})
        self.assertEqual(retry.status, "promotion_already_attempted")
        self.assertEqual(base.prepare_calls, [])
        self.assertEqual(base.evaluate_calls, [])

    def test_promotion_selects_fresh_trial_seeds_disjoint_from_existing_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            promote_candidate_if_eligible,
        )
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

        action = list(range(20))
        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
        key = candidate_key(action, full_context)
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
            evidence = store.trial_evidence_for_action(action, full_context)

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

    def test_promotion_allows_equal_cost_candidates_for_deterministic_tie_ranking(self):
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            base = _PromotionBase()
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                candidate_store=store,
                action_indices=list(range(20)),
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.9),
                priority=3,
                variable_cost=0.5,
                frontier_cost=0.5,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.9),
            )

        self.assertEqual(result.status, "promoted")
        self.assertEqual(len(base.evaluate_calls), 1)

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
