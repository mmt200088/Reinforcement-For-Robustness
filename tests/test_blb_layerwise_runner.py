from __future__ import annotations

import ast
from dataclasses import asdict
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
            compute_saving=None, communication_saving=None,
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
        compute = float(cost if compute_saving is None else compute_saving)
        communication = float(
            cost if communication_saving is None else communication_saving
        )
        return {
            "variable_cost": cost,
            "compute_saving": compute,
            "communication_saving": communication,
            "robust_floor": min(compute, communication),
            "secondary_progress": 0.5 * (compute + communication),
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

    def test_strict_rank_and_pareto_use_both_resource_axes(self):
        from blb_stage2_rl.layerwise_runner import (
            strict_rank_key,
            strict_resource_pareto_frontier,
        )

        common = {"cost": 0.99, "probabilities": [0.9] * 6}
        candidates = {
            "a": self._candidate(
                **common, compute_saving=0.2, communication_saving=0.9,
                full_vector=[2],
            ),
            "b": self._candidate(
                **common, compute_saving=0.3, communication_saving=0.3,
                full_vector=[1],
            ),
            "c": self._candidate(
                **common, compute_saving=0.3, communication_saving=0.5,
                full_vector=[0],
            ),
        }

        self.assertLess(strict_rank_key(candidates["c"]), strict_rank_key(candidates["b"]))
        self.assertLess(strict_rank_key(candidates["b"]), strict_rank_key(candidates["a"]))
        frontier = strict_resource_pareto_frontier(candidates)
        self.assertEqual(list(frontier), ["c", "a"])

    def test_protected_k1_preserves_every_potential_resource_frontier_point(self):
        from blb_stage2_rl.layerwise_runner import (
            _can_expand_resource_frontier,
        )

        common = {"cost": 0.5, "probabilities": [0.9] * 6}
        accepted = {
            "compute-heavy": self._candidate(
                **common,
                compute_saving=0.8,
                communication_saving=0.3,
                full_vector=[0],
            ),
            "balanced": self._candidate(
                **common,
                compute_saving=0.5,
                communication_saving=0.5,
                full_vector=[1],
            ),
        }
        self.assertFalse(_can_expand_resource_frontier(
            {"compute_saving": 0.4, "communication_saving": 0.4},
            accepted,
        ))
        self.assertTrue(_can_expand_resource_frontier(
            {"compute_saving": 0.4, "communication_saving": 0.6},
            accepted,
        ))
        self.assertTrue(_can_expand_resource_frontier(
            {"compute_saving": 0.5, "communication_saving": 0.5},
            accepted,
        ))

    def test_protected_k1_direct_k5_bypass_is_explicitly_annotated(self):
        from blb_stage2_rl.layerwise_runner import (
            _annotate_protected_k1_exact_results,
        )

        info = {
            "statistical_trials": {
                "loss": [0.3] * 5,
                "metric1": [0.9] * 5,
                "metric2": [0.8] * 5,
            },
            "probe_diagnostics": {},
        }
        _annotate_protected_k1_exact_results(
            [(object(), 0.0, True, info)],
            guard_sigma=4.0,
            reason="protected_by_resource_frontier",
        )
        self.assertTrue(info["protected_k1_enabled"])
        self.assertFalse(info["protected_k1_screened"])
        self.assertEqual(info["protected_k1_trials_executed"], 5)
        self.assertTrue(info["protected_k1_stability_measured"])
        self.assertEqual(
            info["protected_k1_reason"],
            "protected_by_resource_frontier",
        )
        self.assertEqual(
            info["probe_diagnostics"]["protected_k1_trials_executed"],
            5,
        )

    def test_protected_k1_fail_open_and_counters_restore_from_checkpoint(self):
        from blb_stage2_rl.layerwise_runner import (
            _restore_protected_k1_runtime_state,
        )

        counters, fail_open_episode = _restore_protected_k1_runtime_state({
            "protected_k1": {
                "enabled_episodes": 120,
                "screened_episodes": 3,
                "k1_only_rejects": 2,
                "audited_episodes": 1,
                "audit_precision_false_negatives": 1,
                "audit_p3_false_negatives": 1,
                "trials_executed": 588,
                "trials_saved_vs_k5": 8,
                "fail_open_episode": 119,
            },
        })
        self.assertEqual(counters["enabled_episodes"], 120)
        self.assertEqual(counters["trials_saved_vs_k5"], 8)
        self.assertEqual(fail_open_episode, 119)
        with self.assertRaises(ValueError):
            _restore_protected_k1_runtime_state({
                "protected_k1": {"trials_saved_vs_k5": -1},
            })

    def test_identical_online_evidence_retargets_existing_assessment(self):
        from blb_stage2_rl.layerwise_runner import (
            _assess_pooled_online_trials,
        )
        from blb_stage2_rl.statistical_constraints import (
            ConstraintAssessment,
            TrialSeries,
            assess_candidate,
        )

        trials = TrialSeries(
            loss=[0.30, 0.31, 0.29, 0.305, 0.295],
            metric1=[0.90, 0.89, 0.91, 0.895, 0.905],
            metric2=[0.80, 0.79, 0.81, 0.795, 0.805],
            seeds=[11, 12, 13, 14, 15],
        )
        fresh = ConstraintAssessment(
            loss_precision_probability=0.91,
            metric1_precision_probability=0.92,
            metric2_precision_probability=0.93,
            loss_stability_probability=0.81,
            metric1_stability_probability=0.82,
            metric2_stability_probability=0.83,
            precision_probability=0.91,
            stability_probability=0.81,
            gate_probability=0.50,
            online_precision_pass=True,
            online_stability_pass=True,
        )

        result = _assess_pooled_online_trials(
            raw_trials=trials,
            pooled_trials=trials,
            fresh_assessment={**asdict(fresh), "bootstrap_seed": 77},
            reference=object(),
            gate_probability=0.80,
            bootstrap_seed=77,
            assess_candidate_fn=assess_candidate,
        )

        self.assertEqual(result.precision_probability, 0.91)
        self.assertEqual(result.stability_probability, 0.81)
        self.assertEqual(result.gate_probability, 0.80)
        self.assertTrue(result.online_precision_pass)
        self.assertTrue(result.online_stability_pass)

    def test_convergence_resets_on_exact_resource_objective_improvement(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker(patience_updates=2)
        tracker.observe_update(
            completed_episodes=120,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_objective=(0.2, 0.55),
            robust_feasible_action_identity="a",
        )
        tracker.observe_update(
            completed_episodes=240,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_objective=(0.2, 0.55),
            robust_feasible_action_identity="a",
        )
        improved = tracker.observe_update(
            completed_episodes=360,
            block4_entropy=0.9,
            k_entropy=0.9,
            robust_feasible_objective=(0.3, 0.3),
            robust_feasible_action_identity="b",
        )

        self.assertEqual(improved.stall_update_windows, 0)
        self.assertEqual(improved.selected_action_stable_update_windows, 0)
        self.assertEqual(improved.best_robust_feasible_objective, (0.3, 0.3))

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

    def test_point_constraint_gate_checks_all_six_metrics_without_probability(self):
        import blb_stage2_rl.layerwise_runner as layerwise_runner

        self.assertTrue(hasattr(layerwise_runner, "point_constraints_pass"))
        if not hasattr(layerwise_runner, "point_constraints_pass"):
            return
        reference = _strict_reference()
        passing = {
            "loss_mean": reference.loss_limit,
            "metric1_mean": reference.metric1_limit,
            "metric2_mean": reference.metric2_limit,
            "loss_std": reference.loss_std_limit,
            "metric1_std": reference.metric1_std_limit,
            "metric2_std": reference.metric2_std_limit,
        }
        self.assertTrue(layerwise_runner.point_constraints_pass(passing, reference))

        violating_values = {
            "loss_mean": reference.loss_limit + 1.0e-3,
            "metric1_mean": reference.metric1_limit - 1.0e-3,
            "metric2_mean": reference.metric2_limit - 1.0e-3,
            "loss_std": reference.loss_std_limit + 1.0e-3,
            "metric1_std": reference.metric1_std_limit + 1.0e-3,
            "metric2_std": reference.metric2_std_limit + 1.0e-3,
        }
        for metric_name, violating_value in violating_values.items():
            with self.subTest(metric_name=metric_name):
                self.assertFalse(layerwise_runner.point_constraints_pass(
                    dict(passing, **{metric_name: violating_value}), reference,
                ))

    def test_three_validation_banks_are_disjoint_and_pool_in_order(self):
        import blb_stage2_rl.layerwise_runner as layerwise_runner
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed
        from blb_stage2_rl.statistical_constraints import TrialSeries

        required = ("LayerwiseValidationBank", "LayerwiseValidationBanks")
        for name in required:
            self.assertTrue(hasattr(layerwise_runner, name))
        if not all(hasattr(layerwise_runner, name) for name in required):
            return

        def reference(probe_seeds):
            seeds = tuple(
                derive_probe_trial_seed(probe_seed, trial_idx)
                for probe_seed in probe_seeds
                for trial_idx in range(5)
            )
            return types.SimpleNamespace(
                **vars(_strict_reference()),
                trials=TrialSeries(
                    loss=[0.30] * len(seeds),
                    metric1=[0.90] * len(seeds),
                    metric2=[0.80] * len(seeds),
                    seeds=seeds,
                ),
            )

        bank_a = layerwise_runner.LayerwiseValidationBank(
            label="A", reference=reference((101, 102, 103, 104, 105)),
            probe_seeds=(101, 102, 103, 104, 105), trials_per_probe=5,
        )
        bank_b = layerwise_runner.LayerwiseValidationBank(
            label="B", reference=reference((201, 202, 203, 204, 205)),
            probe_seeds=(201, 202, 203, 204, 205), trials_per_probe=5,
        )
        bank_c = layerwise_runner.LayerwiseValidationBank(
            label="C", reference=reference((301, 302, 303, 304, 305)),
            probe_seeds=(301, 302, 303, 304, 305), trials_per_probe=5,
        )
        pooled_ab = reference(bank_a.probe_seeds + bank_b.probe_seeds)
        pooled_abc = reference(
            bank_a.probe_seeds + bank_b.probe_seeds + bank_c.probe_seeds
        )
        banks = layerwise_runner.LayerwiseValidationBanks(
            bank_a=bank_a,
            bank_b=bank_b,
            bank_c=bank_c,
            promotion_reference=pooled_ab,
            final_reference=pooled_abc,
        )

        self.assertEqual(banks.bank_a.trial_count, 25)
        self.assertEqual(banks.promotion_trial_count, 50)
        self.assertEqual(banks.final_trial_count, 75)
        self.assertEqual(
            tuple(banks.promotion_reference.trials.seeds),
            bank_a.trial_seeds + bank_b.trial_seeds,
        )
        self.assertEqual(
            tuple(banks.final_reference.trials.seeds),
            bank_a.trial_seeds + bank_b.trial_seeds + bank_c.trial_seeds,
        )

        mismatched_ab = reference(bank_a.probe_seeds + bank_b.probe_seeds)
        mismatched_ab.trials = TrialSeries(
            loss=(0.31,) + mismatched_ab.trials.loss[1:],
            metric1=mismatched_ab.trials.metric1,
            metric2=mismatched_ab.trials.metric2,
            seeds=mismatched_ab.trials.seeds,
        )
        with self.assertRaisesRegex(ValueError, "exact Bank A then Bank B trials"):
            layerwise_runner.LayerwiseValidationBanks(
                bank_a=bank_a,
                bank_b=bank_b,
                bank_c=bank_c,
                promotion_reference=mismatched_ab,
                final_reference=pooled_abc,
            )

        short_bank_a = layerwise_runner.LayerwiseValidationBank(
            label="A", reference=reference((101, 102, 103, 104)),
            probe_seeds=(101, 102, 103, 104), trials_per_probe=5,
        )
        with self.assertRaisesRegex(ValueError, "exactly 25 trials"):
            layerwise_runner.LayerwiseValidationBanks(
                bank_a=short_bank_a,
                bank_b=bank_b,
                bank_c=bank_c,
                promotion_reference=reference(
                    short_bank_a.probe_seeds + bank_b.probe_seeds
                ),
                final_reference=reference(
                    short_bank_a.probe_seeds
                    + bank_b.probe_seeds
                    + bank_c.probe_seeds
                ),
            )

    def test_layerwise_validation_bank_config_is_fixed_to_25_25_25(self):
        from blb_stage2_rl.layerwise_runner import (
            validate_layerwise_validation_bank_config,
        )

        valid = types.SimpleNamespace(
            baseline_groups=5,
            baseline_trials_per_group=5,
            promotion_validation_trials=25,
            final_selection_validation_trials=25,
        )
        self.assertEqual(
            validate_layerwise_validation_bank_config(valid), (5, 5),
        )

        for field, value in (
            ("baseline_groups", 4),
            ("baseline_trials_per_group", 4),
            ("promotion_validation_trials", 20),
            ("final_selection_validation_trials", 20),
        ):
            invalid = types.SimpleNamespace(**vars(valid))
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "A=25, B=25, C=25",
            ):
                validate_layerwise_validation_bank_config(invalid)

    def test_three_bank_convergence_contract_cannot_stop_before_90k_or_100_updates(self):
        from blb_stage2_rl.layerwise_runner import (
            validate_layerwise_three_bank_convergence_config,
        )

        valid = types.SimpleNamespace(
            convergence_min_episodes=90_000,
            convergence_patience_updates=100,
        )
        self.assertEqual(
            validate_layerwise_three_bank_convergence_config(valid),
            (90_000, 100),
        )
        conservative = types.SimpleNamespace(
            convergence_min_episodes=100_000,
            convergence_patience_updates=120,
        )
        self.assertEqual(
            validate_layerwise_three_bank_convergence_config(conservative),
            (100_000, 120),
        )
        for field, value in (
            ("convergence_min_episodes", 89_999),
            ("convergence_patience_updates", 99),
        ):
            invalid = types.SimpleNamespace(**vars(valid))
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "three-bank convergence",
            ):
                validate_layerwise_three_bank_convergence_config(invalid)

    def test_episode_limit_resume_allows_only_equal_or_larger_caps(self):
        from blb_stage2_rl.layerwise_runner import (
            validate_layerwise_episode_limit_extension,
        )

        self.assertEqual(
            validate_layerwise_episode_limit_extension(150_000, 150_000),
            150_000,
        )
        self.assertEqual(
            validate_layerwise_episode_limit_extension(150_000, 200_000),
            200_000,
        )
        self.assertEqual(
            validate_layerwise_episode_limit_extension(150_000, 0), 0,
        )
        with self.assertRaisesRegex(RuntimeError, "cannot shrink"):
            validate_layerwise_episode_limit_extension(150_000, 149_999)
        with self.assertRaisesRegex(RuntimeError, "cannot become bounded"):
            validate_layerwise_episode_limit_extension(0, 200_000)

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

    def test_layerwise_compact_candidate_store_wiring_preserves_f1_and_f4_payload_rules(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "layerwise_runner.py"
        )
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        def function_source(name):
            node = next(
                candidate
                for candidate in ast.walk(tree)
                if isinstance(candidate, ast.FunctionDef)
                and candidate.name == name
            )
            return ast.get_source_segment(source, node)

        def trial_append_source(function_name, fidelity=None):
            function = function_source(function_name)
            return next(
                ast.get_source_segment(function, node)
                for node in ast.walk(ast.parse(function))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append_trial_group"
                and (
                    fidelity is None
                    or f'"fidelity": "{fidelity}"' in ast.get_source_segment(
                        function, node,
                    )
                )
            )

        f1_call = trial_append_source("train_layerwise", "F1")
        promotion_f4_call = trial_append_source(
            "promote_candidate_if_eligible", "F4",
        )
        bank_f4_call = trial_append_source("_collect_fixed_validation_bank")

        self.assertIn("compact=True", f1_call)
        self.assertIn("boosted_overrides_hash", f1_call)
        self.assertIn("boosted_overrides_provenance", f1_call)
        self.assertNotIn('"boosted_overrides":', f1_call)
        self.assertIn("compact=True", promotion_f4_call)
        self.assertIn('"boosted_overrides":', promotion_f4_call)
        self.assertIn("compact=True", bank_f4_call)
        self.assertIn(
            '"boosted_overrides":',
            function_source("_collect_fixed_validation_bank"),
        )
        self.assertIn(
            '"boosted_overrides": _serialize_boosted_overrides(',
            function_source("_record_final_revalidation_outcome"),
        )

    def test_compact_promotion_status_helper_delegates_to_candidate_store_api(self):
        from blb_stage2_rl.layerwise_runner import _append_promotion_status

        append = mock.Mock()
        append_promotion_status = mock.Mock()
        store = types.SimpleNamespace(
            append=append,
            append_promotion_status=append_promotion_status,
        )
        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1", "fidelity": "F4"}
        metadata = {"boosted_overrides": [{"block_idx": 4, "layer_idx": 3}]}

        _append_promotion_status(
            store,
            action,
            context,
            status="promoted",
            metadata=metadata,
        )

        append.assert_not_called()
        append_promotion_status.assert_called_once_with(
            action,
            context,
            status="promoted",
            metadata=metadata,
        )

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

    def test_resumed_best_reward_uses_historical_diagnostics_maximum(self):
        from blb_stage2_rl.sequential_runner import resolve_resumed_best_reward

        cases = (
            ("historical maximum", {"reward": 4.0}, 9.5, 9.5),
            ("strict maximum", {"reward": 9.5}, 4.0, 9.5),
            ("strict only", {"reward": 4.0}, None, 4.0),
            ("no history", {}, None, -math.inf),
            ("malformed strict", {"reward": "not-a-number"}, None, -math.inf),
            ("nonfinite strict nan", {"reward": float("nan")}, None, -math.inf),
            ("nonfinite strict inf", {"reward": float("inf")}, None, -math.inf),
            ("history survives malformed strict", {"reward": object()}, 9.5, 9.5),
            ("nonfinite history ignored", {"reward": 4.0}, float("nan"), 4.0),
        )
        for label, resumed_best, historical_best, expected in cases:
            with self.subTest(label=label):
                actual = resolve_resumed_best_reward(resumed_best, historical_best)
                if expected == -math.inf:
                    self.assertTrue(math.isinf(actual))
                    self.assertLess(actual, 0.0)
                else:
                    self.assertEqual(actual, expected)

    def test_layerwise_branch_wires_restored_diagnostics_into_resumed_best_reward(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "sequential_runner.py"
        )
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        initializers = [
            node.value
            for node in ast.walk(branch)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "best_reward_so_far"
                for target in node.targets
            )
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "resolve_resumed_best_reward"
        ]

        self.assertEqual(len(initializers), 1)
        self.assertEqual(
            ast.unparse(initializers[0]),
            "resolve_resumed_best_reward(resumed_best, "
            "diag_recorder.best_episode_return)",
        )

    def test_layerwise_stage2_builds_one_shared_probe_owner(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "sequential_runner.py"
        )
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        build_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "build_probe_runner"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "build_probe_runner"
                )
            )
        ]
        view_keys = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "view"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }
        attribute_names = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        }

        self.assertEqual(len(build_calls), 1)
        self.assertIn("_shared_probe_runner_owner", attribute_names)
        self.assertTrue({"F1", "F4"}.issubset(view_keys))

    def test_authoritative_validation_reuses_shared_five_device_probe_pool(self):
        from blb_stage2_rl.sequential_runner import (
            _build_authoritative_validation_env,
        )

        class _ProbeView:
            def __init__(self, owner, batch_set_key):
                self.owner = owner
                self.batch_set_key = str(batch_set_key)

            @property
            def pool_id(self):
                return self.owner.pool_id

        class _SharedProbeOwner:
            def __init__(self):
                self.pool_id = "five-device-pool"
                self.devices = [f"cuda:{device_id}" for device_id in range(5)]
                self._process_workers = [object() for _ in range(4)]
                self.registrations = []

            @property
            def num_workers(self):
                return 1 + len(self._process_workers)

            def register_batch_set(self, key, batches):
                self.registrations.append((str(key), tuple(batches)))

            def view(self, batch_set_key):
                return _ProbeView(self, batch_set_key)

        validation_batches = ["full-batch-0", "full-batch-1"]
        validation_examples = list(range(5))
        owner = _SharedProbeOwner()
        base_env = types.SimpleNamespace(
            env_cfg=types.SimpleNamespace(
                probe_batch_count=1,
                persistent_probe_install=True,
            ),
            probe_batches=["online-batch"],
            probe_runner=owner.view("F1"),
            _shared_probe_runner_owner=owner,
            baseline={"scope": "F1"},
            reward_weights={"scope": "F1"},
            bridge=object(),
        )
        evaluator = types.SimpleNamespace(
            dataset_splits={"validation_full": validation_examples},
            model=object(),
            reversible_handler=object(),
            layers_attribute="encoder.layer",
            is_regression=False,
        )
        runner = types.SimpleNamespace(
            _build_validation_full_batches=lambda _ev: validation_batches,
        )

        with mock.patch(
            "blb_stage2_rl.probe_runner.build_probe_runner",
            side_effect=AssertionError("F4 must not allocate a second pool"),
        ) as duplicate_builder:
            promotion_env, example_count = _build_authoritative_validation_env(
                runner=runner,
                ev=evaluator,
                base_env=base_env,
                train_cfg=types.SimpleNamespace(profile="mrpc"),
                reward_devices=[0, 1, 2, 3, 4],
                log=lambda _message: None,
            )

        duplicate_builder.assert_not_called()
        self.assertEqual(example_count, len(validation_examples))
        self.assertEqual(owner.num_workers, 5)
        self.assertEqual(len(owner._process_workers), 4)
        self.assertEqual(
            owner.devices,
            [f"cuda:{device_id}" for device_id in range(5)],
        )
        self.assertEqual(
            owner.registrations,
            [("F4", tuple(validation_batches))],
        )
        self.assertEqual(base_env.probe_runner.batch_set_key, "F1")
        self.assertEqual(promotion_env.probe_runner.batch_set_key, "F4")
        self.assertEqual(
            base_env.probe_runner.pool_id,
            promotion_env.probe_runner.pool_id,
        )
        self.assertEqual(promotion_env.probe_batches, validation_batches)
        self.assertIsNone(promotion_env._installed_action_hash)
        self.assertFalse(promotion_env.env_cfg.persistent_probe_install)

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

    def test_objective_convergence_requires_minimum_episodes_and_revalidation(self):
        from blb_stage2_rl.layerwise_runner import LayerwiseConvergenceTracker

        tracker = LayerwiseConvergenceTracker(
            patience_updates=100, minimum_episodes=90_000,
        )
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
        self.assertFalse(state.plateau_ready)
        self.assertFalse(state.converged)
        self.assertEqual(state.termination_reason, "running")

        state = tracker.observe_update(
            completed_episodes=90_000,
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

        tracker = LayerwiseConvergenceTracker(
            patience_updates=100, minimum_episodes=90_000,
        )
        state = tracker.state_dict()

        self.assertEqual(state["patience_updates"], 100)
        self.assertEqual(state["minimum_episodes"], 90_000)
        self.assertNotIn("maximum_episodes", state)

        incompatible = dict(state, patience_updates=99)
        with self.assertRaisesRegex(ValueError, "convergence contract mismatch"):
            tracker.load_state_dict(incompatible)
        incompatible = dict(state, minimum_episodes=80_000)
        with self.assertRaisesRegex(ValueError, "convergence contract mismatch"):
            tracker.load_state_dict(incompatible)

    def test_layerwise_contract_records_multifidelity_and_honest_stop_statuses(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "blb_stage2_rl"
            / "sequential_runner.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn(
            "dual_resource_maxmin_shapley_three_bank_convergence_v10", source,
        )
        self.assertIn('"F1": {', source)
        self.assertIn('"F4": {', source)
        self.assertIn('"minimum_episodes": convergence_min_episodes', source)
        self.assertNotIn('"maximum_episodes": convergence_max_episodes', source)
        self.assertIn('completion_status = "max_episodes_reached"', source)
        self.assertIn('"strict_revalidation_required": True', source)
        self.assertIn(
            '"strict_revalidation_diagnostic_probability": float(', source,
        )
        self.assertNotIn('"strict_revalidation_probability": float(', source)
        self.assertNotIn('completion_status = "bounded_budget_exhausted"', source)

    def test_layerwise_algorithm_identity_excludes_extendable_runtime_cap(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )

        self.assertIn(
            'algorithm_termination["episode_limit"] = "runtime_extendable"',
            source,
        )
        self.assertIn('"termination": algorithm_termination', source)
        self.assertIn(
            "validate_layerwise_episode_limit_extension(", source,
        )

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

    def test_p3_resource_score_supports_all_24_bert_large_layers(self):
        from blb_stage2_rl.layerwise_runner import redistribute_layerwise_rewards

        layer_resources = (0.01,) * 24
        rewards = redistribute_layerwise_rewards(
            terminal_reward=2.0,
            priority=3,
            ppo_resource_score=sum(layer_resources),
            layer_resource_rewards=layer_resources,
        )

        self.assertEqual(len(rewards), 24)
        self.assertEqual(rewards[:-1], layer_resources[:-1])
        self.assertAlmostEqual(sum(rewards), 2.0)

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
    @staticmethod
    def _counting_probe_owner():
        owner = types.SimpleNamespace(
            closed=False,
            close_calls=0,
            remote_close_count=0,
        )

        def close():
            owner.close_calls += 1
            if not owner.closed:
                owner.closed = True
                owner.remote_close_count += 1

        owner.close = close
        return owner

    def _run_with_mocked_locked_stage2(self, locked_impl):
        from blb_stage2_rl import layerwise_runner
        from blb_stage2_rl import runner as runner_module
        from blb_stage2_rl import sequential_runner

        class _NoopRunLock:
            def __init__(self, _path):
                self.path = _path

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        with mock.patch.object(
            layerwise_runner, "LayerwiseRunLock", _NoopRunLock,
        ):
            with mock.patch.object(
                runner_module,
                "resolve_blb_persistence_dir",
                return_value="/tmp/shared-probe-lifecycle-test",
            ):
                with mock.patch.object(
                    sequential_runner,
                    "_run_sequential_via_runner_locked",
                    side_effect=locked_impl,
                ):
                    return sequential_runner.run_sequential_via_runner(
                        runner=types.SimpleNamespace(evaluator=object()),
                        train_cfg=object(),
                        fixed_gelu=None,
                        fixed_softmax=None,
                        fixed_label="test",
                        fixed_source="test",
                    )

    def test_layerwise_branch_derives_policy_and_identity_from_model_depth(self):
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
            "layerwise_horizon = int(layerwise_env.horizon)",
            branch_source,
        )
        self.assertIn("horizon=layerwise_horizon", branch_source)
        self.assertIn("num_layers=layerwise_horizon", branch_source)
        self.assertIn(
            "step_layer_indices=tuple(range(layerwise_horizon))",
            branch_source,
        )
        self.assertIn("layerwise_action_space_version(", branch_source)
        self.assertIn("max_compute_saving_units(", branch_source)
        self.assertIn("max_communication_saving_units(", branch_source)
        self.assertGreaterEqual(branch_source.count("layerwise_horizon"), 8)
        self.assertIn(
            "num_layers=int(evaluator.total_layers)",
            branch_source,
        )
        self.assertIn(
            "model_type=layerwise_model_type",
            branch_source,
        )
        self.assertIn(
            "fusion_count=2 * layerwise_horizon + b4_count",
            branch_source,
        )
        self.assertIn(
            "fusion_count_b2=layerwise_horizon",
            branch_source,
        )
        self.assertIn(
            "fusion_count_b5=layerwise_horizon",
            branch_source,
        )
        self.assertNotIn("valid_steps=12", branch_source)
        self.assertNotIn("steps_taken=12", branch_source)
        self.assertNotIn("fusion_count=24", branch_source)
        self.assertNotIn("fusion_count_b2=12", branch_source)
        self.assertNotIn("fusion_count_b5=12", branch_source)
        self.assertNotIn("/ 12.0", branch_source)
        self.assertNotIn("horizon=12", branch_source)
        self.assertNotIn("num_layers=12", branch_source)

    def test_stage2_runner_uses_shared_calibrated_action_context(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        runner = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_sequential_via_runner_locked"
        )
        runner_source = ast.get_source_segment(source, runner)

        self.assertIn(
            "load_calibrated_stage2_action_context(",
            runner_source,
        )
        self.assertIn(
            "validate_calibrated_stage2_action_context(",
            runner_source,
        )
        self.assertIn(
            "baseline.typical_fusion_count = float(base_env.num_layers)",
            runner_source,
        )
        self.assertIn(
            "stage2_model_type = resolve_stage2_model_type(",
            runner_source,
        )
        self.assertIn("model_type=stage2_model_type", runner_source)
        self.assertNotIn("load_static_skeletons_baseline(", runner_source)
        self.assertNotIn("static_skeletons_baseline_to_action(", runner_source)

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
        resource_contract = {
            "algorithm_contract_hash": "algorithm-v9",
            "resource_secondary_epsilon": 1.0e-4,
            "compute_axis_denominator": 12,
            "communication_axis_denominator": 295,
            "resource_credit_mode": "two_family_shapley_per_slot_v1",
            "strict_resource_order": ["robust_floor", "secondary_progress"],
        }
        first = binder(base, first_order, "cost-v1", resource_contract)
        second = binder(base, second_order, "cost-v1", resource_contract)

        self.assertEqual(first["k_levels"], list(first_order))
        self.assertEqual(second["k_levels"], list(second_order))
        self.assertNotEqual(candidate_key([0], first), candidate_key([0], second))

    def test_layerwise_candidate_identity_binds_every_resource_contract_field(self):
        from blb_stage2_rl.candidate_store import candidate_key
        from blb_stage2_rl.layerwise_runner import bind_layerwise_candidate_identity

        contract = {
            "algorithm_contract_hash": "algorithm-v9",
            "resource_secondary_epsilon": 1.0e-4,
            "compute_axis_denominator": 12,
            "communication_axis_denominator": 295,
            "resource_credit_mode": "two_family_shapley_per_slot_v1",
            "strict_resource_order": ["robust_floor", "secondary_progress"],
        }
        base = bind_layerwise_candidate_identity(
            {"action_space_version": "stage2_layerwise_12x6_v1"},
            (8, 9, 11, 13, 10, 12),
            "dual_resource_maxmin_shapley_v1",
            contract,
        )
        base_key = candidate_key([0], base)
        mutations = {
            "algorithm_contract_hash": "algorithm-v10",
            "resource_secondary_epsilon": 2.0e-4,
            "compute_axis_denominator": 24,
            "communication_axis_denominator": 294,
            "resource_credit_mode": "other_credit_mode",
            "strict_resource_order": ["secondary_progress", "robust_floor"],
        }

        for field_name, replacement in mutations.items():
            changed_contract = {**contract, field_name: replacement}
            changed = bind_layerwise_candidate_identity(
                {"action_space_version": "stage2_layerwise_12x6_v1"},
                (8, 9, 11, 13, 10, 12),
                "dual_resource_maxmin_shapley_v1",
                changed_contract,
            )
            self.assertNotEqual(
                candidate_key([0], changed),
                base_key,
                msg=field_name,
            )

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

    def test_cuda_rng_role_registry_survives_shrink_and_reexpansion(self):
        from blb_stage2_rl.sequential_runner import (
            merge_cuda_rng_role_registry,
            resolve_cuda_rng_role_registry,
        )

        checkpoint = {
            "cuda_rng_role_registry_version": 1,
            "cuda_rng_state_by_role": ["r0", "r1", "r2", "r3", "r4"],
            "cuda_rng_active_role_count": 5,
        }
        registry, active = resolve_cuda_rng_role_registry(
            checkpoint,
            active_role_count=4,
            new_role_state_factory=lambda role: f"new-{role}",
        )
        self.assertEqual(active, ["r0", "r1", "r2", "r3"])
        self.assertEqual(registry, ["r0", "r1", "r2", "r3", "r4"])

        registry = merge_cuda_rng_role_registry(
            registry,
            ["n0", "n1", "n2", "n3"],
        )
        resumed_registry, resumed_active = resolve_cuda_rng_role_registry(
            {
                "cuda_rng_role_registry_version": 1,
                "cuda_rng_state_by_role": registry,
                "cuda_rng_active_role_count": 4,
            },
            active_role_count=5,
            new_role_state_factory=lambda role: f"new-{role}",
        )
        self.assertEqual(
            resumed_active,
            ["n0", "n1", "n2", "n3", "r4"],
        )
        self.assertEqual(resumed_registry, resumed_active)

    def test_cuda_rng_role_registry_initializes_only_never_seen_roles(self):
        from blb_stage2_rl.sequential_runner import (
            resolve_cuda_rng_role_registry,
        )

        initialized = []
        registry, active = resolve_cuda_rng_role_registry(
            {
                "cuda_rng_role_registry_version": 1,
                "cuda_rng_state_by_role": ["r0", "r1"],
                "cuda_rng_active_role_count": 2,
            },
            active_role_count=4,
            new_role_state_factory=lambda role: (
                initialized.append(role) or f"new-{role}"
            ),
        )

        self.assertEqual(initialized, [2, 3])
        self.assertEqual(active, ["r0", "r1", "new-2", "new-3"])
        self.assertEqual(registry, active)

    def test_legacy_cuda_rng_checkpoint_rejects_changed_gpu_count(self):
        from blb_stage2_rl.sequential_runner import (
            resolve_cuda_rng_role_registry,
        )

        with self.assertRaisesRegex(RuntimeError, "legacy.*GPU count"):
            resolve_cuda_rng_role_registry(
                {"cuda_rng_state_all": ["r0", "r1", "r2", "r3", "r4"]},
                active_role_count=4,
                new_role_state_factory=lambda role: f"new-{role}",
            )

    def test_layerwise_gpu_recovery_restart_waits_for_committed_checkpoint(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        tree = ast.parse(source)
        branch = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_layerwise_training_branch"
        )
        callback = next(
            node for node in branch.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "on_layerwise_update"
        )
        callback_source = ast.get_source_segment(source, callback)

        diagnostics_commit = callback_source.index(
            "diag_recorder.record_ppo_update(update_stats)"
        )
        checkpoint_commit = callback_source.index(
            "save_layerwise_checkpoint("
        )
        restart_boundary = callback_source.index(
            "raise_if_elastic_gpu_restart_requested()"
        )
        self.assertLess(diagnostics_commit, checkpoint_commit)
        self.assertLess(checkpoint_commit, restart_boundary)

    def test_layerwise_training_wraps_primary_cuda_device_failures(self):
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

        self.assertIn("except ElasticGPUFailure:", branch_source)
        self.assertIn("is_recoverable_gpu_failure(exc)", branch_source)
        self.assertIn(
            'ElasticGPUFailure(\n'
            '                    device="cuda:0",\n'
            '                    role="learner-primary",',
            branch_source,
        )

    def test_layerwise_checkpoint_contract_fails_before_mutating_training_state(self):
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
        validation = branch_source.rindex(
            "validate_layerwise_checkpoint_metadata("
        )

        self.assertLess(validation, branch_source.index("policy.load_state_dict("))
        self.assertLess(validation, branch_source.index("optimizer.load_state_dict("))
        self.assertLess(
            validation,
            branch_source.index("candidate_store.recover_to_checkpoint_size("),
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

    def test_outer_owner_holder_closes_after_pre_branch_failure(self):
        owner = self._counting_probe_owner()

        def fail_after_owner_creation(**kwargs):
            holder = kwargs.get("probe_runner_owner_holder")
            self.assertIsNotNone(holder)
            holder.bind(owner)
            raise RuntimeError("F4 registration failure")

        with self.assertRaisesRegex(RuntimeError, "F4 registration failure"):
            self._run_with_mocked_locked_stage2(fail_after_owner_creation)

        self.assertEqual(owner.close_calls, 1)
        self.assertEqual(owner.remote_close_count, 1)

    def test_outer_owner_holder_closes_after_non_layerwise_return(self):
        owner = self._counting_probe_owner()

        def return_after_owner_creation(**kwargs):
            holder = kwargs.get("probe_runner_owner_holder")
            self.assertIsNotNone(holder)
            holder.bind(owner)
            return {"status": "non-layerwise-complete"}

        result = self._run_with_mocked_locked_stage2(
            return_after_owner_creation,
        )

        self.assertEqual(result, {"status": "non-layerwise-complete"})
        self.assertEqual(owner.close_calls, 1)
        self.assertEqual(owner.remote_close_count, 1)

    def test_outer_owner_holder_is_idempotent_after_layerwise_cleanup(self):
        owner = self._counting_probe_owner()

        def return_after_inner_cleanup(**kwargs):
            holder = kwargs.get("probe_runner_owner_holder")
            self.assertIsNotNone(holder)
            holder.bind(owner)
            owner.close()
            return {"status": "layerwise-complete"}

        result = self._run_with_mocked_locked_stage2(
            return_after_inner_cleanup,
        )

        self.assertEqual(result, {"status": "layerwise-complete"})
        self.assertGreaterEqual(owner.close_calls, 1)
        self.assertEqual(owner.remote_close_count, 1)

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
            'checkpoint.get("strict_pareto_frontier")',
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
            "restored_diagnostics = diag_recorder.restore_existing()",
            "completed_episode_count = int(start_episode)",
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
            '"strict_pareto_frontier": copy.deepcopy(strict_pareto_frontier)',
            '"cuda_rng_role_registry_version": 1',
            '"cuda_rng_state_by_role": cuda_rng_role_registry',
            '"cuda_rng_active_role_count": len(active_cuda_rng_states)',
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
            '"dual_resource_maxmin_shapley_three_bank_convergence_v10"',
            branch_source,
        )
        self.assertIn('"algorithm_contract_hash": algorithm_contract_hash', branch_source)
        self.assertIn('"run_context_hash": run_context_hash', branch_source)
        self.assertIn("validate_layerwise_checkpoint_metadata(", branch_source)
        self.assertIn('"cost_model_revision": LAYERWISE_COST_MODEL_REVISION', source)
        self.assertIn('"resource_secondary_epsilon":', branch_source)
        self.assertIn('"compute_axis_denominator":', branch_source)
        self.assertIn('"communication_axis_denominator":', branch_source)
        self.assertIn('"resource_credit_mode": "two_family_shapley_per_slot_v1"', branch_source)
        self.assertIn(
            '"strict_resource_order": ["robust_floor", "secondary_progress"]',
            branch_source,
        )
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
        self.assertIn('"mode": "convergence_or_max_episodes"', branch_source)
        self.assertIn('"minimum_episodes": convergence_min_episodes', branch_source)
        self.assertIn('"validation_banks": authoritative_validation_banks.contract_payload()', branch_source)
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
            '"feasible,robust_floor,secondary_progress,confidence_vector,"',
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

        self.assertIn('"stage2_layerwise_robust_run_v5"', branch_source)
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
        self.assertIn('"max_episodes_reached"', branch_source)
        self.assertNotIn('"bounded_budget_exhausted"', branch_source)
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

    def __init__(
            self,
            probabilities=0.7,
            evidence_mode="valid",
            invalid=False,
            num_layers=12,
    ):
        self.horizon = int(num_layers)
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
        done = self._step == self.horizon
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
        from blb_stage2_rl.layerwise_action import compute_variable_cost_from_action_matrix

        objective = compute_variable_cost_from_action_matrix(self.actions)
        resource_objective = {
            "compute_saving": objective.compute_saving,
            "communication_saving": objective.communication_saving,
            "robust_floor": objective.robust_floor,
            "secondary_progress": objective.secondary_progress,
            "ppo_resource_score": objective.ppo_resource_score,
            "compute_shapley_credit": objective.compute_shapley_credit,
            "communication_shapley_credit": objective.communication_shapley_credit,
            "fusion_count": objective.fusion_count,
            "removed_k_bits": objective.removed_k_bits,
            "layer_resource_rewards": list(objective.layer_resource_rewards),
            "slot_resource_rewards": [list(row) for row in objective.slot_resource_rewards],
        }
        self.last_resource_objective = resource_objective
        return np.zeros(4, dtype=np.float32), (-5.0 if self._invalid else 7.5), True, {
            "policy_actions": [row[:] for row in self.actions],
            "pending_full_vector": list(range(20)),
            "resource_objective": resource_objective,
            "variable_cost": dict(resource_objective),
            "layer_summaries": [
                {"all_valid": True} for _ in range(self.horizon)
            ],
        }


class _DeferredFakeLayerwiseEnv(_FakeLayerwiseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._deferred = False
        self._pending_terminal_probe = None
        self.grouped_calls = []
        self.base.env_cfg = types.SimpleNamespace(
            persistent_probe_install=True,
        )
        self.base.probe_runner = types.SimpleNamespace(
            num_workers=4,
            pool_generation=0,
            run_action_trial_groups=lambda *_args, **_kwargs: None,
        )
        self.base.evaluate_prepared_terminal_batch = self._evaluate_prepared

    def configure_terminal_probe_deferral(self, enabled):
        self._deferred = bool(enabled)

    @property
    def pending_terminal_probe(self):
        return self._pending_terminal_probe

    def reset(self, *, seed=None):
        self._pending_terminal_probe = None
        return super().reset(seed=seed)

    def step(self, action):
        result = super().step(action)
        state, reward, done, info = result
        if not done or not self._deferred:
            return result
        prepared = {
            "probe_base_seed": int(self.base.probe_noise_seed),
            "runtime_info": self.runtime_terminal_info,
            "terminal_reward": float(reward),
        }
        self._pending_terminal_probe = {
            "prepared": prepared,
            "boosted_overrides": dict(self.boosted_overrides),
        }
        self.runtime_terminal_info = None
        return state, 0.0, True, {**info, "terminal_probe_deferred": True}

    def _evaluate_prepared(self, prepared, **kwargs):
        items = list(prepared)
        self.grouped_calls.append((
            [int(item["probe_base_seed"]) for item in items],
            dict(kwargs),
        ))
        return [
            (
                np.zeros(4, dtype=np.float32),
                float(item["terminal_reward"]),
                True,
                item["runtime_info"],
            )
            for item in items
        ]


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


def _three_validation_banks():
    from blb_stage2_rl.layerwise_runner import (
        LayerwiseValidationBank,
        LayerwiseValidationBanks,
    )
    from blb_stage2_rl.seed_utils import derive_probe_trial_seed
    from blb_stage2_rl.statistical_constraints import TrialSeries

    def reference(probe_seeds):
        seeds = tuple(
            derive_probe_trial_seed(probe_seed, trial_idx)
            for probe_seed in probe_seeds
            for trial_idx in range(5)
        )
        return types.SimpleNamespace(
            **vars(_strict_reference()),
            trials=TrialSeries(
                loss=[0.30 + 0.0001 * (idx % 5) for idx in range(len(seeds))],
                metric1=[0.90 - 0.0001 * (idx % 5) for idx in range(len(seeds))],
                metric2=[0.80 - 0.0001 * (idx % 5) for idx in range(len(seeds))],
                seeds=seeds,
            ),
        )

    a_seeds = (101, 102, 103, 104, 105)
    b_seeds = (201, 202, 203, 204, 205)
    c_seeds = (301, 302, 303, 304, 305)
    return LayerwiseValidationBanks(
        bank_a=LayerwiseValidationBank(
            "A", reference(a_seeds), a_seeds, 5,
        ),
        bank_b=LayerwiseValidationBank(
            "B", reference(b_seeds), b_seeds, 5,
        ),
        bank_c=LayerwiseValidationBank(
            "C", reference(c_seeds), c_seeds, 5,
        ),
        promotion_reference=reference(a_seeds + b_seeds),
        final_reference=reference(a_seeds + b_seeds + c_seeds),
    )


def _resource_objective_for_matrix(action_matrix):
    from blb_stage2_rl.layerwise_action import compute_variable_cost_from_action_matrix

    objective = compute_variable_cost_from_action_matrix(action_matrix)
    return {
        "compute_saving": objective.compute_saving,
        "communication_saving": objective.communication_saving,
        "robust_floor": objective.robust_floor,
        "secondary_progress": objective.secondary_progress,
        "ppo_resource_score": objective.ppo_resource_score,
    }


def _fake_policy_action_matrix():
    return [[1, 0, 4, 3, 2, 1]] + [[1, 5, 4, 3, 2, 1] for _ in range(11)]


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
            terminal_eval_batch_size=1,
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

    def test_exact_terminal_batch_size_uses_only_the_cross_episode_imbalance(self):
        from blb_stage2_rl.layerwise_runner import resolve_exact_terminal_batch_size

        self.assertEqual(resolve_exact_terminal_batch_size(4, 5, 4), 4)
        self.assertEqual(resolve_exact_terminal_batch_size(2, 5, 4), 2)
        self.assertEqual(resolve_exact_terminal_batch_size(4, 5, 5), 1)
        self.assertEqual(resolve_exact_terminal_batch_size(1, 5, 4), 1)

    def test_grouped_terminal_probes_finalize_in_order_at_the_ppo_boundary(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise
        from blb_stage2_rl.seed_utils import derive_layerwise_episode_probe_seed

        env = _DeferredFakeLayerwiseEnv(probabilities=0.7)
        config = self._train_cfg(total_episodes=4, update_every=4)
        config.terminal_eval_batch_size = 4
        buffer = _FakeBuffer()
        completed = []
        updates = []

        def fake_update(_policy, _optimizer, rollout, _cfg, _device, **_kwargs):
            updates.append(len(rollout))
            return {"entropy": 0.0, "n_samples": len(rollout)}

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=env,
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=buffer,
                ppo_update_fn=fake_update,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
                on_episode_end=lambda record: completed.append(record.episode_index),
            )

        expected_seeds = [
            derive_layerwise_episode_probe_seed(42, episode, trial_count=5)
            for episode in range(4)
        ]
        self.assertEqual(env.grouped_calls, [(
            expected_seeds,
            {"num_trials_per_action": 5, "validation_required": False},
        )])
        self.assertEqual(completed, [0, 1, 2, 3])
        self.assertEqual(updates, [48])
        self.assertEqual(summary["episode_rewards"], [7.5] * 4)
        self.assertEqual([record.episode_index for record in summary["episode_records"]], [0, 1, 2, 3])

    def test_terminal_batch_rebalances_after_probe_pool_shrinks(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise
        from blb_stage2_rl.seed_utils import derive_layerwise_episode_probe_seed

        class ShrinkingProbeEnv(_DeferredFakeLayerwiseEnv):
            def __init__(self):
                super().__init__(probabilities=0.7)
                self.base.probe_runner.num_workers = 5
                self.defer_history = []
                self.completed_terminal_episodes = 0

            def configure_terminal_probe_deferral(self, enabled):
                super().configure_terminal_probe_deferral(enabled)
                self.defer_history.append(bool(enabled))

            def step(self, action):
                result = super().step(action)
                if result[2]:
                    self.completed_terminal_episodes += 1
                    if self.completed_terminal_episodes == 1:
                        self.base.probe_runner.num_workers = 4
                        self.base.probe_runner.pool_generation = 1
                return result

        env = ShrinkingProbeEnv()
        config = self._train_cfg(total_episodes=5, update_every=5)
        config.terminal_eval_batch_size = 4
        buffer = _FakeBuffer()
        completed = []
        updates = []

        def fake_update(_policy, _optimizer, rollout, _cfg, _device, **_kwargs):
            updates.append(len(rollout))
            return {"entropy": 0.0, "n_samples": len(rollout)}

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=env,
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=buffer,
                ppo_update_fn=fake_update,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
                on_episode_end=lambda record: completed.append(
                    record.episode_index
                ),
            )

        expected_grouped_seeds = [
            derive_layerwise_episode_probe_seed(
                42, episode, trial_count=5,
            )
            for episode in range(1, 5)
        ]
        self.assertEqual(env.defer_history, [False, True])
        self.assertEqual(
            env.grouped_calls,
            [(
                expected_grouped_seeds,
                {
                    "num_trials_per_action": 5,
                    "validation_required": False,
                },
            )],
        )
        self.assertEqual(completed, [0, 1, 2, 3, 4])
        self.assertEqual(updates, [60])
        self.assertEqual(
            summary["probe_pool_schedule"],
            [
                {
                    "first_episode": 0,
                    "pool_generation": 0,
                    "worker_count": 5,
                    "terminal_batch_size": 1,
                },
                {
                    "first_episode": 1,
                    "pool_generation": 1,
                    "worker_count": 4,
                    "terminal_batch_size": 4,
                },
            ],
        )

    def test_protected_k1_reject_never_enters_candidate_store(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        env = _DeferredFakeLayerwiseEnv(probabilities=0.1)
        env.base.probe_runner.run_action_trial_groups_at_indices = (
            lambda *_args, **_kwargs: None
        )
        protected_calls = []

        def evaluate_protected(prepared, **kwargs):
            protected_calls.append(dict(kwargs))
            outputs = []
            for item in prepared:
                info = dict(item["runtime_info"])
                info.update({
                    "reward_breakdown": types.SimpleNamespace(priority=1),
                    "statistical_trials": {
                        name: values[:1]
                        for name, values in info["statistical_trials"].items()
                    },
                    "protected_k1_enabled": True,
                    "protected_k1_screened": True,
                    "protected_k1_audited": False,
                    "protected_k1_k1_only_reject": True,
                    "protected_k1_audit_precision_false_negative": False,
                    "protected_k1_audit_p3_false_negative": False,
                    "protected_k1_reason": "extreme_precision_failure",
                    "protected_k1_guard_sigma": 4.0,
                    "protected_k1_worst_precision_z": 5.0,
                    "protected_k1_trials_executed": 1,
                    "forward_ran": True,
                })
                outputs.append((
                    np.zeros(4, dtype=np.float32),
                    -3.0,
                    True,
                    info,
                ))
            return outputs

        env.base.evaluate_prepared_terminal_batch_protected_k1 = (
            evaluate_protected
        )
        env.base.evaluate_prepared_terminal_batch = evaluate_protected
        config = self._train_cfg(total_episodes=1, update_every=1)
        config.protected_k1_enabled = True
        config.protected_k1_guard_sigma = 4.0
        config.protected_k1_audit_fraction = 0.02
        buffer = _FakeBuffer()

        with (
                tempfile.TemporaryDirectory() as td,
                mock.patch(
                    "blb_stage2_rl.layerwise_runner."
                    "_can_expand_resource_frontier",
                    return_value=False,
                ),
        ):
            path = Path(td) / "candidates.jsonl"
            summary = train_layerwise(
                env=env,
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=CandidateStore(path),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=buffer,
                ppo_update_fn=(
                    lambda *_args, **_kwargs: {
                        "entropy": 0.0,
                        "n_samples": len(buffer),
                    }
                ),
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.1),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )
            candidate_rows = (
                path.read_text(encoding="utf-8").splitlines()
                if path.exists() else []
            )

        record = summary["episode_records"][0]
        self.assertEqual(candidate_rows, [])
        self.assertEqual(record.fresh_trial_count, 1)
        self.assertEqual(record.pooled_trial_count, 0)
        self.assertEqual(record.promotion_status, "protected_k1_screened")
        self.assertEqual(record.reward_evidence, "protected_k1_precision_only")
        self.assertEqual(record.ranking_evidence, "not_eligible_protected_k1")
        self.assertTrue(record.protected_k1_k1_only_reject)
        self.assertEqual(summary["protected_k1"]["trials_saved_vs_k5"], 4)
        self.assertEqual(
            protected_calls[0],
            {
                "absolute_episodes": [0],
                "force_k5_mask": [False],
                "guard_sigma": 4.0,
                "audit_fraction": 0.02,
                "audit_seed": 42,
                "validation_required": False,
            },
        )

    def test_layerwise_branch_propagates_runtime_terminal_batch_size(self):
        source = Path("blb_stage2_rl/sequential_runner.py").read_text(
            encoding="utf-8",
        )
        self.assertIn(
            "terminal_eval_batch_size=int(train_cfg.terminal_eval_batch_size)",
            source,
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

    def test_train_collects_all_24_bert_large_transitions(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        env = _FakeLayerwiseEnv(probabilities=0.7, num_layers=24)
        policy = _FakePolicy()
        buffer = _FakeBuffer()

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=env,
                policy=policy,
                train_cfg=self._train_cfg(),
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-24-v1"},
                optimizer=object(),
                rollout_buffer=buffer,
                ppo_update_fn=lambda *_args, **_kwargs: {
                    "entropy": 0.0,
                    "n_samples": len(buffer),
                },
                assess_candidate_fn=lambda trials, *_args, **_kwargs: _assessment(0.7),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(len(buffer.transitions), 24)
        self.assertEqual(
            [row["done"] for row in buffer.transitions],
            [False] * 23 + [True],
        )
        self.assertEqual(len(summary["episode_records"][0].action_matrix), 24)
        self.assertEqual(summary["episode_records"][0].step_count, 24)

    def test_max_episode_cap_still_runs_bank_c_and_keeps_resumable_result(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        env = _FakeLayerwiseEnv(probabilities=0.9)
        promotion_base = _PromotionBase(fresh_probability=0.01)
        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=env,
                promotion_base_env=promotion_base,
                validation_banks=_three_validation_banks(),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=1, update_every=2),
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {"entropy": 0.0},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertFalse(summary["converged"])
        self.assertTrue(summary["strict_revalidation_passed"])
        self.assertEqual(summary["strict_revalidation_status"], "passed")
        self.assertEqual(summary["termination_reason"], "max_episodes_reached")
        self.assertEqual(
            summary["best_promotion_evidence"]["trial_count"], 75,
        )
        self.assertEqual(len(promotion_base.evaluate_calls), 15)

    def test_max_cap_falls_back_to_dominated_ab_candidate_when_winner_fails_c(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        high_matrix = _fake_policy_action_matrix()
        low_matrix = [[0] * 6 for _ in range(12)]

        def candidate(key, vector, matrix):
            return {
                "candidate_key": key,
                **_resource_objective_for_matrix(matrix),
                "variable_cost": _resource_objective_for_matrix(matrix)[
                    "ppo_resource_score"
                ],
                "assessment": _assessment(0.99),
                "metrics": {
                    "loss_mean": 0.30,
                    "loss_std": 0.001,
                    "metric1_mean": 0.90,
                    "metric1_std": 0.001,
                    "metric2_mean": 0.80,
                    "metric2_std": 0.001,
                },
                "constraint_safety_margins": [0.1] * 6,
                "action_matrix": matrix,
                "full_vector": vector,
                "boosted_overrides": {},
                "reward": 1.0,
                "promotion_trials": None,
                "final_revalidation_status": "not_run",
            }

        high = candidate("high", list(range(20)), high_matrix)
        low = candidate("low", list(range(20, 40)), low_matrix)
        self.assertGreater(
            high["robust_floor"], low["robust_floor"],
        )

        def certification(**kwargs):
            selected = kwargs["candidate"]
            passed = selected["candidate_key"] == "low"
            metrics = {
                "loss_mean": 0.30 if passed else 0.50,
                "loss_std": 0.001,
                "metric1_mean": 0.90,
                "metric1_std": 0.001,
                "metric2_mean": 0.80,
                "metric2_std": 0.001,
            }
            return types.SimpleNamespace(
                status=(
                    "final_revalidation_passed"
                    if passed else "bank_c_point_failed"
                ),
                evidence=types.SimpleNamespace(
                    trial_count=75,
                    trials=types.SimpleNamespace(
                        loss=(metrics["loss_mean"],) * 75,
                        metric1=(metrics["metric1_mean"],) * 75,
                        metric2=(metrics["metric2_mean"],) * 75,
                        seeds=tuple(range(75)),
                    ),
                ),
                assessment=_assessment(0.99),
                metrics=metrics,
            )
        no_promotion = types.SimpleNamespace(
            status="not_evaluated",
            trial_count=0,
            fresh_trial_count=0,
            evidence=None,
            assessment=None,
            metrics=None,
        )
        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.restore_promoted_candidates",
            return_value={"high": high, "low": low},
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner.promote_candidate_if_eligible",
            return_value=no_promotion,
        ), mock.patch(
            "blb_stage2_rl.layerwise_runner.certify_candidate_with_bank_c",
            side_effect=certification,
        ):
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.9),
                promotion_base_env=_PromotionBase(),
                validation_banks=_three_validation_banks(),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=1),
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {"entropy": 0.0},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertEqual(summary["best_full_vector"], low["full_vector"])
        self.assertEqual(summary["strict_revalidation_status"], "passed")
        self.assertTrue(summary["strict_revalidation_passed"])

    def test_train_bounded_history_can_be_disabled_without_changing_callbacks(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        completed = []
        ppo_updates = []

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(evidence_mode="missing", invalid=True),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=37, update_every=8),
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
                on_episode_end=lambda record: completed.append(
                    (record.episode_index, record.reward)
                ),
                on_ppo_update_end=lambda metrics, count, _record: ppo_updates.append(
                    (count, metrics["completed_episodes"])
                ),
                retain_history=False,
            )

        self.assertEqual(summary["completed_episodes"], 37)
        self.assertEqual(summary["episode_records"], [])
        self.assertEqual(summary["episode_rewards"], [])
        self.assertEqual(summary["ppo_metrics"], [])
        self.assertEqual(completed, [(index, -5.0) for index in range(37)])
        self.assertEqual(
            ppo_updates,
            [(8, 8), (16, 16), (24, 24), (32, 32), (37, 37)],
        )

    def test_train_bounded_history_preserves_nonzero_resume_callback_identities(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        config = self._train_cfg(total_episodes=5, update_every=2)
        config.absolute_episode_start = 40
        config.planned_total_episodes = 45
        completed = []
        ppo_updates = []

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(evidence_mode="missing", invalid=True),
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
                on_episode_end=lambda record: completed.append(record.episode_index),
                on_ppo_update_end=lambda metrics, count, _record: ppo_updates.append(
                    (count, metrics["completed_episodes"])
                ),
                retain_history=False,
            )

        self.assertEqual(summary["completed_episodes"], 45)
        self.assertEqual(summary["episode_records"], [])
        self.assertEqual(summary["episode_rewards"], [])
        self.assertEqual(summary["ppo_metrics"], [])
        self.assertEqual(completed, [40, 41, 42, 43, 44])
        self.assertEqual(ppo_updates, [(42, 42), (44, 44), (45, 45)])

    def test_train_bounded_history_defaults_to_complete_direct_call_lists(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import train_layerwise

        with tempfile.TemporaryDirectory() as td:
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(evidence_mode="missing", invalid=True),
                policy=_FakePolicy(),
                train_cfg=self._train_cfg(total_episodes=3, update_every=2),
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
        self.assertEqual(len(summary["episode_records"]), 3)
        self.assertEqual(summary["episode_rewards"], [-5.0, -5.0, -5.0])
        self.assertEqual(len(summary["ppo_metrics"]), 2)

    def test_exhaustive_c_fallback_reports_surviving_certified_tie_as_passed(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            _certify_strict_best_candidates,
        )

        matrix = _fake_policy_action_matrix()

        def candidate(key, probability, vector):
            resource = _resource_objective_for_matrix(matrix)
            return {
                "candidate_key": key,
                **resource,
                "variable_cost": resource["ppo_resource_score"],
                "assessment": _assessment(probability),
                "metrics": {
                    "loss_mean": 0.30,
                    "loss_std": 0.001,
                    "metric1_mean": 0.90,
                    "metric1_std": 0.001,
                    "metric2_mean": 0.80,
                    "metric2_std": 0.001,
                },
                "constraint_safety_margins": [0.1] * 6,
                "action_matrix": matrix,
                "full_vector": vector,
                "boosted_overrides": {},
                "reward": 1.0,
                "promotion_trials": None,
                "final_revalidation_status": "not_run",
            }

        candidates = {
            "a": candidate("a", 0.99, list(range(20))),
            "b": candidate("b", 0.98, list(range(20, 40))),
        }
        calls = []

        def certification(**kwargs):
            key = kwargs["candidate"]["candidate_key"]
            calls.append(key)
            passed = key == "a"
            metrics = {
                "loss_mean": 0.30 if passed else 0.50,
                "loss_std": 0.001,
                "metric1_mean": 0.90,
                "metric1_std": 0.001,
                "metric2_mean": 0.80,
                "metric2_std": 0.001,
            }
            return types.SimpleNamespace(
                status=(
                    "final_revalidation_passed"
                    if passed else "bank_c_point_failed"
                ),
                evidence=types.SimpleNamespace(
                    trial_count=75,
                    trials=types.SimpleNamespace(
                        loss=(metrics["loss_mean"],) * 75,
                        metric1=(metrics["metric1_mean"],) * 75,
                        metric2=(metrics["metric2_mean"],) * 75,
                        seeds=tuple(range(75)),
                    ),
                ),
                assessment=_assessment(0.01 if passed else 0.99),
                metrics=metrics,
            )

        with tempfile.TemporaryDirectory() as td, mock.patch(
            "blb_stage2_rl.layerwise_runner.certify_candidate_with_bank_c",
            side_effect=certification,
        ):
            status, winner = _certify_strict_best_candidates(
                env=types.SimpleNamespace(base=None),
                promotion_base_env=None,
                candidate_store=CandidateStore(Path(td) / "candidates.jsonl"),
                identity_context={"action_space_version": "layerwise-v1"},
                accepted_candidates=candidates,
                bootstrap_seed=99,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                final_probability=0.95,
                validation_banks=_three_validation_banks(),
                exhaustive_fallback=True,
            )

        self.assertEqual(calls, ["a", "b"])
        self.assertEqual(status, "passed")
        self.assertEqual(winner["candidate_key"], "a")
        self.assertEqual(candidates["a"]["final_revalidation_status"], "passed")

    def test_max_cap_does_not_export_an_uncertified_ab_candidate(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            promote_candidate_if_eligible,
            train_layerwise,
        )

        class AlwaysFailBankC(_PromotionBase):
            def evaluate_prepared_terminal_batch(self, prepared, **kwargs):
                del prepared, kwargs
                raise RuntimeError("transient validation backend failure")

        context = {"action_space_version": "layerwise-v1"}
        action = list(range(20))
        action_matrix = [[0] * 6 for _ in range(12)]
        banks = _three_validation_banks()
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            promoted = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=_PromotionBase()),
                promotion_base_env=_PromotionBase(),
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=action_matrix,
                assessment=_assessment(0.99),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                validation_banks=banks,
            )
            self.assertEqual(promoted.status, "promoted")
            config = self._train_cfg(total_episodes=0)
            config.absolute_episode_start = 150_000
            config.planned_total_episodes = 150_000
            summary = train_layerwise(
                env=_FakeLayerwiseEnv(probabilities=0.9),
                promotion_base_env=AlwaysFailBankC(),
                validation_banks=banks,
                policy=_FakePolicy(),
                train_cfg=config,
                candidate_store=store,
                identity_context=context,
                optimizer=object(),
                rollout_buffer=_FakeBuffer(),
                ppo_update_fn=lambda *_args, **_kwargs: {"entropy": 0.0},
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                step_adapter_fn=lambda spec, _max_dim, _max_levels: (
                    np.asarray(spec.slot_mask, dtype=bool),
                    np.asarray(spec.slot_dims, dtype=np.int64),
                ),
            )

        self.assertIsNone(summary["best_action"])
        self.assertIsNotNone(summary["bank_b_best"])
        self.assertEqual(summary["strict_revalidation_status"], "failed_evaluation")

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
        objective = _resource_objective_for_matrix(action_matrix)
        config.convergence_resume_state = {
            "best_robust_feasible_objective": [
                objective["robust_floor"], objective["secondary_progress"],
            ],
            "current_robust_feasible_objective": [
                objective["robust_floor"], objective["secondary_progress"],
            ],
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
            "action_matrix": _fake_policy_action_matrix(),
            "full_vector": list(range(20)),
            "boosted_overrides": {},
            "reward": 1.4,
            "promotion_trials": None,
            "constraint_safety_margins": [0.1] * 6,
        }
        config = self._train_cfg(total_episodes=0, update_every=1)
        objective = _resource_objective_for_matrix(frontier["action_matrix"])
        config.convergence_resume_state = {
            "best_robust_feasible_objective": [
                objective["robust_floor"], objective["secondary_progress"],
            ],
            "current_robust_feasible_objective": [
                objective["robust_floor"], objective["secondary_progress"],
            ],
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
        objective = _resource_objective_for_matrix(action_matrix)
        config.convergence_resume_state = {
            "best_robust_feasible_objective": [
                objective["robust_floor"], objective["secondary_progress"],
            ],
            "current_robust_feasible_objective": [
                objective["robust_floor"], objective["secondary_progress"],
            ],
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
        matrix = _fake_policy_action_matrix()
        expected_resource = compute_variable_cost_from_action_matrix(matrix)
        expected_cost = expected_resource.normalized
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
                "best_robust_feasible_objective": [
                    expected_resource.robust_floor,
                    expected_resource.secondary_progress,
                ],
                "current_robust_feasible_objective": [
                    expected_resource.robust_floor,
                    expected_resource.secondary_progress,
                ],
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
        expected_resource = _resource_objective_for_matrix(matrix)
        self.assertEqual(candidate["reward"], 1.25)
        self.assertEqual(
            candidate["ppo_resource_score"],
            expected_resource["ppo_resource_score"],
        )
        self.assertEqual(candidate["variable_cost"], expected_resource["ppo_resource_score"])
        self.assertEqual(candidate["robust_floor"], expected_resource["robust_floor"])
        self.assertEqual(
            candidate["secondary_progress"],
            expected_resource["secondary_progress"],
        )
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
        matrix = _fake_policy_action_matrix()
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
        self.assertEqual(summary["best_variable_cost"], expected_cost)
        self.assertEqual(summary["stall_update_windows"], 1)
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
        matrix = _fake_policy_action_matrix()
        expected_cost = _resource_objective_for_matrix(matrix)["ppo_resource_score"]
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
        self.assertEqual(
            observed_updates[0]["strict_best"]["variable_cost"], expected_cost,
        )
        self.assertEqual(summary["best_reward"], 1.25)


class _PromotionBase:
    def __init__(self, fresh_probability=0.9):
        self.statistical_reference = _strict_reference()
        self.prepare_calls = []
        self.evaluate_calls = []
        self.fresh_probability = fresh_probability
        self.probe_noise_seed = None
        self.clear_calls = 0
        self._installed_action_hash = "installed-action"

    def clear_installed_blb(self):
        self.clear_calls += 1

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

    def test_latest_promotion_status_reuses_warmed_candidate_index(self):
        from blb_stage2_rl.layerwise_runner import (
            _latest_promotion_status,
            evidence_identity_context,
        )

        action = list(range(20))
        context = evidence_identity_context(
            {"action_space_version": "layerwise-v1"}, "F4",
        )
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "effective_action_indices": action,
                "identity_context": context,
                "promotion_status": "bank_a_point_failed",
                "promotion_metadata": {"generation": 1},
            })
            store = type(store)(store.path)
            store.trial_count_for_action(action, context)
            store.append_promotion_status(
                action,
                context,
                status="promoted",
                metadata={"generation": 2},
            )

            with (
                mock.patch.object(
                    store,
                    "iter_active_records",
                    side_effect=AssertionError("hot lookup must not scan JSONL"),
                ),
                mock.patch.object(
                    store,
                    "_iter_active_records",
                    side_effect=AssertionError("hot lookup must reuse the index"),
                ),
            ):
                status, metadata = _latest_promotion_status(
                    store, action, context,
                )

        self.assertEqual(status, "promoted")
        self.assertEqual(metadata, {"generation": 2})

    def test_promotion_tops_up_five_to_25_through_real_chain_once(self):
        from blb_stage2_rl.layerwise_action import compute_variable_cost_from_action_matrix
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        action_matrix = [[0] * 6 for _ in range(12)]
        expected_objective = compute_variable_cost_from_action_matrix(action_matrix)
        with tempfile.TemporaryDirectory() as td:
            store = self._store_with_five(td)
            base = _PromotionBase()
            env = types.SimpleNamespace(base=base)
            kwargs = dict(
                env=env,
                candidate_store=store,
                action_indices=list(range(20)),
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=action_matrix,
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
        self.assertEqual(
            base.prepare_calls[0][1]["external_cost_score"],
            expected_objective.ppo_resource_score,
        )
        self.assertEqual(
            base.prepare_calls[0][1]["external_cost_rank"],
            expected_objective.ppo_resource_score,
        )
        self.assertEqual(
            base.prepare_calls[0][1]["external_resource_objective"]["robust_floor"],
            expected_objective.robust_floor,
        )
        self.assertEqual(
            base.prepare_calls[0][1]["boosted_overrides"],
            {(4, 3): {"v_mask_rescale_sf": 47}},
        )
        self.assertEqual(base.evaluate_calls[0][1]["num_trials_per_action"], 20)
        self.assertTrue(base.evaluate_calls[0][1]["validation_required"])

    def test_three_bank_promotion_uses_a_then_b_point_gates_not_probability(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import promote_candidate_if_eligible

        action = list(range(20))
        base = _PromotionBase(fresh_probability=0.01)
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                action_indices=action,
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.01),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=_three_validation_banks(),
            )
            repeated = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                action_indices=action,
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.01),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=_three_validation_banks(),
            )

        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.trial_count, 50)
        self.assertEqual(result.fresh_trial_count, 50)
        self.assertEqual(repeated.status, "already_promoted")
        self.assertEqual(len(base.evaluate_calls), 10)
        self.assertEqual(
            [call[1]["num_trials_per_action"] for call in base.evaluate_calls],
            [5] * 10,
        )

    def test_three_bank_promotion_resumes_from_complete_fixed_seed_groups(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            evidence_identity_context,
            promote_candidate_if_eligible,
        )
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = list(range(20))
        context = {"action_space_version": "layerwise-v1"}
        full_context = evidence_identity_context(context, "F4")
        banks = _three_validation_banks()
        base = _PromotionBase(fresh_probability=0.01)
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            for group_index in range(2):
                seeds = banks.bank_a.trial_seeds[
                    group_index * 5:(group_index + 1) * 5
                ]
                store.append_trial_group(
                    action,
                    TrialSeries(
                        loss=[0.30 + 0.001 * i for i in range(5)],
                        metric1=[0.90 - 0.001 * i for i in range(5)],
                        metric2=[0.80 - 0.001 * i for i in range(5)],
                        seeds=seeds,
                    ),
                    {"identity_context": full_context, "fidelity": "F4"},
                )
            result = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=[[0] * 6 for _ in range(12)],
                assessment=_assessment(0.01),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=banks,
            )

        self.assertEqual(result.status, "promoted")
        self.assertEqual(result.trial_count, 50)
        self.assertEqual(result.fresh_trial_count, 40)
        self.assertEqual(len(base.evaluate_calls), 8)
        self.assertEqual(
            tuple(result.evidence.trials.seeds),
            banks.bank_a.trial_seeds + banks.bank_b.trial_seeds,
        )

    def test_bank_c_certification_pools_75_trials_and_ignores_probability(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            certify_candidate_with_bank_c,
            promote_candidate_if_eligible,
        )

        action = list(range(20))
        action_matrix = [[0] * 6 for _ in range(12)]
        base = _PromotionBase(fresh_probability=0.01)
        banks = _three_validation_banks()
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            promoted = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                action_indices=action,
                identity_context={"action_space_version": "layerwise-v1"},
                action_matrix=action_matrix,
                assessment=_assessment(0.01),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=banks,
            )
            certified = certify_candidate_with_bank_c(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                identity_context={"action_space_version": "layerwise-v1"},
                candidate={
                    "candidate_key": promoted.evidence.candidate_key,
                    "full_vector": action,
                    "action_matrix": action_matrix,
                    "boosted_overrides": {},
                    "reward": 1.0,
                },
                bootstrap_seed=99,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=banks,
            )

        self.assertEqual(certified.status, "final_revalidation_passed")
        self.assertEqual(certified.trial_count, 75)
        self.assertEqual(certified.fresh_trial_count, 25)
        self.assertEqual(len(base.evaluate_calls), 15)

    def test_bank_c_transient_failure_is_retryable_after_restore(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            certify_candidate_with_bank_c,
            promote_candidate_if_eligible,
            restore_promoted_candidates,
        )

        class FailNextEvaluation(_PromotionBase):
            def __init__(self):
                super().__init__()
                self.fail_next = False

            def evaluate_prepared_terminal_batch(self, prepared, **kwargs):
                if self.fail_next:
                    self.fail_next = False
                    raise RuntimeError("transient validation backend failure")
                return super().evaluate_prepared_terminal_batch(prepared, **kwargs)

        action = list(range(20))
        action_matrix = [[0] * 6 for _ in range(12)]
        context = {"action_space_version": "layerwise-v1"}
        base = FailNextEvaluation()
        banks = _three_validation_banks()
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            promoted = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=action_matrix,
                assessment=_assessment(0.99),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                validation_banks=banks,
            )
            candidate = {
                "candidate_key": promoted.evidence.candidate_key,
                "full_vector": action,
                "action_matrix": action_matrix,
                "boosted_overrides": {},
                "reward": 1.0,
            }
            base.fail_next = True
            failed = certify_candidate_with_bank_c(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                identity_context=context,
                candidate=candidate,
                bootstrap_seed=99,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                validation_banks=banks,
            )
            restored = restore_promoted_candidates(
                candidate_store=store,
                identity_context=context,
                statistical_reference=banks.promotion_reference,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                validation_banks=banks,
            )
            retried = certify_candidate_with_bank_c(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                identity_context=context,
                candidate=next(iter(restored.values())),
                bootstrap_seed=100,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                validation_banks=banks,
            )

        self.assertEqual(failed.status, "failed_evaluation")
        self.assertEqual(len(restored), 1)
        self.assertEqual(retried.status, "final_revalidation_passed")
        self.assertEqual(retried.trial_count, 75)

    def test_three_bank_restore_uses_point_gate_and_preserves_final_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            certify_candidate_with_bank_c,
            promote_candidate_if_eligible,
            restore_promoted_candidates,
        )

        action = list(range(20))
        action_matrix = [[0] * 6 for _ in range(12)]
        context = {"action_space_version": "layerwise-v1"}
        base = _PromotionBase(fresh_probability=0.01)
        banks = _three_validation_banks()
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidates.jsonl")
            promoted = promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=action_matrix,
                assessment=_assessment(0.01),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=banks,
            )
            certify_candidate_with_bank_c(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                identity_context=context,
                candidate={
                    "candidate_key": promoted.evidence.candidate_key,
                    "full_vector": action,
                    "action_matrix": action_matrix,
                    "boosted_overrides": {},
                    "reward": 1.0,
                },
                bootstrap_seed=99,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=banks,
            )
            restored = restore_promoted_candidates(
                candidate_store=store,
                identity_context=context,
                statistical_reference=banks.promotion_reference,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.01),
                validation_banks=banks,
            )

        self.assertEqual(len(restored), 1)
        candidate = next(iter(restored.values()))
        self.assertEqual(candidate["final_revalidation_status"], "passed")
        self.assertEqual(len(candidate["promotion_trials"].loss), 75)
        self.assertTrue(all(
            value >= -1.0e-12
            for value in candidate["constraint_safety_margins"]
        ))

    def test_each_authoritative_point_gate_blocks_its_own_failure(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.layerwise_runner import (
            certify_candidate_with_bank_c,
            promote_candidate_if_eligible,
        )

        action = list(range(20))
        action_matrix = [[0] * 6 for _ in range(12)]
        context = {"action_space_version": "layerwise-v1"}

        def promote(store, base, banks):
            return promote_candidate_if_eligible(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                action_indices=action,
                identity_context=context,
                action_matrix=action_matrix,
                assessment=_assessment(0.99),
                priority=3,
                variable_cost=0.0,
                frontier_cost=None,
                boosted_overrides={},
                bootstrap_seed=77,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                validation_banks=banks,
            )

        with tempfile.TemporaryDirectory() as td:
            banks = _three_validation_banks()
            banks.bank_a.reference.metric2_limit = 0.99
            base = _PromotionBase()
            result = promote(
                CandidateStore(Path(td) / "bank_a.jsonl"), base, banks,
            )
            self.assertEqual(result.status, "bank_a_point_failed")
            self.assertEqual(len(base.evaluate_calls), 5)

            banks = _three_validation_banks()
            banks.promotion_reference.metric1_std_limit = 1.0e-6
            base = _PromotionBase()
            result = promote(
                CandidateStore(Path(td) / "bank_b.jsonl"), base, banks,
            )
            self.assertEqual(result.status, "bank_b_point_failed")
            self.assertEqual(len(base.evaluate_calls), 10)

            banks = _three_validation_banks()
            base = _PromotionBase()
            store = CandidateStore(Path(td) / "bank_c.jsonl")
            promoted = promote(store, base, banks)
            self.assertEqual(promoted.status, "promoted")
            banks.final_reference.loss_std_limit = 1.0e-6
            certified = certify_candidate_with_bank_c(
                env=types.SimpleNamespace(base=base),
                promotion_base_env=base,
                candidate_store=store,
                identity_context=context,
                candidate={
                    "candidate_key": promoted.evidence.candidate_key,
                    "full_vector": action,
                    "action_matrix": action_matrix,
                    "boosted_overrides": {},
                    "reward": 1.0,
                },
                bootstrap_seed=99,
                assess_candidate_fn=lambda *_args, **_kwargs: _assessment(0.99),
                validation_banks=banks,
            )
            self.assertEqual(certified.status, "bank_c_point_failed")
            self.assertEqual(len(base.evaluate_calls), 15)

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
                if row.get("record_type") == "candidate_promotion_status_v2"
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
        self.assertEqual(online_base.clear_calls, 2)
        self.assertEqual(full_base.clear_calls, 1)
        self.assertIsNone(online_base._installed_action_hash)

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
