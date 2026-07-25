from __future__ import annotations

import importlib.util
import unittest
from unittest import mock


def _reference():
    from blb_stage2_rl.statistical_constraints import (
        TrialSeries,
        build_baseline_reference,
    )

    groups = []
    for group_index in range(5):
        offset = 0.0002 * group_index
        groups.append(TrialSeries(
            loss=[0.340 + offset + 0.0001 * trial for trial in range(5)],
            metric1=[0.880 - offset - 0.0001 * trial for trial in range(5)],
            metric2=[0.878 - offset - 0.0001 * trial for trial in range(5)],
            seeds=[100 * group_index + trial for trial in range(5)],
        ))
    return build_baseline_reference(
        groups,
        precision_tolerance=0.001,
        stability_multiplier=2.0,
        bootstrap_samples=512,
        seed=123,
    )


class ProtectedK1PureTests(unittest.TestCase):
    def test_extreme_precision_failure_screens_but_boundary_candidate_fails_open(self):
        from blb_stage2_rl.protected_k1 import decide_protected_k1

        reference = _reference()
        extreme = (
            reference.loss_limit + 5.0 * reference.loss_std,
            reference.metric1_mean,
            reference.metric2_mean,
        )
        screened = decide_protected_k1(
            extreme,
            reference,
            guard_sigma=4.0,
            gate_probability=0.5,
            force_k5=False,
        )
        self.assertTrue(screened.screened)
        self.assertEqual(screened.violating_channels, ("loss",))
        self.assertGreater(screened.worst_precision_z, 4.0)

        boundary = (
            reference.loss_limit + 3.9 * reference.loss_std,
            reference.metric1_mean,
            reference.metric2_mean,
        )
        self.assertFalse(decide_protected_k1(
            boundary,
            reference,
            guard_sigma=4.0,
            gate_probability=0.5,
            force_k5=False,
        ).screened)

    def test_potential_frontier_candidate_always_receives_k5(self):
        from blb_stage2_rl.protected_k1 import decide_protected_k1

        reference = _reference()
        extreme = (
            reference.loss_limit + 10.0 * reference.loss_std,
            reference.metric1_limit,
            reference.metric2_limit,
        )
        decision = decide_protected_k1(
            extreme,
            reference,
            guard_sigma=4.0,
            gate_probability=0.5,
            force_k5=True,
        )
        self.assertFalse(decision.screened)
        self.assertEqual(decision.reason, "protected_by_frontier")

    def test_reject_probability_never_exceeds_the_reward_p1_boundary(self):
        from blb_stage2_rl.protected_k1 import decide_protected_k1
        from blb_stage2_rl.statistical_constraints import ConstraintAssessment

        reference = _reference()
        assessment = ConstraintAssessment(
            loss_precision_probability=0.6,
            metric1_precision_probability=1.0,
            metric2_precision_probability=1.0,
            loss_stability_probability=1.0,
            metric1_stability_probability=1.0,
            metric2_stability_probability=1.0,
            precision_probability=0.6,
            stability_probability=1.0,
            gate_probability=0.8,
            online_precision_pass=False,
            online_stability_pass=True,
        )
        with mock.patch(
                "blb_stage2_rl.protected_k1.assess_single_trial_precision",
                return_value=assessment,
        ):
            decision = decide_protected_k1(
                (
                    reference.loss_limit + 5.0 * reference.loss_std,
                    reference.metric1_mean,
                    reference.metric2_mean,
                ),
                reference,
                guard_sigma=4.0,
                gate_probability=0.8,
                force_k5=False,
            )
        self.assertFalse(decision.screened)
        self.assertEqual(decision.reason, "bootstrap_fail_open")

    def test_single_trial_assessment_does_not_claim_measured_stability(self):
        from blb_stage2_rl.protected_k1 import assess_single_trial_precision

        reference = _reference()
        assessment = assess_single_trial_precision(
            (
                reference.loss_limit + 5.0 * reference.loss_std,
                reference.metric1_mean,
                reference.metric2_mean,
            ),
            reference,
            gate_probability=0.5,
        )
        self.assertLess(assessment.precision_probability, 0.5)
        self.assertEqual(assessment.loss_stability_probability, 1.0)
        self.assertEqual(assessment.metric1_stability_probability, 1.0)
        self.assertEqual(assessment.metric2_stability_probability, 1.0)

    def test_audit_selection_is_deterministic_and_fractional(self):
        from blb_stage2_rl.protected_k1 import should_audit_protected_k1

        first = [
            should_audit_protected_k1(42, episode, 0.10)
            for episode in range(1000)
        ]
        second = [
            should_audit_protected_k1(42, episode, 0.10)
            for episode in range(1000)
        ]
        self.assertEqual(first, second)
        self.assertGreater(sum(first), 50)
        self.assertLess(sum(first), 150)
        self.assertFalse(should_audit_protected_k1(42, 0, 0.0))
        self.assertTrue(should_audit_protected_k1(42, 0, 1.0))

    def test_config_defaults_are_safe_and_validated(self):
        from blb_stage2_rl.protected_k1 import ProtectedK1Config

        config = ProtectedK1Config()
        self.assertFalse(config.enabled)
        self.assertEqual(config.guard_sigma, 4.0)
        self.assertEqual(config.audit_fraction, 0.02)
        with self.assertRaises(ValueError):
            ProtectedK1Config(guard_sigma=0.0)
        with self.assertRaises(ValueError):
            ProtectedK1Config(audit_fraction=1.1)


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "torch is required to import probe_runner",
)
class ProbeRunnerTrialSubsetTests(unittest.TestCase):
    def test_explicit_trial_subset_preserves_trial_indices_and_seed_order(self):
        from blb_stage2_rl.probe_runner import ProbeRunner
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

        class Worker:
            def __init__(self, name):
                self.device = name
                self.probe_batches = (object(),)
                self.installed = []
                self.calls = []

            def install(self, decoded):
                self.installed.append(decoded)

            def run_trial(self, trial_index, base_seed, _batch_set_key):
                self.calls.append((trial_index, base_seed))
                return (
                    float(trial_index),
                    float(base_seed),
                    float(derive_probe_trial_seed(base_seed, trial_index)),
                )

        workers = [Worker("cuda:0"), Worker("cuda:1")]
        runner = ProbeRunner(workers)
        results = runner.run_action_trial_groups_at_indices(
            ["a", "b"],
            base_seeds=[100, 200],
            trial_indices=[1, 2, 3, 4],
        )

        self.assertEqual(
            [[int(row[0]) for row in action] for action in results],
            [[1, 2, 3, 4], [1, 2, 3, 4]],
        )
        self.assertEqual(
            runner.last_diagnostics.per_worker_action_trial_indices,
            [[(0, 1), (0, 3), (1, 1), (1, 3)],
             [(0, 2), (0, 4), (1, 2), (1, 4)]],
        )
        self.assertEqual(runner.last_diagnostics.trials_per_action, 4)
        expected_seeds = {
            derive_probe_trial_seed(base_seed, trial_index)
            for base_seed in (100, 200)
            for trial_index in (1, 2, 3, 4)
        }
        observed_seeds = {
            seed
            for worker_seeds in runner.last_diagnostics.per_worker_trial_seeds
            for seed in worker_seeds
        }
        self.assertEqual(observed_seeds, expected_seeds)

    def test_two_stage_env_preserves_k5_indices_and_audits_false_negatives(self):
        import numpy as np

        from blb_stage2_rl.env import BLBStage2Env
        from blb_stage2_rl.probe_runner import ProbeRunnerDiagnostics
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

        reference = _reference()

        class Runner:
            def __init__(self):
                self.calls = []
                self.last_diagnostics = None

            def run_action_trial_groups_at_indices(
                    self, actions, *, base_seeds, trial_indices,
            ):
                indices = list(trial_indices)
                self.calls.append((list(actions), list(base_seeds), indices))
                tasks = [
                    (action_index, trial_index)
                    for action_index in range(len(actions))
                    for trial_index in indices
                ]
                self.last_diagnostics = ProbeRunnerDiagnostics(
                    k=len(tasks),
                    wall_seconds=float(len(tasks)),
                    per_worker_seconds=[float(len(tasks))],
                    per_worker_trial_counts=[len(tasks)],
                    per_worker_trial_indices=[list(range(len(tasks)))],
                    per_worker_trial_seeds=[[
                        derive_probe_trial_seed(
                            base_seeds[action_index], trial_index,
                        )
                        for action_index, trial_index in tasks
                    ]],
                    devices=["cuda:0"],
                    multi_action=True,
                    action_count=len(actions),
                    trials_per_action=len(indices),
                    per_worker_action_trial_indices=[tasks],
                )
                output = []
                for action in actions:
                    rows = []
                    for trial_index in indices:
                        if action == "bad" and trial_index == 0:
                            rows.append((
                                reference.loss_limit + 5.0 * reference.loss_std,
                                reference.metric1_mean,
                                reference.metric2_mean,
                            ))
                        else:
                            rows.append((
                                reference.loss_mean,
                                reference.metric1_mean,
                                reference.metric2_mean,
                            ))
                    output.append(rows)
                return output

        class Env(BLBStage2Env):
            def __init__(self):
                self.probe_runner = Runner()
                self.statistical_reference = reference
                self.statistical_gate_probability = 0.5
                self._probe_eval_counter = 0
                self._installed_config_fingerprint = None

            def _finish_prepared_terminal_probe(
                    self, _prepared, metrics, *,
                    probe_diagnostics=None,
                    forward_ran=True,
                    eval_error=None,
                    constraint_assessment_override=None,
            ):
                del eval_error
                priority = 1 if constraint_assessment_override is not None else 3
                info = {
                    "reward_breakdown": {"priority": priority},
                    "statistical_trials": {
                        "loss": list(metrics.loss_trials),
                        "metric1": list(metrics.metric1_trials),
                        "metric2": list(metrics.metric2_trials),
                        "seeds": list(metrics.trial_seeds),
                    },
                    "metrics": metrics,
                    "invalid": False,
                    "forward_ran": bool(forward_ran),
                    "probe_diagnostics": dict(probe_diagnostics or {}),
                }
                return np.zeros(1), float(priority), True, info

        env = Env()
        prepared = [
            {
                "decoded": action,
                "probe_base_seed": 100 + index,
                "requires_forward": True,
            }
            for index, action in enumerate(("bad", "good", "bad"))
        ]
        results = env.evaluate_prepared_terminal_batch_protected_k1(
            prepared,
            absolute_episodes=[10, 11, 12],
            force_k5_mask=[False, False, True],
            guard_sigma=4.0,
            audit_fraction=0.0,
            audit_seed=42,
        )
        self.assertEqual(
            env.probe_runner.calls,
            [
                (["bad", "good", "bad"], [100, 101, 102], [0]),
                (["good", "bad"], [101, 102], [1, 2, 3, 4]),
            ],
        )
        self.assertEqual(
            [len(result[3]["statistical_trials"]["loss"]) for result in results],
            [1, 5, 5],
        )
        self.assertTrue(results[0][3]["protected_k1_k1_only_reject"])
        self.assertEqual(
            results[1][3]["statistical_trials"]["seeds"],
            [derive_probe_trial_seed(101, index) for index in range(5)],
        )
        self.assertEqual(
            results[1][3]["probe_diagnostics"]["per_worker_trial_indices"],
            [[0, 1, 2, 3, 4]],
        )
        self.assertEqual(
            results[0][3]["probe_diagnostics"]["per_worker_trial_indices"],
            [[0]],
        )
        self.assertEqual(
            results[1][3]["probe_diagnostics"]["per_worker_trial_counts"],
            [5],
        )
        self.assertEqual(
            results[2][3]["protected_k1_reason"],
            "protected_by_frontier",
        )

        audited_env = Env()
        audited = audited_env.evaluate_prepared_terminal_batch_protected_k1(
            prepared[:1],
            absolute_episodes=[10],
            force_k5_mask=[False],
            guard_sigma=4.0,
            audit_fraction=1.0,
            audit_seed=42,
        )[0][3]
        self.assertTrue(audited["protected_k1_audited"])
        self.assertFalse(audited["protected_k1_k1_only_reject"])
        self.assertTrue(audited["protected_k1_audit_precision_false_negative"])
        self.assertTrue(audited["protected_k1_audit_p3_false_negative"])
        self.assertEqual(len(audited["statistical_trials"]["loss"]), 5)


if __name__ == "__main__":
    unittest.main()
