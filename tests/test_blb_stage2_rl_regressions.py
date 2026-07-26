import csv
import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


class BLBProbeTrialAggregationRegressionTests(unittest.TestCase):
    def test_aggregate_probe_trials_preserves_sanitized_trials_and_unbiased_std(self):
        from blb_stage2_rl.env import BLBStage2Env

        metrics = BLBStage2Env._aggregate_probe_trials(
            object(),
            [float("nan"), 2.0, float("inf")],
            [float("nan"), 0.5, float("inf")],
            [float("-inf"), 0.25, float("nan")],
            trial_seeds=[101, 102, 103],
        )

        self.assertEqual(metrics.loss_trials, (100.0, 2.0, 100.0))
        self.assertEqual(metrics.metric1_trials, (0.0, 0.5, 1.0))
        self.assertEqual(metrics.metric2_trials, (0.0, 0.25, 0.0))
        self.assertEqual(metrics.trial_seeds, (101, 102, 103))
        self.assertAlmostEqual(metrics.loss_std, np.std([100.0, 2.0, 100.0], ddof=1))
        self.assertAlmostEqual(metrics.metric1_std, np.std([0.0, 0.5, 1.0], ddof=1))
        self.assertAlmostEqual(metrics.metric2_std, np.std([0.0, 0.25, 0.0], ddof=1))

        single_trial = BLBStage2Env._aggregate_probe_trials(
            object(), [0.5], [0.8], [0.7], trial_seeds=[104],
        )
        self.assertEqual(single_trial.loss_std, 0.0)
        self.assertEqual(single_trial.metric1_std, 0.0)
        self.assertEqual(single_trial.metric2_std, 0.0)

    def test_metrics_from_trial_results_preserves_sanitized_trials_seeds_and_ddof_one(self):
        from blb_stage2_rl.env import BLBStage2Env

        metrics = BLBStage2Env._metrics_from_trial_results(
            [(float("inf"), float("nan"), 0.4), (2.0, 0.6, float("inf"))],
            trial_seeds=[41, 42],
        )

        self.assertEqual(metrics.loss_trials, (100.0, 2.0))
        self.assertEqual(metrics.metric1_trials, (0.0, 0.6))
        self.assertEqual(metrics.metric2_trials, (0.4, 1.0))
        self.assertEqual(metrics.trial_seeds, (41, 42))
        self.assertAlmostEqual(metrics.loss_std, np.std([100.0, 2.0], ddof=1))
        self.assertAlmostEqual(metrics.metric1_std, np.std([0.0, 0.6], ddof=1))
        self.assertAlmostEqual(metrics.metric2_std, np.std([0.4, 1.0], ddof=1))

    def test_trial_aggregation_rejects_empty_explicit_seeds_for_nonempty_trials(self):
        from blb_stage2_rl.env import BLBStage2Env

        with self.assertRaisesRegex(ValueError, "trial_seeds length"):
            BLBStage2Env._aggregate_probe_trials(
                object(), [0.5], [0.8], [0.7], trial_seeds=[],
            )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            BLBStage2Env._aggregate_probe_trials(
                object(), [], [0.8], [], trial_seeds=[],
            )
        empty = BLBStage2Env._aggregate_probe_trials(
            object(), [], [], [], trial_seeds=[],
        )
        self.assertEqual(empty.trial_seeds, ())

    def test_metrics_from_trial_results_rejects_misaligned_seeds(self):
        from blb_stage2_rl.env import BLBStage2Env

        with self.assertRaisesRegex(ValueError, "trial_seeds length"):
            BLBStage2Env._metrics_from_trial_results(
                [(0.5, 0.8, 0.7), (0.6, 0.9, 0.8)],
                trial_seeds=[1],
            )

    def test_probe_runner_reconstructs_trial_seeds_in_trial_index_order(self):
        from blb_stage2_rl.env import BLBStage2Env

        class FakeProbeRunner:
            def __init__(self):
                self.last_diagnostics = SimpleNamespace(
                    k=2,
                    wall_seconds=0.1,
                    per_worker_seconds=[0.1, 0.1],
                    per_worker_trial_counts=[1, 1],
                    per_worker_trial_indices=[[1], [0]],
                    per_worker_trial_seeds=[[701], [700]],
                    devices=["cuda:0", "cuda:1"],
                    speedup_vs_sequential=2.0,
                )

            def run_trials(self, k, *, base_seed):
                self.base_seed = base_seed
                self.k = k
                return [(0.2, 0.8, 0.7), (0.3, 0.9, 0.8)]

        env = BLBStage2Env.__new__(BLBStage2Env)
        env.probe_runner = FakeProbeRunner()
        env.probe_noise_seed = None
        env._probe_eval_counter = 0
        env._derive_probe_base_seed = lambda: 99
        env._last_probe_diagnostics = {}

        metrics = env._eval_on_probe(2)

        self.assertEqual(metrics.trial_seeds, (700, 701))

    def test_fast_multi_action_probe_attaches_each_candidate_seed_in_local_order(self):
        from blb_stage2_rl.env import BLBStage2Env

        class FakeProbeRunner:
            def __init__(self):
                self.last_diagnostics = SimpleNamespace(
                    k=2,
                    wall_seconds=0.1,
                    per_worker_seconds=[0.1, 0.1],
                    per_worker_trial_counts=[1, 1],
                    per_worker_trial_indices=[[1], [0]],
                    per_worker_trial_seeds=[[901], [900]],
                    devices=["cuda:0", "cuda:1"],
                    speedup_vs_sequential=2.0,
                    multi_action=True,
                )

            def run_action_trials_once(self, decoded_by_trial, *, base_seed):
                self.base_seed = base_seed
                return [(0.2, 0.8, 0.7), (0.3, 0.9, 0.8)]

        env = BLBStage2Env.__new__(BLBStage2Env)
        env.probe_runner = FakeProbeRunner()
        env._probe_eval_counter = 0
        env._derive_probe_base_seed = lambda: 123
        env._finish_prepared_terminal_probe = (
            lambda item, metrics, **_kwargs: (None, 0.0, True, {"metrics": metrics})
        )
        prepared = [{"requires_forward": True, "decoded": object()} for _ in range(2)]

        results = env.evaluate_prepared_terminal_batch(prepared, num_trials_per_action=1)

        self.assertEqual(results[0][3]["metrics"].trial_seeds, (900,))
        self.assertEqual(results[1][3]["metrics"].trial_seeds, (901,))
        self.assertEqual(results[0][3]["metrics"].loss_trials, (0.2,))
        self.assertEqual(results[1][3]["metrics"].metric1_trials, (0.9,))

    def test_grouped_k5_probe_preserves_each_episode_seed_and_trial_order(self):
        from blb_stage2_rl.env import BLBStage2Env
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

        class FakeProbeRunner:
            num_workers = 4

            def __init__(self):
                self.calls = []
                self.last_diagnostics = SimpleNamespace(
                    k=10,
                    wall_seconds=0.8,
                    per_worker_seconds=[0.8] * 4,
                    per_worker_trial_counts=[3, 3, 2, 2],
                    per_worker_trial_indices=[[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]],
                    per_worker_trial_seeds=[[], [], [], []],
                    per_worker_action_trial_indices=[
                        [(0, 0), (0, 4), (1, 3)],
                        [(0, 1), (1, 0), (1, 4)],
                        [(0, 2), (1, 1)],
                        [(0, 3), (1, 2)],
                    ],
                    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    speedup_vs_sequential=3.5,
                    multi_action=True,
                    action_count=2,
                    trials_per_action=5,
                )

            def run_action_trial_groups(self, decoded, *, base_seeds, k):
                self.calls.append((list(decoded), list(base_seeds), int(k)))
                return [
                    [(10.0 + trial, 0.80 + trial / 100.0, 0.70) for trial in range(k)],
                    [(20.0 + trial, 0.90 + trial / 100.0, 0.75) for trial in range(k)],
                ]

        env = BLBStage2Env.__new__(BLBStage2Env)
        env.probe_runner = FakeProbeRunner()
        env._probe_eval_counter = 11
        env._step_idx = 0
        env._last_probe_diagnostics = {}
        env._finish_prepared_terminal_probe = (
            lambda item, metrics, **kwargs: (
                None,
                float(item["reward"]),
                True,
                {
                    "metrics": metrics,
                    "probe_diagnostics": kwargs.get("probe_diagnostics"),
                },
            )
        )
        actions = [object(), object()]
        prepared = [
            {
                "requires_forward": True,
                "decoded": actions[0],
                "probe_base_seed": 101,
                "reward": 1.0,
            },
            {
                "requires_forward": True,
                "decoded": actions[1],
                "probe_base_seed": 202,
                "reward": 2.0,
            },
        ]

        results = env.evaluate_prepared_terminal_batch(
            prepared, num_trials_per_action=5,
        )

        self.assertEqual(env.probe_runner.calls, [(actions, [101, 202], 5)])
        self.assertEqual(env._probe_eval_counter, 13)
        self.assertEqual([result[1] for result in results], [1.0, 2.0])
        self.assertEqual(
            results[0][3]["metrics"].trial_seeds,
            tuple(derive_probe_trial_seed(101, idx) for idx in range(5)),
        )
        self.assertEqual(
            results[1][3]["metrics"].trial_seeds,
            tuple(derive_probe_trial_seed(202, idx) for idx in range(5)),
        )
        self.assertEqual(
            results[0][3]["metrics"].loss_trials,
            (10.0, 11.0, 12.0, 13.0, 14.0),
        )
        self.assertEqual(
            results[1][3]["metrics"].metric1_trials,
            tuple(0.90 + trial / 100.0 for trial in range(5)),
        )
        first_diag = results[0][3]["probe_diagnostics"]
        self.assertEqual(first_diag["k"], 5)
        self.assertEqual(first_diag["group_k"], 10)
        self.assertEqual(first_diag["action_count"], 1)
        self.assertEqual(first_diag["group_action_count"], 2)

    def test_grouped_k5_mixed_batch_preserves_order_seed_and_step_index(self):
        from blb_stage2_rl.env import BLBStage2Env
        from blb_stage2_rl.reward import EpisodeMetrics
        from blb_stage2_rl.seed_utils import derive_probe_trial_seed

        class FakeProbeRunner:
            num_workers = 4

            def __init__(self):
                self.grouped_calls = []
                self.install_calls = []
                self.last_diagnostics = SimpleNamespace(
                    k=5,
                    wall_seconds=0.4,
                    per_worker_seconds=[0.4] * 4,
                    per_worker_trial_counts=[2, 1, 1, 1],
                    per_worker_trial_indices=[[0, 4], [1], [2], [3]],
                    per_worker_trial_seeds=[[], [], [], []],
                    per_worker_action_trial_indices=[
                        [(0, 0), (0, 4)],
                        [(0, 1)],
                        [(0, 2)],
                        [(0, 3)],
                    ],
                    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
                    speedup_vs_sequential=3.0,
                    multi_action=True,
                    action_count=1,
                    trials_per_action=5,
                )

            def run_action_trial_groups(self, decoded, *, base_seeds, k):
                self.grouped_calls.append(
                    (list(decoded), list(base_seeds), int(k))
                )
                return [[
                    (30.0 + trial, 0.8, 0.7)
                    for trial in range(int(k))
                ]]

            def install_action(self, decoded):
                self.install_calls.append(decoded)

        env = BLBStage2Env.__new__(BLBStage2Env)
        env.probe_runner = FakeProbeRunner()
        env.probe_noise_seed = 404
        env._probe_eval_counter = 7
        env._step_idx = 17
        env._installed_config_fingerprint = None
        env._last_probe_diagnostics = {}
        env.env_cfg = SimpleNamespace(persistent_probe_install=True)
        env._placeholder_metrics_for_invalid = lambda: EpisodeMetrics()
        env._eval_on_probe = lambda k: EpisodeMetrics(
            trial_seeds=tuple(999 for _ in range(int(k)))
        )
        finish_order = []

        def finish(item, metrics, **kwargs):
            finish_order.append((
                item["label"],
                int(env._step_idx),
                bool(kwargs.get("forward_ran")),
                tuple(metrics.trial_seeds),
            ))
            env._step_idx += 1
            return (
                None,
                float(item["reward"]),
                True,
                {
                    "metrics": metrics,
                    "probe_diagnostics": kwargs.get("probe_diagnostics"),
                },
            )

        env._finish_prepared_terminal_probe = finish
        decoded = object()
        prepared = [
            {
                "label": "invalid-0",
                "reward": 0.0,
                "requires_forward": False,
                "_step_idx_before_finish": 17,
            },
            {
                "label": "valid-1",
                "reward": 1.0,
                "requires_forward": True,
                "decoded": decoded,
                "probe_base_seed": 202,
                "final_config_fingerprint": "valid-1",
                "info": {"timing": {}},
                "_step_idx_before_finish": 17,
            },
            {
                "label": "invalid-2",
                "reward": 2.0,
                "requires_forward": False,
                "_step_idx_before_finish": 17,
            },
            {
                "label": "invalid-3",
                "reward": 3.0,
                "requires_forward": False,
                "_step_idx_before_finish": 17,
            },
        ]

        results = env.evaluate_prepared_terminal_batch(
            prepared, num_trials_per_action=5,
        )

        self.assertEqual(
            env.probe_runner.grouped_calls,
            [([decoded], [202], 5)],
        )
        self.assertEqual(env.probe_runner.install_calls, [])
        self.assertEqual(
            [item[0] for item in finish_order],
            ["invalid-0", "valid-1", "invalid-2", "invalid-3"],
        )
        self.assertEqual([item[1] for item in finish_order], [17, 18, 19, 20])
        self.assertEqual(env._step_idx, 21)
        expected_seeds = tuple(
            derive_probe_trial_seed(202, trial)
            for trial in range(5)
        )
        self.assertEqual(finish_order[1][3], expected_seeds)
        self.assertEqual(results[1][3]["metrics"].trial_seeds, expected_seeds)
        valid_diag = results[1][3]["probe_diagnostics"]
        self.assertEqual(valid_diag["k"], 5)
        self.assertEqual(valid_diag["group_k"], 5)
        self.assertEqual(valid_diag["action_count"], 1)
        self.assertEqual(valid_diag["group_action_count"], 1)


@contextlib.contextmanager
def _workspace_tempdir():
    root = Path(__file__).resolve().parents[1] / "tmp_tests"
    root.mkdir(exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class BLBInstallLogRegressionTests(unittest.TestCase):
    def test_blb_install_log_helper_defaults_quiet(self):
        import function_handler

        previous = os.environ.pop("BLB_NOISE_INSTALL_LOGS", None)
        try:
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                function_handler._print_blb_install("hidden by default")
            self.assertEqual(quiet.getvalue(), "")
        finally:
            if previous is not None:
                os.environ["BLB_NOISE_INSTALL_LOGS"] = previous

    def test_blb_install_log_helper_respects_quiet_env(self):
        import function_handler

        previous = os.environ.get("BLB_NOISE_INSTALL_LOGS")
        try:
            os.environ["BLB_NOISE_INSTALL_LOGS"] = "0"
            quiet = io.StringIO()
            with contextlib.redirect_stdout(quiet):
                function_handler._print_blb_install("hidden")
            self.assertEqual(quiet.getvalue(), "")

            os.environ["BLB_NOISE_INSTALL_LOGS"] = "1"
            loud = io.StringIO()
            with contextlib.redirect_stdout(loud):
                function_handler._print_blb_install("shown")
            self.assertEqual(loud.getvalue(), "shown\n")
        finally:
            if previous is None:
                os.environ.pop("BLB_NOISE_INSTALL_LOGS", None)
            else:
                os.environ["BLB_NOISE_INSTALL_LOGS"] = previous


class BLBActionFinalEvalRegressionTests(unittest.TestCase):
    def test_clean_baseline_reuses_single_configuration_install_without_eval_cache(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        class FakeEvaluator:
            total_layers = 1
            dataset_key = "mrpc"
            layers_attribute = "encoder.layer"

            def __init__(self):
                self.calls = 0
                self.applied = []
                self.dataloaders = {"validation_full": object()}

            def apply_configuration(self, gelu, softmax):
                self.applied.append((tuple(gelu), tuple(softmax)))

            def _resolve_eval_split(self, *, use_train, split):
                if use_train:
                    raise AssertionError("clean baseline final-eval must use validation")
                if split != "validation_full":
                    raise AssertionError(f"unexpected split: {split}")
                return "validation_full"

            def _run_evaluation(self, _loader, *, use_train, split_name):
                if use_train:
                    raise AssertionError("clean baseline final-eval must use validation")
                if split_name != "validation_full":
                    raise AssertionError(f"unexpected split: {split_name}")
                self.calls += 1
                return (
                    0.30 + (0.01 * self.calls),
                    0.80 + (0.01 * self.calls),
                    0.70 + (0.02 * self.calls),
                    10.0 * self.calls,
                )

        runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        runner.evaluator = FakeEvaluator()
        runner.repeat_n = 3
        clear_calls = []
        runner._clear_all_noise = lambda: clear_calls.append("clear")

        result = runner._evaluate_clean_baseline(
            baseline_stage1_gelu=np.asarray([1], dtype=int),
            baseline_stage1_softmax=np.asarray([6], dtype=int),
        )

        self.assertEqual(runner.evaluator.calls, 3)
        self.assertEqual(runner.evaluator.applied, [((1,), (6,))])
        self.assertEqual(clear_calls, ["clear"])
        self.assertEqual(result["evaluation_n"], 3)
        self.assertEqual(result["evaluation_protocol"], "repeated_mean_n=3")
        self.assertAlmostEqual(result["loss"], 0.32)
        self.assertAlmostEqual(result["p"], 0.82)
        self.assertAlmostEqual(result["s"], 0.74)
        self.assertGreater(result["loss_std"], 0.0)
        self.assertIn("repeat_evaluation", result)

    def test_blb_repeat_reuses_single_bridge_install_without_eval_cache(self):
        import Paean.blb_action_eval as mod
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        class FakeBridge:
            instances = []

            def __init__(self, _handler, *, layers_attribute):
                self.layers_attribute = layers_attribute
                self.apply_calls = 0
                self.clear_calls = 0
                FakeBridge.instances.append(self)

            def apply(self, **_kwargs):
                self.apply_calls += 1

            def clear(self):
                self.clear_calls += 1

        class FakeDecoded:
            block1_cfgs = {}
            block2_cfgs = {}
            block3_cfgs = {}
            block4_cfgs = {}
            block5_cfgs = {}

        class FakeEvaluator:
            total_layers = 1
            dataset_key = "mrpc"
            layers_attribute = "encoder.layer"

            def __init__(self):
                self.reversible_handler = object()
                self.dataloaders = {"validation_full": object()}
                self.apply_calls = []
                self.eval_calls = 0

            def apply_configuration(self, gelu, softmax):
                self.apply_calls.append((tuple(gelu), tuple(softmax)))

            def _resolve_eval_split(self, *, use_train, split):
                if use_train:
                    raise AssertionError("BLB action final-eval must use validation")
                if split != "validation_full":
                    raise AssertionError(f"unexpected split: {split}")
                return "validation_full"

            def _run_evaluation(self, _loader, *, use_train, split_name):
                if use_train:
                    raise AssertionError("BLB action final-eval must use validation")
                if split_name != "validation_full":
                    raise AssertionError(f"unexpected split: {split_name}")
                self.eval_calls += 1
                return (
                    0.30 + (0.01 * self.eval_calls),
                    0.80 + (0.01 * self.eval_calls),
                    0.70 + (0.02 * self.eval_calls),
                    10.0 * self.eval_calls,
                )

        old_bridge = mod.BLBNoiseRLBridge
        try:
            mod.BLBNoiseRLBridge = FakeBridge
            runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
            runner.evaluator = FakeEvaluator()
            runner.repeat_n = 3
            legacy_clear_calls = []
            all_clear_calls = []
            verify_calls = []
            runner._clear_legacy_noise = lambda: legacy_clear_calls.append("legacy")
            runner._clear_all_noise = lambda: all_clear_calls.append("all")
            runner._verify_model_installation = (
                lambda _bridge, _decoded: verify_calls.append("verify") or {"ok": True}
            )

            single, repeat = runner._run_blb_eval(
                FakeDecoded(),
                gelu=np.asarray([1], dtype=int),
                softmax=np.asarray([6], dtype=int),
            )
        finally:
            mod.BLBNoiseRLBridge = old_bridge

        self.assertEqual(runner.evaluator.eval_calls, 3)
        self.assertEqual(runner.evaluator.apply_calls, [((1,), (6,))])
        self.assertEqual(len(FakeBridge.instances), 1)
        self.assertEqual(FakeBridge.instances[0].apply_calls, 1)
        self.assertEqual(FakeBridge.instances[0].clear_calls, 1)
        self.assertEqual(legacy_clear_calls, ["legacy"])
        self.assertEqual(all_clear_calls, ["all"])
        self.assertEqual(verify_calls, ["verify"])
        self.assertAlmostEqual(single["loss"], 0.32)
        self.assertEqual(repeat["stats"]["n"], 3)
        self.assertEqual(single["install_verification"], {"ok": True})

    def test_final_eval_has_no_profile_only_max_sfs_loader(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        self.assertFalse(hasattr(BLBActionFinalEvaluationModule, "_load_max_sfs"))

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

    def test_action_candidate_applies_replan_cfg_before_model_forward(self):
        import Paean.blb_action_eval as mod
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule
        from rescale_optimizer_bridge import RescaleOptimizerOutput

        class FakeCfg:
            def __init__(self):
                self.marker = "action_decoded"

        class FakeDecoded:
            def __init__(self):
                self.first_input_sf = 0
                self.block1_cfgs = {}
                self.block2_cfgs = {0: FakeCfg()}
                self.block3_cfgs = {}
                self.block4_cfgs = {}
                self.block5_cfgs = {}

            def cfgs_dict(self):
                return {
                    "block1": self.block1_cfgs,
                    "block2": self.block2_cfgs,
                    "block3": self.block3_cfgs,
                    "block4": self.block4_cfgs,
                    "block5": self.block5_cfgs,
                }

        class FakeEvaluator:
            total_layers = 1
            dataset_key = "mrpc"

            def get_simulated_cost(self, _gelu, _softmax):
                return 10.0, 4.0, 6.0

        fake_decoded = FakeDecoded()
        signals = type(
            "Signals",
            (),
            {
                "any_invalid": False,
                "total_bits_sum": 237,
                "total_fusion_count": 1,
                "invalid_block_count": 0,
                "valid_block_count": 1,
            },
        )()
        outputs = {
            "block2_mrpc_L0": RescaleOptimizerOutput(
                config_name="block2_mrpc_L0",
                fusion_count=1,
                total_bits=237,
                invalid_chain=None,
                valid=True,
                raw={"result": {"valid": True}, "new_compact_config": {"unit": True}},
            )
        }

        old_action_vector_to_cfgs = mod.action_vector_to_cfgs
        old_materialize_decoded_action = mod.materialize_decoded_action
        old_avg_truncation_k_in_action = mod.avg_truncation_k_in_action
        expected_outputs = outputs
        try:
            mod.action_vector_to_cfgs = lambda **_kwargs: fake_decoded
            mod.avg_truncation_k_in_action = lambda *_args, **_kwargs: 13.0

            def fake_materialize_decoded_action(
                    *, decoded, cfgs_dict, outputs, **_kwargs,
                    ):
                cfg = cfgs_dict["block2"][0]
                self.assertEqual(cfg.marker, "action_decoded")
                self.assertIs(outputs, expected_outputs)
                cfg.marker = "replan_applied"
                return SimpleNamespace(
                    decoded=decoded,
                    failure_reason=None,
                    final_config_fingerprint="materialized-test-config",
                    optimizer_invalid=False,
                    replan_application={
                        "applied_before_forward": True,
                        "model_uses_replan_config": True,
                        "expected_config_count": 1,
                        "applied_config_count": 1,
                        "invalid_config_count": 0,
                        "missing_compact_config_count": 0,
                        "missing_decoded_cfg_count": 0,
                        "apply_error_count": 0,
                        "override_total": 1,
                        "per_config": {},
                        "optimizer_cfg_overrides": {},
                    },
                )

            mod.materialize_decoded_action = fake_materialize_decoded_action

            runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
            runner.evaluator = FakeEvaluator()
            runner.repeat_n = 1
            runner.rescale_optimizer_mode = "cfg_derived"
            runner.rescale_invoker_kind = "stub"
            runner.rescale_optimizer_root = ""
            runner.rescale_bridge = type(
                "Bridge",
                (),
                {"invoker": type("Invoker", (), {"baselines": {"block2_mrpc": ([0], [], [])}})()},
            )()
            runner._optimizer_outputs = lambda _profile, _cfgs_dict: (outputs, signals)
            runner._config_details = lambda decoded, *_args: {
                "marker_seen_by_details": decoded.block2_cfgs[0].marker
            }
            runner._build_feasibility_report = lambda **kwargs: {
                "feasible": bool(kwargs["apply_ok"]),
                "diagnostic_feasible": bool(kwargs["apply_ok"]),
                "strict_feasible": bool(kwargs["apply_ok"]),
            }

            marker_seen_by_forward = {}

            def fake_run_blb_eval(decoded, **_kwargs):
                marker_seen_by_forward["value"] = decoded.block2_cfgs[0].marker
                return (
                    {
                        "loss": 0.25,
                        "p": 0.875,
                        "s": 0.8,
                        "time_ms": 12.0,
                        "install_verification": {"ok": True},
                    },
                    None,
                )

            runner._run_blb_eval = fake_run_blb_eval

            result = runner._evaluate_action_candidate(
                name="candidate",
                action_vec=np.zeros(1, dtype=int),
                overrides={},
                gelu=np.ones(1, dtype=int),
                softmax=np.ones(1, dtype=int) * 2,
                report_constraints={},
                max_sfs={},
            )
        finally:
            mod.action_vector_to_cfgs = old_action_vector_to_cfgs
            mod.materialize_decoded_action = old_materialize_decoded_action
            mod.avg_truncation_k_in_action = old_avg_truncation_k_in_action

        self.assertEqual(marker_seen_by_forward["value"], "replan_applied")
        self.assertTrue(result["replan_application"]["model_uses_replan_config"])
        self.assertEqual(result["replan_application"]["applied_config_count"], 1)
        self.assertEqual(result["config_details"]["marker_seen_by_details"], "replan_applied")

    def test_model_installation_verification_matches_current_blb_bridge_semantics(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        class FakeDecoded:
            pass

        class FakeHandler:
            def __init__(self, decoded):
                self.block1_cfg_per_layer = dict(decoded.block1_cfgs)
                self.block2_cfg_per_layer = dict(decoded.block2_cfgs)
                self.block3_cfg_per_layer = dict(decoded.block3_cfgs)
                self.block4_cfg_per_layer = dict(decoded.block4_cfgs)
                self.block5_cfg_per_layer = dict(decoded.block5_cfgs)

            def get_active_blb_noise_layers(self):
                return {
                    "block1": {0, 1},
                    "block2": {0, 1},
                    "block3": {0, 1},
                    "block4": {0, 1},
                    "block5": {0, 1},
                    "first_input": set(),
                }

        class FakeBridge:
            def installed_layers(self):
                return {
                    0: {"block1", "block2", "block3", "block4", "block5"},
                    1: {"block1", "block2", "block3", "block4", "block5"},
                }

        decoded = FakeDecoded()
        decoded.block1_cfgs = {0: object(), 1: object()}
        decoded.block2_cfgs = {0: object(), 1: object()}
        decoded.block3_cfgs = {0: object(), 1: object()}
        decoded.block4_cfgs = {0: object(), 1: object()}
        decoded.block5_cfgs = {0: object(), 1: object()}

        runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        runner.evaluator = type(
            "Evaluator",
            (),
            {
                "total_layers": 2,
                "reversible_handler": FakeHandler(decoded),
            },
        )()

        result = runner._verify_model_installation(FakeBridge(), decoded)

        self.assertTrue(result["model_will_use_selected_cfg"])
        self.assertEqual(result["expected_active_layers"]["block1"], [0, 1])
        self.assertEqual(result["expected_active_layers"]["block3"], [0, 1])
        self.assertEqual(result["expected_active_layers"]["first_input"], [])

    def test_action_candidate_skips_model_forward_when_optimizer_invalid(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule
        from blb_stage2_rl.action_space import load_max_sfs, make_all_max_action_vector

        class FakeEvaluator:
            total_layers = 1
            dataset_key = "mrpc"

            def get_simulated_cost(self, _gelu, _softmax):
                return 10.0, 4.0, 6.0

        signals = type(
            "Signals",
            (),
            {
                "any_invalid": True,
                "total_bits_sum": 0,
                "total_fusion_count": 0,
                "invalid_block_count": 1,
                "valid_block_count": 0,
            },
        )()

        runner = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
        runner.evaluator = FakeEvaluator()
        runner._optimizer_outputs = lambda _profile, _cfgs_dict: ({}, signals)
        runner._config_details = lambda *_args, **_kwargs: {}
        runner._is_feasible = lambda *_args, **_kwargs: True

        def fail_if_forwarded(*_args, **_kwargs):
            raise AssertionError("invalid optimizer output must skip model forward")

        runner._run_blb_eval = fail_if_forwarded

        result = runner._evaluate_action_candidate(
            name="candidate",
            action_vec=make_all_max_action_vector(num_layers=1),
            overrides={},
            gelu=np.ones(1, dtype=int),
            softmax=np.ones(1, dtype=int) * 2,
            report_constraints={},
            max_sfs=load_max_sfs("mrpc"),
        )

        self.assertTrue(result["any_invalid"])
        self.assertTrue(result["skipped_forward"])
        self.assertEqual(result["forward_skipped_reason"], "optimizer_invalid_chain")
        self.assertFalse(result["feasible"])
        self.assertEqual(result["evaluation_n"], 0)
        self.assertEqual(result["p"], 0.0)
        self.assertTrue(result["install_verification"]["skipped"])

    def test_full_noise_config_excludes_deprecated_first_input(self):
        from Paean.blb_action_eval import BLBActionFinalEvaluationModule

        decoded = type(
            "Decoded",
            (),
            {
                "first_input_sf": 30,
                "block1_cfgs": {},
                "block2_cfgs": {},
                "block3_cfgs": {},
                "block4_cfgs": {},
                "block5_cfgs": {},
            },
        )()

        config = BLBActionFinalEvaluationModule._full_noise_config(decoded)

        self.assertFalse(
            any(entry.get("block") == "first_input" for entry in config["entries"])
        )


class BLBPolicyWarmstartRegressionTests(unittest.TestCase):
    def test_kind_drop_action_changes_only_target_kind_slots(self):
        from blb_stage2_rl.action_space import (
            action_dims_for_config,
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
        )
        from blb_stage2_rl.runner import _build_kind_drop_action

        baseline = make_all_max_action_vector(num_layers=2)
        desc = describe_action_vector(
            baseline,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=2,
            gelu_degree=[1, 4],
            attn_degree=[2, 5],
            profile="mrpc",
        )
        records = list(desc["records"])
        action, touched = _build_kind_drop_action(
            baseline,
            records,
            action_dims_for_config(2),
            kinds=("M", "S"),
            radius=1,
        )

        self.assertGreater(len(touched), 0)
        self.assertFalse(np.array_equal(action, baseline))
        for idx, (actual, expected) in enumerate(zip(action.tolist(), baseline.tolist())):
            if idx in touched:
                self.assertIn(records[idx]["kind"], ("M", "S"))
                self.assertLess(actual, expected)
            else:
                self.assertEqual(actual, expected)

    def test_warmstart_action_mode_expires_by_absolute_episode(self):
        from blb_stage2_rl.runner import _warmstart_action_mode

        kwargs = {
            "anchor_episodes": 30,
            "cost_probe_count": 4,
            "neighbor_sampling": True,
            "has_mutable_neighbors": True,
            "neighbor_ramp_episodes": 1200,
        }

        self.assertEqual(
            _warmstart_action_mode(episode_index=0, **kwargs),
            ("anchor", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=30, **kwargs),
            ("cost_probe", 0),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=34, **kwargs),
            ("neighbor", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=1199, **kwargs),
            ("neighbor", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=1200, **kwargs),
            ("policy", -1),
        )
        self.assertEqual(
            _warmstart_action_mode(episode_index=50000, **kwargs),
            ("policy", -1),
        )

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

        expected = preferred.copy()
        # The deprecated first_input tail is a fixed compatibility placeholder,
        # not a sampled policy head.
        expected[-1] = 0
        self.assertEqual(action.squeeze(0).tolist(), expected.tolist())


class BLBTraceWriterRegressionTests(unittest.TestCase):
    def test_stage2_noise_log_formatters_use_readable_chinese(self):
        from blb_stage2_rl.runner import (
            _format_blb_best_log,
            _format_blb_episode_error_log,
            _format_blb_rollout_summary_log,
            _format_blb_train_iter_log,
            _format_warmstart_cost_probe_log,
        )

        error_text = (
            "Rescale_optimizer invalid blocks: "
            "block5_n1_L7: new chain has prime(s) > q_max=60 at stage(s) [1]; "
            "fusion cannot reduce. Reject.; "
            "block3_exp_n2_L0: replan FAILED: a prime < q_min=30 could not be fused "
            "after 0 successful fusion(s)."
        )
        formatted_error = _format_blb_episode_error_log(485, error_text)

        self.assertIn("【BLB 单回合错误】", formatted_error)
        self.assertIn("回合（episode）：485", formatted_error)
        self.assertIn("失败位置", formatted_error)
        self.assertIn("block5_n1_L7", formatted_error)
        self.assertIn("block3_exp_n2_L0", formatted_error)
        self.assertIn("新模数链", formatted_error)
        self.assertIn("重新规划（replan）失败", formatted_error)
        self.assertNotIn("[BLB episode error]", formatted_error)
        self.assertNotIn("new chain has prime(s)", formatted_error)

        formatted_rollout = _format_blb_rollout_summary_log(
            update_count=5,
            episode=600,
            total_episodes=80000,
            reward_mean=-36.691,
            reward_max=-3.400,
            reward_min=-100.188,
            invalid_count=11,
            priority_counts={1: 1, 2: 77, 3: 31},
            anchor_count=0,
            cost_probe_count=0,
            neighborhood_count=120,
            policy_loss=0.0128,
            value_loss=1711.9232,
            entropy=1071.4326,
            clip_fraction=0.25,
            entropy_by_kind={"F": 1.44, "W": 1.48, "M": 0.92},
        )

        self.assertIn("【BLB Rollout 汇总】", formatted_rollout)
        self.assertIn("训练进度（episode）：600 / 80000", formatted_rollout)
        self.assertIn("奖励（reward，均值 / 最大 / 最小）：-36.691 / -3.400 / -100.188", formatted_rollout)
        self.assertIn("优先级计数（P0/P1/P2/P3）", formatted_rollout)
        self.assertIn("P0 无效=11", formatted_rollout)
        self.assertIn("动作来源（A/C/N）", formatted_rollout)
        self.assertIn("槽位熵（H_kind）", formatted_rollout)
        self.assertIn("F=1.44", formatted_rollout)

        formatted_probes = _format_warmstart_cost_probe_log(
            [("drop_kind_M", object(), [1, 2]), ("drop_kind_MS", object(), [3])]
        )
        self.assertIn("预热（warmstart）成本探针", formatted_probes)
        self.assertIn("drop_kind_M：影响槽位 2 个", formatted_probes)

        formatted_best = _format_blb_best_log(
            episode=31,
            best_reward=2.7333,
            previous_reward_label="0.0000",
            priority=3,
            diff_text="L0.B2.M.gamma 2->1; L0.B2.M.q_mask1 2->1",
        )
        self.assertIn("【BLB 新最佳】", formatted_best)
        self.assertIn("回合（episode）：31", formatted_best)
        self.assertIn("当前奖励（reward）：2.7333", formatted_best)
        self.assertIn("上一最佳奖励：0.0000", formatted_best)
        self.assertIn("变化位置", formatted_best)
        self.assertIn("L0.B2.M.gamma", formatted_best)
        self.assertNotIn("[BLB best]", formatted_best)

        formatted_iter = _format_blb_train_iter_log(
            episode=120,
            total_episodes=80000,
            return_mean=-25.817,
            return_max=2.733,
            best_reward=2.733,
            policy_loss=0.2411,
            value_loss=1298.1715,
            entropy=1071.0776,
            clip_fraction=0.8708,
        )
        self.assertIn("【BLB 训练迭代】", formatted_iter)
        self.assertIn("近期回报（return，均值 / 最大）", formatted_iter)
        self.assertIn("best_reward=2.733", formatted_iter)

        evaluator_source = (Path(__file__).resolve().parents[1] / "layer_importance_evaluator.py").read_text(
            encoding="utf-8",
        )
        runner_source = (Path(__file__).resolve().parents[1] / "blb_stage2_rl" / "runner.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("二阶段噪声 RL 日志开始（Stage-2 noise RL log started）", evaluator_source)
        self.assertIn("一阶段 PPO 学习率（Stage-1 PPO LR）", evaluator_source)
        self.assertNotIn("?????? RL", evaluator_source)
        self.assertIn("BLB 单候选安装日志", runner_source)
        self.assertNotIn("per-candidate install logs suppressed", runner_source)

    def test_status_board_publishes_live_top_level_fields(self):
        from blb_stage2_rl.persistence import BLBStatusBoard

        with _workspace_tempdir() as td:
            board = BLBStatusBoard(td, total_episodes=240, profile="mrpc")
            board.set_best(
                best_reward=0.0,
                best_action_vec=[1, 2, 3],
                best_breakdown={"priority": 3, "invalid": False},
                best_episode=0,
            )
            board.update_after_episode(
                120,
                -30.0,
                breakdown={"priority": 0, "invalid": True},
            )
            board.update_after_ppo_update(1, {"policy_loss": 0.25})

            payload = json.loads(Path(board.path).read_text(encoding="utf-8"))

        self.assertEqual(payload["episode"], 120)
        self.assertEqual(payload["completed_episodes"], 120)
        self.assertEqual(payload["best_reward"], 0.0)
        self.assertEqual(payload["best"]["reward"], 0.0)
        self.assertEqual(payload["last_reward"], -30.0)
        self.assertEqual(payload["last_priority"], 0)
        self.assertTrue(payload["last_invalid"])
        self.assertIsNotNone(payload["updated_at"])

    def test_persistence_report_writers_stream_line_outputs(self):
        from blb_stage2_rl import persistence

        with _workspace_tempdir() as td:
            root = Path(td)
            report_paths = {
                str(root / "blb_stage2_best_action_full.md"),
                str(root / persistence.BLB_FINAL_REPORT_MD),
                str(root / persistence.BLB_ERROR_TXT),
            }
            original_open = open

            def fake_open(path, *args, **kwargs):
                path_str = str(path)
                if path_str not in report_paths:
                    return original_open(path, *args, **kwargs)
                handle = mock.MagicMock()
                handle.__enter__.return_value = handle
                handle.__exit__.return_value = None

                def reject_full_document_write(text):
                    if not isinstance(text, str) or text.count("\n") <= 3:
                        return
                    if path_str.endswith(persistence.BLB_ERROR_TXT):
                        if "Traceback:" in text and "Python:" in text:
                            raise AssertionError("persistence reports should stream lines")
                        return
                    else:
                        raise AssertionError("persistence reports should stream lines")

                handle.write.side_effect = reject_full_document_write
                return handle

            with mock.patch("builtins.open", side_effect=fake_open):
                action_paths = persistence.write_action_description_files(
                    td,
                    {
                        "profile": "mrpc",
                        "num_layers": 1,
                        "action_length": 1,
                        "summary": {"record_count": 1},
                        "records": [{
                            "global_index": 0,
                            "slot_label": "L00.B1.F.example",
                            "label": "L00.B1.F.example",
                            "location": "L00.B1",
                            "operation": "example",
                            "kind": "F",
                            "layer": 0,
                            "block": 1,
                            "distribution": "F",
                            "action_index": 1,
                            "effective": True,
                            "effective_value": 12,
                            "scaling_factor": 12,
                            "level_values": [8, 12],
                        }],
                    },
                )
                final_report_path = persistence.write_blb_final_report(
                    td,
                    run_basename="unit",
                    profile="mrpc",
                    total_episodes=10,
                    completed_episodes=10,
                    elapsed_sec=1.0,
                    best_reward=1.0,
                    best_breakdown={"terminal_priority": 3},
                    best_action_vec=[1, 2, 3],
                    baseline={"loss": 0.3},
                    reward_weights={"cost": 1.0},
                    episode_returns=[0.1, 0.2, 0.3],
                    rescale_invoker_kind="unit",
                )
                try:
                    raise RuntimeError("boom")
                except RuntimeError as exc:
                    crash_path = persistence.dump_crash_report(td, exc=exc)

            self.assertEqual(action_paths["md"], str(root / "blb_stage2_best_action_full.md"))
            self.assertEqual(final_report_path, str(root / persistence.BLB_FINAL_REPORT_MD))
            self.assertEqual(crash_path, str(root / persistence.BLB_ERROR_TXT))

    def test_trace_writer_appends_structured_rollout_rows(self):
        from blb_stage2_rl.persistence import append_blb_episode_trace_row

        with _workspace_tempdir() as td:
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

    def test_trace_writer_migrates_old_header_after_cost_probe_column_added(self):
        from blb_stage2_rl.persistence import (
            BLB_EPISODE_TRACE_CSV,
            BLB_TRACE_FIELDNAMES,
            append_blb_episode_trace_row,
        )

        with _workspace_tempdir() as td:
            trace_path = Path(td) / BLB_EPISODE_TRACE_CSV
            old_header = [field for field in BLB_TRACE_FIELDNAMES if field != "cost_probe_count"]
            old_row = {field: "" for field in old_header}
            old_row.update({
                "episode": 120,
                "total_episodes": 80000,
                "ppo_update_count": 1,
                "anchor_count": 30,
                "policy_loss": 0.01,
                "value_loss": 10.0,
                "entropy": 100.0,
                "clip_fraction": 0.1,
                "n_samples": 120,
            })
            malformed_new_row = {field: "" for field in BLB_TRACE_FIELDNAMES}
            malformed_new_row.update({
                "episode": 520,
                "total_episodes": 80000,
                "ppo_update_count": 4,
                "anchor_count": 0,
                "cost_probe_count": 4,
                "policy_loss": 0.02,
                "value_loss": 2437.0,
                "entropy": 1071.0,
                "clip_fraction": 0.744,
                "n_samples": 120,
            })
            with trace_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(old_header)
                writer.writerow([old_row.get(field, "") for field in old_header])
                writer.writerow([malformed_new_row.get(field, "") for field in BLB_TRACE_FIELDNAMES])

            append_blb_episode_trace_row(
                td,
                {
                    "episode": 640,
                    "total_episodes": 80000,
                    "ppo_update_count": 5,
                    "anchor_count": 0,
                    "cost_probe_count": 0,
                    "policy_loss": 0.03,
                    "value_loss": 20.0,
                    "entropy": 90.0,
                    "clip_fraction": 0.2,
                    "n_samples": 120,
                },
            )

            with trace_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(list(rows[0].keys()), list(BLB_TRACE_FIELDNAMES))
        self.assertEqual(rows[0]["cost_probe_count"], "0")
        self.assertEqual(rows[0]["policy_loss"], "0.01")
        self.assertEqual(rows[1]["cost_probe_count"], "4")
        self.assertEqual(rows[1]["policy_loss"], "0.02")
        self.assertEqual(rows[1]["value_loss"], "2437.0")
        self.assertEqual(rows[1]["entropy"], "1071.0")
        self.assertEqual(rows[1]["clip_fraction"], "0.744")
        self.assertEqual(rows[1]["n_samples"], "120")
        self.assertEqual(rows[2]["episode"], "640")

    def test_trace_writer_persists_rollout_eval_diagnostics(self):
        from blb_stage2_rl.persistence import append_blb_episode_trace_row

        with _workspace_tempdir() as td:
            trace_path = append_blb_episode_trace_row(
                td,
                {
                    "episode": 120,
                    "total_episodes": 120,
                    "ppo_update_count": 1,
                    "rollout_reward_mean": -30.0,
                    "rollout_metric1_mean": 0.875,
                    "rollout_metric2_mean": 0.8125,
                    "rollout_loss_mean": 0.341,
                    "rollout_loss_std_mean": 0.002,
                    "apply_error_count": 1,
                    "eval_error_count": 0,
                    "last_error": "BLB apply failed: example",
                    "best_reward": -30.0,
                },
            )

            with Path(trace_path).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertIn("rollout_metric1_mean", rows[0])
        self.assertEqual(rows[0]["rollout_metric1_mean"], "0.875")
        self.assertEqual(rows[0]["apply_error_count"], "1")
        self.assertIn("BLB apply failed", rows[0]["last_error"])


class BLBRewardRegressionTests(unittest.TestCase):
    def test_optimizer_invalid_is_cost_layer_after_accuracy_and_stability(self):
        """Priority semantics are preserved under bounded continuous reward.

        Current Stage-2 reward no longer uses the old +20/+40 tier bonuses by
        default. The regression invariant is the gate order: P1 blocks metric
        feasibility, P2 blocks cost reward, and only P3 can receive cost.
        """
        from blb_stage2_rl.reward import (
            BaselineCostStats,
            EpisodeMetrics,
            RewardWeights,
            compute_reward,
        )

        # --- Case 1: acc fail (well below threshold) + invalid_chain ---
        # metric_ok=False (both acc_violation AND invalid trigger it).
        # priority label = 1 (acc/invalid takes precedence).
        accuracy_breakdown = compute_reward(
            EpisodeMetrics(metric1_mean=0.0, loss_mean=float("inf"), loss_std=float("inf")),
            type("Signals", (), {"any_invalid": True, "total_bits_sum": 0, "total_fusion_count": 0})(),
            action_avg_k=13.0,
            baseline=BaselineCostStats(metric1_mean=0.875),
            weights=RewardWeights(baseline_metric1=0.875),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=True,
        )
        self.assertTrue(accuracy_breakdown.invalid)
        self.assertEqual(accuracy_breakdown.priority, 1)
        self.assertGreater(accuracy_breakdown.acc_violation, 0.0)
        self.assertFalse(accuracy_breakdown.metric_ok)
        self.assertEqual(accuracy_breakdown.tier_bonus, 0.0)
        # Total reward is clipped shaping only (no tier bonus), bounded in
        # the [-5, +5] range so PPO advantages stay well-conditioned.
        self.assertLessEqual(accuracy_breakdown.reward, 5.0)
        self.assertGreaterEqual(accuracy_breakdown.reward, -10.0)

        # --- Case 2: acc OK but optimizer invalid ---
        # invalid is a hard P1 failure under the current reward contract:
        # metric_ok=False, priority=1, tier_bonus=0.
        cost_breakdown = compute_reward(
            EpisodeMetrics(metric1_mean=0.9, loss_mean=0.2, loss_std=0.1),
            type("Signals", (), {"any_invalid": True, "total_bits_sum": 200, "total_fusion_count": 0})(),
            action_avg_k=13.0,
            baseline=BaselineCostStats(total_bits_sum=200, metric1_mean=0.875),
            weights=RewardWeights(baseline_metric1=0.875, invalid_penalty=5.0),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=True,
        )
        self.assertTrue(cost_breakdown.invalid)
        self.assertEqual(cost_breakdown.priority, 1)
        self.assertFalse(cost_breakdown.metric_ok)
        self.assertEqual(cost_breakdown.tier_bonus, 0.0)
        # invalid_penalty contributes -5 to shaping; clipped at -5.
        self.assertEqual(cost_breakdown.invalid_term, -5.0)

        # --- Case 3: nonfinite stability, acc OK, no invalid ---
        # metric_ok=True (acc OK + not invalid), but nonfinite std is a hard P2
        # stability failure. It receives no tier bonus and no cost reward.
        nonfinite_stability = compute_reward(
            EpisodeMetrics(metric1_mean=0.9, loss_mean=0.2, loss_std=float("inf")),
            type("Signals", (), {"any_invalid": False, "total_bits_sum": 200, "total_fusion_count": 0})(),
            action_avg_k=13.0,
            baseline=BaselineCostStats(total_bits_sum=200, metric1_mean=0.875),
            weights=RewardWeights(baseline_metric1=0.875, lambda_stab=5.0),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=False,
        )
        self.assertEqual(nonfinite_stability.priority, 2)
        self.assertTrue(nonfinite_stability.metric_ok)
        self.assertFalse(nonfinite_stability.stab_ok)
        self.assertEqual(nonfinite_stability.tier_bonus, 0.0)
        self.assertEqual(nonfinite_stability.cost_score, 0.0)
        self.assertLessEqual(nonfinite_stability.reward, 0.0)

        # --- Case 4: everything OK → both tier bonuses ---
        all_ok = compute_reward(
            EpisodeMetrics(metric1_mean=0.9, loss_mean=0.2, loss_std=0.5),
            type("Signals", (), {"any_invalid": False, "total_bits_sum": 100, "total_fusion_count": 0})(),
            action_avg_k=11.0,
            baseline=BaselineCostStats(total_bits_sum=200, metric1_mean=0.875, avg_k=13.0),
            weights=RewardWeights(baseline_metric1=0.875),
            acc_threshold=0.865,
            stab_threshold=1.0,
            any_invalid=False,
        )
        self.assertEqual(all_ok.priority, 3)
        self.assertTrue(all_ok.metric_ok)
        self.assertTrue(all_ok.stab_ok)
        self.assertEqual(all_ok.tier_bonus, 0.0)
        self.assertGreater(all_ok.cost_score, 0.0)
        self.assertGreater(all_ok.reward, nonfinite_stability.reward)
        self.assertLessEqual(all_ok.reward, 5.0)

    def test_metric_std_sampling_jitter_does_not_drop_stability_tier(self):
        """5-trial MRPC metric std jitter should not masquerade as instability.

        Reward v3 adds metric1_std / metric2_std to the stability gate. The
        first reward-v3 10k run failed at 345 episodes because normal discrete
        probe jitter repeatedly landed in P2(stab), even though m1/loss were
        healthy and there were no P1/invalid/loss-cap events. The metric-std
        floor keeps tiny sampling swings in the top tier while preserving P2
        for materially unstable metrics.
        """
        from blb_stage2_rl.reward import (
            BaselineCostStats,
            EpisodeMetrics,
            RewardWeights,
            compute_reward,
        )

        signals = type(
            "Signals", (),
            {"any_invalid": False, "total_bits_sum": 200, "total_fusion_count": 0},
        )()
        baseline = BaselineCostStats(
            total_bits_sum=200,
            total_fusion_count=0,
            avg_k=13.0,
            metric1_mean=0.875,
            metric2_mean=0.875,
            metric1_std=0.002,
            metric2_std=0.002,
            loss_std=0.002,
        )
        weights = RewardWeights(baseline_metric1=0.875, baseline_metric2=0.875)

        jitter = compute_reward(
            EpisodeMetrics(
                metric1_mean=0.872,
                metric2_mean=0.872,
                metric1_std=0.006,
                metric2_std=0.006,
                loss_std=0.003,
            ),
            signals,
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
            acc_threshold=0.865,
            acc_threshold_m2=0.865,
            stab_threshold=0.01,
            any_invalid=False,
        )
        self.assertEqual(jitter.priority, 3)
        self.assertTrue(jitter.stab_ok)
        self.assertEqual(jitter.tier_bonus, 0.0)

        material_instability = compute_reward(
            EpisodeMetrics(
                metric1_mean=0.872,
                metric2_mean=0.872,
                metric1_std=0.03,
                metric2_std=0.03,
                loss_std=0.003,
            ),
            signals,
            action_avg_k=13.0,
            baseline=baseline,
            weights=weights,
            acc_threshold=0.865,
            acc_threshold_m2=0.865,
            stab_threshold=0.01,
            any_invalid=False,
        )
        self.assertEqual(material_instability.priority, 2)
        self.assertFalse(material_instability.stab_ok)
        self.assertEqual(material_instability.tier_bonus, 0.0)
        self.assertLess(material_instability.reward, jitter.reward)

    def test_runner_best_selection_uses_hard_constraints_before_reward(self):
        from blb_stage2_rl.runner import is_better_blb_candidate

        invalid_high_reward = {
            "invalid": True,
            "acc_violation": 0.0,
            "stab_violation": 0.0,
        }
        valid_accuracy_failure = {
            "invalid": False,
            "acc_violation": 0.2,
            "stab_violation": 0.0,
        }
        valid_safe_lower_reward = {
            "invalid": False,
            "acc_violation": 0.0,
            "stab_violation": 0.0,
        }
        valid_safe_higher_reward = {
            "invalid": False,
            "acc_violation": 0.0,
            "stab_violation": 0.0,
        }

        self.assertFalse(
            is_better_blb_candidate(
                candidate_reward=-100.0,
                candidate_breakdown=valid_accuracy_failure,
                best_reward=-30.0,
                best_breakdown=invalid_high_reward,
            )
        )
        self.assertTrue(
            is_better_blb_candidate(
                candidate_reward=1.0,
                candidate_breakdown=valid_safe_higher_reward,
                best_reward=0.5,
                best_breakdown=valid_safe_lower_reward,
            )
        )
        self.assertTrue(
            is_better_blb_candidate(
                candidate_reward=-30.0,
                candidate_breakdown=invalid_high_reward,
                best_reward=-100.0,
                best_breakdown=valid_accuracy_failure,
            )
        )

    def test_baseline_preflight_stability_threshold_uses_noisy_baseline(self):
        from blb_stage2_rl.runner import _baseline_preflight_stability_threshold

        widened = _baseline_preflight_stability_threshold(
            current_threshold=0.001,
            observed_loss_std=0.0018,
            tolerance=0.005,
        )

        self.assertGreater(widened, 0.0018)
        self.assertAlmostEqual(
            _baseline_preflight_stability_threshold(
                current_threshold=0.003,
                observed_loss_std=0.0018,
                tolerance=0.005,
            ),
            0.003,
        )

    def test_neighbor_indices_use_real_k_order_not_action_index_order(self):
        from blb_stage2_rl.action_space import K_LEVELS
        from blb_stage2_rl.runner import _allowed_neighbor_indices

        baseline_idx = int(K_LEVELS.index(max(K_LEVELS)))
        allowed = _allowed_neighbor_indices(
            kind="K",
            baseline_idx=baseline_idx,
            dim=len(K_LEVELS),
            radius=1,
        )

        self.assertEqual([K_LEVELS[i] for i in allowed], [13, 12])

    def test_neighborhood_curriculum_expands_slowly(self):
        from blb_stage2_rl.runner import _neighborhood_curriculum

        self.assertEqual(
            _neighborhood_curriculum(
                episode_offset=0,
                ramp_episodes=100,
                max_mutations=8,
                max_radius=2,
            ),
            (1, 1),
        )
        self.assertEqual(
            _neighborhood_curriculum(
                episode_offset=100,
                ramp_episodes=100,
                max_mutations=8,
                max_radius=2,
            ),
            (8, 2),
        )


class BLBOptimizerBaselineRegressionTests(unittest.TestCase):
    def test_bridge_baseline_evaluation_bypasses_cfg_derived_payload(self):
        from rescale_optimizer_bridge import RescaleOptimizerBridge

        class RecordingInvoker:
            def __init__(self):
                self.calls = []

            def __call__(self, config_name, payload):
                self.calls.append((config_name, payload))
                return {
                    "fusion_count": 0,
                    "valid": True,
                    "result": {
                        "valid": True,
                        "chain": {"total_bits": 123},
                        "invalid_chain": None,
                    },
                }

        invoker = RecordingInvoker()
        bridge = RescaleOptimizerBridge(invoker=invoker)

        out = bridge.evaluate_baseline(config_name="block2_mrpc_L7")

        self.assertEqual(invoker.calls, [("block2_mrpc", {})])
        self.assertEqual(out.config_name, "block2_mrpc_L7")
        self.assertTrue(out.valid)
        self.assertEqual(out.total_bits, 123)

    def test_env_all_max_action_uses_optimizer_baseline_scoring(self):
        from blb_stage2_rl.action_space import (
            avg_truncation_k_in_action,
            load_max_sfs,
            make_all_max_action_vector,
        )
        from blb_stage2_rl.env import BLBStage2Env, BLBStage2EnvConfig, ProbeBatch
        from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
        from rescale_optimizer_bridge import RescaleOptimizerOutput

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))

            def forward(self, **_kwargs):
                logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
                return type("Output", (), {"logits": logits})()

        class FakeHandler:
            def __getattr__(self, name):
                if name.startswith(("replace_", "restore_")):
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

        class FakeRescaleBridge:
            def __init__(self):
                self.calls = []
                self.invoker = SimpleNamespace(baselines={
                    "block2_mrpc": ([0], [], []),
                    "block3_exp_n4": ([0], [], []),
                    "block4": ([0], [], []),
                    "block5_n4": ([0], [], []),
                })

            def evaluate_blocks(self, requests):
                self.calls.append("cfg-derived")
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=100,
                        invalid_chain=None,
                        valid=True,
                        raw={
                            "result": {"valid": True},
                            "new_compact_config": {},
                        },
                    )
                    for name in requests
                }

            def evaluate_baseline_blocks(self, requests):
                self.calls.append("baseline")
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=100,
                        invalid_chain=None,
                        valid=True,
                        raw={
                            "result": {"valid": True},
                            "new_compact_config": {},
                        },
                    )
                    for name in requests
                }

        bridge = FakeRescaleBridge()
        all_max_action = make_all_max_action_vector(num_layers=1)
        probe = ProbeBatch(
            input_ids=torch.ones((2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 4), dtype=torch.long),
            labels=torch.tensor([1, 0], dtype=torch.long),
        )
        env = BLBStage2Env(
            handler=FakeHandler(),
            model=TinyModel(),
            probe_batches=[probe],
            rescale_bridge=bridge,
            # RO still evaluates 4 SF/fusion blocks. Layer-0 Block1 K is
            # model-side only and does not add a replan request.
            baseline=BaselineCostStats(
                total_bits_sum=400,
                total_fusion_count=0,
                avg_k=avg_truncation_k_in_action(all_max_action, 1),
            ),
            reward_weights=RewardWeights(),
            acc_threshold=0.5,
            stab_threshold=10.0,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            env_cfg=BLBStage2EnvConfig(profile="mrpc", num_trials_per_step=1),
        )

        _obs, reward, done, info = env.step(all_max_action)

        self.assertEqual(bridge.calls, ["cfg-derived"])
        self.assertTrue(done)
        self.assertFalse(info["invalid"])
        self.assertFalse(info["reward_breakdown"].invalid)
        # The all-max action matches the test baseline (same total_bits, fusion,
        # avg_k), so cost_score / k_drop / bits_drop must be zero. v2-style
        # Current bounded reward does not emit the historical +40 tier bonus.
        # This regression guards the optimizer-baseline scoring path: the
        # baseline action has zero cost-side gain while still passing the gates.
        breakdown = info["reward_breakdown"]
        self.assertEqual(breakdown.k_drop, 0.0)
        self.assertEqual(breakdown.bits_drop, 0.0)
        self.assertEqual(breakdown.fusion_count, 0.0)
        self.assertEqual(breakdown.cost_score, 0.0)
        self.assertTrue(breakdown.metric_ok)
        self.assertTrue(breakdown.stab_ok)
        self.assertEqual(breakdown.priority, 3)
        self.assertGreaterEqual(reward, -5.0)
        self.assertLessEqual(reward, 5.0)

    def test_env_runs_forward_even_when_optimizer_invalid(self):
        from blb_stage2_rl.action_space import load_max_sfs, make_all_max_action_vector
        from blb_stage2_rl.env import BLBStage2Env, BLBStage2EnvConfig, ProbeBatch
        from blb_stage2_rl.reward import BaselineCostStats, RewardWeights
        from rescale_optimizer_bridge import RescaleOptimizerOutput

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.forward_count = 0

            def forward(self, **_kwargs):
                self.forward_count += 1
                logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
                return type("Output", (), {"logits": logits})()

        class FakeHandler:
            def __getattr__(self, name):
                if name.startswith(("replace_", "restore_")):
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

        class FakeRescaleBridge:
            def evaluate_blocks(self, requests):
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=100,
                        invalid_chain={"reason": "unit"},
                        valid=False,
                        raw={},
                    )
                    for name in requests
                }

        probe = ProbeBatch(
            input_ids=torch.ones((2, 4), dtype=torch.long),
            attention_mask=torch.ones((2, 4), dtype=torch.long),
            labels=torch.tensor([1, 0], dtype=torch.long),
        )
        model = TinyModel()
        env = BLBStage2Env(
            handler=FakeHandler(),
            model=model,
            probe_batches=[probe],
            rescale_bridge=FakeRescaleBridge(),
            baseline=BaselineCostStats(total_bits_sum=400, total_fusion_count=0, avg_k=13.0),
            reward_weights=RewardWeights(invalid_penalty=30.0),
            acc_threshold=0.5,
            stab_threshold=10.0,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            env_cfg=BLBStage2EnvConfig(profile="mrpc", num_trials_per_step=1),
        )

        _obs, _reward, done, info = env.step(make_all_max_action_vector(num_layers=1))

        self.assertTrue(done)
        self.assertTrue(info["invalid"])
        # When Rescale_optimizer reports any_invalid, env.step short-circuits the
        # model forward and emits a P1 invalid reward with the invalid_penalty
        # docked. This was the behaviour the user asked for on
        # 2026-05-17 ("出现 invalid chain 再去做推理就没有意义了") and is
        # documented in CLAUDE.md → "Sequential invalid-action mask + skip-forward".
        # The reward priority / invalid_penalty contract is preserved; only the
        # wasted model forward is skipped.
        self.assertFalse(info["forward_ran"])
        self.assertEqual(model.forward_count, 0)
        self.assertEqual(
            info.get("forward_skipped_reason"),
            "optimizer_invalid_chain",
        )
        self.assertEqual(info["reward_breakdown"].priority, 1)
        self.assertTrue(info["reward_breakdown"].invalid)
        self.assertEqual(info["reward_breakdown"].r_invalid, -30.0)

    def test_real_mrpc_all_max_baseline_optimizer_outputs_are_valid(self):
        from blb_stage2_rl.action_space import (
            action_vector_to_cfgs,
            build_optimizer_requests,
            load_max_sfs,
            make_all_max_action_vector,
        )
        from rescale_optimizer_bridge import (
            InProcessInvoker,
            RescaleOptimizerBridge,
            aggregate_optimizer_signals,
        )

        gelu_degree = [1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1]
        attn_degree = [2, 2, 5, 5, 5, 2, 5, 2, 5, 5, 5, 2]
        decoded = action_vector_to_cfgs(
            make_all_max_action_vector(num_layers=12),
            load_max_sfs("mrpc"),
            num_layers=12,
            gelu_degree=gelu_degree,
            attn_degree=attn_degree,
        )
        requests = build_optimizer_requests("mrpc", decoded.cfgs_dict())
        bridge = RescaleOptimizerBridge(
            invoker=InProcessInvoker.from_profile(
                rescale_optimizer_root="Rescale_optimizer",
                profile="mrpc",
            )
        )

        outputs = {
            config_name: bridge.evaluate_baseline(config_name=config_name)
            for config_name in requests
        }
        signals = aggregate_optimizer_signals(outputs)

        self.assertFalse(signals.any_invalid, signals.invalid_chains)
        # RO has 59 SF/fusion requests; model materialization separately
        # contains all 60 K actions, including layer-0 Block1.
        self.assertEqual(signals.valid_block_count, 59)
        self.assertEqual(signals.invalid_block_count, 0)

    def test_real_mrpc_all_max_cfg_derived_optimizer_outputs_are_valid(self):
        from blb_stage2_rl.action_space import (
            action_vector_to_cfgs,
            build_optimizer_requests,
            load_max_sfs,
            make_all_max_action_vector,
        )
        from rescale_optimizer_bridge import (
            InProcessInvoker,
            RescaleOptimizerBridge,
            aggregate_optimizer_signals,
        )

        decoded = action_vector_to_cfgs(
            make_all_max_action_vector(num_layers=12),
            load_max_sfs("mrpc"),
            num_layers=12,
            gelu_degree=4,
            attn_degree=4,
        )
        requests = build_optimizer_requests("mrpc", decoded.cfgs_dict())
        bridge = RescaleOptimizerBridge(
            invoker=InProcessInvoker.from_profile(
                rescale_optimizer_root="Rescale_optimizer",
                profile="mrpc",
            )
        )

        outputs = bridge.evaluate_blocks(requests)
        signals = aggregate_optimizer_signals(outputs)

        self.assertFalse(signals.any_invalid, signals.invalid_chains)
        # RO has 59 SF/fusion requests; model materialization separately
        # contains all 60 K actions, including layer-0 Block1.
        self.assertEqual(signals.valid_block_count, 59)
        self.assertEqual(signals.invalid_block_count, 0)
        self.assertEqual(
            outputs["block4_L0"].raw["delta_overrides"]["ctct_rot_softmax_mul_v"],
            39,
        )


class BLBActionDescriptionRegressionTests(unittest.TestCase):
    def test_action_description_names_every_noise_point_and_truncation(self):
        from blb_stage2_rl.action_space import (
            describe_action_vector,
            load_max_sfs,
            make_all_max_action_vector,
        )

        action = make_all_max_action_vector(num_layers=1)
        desc = describe_action_vector(
            action,
            max_sfs=load_max_sfs("mrpc"),
            num_layers=1,
            gelu_degree=[4],
            attn_degree=[2],
            profile="mrpc",
        )
        records = desc["records"]

        self.assertEqual(desc["action_length"], len(action))
        self.assertEqual(desc["num_layers"], 1)
        self.assertGreaterEqual(len(records), len(action))

        first_input = [r for r in records if r["block"] == "first_input"][0]
        self.assertEqual(first_input["field"], "first_input_sf")
        self.assertEqual(first_input["value_type"], "scaling_factor")
        self.assertEqual(first_input["N"], 8192)

        truncations = [r for r in records if r["value_type"] == "truncation_k"]
        self.assertTrue(any(r["block"] == "block2" and r["value"] == 13 for r in truncations))
        self.assertTrue(all("location" in r and "operation" in r for r in records))
        self.assertTrue(any(r["field"] == "square_rescale_sf_0" and r["block"] == "block3" for r in records))


class BLBPlaybookArtifactRegressionTests(unittest.TestCase):
    def test_phase0_preflight_report_names_current_operator_entrypoints(self):
        from scripts.blb_phase0_preflight import build_phase0_entrypoint_report

        repo_root = Path(__file__).resolve().parents[1]
        report = build_phase0_entrypoint_report(repo_root)

        self.assertIn("llama_7B_LayerImportance.sh", report)
        self.assertIn("run rl --preset mrpc-blb-stage2-rl", report)
        self.assertIn("blb_stage2_rl/runner.py", report)
        self.assertIn("Rescale_optimizer", report)

    def test_candidate_store_hash_fidelity_and_rank_key_are_stable(self):
        from blb_stage2_rl.candidate_store import (
            CandidateStore,
            action_hash,
            candidate_rank_key,
        )

        action = [4, 3, 2, 1]
        self.assertEqual(action_hash(action), action_hash(np.array(action, dtype=int)))
        self.assertLess(
            candidate_rank_key({"valid": True, "acc_violation": 0.0, "stability_violation": 0.0, "normalized_cost": 0.4}),
            candidate_rank_key({"valid": True, "acc_violation": 0.0, "stability_violation": 0.0, "normalized_cost": 0.8}),
        )
        self.assertGreater(
            candidate_rank_key({"valid": True, "acc_violation": 0.0, "stability_violation": 0.1, "normalized_cost": 0.1}),
            candidate_rank_key({"valid": False, "acc_violation": 0.0, "stability_violation": 0.0, "normalized_cost": 0.0}),
        )

        with _workspace_tempdir() as td:
            store = CandidateStore(Path(td) / "candidate_store.jsonl")
            self.assertTrue(store.should_evaluate(action, "F1"))
            store.append({"action_indices": action, "fidelity": "F1", "valid": True})
            self.assertFalse(store.should_evaluate(action, "F1"))
            # F2/F3 were dropped on 2026-05-16 (see candidate_store.FIDELITY_ORDER
            # docstring). The active ladder is F0 → F1 → F4, so promotion past
            # F1 must be checked with F4.
            self.assertTrue(store.should_evaluate(action, "F4"))
            store.append({"action_indices": action, "fidelity": "F4", "valid": True})
            self.assertFalse(store.should_evaluate(action, "F4"))

    def test_candidate_action_hash_avoids_json_dumps_for_integer_vectors(self):
        from blb_stage2_rl import candidate_store

        action = [4, 3, 2, -1]
        expected = hashlib.sha256(b"[4,3,2,-1]").hexdigest()

        self.assertEqual(candidate_store.action_hash(action), expected)
        source = inspect.getsource(candidate_store._action_hash_from_tuple)
        self.assertNotIn("json.dumps", source)

    def test_registry_export_records_action_values_and_current_slot_count(self):
        from blb_stage2_rl.action_space import K_LEVELS
        from scripts.blb_export_action_registry import build_registry_payload

        payload = build_registry_payload(
            profile="mrpc",
            num_layers=1,
            gelu_degree=[4],
            attn_degree=[2],
        )
        records = payload["slot_registry_full"]

        self.assertEqual(payload["schema"], "blb_action_registry_export_v1")
        self.assertTrue(records)
        self.assertTrue(all(r["action_values"] for r in records))
        self.assertTrue(all("scale_semantics" in r for r in records))

        k_records = [r for r in records if r["value_type"] == "truncation_k"]
        self.assertTrue(k_records)
        expected_k_idx = list(K_LEVELS).index(max(K_LEVELS))
        self.assertTrue(all(r["all_max_action_index"] == expected_k_idx for r in k_records))

        self.assertEqual(payload["summary"]["per_layer_slot_count"], 73)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block1"], 9)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block2"], 23)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block3"], 8)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block4"], 17)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"]["block5"], 16)
        slot_check = payload["current_code_slot_check_markdown"]
        self.assertIn("expected_slots_per_layer", slot_check)
        self.assertIn("status", slot_check)

    def test_f0_eval_record_contains_action_hash_rank_key_and_optimizer_summary(self):
        from blb_stage2_rl.candidate_store import action_hash
        from scripts.blb_eval_action import build_f0_candidate_record

        action = [4, 4, 3, 2]
        signals = type(
            "Signals",
            (),
            {
                "any_invalid": False,
                "total_bits_sum": 1200,
                "total_fusion_count": 7,
                "invalid_chains": {},
            },
        )()

        record = build_f0_candidate_record(
            action,
            source="unit",
            signals=signals,
            baseline_total_bits=2400,
        )

        self.assertEqual(record["fidelity"], "F0")
        self.assertEqual(record["action_hash"], action_hash(action))
        self.assertEqual(record["action_vector_hash"], action_hash(action))
        self.assertEqual(record["normalized_cost"], 0.5)
        self.assertEqual(record["optimizer"]["total_bits_sum"], 1200)
        self.assertEqual(record["rank_key"], [0.0, 1200.0, 7.0])

    def test_f0_eval_all_max_uses_optimizer_baseline_path(self):
        import scripts.blb_eval_action as mod
        from rescale_optimizer_bridge import RescaleOptimizerOutput

        calls = []

        class FakeInvoker:
            @classmethod
            def from_profile(cls, **_kwargs):
                return object()

        class FakeBridge:
            def __init__(self, invoker):
                self.invoker = invoker

            def evaluate_blocks(self, requests):
                calls.append("cfg-derived")
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=100,
                        invalid_chain=None,
                        valid=True,
                        raw={},
                    )
                    for name in requests
                }

            def evaluate_baseline_blocks(self, requests):
                calls.append("baseline")
                return {
                    name: RescaleOptimizerOutput(
                        config_name=name,
                        fusion_count=0,
                        total_bits=100,
                        invalid_chain=None,
                        valid=True,
                        raw={},
                    )
                    for name in requests
                }

        old_invoker = mod.InProcessInvoker
        old_bridge = mod.RescaleOptimizerBridge
        mod.InProcessInvoker = FakeInvoker
        mod.RescaleOptimizerBridge = FakeBridge
        try:
            with _workspace_tempdir() as td:
                record = mod.run_f0_eval(
                    profile="mrpc",
                    num_layers=1,
                    action_json=None,
                    output_dir=td,
                    source="all_max_f0",
                    rescale_optimizer_root="unused",
                    baseline_total_bits=None,
                )
        finally:
            mod.InProcessInvoker = old_invoker
            mod.RescaleOptimizerBridge = old_bridge

        self.assertEqual(calls, ["cfg-derived"])
        self.assertTrue(record["valid"])


class BLBProbeSizingRegressionTests(unittest.TestCase):
    def test_probe_batch_count_covers_requested_probe_size(self):
        from blb_stage2_rl.runner import _effective_probe_batch_count

        class Ev:
            batch_size = 16
            stage2_probe_size = 256

        class Cfg:
            probe_batch_count = 4

        self.assertEqual(_effective_probe_batch_count(Ev(), Cfg()), 16)

    def test_explicit_probe_batch_count_override_still_works(self):
        from blb_stage2_rl.runner import _effective_probe_batch_count

        class Ev:
            batch_size = 16
            stage2_probe_size = 256
            blb_v3_probe_batch_count = 3

        class Cfg:
            probe_batch_count = 4

        self.assertEqual(_effective_probe_batch_count(Ev(), Cfg()), 3)


class BLBPersistencePathRegressionTests(unittest.TestCase):
    def test_blb_progress_stays_under_stage2_noise_progress(self):
        from blb_stage2_rl.runner import resolve_blb_persistence_dir

        class DummyEvaluator:
            pass

        with _workspace_tempdir() as td:
            ev = DummyEvaluator()
            ev.run_output_dir = td
            path = Path(resolve_blb_persistence_dir(ev))

        self.assertEqual(path.name, "progress")
        self.assertEqual(path.parent.name, "stage2_noise")


if __name__ == "__main__":
    unittest.main()
