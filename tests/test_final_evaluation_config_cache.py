from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import final_evaluation_module as fem


class FinalEvaluationConfigCacheTest(unittest.TestCase):
    def _make_runner(self, config_path: Path):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.config_path = str(config_path)
        runner.evaluator = SimpleNamespace(
            total_layers=2,
            model_type="bert-base",
            dataset_key="mrpc",
            model=None,
            log=lambda *_args, **_kwargs: None,
        )
        runner.allowed_gelu_selected = [1, 2, 4]
        runner.allowed_softmax = [6]
        runner.input_noise_allowed = [1]
        runner.weight_noise_allowed = [1]
        runner.wffn1_noise_allowed = [1]
        return runner

    def test_stage1_and_stage2_json_resolution_share_loaded_config(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "final_eval_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "_comment": "ignored",
                        "bert-base": {
                            "mrpc": {
                                "stage1": {"gelu": [1, 2], "softmax": [6, 6]},
                                "stage2": {
                                    key: [1, 1]
                                    for key in fem.NOISE_SCALING_FACTOR_KEYS
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            runner = self._make_runner(config_path)
            original_load = fem.json.load
            load_count = 0

            def counting_load(handle):
                nonlocal load_count
                load_count += 1
                return original_load(handle)

            with mock.patch.object(fem.json, "load", side_effect=counting_load):
                runner._resolve_stage1_from_json(total_layers=2)
                runner._resolve_stage2_from_json(total_layers=2)

            self.assertEqual(load_count, 1)

    def test_stage2_total_cost_sampling_reuses_count_solution_maps(self):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.input_noise_allowed = [1]
        runner.weight_noise_allowed = [1]
        runner.wffn1_noise_allowed = [1]
        runner.evaluator = SimpleNamespace(
            INPUT_NOISE_COST_MAP={1: 1.0 / 40.0},
            WEIGHT_NOISE_COST_MAP={1: 1.0 / 40.0},
            WFFN1_NOISE_COST_MAP={1: 1.0 / 40.0},
        )
        rng = np.random.default_rng(123)
        target_total = len(fem.BREAKDOWN_KEYS) / 40.0

        with mock.patch.object(
            runner,
            "_enumerate_stage2_count_solutions",
            wraps=runner._enumerate_stage2_count_solutions,
        ) as wrapped:
            cfg1 = runner._sample_stage2_total_cost(rng, target_total, total_layers=1)
            cfg2 = runner._sample_stage2_total_cost(rng, target_total, total_layers=1)

        self.assertIsNotNone(cfg1)
        self.assertIsNotNone(cfg2)
        self.assertEqual(wrapped.call_count, len(fem.BREAKDOWN_KEYS))

    def test_stage2_equiv_sampling_uses_exact_cached_count_solutions(self):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.input_noise_allowed = [1, 2]
        runner.weight_noise_allowed = [1, 2]
        runner.wffn1_noise_allowed = [1, 2]
        runner.evaluator = SimpleNamespace(
            INPUT_NOISE_COST_MAP={1: 1.0 / 40.0, 2: 2.0 / 40.0},
            WEIGHT_NOISE_COST_MAP={1: 1.0 / 40.0, 2: 2.0 / 40.0},
            WFFN1_NOISE_COST_MAP={1: 1.0 / 40.0, 2: 2.0 / 40.0},
        )
        rng = np.random.default_rng(123)
        breakdown = {short: 2.0 / 40.0 for short in fem.BREAKDOWN_KEYS}

        with mock.patch.object(
            runner,
            "_stage2_cost_matched_array",
            side_effect=AssertionError("should use cached exact count solutions"),
        ):
            cfg = runner._sample_stage2_equiv(rng, breakdown, total_layers=2)

        self.assertIsNotNone(cfg)
        self.assertEqual(runner._stage2_config_cost_key(cfg), len(fem.BREAKDOWN_KEYS) * 2)

    def test_stage2_total_cost_sampling_reuses_count_combo_plan(self):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.input_noise_allowed = [1, 2]
        runner.weight_noise_allowed = [1, 2]
        runner.wffn1_noise_allowed = [1, 2]
        runner.evaluator = SimpleNamespace(
            INPUT_NOISE_COST_MAP={1: 1.0 / 40.0, 2: 2.0 / 40.0},
            WEIGHT_NOISE_COST_MAP={1: 1.0 / 40.0, 2: 2.0 / 40.0},
            WFFN1_NOISE_COST_MAP={1: 1.0 / 40.0, 2: 2.0 / 40.0},
        )
        key_scan_count = 0

        class CountingSolutionMap(dict):
            def keys(self):
                nonlocal key_scan_count
                key_scan_count += 1
                return super().keys()

        original = runner._enumerate_stage2_count_solutions

        def counting_enumerate(*args, **kwargs):
            return CountingSolutionMap(original(*args, **kwargs))

        rng = np.random.default_rng(123)
        target_total = len(fem.BREAKDOWN_KEYS) * 2.0 / 40.0
        with mock.patch.object(
            runner,
            "_enumerate_stage2_count_solutions",
            side_effect=counting_enumerate,
        ):
            configs = [
                runner._sample_stage2_total_cost(rng, target_total, total_layers=2)
                for _ in range(3)
            ]

        self.assertTrue(all(cfg is not None for cfg in configs))
        self.assertEqual(key_scan_count, len(fem.BREAKDOWN_KEYS))

    def test_stage1_total_cost_sampling_reuses_feasible_pairs(self):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.allowed_gelu_random = [1, 2]
        runner.allowed_softmax = [6]
        key_scan_count = 0

        class CountingSolutionMap(dict):
            def keys(self):
                nonlocal key_scan_count
                key_scan_count += 1
                return super().keys()

        gelu_solution_map = CountingSolutionMap({
            1: [(2, 0)],
            2: [(0, 2)],
        })
        softmax_solution_map = {
            3: [(2,)],
        }
        rng = np.random.default_rng(123)
        configs = [
            runner._sample_stage1_total_cost(
                rng,
                gelu_solution_map,
                softmax_solution_map,
                target_total_cost=2.0,
            )
            for _ in range(3)
        ]

        self.assertTrue(all(cfg is not None for cfg in configs))
        self.assertEqual(key_scan_count, 1)

    def test_stage2_only_random_generation_skips_stage1_solution_enumeration(self):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.final_eval_only = False
        runner.stage1_budget_trials = 0
        runner.stage2_budget_trials = 1
        runner.permutation_trials = 0
        runner.cost_equivalent_trials = 0
        runner.budget_equivalent_trials = 0
        runner.allowed_gelu_random = [1, 2]
        runner.allowed_softmax = [6]
        runner.input_noise_allowed = [1]
        runner.weight_noise_allowed = [1]
        runner.wffn1_noise_allowed = [1]
        runner.evaluator = SimpleNamespace(
            log=lambda *_args, **_kwargs: None,
            GELU_COST_MAP={1: 0.5, 2: 1.0},
            SOFTMAX_COST_MAP={6: 0.0},
            INPUT_NOISE_COST_MAP={1: 1.0 / 40.0},
            WEIGHT_NOISE_COST_MAP={1: 1.0 / 40.0},
            WFFN1_NOISE_COST_MAP={1: 1.0 / 40.0},
        )
        opt_gelu = np.array([1, 1], dtype=int)
        opt_softmax = np.array([6, 6], dtype=int)
        opt_noise_cfg = {
            full: np.array([1, 1], dtype=int)
            for full in fem.NOISE_SCALING_FACTOR_KEYS
        }

        def build_result(*_args, **_kwargs):
            return {}, None

        with mock.patch.object(
            runner,
            "_enumerate_cost_solutions",
            side_effect=AssertionError("stage1 maps should not be built"),
        ):
            runner._generate_random_results(
                opt_gelu,
                opt_softmax,
                opt_noise_cfg,
                opt_stage1_tot_c=1.0,
                opt_stage2_tot_c=len(fem.BREAKDOWN_KEYS) * 2.0 / 40.0,
                opt_breakdown={short: 2.0 / 40.0 for short in fem.BREAKDOWN_KEYS},
                total_layers=2,
                build_result=build_result,
            )


if __name__ == "__main__":
    unittest.main()
