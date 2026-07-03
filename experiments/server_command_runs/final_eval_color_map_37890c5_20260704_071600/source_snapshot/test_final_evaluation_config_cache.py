from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
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

    def test_stage2_count_combo_plan_caches_feasible_keys_by_state(self):
        class CountingKeys:
            def __init__(self, values):
                self.values = tuple(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(self.values)

        key_options = (CountingKeys(range(5)), CountingKeys(range(5)))
        suffix_possible = (
            frozenset(range(9)),
            frozenset(range(5)),
            frozenset({0}),
        )
        combo_plan = (key_options, suffix_possible, {})
        solution_maps = [
            {key: [(key,)] for key in range(5)},
            {key: [(key,)] for key in range(5)},
        ]
        rng = np.random.default_rng(123)

        for _ in range(4):
            choice = fem.UnifiedFinalEvaluationModule._sample_stage2_count_combo(
                rng,
                solution_maps,
                target_key=4,
                combo_plan=combo_plan,
            )
            self.assertIsNotNone(choice)

        self.assertEqual(key_options[0].iterations, 1)

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

    def test_random_summary_vectorizes_dominance_checks(self):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.evaluator = SimpleNamespace(get_num_metrics=lambda: 2)
        selected = {
            "loss": 0.30,
            "p": 0.90,
            "s": 0.85,
            "stage1_tot_c": 10.0,
            "stage2_tot_c": 20.0,
        }
        random_results = [
            {
                "family": "cost_matched",
                "feasible": True,
                "loss": 0.40,
                "p": 0.80,
                "s": 0.80,
                "loss_delta_vs_baseline": 0.02,
                "p_delta_vs_baseline": -0.01,
                "s_delta_vs_baseline": -0.02,
                "total_cost": 31.0,
                "stage1_tot_c": 11.0,
                "stage2_tot_c": 22.0,
            },
            {
                "family": "cost_matched",
                "feasible": False,
                "loss": 0.20,
                "p": 0.95,
                "s": 0.90,
                "loss_delta_vs_baseline": -0.01,
                "p_delta_vs_baseline": 0.02,
                "s_delta_vs_baseline": 0.03,
                "total_cost": 28.0,
                "stage1_tot_c": 9.0,
                "stage2_tot_c": 19.0,
            },
        ]

        with mock.patch.object(
            runner,
            "_dominates",
            side_effect=AssertionError("dominance should be computed inline"),
        ):
            summary = runner._summarize_random_results(selected, random_results, num_metrics=2)

        self.assertEqual(summary["by_family"]["cost_matched"]["count"], 2)
        self.assertEqual(summary["by_family"]["cost_matched"]["dominance_rate"], 0.5)
        self.assertEqual(summary["overall"]["dominance_rate"], 0.5)

    def test_random_summary_uses_running_stats_without_np_materialization(self):
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        selected = {
            "loss": 0.30,
            "p": 0.90,
            "s": 0.85,
            "stage1_tot_c": 10.0,
            "stage2_tot_c": 20.0,
        }
        random_results = [
            {
                "family": "cost_matched",
                "feasible": True,
                "loss": 0.40,
                "p": 0.80,
                "s": 0.80,
                "loss_delta_vs_baseline": 0.02,
                "p_delta_vs_baseline": -0.01,
                "s_delta_vs_baseline": -0.02,
                "total_cost": 31.0,
                "stage1_tot_c": 11.0,
                "stage2_tot_c": 22.0,
                "loss_var": 0.04,
            },
            {
                "family": "cost_matched",
                "feasible": False,
                "loss": 0.20,
                "p": 0.95,
                "s": 0.90,
                "loss_delta_vs_baseline": -0.01,
                "p_delta_vs_baseline": 0.02,
                "s_delta_vs_baseline": 0.03,
                "total_cost": 28.0,
                "stage1_tot_c": 9.0,
                "stage2_tot_c": 19.0,
                "loss_var": 0.02,
            },
        ]

        with mock.patch.object(
            fem.np,
            "mean",
            side_effect=AssertionError("random summary should use running mean stats"),
        ), mock.patch.object(
            fem.np,
            "std",
            side_effect=AssertionError("random summary should use running std stats"),
        ):
            summary = runner._summarize_random_results(selected, random_results, num_metrics=2)

        family = summary["by_family"]["cost_matched"]
        self.assertEqual(family["count"], 2)
        self.assertAlmostEqual(family["loss_mean"], 0.30)
        self.assertAlmostEqual(family["loss_std"], 0.10)
        self.assertAlmostEqual(family["total_cost_mean"], 29.5)
        self.assertAlmostEqual(family["total_cost_std"], 1.5)
        self.assertAlmostEqual(family["loss_eval_variance_mean"], 0.03)
        self.assertAlmostEqual(summary["overall"]["dominance_rate"], 0.5)

    def test_relative_metric_attach_does_not_copy_random_results(self):
        source = inspect.getsource(fem.UnifiedFinalEvaluationModule.run)

        self.assertNotIn("+ list(random_results)", source)

    def test_final_eval_plots_iterate_axes_without_flat_list_copy(self):
        comparison_source = inspect.getsource(fem.UnifiedFinalEvaluationModule._plot_results)
        variance_source = inspect.getsource(fem.UnifiedFinalEvaluationModule._plot_variance_results)

        self.assertNotIn("list(axes.flat)[:3]", comparison_source)
        self.assertNotIn("list(axes.flat)[:3]", variance_source)

    def test_final_eval_summary_bar_chart_collects_series_once(self):
        source = inspect.getsource(fem.UnifiedFinalEvaluationModule._plot_results)

        self.assertNotIn(
            'feasible = [summary["by_family"][f]["feasible_rate"] for f in families]',
            source,
        )
        self.assertNotIn(
            'dominance = [summary["by_family"][f]["dominance_rate"] for f in families]',
            source,
        )

    def test_ordered_families_reuses_static_preferred_order(self):
        source = inspect.getsource(fem.UnifiedFinalEvaluationModule._ordered_families)

        self.assertNotIn("self._family_colors().keys()", source)
        self.assertNotIn("preferred = list(", source)

    def test_final_eval_plots_reuse_static_family_color_map(self):
        comparison_source = inspect.getsource(fem.UnifiedFinalEvaluationModule._plot_results)
        variance_source = inspect.getsource(fem.UnifiedFinalEvaluationModule._plot_variance_results)

        self.assertNotIn("self._family_colors()", comparison_source)
        self.assertNotIn("self._family_colors()", variance_source)

    def test_float_stat_helpers_stream_values_without_clean_lists(self):
        values = [1.0, None, float("nan"), 3.0, float("inf")]

        with mock.patch.object(
            fem.np,
            "mean",
            side_effect=AssertionError("mean helper should stream finite values"),
        ), mock.patch.object(
            fem.np,
            "std",
            side_effect=AssertionError("std helper should stream finite values"),
        ):
            mean = fem.UnifiedFinalEvaluationModule._mean_float_or_none(values)
            std = fem.UnifiedFinalEvaluationModule._std_float_or_none(values)

        self.assertEqual(mean, 2.0)
        self.assertEqual(std, 1.0)
        self.assertIsNone(
            fem.UnifiedFinalEvaluationModule._mean_float_or_none([None, float("nan")])
        )
        self.assertIsNone(
            fem.UnifiedFinalEvaluationModule._std_float_or_none([None, float("nan")])
        )

    def test_variance_plot_streams_group_means_without_numpy_mean(self):
        class FakeAxis:
            def __init__(self):
                self.bar_calls = []

            def scatter(self, *_args, **_kwargs):
                pass

            def axhline(self, *_args, **_kwargs):
                pass

            def text(self, *_args, **_kwargs):
                pass

            def set_title(self, *_args, **_kwargs):
                pass

            def set_ylabel(self, *_args, **_kwargs):
                pass

            def set_xlabel(self, *_args, **_kwargs):
                pass

            def grid(self, *_args, **_kwargs):
                pass

            def margins(self, *_args, **_kwargs):
                pass

            def ticklabel_format(self, *_args, **_kwargs):
                pass

            def legend(self, *_args, **_kwargs):
                pass

            def set_xlim(self, *_args, **_kwargs):
                pass

            def set_xticks(self, *_args, **_kwargs):
                pass

            def set_xticklabels(self, *_args, **_kwargs):
                pass

            def bar(self, x, height, **kwargs):
                self.bar_calls.append(
                    {
                        "x": list(np.asarray(x, dtype=float).reshape(-1)),
                        "height": list(height),
                        "label": kwargs.get("label"),
                    }
                )

        class FakeFigure:
            def suptitle(self, *_args, **_kwargs):
                pass

        axes = np.array(
            [[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]],
            dtype=object,
        )
        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.__path__ = []
        fake_matplotlib.use = lambda *_args, **_kwargs: None
        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda *_args, **_kwargs: (FakeFigure(), axes)
        fake_pyplot.savefig = lambda *_args, **_kwargs: None
        fake_pyplot.close = lambda *_args, **_kwargs: None

        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.results_dir = "/tmp/final-eval-test"
        logs = []
        runner.evaluator = SimpleNamespace(
            dataset_key="mrpc",
            log=lambda message, *_args, **_kwargs: logs.append(str(message)),
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
            },
        ), mock.patch.object(
            fem.np,
            "mean",
            side_effect=AssertionError("variance plot should stream group means"),
        ):
            plot_path = runner._plot_variance_results(
                metric_short_names=["Accuracy", "F1"],
                num_metrics=2,
                baseline={
                    "loss_var": 0.1,
                    "p_var": 0.2,
                    "s_var": 0.3,
                    "total_cost": 10.0,
                },
                optimized={
                    "loss_var": 0.4,
                    "p_var": 0.5,
                    "s_var": 0.6,
                    "total_cost": 11.0,
                },
                stage1_fixed_max={
                    "loss_var": 0.7,
                    "p_var": 0.8,
                    "s_var": 0.9,
                    "total_cost": 12.0,
                },
                random_results=[
                    {
                        "family": "Budget",
                        "loss_var": 1.0,
                        "p_var": 2.0,
                        "s_var": 4.0,
                        "total_cost": 13.0,
                    },
                    {
                        "family": "Budget",
                        "loss_var": 3.0,
                        "p_var": 6.0,
                        "s_var": 8.0,
                        "total_cost": 14.0,
                    },
                ],
            )

        self.assertEqual(
            plot_path,
            "/tmp/final-eval-test/final_eval_variance_mrpc.png",
            "\n".join(logs),
        )
        bar_axis = axes[1, 1]
        self.assertEqual([call["label"] for call in bar_axis.bar_calls], ["Loss", "Accuracy", "F1"])
        self.assertEqual(bar_axis.bar_calls[0]["height"][2], 2.0)
        self.assertEqual(bar_axis.bar_calls[1]["height"][2], 4.0)
        self.assertEqual(bar_axis.bar_calls[2]["height"][2], 6.0)

    def test_variance_plot_scans_random_points_once_per_panel(self):
        class FakeAxis:
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return lambda *_args, **_kwargs: None

        class FakeFigure:
            def suptitle(self, *_args, **_kwargs):
                pass

        class LimitedGetDict(dict):
            def __init__(self, *args, max_total_cost_gets, **kwargs):
                super().__init__(*args, **kwargs)
                self.max_total_cost_gets = max_total_cost_gets
                self.total_cost_gets = 0

            def get(self, key, default=None):
                if key == "total_cost":
                    self.total_cost_gets += 1
                    if self.total_cost_gets > self.max_total_cost_gets:
                        raise AssertionError("variance plot should scan each point once per panel")
                return super().get(key, default)

        axes = np.array(
            [[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]],
            dtype=object,
        )
        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.__path__ = []
        fake_matplotlib.use = lambda *_args, **_kwargs: None
        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda *_args, **_kwargs: (FakeFigure(), axes)
        fake_pyplot.savefig = lambda *_args, **_kwargs: None
        fake_pyplot.close = lambda *_args, **_kwargs: None

        random_point = LimitedGetDict(
            {
                "family": "Budget",
                "loss_var": 1.0,
                "p_var": 2.0,
                "s_var": 3.0,
                "total_cost": 13.0,
            },
            max_total_cost_gets=3,
        )
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.results_dir = "/tmp/final-eval-test"
        logs = []
        runner.evaluator = SimpleNamespace(
            dataset_key="mrpc",
            log=lambda message, *_args, **_kwargs: logs.append(str(message)),
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
            },
        ):
            plot_path = runner._plot_variance_results(
                metric_short_names=["Accuracy", "F1"],
                num_metrics=2,
                baseline={"loss_var": 0.1, "p_var": 0.2, "s_var": 0.3},
                optimized={
                    "loss_var": 0.4,
                    "p_var": 0.5,
                    "s_var": 0.6,
                    "total_cost": 11.0,
                },
                stage1_fixed_max={
                    "loss_var": 0.7,
                    "p_var": 0.8,
                    "s_var": 0.9,
                    "total_cost": 12.0,
                },
                random_results=[random_point],
            )

        self.assertEqual(
            plot_path,
            "/tmp/final-eval-test/final_eval_variance_mrpc.png",
            "\n".join(logs),
        )
        self.assertEqual(random_point.total_cost_gets, 3)

    def test_comparison_plot_scans_random_points_once_per_panel(self):
        class FakeAxis:
            def __getattr__(self, name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return lambda *_args, **_kwargs: None

        class FakeFigure:
            def suptitle(self, *_args, **_kwargs):
                pass

        class LimitedGetDict(dict):
            def __init__(self, *args, max_total_cost_gets, **kwargs):
                super().__init__(*args, **kwargs)
                self.max_total_cost_gets = max_total_cost_gets
                self.total_cost_gets = 0

            def get(self, key, default=None):
                if key == "total_cost":
                    self.total_cost_gets += 1
                    if self.total_cost_gets > self.max_total_cost_gets:
                        raise AssertionError("comparison plot should scan each point once per panel")
                return super().get(key, default)

        axes = np.array(
            [[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]],
            dtype=object,
        )
        fake_matplotlib = ModuleType("matplotlib")
        fake_matplotlib.__path__ = []
        fake_matplotlib.use = lambda *_args, **_kwargs: None
        fake_pyplot = ModuleType("matplotlib.pyplot")
        fake_pyplot.subplots = lambda *_args, **_kwargs: (FakeFigure(), axes)
        fake_pyplot.savefig = lambda *_args, **_kwargs: None
        fake_pyplot.close = lambda *_args, **_kwargs: None

        random_point = LimitedGetDict(
            {
                "family": "Budget",
                "loss": 0.4,
                "p": 0.8,
                "s": 0.7,
                "total_cost": 13.0,
            },
            max_total_cost_gets=3,
        )
        runner = fem.UnifiedFinalEvaluationModule.__new__(fem.UnifiedFinalEvaluationModule)
        runner.results_dir = "/tmp/final-eval-test"
        logs = []
        runner.evaluator = SimpleNamespace(
            dataset_key="mrpc",
            log=lambda message, *_args, **_kwargs: logs.append(str(message)),
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
            },
        ):
            plot_path = runner._plot_results(
                metric_short_names=["Accuracy", "F1"],
                num_metrics=2,
                baseline={"loss": 0.5, "p": 0.75, "s": 0.70},
                optimized={"loss": 0.3, "p": 0.9, "s": 0.85, "total_cost": 11.0},
                stage1_fixed_max={"loss": 0.35, "p": 0.88, "s": 0.82, "total_cost": 12.0},
                random_results=[random_point],
                summary={
                    "by_family": {
                        "Budget": {
                            "feasible_rate": 1.0,
                            "dominance_rate": 0.0,
                        }
                    }
                },
            )

        self.assertEqual(
            plot_path,
            "/tmp/final-eval-test/final_eval_comparison_mrpc.png",
            "\n".join(logs),
        )
        self.assertEqual(random_point.total_cost_gets, 3)

    def test_numeric_axis_limits_stream_float_values_once(self):
        class FakeAxis:
            def __init__(self):
                self.xlim = None

            def set_xlim(self, left, right):
                self.xlim = (left, right)

        class LimitedFloat:
            def __init__(self, value):
                self.value = value
                self.float_calls = 0

            def __float__(self):
                self.float_calls += 1
                if self.float_calls > 1:
                    raise AssertionError("axis limit helper should convert each value once")
                return float(self.value)

        lo = LimitedFloat(2.0)
        hi = LimitedFloat(4.0)
        ax = FakeAxis()

        fem.UnifiedFinalEvaluationModule._set_numeric_axis_limits(
            ax,
            [None, lo, float("nan"), hi],
        )

        self.assertEqual(lo.float_calls, 1)
        self.assertEqual(hi.float_calls, 1)
        self.assertEqual(ax.xlim, (1.84, 4.16))


if __name__ == "__main__":
    unittest.main()
