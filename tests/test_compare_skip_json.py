import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class CompareSkipJsonTests(unittest.TestCase):
    def test_normalize_compare_side_config_requires_json_when_skipping(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with self.assertRaises(compare_runner.CompareRunnerError):
            compare_runner.normalize_compare_side_config(
                label="RL",
                skip_stage1_search=True,
                final_eval_config_source="search",
                final_eval_config_path="",
                skip_noise_search=False,
                noise_eval_config_source="search",
                noise_eval_config_path="",
            )

    def test_ensure_stage1_eval_json_honors_requested_json_source(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "rl_run"
            stage1_dir = run_dir / "stage1_final_eval"
            requested_json = Path(tmpdir) / "stage1_saved.json"
            requested_json.write_text("{}", encoding="utf-8")

            captured = {}

            class FakeEvaluator:
                def __init__(self):
                    self.stage1_final_eval_dir = str(stage1_dir)
                    self.noise_final_eval_dir = str(run_dir / "stage2_noise_final_eval")

                def log(self, message):
                    del message

            class FakeRunner:
                def __init__(self, *, evaluator, config_source, config_path, **kwargs):
                    del kwargs
                    captured["config_source"] = config_source
                    captured["config_path"] = config_path
                    captured["results_dir"] = evaluator.stage1_final_eval_dir

                def run(self, **kwargs):
                    captured["run_kwargs"] = kwargs
                    summary_path = (
                        Path(captured["results_dir"]) / "final_eval_results_mrpc.json"
                    )
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(
                        json.dumps(
                            {
                                "status": "ok",
                                "selected_source": captured["config_source"],
                                "selected": {
                                    "gelu": [4, 4],
                                    "softmax": [6, 6],
                                },
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    return {"summary_path": str(summary_path)}

            fake_context = SimpleNamespace(
                base_gelu=np.asarray([4, 4], dtype=int),
                base_softmax=np.asarray([6, 6], dtype=int),
                base_tot_c=10.0,
                base_g_c=5.0,
                base_s_c=5.0,
                limit_loss=0.2,
                limit_p=0.8,
                limit_s=0.7,
            )

            fake_eval_module = SimpleNamespace(FinalEvaluationModule=FakeRunner)
            fake_ga_module = SimpleNamespace(
                GeneticFinalEvaluationModule=FakeRunner,
                build_stage1_context=lambda evaluator, log_fn=None, include_distribution=False: fake_context,
            )
            side_config = compare_runner.CompareSideConfig(
                skip_stage1_search=True,
                final_eval_config_source="json",
                final_eval_config_path=str(requested_json),
            )

            with patch.object(
                compare_runner,
                "build_compare_evaluator",
                return_value=FakeEvaluator(),
            ), patch.dict(
                sys.modules,
                {
                    "final_evaluation_module": fake_eval_module,
                    "genetic_search_module": fake_ga_module,
                },
            ):
                json_path, warnings = compare_runner.ensure_stage1_eval_json(
                    algorithm="rl",
                    run_dir=run_dir,
                    side_config=side_config,
                    dataset="mrpc",
                    base_model="dummy-model",
                    data_path="mrpc",
                    batch_size=16,
                    stage1_rl_lr="1e-4",
                    stage2_rl_lr="1e-4",
                    random_seed=42,
                    perm_trials=10,
                    cost_trials=10,
                    budget_trials=10,
                    noise_eval_repeat_n=1,
                )

            self.assertTrue(json_path.is_file())
            self.assertEqual(captured["config_source"], "json")
            self.assertEqual(captured["config_path"], str(requested_json))
            self.assertIsNone(captured["run_kwargs"]["search_best_config"])
            self.assertTrue(any("json" in item for item in warnings))

    def test_ensure_stage2_eval_json_honors_requested_json_source(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "ga_run"
            stage1_summary = run_dir / "stage1_final_eval" / "final_eval_results_mrpc.json"
            stage1_summary.parent.mkdir(parents=True, exist_ok=True)
            stage1_summary.write_text(
                json.dumps(
                    {
                        "selected": {
                            "gelu": [4, 4],
                            "softmax": [6, 6],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            stage2_dir = run_dir / "stage2_noise_final_eval"
            requested_json = Path(tmpdir) / "stage2_saved.json"
            requested_json.write_text("{}", encoding="utf-8")

            captured = {}

            class FakeEvaluator:
                def __init__(self):
                    self.stage1_final_eval_dir = str(run_dir / "stage1_final_eval")
                    self.noise_final_eval_dir = str(stage2_dir)

                def log(self, message):
                    del message

            class FakeNoiseRunner:
                def __init__(self, *, evaluator, config_source, config_path, **kwargs):
                    del kwargs
                    captured["config_source"] = config_source
                    captured["config_path"] = config_path
                    captured["results_dir"] = evaluator.noise_final_eval_dir

                def run(self, **kwargs):
                    captured["run_kwargs"] = kwargs
                    summary_path = (
                        Path(captured["results_dir"]) / "noise_final_eval_results_mrpc.json"
                    )
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(
                        json.dumps(
                            {
                                "status": "ok",
                                "selected_source": captured["config_source"],
                                "selected": {
                                    "noise_config": {"x": [20, 20]},
                                    "loss": 0.1,
                                    "p": 0.9,
                                    "s": 0.8,
                                    "tot_c": 3.0,
                                    "time_ms": 1.0,
                                    "feasible": True,
                                },
                                "baseline": {
                                    "loss": 0.2,
                                    "p": 0.8,
                                    "s": 0.7,
                                    "tot_c": 4.0,
                                    "time_ms": 1.2,
                                    "feasible": True,
                                },
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    return {"summary_path": str(summary_path)}

            fake_context = SimpleNamespace(
                search_limits={"loss": 0.2, "metric1": 0.8, "metric2": 0.7},
                cost_reference_noise_config={"x": [18, 18]},
                cost_reference_tot_c=4.0,
            )

            fake_ga_module = SimpleNamespace(
                GeneticNoiseFinalEvaluationModule=FakeNoiseRunner,
                build_stage1_context=lambda evaluator, log_fn=None, include_distribution=False: None,
                build_stage2_context=lambda evaluator, fixed_gelu, fixed_softmax, log_fn=None: fake_context,
            )
            fake_noise_module = SimpleNamespace(NoiseFinalEvaluationModule=FakeNoiseRunner)
            side_config = compare_runner.CompareSideConfig(
                skip_noise_search=True,
                noise_eval_config_source="json",
                noise_eval_config_path=str(requested_json),
            )

            with patch.object(
                compare_runner,
                "build_compare_evaluator",
                return_value=FakeEvaluator(),
            ), patch.dict(
                sys.modules,
                {
                    "genetic_search_module": fake_ga_module,
                    "noise_final_evaluation_module": fake_noise_module,
                },
            ):
                json_path, warnings = compare_runner.ensure_stage2_eval_json(
                    algorithm="ga",
                    run_dir=run_dir,
                    side_config=side_config,
                    dataset="mrpc",
                    base_model="dummy-model",
                    data_path="mrpc",
                    batch_size=16,
                    stage1_rl_lr="1e-4",
                    stage2_rl_lr="1e-4",
                    random_seed=42,
                    perm_trials=10,
                    cost_trials=10,
                    budget_trials=10,
                    noise_eval_repeat_n=1,
                )

            self.assertTrue(json_path.is_file())
            self.assertEqual(captured["config_source"], "json")
            self.assertEqual(captured["config_path"], str(requested_json))
            self.assertIsNone(captured["run_kwargs"]["search_best_noise_config"])
            self.assertTrue(any("json" in item for item in warnings))

    def test_ensure_stage2_eval_json_uses_safe_evaluator_flags_for_search_fallback(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "ga_run"
            stage1_summary = run_dir / "stage1_final_eval" / "final_eval_results_mrpc.json"
            stage1_summary.parent.mkdir(parents=True, exist_ok=True)
            stage1_summary.write_text(
                json.dumps(
                    {
                        "selected": {
                            "gelu": [4, 4],
                            "softmax": [6, 6],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            evaluator_kwargs = {}

            class FakeEvaluator:
                def __init__(self):
                    self.stage1_final_eval_dir = str(run_dir / "stage1_final_eval")
                    self.noise_final_eval_dir = str(run_dir / "stage2_noise_final_eval")

                def log(self, message):
                    del message

            class FakeNoiseRunner:
                def __init__(self, *, evaluator, config_source, config_path, **kwargs):
                    del kwargs
                    self.results_dir = Path(evaluator.noise_final_eval_dir)
                    self.config_source = config_source
                    self.config_path = config_path

                def run(self, **kwargs):
                    del kwargs
                    summary_path = self.results_dir / "noise_final_eval_results_mrpc.json"
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(
                        json.dumps({"status": "ok", "selected_source": self.config_source}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    return {"summary_path": str(summary_path)}

            fake_context = SimpleNamespace(
                search_limits={"loss": 0.2, "metric1": 0.8, "metric2": 0.7},
                cost_reference_noise_config={"x": [18, 18]},
                cost_reference_tot_c=4.0,
            )

            fake_ga_module = SimpleNamespace(
                GeneticNoiseFinalEvaluationModule=FakeNoiseRunner,
                build_stage1_context=lambda evaluator, log_fn=None, include_distribution=False: None,
                build_stage2_context=lambda evaluator, fixed_gelu, fixed_softmax, log_fn=None: fake_context,
            )
            fake_noise_module = SimpleNamespace(NoiseFinalEvaluationModule=FakeNoiseRunner)
            side_config = compare_runner.CompareSideConfig(
                skip_stage1_search=False,
                final_eval_config_source="search",
                final_eval_config_path="",
                skip_noise_search=False,
                noise_eval_config_source="search",
                noise_eval_config_path="",
            )

            def _fake_build_compare_evaluator(**kwargs):
                evaluator_kwargs.update(kwargs)
                return FakeEvaluator()

            with patch.object(
                compare_runner,
                "build_compare_evaluator",
                side_effect=_fake_build_compare_evaluator,
            ), patch.dict(
                sys.modules,
                {
                    "genetic_search_module": fake_ga_module,
                    "noise_final_evaluation_module": fake_noise_module,
                },
            ):
                json_path, warnings = compare_runner.ensure_stage2_eval_json(
                    algorithm="ga",
                    run_dir=run_dir,
                    side_config=side_config,
                    dataset="mrpc",
                    base_model="dummy-model",
                    data_path="mrpc",
                    batch_size=16,
                    stage1_rl_lr="1e-4",
                    stage2_rl_lr="1e-4",
                    random_seed=42,
                    perm_trials=10,
                    cost_trials=10,
                    budget_trials=10,
                    noise_eval_repeat_n=1,
                )

            self.assertTrue(json_path.is_file())
            self.assertEqual(evaluator_kwargs["skip_stage1_rl"], True)
            self.assertEqual(evaluator_kwargs["skip_stage1_final_eval"], True)
            self.assertEqual(evaluator_kwargs["final_eval_config_source"], "json")
            self.assertEqual(evaluator_kwargs["skip_noise_rl"], False)
            self.assertEqual(evaluator_kwargs["skip_noise_final_eval"], False)
            self.assertEqual(evaluator_kwargs["noise_eval_config_source"], "search")
            self.assertTrue(any("Stage-2" in item for item in warnings))

    def test_ensure_stage2_eval_json_passes_stage1_selection_into_runner(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "rl_run"
            stage1_summary = run_dir / "stage1_final_eval" / "final_eval_results_mrpc.json"
            stage1_summary.parent.mkdir(parents=True, exist_ok=True)
            stage1_summary.write_text(
                json.dumps(
                    {
                        "selected": {
                            "gelu": [4, 2, 1],
                            "softmax": [6, 5, 4],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            requested_json = Path(tmpdir) / "stage2_saved.json"
            requested_json.write_text("{}", encoding="utf-8")
            captured = {}

            class FakeEvaluator:
                def __init__(self):
                    self.stage1_final_eval_dir = str(run_dir / "stage1_final_eval")
                    self.noise_final_eval_dir = str(run_dir / "stage2_noise_final_eval")

                def log(self, message):
                    del message

            class FakeNoiseRunner:
                def __init__(self, *, evaluator, config_source, config_path, **kwargs):
                    del kwargs
                    self.results_dir = Path(evaluator.noise_final_eval_dir)
                    self.config_source = config_source
                    self.config_path = config_path

                def run(self, **kwargs):
                    captured["run_kwargs"] = kwargs
                    summary_path = self.results_dir / "noise_final_eval_results_mrpc.json"
                    summary_path.parent.mkdir(parents=True, exist_ok=True)
                    summary_path.write_text(
                        json.dumps(
                            {
                                "status": "ok",
                                "selected_source": self.config_source,
                                "fixed_stage1_config": {
                                    "gelu": [int(x) for x in np.asarray(kwargs["fixed_gelu"], dtype=int)],
                                    "softmax": [int(x) for x in np.asarray(kwargs["fixed_softmax"], dtype=int)],
                                },
                                "selected": {
                                    "noise_config": {"x": [20, 20, 20]},
                                    "loss": 0.1,
                                    "p": 0.9,
                                    "s": 0.8,
                                    "tot_c": 3.0,
                                    "time_ms": 1.0,
                                    "feasible": True,
                                },
                                "baseline": {
                                    "loss": 0.2,
                                    "p": 0.8,
                                    "s": 0.7,
                                    "tot_c": 4.0,
                                    "time_ms": 1.2,
                                    "feasible": True,
                                },
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    return {"summary_path": str(summary_path)}

            fake_context = SimpleNamespace(
                search_limits={"loss": 0.2, "metric1": 0.8, "metric2": 0.7},
                cost_reference_noise_config={"x": [18, 18, 18]},
                cost_reference_tot_c=4.0,
            )

            fake_ga_module = SimpleNamespace(
                GeneticNoiseFinalEvaluationModule=FakeNoiseRunner,
                build_stage1_context=lambda evaluator, log_fn=None, include_distribution=False: None,
                build_stage2_context=lambda evaluator, fixed_gelu, fixed_softmax, log_fn=None: fake_context,
            )
            fake_noise_module = SimpleNamespace(NoiseFinalEvaluationModule=FakeNoiseRunner)
            side_config = compare_runner.CompareSideConfig(
                skip_noise_search=True,
                noise_eval_config_source="json",
                noise_eval_config_path=str(requested_json),
            )

            with patch.object(
                compare_runner,
                "build_compare_evaluator",
                return_value=FakeEvaluator(),
            ), patch.dict(
                sys.modules,
                {
                    "genetic_search_module": fake_ga_module,
                    "noise_final_evaluation_module": fake_noise_module,
                },
            ):
                json_path, _ = compare_runner.ensure_stage2_eval_json(
                    algorithm="rl",
                    run_dir=run_dir,
                    side_config=side_config,
                    dataset="mrpc",
                    base_model="dummy-model",
                    data_path="mrpc",
                    batch_size=16,
                    stage1_rl_lr="1e-4",
                    stage2_rl_lr="1e-4",
                    random_seed=42,
                    perm_trials=10,
                    cost_trials=10,
                    budget_trials=10,
                    noise_eval_repeat_n=3,
                )

            self.assertTrue(json_path.is_file())
            self.assertEqual(
                list(np.asarray(captured["run_kwargs"]["fixed_gelu"], dtype=int)),
                [4, 2, 1],
            )
            self.assertEqual(
                list(np.asarray(captured["run_kwargs"]["fixed_softmax"], dtype=int)),
                [6, 5, 4],
            )

    def test_build_stage2_compare_payload_keeps_fixed_stage1_and_repeat_stats(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            compare_root = Path(tmpdir) / "compare_root"
            rl_run_dir = compare_root / "children" / "rl"
            ga_run_dir = compare_root / "children" / "ga"
            rl_json = compare_root / "reports" / "rl_stage2.json"
            ga_json = compare_root / "reports" / "ga_stage2.json"
            rl_json.parent.mkdir(parents=True, exist_ok=True)

            rl_json.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "fixed_stage1_config": {"gelu": [4, 2], "softmax": [6, 5]},
                        "selected_source": "search",
                        "selected": {
                            "loss": 0.10,
                            "p": 0.90,
                            "s": 0.80,
                            "tot_c": 3.0,
                            "time_ms": 1.1,
                            "noise_config": {"x": [20, 20]},
                            "breakdown": {"x": 1.0},
                        },
                        "baseline": {
                            "loss": 0.20,
                            "p": 0.80,
                            "s": 0.70,
                            "tot_c": 4.0,
                            "time_ms": 1.3,
                        },
                        "repeat_evaluation": {
                            "stats": {
                                "n": 3,
                                "loss_mean": 0.10,
                                "loss_std": 0.01,
                                "loss_min": 0.09,
                                "loss_max": 0.11,
                                "p_mean": 0.90,
                                "p_std": 0.02,
                                "p_min": 0.88,
                                "p_max": 0.92,
                                "s_mean": 0.80,
                                "s_std": 0.03,
                                "s_min": 0.77,
                                "s_max": 0.83,
                                "time_ms_mean": 1.10,
                                "time_ms_std": 0.05,
                                "time_ms_min": 1.04,
                                "time_ms_max": 1.16,
                            },
                            "trials": [{"trial": 1}],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ga_json.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "fixed_stage1_config": {"gelu": [4, 1], "softmax": [6, 4]},
                        "selected_source": "json",
                        "selected": {
                            "loss": 0.12,
                            "p": 0.88,
                            "s": 0.79,
                            "tot_c": 2.8,
                            "time_ms": 1.0,
                            "noise_config": {"x": [18, 18]},
                            "breakdown": {"x": 0.8},
                        },
                        "baseline": {
                            "loss": 0.22,
                            "p": 0.78,
                            "s": 0.69,
                            "tot_c": 4.1,
                            "time_ms": 1.2,
                        },
                        "repeat_evaluation": {
                            "stats": {
                                "n": 4,
                                "loss_mean": 0.12,
                                "loss_std": 0.02,
                                "loss_min": 0.10,
                                "loss_max": 0.14,
                                "p_mean": 0.88,
                                "p_std": 0.01,
                                "p_min": 0.87,
                                "p_max": 0.89,
                                "s_mean": 0.79,
                                "s_std": 0.02,
                                "s_min": 0.77,
                                "s_max": 0.81,
                                "time_ms_mean": 1.00,
                                "time_ms_std": 0.04,
                                "time_ms_min": 0.95,
                                "time_ms_max": 1.05,
                            },
                            "trials": [{"trial": 1}],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            payload = compare_runner.build_stage_compare_payload(
                stage_label="stage2",
                dataset="mrpc",
                compare_root=compare_root,
                rl_run_dir=rl_run_dir,
                ga_run_dir=ga_run_dir,
                rl_json_path=rl_json,
                ga_json_path=ga_json,
                rl_warnings=[],
                ga_warnings=[],
                process_meta={"rl": {"state": "completed"}, "ga": {"state": "completed"}},
            )

            self.assertEqual(
                payload["sides"]["rl"]["stage1_selected_config"],
                {"gelu": [4, 2], "softmax": [6, 5]},
            )
            self.assertEqual(
                payload["sides"]["ga"]["stage1_selected_config"],
                {"gelu": [4, 1], "softmax": [6, 4]},
            )
            summary = payload["stage2_repeat_summary"]
            self.assertEqual(summary["rl_repeat_count"], 3)
            self.assertEqual(summary["ga_repeat_count"], 4)
            loss_row = next(item for item in summary["metrics"] if item["key"] == "loss")
            self.assertAlmostEqual(loss_row["rl"]["var"], 0.0001)
            self.assertAlmostEqual(loss_row["ga"]["var"], 0.0004)
            self.assertEqual(loss_row["winner"], "rl")

    def test_noise_final_evaluation_repeat_n_respects_user_value(self):
        try:
            import noise_final_evaluation_module as noise_eval_module
        except ImportError as exc:
            self.skipTest(f"noise_final_evaluation_module import unavailable: {exc}")

        evaluator = SimpleNamespace(noise_final_eval_dir="tmp/noise_eval")
        module = noise_eval_module.NoiseFinalEvaluationModule(
            evaluator=evaluator,
            repeat_n=1,
        )
        self.assertEqual(module.repeat_n, 1)


if __name__ == "__main__":
    unittest.main()
