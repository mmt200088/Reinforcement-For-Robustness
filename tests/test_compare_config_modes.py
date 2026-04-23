import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def _stage1_result(dataset: str, layers: int) -> dict:
    return {
        "dataset": dataset,
        "selected": {
            "gelu": [4] * layers,
            "softmax": [6] * layers,
        },
    }


def _stage2_config_template(dataset: str, model_type: str, layers: int) -> dict:
    payload = {
        "x": [20] * layers,
        "wq": [20] * layers,
        "wk": [20] * layers,
        "wv": [20] * layers,
        "wo": [20] * layers,
        "wffn1": [20] * layers,
        "wffn2": [20] * layers,
    }
    return {model_type: {dataset: payload}}


def _stage2_result(dataset: str, layers: int) -> dict:
    return {
        "dataset": dataset,
        "fixed_stage1_config": {
            "gelu": [4] * layers,
            "softmax": [6] * layers,
        },
        "selected": {
            "noise_config": {
                "x": [20] * layers,
                "wq": [20] * layers,
                "wk": [20] * layers,
                "wv": [20] * layers,
                "wo": [20] * layers,
                "wffn1": [20] * layers,
                "wffn2": [20] * layers,
            }
        },
    }


class CompareConfigModeTests(unittest.TestCase):
    def test_resolve_direct_side_spec_accepts_result_and_template_mix(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage1_json = root / "glue_stage1_best_ppo_result.json"
            stage2_json = root / "glue_final_configs_best_ppo.json"
            stage1_json.write_text(
                json.dumps(_stage1_result("mrpc", 12), ensure_ascii=False),
                encoding="utf-8",
            )
            stage2_json.write_text(
                json.dumps(
                    _stage2_config_template("mrpc", "bert-base", 12),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            spec = compare_runner.resolve_direct_side_spec(
                algorithm="rl",
                dataset="mrpc",
                model_type="bert-base",
                run_dir=root / "children" / "rl",
                stage1_json_path=str(stage1_json),
                stage2_json_path=str(stage2_json),
            )

            self.assertEqual(spec.stage1_input_kind, "result_json")
            self.assertEqual(spec.stage2_input_kind, "config_json")
            self.assertFalse(spec.side_config.skip_stage1_search)
            self.assertTrue(spec.side_config.skip_noise_search)
            self.assertEqual(spec.side_config.final_eval_config_source, "json")
            self.assertEqual(spec.side_config.final_eval_config_path, str(stage2_json))

    def test_resolve_direct_side_spec_rejects_model_layer_mismatch(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage1_json = root / "glue_stage1_best_ppo_result.json"
            stage2_json = root / "glue_stage2_best_ppo_result.json"
            stage1_json.write_text(
                json.dumps(_stage1_result("mrpc", 24), ensure_ascii=False),
                encoding="utf-8",
            )
            stage2_json.write_text(
                json.dumps(_stage2_result("mrpc", 24), ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaises(compare_runner.CompareRunnerError):
                compare_runner.resolve_direct_side_spec(
                    algorithm="rl",
                    dataset="mrpc",
                    model_type="bert-base",
                    run_dir=root / "children" / "rl",
                    stage1_json_path=str(stage1_json),
                    stage2_json_path=str(stage2_json),
                )

    def test_resolve_persistent_side_spec_rejects_metadata_model_mismatch(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = compare_runner.persistent_run_dir_for_compare(
                persistent_root=root,
                algorithm="rl",
                model_type="bert-base",
                dataset="mrpc",
                stage1_accuracy_tolerance=0.005,
                stage2_limit_tolerance=0.05,
                stage2_stability_tolerance=0.05,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "algorithm": "rl",
                        "model_type": "bert-large",
                        "dataset": "mrpc",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(compare_runner.CompareRunnerError):
                compare_runner.resolve_persistent_side_spec(
                    algorithm="rl",
                    dataset="mrpc",
                    model_type="bert-base",
                    persistent_root=root,
                    stage1_accuracy_tolerance=0.005,
                    stage2_limit_tolerance=0.05,
                    stage2_stability_tolerance=0.05,
                )

    def test_evaluation_only_compare_evaluates_one_side_at_a_time(self):
        try:
            import rl_ga_compare_runner as compare_runner
        except ImportError as exc:
            self.skipTest(f"rl_ga_compare_runner import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rl_json = root / "glue_final_configs_best_ppo.json"
            ga_json = root / "glue_final_configs_best_genetic.json"
            template = {
                "bert-base": {
                    "mrpc": {
                        "stage1": {
                            "gelu": [4] * 12,
                            "softmax": [6] * 12,
                        },
                        "stage2": _stage2_config_template(
                            "mrpc", "bert-base", 12
                        )["bert-base"]["mrpc"],
                    }
                }
            }
            for path in (rl_json, ga_json):
                path.write_text(json.dumps(template, ensure_ascii=False), encoding="utf-8")

            args = SimpleNamespace(
                output_dir=str(root / "compare"),
                compare_config_mode="direct",
                compare_persistent_root="",
                rl_compare_stage1_json=str(rl_json),
                rl_compare_stage2_json=str(rl_json),
                ga_compare_stage1_json=str(ga_json),
                ga_compare_stage2_json=str(ga_json),
                rl_compare_stage1_accuracy_tolerance=None,
                rl_compare_stage2_limit_tolerance=None,
                rl_compare_stage2_stability_tolerance=None,
                ga_compare_stage1_accuracy_tolerance=None,
                ga_compare_stage2_limit_tolerance=None,
                ga_compare_stage2_stability_tolerance=None,
                model_type="bert-base",
                base_model="dummy-model",
                data_path="mrpc",
                dataset="mrpc",
                batch_size=1,
                stage1_search_lr="1e-4",
                stage2_search_lr="1e-4",
                random_seed=42,
                perm_trials=1,
                cost_trials=1,
                budget_trials=1,
                stage2_compare_repeats=1,
                stage2_k_trials=None,
                stage2_probe_size=None,
                dry_run=False,
            )
            order = []

            def fake_ensure_final_eval_json(**kwargs):
                algorithm = kwargs["algorithm"]
                run_dir = kwargs["run_dir"]
                dataset = kwargs["dataset"]
                order.append(f"ensure:{algorithm}")
                path = compare_runner.final_eval_json(run_dir, dataset)
                payload = {
                    "dataset": dataset,
                    "baseline": {
                        "loss": 1.0,
                        "p": 0.7,
                        "s": 0.7,
                        "stage1_tot_c": 72.0,
                        "stage2_tot_c": 0.0,
                        "gelu": [4] * 12,
                        "softmax": [6] * 12,
                    },
                    "optimized_stage1": {
                        "gelu": [4] * 12,
                        "softmax": [6] * 12,
                    },
                    "optimized": {
                        "loss": 0.9,
                        "p": 0.8,
                        "s": 0.8,
                        "stage1_tot_c": 72.0,
                        "stage2_tot_c": 10.0,
                        "gelu": [4] * 12,
                        "softmax": [6] * 12,
                        "noise_config": {
                            "input_noise_scaling_factors": [20] * 12,
                        },
                    },
                }
                compare_runner.write_json(path, payload)
                return path, []

            def fake_cleanup_cuda_memory():
                order.append("cleanup")

            with patch.object(
                compare_runner,
                "ensure_final_eval_json",
                side_effect=fake_ensure_final_eval_json,
            ), patch.object(
                compare_runner,
                "cleanup_cuda_memory",
                side_effect=fake_cleanup_cuda_memory,
            ):
                rc = compare_runner.run_evaluation_only_compare(args)

            self.assertEqual(rc, 0)
            self.assertEqual(order, ["ensure:rl", "cleanup", "ensure:ga", "cleanup"])

if __name__ == "__main__":
    unittest.main()
