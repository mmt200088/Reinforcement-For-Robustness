import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


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

if __name__ == "__main__":
    unittest.main()
