import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _write_compare_direct_jsons(root: Path) -> dict:
    stage1_template = {
        "bert-base": {
            "mrpc": {
                "gelu": [4] * 12,
                "softmax": [6] * 12,
            }
        }
    }
    stage2_template = {
        "bert-base": {
            "mrpc": {
                "x": [20] * 12,
                "wq": [20] * 12,
                "wk": [20] * 12,
                "wv": [20] * 12,
                "wo": [20] * 12,
                "wffn1": [20] * 12,
                "wffn2": [20] * 12,
            }
        }
    }
    paths = {
        "rl_stage1": root / "glue_configs_best_ppo.json",
        "rl_stage2": root / "glue_noise_configs_best_ppo.json",
        "ga_stage1": root / "glue_configs_best_genetic.json",
        "ga_stage2": root / "glue_noise_configs_best_genetic.json",
    }
    for key, path in paths.items():
        payload = stage1_template if "stage1" in key else stage2_template
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return paths


class OutputLayoutRegressionTests(unittest.TestCase):
    def test_resolve_run_output_layout_is_lazy(self):
        try:
            import layer_importance_evaluator as evaluator_module
        except ImportError as exc:
            self.skipTest(f"layer_importance_evaluator import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "lazy_layout_run"
            layout = evaluator_module.resolve_run_output_layout(
                str(run_dir),
                search_algorithm="rl",
            )

            self.assertFalse(run_dir.exists())
            self.assertEqual(
                layout["log_file"],
                os.path.join(
                    str(run_dir),
                    "stage1",
                    "pruning_search_log.txt",
                ),
            )
            self.assertEqual(
                layout["noise_log_file"],
                os.path.join(
                    str(run_dir),
                    "stage2_noise",
                    "pruning_search_log.txt",
                ),
            )

    def test_activate_noise_logging_creates_stage2_dir_only_when_needed(self):
        try:
            import layer_importance_evaluator as evaluator_module
        except ImportError as exc:
            self.skipTest(f"layer_importance_evaluator import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "ga_run"
            layout = evaluator_module.resolve_run_output_layout(
                str(run_dir),
                search_algorithm="ga",
            )

            evaluator = evaluator_module.LayerImportanceEvaluator.__new__(
                evaluator_module.LayerImportanceEvaluator
            )
            evaluator.log_file = layout["log_file"]
            evaluator.noise_log_file = layout["noise_log_file"]
            evaluator.active_log_file = evaluator.log_file
            evaluator._noise_log_initialized = False
            evaluator.stage1_rl_lr_raw = "1e-4"
            evaluator.stage2_rl_lr_raw = "1e-4"
            evaluator.stage1_ppo_lr_initial = 1e-4
            evaluator.stage2_ppo_lr_initial = 1e-4
            evaluator.stage1_ppo_lr_mode = "direct"
            evaluator.stage2_ppo_lr_mode = "direct"
            evaluator.stage1_rl_episodes = 300
            evaluator.stage2_rl_episodes = 500
            evaluator.run_output_dir = str(run_dir)

            self.assertFalse((run_dir / "stage2_noise").exists())

            previous_log_file = evaluator.activate_noise_logging()

            self.assertEqual(previous_log_file, layout["log_file"])
            self.assertEqual(evaluator.active_log_file, layout["noise_log_file"])
            self.assertTrue((run_dir / "stage2_noise").is_dir())
            self.assertTrue(Path(layout["noise_log_file"]).is_file())

            evaluator.restore_log_file(previous_log_file)
            self.assertEqual(evaluator.active_log_file, layout["log_file"])

    def test_genetic_json_dump_creates_missing_stage2_dir(self):
        try:
            import genetic_search_module as genetic_module
        except ImportError as exc:
            self.skipTest(f"genetic_search_module import unavailable: {exc}")

        with TemporaryDirectory() as tmpdir:
            result_path = (
                Path(tmpdir)
                / "compare_run"
                / "children"
                / "ga"
                / "stage2_noise"
                / "noise_ga_search_results.json"
            )

            self.assertFalse(result_path.parent.exists())

            genetic_module._json_dump(
                str(result_path),
                {"status": "ok", "result_path": str(result_path)},
            )

            self.assertTrue(result_path.parent.is_dir())
            self.assertTrue(result_path.is_file())
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")

    def test_compare_runner_dry_run_does_not_create_unused_logs_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "compare_dry_run"
            json_paths = _write_compare_direct_jsons(Path(tmpdir))
            command = [
                sys.executable,
                "rl_ga_compare_runner.py",
                "--base-model",
                "dummy-model",
                "--data-path",
                "mrpc",
                "--dataset",
                "mrpc",
                "--output-dir",
                str(output_dir),
                "--model-type",
                "bert-base",
                "--compare-config-mode",
                "direct",
                "--rl-compare-stage1-json",
                str(json_paths["rl_stage1"]),
                "--rl-compare-stage2-json",
                str(json_paths["rl_stage2"]),
                "--ga-compare-stage1-json",
                str(json_paths["ga_stage1"]),
                "--ga-compare-stage2-json",
                str(json_paths["ga_stage2"]),
                "--stage2-compare-repeats",
                "7",
                "--dry-run",
            ]

            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )
            self.assertTrue((output_dir / "meta" / "compare_metadata.json").is_file())
            self.assertFalse((output_dir / "logs").exists())
            self.assertFalse((output_dir / "reports").exists())
            self.assertFalse((output_dir / "children").exists())
            metadata = json.loads((output_dir / "meta" / "compare_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["compare_config_mode"], "direct")
            self.assertEqual(metadata["model_type"], "bert-base")
            self.assertEqual(metadata["stage2_compare_repeats"], 7)
            self.assertEqual(metadata["rl_side"]["stage1_json_path"], str(json_paths["rl_stage1"]))
            self.assertEqual(metadata["ga_side"]["stage2_json_path"], str(json_paths["ga_stage2"]))

    def test_compare_runner_deprecated_noise_eval_repeat_alias(self):
        repo_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "compare_dry_run_alias"
            json_paths = _write_compare_direct_jsons(Path(tmpdir))
            command = [
                sys.executable,
                "rl_ga_compare_runner.py",
                "--base-model",
                "dummy-model",
                "--data-path",
                "mrpc",
                "--dataset",
                "mrpc",
                "--output-dir",
                str(output_dir),
                "--model-type",
                "bert-base",
                "--compare-config-mode",
                "direct",
                "--rl-compare-stage1-json",
                str(json_paths["rl_stage1"]),
                "--rl-compare-stage2-json",
                str(json_paths["rl_stage2"]),
                "--ga-compare-stage1-json",
                str(json_paths["ga_stage1"]),
                "--ga-compare-stage2-json",
                str(json_paths["ga_stage2"]),
                "--noise-eval-repeat",
                "9",
                "--dry-run",
            ]

            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr or completed.stdout,
            )
            metadata = json.loads(
                (output_dir / "meta" / "compare_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["stage2_compare_repeats"], 9)
            self.assertTrue(
                any("noise-eval-repeat" in item for item in metadata["warnings"])
            )

    def test_compare_runner_rejects_duplicate_repeat_flags(self):
        repo_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "compare_dry_run_conflict"
            json_paths = _write_compare_direct_jsons(Path(tmpdir))
            command = [
                sys.executable,
                "rl_ga_compare_runner.py",
                "--base-model",
                "dummy-model",
                "--data-path",
                "mrpc",
                "--dataset",
                "mrpc",
                "--output-dir",
                str(output_dir),
                "--model-type",
                "bert-base",
                "--compare-config-mode",
                "direct",
                "--rl-compare-stage1-json",
                str(json_paths["rl_stage1"]),
                "--rl-compare-stage2-json",
                str(json_paths["rl_stage2"]),
                "--ga-compare-stage1-json",
                str(json_paths["ga_stage1"]),
                "--ga-compare-stage2-json",
                str(json_paths["ga_stage2"]),
                "--noise-eval-repeat",
                "9",
                "--stage2-compare-repeats",
                "7",
                "--dry-run",
            ]

            completed = subprocess.run(
                command,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("stage2-compare-repeats", completed.stderr + completed.stdout)


if __name__ == "__main__":
    unittest.main()
