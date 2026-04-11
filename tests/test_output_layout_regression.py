import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


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

    def test_compare_runner_dry_run_does_not_create_unused_logs_dir(self):
        repo_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "compare_dry_run"
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
                "--stage1-search-episodes",
                "170",
                "--stage2-search-episodes",
                "170",
                "--stage1-search-generations",
                "6",
                "--stage2-search-generations",
                "5",
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
            self.assertEqual(metadata["rl_stage1_search_episodes"], 170)
            self.assertEqual(metadata["ga_stage1_search_generations"], 6)
            self.assertEqual(metadata["rl_side_config"]["final_eval_config_source"], "search")
            self.assertEqual(metadata["ga_side_config"]["noise_eval_config_source"], "search")


if __name__ == "__main__":
    unittest.main()
