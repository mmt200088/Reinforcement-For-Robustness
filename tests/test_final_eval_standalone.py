import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class FinalEvalStandaloneTests(unittest.TestCase):
    def test_preset_parser_uses_independent_output_root(self):
        from final_eval.config import parse_final_eval_settings

        settings = parse_final_eval_settings(
            ["--preset", "mrpc-final-eval-only", "--dry-run"]
        )

        self.assertEqual(settings.dataset, "mrpc")
        self.assertEqual(settings.algorithm, "rl")
        self.assertEqual(settings.source, "json")
        self.assertEqual(settings.repeat, 50)
        self.assertEqual(settings.perm_trials, 0)
        self.assertTrue(settings.output_root.endswith(str(Path("final_eval") / "outputs")))

    def test_cli_dry_run_builds_final_eval_only_command(self):
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "final_eval.run_final_eval",
                "--preset",
                "mrpc-final-eval-only",
                "--dry-run",
            ],
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
        self.assertIn("--final_eval_only true", completed.stdout)
        self.assertIn("final_eval/outputs", completed.stdout.replace("\\", "/"))
        self.assertIn("--final_eval_random_enabled false", completed.stdout)

    def test_action_grid_range_expands_cartesian_product(self):
        from final_eval.action_grid import build_action_candidates

        candidates = build_action_candidates(
            num_layers=2,
            profile="mrpc",
            fixed_specs=[],
            range_specs=["truncation=8,9,11,13", "wffn1=18,20"],
        )

        self.assertEqual(len(candidates), 8)
        pairs = {
            (candidate.overrides["truncation"], candidate.overrides["wffn1"])
            for candidate in candidates
        }
        self.assertIn((8, 18), pairs)
        self.assertIn((13, 20), pairs)

    def test_random_mode_rejects_ranges(self):
        from final_eval.config import parse_final_eval_settings

        with self.assertRaises(ValueError):
            parse_final_eval_settings(
                [
                    "--preset",
                    "mrpc-final-eval-only",
                    "--random",
                    "--range",
                    "truncation=8,9",
                    "--dry-run",
                ]
            )

    def test_random_flag_supplies_default_random_counts(self):
        from final_eval.config import parse_final_eval_settings

        settings = parse_final_eval_settings(
            ["--preset", "mrpc-final-eval-only", "--random", "--dry-run"]
        )

        self.assertTrue(settings.random_enabled)
        self.assertEqual(settings.perm_trials, 10)
        self.assertEqual(settings.stage2_budget_trials, 10)

    def test_embedded_training_call_forces_search_but_uses_preset_counts(self):
        from final_eval.embedded import run_embedded_final_eval

        captured = {}

        class FakeEvaluator:
            run_output_dir = str(Path("rl_results") / "persistent" / "rl" / "bert-base" / "mrpc" / "slug")
            search_algorithm = "rl"

            def log(self, _message):
                pass

        class FakeModule:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run(self, **kwargs):
                captured["run_kwargs"] = kwargs
                return {"summary_path": "fake.json"}

        with TemporaryDirectory() as tmpdir:
            result = run_embedded_final_eval(
                evaluator=FakeEvaluator(),
                search_best_stage1={"gelu": [4] * 12, "softmax": [6] * 12},
                search_best_stage2={"input_noise_scaling_factors": [20] * 12},
                baseline_stage1_gelu=[4] * 12,
                baseline_stage1_softmax=[6] * 12,
                baseline_noise_tot_c=1.0,
                limit_loss=1.0,
                limit_p=0.0,
                limit_s=0.0,
                preset_name="default",
                output_root=tmpdir,
                module_cls=FakeModule,
            )

        self.assertEqual(result["summary_path"], "fake.json")
        self.assertEqual(captured["config_source"], "search")
        self.assertEqual(captured["repeat_n"], 50)
        self.assertEqual(captured["permutation_trials"], 10)
        self.assertIn("final_eval", captured["results_dir"].replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
