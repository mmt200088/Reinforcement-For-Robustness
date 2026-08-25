from pathlib import Path
import io
import json
import tempfile
from contextlib import redirect_stdout
import unittest
from unittest import mock

from rfr.cli import evaluate as run_final_eval
from rfr.cli.evaluation_config import FinalEvalSettings
from rfr.cli.evaluate import configuration_lines, estimate_workload


class FinalEvalWorkloadEstimateTest(unittest.TestCase):
    def test_rl_final_eval_command_disables_stage2_training_episode_limit(self):
        command = run_final_eval.build_command(FinalEvalSettings(algorithm="rl"))

        stage2_index = command.index("--stage2_rl_episodes")
        self.assertEqual(command[stage2_index + 1], "0")

    def test_batch_manifest_counts_candidates_in_one_launcher_process(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "batch.json"
            manifest.write_text(
                json.dumps({
                    "schema_version": "paean_action_batch_v1",
                    "candidates": [
                        {"name": "first", "action_config": "first.json"},
                        {"name": "second", "action_config": "second.json"},
                    ],
                }),
                encoding="utf-8",
            )
            settings = FinalEvalSettings(
                action_config=str(manifest),
                cost_match_count=0,
            )

            workload = estimate_workload(settings)

        self.assertEqual(workload["selected_config_count"], 2)
        self.assertEqual(workload["total_config_count"], 2)
        self.assertEqual(workload["launcher_processes"], 1)

    def test_counts_action_range_product_and_random_controls(self):
        settings = FinalEvalSettings(
            repeat=3,
            random_enabled=True,
            perm_trials=2,
            cost_trials=1,
            budget_trials=0,
            stage1_budget_trials=1,
            stage2_budget_trials=1,
            cost_match_count=4,
            action_ranges=(
                "block3.truncation=8,9,10",
                "layer2.block5.wffn1_sf=18,20",
            ),
        )

        workload = estimate_workload(settings)

        self.assertEqual(workload["action_range_dimensions"], 2)
        self.assertEqual(workload["selected_config_count"], 6)
        self.assertEqual(workload["legacy_random_control_count"], 5)
        self.assertEqual(workload["cost_matched_random_count"], 4)
        self.assertEqual(workload["total_config_count"], 15)
        self.assertEqual(workload["total_repeated_evaluations"], 45)
        self.assertTrue(workload["gpu_parallelism_candidate"])

    def test_default_cost_match_count_is_counted_even_without_legacy_random(self):
        settings = FinalEvalSettings(random_enabled=False, cost_match_count=50)

        workload = estimate_workload(settings)

        self.assertEqual(workload["selected_config_count"], 1)
        self.assertEqual(workload["legacy_random_control_count"], 0)
        self.assertEqual(workload["cost_matched_random_count"], 50)
        self.assertEqual(workload["total_config_count"], 51)
        self.assertTrue(workload["gpu_parallelism_candidate"])

    def test_configuration_lines_include_workload_summary(self):
        settings = FinalEvalSettings(
            repeat=5,
            random_enabled=False,
            cost_match_count=0,
            action_ranges=("block3.truncation=8,9",),
        )

        lines = configuration_lines(
            settings,
            Path("/tmp/paean-final-eval"),
            ["python", "-m", "rfr.cli.run"],
            include_command=False,
        )
        text = "\n".join(lines)

        self.assertIn("  workload:", text)
        self.assertIn("    selected_configs: 2", text)
        self.assertIn("    total_configs: 2", text)
        self.assertIn("    repeat: 5", text)
        self.assertIn("    total_repeated_evaluations: 10", text)
        self.assertIn("    gpu_parallelism_candidate: true", text)

    def test_list_presets_streams_only_first_line(self):
        with self.subTest("list-presets avoids whole-file read"):
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                preset_dir = Path(td)
                preset = preset_dir / "stream.conf"
                preset.write_text("# stream preset\n--repeat 5\n", encoding="utf-8")
                original_read_text = Path.read_text

                def fail_read_text(path, *_args, **_kwargs):
                    if Path(path) == preset:
                        raise AssertionError("--list-presets should not read the whole preset")
                    return original_read_text(path, *_args, **_kwargs)

                buffer = io.StringIO()
                with (
                    mock.patch.object(run_final_eval, "PRESET_DIR", preset_dir),
                    mock.patch.object(Path, "read_text", fail_read_text),
                    redirect_stdout(buffer),
                ):
                    rc = run_final_eval.main(["--list-presets"])

        self.assertEqual(rc, 0)
        self.assertIn("stream", buffer.getvalue())
        self.assertIn("stream preset", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
