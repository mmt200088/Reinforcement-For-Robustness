import json
import importlib.util
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from json_utils import to_jsonable as shared_to_jsonable
from rl_data_points import RLDataPointWriter, make_unique_run_id, to_jsonable


REPO_ROOT = Path(__file__).resolve().parents[1]


class RLDataPointWriterTest(unittest.TestCase):
    def test_writes_stage1_training_data_points_as_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            writer = RLDataPointWriter(
                root_dir=root,
                run_id="bert-large/mrpc test",
                stage="stage1",
                model_type="bert-large",
                dataset="mrpc",
            )

            writer.write_manifest({"layers": np.int64(24), "lr": np.float32(2e-5)})
            writer.write_step({"episode": 0, "state": np.array([1, 2]), "prob": np.float32(0.5)})
            writer.write_episode({"episode": 0, "reward": np.float64(1.25)})
            writer.write_ppo_update({"update": 1, "entropy": np.float32(0.9)})
            writer.write_summary({"stop_reason": "smoke"})
            writer.close()

            self.assertTrue((writer.run_dir / "manifest.json").is_file())
            self.assertTrue((writer.run_dir / "steps.jsonl").is_file())
            self.assertTrue((writer.run_dir / "episodes.jsonl").is_file())
            self.assertTrue((writer.run_dir / "ppo_updates.jsonl").is_file())
            self.assertTrue((writer.run_dir / "summary.json").is_file())

            manifest = json.loads((writer.run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["layers"], 24)
            self.assertEqual(manifest["stage"], "stage1")
            self.assertEqual(manifest["model_type"], "bert-large")
            self.assertEqual(manifest["dataset"], "mrpc")

            step = json.loads((writer.run_dir / "steps.jsonl").read_text().splitlines()[0])
            self.assertEqual(step["state"], [1, 2])
            self.assertEqual(step["prob"], 0.5)

    def test_to_jsonable_handles_nested_numpy_values(self):
        value = {"a": np.int64(3), "b": [np.float32(1.5), np.array([2, 4])]}
        self.assertEqual(to_jsonable(value), {"a": 3, "b": [1.5, [2, 4]]})
        self.assertIs(to_jsonable, shared_to_jsonable)

    def test_to_jsonable_handles_paths_and_dataclasses(self):
        @dataclass
        class Payload:
            path: Path
            value: np.int64

        self.assertEqual(
            to_jsonable(Payload(Path("reports/out.html"), np.int64(7))),
            {"path": "reports/out.html", "value": 7},
        )

    def test_to_jsonable_does_not_import_torch_for_json_native_scalars(self):
        import builtins

        value = {
            "episode": 1,
            "done": False,
            "reward": 1.25,
            "note": "ok",
            "missing": None,
            "nested": [2, True, "x"],
        }
        torch_imports = 0
        original_import = builtins.__import__

        def counting_import(name, *args, **kwargs):
            nonlocal torch_imports
            if name == "torch":
                torch_imports += 1
                raise ModuleNotFoundError("torch intentionally hidden")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=counting_import):
            self.assertEqual(to_jsonable(value), value)

        self.assertEqual(torch_imports, 0)

    def test_make_unique_run_id_preserves_base_and_separates_invocations(self):
        first = make_unique_run_id(
            "Parting Chapter/stage1/bert large mrpc",
            started_at="2026-06-26T01:00:00Z",
            pid=123,
        )
        second = make_unique_run_id(
            "Parting Chapter/stage1/bert large mrpc",
            started_at="2026-06-26T01:00:01Z",
            pid=124,
        )

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("Parting_Chapter_stage1_bert_large_mrpc__"))
        self.assertIn("20260626T010000Z", first)
        self.assertIn("pid123", first)

    def test_stage1_loop_integrates_structured_data_writer(self):
        source = (REPO_ROOT / "layer_importance_evaluator.py").read_text()
        self.assertIn("RLDataPointWriter", source)
        self.assertIn("make_unique_run_id", source)
        self.assertIn("stage1_data_writer.write_step", source)
        self.assertIn("stage1_data_writer.write_episode", source)
        self.assertIn("stage1_data_writer.write_ppo_update", source)
        self.assertIn("rl_training_data_points", source)

    def test_stage2_loop_integrates_structured_data_writer(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text()
        self.assertIn("RLDataPointWriter", source)
        self.assertIn("stage2_data_writer", source)
        self.assertIn("data_point_writer=stage2_data_writer", source)
        self.assertIn("rl_training_data_points", source)

    def test_stage2_diagnostics_can_mirror_to_structured_data_writer(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            writer = RLDataPointWriter(
                root_dir=root / "rl_training_data_points",
                run_id="stage2/mrpc run",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            recorder = module.RLDiagnosticsRecorder(
                output_dir=str(root / "progress"),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=writer,
            )

            recorder.record_episode(
                episode_stats=module.EpisodeStats(
                    episode=0,
                    total_reward=1.0,
                    terminal_reward=0.5,
                    per_step_sum=0.5,
                    valid_steps=47,
                    invalid_steps=0,
                    steps_taken=47,
                    total_bits=123,
                    fusion_count=1,
                    first_invalid_step=None,
                    first_invalid_block=None,
                    first_invalid_layer=None,
                    early_terminated=False,
                ),
                full_action_vec=np.array([0, 1, 2]),
                is_new_best=True,
                best_reward_so_far=1.0,
            )
            recorder.record_ppo_update(module.PPOUpdateStats(
                update=1,
                completed_episodes=1,
                policy_loss=0.1,
                value_loss=0.2,
                entropy=0.3,
                clip_fraction=0.4,
                n_samples=47,
                window_mean_return=1.0,
                window_max_return=1.0,
                window_min_return=1.0,
                window_mean_invalid=0.0,
                best_reward_so_far=1.0,
                elapsed_sec=2.0,
            ))
            recorder.finalize()
            writer.close()

            stage2_dir = writer.run_dir
            self.assertTrue((stage2_dir / "episodes.jsonl").is_file())
            self.assertTrue((stage2_dir / "ppo_updates.jsonl").is_file())
            self.assertTrue((stage2_dir / "summary.json").is_file())
            episode = json.loads((stage2_dir / "episodes.jsonl").read_text().splitlines()[0])
            self.assertEqual(episode["fusion_count"], 1)
            self.assertEqual(episode["full_action_vec"], [0, 1, 2])
            update = json.loads((stage2_dir / "ppo_updates.jsonl").read_text().splitlines()[0])
            self.assertEqual(update["completed_episodes"], 1)


if __name__ == "__main__":
    unittest.main()
