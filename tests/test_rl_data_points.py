from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from json_utils import (
    json_default,
    read_json_file,
    stable_json_hash,
    stable_json_key,
    to_jsonable as shared_to_jsonable,
    write_json_file,
)
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

    def test_jsonl_writer_batches_os_flushes(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=Path(td),
                run_id="flush-test",
                stage="stage1",
                model_type="bert-base",
                dataset="mrpc",
                jsonl_buffer_size=4096,
                jsonl_flush_interval=2,
            )
            fake_handle = mock.MagicMock()

            with mock.patch.object(Path, "open", return_value=fake_handle) as open_mock:
                writer.write_episode({"episode": 0})
                writer.write_episode({"episode": 1})

            open_mock.assert_called_once()
            self.assertEqual(open_mock.call_args.kwargs["buffering"], 4096)
            self.assertEqual(fake_handle.write.call_count, 2)
            fake_handle.flush.assert_called_once()

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

    def test_json_default_is_shared_json_dump_adapter(self):
        payload = {"array": np.array([1, 2]), "scalar": np.float32(1.25), "path": Path("x/y")}
        encoded = json.dumps(payload, default=json_default, sort_keys=True)
        self.assertEqual(json.loads(encoded), {"array": [1, 2], "path": "x/y", "scalar": 1.25})

        with self.assertRaises(TypeError):
            json.dumps({"bad": object()}, default=json_default)

    def test_stable_json_key_and_hash_normalize_common_values(self):
        a = {"b": np.int64(2), "a": Path("x")}
        b = {"a": "x", "b": 2}
        self.assertEqual(stable_json_key(a), stable_json_key(b))
        self.assertEqual(stable_json_hash(a), stable_json_hash(b))

    def test_write_json_file_creates_parent_and_normalizes_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "payload.json"
            written = write_json_file(
                path,
                {"b": np.int64(2), "a": Path("x")},
                sort_keys=True,
            )

            text = path.read_text(encoding="utf-8")

        self.assertEqual(written, path)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual(json.loads(text), {"a": "x", "b": 2})

    def test_read_json_file_reads_artifact_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.json"
            path.write_text('{"a": 1, "b": [2]}', encoding="utf-8")

            self.assertEqual(read_json_file(path), {"a": 1, "b": [2]})

    def test_read_json_file_default_handles_optional_sidecars(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.json"
            broken = Path(td) / "broken.json"
            broken.write_text("{", encoding="utf-8")

            self.assertEqual(read_json_file(missing, default={}), {})
            self.assertEqual(read_json_file(broken, default=[]), [])

    def test_json_artifact_scripts_use_shared_writer(self):
        checks = {
            "scripts/blb_f0_scan_feasible_domain.py": "from json_utils import read_json_file, stable_json_hash, write_json_file",
            "scripts/blb_eval_action.py": "from json_utils import read_json_file, write_json_file",
            "scripts/blb_compare_optimizer_modes.py": "from json_utils import write_json_file",
            "scripts/optimization_evidence_bundle.py": "from json_utils import write_json_file",
            "scripts/gpu_utilization_report.py": "from json_utils import write_json_file",
            "scripts/server_resource_snapshot.py": "from json_utils import write_json_file",
            "scripts/stage1_parallel_report.py": "from json_utils import write_json_file",
            "scripts/stage2_reward_probe_scaling_report.py": "from json_utils import write_json_file",
            "scripts/report_fusion_count_map.py": "from json_utils import read_json_file, write_json_file",
            "scripts/run_fusion_count_action_eval.py": "from json_utils import read_json_file, write_json_file",
            "scripts/run_fusion_count_action_eval_rlpath.py": "from json_utils import read_json_file, to_jsonable, write_json_file",
            "scripts/blb_apply_precision_boost.py": "from json_utils import read_json_file, write_json_file",
            "scripts/blb_make_fusion_fixed_action_config.py": "from json_utils import read_json_file, write_json_file",
            "scripts/blb_make_run_manifest.py": "from json_utils import read_json_file, write_json_file",
            "scripts/blb_build_fusion_count_map.py": "from json_utils import write_json_file",
            "scripts/blb_orphan_slot_audit.py": "from json_utils import read_json_file, write_json_file",
            "scripts/stage1_plaintext_repeat_eval.py": "from json_utils import write_json_file",
        }
        for rel, needle in checks.items():
            with self.subTest(path=rel):
                text = (REPO_ROOT / rel).read_text(encoding="utf-8")
                self.assertIn(needle, text)
                self.assertNotIn("def _write_json(", text)

    def test_json_artifact_scripts_use_shared_reader(self):
        checks = {
            "scripts/fusion_count_action_eval_common.py": "from json_utils import read_json_file",
            "scripts/run_fusion_count_action_eval.py": "from json_utils import read_json_file",
            "scripts/run_fusion_count_action_eval_rlpath.py": "from json_utils import read_json_file",
            "scripts/blb_apply_precision_boost.py": "from json_utils import read_json_file",
            "scripts/blb_make_fusion_fixed_action_config.py": "from json_utils import read_json_file",
            "scripts/blb_make_run_manifest.py": "from json_utils import read_json_file",
            "scripts/blb_eval_action.py": "from json_utils import read_json_file",
            "scripts/blb_f0_scan_feasible_domain.py": "from json_utils import read_json_file",
            "blb_stage2_rl/action_space.py": "from json_utils import read_json_file",
            "blb_stage2_rl/action_mask.py": "from json_utils import read_json_file",
            "blb_stage2_rl/fusion_count_map.py": "from json_utils import read_json_file",
            "blb_stage2_rl/skeleton_stage_map.py": "from json_utils import read_json_file",
            "scripts/blb_diagnose_invalid_blocks.py": "from json_utils import read_json_file",
            "scripts/blb_diag_block2_boost.py": "from json_utils import read_json_file",
            "scripts/report_fusion_count_map.py": "from json_utils import read_json_file",
            "scripts/blb_verify_boosted_install.py": "from json_utils import read_json_file",
            "scripts/blb_verify_noise_install.py": "from json_utils import read_json_file",
            "scripts/blb_orphan_slot_audit.py": "from json_utils import read_json_file",
            "reports/generate_blb_mapping_html_reports.py": "from json_utils import read_json_file",
            "tools/paper_figures.py": "from json_utils import read_json_file",
            "Paean/action_grid.py": "from json_utils import read_json_file",
            "Paean/blb_action_eval.py": "from json_utils import read_json_file",
        }
        for rel, needle in checks.items():
            with self.subTest(path=rel):
                text = (REPO_ROOT / rel).read_text(encoding="utf-8")
                self.assertIn(needle, text)
                self.assertNotIn("json.loads(path.read_text", text)

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

    def test_stage2_diagnostics_reuses_primary_jsonl_handles(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_jsonl_reuse_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as td:
            recorder = module.RLDiagnosticsRecorder(
                output_dir=str(Path(td) / "progress"),
                num_layers=12,
                num_action_slots=3,
            )
            opened = {}

            def fake_open(path, *args, **kwargs):
                handle = mock.MagicMock()
                handle.__enter__.return_value = handle
                handle.__exit__.return_value = None
                opened.setdefault(str(path), []).append((handle, args, kwargs))
                return handle

            episode_stats = module.EpisodeStats(
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
            )
            update_stats = module.PPOUpdateStats(
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
            )

            with mock.patch("builtins.open", side_effect=fake_open):
                recorder.record_episode(
                    episode_stats=episode_stats,
                    full_action_vec=None,
                    is_new_best=False,
                    best_reward_so_far=1.0,
                )
                recorder.record_episode(
                    episode_stats=episode_stats,
                    full_action_vec=None,
                    is_new_best=False,
                    best_reward_so_far=1.0,
                )
                recorder.record_ppo_update(update_stats)
                recorder.record_ppo_update(update_stats)

            episode_handles = opened[recorder.episodes_path]
            ppo_handles = opened[recorder.ppo_updates_path]
            self.assertEqual(len(episode_handles), 1)
            self.assertEqual(len(ppo_handles), 1)
            self.assertEqual(episode_handles[0][2]["buffering"], 1024 * 1024)
            self.assertEqual(ppo_handles[0][2]["buffering"], 1024 * 1024)
            self.assertEqual(episode_handles[0][0].write.call_count, 2)
            self.assertEqual(ppo_handles[0][0].write.call_count, 2)


if __name__ == "__main__":
    unittest.main()
