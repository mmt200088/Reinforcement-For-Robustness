from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import rl_data_points
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
    def test_diagnostics_restore_reconciles_primary_and_structured_mirror(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_reconcile_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        def episode(index):
            return module.EpisodeStats(
                episode=index,
                total_reward=1.0,
                terminal_reward=1.0,
                per_step_sum=0.0,
                valid_steps=12,
                invalid_steps=0,
                steps_taken=12,
                total_bits=0,
                fusion_count=24,
                first_invalid_step=None,
                first_invalid_block=None,
                first_invalid_layer=None,
                early_terminated=False,
            )

        with tempfile.TemporaryDirectory() as td:
            progress = Path(td) / "progress"
            writer = RLDataPointWriter(
                root_dir=Path(td) / "points",
                run_id="reconcile",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            first = module.RLDiagnosticsRecorder(
                output_dir=str(progress),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=writer,
                strict_writes=True,
            )
            first.record_episode(
                episode_stats=episode(0),
                full_action_vec=np.array([0, 1, 2]),
                is_new_best=False,
                best_reward_so_far=1.0,
            )
            first.flush_mandatory()
            # Simulate a crash after the primary append but before its mirror.
            writer_episode_path = writer.run_dir / "episodes.jsonl"
            writer_episode_path.write_text("", encoding="utf-8")
            first._close_primary_jsonl()
            writer.close()

            resumed_writer = RLDataPointWriter(
                root_dir=Path(td) / "points",
                run_id="reconcile",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            resumed = module.RLDiagnosticsRecorder(
                output_dir=str(progress),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=resumed_writer,
                strict_writes=True,
            )
            restored = resumed.restore_existing()
            resumed.flush_mandatory()

            mirror_rows = [
                json.loads(line)
                for line in writer_episode_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(restored["episodes"], 1)
        self.assertEqual([row["episode"] for row in mirror_rows], [0])
        self.assertEqual(mirror_rows[0]["full_action_vec"], [0, 1, 2])

    def test_diagnostics_restore_rejects_primary_mirror_conflict(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_conflict_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as td:
            progress = Path(td) / "progress"
            diagnostics = progress / "diagnostics"
            diagnostics.mkdir(parents=True)
            primary = {
                "episode": 0,
                "total_reward": 1.0,
                "terminal_reward": 1.0,
                "per_step_sum": 0.0,
                "valid_steps": 12,
                "invalid_steps": 0,
                "steps_taken": 12,
                "total_bits": 0,
                "fusion_count": 24,
                "first_invalid_step": None,
                "first_invalid_block": None,
                "first_invalid_layer": None,
                "early_terminated": False,
            }
            (diagnostics / "episodes.jsonl").write_text(
                json.dumps(primary) + "\n", encoding="utf-8",
            )
            writer = RLDataPointWriter(
                root_dir=Path(td) / "points",
                run_id="conflict",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            writer.write_episode({**primary, "total_reward": 2.0})
            writer.flush()
            recorder = module.RLDiagnosticsRecorder(
                output_dir=str(progress),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=writer,
                strict_writes=True,
            )

            with self.assertRaisesRegex(RuntimeError, "conflict"):
                recorder.restore_existing()

    def test_diagnostics_restore_backfills_primary_from_structured_tail(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_primary_backfill_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        payload = {
            "episode": 0,
            "total_reward": 1.0,
            "terminal_reward": 1.0,
            "per_step_sum": 0.0,
            "valid_steps": 12,
            "invalid_steps": 0,
            "steps_taken": 12,
            "total_bits": 0,
            "fusion_count": 24,
            "first_invalid_step": None,
            "first_invalid_block": None,
            "first_invalid_layer": None,
            "early_terminated": False,
            "full_action_vec": [0, 1, 2],
        }
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=Path(td) / "points",
                run_id="backfill",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            writer.write_episode(payload)
            writer.flush()
            recorder = module.RLDiagnosticsRecorder(
                output_dir=str(Path(td) / "progress"),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=writer,
                strict_writes=True,
            )

            restored = recorder.restore_existing()

        self.assertEqual(restored, {"episodes": 1, "ppo_updates": 0})
        self.assertEqual(recorder._all_episode_returns, [1.0])

    def test_diagnostics_checkpoint_sizes_roll_back_both_jsonl_trees(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_sizes_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=Path(td) / "points",
                run_id="sizes",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            recorder = module.RLDiagnosticsRecorder(
                output_dir=str(Path(td) / "progress"),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=writer,
                strict_writes=True,
            )
            recorder._write_primary_jsonl(recorder.episodes_path, {"episode": 0})
            writer.write_episode({"episode": 0})
            recorder.flush_mandatory()
            committed = recorder.committed_jsonl_sizes()
            recorder._write_primary_jsonl(recorder.episodes_path, {"episode": 1})
            writer.write_episode({"episode": 1})
            recorder.flush_mandatory()
            recorder._close_primary_jsonl()
            writer.close()

            resumed_writer = RLDataPointWriter(
                root_dir=Path(td) / "points",
                run_id="sizes",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            resumed = module.RLDiagnosticsRecorder(
                output_dir=str(Path(td) / "progress"),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=resumed_writer,
                strict_writes=True,
            )
            resumed.recover_to_checkpoint_sizes(committed)

            primary_rows = list(
                json.loads(line)
                for line in Path(resumed.episodes_path).read_text().splitlines()
            )
            mirror_rows = list(
                json.loads(line)
                for line in (resumed_writer.run_dir / "episodes.jsonl").read_text().splitlines()
            )

        self.assertEqual([row["episode"] for row in primary_rows], [0])
        self.assertEqual([row["episode"] for row in mirror_rows], [0])

    def test_layerwise_robust_episode_is_strict_json_and_auditable(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=td,
                run_id="layerwise-robust",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            probabilities = {
                name: 0.9
                for name in (
                    "loss_precision_probability",
                    "metric1_precision_probability",
                    "metric2_precision_probability",
                    "loss_stability_probability",
                    "metric1_stability_probability",
                    "metric2_stability_probability",
                )
            }
            payload = {
                "episode": 7,
                "fresh_trials": {
                    "loss": [0.3] * 5,
                    "metric1": [0.88] * 5,
                    "metric2": [0.87] * 5,
                    "seeds": [11, 12, 13, 14, 15],
                },
                "pooled_trials": {
                    "loss": [0.3] * 25,
                    "metric1": [0.88] * 25,
                    "metric2": [0.87] * 25,
                    "seeds": list(range(25)),
                },
                "fresh_constraint_probabilities": probabilities,
                "pooled_constraint_probabilities": probabilities,
                "fresh_trial_count": 5,
                "pooled_trial_count": 25,
                "reward_evidence": "fresh_trials",
                "ranking_evidence": "pooled_prefix_trials",
                "constraint_thresholds": {
                    "online": 0.50, "promotion": 0.80, "final": 0.95,
                },
                "variable_cost": 0.625,
                "layer_action_matrix": [[0, 0, 3, 3, 3, 3]] + [[1, 3, 3, 3, 3, 3]] * 11,
                "block4_entropy": 0.08,
                "k_entropy": 0.09,
                "promotion_trial_count": 25,
                "promotion_status": "promoted",
                "convergence_state": {"converged": False, "stall_update_windows": 4},
            }
            writer.write_episode(payload)
            writer.write_summary({"strict_best_assessment": probabilities})
            writer.close()

            text = (writer.run_dir / "episodes.jsonl").read_text(encoding="utf-8")
            row = json.loads(text, parse_constant=lambda value: self.fail(value))
            self.assertEqual(len(row["layer_action_matrix"]), 12)
            self.assertEqual(len(row["fresh_constraint_probabilities"]), 6)
            self.assertEqual(len(row["pooled_constraint_probabilities"]), 6)
            self.assertEqual(row["fresh_trials"]["seeds"], [11, 12, 13, 14, 15])
            self.assertEqual(len(row["pooled_trials"]["seeds"]), 25)

    def test_nonfinite_structured_values_are_written_as_json_null(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=td,
                run_id="finite-json",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            writer.write_episode({"episode": 0, "bad": float("nan")})
            writer.close()
            row = json.loads(
                (writer.run_dir / "episodes.jsonl").read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(value),
            )
            self.assertIsNone(row["bad"])

    def test_nonfinite_numpy_manifest_and_summary_values_are_json_null(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=td,
                run_id="finite-numpy-json",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )
            payload = {
                "scalar": np.float32(float("nan")),
                "array": np.asarray([1.0, float("inf")]),
            }
            writer.write_manifest(payload)
            writer.write_summary(payload)
            writer.close()

            manifest = json.loads(
                (writer.run_dir / "manifest.json").read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(value),
            )
            summary = json.loads(
                (writer.run_dir / "summary.json").read_text(encoding="utf-8"),
                parse_constant=lambda value: self.fail(value),
            )

        self.assertIsNone(manifest["scalar"])
        self.assertEqual(manifest["array"], [1.0, None])
        self.assertIsNone(summary["scalar"])
        self.assertEqual(summary["array"], [1.0, None])

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
            newline_writes = [
                call for call in fake_handle.write.call_args_list if call.args == ("\n",)
            ]
            self.assertEqual(len(newline_writes), 2)
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

    def test_stable_json_hash_streams_without_materializing_key(self):
        import json_utils

        payload = {"b": [np.int64(2), Path("x")], "a": {"flag": True}}
        expected = hashlib.sha256(stable_json_key(payload).encode("utf-8")).hexdigest()

        with mock.patch.object(
            json_utils,
            "stable_json_key",
            side_effect=AssertionError("stable_json_hash should stream canonical JSON"),
        ):
            self.assertEqual(json_utils.stable_json_hash(payload), expected)

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

    def test_write_json_file_streams_to_file_handle(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "payload.json"
            original_write_text = Path.write_text

            def fail_write_text(p, *args, **kwargs):
                if Path(p) == path:
                    raise AssertionError("write_json_file should not materialize full JSON text")
                return original_write_text(p, *args, **kwargs)

            with mock.patch.object(Path, "write_text", fail_write_text):
                written = write_json_file(path, {"a": np.int64(1)})

            text = path.read_text(encoding="utf-8")

        self.assertEqual(written, path)
        self.assertEqual(json.loads(text), {"a": 1})
        self.assertTrue(text.endswith("\n"))

    def test_training_data_manifest_and_summary_use_streaming_json_helpers(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=Path(td),
                run_id="stream-json-test",
                stage="stage1",
                model_type="bert-base",
                dataset="mrpc",
            )
            writer.write_manifest({"first": np.int64(1)})
            manifest_path = writer.run_dir / "manifest.json"
            summary_path = writer.run_dir / "summary.json"
            original_read_text = Path.read_text
            original_write_text = Path.write_text

            def fail_read_text(path, *args, **kwargs):
                if Path(path) == manifest_path:
                    raise AssertionError("manifest merge should stream existing JSON")
                return original_read_text(path, *args, **kwargs)

            def fail_write_text(path, *args, **kwargs):
                if Path(path) in {manifest_path, summary_path}:
                    raise AssertionError("training data JSON artifacts should stream writes")
                return original_write_text(path, *args, **kwargs)

            with (
                mock.patch.object(Path, "read_text", fail_read_text),
                mock.patch.object(Path, "write_text", fail_write_text),
            ):
                writer.write_manifest({"second": np.float32(2.5)})
                writer.write_summary({"done": True})
            writer.close()

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["first"], 1)
        self.assertEqual(manifest["second"], 2.5)
        self.assertEqual(summary, {"done": True})

    def test_training_data_jsonl_writer_streams_rows_without_json_dumps(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=Path(td),
                run_id="stream-jsonl-test",
                stage="stage1",
                model_type="bert-base",
                dataset="mrpc",
            )

            with mock.patch.object(
                rl_data_points.json,
                "dumps",
                side_effect=AssertionError("JSONL writer should stream through iterencode"),
            ):
                writer.write_episode({"episode": np.int64(1), "reward": np.float32(0.5)})
            writer.close()

            rows = (writer.run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]), {"episode": 1, "reward": 0.5})

    def test_training_data_jsonl_writer_avoids_eager_payload_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=Path(td),
                run_id="direct-json-encoder-test",
                stage="stage1",
                model_type="bert-base",
                dataset="mrpc",
            )
            payload = {
                "z": np.array([1, 2]),
                "a": Path("x/y"),
                "nested": {
                    "scalar": np.float32(1.25),
                    "native": [True, None, 3],
                },
            }

            try:
                with mock.patch.object(
                    rl_data_points,
                    "to_jsonable",
                    side_effect=AssertionError(
                        "JSONEncoder.default should normalize only non-native leaves"
                    ),
                ):
                    writer.write_episode(payload)
            finally:
                writer.close()

            row = (writer.run_dir / "episodes.jsonl").read_text(encoding="utf-8")

        self.assertEqual(
            row,
            '{"a": "x/y", "nested": {"native": [true, null, 3], '
            '"scalar": 1.25}, "z": [1, 2]}\n',
        )

    def test_read_json_file_reads_artifact_payload(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.json"
            path.write_text('{"a": 1, "b": [2]}', encoding="utf-8")

            self.assertEqual(read_json_file(path), {"a": 1, "b": [2]})

    def test_read_json_file_uses_streaming_json_loader(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "payload.json"
            path.write_text('{"a": 1, "b": [2]}', encoding="utf-8")
            original_read_text = Path.read_text

            def fail_read_text(p, *args, **kwargs):
                if Path(p) == path:
                    raise AssertionError("read_json_file should not materialize the whole file")
                return original_read_text(p, *args, **kwargs)

            with mock.patch.object(Path, "read_text", fail_read_text):
                payload = read_json_file(path)

        self.assertEqual(payload, {"a": 1, "b": [2]})

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
            "scripts/diagnose_block4_fusion_install.py": "from json_utils import write_json_file",
            "scripts/stage1_plaintext_repeat_eval.py": "from json_utils import write_json_file",
            "reports/generate_blb_mapping_html_reports.py": "from json_utils import read_json_file, write_json_file",
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
            "scripts/render_fusion_count_slots_eval_report.py": "from json_utils import read_json_file",
            "tools/paper_figures.py": "from json_utils import read_json_file",
            "Paean/action_grid.py": "from json_utils import read_json_file",
            "Paean/blb_action_eval.py": "from json_utils import read_json_file",
            "generate_glue_submission.py": "from json_utils import read_json_file",
        }
        for rel, needle in checks.items():
            with self.subTest(path=rel):
                text = (REPO_ROOT / rel).read_text(encoding="utf-8")
                self.assertIn(needle, text)
                self.assertNotIn("json.loads(path.read_text", text)
                self.assertNotIn("json.loads(open(", text)

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

    def test_manifest_writes_merge_instead_of_replace(self):
        with tempfile.TemporaryDirectory() as td:
            writer = RLDataPointWriter(
                root_dir=Path(td),
                run_id="merge test",
                stage="stage2",
                model_type="bert-base",
                dataset="mrpc",
            )

            writer.write_manifest({"baseline_preflight_trial_count": 5})
            writer.write_manifest({"schema_version": 2})
            writer.close()

            manifest = json.loads((writer.run_dir / "manifest.json").read_text())
            self.assertEqual(manifest["baseline_preflight_trial_count"], 5)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["stage"], "stage2")

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
            recorder.set_meta({
                "stage2_k_trials": 5,
                "reward_design": "stage1_aligned",
                "borderline_retest_enabled": False,
                "borderline_retest_trials_multiplier": 1,
                "baseline_preflight_metrics": {
                    "trial_count": 5,
                    "limit_tolerance": 0.001,
                    "stability_tolerance": 3.5,
                    "loss_mean": 0.365,
                    "loss_std": 0.002,
                    "metric1_mean": 0.864,
                    "metric1_std": 0.001,
                    "metric2_mean": 0.864,
                    "metric2_std": 0.001,
                    "loss_threshold": 0.366,
                    "metric1_threshold": 0.863,
                    "metric2_threshold": 0.863,
                    "loss_std_threshold": 0.01,
                    "metric1_std_threshold": 0.01,
                    "metric2_std_threshold": 0.01,
                },
            })

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
            manifest = json.loads((stage2_dir / "manifest.json").read_text())
            self.assertEqual(manifest["stage2_k_trials"], 5)
            self.assertEqual(manifest["baseline_preflight_trial_count"], 5)
            self.assertEqual(manifest["precision_tolerance"], 0.001)
            self.assertEqual(manifest["stability_tolerance"], 3.5)
            self.assertEqual(manifest["reward_design"], "stage1_aligned")
            self.assertIs(manifest["borderline_retest_enabled"], False)
            self.assertEqual(manifest["borderline_retest_trials_multiplier"], 1)
            self.assertEqual(manifest["trainer_gate_baseline"]["trial_count"], 5)
            episode = json.loads((stage2_dir / "episodes.jsonl").read_text().splitlines()[0])
            self.assertEqual(episode["fusion_count"], 1)
            self.assertEqual(episode["full_action_vec"], [0, 1, 2])
            update = json.loads((stage2_dir / "ppo_updates.jsonl").read_text().splitlines()[0])
            self.assertEqual(update["completed_episodes"], 1)

    def test_layerwise_strict_diagnostics_propagate_mandatory_write_failures(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_strict_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        writer = mock.Mock()
        writer.write_manifest.side_effect = OSError("manifest disk failure")
        with tempfile.TemporaryDirectory() as td:
            recorder = module.RLDiagnosticsRecorder(
                output_dir=str(Path(td) / "progress"),
                num_layers=12,
                num_action_slots=3,
                data_point_writer=writer,
                strict_writes=True,
            )
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                recorder.set_meta({"reward_design": "robust_constrained"})

            writer.write_manifest.side_effect = None
            with mock.patch.object(
                recorder,
                "_write_primary_jsonl",
                side_effect=OSError("episode disk failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "episodes.jsonl"):
                    recorder.record_episode(
                        episode_stats=module.EpisodeStats(
                            episode=0,
                            total_reward=1.0,
                            terminal_reward=1.0,
                            per_step_sum=0.0,
                            valid_steps=12,
                            invalid_steps=0,
                            steps_taken=12,
                            total_bits=0,
                            fusion_count=24,
                            first_invalid_step=None,
                            first_invalid_block=None,
                            first_invalid_layer=None,
                            early_terminated=False,
                        ),
                        full_action_vec=np.array([0, 1, 2]),
                        is_new_best=False,
                        best_reward_so_far=1.0,
                    )

    def test_layerwise_branch_enables_strict_diagnostics_writes(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text()
        self.assertIn("strict_writes=True", source)

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

            with (
                mock.patch("builtins.open", side_effect=fake_open),
                mock.patch.object(
                    module.json,
                    "dumps",
                    side_effect=AssertionError("primary diagnostics JSONL should stream rows"),
                ),
            ):
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
            self.assertEqual(episode_handles[0][0].writelines.call_count, 2)
            self.assertEqual(ppo_handles[0][0].writelines.call_count, 2)
            episode_newlines = [
                call for call in episode_handles[0][0].write.call_args_list
                if call.args == ("\n",)
            ]
            ppo_newlines = [
                call for call in ppo_handles[0][0].write.call_args_list
                if call.args == ("\n",)
            ]
            self.assertEqual(len(episode_newlines), 2)
            self.assertEqual(len(ppo_newlines), 2)

    def test_stage2_diagnostics_restore_rebuilds_full_history_without_reappend(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_resume_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        def episode(module, index):
            return module.EpisodeStats(
                episode=index,
                total_reward=float(index + 1),
                terminal_reward=float(index + 1),
                per_step_sum=0.0,
                valid_steps=12,
                invalid_steps=0,
                steps_taken=12,
                total_bits=0,
                fusion_count=24 + index,
                first_invalid_step=None,
                first_invalid_block=None,
                first_invalid_layer=None,
                early_terminated=False,
            )

        def update(module, index):
            return module.PPOUpdateStats(
                update=index,
                completed_episodes=index * 120,
                policy_loss=0.1,
                value_loss=0.2,
                entropy=0.3,
                clip_fraction=0.4,
                n_samples=1440,
                window_mean_return=1.0,
                window_max_return=1.0,
                window_min_return=1.0,
                window_mean_invalid=0.0,
                best_reward_so_far=1.0,
                elapsed_sec=2.0,
            )

        with tempfile.TemporaryDirectory() as td:
            output = str(Path(td) / "progress")
            first = module.RLDiagnosticsRecorder(
                output_dir=output, num_layers=12, num_action_slots=3,
            )
            first.record_episode(
                episode_stats=episode(module, 0),
                full_action_vec=np.array([0, 1, 2]),
                is_new_best=False,
                best_reward_so_far=1.0,
            )
            first.record_episode(
                episode_stats=episode(module, 1),
                full_action_vec=np.array([1, 2, 0]),
                is_new_best=False,
                best_reward_so_far=2.0,
            )
            first.record_ppo_update(update(module, 1))
            first.finalize()

            resumed = module.RLDiagnosticsRecorder(
                output_dir=output, num_layers=12, num_action_slots=3,
            )
            restored = resumed.restore_existing()
            self.assertEqual(restored, {"episodes": 2, "ppo_updates": 1})
            self.assertEqual(resumed._all_episode_returns, [1.0, 2.0])
            self.assertEqual(len(resumed._ppo_history), 1)
            self.assertEqual(int(resumed._action_hist.sum()), 6)

            resumed.record_episode(
                episode_stats=episode(module, 2),
                full_action_vec=np.array([2, 0, 1]),
                is_new_best=False,
                best_reward_so_far=3.0,
            )
            resumed.record_ppo_update(update(module, 2))
            resumed.finalize()

            episode_rows = (
                Path(output) / "diagnostics" / "episodes.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            update_rows = (
                Path(output) / "diagnostics" / "ppo_updates.jsonl"
            ).read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(episode_rows), 3)
        self.assertEqual(len(update_rows), 2)

    def test_stage2_diagnostics_restore_accepts_invalid_nonfinite_metrics(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_invalid_resume_for_test",
            REPO_ROOT / "blb_stage2_rl" / "diagnostics.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as td:
            output = str(Path(td) / "progress")
            first = module.RLDiagnosticsRecorder(
                output_dir=output, num_layers=12, num_action_slots=3,
                strict_writes=True,
            )
            first.record_episode(
                episode_stats=module.EpisodeStats(
                    episode=0,
                    total_reward=-5.0,
                    terminal_reward=-5.0,
                    per_step_sum=0.0,
                    valid_steps=11,
                    invalid_steps=1,
                    steps_taken=12,
                    total_bits=0,
                    fusion_count=24,
                    first_invalid_step=11,
                    first_invalid_block=4,
                    first_invalid_layer=11,
                    early_terminated=True,
                    terminal_priority=1,
                    terminal_loss_mean=float("inf"),
                    terminal_loss_std=float("inf"),
                ),
                full_action_vec=np.array([0, 1, 2]),
                is_new_best=False,
                best_reward_so_far=-5.0,
            )
            first.finalize()

            persisted = json.loads(
                (Path(output) / "diagnostics" / "episodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIsNone(persisted["terminal_loss_mean"])
            self.assertIsNone(persisted["terminal_loss_std"])

            resumed = module.RLDiagnosticsRecorder(
                output_dir=output, num_layers=12, num_action_slots=3,
                strict_writes=True,
            )
            restored = resumed.restore_existing()

        self.assertEqual(restored, {"episodes": 1, "ppo_updates": 0})
        self.assertEqual(resumed._all_priority, [1])
        self.assertEqual(len(resumed._top_candidates), 1)
        restored_payload = resumed._top_candidates[0][2]
        self.assertTrue(np.isinf(restored_payload["terminal_loss_mean"]))
        self.assertTrue(np.isinf(restored_payload["terminal_loss_std"]))

    def test_layerwise_checkpoint_flushes_mandatory_logs_before_save(self):
        source = (REPO_ROOT / "blb_stage2_rl" / "sequential_runner.py").read_text()
        flush_pos = source.index("diag_recorder.committed_jsonl_sizes()")
        checkpoint_pos = source.index("torch.save(checkpoint, tmp_path)", flush_pos)
        self.assertLess(flush_pos, checkpoint_pos)

    def test_stage2_diagnostics_streams_human_report_writes(self):
        spec = importlib.util.spec_from_file_location(
            "blb_stage2_diagnostics_report_stream_for_test",
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
                full_action_vec=None,
                is_new_best=False,
                best_reward_so_far=1.0,
            )
            recorder._close_primary_jsonl()

            report_paths = {
                recorder.summary_md_path + ".tmp",
                recorder.pareto_html_path + ".tmp",
            }
            handles = {}
            original_open = open

            def fake_open(path, *args, **kwargs):
                if str(path) not in report_paths:
                    return original_open(path, *args, **kwargs)
                handle = mock.MagicMock()
                handle.__enter__.return_value = handle
                handle.__exit__.return_value = None

                def reject_full_document_write(text):
                    if isinstance(text, str) and text.count("\n") > 3:
                        raise AssertionError("human diagnostics reports should stream lines")

                handle.write.side_effect = reject_full_document_write
                handles[str(path)] = handle
                return handle

            with (
                mock.patch("builtins.open", side_effect=fake_open),
                mock.patch.object(module.os, "replace") as replace_mock,
            ):
                recorder._write_summary_md()
                recorder._write_pareto_html([{"episode": 0, "total_reward": 1.0}])

            self.assertEqual(set(handles), report_paths)
            for handle in handles.values():
                self.assertGreater(handle.write.call_count, 1)
            self.assertEqual(replace_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
