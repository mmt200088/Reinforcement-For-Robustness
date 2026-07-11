import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from json_utils import to_jsonable
from scripts.fusion_count_action_eval_common import (
    load_rlpath_action_configs,
    rlpath_config_group_key,
    rlpath_group_key,
    unique_rlpath_action_configs,
)


class NoCopyMapping:
    def __init__(self, payload):
        self.payload = dict(payload)

    def get(self, key, default=None):
        return self.payload.get(key, default)

    def __getitem__(self, key):
        return self.payload[key]

    def __iter__(self):
        raise AssertionError("unique config selection should not copy mappings")

    def __len__(self):
        raise AssertionError("unique config selection should not copy mappings")


class FakePredictionRecorder:
    def __init__(self, *, rows=None, finish_error=None):
        self.rows = list(rows or [{"dataset_idx": 7}, {"dataset_idx": 9}])
        self.finish_error = finish_error
        self.begin_calls = []
        self.finish_calls = []
        self.abort_count = 0

    def begin_group(self, *, run_seed, group):
        self.begin_calls.append((run_seed, group))

    def finish_group(self, *, trial_seeds):
        self.finish_calls.append(list(trial_seeds))
        if self.finish_error is not None:
            raise self.finish_error
        return self.rows

    def abort_group(self):
        self.abort_count += 1


class FakePredictionWriter:
    def __init__(self):
        self.written_rows = []

    def write_rows(self, rows):
        self.written_rows.extend(rows)


class FakeCaptureSeqEnv:
    def __init__(self, *, commit_error=None):
        self.base = SimpleNamespace(probe_noise_seed=None)
        self._step_idx = 0
        self._schedule = [SimpleNamespace(
            step_idx=0,
            layer_idx=0,
            block_idx=2,
            graph_key_suffix="block2_mrpc",
        )]
        self.commit_error = commit_error

    def reset(self, *, seed):
        self._step_idx = 0

    def evaluate_step(self, action, *, map_option_id_override=None):
        return {
            "valid": True,
            "fusion_count": 0,
            "boosted_field_values": None,
        }

    def commit_step(self, _eval_info, *, defer_terminal_forward):
        if self.commit_error is not None:
            raise self.commit_error
        self._step_idx = 1
        self.base.fixed_eval_trial_metrics = {
            "loss": [0.3, 0.31],
            "metric1": [0.88, 0.89],
            "metric2": [0.87, 0.88],
        }
        return np.zeros(1), 1.0, True, {
            "terminal_info": {
                "metrics": {"loss_mean": 0.305},
                "probe_diagnostics": {
                    "per_worker_trial_seeds": [[42, 2654435739]],
                },
            },
            "replan_application": {
                "applied_before_forward": True,
                "model_uses_replan_config": True,
            },
        }


class FakeDataset:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self._columns = None

    def shuffle(self, *, seed):
        return self

    def map(self, fn):
        for row in self._rows:
            row.update(fn(row))
        return self

    def rename_column(self, old, new):
        for row in self._rows:
            row[new] = row.pop(old)
        return self

    def set_format(self, *, type, columns):
        self._columns = tuple(columns)

    def __iter__(self):
        for row in self._rows:
            if self._columns is None:
                yield dict(row)
            else:
                yield {key: row[key] for key in self._columns}

    def __getitem__(self, index):
        row = self._rows[index]
        if self._columns is None:
            return dict(row)
        return {key: row[key] for key in self._columns}


class FakeHookHandle:
    def __init__(self):
        self.remove_count = 0

    def remove(self):
        self.remove_count += 1


class FakeHookModel:
    def __init__(self):
        self.register_calls = []
        self.handle = FakeHookHandle()

    def register_forward_hook(self, hook, *, with_kwargs):
        self.register_calls.append((hook, with_kwargs))
        return self.handle


class FusionCountActionEvalRLPathTest(unittest.TestCase):
    def test_group_seed_offsets_independent_groups(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        self.assertEqual(rlpath._group_seed(100, 2, shared=False), 102)
        self.assertEqual(rlpath._group_seed(100, 2, shared=True), 100)

    def test_trial_metric_payload_preserves_float_trials(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        payload = rlpath._trial_metric_payload(
            [0.2, 0.3],
            [0.8, 0.9],
            [0.7, 0.8],
        )

        self.assertEqual(
            payload,
            {
                "loss": [0.2, 0.3],
                "metric1": [0.8, 0.9],
                "metric2": [0.7, 0.8],
            },
        )

    def test_trial_metric_payload_is_strict_json_safe(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        payload = rlpath._trial_metric_payload(
            [np.float32(0.25), float("nan")],
            [np.float64(0.75), float("inf")],
            [np.float32(0.5), float("-inf")],
        )

        self.assertEqual(
            payload,
            {
                "loss": [0.25, {"non_finite": "nan"}],
                "metric1": [0.75, {"non_finite": "positive_infinity"}],
                "metric2": [0.5, {"non_finite": "negative_infinity"}],
            },
        )
        self.assertIs(type(payload["loss"][0]), float)
        self.assertIs(type(payload["metric1"][0]), float)
        self.assertIs(type(payload["metric2"][0]), float)
        json.dumps(payload, allow_nan=False)

    def test_fixed_map_option_uses_explicit_override_on_canonical_env_path(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        class FakeSeqEnv:
            def __init__(self):
                self.base = SimpleNamespace(probe_noise_seed=None)
                self._step_idx = 0
                self._schedule = [SimpleNamespace(
                    step_idx=0,
                    layer_idx=0,
                    block_idx=2,
                    graph_key_suffix="block2_mrpc",
                    map_option_ids=(1,),
                )]
                self.evaluated_actions = []
                self.map_option_overrides = []

            def reset(self, *, seed):
                self._step_idx = 0

            def evaluate_step(self, action, *, map_option_id_override=None):
                self.evaluated_actions.append(list(action))
                self.map_option_overrides.append(map_option_id_override)
                return {
                    "valid": True,
                    "fusion_count": int(map_option_id_override or 0),
                    "boosted_field_values": None,
                }

            def commit_step(self, _eval_info, *, defer_terminal_forward):
                self._step_idx = 1
                self.base.fixed_eval_trial_metrics = {
                    "loss": [0.29, 0.31],
                    "metric1": [0.87, 0.89],
                    "metric2": [0.86, 0.88],
                }
                terminal_info = {
                    "metrics": SimpleNamespace(
                        loss_mean=0.3,
                        loss_std=0.01,
                        metric1_mean=0.88,
                        metric1_std=0.01,
                        metric2_mean=0.87,
                        metric2_std=0.01,
                    ),
                    "fusion_action_steps": [{
                        "block_idx": 2,
                        "fusion_count": 0,
                        "k_value": 13,
                        "graph_key": "block2_mrpc_L0",
                    }],
                }
                return np.zeros(1), 1.0, True, {
                    "terminal_info": terminal_info,
                    "replan_application": {
                        "applied_before_forward": True,
                        "model_uses_replan_config": True,
                    },
                }

        env = FakeSeqEnv()
        old_deps = rlpath._RUNTIME_DEPS
        try:
            rlpath._RUNTIME_DEPS = {"K_LEVELS": (8, 9, 11, 13, 10, 12)}
            result = rlpath._run_group(
                env,
                {
                    "name": "fixed_b2",
                    "path": Path("fixed_b2.json"),
                    "baseline_k_index": 3,
                    "group": {"option_by_graph": {"block2_mrpc": 0}},
                },
                seed=42,
            )
        finally:
            rlpath._RUNTIME_DEPS = old_deps

        self.assertEqual(env.evaluated_actions, [[0, 3]])
        self.assertEqual(env.map_option_overrides, [0])
        self.assertEqual(result["step_records"][0]["policy_option_index"], 0)
        self.assertEqual(result["step_records"][0]["map_option_id"], 0)
        self.assertTrue(result["step_records"][0]["model_uses_replan_config"])
        self.assertTrue(
            result["step_records"][0]["replan_application"]["model_uses_replan_config"]
        )
        self.assertEqual(result["trial_metrics"]["loss"], [0.29, 0.31])
        self.assertEqual(result["fusion_total"], 0)
        self.assertNotIn("prediction_capture", result)

    def test_run_group_arms_finishes_and_writes_prediction_rows(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        env = FakeCaptureSeqEnv()
        recorder = FakePredictionRecorder()
        writer = FakePredictionWriter()
        cfg = {
            "name": "fixed_b2",
            "path": Path("fixed_b2.json"),
            "baseline_k_index": 3,
            "group": {"option_by_graph": {"block2_mrpc": 0}},
        }
        old_deps = rlpath._RUNTIME_DEPS
        try:
            rlpath._RUNTIME_DEPS = {"K_LEVELS": (8, 9, 11, 13, 10, 12)}
            result = rlpath._run_group(
                env,
                cfg,
                seed=42,
                prediction_recorder=recorder,
                prediction_writer=writer,
            )
        finally:
            rlpath._RUNTIME_DEPS = old_deps

        self.assertEqual(recorder.begin_calls, [(42, "fixed_b2")])
        self.assertEqual(recorder.finish_calls, [[42, 2654435739]])
        self.assertEqual(writer.written_rows, recorder.rows)
        self.assertEqual(result["prediction_capture"], {"row_count": 2})
        self.assertEqual(recorder.abort_count, 0)

    def test_run_group_aborts_prediction_recorder_on_commit_error(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        recorder = FakePredictionRecorder()
        writer = FakePredictionWriter()
        env = FakeCaptureSeqEnv(commit_error=RuntimeError("terminal failure"))
        cfg = {
            "name": "fixed_b2",
            "path": Path("fixed_b2.json"),
            "baseline_k_index": 3,
            "group": {},
        }
        old_deps = rlpath._RUNTIME_DEPS
        try:
            rlpath._RUNTIME_DEPS = {"K_LEVELS": (8, 9, 11, 13, 10, 12)}
            with self.assertRaisesRegex(RuntimeError, "terminal failure"):
                rlpath._run_group(
                    env,
                    cfg,
                    seed=42,
                    prediction_recorder=recorder,
                    prediction_writer=writer,
                )
        finally:
            rlpath._RUNTIME_DEPS = old_deps

        self.assertEqual(recorder.abort_count, 1)
        self.assertEqual(writer.written_rows, [])

    def test_run_group_aborts_prediction_recorder_on_finish_error(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        recorder = FakePredictionRecorder(
            finish_error=RuntimeError("terminal capture failure")
        )
        env = FakeCaptureSeqEnv()
        cfg = {
            "name": "fixed_b2",
            "path": Path("fixed_b2.json"),
            "baseline_k_index": 3,
            "group": {},
        }
        old_deps = rlpath._RUNTIME_DEPS
        try:
            rlpath._RUNTIME_DEPS = {"K_LEVELS": (8, 9, 11, 13, 10, 12)}
            with self.assertRaisesRegex(RuntimeError, "terminal capture failure"):
                rlpath._run_group(
                    env,
                    cfg,
                    seed=42,
                    prediction_recorder=recorder,
                    prediction_writer=FakePredictionWriter(),
                )
        finally:
            rlpath._RUNTIME_DEPS = old_deps

        self.assertEqual(recorder.finish_calls, [[42, 2654435739]])
        self.assertEqual(recorder.abort_count, 1)

    def test_prediction_capture_is_disabled_by_default(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        args = rlpath._parser().parse_args([
            "--action-dir", "actions",
            "--original-json", "original.json",
            "--output-json", "result.json",
            "--output-html", "result.html",
        ])

        self.assertEqual(args.prediction_jsonl, "")

    def test_tokenize_glue_preserves_mrpc_idx_in_identity_catalog(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        rows = [
            {"idx": 17, "sentence1": "left", "sentence2": "right", "label": 1},
            {"idx": 9, "sentence1": "up", "sentence2": "down", "label": 0},
        ]
        data = {
            "train": FakeDataset(rows),
            "validation": FakeDataset(rows),
        }

        def tokenizer(sentence1, sentence2, **kwargs):
            del sentence2, kwargs
            token = 11 if sentence1 == "left" else 12
            return {
                "input_ids": [101, token, 102],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
            }

        _train, validation, catalog = rlpath._tokenize_glue(
            data,
            task="mrpc",
            tokenizer=tokenizer,
            seed=42,
            include_identity_catalog=True,
        )

        self.assertEqual(catalog.dataset_indices, (17, 9))
        self.assertNotIn("idx", validation[0])
        resolver = catalog.new_trial_resolver()
        self.assertEqual(
            resolver.resolve([101, 11, 102], [1, 1, 1], [0, 0, 0], 1),
            17,
        )

    def test_tokenize_glue_default_does_not_build_identity_catalog(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        rows = [
            {"idx": 17, "sentence1": "left", "sentence2": "right", "label": 1},
        ]
        data = {
            "train": FakeDataset(rows),
            "validation": FakeDataset(rows),
        }

        def tokenizer(sentence1, sentence2, **kwargs):
            del sentence1, sentence2, kwargs
            return {
                "input_ids": [101, 11, 102],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
            }

        tokenized = rlpath._tokenize_glue(
            data,
            task="mrpc",
            tokenizer=tokenizer,
            seed=42,
        )

        self.assertEqual(len(tokenized), 2)

    def test_main_without_prediction_opt_in_registers_no_hook_or_metadata(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output_json = root / "result.json"
            model = FakeHookModel()
            evaluator = SimpleNamespace(model=model)
            config = {
                "name": "fixed_b2",
                "path": Path("fixed_b2.json"),
                "baseline_k_index": 3,
                "group": {},
                "group_key": "fixed",
            }
            argv = [
                "run_fusion_count_action_eval_rlpath.py",
                "--action-dir", str(root),
                "--original-json", str(root / "original.json"),
                "--output-json", str(output_json),
                "--output-html", str(root / "result.html"),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                rlpath, "load_rlpath_action_configs", return_value=[config]
            ), mock.patch.object(
                rlpath, "unique_rlpath_action_configs", return_value=[config]
            ), mock.patch.object(
                rlpath, "_build_evaluator", return_value=evaluator
            ), mock.patch.object(
                rlpath, "_build_seq_env", return_value=(SimpleNamespace(), {})
            ), mock.patch.object(
                rlpath, "read_json_file", return_value={"group_results": []}
            ), mock.patch.object(
                rlpath,
                "_run_group",
                return_value={"name": "fixed_b2", "metrics": {}},
            ), mock.patch.object(rlpath, "write_rendered_html"):
                self.assertEqual(rlpath.main(), 0)

            result = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(model.register_calls, [])
        self.assertNotIn("prediction_artifact", result)

    def test_main_removes_prediction_hook_when_group_evaluation_fails(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        writers = []

        class Recorder:
            def __init__(self, *, catalog, probe_batch_count):
                self.catalog = catalog
                self.probe_batch_count = probe_batch_count

            def hook(self, module, args, kwargs, output):
                pass

        class Writer:
            def __init__(self, path):
                self.path = path
                self.row_count = 0
                self.close_count = 0
                self.abort_count = 0
                writers.append(self)

            def close(self):
                self.close_count += 1

            def abort(self):
                self.abort_count += 1
                self.close()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = FakeHookModel()
            catalog = SimpleNamespace(dataset_indices=(17, 9))
            evaluator = SimpleNamespace(
                model=model,
                fixed_eval_identity_catalog=catalog,
            )
            seq_env = SimpleNamespace(
                base=SimpleNamespace(
                    env_cfg=SimpleNamespace(probe_batch_count=7),
                ),
            )
            config = {
                "name": "fixed_b2",
                "path": Path("fixed_b2.json"),
                "baseline_k_index": 3,
                "group": {},
                "group_key": "fixed",
            }
            argv = [
                "run_fusion_count_action_eval_rlpath.py",
                "--action-dir", str(root),
                "--original-json", str(root / "original.json"),
                "--output-json", str(root / "result.json"),
                "--output-html", str(root / "result.html"),
                "--prediction-jsonl", str(root / "predictions.jsonl"),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                rlpath, "load_rlpath_action_configs", return_value=[config]
            ), mock.patch.object(
                rlpath, "unique_rlpath_action_configs", return_value=[config]
            ), mock.patch.object(
                rlpath, "_build_evaluator", return_value=evaluator
            ), mock.patch.object(
                rlpath, "_build_seq_env", return_value=(seq_env, {})
            ), mock.patch.object(
                rlpath, "read_json_file", return_value={"group_results": []}
            ), mock.patch.object(
                rlpath, "_run_group", side_effect=RuntimeError("group failed")
            ), mock.patch.object(
                rlpath, "ForwardPredictionRecorder", Recorder, create=True
            ), mock.patch.object(
                rlpath, "PredictionJsonlWriter", Writer, create=True
            ):
                with self.assertRaisesRegex(RuntimeError, "group failed"):
                    rlpath.main()

        self.assertEqual(len(model.register_calls), 1)
        self.assertTrue(model.register_calls[0][1])
        self.assertEqual(model.handle.remove_count, 1)
        self.assertEqual(writers[0].abort_count, 1)
        self.assertGreaterEqual(writers[0].close_count, 1)

    def test_main_atomically_promotes_predictions_and_reports_committed_rows(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prediction_path = root / "predictions.jsonl"
            prediction_path.write_text("previous artifact\n", encoding="utf-8")
            output_json = root / "result.json"
            model = FakeHookModel()
            catalog = SimpleNamespace(dataset_indices=(17, 9))
            evaluator = SimpleNamespace(
                model=model,
                fixed_eval_identity_catalog=catalog,
            )
            seq_env = SimpleNamespace(
                base=SimpleNamespace(
                    env_cfg=SimpleNamespace(probe_batch_count=1),
                ),
            )
            config = {
                "name": "fixed_b2",
                "path": Path("fixed_b2.json"),
                "baseline_k_index": 3,
                "group": {},
                "group_key": "fixed",
            }

            def run_group(_env, _cfg, **kwargs):
                self.assertEqual(
                    prediction_path.read_text(encoding="utf-8"),
                    "previous artifact\n",
                )
                kwargs["prediction_writer"].write_rows([
                    {"dataset_idx": 17, "logits": [0.1, 0.2]},
                    {"dataset_idx": 9, "logits": [0.3, 0.4]},
                ])
                return {"name": "fixed_b2", "metrics": {}}

            argv = [
                "run_fusion_count_action_eval_rlpath.py",
                "--action-dir", str(root),
                "--original-json", str(root / "original.json"),
                "--output-json", str(output_json),
                "--output-html", str(root / "result.html"),
                "--prediction-jsonl", str(prediction_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                rlpath, "load_rlpath_action_configs", return_value=[config]
            ), mock.patch.object(
                rlpath, "unique_rlpath_action_configs", return_value=[config]
            ), mock.patch.object(
                rlpath, "_build_evaluator", return_value=evaluator
            ), mock.patch.object(
                rlpath, "_build_seq_env", return_value=(seq_env, {})
            ), mock.patch.object(
                rlpath, "read_json_file", return_value={"group_results": []}
            ), mock.patch.object(
                rlpath, "_run_group", side_effect=run_group
            ), mock.patch.object(rlpath, "write_rendered_html"):
                self.assertEqual(rlpath.main(), 0)

            payloads = [
                json.loads(line)
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
            ]
            result = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual([row["dataset_idx"] for row in payloads], [17, 9])
        self.assertEqual(
            result["prediction_artifact"],
            {
                "schema_version": "fusion-count-per-example-v1",
                "path": str(prediction_path),
                "row_count": 2,
                "dataset_indices": [17, 9],
            },
        )

    def test_main_cleans_partial_temp_and_preserves_final_on_later_group_failure(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        writers = []

        class TrackingWriter(rlpath.PredictionJsonlWriter):
            def __init__(self, path):
                super().__init__(path)
                self.abort_count = 0
                self.close_count = 0
                writers.append(self)

            def close(self):
                self.close_count += 1
                super().close()

            def abort(self):
                self.abort_count += 1
                super().abort()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prediction_path = root / "predictions.jsonl"
            prediction_path.write_text("previous artifact\n", encoding="utf-8")
            model = FakeHookModel()
            evaluator = SimpleNamespace(
                model=model,
                fixed_eval_identity_catalog=SimpleNamespace(dataset_indices=(17,)),
            )
            seq_env = SimpleNamespace(
                base=SimpleNamespace(
                    env_cfg=SimpleNamespace(probe_batch_count=1),
                ),
            )
            configs = [
                {
                    "name": "first",
                    "path": Path("first.json"),
                    "baseline_k_index": 3,
                    "group": {},
                    "group_key": "first",
                },
                {
                    "name": "second",
                    "path": Path("second.json"),
                    "baseline_k_index": 3,
                    "group": {},
                    "group_key": "second",
                },
            ]
            call_count = 0

            def run_group(_env, cfg, **kwargs):
                nonlocal call_count
                call_count += 1
                kwargs["prediction_writer"].write_rows([
                    {"group": cfg["name"], "dataset_idx": 17},
                ])
                if call_count == 2:
                    raise RuntimeError("later group failed")
                return {"name": cfg["name"], "metrics": {}}

            argv = [
                "run_fusion_count_action_eval_rlpath.py",
                "--action-dir", str(root),
                "--original-json", str(root / "original.json"),
                "--output-json", str(root / "result.json"),
                "--output-html", str(root / "result.html"),
                "--prediction-jsonl", str(prediction_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                rlpath, "load_rlpath_action_configs", return_value=configs
            ), mock.patch.object(
                rlpath, "unique_rlpath_action_configs", return_value=configs
            ), mock.patch.object(
                rlpath, "_build_evaluator", return_value=evaluator
            ), mock.patch.object(
                rlpath, "_build_seq_env", return_value=(seq_env, {})
            ), mock.patch.object(
                rlpath, "read_json_file", return_value={"group_results": []}
            ), mock.patch.object(
                rlpath, "_run_group", side_effect=run_group
            ), mock.patch.object(
                rlpath, "PredictionJsonlWriter", TrackingWriter
            ):
                with self.assertRaisesRegex(RuntimeError, "later group failed"):
                    rlpath.main()

            remaining = set(root.iterdir())
            final_contents = prediction_path.read_text(encoding="utf-8")

        self.assertEqual(call_count, 2)
        self.assertEqual(final_contents, "previous artifact\n")
        self.assertEqual(remaining, {prediction_path})
        self.assertEqual(writers[0].abort_count, 1)
        self.assertGreaterEqual(writers[0].close_count, 1)

    def test_run_group_clears_stale_trials_and_requires_committed_replan_evidence(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        class FakeSeqEnv:
            def __init__(self):
                self.base = SimpleNamespace(probe_noise_seed=None)
                self._step_idx = 0
                self._schedule = [SimpleNamespace(
                    step_idx=0,
                    layer_idx=0,
                    block_idx=2,
                    graph_key_suffix="block2_mrpc",
                )]
                self.commit_count = 0

            def reset(self, *, seed):
                self._step_idx = 0

            def evaluate_step(self, action, *, map_option_id_override=None):
                return {
                    "valid": True,
                    "fusion_count": 0,
                    "boosted_field_values": None,
                }

            def commit_step(self, _eval_info, *, defer_terminal_forward):
                self._step_idx = 1
                self.commit_count += 1
                info = {"terminal_info": {}}
                if self.commit_count == 1:
                    self.base.fixed_eval_trial_metrics = {
                        "loss": [0.3],
                        "metric1": [0.88],
                        "metric2": [0.87],
                    }
                    info["replan_application"] = {
                        "applied_before_forward": True,
                        "model_uses_replan_config": True,
                    }
                return np.zeros(1), 0.0, True, info

        cfg = {
            "name": "fixed_b2",
            "path": Path("fixed_b2.json"),
            "baseline_k_index": 3,
            "group": {"option_by_graph": {"block2_mrpc": 0}},
        }
        env = FakeSeqEnv()
        old_deps = rlpath._RUNTIME_DEPS
        try:
            rlpath._RUNTIME_DEPS = {"K_LEVELS": (8, 9, 11, 13, 10, 12)}
            first = rlpath._run_group(env, cfg, seed=42)
            second = rlpath._run_group(env, cfg, seed=43)
        finally:
            rlpath._RUNTIME_DEPS = old_deps

        self.assertEqual(first["trial_metrics"]["loss"], [0.3])
        self.assertTrue(first["step_records"][0]["model_uses_replan_config"])
        self.assertEqual(second["trial_metrics"], {})
        self.assertEqual(second["step_records"][0]["replan_application"], {})
        self.assertFalse(second["step_records"][0]["model_uses_replan_config"])

    def test_module_import_is_dependency_light(self):
        code = """
import builtins

real_import = builtins.__import__
blocked = ("torch", "transformers", "blb_stage2_rl")

def guarded_import(name, *args, **kwargs):
    if name.startswith(blocked):
        raise AssertionError(f"heavy dependency imported: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import scripts.run_fusion_count_action_eval_rlpath as rlpath
assert callable(rlpath.load_rlpath_action_configs)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_load_action_configs_does_not_retain_full_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "candidate.json"
            config.write_text(
                json.dumps(
                    {
                        "baseline_k_index": 2,
                        "group": {
                            "name": "candidate",
                            "option_by_graph": {"block2_mrpc": 1},
                            "option_by_step": {"0": 1},
                        },
                        "large_unused_payload": [{"i": i} for i in range(64)],
                    }
                ),
                encoding="utf-8",
            )

            configs = load_rlpath_action_configs(root)

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "candidate")
        self.assertEqual(configs[0]["baseline_k_index"], 2)
        self.assertEqual(configs[0]["group"]["option_by_graph"], {"block2_mrpc": 1})
        self.assertNotIn("payload", configs[0])
        self.assertEqual(configs[0]["group_key"], rlpath_group_key(configs[0]))

    def test_load_action_configs_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_summary.json").write_text("{}", encoding="utf-8")
            (root / "._candidate.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "candidate.json").write_text(
                json.dumps(
                    {
                        "baseline_k_index": 2,
                        "group": {"name": "candidate", "option_by_graph": {"block2_mrpc": 1}},
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("action config loading should not use Path.glob"),
            ):
                configs = load_rlpath_action_configs(root)

        self.assertEqual([cfg["name"] for cfg in configs], ["candidate"])

    def test_group_key_uses_retained_group_and_baseline_fields(self):
        left = {
            "name": "a",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 2,
        }
        right = {
            "name": "b",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 3,
        }

        self.assertNotEqual(rlpath_group_key(left), rlpath_group_key(right))

    def test_jsonable_reuses_json_native_nested_payloads(self):
        steps = [
            {"step_idx": i, "valid": bool(i % 2), "nested": {"fusion_count": i}}
            for i in range(8)
        ]

        converted = to_jsonable(steps, stringify_unknown=True, preserve_native=True)

        self.assertIs(converted, steps)
        self.assertIs(converted[0]["nested"], steps[0]["nested"])

    def test_jsonable_converts_only_branches_that_need_conversion(self):
        import numpy as np

        steps = [{"step_idx": i, "valid": True} for i in range(8)]
        payload = {"steps": steps, "array": np.array([1, 2, 3])}

        converted = to_jsonable(payload, stringify_unknown=True, preserve_native=True)

        self.assertIsNot(converted, payload)
        self.assertIs(converted["steps"], steps)
        self.assertEqual(converted["array"], [1, 2, 3])

    def test_jsonable_does_not_reconvert_changed_list_item(self):
        class CountedString:
            calls = 0

            def __str__(self):
                type(self).calls += 1
                return "converted"

        native = {"step_idx": 1, "valid": True}
        converted = to_jsonable([native, CountedString()], stringify_unknown=True, preserve_native=True)

        self.assertEqual(converted, [native, "converted"])
        self.assertIs(converted[0], native)
        self.assertEqual(CountedString.calls, 1)

    def test_jsonable_does_not_reconvert_changed_dict_item(self):
        class CountedString:
            calls = 0

            def __str__(self):
                type(self).calls += 1
                return "converted"

        native = {"step_idx": 1, "valid": True}
        converted = to_jsonable({"native": native, "custom": CountedString()}, stringify_unknown=True, preserve_native=True)

        self.assertEqual(converted, {"native": native, "custom": "converted"})
        self.assertIs(converted["native"], native)
        self.assertEqual(CountedString.calls, 1)

    def test_unique_configs_reuses_first_config_without_copying(self):
        first = NoCopyMapping({
            "name": "first",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 2,
        })
        duplicate = NoCopyMapping({
            "name": "duplicate",
            "group": {"option_by_graph": {"block2_mrpc": 1}, "option_by_step": {"0": 1}},
            "baseline_k_index": 2,
        })

        unique = unique_rlpath_action_configs([first, duplicate])

        self.assertEqual(len(unique), 1)
        self.assertIs(next(iter(unique)), first)

    def test_unique_configs_reuses_cached_group_key(self):
        first = {"name": "first", "group_key": "same"}
        duplicate = {"name": "duplicate", "group_key": "same"}

        with mock.patch(
            "scripts.fusion_count_action_eval_common.rlpath_group_key",
            side_effect=AssertionError("cached group_key should be reused"),
        ):
            unique = unique_rlpath_action_configs([first, duplicate])

        self.assertEqual(len(unique), 1)
        self.assertIs(next(iter(unique)), first)

    def test_config_group_key_reuses_cached_group_key(self):
        self.assertEqual(rlpath_config_group_key({"group_key": "cached"}), "cached")

    def test_rlpath_script_has_no_local_common_wrappers(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        forbidden = [
            "def _resolve(",
            "def _json_int_list(",
            "def _iter_action_config_paths(",
            "def _load_action_configs(",
            "def _group_key(",
            "def _config_group_key(",
            "def _unique_configs(",
        ]
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("resolve_repo_path", source)
        self.assertIn("unique_rlpath_action_configs", source)

    def test_main_streams_stdout_json_without_json_dumps_string(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        main_source = source[source.index("def main("):]

        self.assertIn("json.dump(", main_source)
        self.assertIn("sys.stdout", main_source)
        self.assertIn('sys.stdout.write("\\n")', main_source)
        self.assertNotIn("print(json.dumps(", main_source)

    def test_main_reuses_static_stage1_default_json_strings(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        parser_source = source[source.index("def _parser("):source.index("def main(")]

        self.assertIn("DEFAULT_STAGE1_GELU_JSON = json.dumps(DEFAULT_STAGE1_GELU)", source)
        self.assertIn("DEFAULT_STAGE1_SOFTMAX_JSON = json.dumps(DEFAULT_STAGE1_SOFTMAX)", source)
        self.assertIn("default=DEFAULT_STAGE1_GELU_JSON", parser_source)
        self.assertIn("default=DEFAULT_STAGE1_SOFTMAX_JSON", parser_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_GELU)", parser_source)
        self.assertNotIn("default=json.dumps(DEFAULT_STAGE1_SOFTMAX)", parser_source)

    def test_main_streams_html_report_without_full_render_string_write(self):
        import scripts.run_fusion_count_action_eval_rlpath as rlpath

        source = Path(rlpath.__file__).read_text(encoding="utf-8")
        main_source = source[source.index("def main("):]

        self.assertTrue(hasattr(rlpath, "write_rendered_html"))
        self.assertTrue(hasattr(rlpath, "_HtmlPartsWriter"))
        self.assertIn("_HtmlPartsWriter(output_html)", source)
        self.assertNotIn("output_html.write_text(_render_html(combined)", main_source)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "parts.html"
            writer = rlpath._HtmlPartsWriter(path)
            writer.append("alpha")
            writer.extend(["beta", "gamma"])
            writer.close()

            self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nbeta\ngamma")
