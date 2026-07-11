import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


class TensorLike:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def _single_example_catalog():
    from scripts.fusion_count_prediction_capture import ExampleIdentityCatalog

    return ExampleIdentityCatalog.from_tokenized_rows([
        {
            "idx": 10,
            "input_ids": [101, 11, 102],
            "token_type_ids": [0, 0, 0],
            "labels": 0,
        }
    ])


def _capture_single_example(recorder, *, logits=None, output=None):
    if output is None:
        output = SimpleNamespace(
            logits=np.asarray(logits if logits is not None else [[2.0, -1.0]])
        )
    recorder.hook(
        None,
        (),
        {
            "input_ids": np.asarray([[101, 11, 102, 0]]),
            "attention_mask": np.asarray([[1, 1, 1, 0]]),
            "token_type_ids": np.asarray([[0, 0, 0, 0]]),
            "labels": np.asarray([0]),
        },
        output,
    )


class ExampleIdentityCatalogTest(unittest.TestCase):
    def test_schema_constant_is_versioned(self):
        from scripts.fusion_count_prediction_capture import PREDICTION_ROW_SCHEMA

        self.assertEqual(PREDICTION_ROW_SCHEMA, "fusion-count-per-example-v1")

    def test_duplicate_token_rows_resolve_in_dataset_order_once_per_trial(self):
        from scripts.fusion_count_prediction_capture import ExampleIdentityCatalog

        rows = [
            {
                "idx": 7,
                "input_ids": [101, 10, 102, 0],
                "attention_mask": [1, 1, 1, 0],
                "token_type_ids": [0, 0, 1, 8],
                "labels": 1,
            },
            {
                "idx": 9,
                "input_ids": [101, 10, 102, 0],
                "attention_mask": [1, 1, 1, 0],
                "token_type_ids": [0, 0, 1, 9],
                "labels": 1,
            },
        ]
        catalog = ExampleIdentityCatalog.from_tokenized_rows(rows)
        resolver = catalog.new_trial_resolver()

        self.assertEqual(catalog.dataset_indices, (7, 9))
        with self.assertRaises(AttributeError):
            catalog.dataset_indices = (9, 7)
        self.assertEqual(
            resolver.resolve(
                [101, 10, 102, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 1, 4, 5],
                1,
            ),
            7,
        )
        self.assertEqual(
            resolver.resolve(
                [101, 10, 102, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 3],
                1,
            ),
            9,
        )
        resolver.assert_complete()

        next_trial = catalog.new_trial_resolver()
        self.assertEqual(
            next_trial.resolve(
                [101, 10, 102, 0],
                [1, 1, 1, 0],
                [0, 0, 1, 0],
                1,
            ),
            7,
        )

    def test_identity_key_includes_token_types_and_gold_label(self):
        from scripts.fusion_count_prediction_capture import ExampleIdentityCatalog

        catalog = ExampleIdentityCatalog.from_tokenized_rows([
            {
                "idx": 1,
                "input_ids": [101, 20, 102],
                "token_type_ids": [0, 0, 0],
                "labels": 0,
            },
            {
                "idx": 2,
                "input_ids": [101, 20, 102],
                "token_type_ids": [0, 1, 1],
                "labels": 0,
            },
            {
                "idx": 3,
                "input_ids": [101, 20, 102],
                "token_type_ids": [0, 0, 0],
                "labels": 1,
            },
        ])
        resolver = catalog.new_trial_resolver()

        self.assertEqual(
            resolver.resolve([101, 20, 102], [1, 1, 1], [0, 1, 1], 0),
            2,
        )
        self.assertEqual(
            resolver.resolve([101, 20, 102], [1, 1, 1], [0, 0, 0], 1),
            3,
        )
        self.assertEqual(
            resolver.resolve([101, 20, 102], [1, 1, 1], [0, 0, 0], 0),
            1,
        )
        resolver.assert_complete()

    def test_resolver_rejects_missing_and_reused_identities(self):
        catalog = _single_example_catalog()
        resolver = catalog.new_trial_resolver()

        with self.assertRaisesRegex(ValueError, "identity"):
            resolver.resolve([101, 99, 102], [1, 1, 1], [0, 0, 0], 0)

        self.assertEqual(
            resolver.resolve([101, 11, 102], [1, 1, 1], [0, 0, 0], 0),
            10,
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            resolver.resolve([101, 11, 102], [1, 1, 1], [0, 0, 0], 0)

    def test_resolver_rejects_incomplete_trial(self):
        resolver = _single_example_catalog().new_trial_resolver()

        with self.assertRaisesRegex(ValueError, "incomplete"):
            resolver.assert_complete()

    def test_catalog_rejects_missing_or_duplicate_dataset_indices(self):
        from scripts.fusion_count_prediction_capture import ExampleIdentityCatalog

        with self.assertRaises((KeyError, ValueError)):
            ExampleIdentityCatalog.from_tokenized_rows([
                {"input_ids": [101], "token_type_ids": [0], "labels": 0}
            ])

        with self.assertRaisesRegex(ValueError, "dataset.*idx"):
            ExampleIdentityCatalog.from_tokenized_rows([
                {"idx": 1, "input_ids": [101], "token_type_ids": [0], "labels": 0},
                {"idx": 1, "input_ids": [102], "token_type_ids": [0], "labels": 1},
            ])

    def test_catalog_rejects_malformed_identity_integer_values(self):
        from scripts.fusion_count_prediction_capture import ExampleIdentityCatalog

        int64 = np.iinfo(np.int64)
        valid_row = {
            "idx": 10,
            "input_ids": [101, 11, 102],
            "attention_mask": [1, 1, 1],
            "token_type_ids": [0, 0, 0],
            "labels": 0,
        }
        cases = (
            ("fractional input_ids", "input_ids", [101, 11.5, 102], "input_ids"),
            ("NaN attention_mask", "attention_mask", [1, float("nan"), 1], "attention_mask"),
            ("infinite token_type_ids", "token_type_ids", [0, float("inf"), 0], "token_type_ids"),
            ("boolean input_ids", "input_ids", [101, True, 102], "input_ids"),
            ("boolean label", "labels", False, "label"),
            ("above int64", "input_ids", [101, int64.max + 1, 102], "input_ids"),
            ("below int64", "token_type_ids", [0, int64.min - 1, 0], "token_type_ids"),
        )

        for name, field, value, error_field in cases:
            with self.subTest(name=name):
                row = {**valid_row, field: value}
                with self.assertRaisesRegex(ValueError, error_field):
                    ExampleIdentityCatalog.from_tokenized_rows([row])

    def test_catalog_rejects_malformed_dataset_indices(self):
        from scripts.fusion_count_prediction_capture import ExampleIdentityCatalog

        int64 = np.iinfo(np.int64)
        invalid_indices = (
            10.5,
            float("nan"),
            float("inf"),
            True,
            int64.max + 1,
            int64.min - 1,
        )

        for dataset_idx in invalid_indices:
            with self.subTest(dataset_idx=dataset_idx):
                with self.assertRaisesRegex(ValueError, "dataset idx"):
                    ExampleIdentityCatalog.from_tokenized_rows([{
                        "idx": dataset_idx,
                        "input_ids": [101, 11, 102],
                        "token_type_ids": [0, 0, 0],
                        "labels": 0,
                    }])


class ForwardPredictionRecorderTest(unittest.TestCase):
    def test_partitions_exact_batches_into_trials_and_emits_complete_rows(self):
        from scripts.fusion_count_prediction_capture import (
            ExampleIdentityCatalog,
            ForwardPredictionRecorder,
        )

        catalog = ExampleIdentityCatalog.from_tokenized_rows([
            {"idx": 10, "input_ids": [101, 11, 102], "token_type_ids": [0, 0, 0], "labels": 0},
            {"idx": 11, "input_ids": [101, 12, 102], "token_type_ids": [0, 0, 0], "labels": 1},
            {"idx": 12, "input_ids": [101, 13, 102], "token_type_ids": [0, 0, 0], "labels": 1},
        ])
        recorder = ForwardPredictionRecorder(catalog=catalog, probe_batch_count=2)
        recorder.begin_group(run_seed=100, group="all_fusion0")

        for _trial_index in range(2):
            recorder.hook(
                None,
                (),
                {
                    "input_ids": np.asarray([
                        [101, 11, 102, 0],
                        [101, 12, 102, 0],
                    ]),
                    "attention_mask": np.asarray([
                        [1, 1, 1, 0],
                        [1, 1, 1, 0],
                    ]),
                    "token_type_ids": np.zeros((2, 4), dtype=np.int64),
                    "labels": np.asarray([0, 1]),
                },
                SimpleNamespace(logits=np.asarray([
                    [0.1, -0.2],
                    [0.5, 0.5],
                ], dtype=np.float64)),
            )
            recorder.hook(
                None,
                (),
                {
                    "input_ids": [[101, 13, 102, 0]],
                    "attention_mask": [[1, 1, 1, 0]],
                    "token_type_ids": [[0, 0, 0, 0]],
                    "labels": [1],
                },
                SimpleNamespace(logits=[[-0.5, 0.8]]),
            )

        rows = recorder.finish_group(trial_seeds=[123, 456])

        self.assertEqual(len(rows), 6)
        self.assertEqual([row["dataset_idx"] for row in rows], [10, 11, 12] * 2)
        self.assertEqual([row["trial_index"] for row in rows], [0, 0, 0, 1, 1, 1])
        self.assertEqual([row["trial_seed"] for row in rows], [123, 123, 123, 456, 456, 456])
        self.assertEqual([row["probe_position"] for row in rows], [0, 1, 2] * 2)
        self.assertEqual([row["predicted_label"] for row in rows], [0, 0, 1] * 2)
        self.assertEqual([row["correct"] for row in rows], [True, False, True] * 2)
        self.assertEqual(rows[0]["schema_version"], "fusion-count-per-example-v1")
        self.assertEqual(rows[0]["run_seed"], 100)
        self.assertEqual(rows[0]["group"], "all_fusion0")
        self.assertEqual(rows[0]["input_ids"], [101, 11, 102, 0])
        self.assertEqual(rows[0]["attention_mask"], [1, 1, 1, 0])
        self.assertEqual(rows[0]["token_type_ids"], [0, 0, 0, 0])
        self.assertEqual(rows[0]["gold_label"], 0)
        self.assertEqual(rows[0]["logits"], [float(np.float32(0.1)), float(np.float32(-0.2))])
        self.assertIs(type(rows[0]["logits"][0]), float)

    def test_accepts_tensor_like_inputs_and_tuple_output_fallback(self):
        from scripts.fusion_count_prediction_capture import ForwardPredictionRecorder

        recorder = ForwardPredictionRecorder(
            catalog=_single_example_catalog(),
            probe_batch_count=1,
        )
        recorder.begin_group(run_seed=5, group="fixed_b2")
        recorder.hook(
            None,
            (),
            {
                "input_ids": TensorLike([[101, 11, 102, 0]]),
                "attention_mask": TensorLike([[1, 1, 1, 0]]),
                "token_type_ids": TensorLike([[0, 0, 0, 0]]),
                "labels": TensorLike([0]),
            },
            (TensorLike([0.25]), TensorLike([[1.5, -2.0]])),
        )

        rows = recorder.finish_group(trial_seeds=[77])

        self.assertEqual(rows[0]["dataset_idx"], 10)
        self.assertEqual(rows[0]["logits"], [1.5, -2.0])

    def test_token_type_ids_are_optional_when_absent_from_catalog_and_forward(self):
        from scripts.fusion_count_prediction_capture import (
            ExampleIdentityCatalog,
            ForwardPredictionRecorder,
        )

        catalog = ExampleIdentityCatalog.from_tokenized_rows([
            {"idx": 4, "input_ids": [101, 30, 102], "labels": 1}
        ])
        recorder = ForwardPredictionRecorder(catalog=catalog, probe_batch_count=1)
        recorder.begin_group(run_seed=8, group="fixed_b5")
        recorder.hook(
            None,
            (),
            {
                "input_ids": [[101, 30, 102, 0]],
                "attention_mask": [[1, 1, 1, 0]],
                "labels": [1],
            },
            SimpleNamespace(logits=[[-1.0, 2.0]]),
        )

        rows = recorder.finish_group(trial_seeds=[9])

        self.assertNotIn("token_type_ids", rows[0])
        self.assertEqual(rows[0]["dataset_idx"], 4)

    def test_hook_rejects_malformed_identity_integer_values(self):
        from scripts.fusion_count_prediction_capture import ForwardPredictionRecorder

        int64 = np.iinfo(np.int64)
        valid_kwargs = {
            "input_ids": [[101, 11, 102, 0]],
            "attention_mask": [[1, 1, 1, 0]],
            "token_type_ids": [[0, 0, 0, 0]],
            "labels": [0],
        }
        cases = (
            ("fractional input_ids", "input_ids", [[101, 11.5, 102, 0]], "input_ids"),
            ("NaN attention_mask", "attention_mask", [[1, float("nan"), 1, 0]], "attention_mask"),
            ("infinite token_type_ids", "token_type_ids", [[0, float("inf"), 0, 0]], "token_type_ids"),
            ("boolean input_ids", "input_ids", [[101, True, 102, 0]], "input_ids"),
            ("boolean label", "labels", np.asarray([False], dtype=np.bool_), "labels"),
            ("above int64", "input_ids", [[101, int64.max + 1, 102, 0]], "input_ids"),
            ("below int64", "token_type_ids", [[0, int64.min - 1, 0, 0]], "token_type_ids"),
        )

        for name, field, value, error_field in cases:
            with self.subTest(name=name):
                recorder = ForwardPredictionRecorder(
                    catalog=_single_example_catalog(),
                    probe_batch_count=1,
                )
                recorder.begin_group(run_seed=1, group=name)
                with self.assertRaisesRegex(ValueError, error_field):
                    recorder.hook(
                        None,
                        (),
                        {**valid_kwargs, field: value},
                        SimpleNamespace(logits=[[2.0, -1.0]]),
                    )

    def test_accepts_mathematically_integral_float_identity_values(self):
        from scripts.fusion_count_prediction_capture import (
            ExampleIdentityCatalog,
            ForwardPredictionRecorder,
        )

        catalog = ExampleIdentityCatalog.from_tokenized_rows([{
            "idx": np.float64(10.0),
            "input_ids": np.asarray([101.0, 11.0, 102.0]),
            "attention_mask": np.asarray([1.0, 1.0, 1.0]),
            "token_type_ids": np.asarray([0.0, 0.0, 0.0]),
            "labels": np.float64(0.0),
        }])
        recorder = ForwardPredictionRecorder(catalog=catalog, probe_batch_count=1)
        recorder.begin_group(run_seed=1, group="integral_float")
        recorder.hook(
            None,
            (),
            {
                "input_ids": np.asarray([[101.0, 11.0, 102.0, 0.0]]),
                "attention_mask": np.asarray([[1.0, 1.0, 1.0, 0.0]]),
                "token_type_ids": np.asarray([[0.0, 0.0, 0.0, 0.0]]),
                "labels": np.asarray([0.0]),
            },
            SimpleNamespace(logits=[[2.0, -1.0]]),
        )

        rows = recorder.finish_group(trial_seeds=[2])

        self.assertEqual(rows[0]["dataset_idx"], 10)
        self.assertEqual(rows[0]["input_ids"], [101, 11, 102, 0])

    def test_rejects_hook_and_group_lifecycle_misuse_and_abort_discards_capture(self):
        from scripts.fusion_count_prediction_capture import ForwardPredictionRecorder

        recorder = ForwardPredictionRecorder(
            catalog=_single_example_catalog(),
            probe_batch_count=1,
        )
        with self.assertRaisesRegex(RuntimeError, "active group"):
            _capture_single_example(recorder)

        recorder.begin_group(run_seed=1, group="first")
        with self.assertRaisesRegex(RuntimeError, "already active"):
            recorder.begin_group(run_seed=2, group="second")
        _capture_single_example(recorder)
        recorder.abort_group()

        recorder.begin_group(run_seed=2, group="second")
        _capture_single_example(recorder)
        rows = recorder.finish_group(trial_seeds=[3])
        self.assertEqual(len(rows), 1)
        with self.assertRaisesRegex(RuntimeError, "active group"):
            recorder.finish_group(trial_seeds=[3])

    def test_rejects_nonpositive_probe_batch_count(self):
        from scripts.fusion_count_prediction_capture import ForwardPredictionRecorder

        with self.assertRaisesRegex(ValueError, "probe_batch_count"):
            ForwardPredictionRecorder(catalog=_single_example_catalog(), probe_batch_count=0)

    def test_rejects_missing_or_extra_forward_batches(self):
        from scripts.fusion_count_prediction_capture import ForwardPredictionRecorder

        missing = ForwardPredictionRecorder(
            catalog=_single_example_catalog(),
            probe_batch_count=2,
        )
        missing.begin_group(run_seed=1, group="missing")
        _capture_single_example(missing)
        with self.assertRaisesRegex(ValueError, "forward"):
            missing.finish_group(trial_seeds=[10])

        extra = ForwardPredictionRecorder(
            catalog=_single_example_catalog(),
            probe_batch_count=1,
        )
        extra.begin_group(run_seed=1, group="extra")
        _capture_single_example(extra)
        _capture_single_example(extra)
        with self.assertRaisesRegex(ValueError, "forward"):
            extra.finish_group(trial_seeds=[10])

    def test_rejects_missing_and_reused_dataset_identities_during_finish(self):
        from scripts.fusion_count_prediction_capture import (
            ExampleIdentityCatalog,
            ForwardPredictionRecorder,
        )

        catalog = ExampleIdentityCatalog.from_tokenized_rows([
            {"idx": 10, "input_ids": [101, 11, 102], "token_type_ids": [0, 0, 0], "labels": 0},
            {"idx": 11, "input_ids": [101, 12, 102], "token_type_ids": [0, 0, 0], "labels": 1},
        ])
        recorder = ForwardPredictionRecorder(catalog=catalog, probe_batch_count=1)
        recorder.begin_group(run_seed=1, group="reuse")
        recorder.hook(
            None,
            (),
            {
                "input_ids": [[101, 11, 102, 0], [101, 11, 102, 0]],
                "attention_mask": [[1, 1, 1, 0], [1, 1, 1, 0]],
                "token_type_ids": [[0, 0, 0, 0], [0, 0, 0, 0]],
                "labels": [0, 0],
            },
            SimpleNamespace(logits=[[2.0, -1.0], [1.0, -0.5]]),
        )

        with self.assertRaisesRegex(ValueError, "identity"):
            recorder.finish_group(trial_seeds=[10])

    def test_hook_rejects_missing_inputs_and_shape_mismatches(self):
        from scripts.fusion_count_prediction_capture import ForwardPredictionRecorder

        valid_kwargs = {
            "input_ids": [[101, 11, 102, 0]],
            "attention_mask": [[1, 1, 1, 0]],
            "token_type_ids": [[0, 0, 0, 0]],
            "labels": [0],
        }
        cases = {
            "missing input_ids": ({key: value for key, value in valid_kwargs.items() if key != "input_ids"}, [[2.0, -1.0]]),
            "missing attention_mask": ({key: value for key, value in valid_kwargs.items() if key != "attention_mask"}, [[2.0, -1.0]]),
            "missing labels": ({key: value for key, value in valid_kwargs.items() if key != "labels"}, [[2.0, -1.0]]),
            "input rank": ({**valid_kwargs, "input_ids": [101, 11, 102, 0]}, [[2.0, -1.0]]),
            "mask shape": ({**valid_kwargs, "attention_mask": [[1, 1, 1]]}, [[2.0, -1.0]]),
            "token type shape": ({**valid_kwargs, "token_type_ids": [[0, 0, 0]]}, [[2.0, -1.0]]),
            "label rank": ({**valid_kwargs, "labels": [[0]]}, [[2.0, -1.0]]),
            "logit batch": (valid_kwargs, [[2.0, -1.0], [1.0, 0.0]]),
        }

        for name, (kwargs, logits) in cases.items():
            with self.subTest(name=name):
                recorder = ForwardPredictionRecorder(
                    catalog=_single_example_catalog(),
                    probe_batch_count=1,
                )
                recorder.begin_group(run_seed=1, group=name)
                with self.assertRaises((KeyError, ValueError)):
                    recorder.hook(
                        None,
                        (),
                        kwargs,
                        SimpleNamespace(logits=logits),
                    )

    def test_hook_rejects_malformed_or_nonfinite_logits(self):
        from scripts.fusion_count_prediction_capture import ForwardPredictionRecorder

        outputs = {
            "one class": SimpleNamespace(logits=[[1.0]]),
            "three classes": SimpleNamespace(logits=[[1.0, 2.0, 3.0]]),
            "nan": SimpleNamespace(logits=[[float("nan"), 1.0]]),
            "infinity": SimpleNamespace(logits=[[float("inf"), 1.0]]),
            "missing": SimpleNamespace(),
        }

        for name, output in outputs.items():
            with self.subTest(name=name):
                recorder = ForwardPredictionRecorder(
                    catalog=_single_example_catalog(),
                    probe_batch_count=1,
                )
                recorder.begin_group(run_seed=1, group=name)
                with self.assertRaises((TypeError, ValueError)):
                    _capture_single_example(recorder, output=output)

    def test_module_import_does_not_load_torch_or_project_runtime(self):
        code = """
import builtins

real_import = builtins.__import__
blocked = ("torch", "transformers", "blb_stage2_rl")

def guarded_import(name, *args, **kwargs):
    if name.startswith(blocked):
        raise AssertionError(f"heavy dependency imported: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import scripts.fusion_count_prediction_capture as capture
assert capture.PREDICTION_ROW_SCHEMA == "fusion-count-per-example-v1"
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
            msg=completed.stdout + completed.stderr,
        )


class PredictionJsonlWriterTest(unittest.TestCase):
    def test_streams_strict_json_rows_and_tracks_row_count(self):
        from scripts.fusion_count_prediction_capture import PredictionJsonlWriter

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "predictions.jsonl"
            with PredictionJsonlWriter(path) as writer:
                writer.write_rows(iter([
                    {
                        "schema_version": "fusion-count-per-example-v1",
                        "dataset_idx": 1,
                        "logits": [0.1, 0.2],
                        "correct": True,
                    },
                    {
                        "schema_version": "fusion-count-per-example-v1",
                        "dataset_idx": 2,
                        "logits": [0.3, 0.4],
                        "correct": False,
                    },
                ]))
                self.assertEqual(writer.row_count, 2)
                writer.write_rows([{"dataset_idx": 3, "logits": [0.5, 0.6]}])
                self.assertEqual(writer.row_count, 3)

            payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([payload["dataset_idx"] for payload in payloads], [1, 2, 3])
        self.assertEqual(payloads[0]["logits"], [0.1, 0.2])

    def test_rejects_nonfinite_json_without_incrementing_row_count(self):
        from scripts.fusion_count_prediction_capture import PredictionJsonlWriter

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "predictions.jsonl"
            with PredictionJsonlWriter(path) as writer:
                writer.write_rows([{"dataset_idx": 1, "logits": [0.1, 0.2]}])
                with self.assertRaises(ValueError):
                    writer.write_rows([{"dataset_idx": 2, "logits": [float("nan"), 0.2]}])
                self.assertEqual(writer.row_count, 1)

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertNotIn("NaN", lines[0])


if __name__ == "__main__":
    unittest.main()
