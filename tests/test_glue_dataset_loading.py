import os
import sys
import tempfile
import types
import unittest
from unittest import mock


_RL_TUNE = None


def _import_rl_tune():
    global _RL_TUNE
    if _RL_TUNE is not None:
        return _RL_TUNE
    with mock.patch.dict(sys.modules, {"fire": types.SimpleNamespace()}, clear=False):
        import rl_tune
    _RL_TUNE = rl_tune
    return _RL_TUNE


class GlueDatasetLoadingRegressionTests(unittest.TestCase):
    def test_equivalent_route_can_use_configured_alternate_endpoint(self):
        rl_tune = _import_rl_tune()
        calls = []
        revision = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"

        class FakeDatasetDict(dict):
            column_names = {
                "train": ["sentence1", "sentence2", "label", "idx"],
                "validation": ["sentence1", "sentence2", "label", "idx"],
                "test": ["sentence1", "sentence2", "label", "idx"],
            }

        def fake_load_dataset(path, *args, **kwargs):
            calls.append((path, args, kwargs))
            if path == "nyu-mll/glue":
                raise RuntimeError(
                    "500 Server Error for url: "
                    f"https://hf-mirror.com/api/datasets/nyu-mll/glue/tree/{revision}/qnli"
                )
            if path == "parquet":
                return FakeDatasetDict({"train": "ok"})
            raise AssertionError(f"unexpected dataset path: {path}")

        with mock.patch.object(
            rl_tune,
            "GLUE_EQUIVALENT_PARQUET_ENDPOINTS",
            ["https://huggingface.co"],
        ):
            with tempfile.TemporaryDirectory() as td:
                data = rl_tune.load_glue_dataset_equivalent(
                    "mrpc",
                    load_dataset_fn=fake_load_dataset,
                    route_log_dir=td,
                )
                log_path = os.path.join(td, "glue_dataset_equivalent_route.txt")
                with open(log_path, "r", encoding="utf-8") as f:
                    log_text = f.read()

        self.assertEqual(data, {"train": "ok"})
        data_files = calls[1][2]["data_files"]
        self.assertTrue(data_files["train"].startswith("https://huggingface.co/datasets/"))
        self.assertIn("candidate_endpoints=https://huggingface.co,https://hf-mirror.com", log_text)
        self.assertIn("endpoint=https://huggingface.co", log_text)
        self.assertIn('original_operation=load_dataset("nyu-mll/glue", "mrpc")', log_text)
        self.assertIn('switched_operation=load_dataset("parquet", data_files=<same GLUE task parquet files>)', log_text)
        self.assertIn("switch_from_endpoint=https://hf-mirror.com", log_text)
        self.assertIn("switch_to_endpoint=https://huggingface.co", log_text)
        self.assertIn("route_change_summary=metadata route -> direct parquet file route", log_text)
        self.assertIn("semantic_equivalence=same_repo=nyu-mll/glue; same_task=mrpc", log_text)

    def test_uses_same_revision_task_parquet_when_glue_metadata_listing_fails(self):
        load_glue_dataset_equivalent = _import_rl_tune().load_glue_dataset_equivalent

        calls = []
        revision = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"

        class FakeDatasetDict(dict):
            column_names = {
                "train": ["sentence1", "sentence2", "label", "idx"],
                "validation": ["sentence1", "sentence2", "label", "idx"],
                "test": ["sentence1", "sentence2", "label", "idx"],
            }

        def fake_load_dataset(path, *args, **kwargs):
            calls.append((path, args, kwargs))
            if path == "nyu-mll/glue":
                raise RuntimeError(
                    "500 Server Error for url: "
                    f"https://hf-mirror.com/api/datasets/nyu-mll/glue/tree/{revision}/qnli"
                )
            if path == "parquet":
                return FakeDatasetDict({"train": "ok"})
            raise AssertionError(f"unexpected dataset path: {path}")

        rl_tune = _import_rl_tune()
        with mock.patch.object(rl_tune, "GLUE_EQUIVALENT_PARQUET_ENDPOINTS", []):
            with mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com/"}, clear=False):
                with tempfile.TemporaryDirectory() as td:
                    data = load_glue_dataset_equivalent(
                        "mrpc",
                        load_dataset_fn=fake_load_dataset,
                        route_log_dir=td,
                    )
                    log_path = os.path.join(td, "glue_dataset_equivalent_route.txt")
                    self.assertTrue(os.path.exists(log_path))
                    with open(log_path, "r", encoding="utf-8") as f:
                        log_text = f.read()

        self.assertEqual(data, {"train": "ok"})
        self.assertEqual(calls[0], ("nyu-mll/glue", ("mrpc",), {}))
        self.assertEqual(calls[1][0], "parquet")
        data_files = calls[1][2]["data_files"]
        self.assertEqual(set(data_files), {"train", "validation", "test"})
        self.assertTrue(data_files["train"].startswith("https://hf-mirror.com/datasets/"))
        self.assertIn(f"/resolve/{revision}/mrpc/train-00000-of-00001.parquet", data_files["train"])
        self.assertIn("task=mrpc", log_text)
        self.assertIn(f"revision={revision}", log_text)
        self.assertIn("primary_error=RuntimeError", log_text)
        self.assertIn("switch_from_endpoint=https://hf-mirror.com", log_text)
        self.assertIn("switch_to_endpoint=https://hf-mirror.com", log_text)

    def test_equivalent_route_can_be_disabled_by_code_variable(self):
        rl_tune = _import_rl_tune()
        calls = []

        def fake_load_dataset(path, *args, **kwargs):
            calls.append(path)
            raise RuntimeError("primary loader failure")

        with mock.patch.object(rl_tune, "ENABLE_GLUE_EQUIVALENT_PARQUET_ROUTE", False):
            with self.assertRaisesRegex(RuntimeError, "primary loader failure"):
                rl_tune.load_glue_dataset_equivalent("mrpc", load_dataset_fn=fake_load_dataset)

        self.assertEqual(calls, ["nyu-mll/glue"])

    def test_equivalent_parquet_route_rejects_wrong_schema(self):
        load_glue_dataset_equivalent = _import_rl_tune().load_glue_dataset_equivalent

        class WrongSchemaDatasetDict(dict):
            column_names = {
                "train": ["sentence1", "sentence2"],
                "validation": ["sentence1", "sentence2"],
                "test": ["sentence1", "sentence2"],
            }

        def fake_load_dataset(path, *args, **kwargs):
            if path == "nyu-mll/glue":
                raise RuntimeError("metadata listing failed")
            return WrongSchemaDatasetDict()

        with self.assertRaisesRegex(RuntimeError, "equivalent parquet"):
            load_glue_dataset_equivalent("mrpc", load_dataset_fn=fake_load_dataset)

    def test_unknown_glue_task_preserves_original_loader_error(self):
        load_glue_dataset_equivalent = _import_rl_tune().load_glue_dataset_equivalent

        def fake_load_dataset(path, *args, **kwargs):
            raise RuntimeError("primary loader failure")

        with self.assertRaisesRegex(RuntimeError, "primary loader failure"):
            load_glue_dataset_equivalent("unknown_task", load_dataset_fn=fake_load_dataset)
