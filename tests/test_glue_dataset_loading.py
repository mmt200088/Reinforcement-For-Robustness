import ast
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from glue_data_protocol import (
    GLUE_DATASET_REVISION,
    SUPPORTED_DATASETS,
)
from mrpc_reproducibility import (
    MRPC_DATASET_REVISION,
    MRPCReproducibilityContext,
    MRPCReproducibilityError,
    build_mrpc_fixture,
)

_MRPC_ROWS = [
    {"idx": 558, "label": 1, "sentence1": "a", "sentence2": "a"},
    {"idx": 18, "label": 0, "sentence1": "b", "sentence2": "x"},
    {"idx": 4053, "label": 1, "sentence1": "c", "sentence2": "c"},
    {"idx": 1039, "label": 0, "sentence1": "d", "sentence2": "y"},
]
_MRPC_FULL_IDS = [4053, 18, 558, 1039]
_MRPC_PROBE_IDS = [4053, 558]


class _MRPCDatasetDict(dict):
    column_names = {
        "train": ["sentence1", "sentence2", "label", "idx"],
        "validation": ["sentence1", "sentence2", "label", "idx"],
        "test": ["sentence1", "sentence2", "label", "idx"],
    }

    def __init__(self, rows=None):
        super().__init__({
            "train": [],
            "validation": list(_MRPC_ROWS if rows is None else rows),
            "test": [],
        })


def _mrpc_fixture():
    return build_mrpc_fixture(
        _MRPC_ROWS,
        full_validation_ids=_MRPC_FULL_IDS,
        probe_ids=_MRPC_PROBE_IDS,
        dataset_revision=MRPC_DATASET_REVISION,
    )


_RL_TUNE = None


def _import_rl_tune():
    global _RL_TUNE
    if _RL_TUNE is not None:
        return _RL_TUNE

    torch_stub = types.ModuleType("torch")
    torch_stub.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(benchmark=False),
    )
    torch_stub.float16 = object()
    torch_stub.__version__ = "0"

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.set_seed = lambda _seed: None
    for name in (
            "AutoConfig", "AutoModelForCausalLM",
            "AutoModelForSequenceClassification", "AutoTokenizer",
            "LlamaTokenizer", "DataCollatorWithPadding", "AutoModel",
    ):
        setattr(transformers_stub, name, object)

    datasets_stub = types.ModuleType("datasets")

    class DownloadConfig:
        def __init__(self, *, local_files_only=False):
            self.local_files_only = bool(local_files_only)

    datasets_stub.DownloadConfig = DownloadConfig
    datasets_stub.load_dataset = lambda *_args, **_kwargs: None
    datasets_stub.load_from_disk = lambda *_args, **_kwargs: None

    with mock.patch.dict(sys.modules, {
            "fire": types.SimpleNamespace(),
            "torch": torch_stub,
            "transformers": transformers_stub,
            "datasets": datasets_stub,
    }, clear=False):
        import rl_tune
    _RL_TUNE = rl_tune
    return _RL_TUNE


class GlueDatasetLoadingRegressionTests(unittest.TestCase):
    def test_public_glue_registry_contains_only_supported_tasks(self):
        rl_tune = _import_rl_tune()
        self.assertEqual(
            tuple(rl_tune.GLUE_PARQUET_SPLITS),
            SUPPORTED_DATASETS,
        )
        self.assertEqual(
            tuple(rl_tune.GLUE_REQUIRED_COLUMNS),
            SUPPORTED_DATASETS,
        )
        self.assertIsNone(rl_tune._glue_parquet_data_files("stsb"))

    def test_loader_rejects_unsupported_task_before_dataset_access(self):
        rl_tune = _import_rl_tune()
        loader = mock.Mock(side_effect=AssertionError("loader must not run"))
        with self.assertRaisesRegex(ValueError, "unsupported dataset"):
            rl_tune.load_glue_dataset_equivalent(
                "stsb",
                load_dataset_fn=loader,
            )
        loader.assert_not_called()

    def test_primary_loader_uses_pinned_glue_revision(self):
        rl_tune = _import_rl_tune()
        calls = []

        def fake_load_dataset(path, task, **kwargs):
            calls.append((path, task, kwargs))
            return _MRPCDatasetDict()

        loaded = rl_tune.load_glue_dataset_equivalent(
            "mrpc",
            load_dataset_fn=fake_load_dataset,
        )

        self.assertIsInstance(loaded, _MRPCDatasetDict)
        self.assertEqual(calls, [(
            "nyu-mll/glue",
            "mrpc",
            {"revision": GLUE_DATASET_REVISION},
        )])

    def test_reproducibility_fixture_selects_ordered_full_and_probe_views(self):
        rl_tune = _import_rl_tune()
        data = _MRPCDatasetDict(list(reversed(_MRPC_ROWS)))

        with mock.patch.object(rl_tune, "MRPC_VALIDATION_ROW_COUNT", 4):
            views = rl_tune.resolve_mrpc_reproducibility_views(
                data,
                data_path="mrpc",
                fixture=_mrpc_fixture(),
            )

        self.assertEqual(
            [row["idx"] for row in views.full_validation],
            _MRPC_FULL_IDS,
        )
        self.assertEqual(
            [row["idx"] for row in views.stability_probe],
            _MRPC_PROBE_IDS,
        )

    def test_reproducibility_views_reject_missing_fixture_and_other_tasks(self):
        rl_tune = _import_rl_tune()
        with self.assertRaisesRegex(
                MRPCReproducibilityError, "requires the MRPC task",
        ):
            rl_tune.resolve_mrpc_reproducibility_views(
                _MRPCDatasetDict(),
                data_path="rte",
                fixture=_mrpc_fixture(),
            )
        with self.assertRaisesRegex(
                MRPCReproducibilityError, "requires a fixture",
        ):
            rl_tune.resolve_mrpc_reproducibility_views(
                _MRPCDatasetDict(),
                data_path="mrpc",
                fixture=None,
            )

    def test_reproducible_pretrained_loads_use_the_pinned_snapshot(self):
        rl_tune = _import_rl_tune()
        approved_revision = "d421614df8fbeb22d6826a24d6397809fdc1e3ff"

        model_kwargs, tokenizer_kwargs = (
            rl_tune.resolve_pretrained_revision_kwargs(
                fixture=_mrpc_fixture(),
                data_path="mrpc",
                model_id="textattack/bert-base-uncased-MRPC",
            )
        )

        self.assertEqual(model_kwargs, {"revision": approved_revision})
        self.assertEqual(tokenizer_kwargs, {"revision": approved_revision})
        self.assertEqual(
            rl_tune.resolve_pretrained_revision_kwargs(
                fixture=None,
                data_path="rte",
                model_id="any/model",
            ),
            ({}, {}),
        )
        with self.assertRaisesRegex(
                MRPCReproducibilityError, "requires model",
        ):
            rl_tune.resolve_pretrained_revision_kwargs(
                fixture=_mrpc_fixture(),
                data_path="mrpc",
                model_id="other/model",
            )

        source = inspect.getsource(rl_tune.train)
        self.assertEqual(source.count("**tokenizer_revision_kwargs"), 2)
        self.assertEqual(source.count("**model_revision_kwargs"), 3)

    def test_train_wires_raw_row_fixture_before_validation_preprocessing(self):
        rl_tune = _import_rl_tune()
        source = inspect.getsource(rl_tune.train)

        self.assertIn("mrpc_reproducibility_fixture_path: str = \"\"", source)
        self.assertIn(
            "mrpc_reproducibility_fixture=mrpc_fixture",
            source,
        )
        self.assertIn("mrpc_views.full_validation", source)
        self.assertIn("mrpc_views.stability_probe", source)
        self.assertIn("MRPCReproducibilityContext(", source)
        self.assertIn(
            "mrpc_reproducibility=mrpc_reproducibility",
            source,
        )
        for forbidden in (
                "formal_dataset_protocol",
                "formal_run_identity",
                "hash_formal_mrpc_tokenized_view",
                "build_formal_mrpc_run_identity",
        ):
            self.assertNotIn(forbidden, source)
        self.assertLess(
            source.index("load_mrpc_fixture("),
            source.index("AutoTokenizer.from_pretrained("),
        )
        self.assertLess(
            source.index("load_mrpc_fixture("),
            source.index("load_glue_dataset_equivalent("),
        )

    def test_train_rejects_missing_fixture_before_runtime_access(self):
        rl_tune = _import_rl_tune()

        def forbidden_runtime_access(*_args, **_kwargs):
            raise AssertionError(
                "runtime access occurred before MRPC fixture validation"
            )

        evaluator_stub = types.SimpleNamespace(
            set_ppo_update_interval=forbidden_runtime_access,
        )
        with tempfile.TemporaryDirectory() as td, mock.patch.object(
                rl_tune,
                "AutoTokenizer",
                types.SimpleNamespace(from_pretrained=forbidden_runtime_access),
        ), mock.patch.dict(
                sys.modules,
                {"layer_importance_evaluator": evaluator_stub},
                clear=False,
        ), mock.patch("builtins.print"):
            with self.assertRaisesRegex(
                    MRPCReproducibilityError,
                    "cannot load MRPC reproducibility fixture",
            ):
                rl_tune.train(
                    base_model="textattack/bert-base-uncased-MRPC",
                    data_path="mrpc",
                    output_dir=td,
                    mrpc_reproducibility_fixture_path=str(
                        Path(td) / "missing-mrpc-fixture.json"
                    ),
                )

    def test_evaluator_frozen_probe_bypasses_runtime_resampling(self):
        source = Path("layer_importance_evaluator.py").read_text(encoding="utf-8")
        method_start = source.index("    def _get_stability_probe(")
        method_end = source.index("    def _prepare_rl_datasets(", method_start)
        method = source[method_start:method_end]

        self.assertIn("self.mrpc_reproducibility.stability_probe", method)
        self.assertIn("MRPC reproducibility probe size mismatch", method)
        self.assertLess(
            method.index("self.mrpc_reproducibility.stability_probe"),
            method.index("self._sample_dataset_by_size("),
        )

    def test_evaluator_returns_exact_frozen_probe_without_sampling(self):
        tree = ast.parse(
            Path("layer_importance_evaluator.py").read_text(encoding="utf-8")
        )
        evaluator_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LayerImportanceEvaluator"
        )
        method_node = next(
            node for node in evaluator_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_stability_probe"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[method_node], type_ignores=[]),
                "<mrpc-stability-probe>",
                "exec",
            ),
            namespace,
        )
        method = namespace["_get_stability_probe"]
        frozen_probe = ["fixture-row-1", "fixture-row-2"]
        evaluator = types.SimpleNamespace(
            mrpc_reproducibility=MRPCReproducibilityContext(
                fixture=_mrpc_fixture(),
                stability_probe=frozen_probe,
            ),
            dataset_splits={"validation_full": ["wrong-runtime-data"]},
            dataset_splits_mm={},
            _sample_dataset_by_size=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("runtime sampling must not execute")
            ),
        )

        actual = method(
            evaluator, "validation_full", len(frozen_probe), probe_seed=999
        )

        self.assertIs(actual[0], frozen_probe)
        self.assertIsNone(actual[1])

    def test_evaluator_rejects_frozen_probe_size_drift_before_sampling(self):
        tree = ast.parse(
            Path("layer_importance_evaluator.py").read_text(encoding="utf-8")
        )
        evaluator_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LayerImportanceEvaluator"
        )
        method_node = next(
            node for node in evaluator_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_stability_probe"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[method_node], type_ignores=[]),
                "<mrpc-stability-probe>",
                "exec",
            ),
            namespace,
        )
        method = namespace["_get_stability_probe"]
        evaluator = types.SimpleNamespace(
            mrpc_reproducibility=MRPCReproducibilityContext(
                fixture=_mrpc_fixture(),
                stability_probe=["fixture-row-1", "fixture-row-2"],
            ),
            dataset_splits={"validation_full": ["wrong-runtime-data"]},
            dataset_splits_mm={},
            _sample_dataset_by_size=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("runtime sampling must not execute")
            ),
        )

        with self.assertRaisesRegex(
                ValueError, "MRPC reproducibility probe size mismatch"):
            method(evaluator, "validation_full", 1, probe_seed=42)

    def test_stage2_fixture_probe_failure_cannot_fall_back_to_other_data(self):
        source = Path("blb_stage2_rl/runner.py").read_text(encoding="utf-8")
        method_start = source.index("    def _build_probe_batches(")
        method_end = source.index(
            "    def _build_validation_full_batches(", method_start
        )
        method = source[method_start:method_end]

        self.assertIn(
            'getattr(ev, "mrpc_reproducibility", None)', method
        )
        self.assertIn("drop_last=False", method)
        self.assertLess(
            method.index('getattr(ev, "mrpc_reproducibility", None)'),
            method.index(
                'ds = ev.dataset_splits.get(split_name) '
                'or ev.dataset_splits.get("train")'
            ),
        )

    def test_raw_row_fixture_validation_runs_on_primary_remote_success(self):
        rl_tune = _import_rl_tune()
        data = _MRPCDatasetDict()

        with mock.patch.object(rl_tune, "MRPC_VALIDATION_ROW_COUNT", 4):
            loaded = rl_tune.load_glue_dataset_equivalent(
                "mrpc",
                load_dataset_fn=lambda *_args, **_kwargs: data,
                mrpc_reproducibility_fixture=_mrpc_fixture(),
            )

        self.assertIs(loaded, data)

    def test_raw_row_mismatch_on_primary_route_does_not_fallback(self):
        rl_tune = _import_rl_tune()
        changed_rows = [dict(row) for row in _MRPC_ROWS]
        changed_rows[0]["sentence2"] = "changed"
        calls = []

        def fake_load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) > 1:
                raise AssertionError("raw-row mismatch must not try another route")
            return _MRPCDatasetDict(changed_rows)

        with mock.patch.object(rl_tune, "MRPC_VALIDATION_ROW_COUNT", 4):
            with self.assertRaises(MRPCReproducibilityError):
                rl_tune.load_glue_dataset_equivalent(
                    "mrpc",
                    load_dataset_fn=fake_load_dataset,
                    mrpc_reproducibility_fixture=_mrpc_fixture(),
                )

        self.assertEqual(len(calls), 1)

    def test_raw_row_fixture_validation_runs_on_local_saved_dataset(self):
        rl_tune = _import_rl_tune()

        def fake_load_dataset(*_args, **_kwargs):
            raise RuntimeError("network unavailable")

        with tempfile.TemporaryDirectory() as td:
            local_mrpc = os.path.join(td, "mrpc")
            os.makedirs(local_mrpc)
            with mock.patch.dict(
                    os.environ, {"GLUE_LOCAL_DATASET_DIR": td}, clear=False,
            ), mock.patch.object(
                    rl_tune, "MRPC_VALIDATION_ROW_COUNT", 4,
            ):
                loaded = rl_tune.load_glue_dataset_equivalent(
                    "mrpc",
                    load_dataset_fn=fake_load_dataset,
                    load_from_disk_fn=lambda _path: _MRPCDatasetDict(),
                    mrpc_reproducibility_fixture=_mrpc_fixture(),
                )

        self.assertIsInstance(loaded, _MRPCDatasetDict)

    def test_raw_row_fixture_validation_runs_on_local_parquet(self):
        rl_tune = _import_rl_tune()
        calls = []

        def fake_load_dataset(path, *_args, **_kwargs):
            calls.append(path)
            if path == "nyu-mll/glue":
                raise RuntimeError("network unavailable")
            if path == "parquet":
                return _MRPCDatasetDict()
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as td:
            local_mrpc = os.path.join(td, "mrpc")
            os.makedirs(local_mrpc)
            for split in ("train", "validation", "test"):
                with open(os.path.join(local_mrpc, f"{split}.parquet"), "wb"):
                    pass
            with mock.patch.dict(
                    os.environ, {"GLUE_LOCAL_DATASET_DIR": td}, clear=False,
            ), mock.patch.object(
                    rl_tune, "MRPC_VALIDATION_ROW_COUNT", 4,
            ):
                loaded = rl_tune.load_glue_dataset_equivalent(
                    "mrpc",
                    load_dataset_fn=fake_load_dataset,
                    load_from_disk_fn=lambda _path: (_ for _ in ()).throw(
                        RuntimeError("not a saved dataset")
                    ),
                    mrpc_reproducibility_fixture=_mrpc_fixture(),
                )

        self.assertIsInstance(loaded, _MRPCDatasetDict)
        self.assertEqual(calls, ["nyu-mll/glue", "parquet"])

    def test_raw_row_fixture_validation_runs_on_hf_local_cache(self):
        rl_tune = _import_rl_tune()
        calls = []

        def fake_load_dataset(path, *_args, **kwargs):
            calls.append((path, kwargs))
            if len(calls) == 1:
                raise RuntimeError("network unavailable")
            if path == "nyu-mll/glue" and kwargs.get("download_config") is not None:
                return _MRPCDatasetDict()
            raise AssertionError((path, kwargs))

        with tempfile.TemporaryDirectory() as td, mock.patch.dict(
                os.environ, {"GLUE_LOCAL_DATASET_DIR": td}, clear=False,
        ), mock.patch.object(
                rl_tune, "MRPC_VALIDATION_ROW_COUNT", 4,
        ):
            loaded = rl_tune.load_glue_dataset_equivalent(
                "mrpc",
                load_dataset_fn=fake_load_dataset,
                mrpc_reproducibility_fixture=_mrpc_fixture(),
            )

        self.assertIsInstance(loaded, _MRPCDatasetDict)
        self.assertEqual(len(calls), 2)

    def test_raw_row_fixture_validation_runs_on_remote_parquet(self):
        rl_tune = _import_rl_tune()
        calls = []

        def fake_load_dataset(path, *_args, **kwargs):
            calls.append((path, kwargs))
            if path == "nyu-mll/glue":
                raise RuntimeError("network unavailable")
            if path == "parquet":
                return _MRPCDatasetDict()
            raise AssertionError(path)

        with tempfile.TemporaryDirectory() as td, mock.patch.dict(
                os.environ, {"GLUE_LOCAL_DATASET_DIR": td}, clear=False,
        ), mock.patch.object(
                rl_tune, "MRPC_VALIDATION_ROW_COUNT", 4,
        ):
            loaded = rl_tune.load_glue_dataset_equivalent(
                "mrpc",
                load_dataset_fn=fake_load_dataset,
                mrpc_reproducibility_fixture=_mrpc_fixture(),
            )

        self.assertIsInstance(loaded, _MRPCDatasetDict)
        self.assertEqual([path for path, _kwargs in calls], [
            "nyu-mll/glue", "nyu-mll/glue", "parquet",
        ])

    def test_uses_local_glue_dataset_dir_when_glue_network_routes_fail(self):
        rl_tune = _import_rl_tune()
        calls = []
        disk_calls = []

        class FakeDatasetDict(dict):
            column_names = {
                "train": ["sentence1", "sentence2", "label", "idx"],
                "validation": ["sentence1", "sentence2", "label", "idx"],
                "test": ["sentence1", "sentence2", "label", "idx"],
            }

        def fake_load_dataset(path, *args, **kwargs):
            calls.append((path, args, kwargs))
            raise RuntimeError("network unavailable")

        def fake_load_from_disk(path):
            disk_calls.append(path)
            return FakeDatasetDict({"train": "ok"})

        with tempfile.TemporaryDirectory() as td:
            local_mrpc = os.path.join(td, "mrpc")
            os.makedirs(local_mrpc)
            with mock.patch.dict(os.environ, {"GLUE_LOCAL_DATASET_DIR": td}, clear=False):
                data = rl_tune.load_glue_dataset_equivalent(
                    "mrpc",
                    load_dataset_fn=fake_load_dataset,
                    load_from_disk_fn=fake_load_from_disk,
                    route_log_dir=td,
                )
            log_path = os.path.join(td, "glue_dataset_local_route.txt")
            with open(log_path, "r", encoding="utf-8") as f:
                log_text = f.read()

        self.assertEqual(data, {"train": "ok"})
        self.assertEqual(calls, [(
            "nyu-mll/glue",
            ("mrpc",),
            {"revision": GLUE_DATASET_REVISION},
        )])
        self.assertEqual(disk_calls, [local_mrpc])
        self.assertIn("route=local_saved_to_disk", log_text)
        self.assertIn(f"path={local_mrpc}", log_text)

    def test_unpinned_mrpc_local_dataset_does_not_require_idx(self):
        rl_tune = _import_rl_tune()

        class FakeDatasetDict(dict):
            column_names = {
                "train": ["sentence1", "sentence2", "label"],
                "validation": ["sentence1", "sentence2", "label"],
                "test": ["sentence1", "sentence2", "label"],
            }

        with tempfile.TemporaryDirectory() as td:
            local_mrpc = os.path.join(td, "mrpc")
            os.makedirs(local_mrpc)
            with mock.patch.dict(
                    os.environ, {"GLUE_LOCAL_DATASET_DIR": td}, clear=False,
            ):
                loaded = rl_tune.load_glue_dataset_equivalent(
                    "mrpc",
                    load_dataset_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("network unavailable")
                    ),
                    load_from_disk_fn=lambda _path: FakeDatasetDict({"train": "ok"}),
                )

        self.assertEqual(loaded, {"train": "ok"})

    def test_uses_hf_local_files_cache_before_remote_parquet_fallback(self):
        rl_tune = _import_rl_tune()
        calls = []

        class FakeDatasetDict(dict):
            column_names = {
                "train": ["sentence1", "sentence2", "label", "idx"],
                "validation": ["sentence1", "sentence2", "label", "idx"],
                "test": ["sentence1", "sentence2", "label", "idx"],
            }

        def fake_load_dataset(path, *args, **kwargs):
            calls.append((path, args, kwargs))
            if len(calls) == 1:
                raise RuntimeError("metadata listing failed")
            if path == "nyu-mll/glue" and kwargs.get("download_config") is not None:
                return FakeDatasetDict({"train": "ok"})
            raise AssertionError(f"unexpected dataset call: {(path, args, kwargs)!r}")

        with tempfile.TemporaryDirectory() as td:
            data = rl_tune.load_glue_dataset_equivalent(
                "mrpc",
                load_dataset_fn=fake_load_dataset,
                route_log_dir=td,
            )
            log_path = os.path.join(td, "glue_dataset_local_route.txt")
            with open(log_path, "r", encoding="utf-8") as f:
                log_text = f.read()

        self.assertEqual(data, {"train": "ok"})
        self.assertEqual(calls[0], (
            "nyu-mll/glue",
            ("mrpc",),
            {"revision": GLUE_DATASET_REVISION},
        ))
        self.assertEqual(calls[1][0], "nyu-mll/glue")
        self.assertEqual(calls[1][1], ("mrpc",))
        self.assertTrue(getattr(calls[1][2]["download_config"], "local_files_only"))
        self.assertEqual(len(calls), 2)
        self.assertIn("route=hf_cache_local_files_only", log_text)

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
        self.assertEqual(calls[1][0], "nyu-mll/glue")
        self.assertTrue(getattr(calls[1][2]["download_config"], "local_files_only"))
        data_files = calls[2][2]["data_files"]
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
        self.assertEqual(calls[0], (
            "nyu-mll/glue",
            ("mrpc",),
            {"revision": GLUE_DATASET_REVISION},
        ))
        self.assertEqual(calls[1][0], "nyu-mll/glue")
        self.assertTrue(getattr(calls[1][2]["download_config"], "local_files_only"))
        self.assertEqual(calls[2][0], "parquet")
        data_files = calls[2][2]["data_files"]
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
