from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

ROWS = [
    {
        "idx": 558,
        "label": 1,
        "sentence1": "Alpha sentence.",
        "sentence2": "Alpha paraphrase.",
    },
    {
        "idx": 18,
        "label": 0,
        "sentence1": "Beta sentence.",
        "sentence2": "Different beta sentence.",
    },
    {
        "idx": 4053,
        "label": 1,
        "sentence1": "Gamma sentence.",
        "sentence2": "Gamma paraphrase.",
    },
    {
        "idx": 1039,
        "label": 0,
        "sentence1": "Delta sentence.",
        "sentence2": "Different delta sentence.",
    },
]
FULL_IDS = [4053, 18, 558, 1039]
PROBE_IDS = [4053, 558]


def _module():
    spec = importlib.util.find_spec("mrpc_reproducibility")
    if spec is None:
        raise AssertionError("mrpc_reproducibility module is missing")
    return importlib.import_module("mrpc_reproducibility")


def _fixture(module):
    return module.build_mrpc_fixture(
        ROWS,
        full_validation_ids=FULL_IDS,
        probe_ids=PROBE_IDS,
        dataset_revision=module.MRPC_DATASET_REVISION,
    )


def _dataset(rows):
    return {"validation": list(rows)}


def _production_fixture_payload(module):
    rows = []
    for sample_id in range(module.MRPC_FULL_EXAMPLE_COUNT):
        label = 0 if sample_id < 129 else 1
        rows.append(
            {
                "idx": sample_id,
                "label": label,
                "sentence1": f"sentence one {sample_id}",
                "sentence2": f"sentence two {sample_id}",
            }
        )
    full_ids = list(reversed(range(module.MRPC_FULL_EXAMPLE_COUNT)))
    zero_probe = list(range(81))
    one_probe = list(range(129, 129 + 175))
    fixture = module.build_mrpc_fixture(
        rows,
        full_validation_ids=full_ids,
        probe_ids=zero_probe + one_probe,
        dataset_revision=module.MRPC_DATASET_REVISION,
    )
    return fixture.as_payload()


class MRPCReproducibilityFixtureTest(unittest.TestCase):
    def test_direct_raw_rows_make_route_order_irrelevant(self):
        module = _module()
        fixture = _fixture(module)
        alternate_rows = []
        for row in reversed(ROWS):
            converted = dict(row)
            converted["idx"] = str(converted["idx"])
            converted["label"] = str(converted["label"])
            alternate_rows.append(converted)

        views = module.resolve_mrpc_validation_views(
            _dataset(alternate_rows),
            fixture,
            expected_row_count=4,
        )

        self.assertEqual(
            [row["idx"] for row in views.full_validation],
            FULL_IDS,
        )
        self.assertEqual(
            [row["idx"] for row in views.stability_probe],
            PROBE_IDS,
        )
        self.assertEqual(fixture.label_histogram, {0: 2, 1: 2})
        self.assertEqual(fixture.probe_label_histogram, {0: 0, 1: 2})

    def test_loaded_rows_must_equal_fixture_rows_directly(self):
        module = _module()
        fixture = _fixture(module)
        mutations = {}

        changed_sentence = deepcopy(ROWS)
        changed_sentence[0]["sentence2"] += " changed"
        mutations["sentence"] = changed_sentence

        changed_label = deepcopy(ROWS)
        changed_label[0]["label"] = 0
        mutations["label"] = changed_label

        mutations["missing"] = deepcopy(ROWS[:-1])

        extra = deepcopy(ROWS)
        extra.append(
            {
                "idx": 9999,
                "label": 1,
                "sentence1": "Extra.",
                "sentence2": "Extra.",
            }
        )
        mutations["extra"] = extra

        replaced_id = deepcopy(ROWS)
        replaced_id[0]["idx"] = 9998
        mutations["id"] = replaced_id

        for name, rows in mutations.items():
            with self.subTest(name=name), self.assertRaises(module.MRPCReproducibilityError):
                module.resolve_mrpc_validation_views(
                    _dataset(rows),
                    fixture,
                    expected_row_count=4,
                )

    def test_invalid_row_scalars_and_duplicate_ids_are_rejected(self):
        module = _module()
        fixture = _fixture(module)
        mutations = {}

        bool_idx = deepcopy(ROWS)
        bool_idx[0]["idx"] = True
        mutations["bool_idx"] = bool_idx

        float_idx = deepcopy(ROWS)
        float_idx[0]["idx"] = 558.0
        mutations["float_idx"] = float_idx

        bool_label = deepcopy(ROWS)
        bool_label[0]["label"] = True
        mutations["bool_label"] = bool_label

        invalid_label = deepcopy(ROWS)
        invalid_label[0]["label"] = 2
        mutations["invalid_label"] = invalid_label

        non_string_sentence = deepcopy(ROWS)
        non_string_sentence[0]["sentence1"] = 123
        mutations["sentence"] = non_string_sentence

        duplicate = deepcopy(ROWS)
        duplicate[1]["idx"] = duplicate[0]["idx"]
        mutations["duplicate"] = duplicate

        for name, rows in mutations.items():
            with self.subTest(name=name), self.assertRaises(module.MRPCReproducibilityError):
                module.resolve_mrpc_validation_views(
                    _dataset(rows),
                    fixture,
                    expected_row_count=4,
                )

    def test_fixture_json_contains_raw_rows_and_no_hash_authority(self):
        module = _module()
        payload = _production_fixture_payload(module)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mrpc_validation.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = module.load_mrpc_fixture(path)

        self.assertEqual(
            len(loaded.canonical_rows),
            module.MRPC_FULL_EXAMPLE_COUNT,
        )
        self.assertEqual(
            len(loaded.probe_ids),
            module.MRPC_PROBE_EXAMPLE_COUNT,
        )
        self.assertEqual(loaded.label_histogram, {0: 129, 1: 279})
        self.assertEqual(loaded.probe_label_histogram, {0: 81, 1: 175})
        self.assertIn("sentence1", payload["canonical_rows"][0])
        self.assertIn("sentence2", payload["canonical_rows"][0])
        forbidden_fragments = ("hash", "sha256", "authority", "parquet")
        self.assertFalse(any(fragment in str(key).lower() for key in payload for fragment in forbidden_fragments))


class MRPCReproducibilityRuntimeTest(unittest.TestCase):
    def _runtime(self, module):
        model_config = types.SimpleNamespace(
            _name_or_path=module.MRPC_MODEL_ID,
            _commit_hash=module.MRPC_MODEL_REVISION,
            num_hidden_layers=module.MRPC_NUM_LAYERS,
        )
        model = types.SimpleNamespace(
            name_or_path=module.MRPC_MODEL_ID,
            config=model_config,
        )
        tokenizer = types.SimpleNamespace(
            name_or_path=module.MRPC_MODEL_ID,
            init_kwargs={
                "_commit_hash": module.MRPC_TOKENIZER_REVISION,
                "name_or_path": module.MRPC_MODEL_ID,
            },
        )
        collator_type = type(
            "DataCollatorWithPadding",
            (),
            {"__module__": "transformers.data.data_collator"},
        )
        collator = collator_type()
        collator.tokenizer = tokenizer
        collator.padding = "max_length"
        collator.max_length = module.MRPC_MAX_LENGTH
        collator.return_tensors = "pt"
        collator.pad_to_multiple_of = 8
        return model, tokenizer, collator

    def test_pinned_model_collator_batch_and_view_sizes_are_validated(self):
        module = _module()
        model, tokenizer, collator = self._runtime(module)

        module.validate_mrpc_evaluation_setup(
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            full_validation=[None] * module.MRPC_FULL_EXAMPLE_COUNT,
            stability_probe=[None] * module.MRPC_PROBE_EXAMPLE_COUNT,
            batch_size=module.MRPC_COMPARATOR_BATCH_SIZE,
        )

        for name, mutation in (
            ("model", lambda: setattr(model.config, "_name_or_path", "other/model")),
            ("revision", lambda: setattr(model.config, "_commit_hash", "0" * 40)),
            ("tokenizer_model", lambda: setattr(tokenizer, "name_or_path", "other/model")),
            ("tokenizer", lambda: tokenizer.init_kwargs.update({"_commit_hash": "0" * 40})),
            ("padding", lambda: setattr(collator, "padding", True)),
            ("max_length", lambda: setattr(collator, "max_length", 64)),
            ("drop_multiple", lambda: setattr(collator, "pad_to_multiple_of", None)),
        ):
            model, tokenizer, collator = self._runtime(module)
            mutation()
            with self.subTest(name=name), self.assertRaises(module.MRPCReproducibilityError):
                module.validate_mrpc_evaluation_setup(
                    model=model,
                    tokenizer=tokenizer,
                    collator=collator,
                    full_validation=[None] * module.MRPC_FULL_EXAMPLE_COUNT,
                    stability_probe=[None] * module.MRPC_PROBE_EXAMPLE_COUNT,
                    batch_size=module.MRPC_COMPARATOR_BATCH_SIZE,
                )

        model, tokenizer, collator = self._runtime(module)
        with self.assertRaisesRegex(module.MRPCReproducibilityError, "batch size"):
            module.validate_mrpc_evaluation_setup(
                model=model,
                tokenizer=tokenizer,
                collator=collator,
                full_validation=[None] * module.MRPC_FULL_EXAMPLE_COUNT,
                stability_probe=[None] * module.MRPC_PROBE_EXAMPLE_COUNT,
                batch_size=32,
            )

    def test_legacy_transformers_tokenizer_commit_metadata_is_optional(self):
        module = _module()
        model, tokenizer, collator = self._runtime(module)
        tokenizer.init_kwargs.pop("_commit_hash")

        module.validate_mrpc_evaluation_setup(
            model=model,
            tokenizer=tokenizer,
            collator=collator,
            full_validation=[None] * module.MRPC_FULL_EXAMPLE_COUNT,
            stability_probe=[None] * module.MRPC_PROBE_EXAMPLE_COUNT,
            batch_size=module.MRPC_COMPARATOR_BATCH_SIZE,
        )

    def test_context_carries_the_exact_frozen_probe_without_identity_hashes(self):
        module = _module()
        fixture = _fixture(module)
        probe = ["probe-row-1", "probe-row-2"]

        context = module.MRPCReproducibilityContext(
            fixture=fixture,
            stability_probe=probe,
        )

        self.assertIs(context.stability_probe, probe)
        self.assertIs(context.fixture, fixture)
        self.assertNotIn("identity", context.__dict__)
        self.assertNotIn("hash", context.__dict__)

    def test_revision_kwargs_are_pinned_only_when_context_is_requested(self):
        module = _module()
        fixture = _fixture(module)

        self.assertEqual(
            module.resolve_mrpc_pretrained_revision_kwargs(
                fixture=None,
                data_path="rte",
                model_id="any/model",
            ),
            ({}, {}),
        )
        self.assertEqual(
            module.resolve_mrpc_pretrained_revision_kwargs(
                fixture=fixture,
                data_path="mrpc",
                model_id="textattack/bert-base-uncased-MRPC",
            ),
            (
                {"revision": module.MRPC_MODEL_REVISION},
                {"revision": module.MRPC_TOKENIZER_REVISION},
            ),
        )
        with self.assertRaises(module.MRPCReproducibilityError):
            module.resolve_mrpc_pretrained_revision_kwargs(
                fixture=fixture,
                data_path="rte",
                model_id=module.MRPC_MODEL_ID,
            )
        with self.assertRaises(module.MRPCReproducibilityError):
            module.resolve_mrpc_pretrained_revision_kwargs(
                fixture=fixture,
                data_path="mrpc",
                model_id="other/model",
            )


if __name__ == "__main__":
    unittest.main()
