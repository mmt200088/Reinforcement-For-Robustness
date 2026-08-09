from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from mrpc_reproducibility import (
    MRPC_DATASET_REVISION,
    MRPCReproducibilityError,
)


class _FakeSplit(list):
    def shuffle(self, *, seed):
        self.shuffle_seed = int(seed)
        return _FakeSplit(reversed(self))


def _builder():
    spec = importlib.util.find_spec("scripts.build_mrpc_reproducibility_fixture")
    if spec is None:
        raise AssertionError("scripts.build_mrpc_reproducibility_fixture is missing")
    return importlib.import_module("scripts.build_mrpc_reproducibility_fixture")


class BuildMRPCReproducibilityFixtureTest(unittest.TestCase):
    def test_reproduces_historical_full_order_and_sorted_probe_positions(self):
        build_mrpc_reproducibility_fixture_from_dataset = _builder().build_mrpc_reproducibility_fixture_from_dataset

        validation = _FakeSplit(
            [
                {"idx": 558, "label": 1, "sentence1": "a", "sentence2": "A"},
                {"idx": 18, "label": 0, "sentence1": "b", "sentence2": "B"},
                {"idx": 4053, "label": 1, "sentence1": "c", "sentence2": "C"},
                {"idx": 1039, "label": 0, "sentence1": "d", "sentence2": "D"},
            ]
        )
        split_calls = []

        def split_fn(indices, **kwargs):
            split_calls.append((list(indices), dict(kwargs)))
            return [2, 0], [1, 3]

        fixture = build_mrpc_reproducibility_fixture_from_dataset(
            validation,
            dataset_revision=MRPC_DATASET_REVISION,
            full_shuffle_seed=42,
            probe_seed=42,
            probe_size=2,
            expected_row_count=4,
            expected_label_histogram={0: 2, 1: 2},
            split_fn=split_fn,
        )

        self.assertEqual(
            fixture.full_validation_ids,
            (1039, 4053, 18, 558),
        )
        self.assertEqual(fixture.probe_ids, (1039, 18))
        self.assertEqual(
            [row.as_dict() for row in fixture.canonical_rows],
            [
                {"idx": 18, "label": 0, "sentence1": "b", "sentence2": "B"},
                {"idx": 558, "label": 1, "sentence1": "a", "sentence2": "A"},
                {"idx": 1039, "label": 0, "sentence1": "d", "sentence2": "D"},
                {"idx": 4053, "label": 1, "sentence1": "c", "sentence2": "C"},
            ],
        )
        self.assertEqual(validation.shuffle_seed, 42)
        self.assertEqual(
            split_calls,
            [
                (
                    [0, 1, 2, 3],
                    {
                        "train_size": 2,
                        "shuffle": True,
                        "random_state": 42,
                        "stratify": [0, 1, 0, 1],
                    },
                )
            ],
        )

    def test_shape_mismatch_is_rejected_without_hash_provenance(self):
        build_mrpc_reproducibility_fixture_from_dataset = _builder().build_mrpc_reproducibility_fixture_from_dataset

        validation = _FakeSplit(
            [
                {"idx": 1, "label": 1, "sentence1": "a", "sentence2": "A"},
            ]
        )
        with self.assertRaisesRegex(MRPCReproducibilityError, "row count mismatch"):
            build_mrpc_reproducibility_fixture_from_dataset(
                validation,
                dataset_revision=MRPC_DATASET_REVISION,
                split_fn=lambda *_args, **_kwargs: ([0], []),
            )

    def test_writer_persists_direct_raw_rows(self):
        builder = _builder()
        build_mrpc_reproducibility_fixture_from_dataset = builder.build_mrpc_reproducibility_fixture_from_dataset
        write_mrpc_reproducibility_fixture = builder.write_mrpc_reproducibility_fixture

        validation = _FakeSplit(
            [
                {"idx": 1, "label": 1, "sentence1": "a", "sentence2": "A"},
            ]
        )
        fixture = build_mrpc_reproducibility_fixture_from_dataset(
            validation,
            dataset_revision=MRPC_DATASET_REVISION,
            probe_size=1,
            expected_row_count=1,
            expected_label_histogram={0: 0, 1: 1},
            split_fn=lambda *_args, **_kwargs: ([0], []),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "mrpc_validation.json"
            write_mrpc_reproducibility_fixture(output, fixture)
            payload = output.read_text(encoding="utf-8")

        self.assertIn('"canonical_rows"', payload)
        self.assertIn('"sentence1": "a"', payload)
        self.assertNotIn('"fixture_hash"', payload)
        self.assertNotIn('"source_parquet_sha256"', payload)

    def test_builder_source_has_no_formal_hash_authority_layer(self):
        builder = _builder()
        source = Path(builder.__file__).read_text(encoding="utf-8")

        for forbidden in (
            "formal_dataset_identity",
            "hashlib",
            "sha256",
            "authority",
            "fixture_hash",
        ):
            self.assertNotIn(forbidden, source.lower())


if __name__ == "__main__":
    unittest.main()
