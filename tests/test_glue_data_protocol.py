from __future__ import annotations

import numpy as np
import pytest

from glue_data_protocol import (
    GlueDataProtocolError,
    SUPPORTED_DATASETS,
    SUPPORTED_MODEL_FAMILIES,
    TRAIN_PROBE_SIZE,
    build_train_probe,
    supported_profiles,
    validate_supported_profile,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = tuple(dict(row) for row in rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return dict(self.rows[key])

    def shuffle(self, seed):
        order = np.random.default_rng(int(seed)).permutation(len(self.rows))
        return self.select(order.tolist())

    def select(self, positions):
        return FakeDataset(self.rows[int(position)] for position in positions)


def fake_binary_dataset(size=600, zero_count=240):
    return FakeDataset(
        {
            "idx": index,
            "label": 0 if index < zero_count else 1,
            "sentence": f"sample-{index}",
        }
        for index in range(size)
    )


def test_supported_matrix_contains_only_six_bert_profiles():
    assert SUPPORTED_DATASETS == ("mrpc", "rte", "sst2")
    assert SUPPORTED_MODEL_FAMILIES == ("bert-base", "bert-large")
    assert supported_profiles() == (
        ("bert-base", "mrpc"),
        ("bert-base", "rte"),
        ("bert-base", "sst2"),
        ("bert-large", "mrpc"),
        ("bert-large", "rte"),
        ("bert-large", "sst2"),
    )


@pytest.mark.parametrize(
    ("model_family", "dataset"),
    (
        ("gpt-2", "mrpc"),
        ("bert-base", "stsb"),
        ("bert-large", "qnli"),
        ("bert-base", "mnli"),
    ),
)
def test_unsupported_profile_fails_closed(model_family, dataset):
    with pytest.raises(ValueError, match="unsupported profile"):
        validate_supported_profile(model_family, dataset)


def test_train_probe_is_deterministic_stratified_and_ordered():
    dataset = fake_binary_dataset()
    first, first_identity = build_train_probe(dataset, dataset="mrpc")
    second, second_identity = build_train_probe(dataset, dataset="mrpc")

    assert len(first) == TRAIN_PROBE_SIZE
    assert first.rows == second.rows
    assert first_identity == second_identity
    assert first_identity.positions == tuple(sorted(first_identity.positions))
    assert len(set(first_identity.positions)) == TRAIN_PROBE_SIZE
    assert dict(first_identity.label_histogram) == {0: 102, 1: 154}


def test_probe_identity_is_independent_of_model_family():
    dataset = fake_binary_dataset()
    _, base_identity = build_train_probe(dataset, dataset="rte")
    _, large_identity = build_train_probe(dataset, dataset="rte")
    assert base_identity.ordered_identity_hash == large_identity.ordered_identity_hash


def test_train_probe_rejects_insufficient_rows():
    with pytest.raises(GlueDataProtocolError, match="fewer than 256"):
        build_train_probe(fake_binary_dataset(size=255), dataset="sst2")


def test_train_probe_rejects_missing_labels():
    dataset = FakeDataset({"idx": index} for index in range(300))
    with pytest.raises(GlueDataProtocolError, match="missing label"):
        build_train_probe(dataset, dataset="mrpc")


def test_train_probe_rejects_single_label():
    dataset = fake_binary_dataset(size=300, zero_count=300)
    with pytest.raises(GlueDataProtocolError, match="both binary labels"):
        build_train_probe(dataset, dataset="rte")


def test_train_probe_rejects_duplicate_ids():
    rows = list(fake_binary_dataset(size=300, zero_count=120).rows)
    rows[-1]["idx"] = rows[0]["idx"]
    with pytest.raises(GlueDataProtocolError, match="duplicate idx"):
        build_train_probe(FakeDataset(rows), dataset="sst2")
