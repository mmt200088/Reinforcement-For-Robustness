from __future__ import annotations

import json
import numpy as np
from pathlib import Path
import pytest
import subprocess
import sys

from glue_data_protocol import (
    GLUE_DATASET_REVISION,
    GlueDataProtocolError,
    SUPPORTED_DATASETS,
    SUPPORTED_MODEL_FAMILIES,
    TRAIN_PROBE_SIZE,
    build_train_probe,
    load_train_probe_fixture,
    resolve_glue_protocol_views,
    supported_profiles,
    validate_supported_profile,
    write_train_probe_fixture,
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


def fake_binary_dataset(size=600, zero_count=240, offset=0):
    return FakeDataset(
        {
            "idx": offset + index,
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


def _fixture_identities():
    identities = {}
    for offset, dataset in enumerate(SUPPORTED_DATASETS):
        _, identity = build_train_probe(
            fake_binary_dataset(offset=offset * 10_000),
            dataset=dataset,
        )
        identities[dataset] = identity
    return identities


def test_train_probe_fixture_round_trips_exact_identities(tmp_path):
    identities = _fixture_identities()
    path = tmp_path / "glue_train_probe_v1.json"
    write_train_probe_fixture(path, identities)
    fixture = load_train_probe_fixture(path)

    assert fixture.dataset_revision == GLUE_DATASET_REVISION
    assert fixture.task_names == SUPPORTED_DATASETS
    for dataset in SUPPORTED_DATASETS:
        assert fixture.identity_for(dataset) == identities[dataset]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(dataset_revision="wrong"), "revision"),
        (
            lambda payload: payload["tasks"].update(extra=payload["tasks"]["mrpc"]),
            "task set",
        ),
        (
            lambda payload: payload["tasks"]["mrpc"].update(
                raw_ids=payload["tasks"]["mrpc"]["raw_ids"][:-1]
                + [payload["tasks"]["mrpc"]["raw_ids"][0]]
            ),
            "duplicate raw IDs",
        ),
    ),
)
def test_train_probe_fixture_rejects_tampering(tmp_path, mutation, message):
    path = tmp_path / "glue_train_probe_v1.json"
    write_train_probe_fixture(path, _fixture_identities())
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GlueDataProtocolError, match=message):
        load_train_probe_fixture(path)


def test_fixture_builder_loads_all_tasks_at_pinned_revision(tmp_path):
    from scripts.build_glue_train_probe_fixture import build_fixture

    calls = []

    def fake_loader(repo, task, revision):
        calls.append((repo, task, revision))
        offset = SUPPORTED_DATASETS.index(task) * 10_000
        return {"train": fake_binary_dataset(offset=offset)}

    output = tmp_path / "fixture.json"
    fixture = build_fixture(output, load_dataset_fn=fake_loader)

    assert fixture.task_names == SUPPORTED_DATASETS
    assert calls == [
        ("nyu-mll/glue", task, GLUE_DATASET_REVISION)
        for task in SUPPORTED_DATASETS
    ]
    assert load_train_probe_fixture(output) == fixture


def test_fixture_builder_runs_as_a_direct_script():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_glue_train_probe_fixture.py",
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_protocol_views_use_fixture_probe_and_preserve_validation(tmp_path):
    raw_train = fake_binary_dataset()
    validation = fake_binary_dataset(size=300, zero_count=120, offset=20_000)
    identities = _fixture_identities()
    fixture_path = tmp_path / "fixture.json"
    write_train_probe_fixture(fixture_path, identities)
    fixture = load_train_probe_fixture(fixture_path)

    views = resolve_glue_protocol_views(
        {"train": raw_train, "validation": validation},
        dataset="mrpc",
        fixture=fixture,
    )

    assert views.train_full is raw_train
    assert views.validation_full is validation
    assert tuple(views.train_probe["idx"]) == identities["mrpc"].raw_ids
    assert views.identity == identities["mrpc"]


def test_protocol_views_reject_training_identity_drift(tmp_path):
    identities = _fixture_identities()
    fixture_path = tmp_path / "fixture.json"
    write_train_probe_fixture(fixture_path, identities)
    fixture = load_train_probe_fixture(fixture_path)
    changed_rows = list(fake_binary_dataset().rows)
    changed_rows[0]["label"] = 1

    with pytest.raises(GlueDataProtocolError, match="identity mismatch"):
        resolve_glue_protocol_views(
            {
                "train": FakeDataset(changed_rows),
                "validation": fake_binary_dataset(size=300),
            },
            dataset="mrpc",
            fixture=fixture,
        )
