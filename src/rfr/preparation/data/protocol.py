"""Shared data protocol for the supported BERT/GLUE search profiles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import train_test_split

from rfr.common.json_utils import read_json_file, stable_json_hash, write_json_file


PROTOCOL_SCHEMA = "glue_train_probe_protocol_v1"
GLUE_DATASET_REPO = "nyu-mll/glue"
GLUE_DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
TRAIN_PROBE_SPLIT = "train_probe"
TRAIN_PROBE_SOURCE_SPLIT = "train"
FINAL_EVAL_SPLIT = "validation_full"
TRAIN_PROBE_SIZE = 256
TRAIN_PROBE_SEED = 42

SUPPORTED_DATASETS = ("mrpc", "rte", "sst2")
SUPPORTED_MODEL_FAMILIES = ("bert-base", "bert-large")


@dataclass(frozen=True)
class TaskSpec:
    input_columns: tuple[str, ...]
    metric_names: tuple[str, str] = ("accuracy", "weighted_f1")


TASK_SPECS = {
    "mrpc": TaskSpec(("sentence1", "sentence2")),
    "rte": TaskSpec(("sentence1", "sentence2")),
    "sst2": TaskSpec(("sentence",)),
}


class GlueDataProtocolError(ValueError):
    """Raised when a dataset cannot satisfy the formal probe contract."""


@dataclass(frozen=True)
class TrainProbeIdentity:
    dataset: str
    source_size: int
    positions: tuple[int, ...]
    raw_ids: tuple[int, ...]
    labels: tuple[int, ...]
    label_histogram: tuple[tuple[int, int], ...]
    ordered_identity_hash: str


@dataclass(frozen=True)
class GlueTrainProbeFixture:
    dataset_revision: str
    identities: tuple[TrainProbeIdentity, ...]

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(identity.dataset for identity in self.identities)

    def identity_for(self, dataset: str) -> TrainProbeIdentity:
        dataset_name = validate_dataset(dataset)
        matches = [
            identity
            for identity in self.identities
            if identity.dataset == dataset_name
        ]
        if len(matches) != 1:
            raise GlueDataProtocolError(
                f"fixture has no unique identity for {dataset_name}"
            )
        return matches[0]


@dataclass(frozen=True)
class GlueProtocolViews:
    train_full: Any
    train_probe: Any
    validation_full: Any
    identity: TrainProbeIdentity


@dataclass(frozen=True)
class GlueDataProtocolContext:
    model_family: str
    dataset: str
    train_probe: Any
    validation_full: Any
    identity: TrainProbeIdentity

    def __post_init__(self) -> None:
        validate_supported_profile(self.model_family, self.dataset)
        if self.identity.dataset != self.dataset:
            raise GlueDataProtocolError(
                "protocol context dataset does not match probe identity"
            )
        if len(self.train_probe) != TRAIN_PROBE_SIZE:
            raise GlueDataProtocolError(
                "protocol context train probe must contain 256 rows"
            )
        if self.validation_full is None or len(self.validation_full) == 0:
            raise GlueDataProtocolError(
                "protocol context requires the full validation split"
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_SCHEMA,
            "model_family": self.model_family,
            "dataset": self.dataset,
            "dataset_repo": GLUE_DATASET_REPO,
            "dataset_revision": GLUE_DATASET_REVISION,
            "source_split": TRAIN_PROBE_SOURCE_SPLIT,
            "search_split": TRAIN_PROBE_SPLIT,
            "final_eval_split": FINAL_EVAL_SPLIT,
            "source_size": int(self.identity.source_size),
            "probe_size": TRAIN_PROBE_SIZE,
            "probe_seed": TRAIN_PROBE_SEED,
            "positions": list(self.identity.positions),
            "raw_ids": list(self.identity.raw_ids),
            "label_histogram": {
                str(label): int(count)
                for label, count in self.identity.label_histogram
            },
            "ordered_identity_hash": self.identity.ordered_identity_hash,
            "validation_size": len(self.validation_full),
        }

    @property
    def dataset_protocol_hash(self) -> str:
        return stable_json_hash(self._identity_payload())

    def as_payload(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["dataset_protocol_hash"] = self.dataset_protocol_hash
        return payload


def supported_profiles() -> tuple[tuple[str, str], ...]:
    return tuple(
        (model_family, dataset)
        for model_family in SUPPORTED_MODEL_FAMILIES
        for dataset in SUPPORTED_DATASETS
    )


def validate_supported_profile(model_family: str, dataset: str) -> None:
    profile = (
        str(model_family or "").strip().lower(),
        str(dataset or "").strip().lower(),
    )
    if profile not in supported_profiles():
        raise ValueError(
            f"unsupported profile: {profile[0] or '<empty>'}/"
            f"{profile[1] or '<empty>'}"
        )


def resolve_model_family(model_id: str) -> str:
    normalized = str(model_id or "").strip().lower()
    if "bert-large" in normalized:
        return "bert-large"
    if "bert-base" in normalized:
        return "bert-base"
    raise ValueError(f"unsupported model: {normalized or '<empty>'}")


def validate_dataset_protocol_binding(
    payload: Mapping[str, Any],
    *,
    expected_hash: str,
    artifact: str,
) -> None:
    actual_schema = str(payload.get("dataset_protocol_schema") or "")
    actual_hash = str(payload.get("dataset_protocol_hash") or "")
    expected = str(expected_hash or "")
    if actual_schema != PROTOCOL_SCHEMA or not expected or actual_hash != expected:
        raise RuntimeError(
            f"{artifact} train-probe protocol mismatch; start a fresh run "
            f"with {PROTOCOL_SCHEMA}"
        )


def validate_dataset(dataset: str) -> str:
    normalized = str(dataset or "").strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise GlueDataProtocolError(f"unsupported dataset: {normalized or '<empty>'}")
    return normalized


def dataset_from_profile(profile: str) -> str:
    normalized = str(profile or "").strip().lower().replace("-", "_")
    if normalized.endswith("_large"):
        normalized = normalized[:-len("_large")]
    return validate_dataset(normalized)


def _required_column(dataset: Any, name: str) -> list[Any]:
    try:
        values = list(dataset[name])
    except Exception as exc:
        raise GlueDataProtocolError(f"training split is missing {name}") from exc
    if len(values) != len(dataset):
        raise GlueDataProtocolError(
            f"training split {name} length does not match the dataset"
        )
    return values


def _identity_hash_payload(
    *,
    dataset: str,
    source_size: int,
    positions: tuple[int, ...],
    raw_ids: tuple[int, ...],
    labels: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "dataset": dataset,
        "source_split": TRAIN_PROBE_SOURCE_SPLIT,
        "source_size": source_size,
        "probe_size": TRAIN_PROBE_SIZE,
        "probe_seed": TRAIN_PROBE_SEED,
        "positions": positions,
        "raw_ids": raw_ids,
        "labels": labels,
    }


def build_train_probe(raw_train: Any, *, dataset: str):
    dataset_name = validate_dataset(dataset)
    source_size = len(raw_train)
    if source_size < TRAIN_PROBE_SIZE:
        raise GlueDataProtocolError("training split has fewer than 256 rows")

    raw_labels = _required_column(raw_train, "label")
    try:
        labels = tuple(int(value) for value in raw_labels)
    except (TypeError, ValueError) as exc:
        raise GlueDataProtocolError("training split labels must be integers") from exc
    if set(labels) != {0, 1}:
        raise GlueDataProtocolError("training split must contain both binary labels")

    raw_ids = _required_column(raw_train, "idx")
    try:
        normalized_ids = tuple(int(value) for value in raw_ids)
    except (TypeError, ValueError) as exc:
        raise GlueDataProtocolError("training split idx values must be integers") from exc
    if len(set(normalized_ids)) != source_size:
        raise GlueDataProtocolError("training split contains duplicate idx values")

    # One ordered, stratified identity is shared by every search-stage evaluator.
    shuffled = raw_train.shuffle(seed=TRAIN_PROBE_SEED)
    shuffled_labels = np.asarray(
        [int(value) for value in _required_column(shuffled, "label")],
        dtype=np.int64,
    )
    positions = np.arange(source_size, dtype=np.int64)
    try:
        selected, _ = train_test_split(
            positions,
            train_size=TRAIN_PROBE_SIZE,
            random_state=TRAIN_PROBE_SEED,
            shuffle=True,
            stratify=shuffled_labels,
        )
    except ValueError as exc:
        raise GlueDataProtocolError(
            "training split cannot produce the formal stratified probe"
        ) from exc
    selected_positions = tuple(
        sorted(int(value) for value in np.asarray(selected).reshape(-1))
    )
    if (
        len(selected_positions) != TRAIN_PROBE_SIZE
        or len(set(selected_positions)) != TRAIN_PROBE_SIZE
    ):
        raise GlueDataProtocolError(
            "formal train probe must contain 256 unique positions"
        )

    probe = shuffled.select(list(selected_positions))
    probe_ids = tuple(int(value) for value in _required_column(probe, "idx"))
    probe_labels = tuple(
        int(value) for value in _required_column(probe, "label")
    )
    identity_payload = _identity_hash_payload(
        dataset=dataset_name,
        source_size=source_size,
        positions=selected_positions,
        raw_ids=probe_ids,
        labels=probe_labels,
    )
    identity = TrainProbeIdentity(
        dataset=dataset_name,
        source_size=source_size,
        positions=selected_positions,
        raw_ids=probe_ids,
        labels=probe_labels,
        label_histogram=tuple(sorted(Counter(probe_labels).items())),
        ordered_identity_hash=stable_json_hash(identity_payload),
    )
    return probe, identity


def _identity_payload(identity: TrainProbeIdentity) -> dict[str, Any]:
    return {
        "source_size": int(identity.source_size),
        "positions": list(identity.positions),
        "raw_ids": list(identity.raw_ids),
        "labels": list(identity.labels),
        "label_histogram": {
            str(label): int(count)
            for label, count in identity.label_histogram
        },
        "ordered_identity_hash": identity.ordered_identity_hash,
    }


def write_train_probe_fixture(
    path: str | Path,
    identities: dict[str, TrainProbeIdentity],
) -> Path:
    if tuple(sorted(identities)) != tuple(sorted(SUPPORTED_DATASETS)):
        raise GlueDataProtocolError("fixture task set must match supported datasets")
    payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "dataset_repo": GLUE_DATASET_REPO,
        "dataset_revision": GLUE_DATASET_REVISION,
        "probe_size": TRAIN_PROBE_SIZE,
        "probe_seed": TRAIN_PROBE_SEED,
        "tasks": {
            dataset: _identity_payload(identities[dataset])
            for dataset in SUPPORTED_DATASETS
        },
    }
    return write_json_file(path, payload, ensure_ascii=True)


def _parse_fixture_identity(dataset: str, payload: Any) -> TrainProbeIdentity:
    if not isinstance(payload, dict):
        raise GlueDataProtocolError(f"fixture identity for {dataset} must be an object")
    expected_fields = {
        "source_size",
        "positions",
        "raw_ids",
        "labels",
        "label_histogram",
        "ordered_identity_hash",
    }
    if set(payload) != expected_fields:
        raise GlueDataProtocolError(f"fixture identity field set mismatch for {dataset}")
    try:
        source_size = int(payload["source_size"])
        positions = tuple(int(value) for value in payload["positions"])
        raw_ids = tuple(int(value) for value in payload["raw_ids"])
        labels = tuple(int(value) for value in payload["labels"])
        stored_histogram = {
            int(label): int(count)
            for label, count in payload["label_histogram"].items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise GlueDataProtocolError(
            f"fixture identity values are invalid for {dataset}"
        ) from exc
    if not (
        len(positions)
        == len(raw_ids)
        == len(labels)
        == TRAIN_PROBE_SIZE
    ):
        raise GlueDataProtocolError(
            f"fixture identity for {dataset} must contain 256 rows"
        )
    if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
        raise GlueDataProtocolError(
            f"fixture positions for {dataset} must be sorted and unique"
        )
    if len(set(raw_ids)) != len(raw_ids):
        raise GlueDataProtocolError(f"fixture contains duplicate raw IDs for {dataset}")
    if set(labels) != {0, 1}:
        raise GlueDataProtocolError(
            f"fixture labels for {dataset} must contain both binary labels"
        )
    histogram = dict(Counter(labels))
    if histogram != stored_histogram:
        raise GlueDataProtocolError(f"fixture label histogram mismatch for {dataset}")
    computed_hash = stable_json_hash(
        _identity_hash_payload(
            dataset=dataset,
            source_size=source_size,
            positions=positions,
            raw_ids=raw_ids,
            labels=labels,
        )
    )
    stored_hash = str(payload["ordered_identity_hash"] or "")
    if computed_hash != stored_hash:
        raise GlueDataProtocolError(f"fixture identity hash mismatch for {dataset}")
    return TrainProbeIdentity(
        dataset=dataset,
        source_size=source_size,
        positions=positions,
        raw_ids=raw_ids,
        labels=labels,
        label_histogram=tuple(sorted(histogram.items())),
        ordered_identity_hash=computed_hash,
    )


def load_train_probe_fixture(path: str | Path) -> GlueTrainProbeFixture:
    try:
        payload = read_json_file(path)
    except Exception as exc:
        raise GlueDataProtocolError(f"cannot load train-probe fixture at {path}") from exc
    if not isinstance(payload, dict):
        raise GlueDataProtocolError("train-probe fixture must be an object")
    expected_fields = {
        "schema_version",
        "dataset_repo",
        "dataset_revision",
        "probe_size",
        "probe_seed",
        "tasks",
    }
    if set(payload) != expected_fields:
        raise GlueDataProtocolError("train-probe fixture field set mismatch")
    if payload["schema_version"] != PROTOCOL_SCHEMA:
        raise GlueDataProtocolError("train-probe fixture schema mismatch")
    if payload["dataset_repo"] != GLUE_DATASET_REPO:
        raise GlueDataProtocolError("train-probe fixture dataset repository mismatch")
    if payload["dataset_revision"] != GLUE_DATASET_REVISION:
        raise GlueDataProtocolError("train-probe fixture revision mismatch")
    if int(payload["probe_size"]) != TRAIN_PROBE_SIZE:
        raise GlueDataProtocolError("train-probe fixture size mismatch")
    if int(payload["probe_seed"]) != TRAIN_PROBE_SEED:
        raise GlueDataProtocolError("train-probe fixture seed mismatch")
    tasks = payload["tasks"]
    if not isinstance(tasks, dict) or tuple(sorted(tasks)) != tuple(
        sorted(SUPPORTED_DATASETS)
    ):
        raise GlueDataProtocolError("train-probe fixture task set mismatch")
    identities = tuple(
        _parse_fixture_identity(dataset, tasks[dataset])
        for dataset in SUPPORTED_DATASETS
    )
    return GlueTrainProbeFixture(
        dataset_revision=GLUE_DATASET_REVISION,
        identities=identities,
    )


def resolve_glue_protocol_views(
    dataset_dict: Any,
    *,
    dataset: str,
    fixture: GlueTrainProbeFixture,
) -> GlueProtocolViews:
    dataset_name = validate_dataset(dataset)
    try:
        raw_train = dataset_dict[TRAIN_PROBE_SOURCE_SPLIT]
    except Exception as exc:
        raise GlueDataProtocolError("GLUE dataset has no training split") from exc
    try:
        validation_full = dataset_dict["validation"]
    except Exception as exc:
        raise GlueDataProtocolError("GLUE dataset has no validation split") from exc

    train_probe, actual_identity = build_train_probe(
        raw_train,
        dataset=dataset_name,
    )
    expected_identity = fixture.identity_for(dataset_name)
    if actual_identity != expected_identity:
        raise GlueDataProtocolError(
            f"training probe identity mismatch for {dataset_name}"
        )
    return GlueProtocolViews(
        train_full=raw_train,
        train_probe=train_probe,
        validation_full=validation_full,
        identity=actual_identity,
    )
