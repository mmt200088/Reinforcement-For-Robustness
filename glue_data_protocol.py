"""Shared data protocol for the supported BERT/GLUE search profiles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from json_utils import stable_json_hash


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


class GlueDataProtocolError(RuntimeError):
    """Raised when a dataset cannot satisfy the formal probe contract."""


@dataclass(frozen=True)
class TrainProbeIdentity:
    dataset: str
    source_size: int
    positions: tuple[int, ...]
    raw_ids: tuple[int, ...]
    label_histogram: tuple[tuple[int, int], ...]
    ordered_identity_hash: str


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


def validate_dataset(dataset: str) -> str:
    normalized = str(dataset or "").strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise GlueDataProtocolError(f"unsupported dataset: {normalized or '<empty>'}")
    return normalized


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
    identity_payload = {
        "schema_version": PROTOCOL_SCHEMA,
        "dataset": dataset_name,
        "source_split": TRAIN_PROBE_SOURCE_SPLIT,
        "source_size": source_size,
        "probe_size": TRAIN_PROBE_SIZE,
        "probe_seed": TRAIN_PROBE_SEED,
        "positions": selected_positions,
        "raw_ids": probe_ids,
        "labels": probe_labels,
    }
    identity = TrainProbeIdentity(
        dataset=dataset_name,
        source_size=source_size,
        positions=selected_positions,
        raw_ids=probe_ids,
        label_histogram=tuple(sorted(Counter(probe_labels).items())),
        ordered_identity_hash=stable_json_hash(identity_payload),
    )
    return probe, identity
