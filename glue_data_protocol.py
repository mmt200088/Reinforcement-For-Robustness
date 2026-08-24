"""Shared data protocol for the supported BERT/GLUE search profiles."""

from __future__ import annotations

from dataclasses import dataclass


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
