"""Validation-data contract for selected-configuration evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from rfr.common.json_utils import read_json_file
from rfr.preparation.data.protocol import FINAL_EVAL_SPLIT, TRAIN_PROBE_SPLIT


def _protocol_hashes(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        hashes = set()
        own_hash = value.get("dataset_protocol_hash")
        if own_hash not in (None, ""):
            hashes.add(str(own_hash))
        for nested in value.values():
            hashes.update(_protocol_hashes(nested))
        return hashes
    if isinstance(value, (list, tuple)):
        hashes = set()
        for nested in value:
            hashes.update(_protocol_hashes(nested))
        return hashes
    return set()


def require_final_evaluation_protocol(
        evaluator: Any,
        *,
        search_results: Sequence[Any],
        ) -> dict[str, Any]:
    protocol_hash = str(getattr(evaluator, "dataset_protocol_hash", "") or "")
    if not protocol_hash:
        raise RuntimeError("final evaluation dataset protocol hash is missing")
    protocol_path = Path(
        str(getattr(evaluator, "dataset_protocol_path", "") or "")
    )
    if not protocol_path.is_file():
        raise RuntimeError("final evaluation dataset protocol is missing")
    protocol_payload = read_json_file(protocol_path)
    if (
            not isinstance(protocol_payload, Mapping)
            or protocol_payload.get("dataset_protocol_hash") != protocol_hash
            or protocol_payload.get("final_eval_split") != FINAL_EVAL_SPLIT
    ):
        raise RuntimeError("final evaluation dataset protocol does not match")

    provided_results = [result for result in search_results if result is not None]
    if not provided_results:
        raise RuntimeError("final evaluation requires a search-best result")
    for result in provided_results:
        if _protocol_hashes(result) != {protocol_hash}:
            raise RuntimeError(
                "final evaluation search result uses a different data protocol"
            )

    dataset_splits = getattr(evaluator, "dataset_splits", None)
    dataloaders = getattr(evaluator, "dataloaders", None)
    if not isinstance(dataset_splits, Mapping) or not isinstance(
            dataloaders, Mapping
    ):
        raise RuntimeError("final evaluation dataset registry is unavailable")
    dataset = dataset_splits.get(FINAL_EVAL_SPLIT)
    dataloader = dataloaders.get(FINAL_EVAL_SPLIT)
    if dataset is None or dataloader is None:
        raise RuntimeError("final evaluation requires the full validation split")
    if dataset is dataset_splits.get(TRAIN_PROBE_SPLIT):
        raise RuntimeError("final evaluation cannot alias the training probe")
    if len(dataset) <= 0:
        raise RuntimeError("final evaluation validation split is empty")
    return {
        "split_name": FINAL_EVAL_SPLIT,
        "dataset": dataset,
        "dataloader": dataloader,
        "example_count": int(len(dataset)),
        "dataset_protocol_hash": protocol_hash,
    }


__all__ = ["require_final_evaluation_protocol"]
