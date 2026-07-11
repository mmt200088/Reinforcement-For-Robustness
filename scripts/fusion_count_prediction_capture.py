"""Capture per-example identities and logits from fixed Stage-2 probes."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
import json
from numbers import Integral, Real
import operator
import os
from pathlib import Path
import tempfile
from typing import Any, Optional, Union

import numpy as np


PREDICTION_ROW_SCHEMA = "fusion-count-per-example-v1"

_IdentityKey = tuple[tuple[int, ...], Optional[tuple[int, ...]], int]
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


def _validated_int64_array(value: Any, *, name: str) -> np.ndarray:
    message = (
        f"{name} must contain finite, integral, non-boolean signed int64 values"
    )
    try:
        values = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc

    normalized: list[int] = []
    for item in values.flat:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(message)
        if isinstance(item, (Integral, np.integer)):
            integer = int(item)
        elif isinstance(item, (Real, np.floating)):
            if not bool(np.isfinite(item)):
                raise ValueError(message)
            integer = int(item)
            if item != integer:
                raise ValueError(message)
        else:
            raise ValueError(message)
        if integer < _INT64_MIN or integer > _INT64_MAX:
            raise ValueError(message)
        normalized.append(integer)

    return np.asarray(normalized, dtype=np.int64).reshape(values.shape)


def _array(
    value: Any,
    *,
    dtype: Any,
    name: str,
    convert_tensor_to_float: bool = False,
) -> np.ndarray:
    converted = value
    try:
        detach = getattr(converted, "detach", None)
        if callable(detach):
            converted = detach()
        if convert_tensor_to_float:
            to_float = getattr(converted, "float", None)
            if callable(to_float):
                converted = to_float()
        to_cpu = getattr(converted, "cpu", None)
        if callable(to_cpu):
            converted = to_cpu()
        to_list = getattr(converted, "tolist", None)
        if callable(to_list):
            converted = to_list()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a regular numeric array") from exc

    if np.dtype(dtype) == np.dtype(np.int64):
        return _validated_int64_array(converted, name=name)
    try:
        return np.asarray(converted, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a regular numeric array") from exc


def _integer_vector(value: Any, *, name: str) -> np.ndarray:
    result = _array(value, dtype=np.int64, name=name)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _integer_scalar(value: Any, *, name: str) -> int:
    result = _array(value, dtype=np.int64, name=name)
    if result.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return int(result.reshape(-1)[0])


def _identity_key(
    input_ids: Any,
    attention_mask: Any,
    token_type_ids: Optional[Any],
    label: Any,
) -> _IdentityKey:
    ids = _integer_vector(input_ids, name="input_ids")
    mask = _integer_vector(attention_mask, name="attention_mask")
    if ids.shape != mask.shape:
        raise ValueError("attention_mask shape must match input_ids")

    token_types: Optional[np.ndarray] = None
    if token_type_ids is not None:
        token_types = _integer_vector(token_type_ids, name="token_type_ids")
        if token_types.shape != ids.shape:
            raise ValueError("token_type_ids shape must match input_ids")

    active = mask != 0
    active_ids = tuple(int(value) for value in ids[active].tolist())
    active_token_types = None
    if token_types is not None:
        active_token_types = tuple(
            int(value) for value in token_types[active].tolist()
        )
    return active_ids, active_token_types, _integer_scalar(label, name="label")


class ExampleIdentityCatalog:
    """Immutable lookup from tokenized MRPC identities to dataset indices."""

    __slots__ = ("_dataset_indices", "_identity_to_indices")

    def __init__(
        self,
        *,
        identity_to_indices: Mapping[_IdentityKey, Sequence[int]],
        dataset_indices: Sequence[int],
    ) -> None:
        self._identity_to_indices = {
            key: tuple(int(dataset_idx) for dataset_idx in indices)
            for key, indices in identity_to_indices.items()
        }
        self._dataset_indices = tuple(int(value) for value in dataset_indices)

    @classmethod
    def from_tokenized_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
    ) -> "ExampleIdentityCatalog":
        identity_to_indices: dict[_IdentityKey, list[int]] = {}
        dataset_indices: list[int] = []
        seen_indices: set[int] = set()

        for row in rows:
            dataset_idx = _integer_scalar(row["idx"], name="dataset idx")
            if dataset_idx in seen_indices:
                raise ValueError(f"duplicate dataset idx: {dataset_idx}")

            input_ids = _integer_vector(row["input_ids"], name="input_ids")
            attention_mask = row.get("attention_mask")
            if attention_mask is None:
                attention_mask = np.ones(input_ids.shape, dtype=np.int64)
            key = _identity_key(
                input_ids,
                attention_mask,
                row.get("token_type_ids"),
                row["labels"],
            )
            identity_to_indices.setdefault(key, []).append(dataset_idx)
            dataset_indices.append(dataset_idx)
            seen_indices.add(dataset_idx)

        return cls(
            identity_to_indices=identity_to_indices,
            dataset_indices=dataset_indices,
        )

    @property
    def dataset_indices(self) -> tuple[int, ...]:
        return self._dataset_indices

    def new_trial_resolver(self) -> "TrialIdentityResolver":
        return TrialIdentityResolver(
            identity_to_indices=self._identity_to_indices,
            dataset_indices=self._dataset_indices,
        )


class TrialIdentityResolver:
    """Consume each catalog identity exactly once for one probe trial."""

    __slots__ = ("_available", "_remaining_indices")

    def __init__(
        self,
        *,
        identity_to_indices: Mapping[_IdentityKey, Sequence[int]],
        dataset_indices: Sequence[int],
    ) -> None:
        self._available = {
            key: deque(int(dataset_idx) for dataset_idx in indices)
            for key, indices in identity_to_indices.items()
        }
        self._remaining_indices = set(int(value) for value in dataset_indices)

    def resolve(
        self,
        input_ids: Any,
        attention_mask: Any,
        token_type_ids: Optional[Any],
        label: Any,
    ) -> int:
        key = _identity_key(input_ids, attention_mask, token_type_ids, label)
        available = self._available.get(key)
        if not available:
            raise ValueError("prediction identity is missing or already reused")

        dataset_idx = int(available.popleft())
        if dataset_idx not in self._remaining_indices:
            raise ValueError(f"dataset identity reused: {dataset_idx}")
        self._remaining_indices.remove(dataset_idx)
        return dataset_idx

    def assert_complete(self) -> None:
        if self._remaining_indices:
            missing = sorted(self._remaining_indices)
            raise ValueError(
                "identity trial incomplete; unresolved dataset idx values: "
                f"{missing[:10]}"
            )


def _extract_logits(output: Any) -> Any:
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    try:
        return output[1]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError("model output does not contain logits") from exc


class ForwardPredictionRecorder:
    """Read-only forward-hook recorder partitioned by deterministic trials."""

    def __init__(
        self,
        *,
        catalog: ExampleIdentityCatalog,
        probe_batch_count: int,
    ) -> None:
        try:
            batch_count = operator.index(probe_batch_count)
        except TypeError as exc:
            raise ValueError("probe_batch_count must be a positive integer") from exc
        if isinstance(probe_batch_count, bool) or batch_count <= 0:
            raise ValueError("probe_batch_count must be a positive integer")

        self._catalog = catalog
        self._probe_batch_count = int(batch_count)
        self._active = False
        self._run_seed: Optional[int] = None
        self._group: Optional[str] = None
        self._captured_batches: list[dict[str, Any]] = []

    def begin_group(self, *, run_seed: int, group: str) -> None:
        if self._active:
            raise RuntimeError("prediction recorder group is already active")
        self._active = True
        self._run_seed = int(run_seed)
        self._group = str(group)
        self._captured_batches = []

    def hook(
        self,
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        del module, args
        self._require_active()
        for name in ("input_ids", "attention_mask", "labels"):
            if name not in kwargs:
                raise KeyError(f"model forward is missing {name}")

        input_ids = _array(kwargs["input_ids"], dtype=np.int64, name="input_ids")
        attention_mask = _array(
            kwargs["attention_mask"],
            dtype=np.int64,
            name="attention_mask",
        )
        labels = _array(kwargs["labels"], dtype=np.int64, name="labels")
        token_type_ids = None
        if kwargs.get("token_type_ids") is not None:
            token_type_ids = _array(
                kwargs["token_type_ids"],
                dtype=np.int64,
                name="token_type_ids",
            )
        logits = _array(
            _extract_logits(output),
            dtype=np.float32,
            name="logits",
            convert_tensor_to_float=True,
        )

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids")
        if token_type_ids is not None and token_type_ids.shape != input_ids.shape:
            raise ValueError("token_type_ids shape must match input_ids")
        if labels.ndim != 1 or labels.shape[0] != input_ids.shape[0]:
            raise ValueError("labels must have shape [batch]")
        if logits.ndim != 2 or logits.shape != (input_ids.shape[0], 2):
            raise ValueError("logits must have shape [batch, 2]")
        if not bool(np.isfinite(logits).all()):
            raise ValueError("logits must contain only finite FP32 values")

        self._captured_batches.append({
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "token_type_ids": (
                token_type_ids.tolist() if token_type_ids is not None else None
            ),
            "labels": labels.tolist(),
            "logits": logits.tolist(),
        })

    def finish_group(self, *, trial_seeds: Sequence[int]) -> list[dict[str, Any]]:
        self._require_active()
        seeds = [int(seed) for seed in trial_seeds]
        expected_batch_count = len(seeds) * self._probe_batch_count
        actual_batch_count = len(self._captured_batches)
        if actual_batch_count != expected_batch_count:
            raise ValueError(
                "captured forward batch count mismatch: "
                f"expected {expected_batch_count}, got {actual_batch_count}"
            )

        rows: list[dict[str, Any]] = []
        for trial_index, trial_seed in enumerate(seeds):
            resolver = self._catalog.new_trial_resolver()
            probe_position = 0
            first_batch = trial_index * self._probe_batch_count
            trial_batches = self._captured_batches[
                first_batch:first_batch + self._probe_batch_count
            ]
            for batch in trial_batches:
                token_type_rows = batch["token_type_ids"]
                for batch_position, input_ids in enumerate(batch["input_ids"]):
                    attention_mask = batch["attention_mask"][batch_position]
                    token_type_ids = (
                        token_type_rows[batch_position]
                        if token_type_rows is not None
                        else None
                    )
                    gold_label = int(batch["labels"][batch_position])
                    logits = batch["logits"][batch_position]
                    dataset_idx = resolver.resolve(
                        input_ids,
                        attention_mask,
                        token_type_ids,
                        gold_label,
                    )
                    predicted_label = 1 if logits[1] > logits[0] else 0

                    row = {
                        "schema_version": PREDICTION_ROW_SCHEMA,
                        "run_seed": int(self._run_seed),
                        "group": str(self._group),
                        "trial_index": int(trial_index),
                        "trial_seed": int(trial_seed),
                        "probe_position": int(probe_position),
                        "dataset_idx": int(dataset_idx),
                        "input_ids": list(input_ids),
                        "attention_mask": list(attention_mask),
                    }
                    if token_type_ids is not None:
                        row["token_type_ids"] = list(token_type_ids)
                    row.update({
                        "gold_label": gold_label,
                        "predicted_label": predicted_label,
                        "correct": bool(predicted_label == gold_label),
                        "logits": [float(logits[0]), float(logits[1])],
                    })
                    rows.append(row)
                    probe_position += 1
            resolver.assert_complete()

        self._clear_group()
        return rows

    def abort_group(self) -> None:
        self._clear_group()

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("prediction recorder has no active group")

    def _clear_group(self) -> None:
        self._active = False
        self._run_seed = None
        self._group = None
        self._captured_batches = []


class PredictionJsonlWriter:
    """Stream strict JSON objects to a transactional sibling temp file."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._output_path = Path(path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(self._output_path.parent),
            prefix=f".{self._output_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self._temp_path: Optional[Path] = Path(self._handle.name)
        self._row_count = 0

    @property
    def row_count(self) -> int:
        return self._row_count

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        if self._handle is None:
            raise ValueError("prediction JSONL writer is closed")
        for row in rows:
            payload = json.dumps(row, allow_nan=False, separators=(",", ":"))
            self._handle.write(payload)
            self._handle.write("\n")
            self._row_count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def commit(self) -> None:
        if self._temp_path is None:
            return
        self.close()
        temp_path = self._temp_path
        try:
            os.replace(temp_path, self._output_path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            self._temp_path = None
            raise
        self._temp_path = None

    def abort(self) -> None:
        self.close()
        if self._temp_path is not None:
            self._temp_path.unlink(missing_ok=True)
            self._temp_path = None

    def __enter__(self) -> "PredictionJsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()
