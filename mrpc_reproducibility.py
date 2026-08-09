"""Frozen raw-row MRPC views for reproducible comparator evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

MRPC_FIXTURE_SCHEMA_VERSION = "mrpc_reproducibility_v1"
MRPC_DATASET_REPO = "nyu-mll/glue"
MRPC_DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
MRPC_MODEL_ID = "textattack/bert-base-uncased-mrpc"
MRPC_MODEL_REVISION = "d421614df8fbeb22d6826a24d6397809fdc1e3ff"
MRPC_TOKENIZER_REVISION = MRPC_MODEL_REVISION
MRPC_FULL_SHUFFLE_SEED = 42
MRPC_PROBE_SEED = 42
MRPC_FULL_EXAMPLE_COUNT = 408
MRPC_PROBE_EXAMPLE_COUNT = 256
MRPC_FULL_LABEL_HISTOGRAM = {0: 129, 1: 279}
MRPC_PROBE_LABEL_HISTOGRAM = {0: 81, 1: 175}
MRPC_COMPARATOR_BATCH_SIZE = 64
MRPC_MAX_LENGTH = 128
MRPC_NUM_LAYERS = 12

_INTEGER_TEXT = re.compile(r"[+-]?\d+\Z")
_FIXTURE_FIELDS = {
    "schema_version",
    "dataset_repo",
    "task",
    "split",
    "dataset_revision",
    "full_shuffle_seed",
    "probe_seed",
    "canonical_rows",
    "full_validation_ids",
    "probe_ids",
}


class MRPCReproducibilityError(RuntimeError):
    """Raised when an MRPC reproducibility input is inconsistent."""


@dataclass(frozen=True)
class MRPCRow:
    idx: int
    label: int
    sentence1: str
    sentence2: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "idx": int(self.idx),
            "label": int(self.label),
            "sentence1": self.sentence1,
            "sentence2": self.sentence2,
        }


@dataclass(frozen=True)
class MRPCFixture:
    canonical_rows: tuple[MRPCRow, ...]
    full_validation_ids: tuple[int, ...]
    probe_ids: tuple[int, ...]
    dataset_revision: str
    full_shuffle_seed: int = MRPC_FULL_SHUFFLE_SEED
    probe_seed: int = MRPC_PROBE_SEED

    @property
    def label_histogram(self) -> dict[int, int]:
        return _label_histogram(self.canonical_rows)

    @property
    def probe_label_histogram(self) -> dict[int, int]:
        rows_by_id = {row.idx: row for row in self.canonical_rows}
        return _label_histogram(rows_by_id[sample_id] for sample_id in self.probe_ids)

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MRPC_FIXTURE_SCHEMA_VERSION,
            "dataset_repo": MRPC_DATASET_REPO,
            "task": "mrpc",
            "split": "validation",
            "dataset_revision": self.dataset_revision,
            "full_shuffle_seed": int(self.full_shuffle_seed),
            "probe_seed": int(self.probe_seed),
            "canonical_rows": [row.as_dict() for row in self.canonical_rows],
            "full_validation_ids": list(self.full_validation_ids),
            "probe_ids": list(self.probe_ids),
        }


@dataclass(frozen=True)
class MRPCValidationViews:
    full_validation: Any
    stability_probe: Any


@dataclass(frozen=True)
class MRPCReproducibilityContext:
    fixture: MRPCFixture
    stability_probe: Any


def _fail(message: str) -> None:
    raise MRPCReproducibilityError(message)


def _exact_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        _fail(f"{field} must be an integer, not bool")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and _INTEGER_TEXT.fullmatch(value):
        return int(value)
    _fail(f"{field} must be an integer or pure integer string")


def _binary_label(value: Any, *, field: str) -> int:
    label = _exact_integer(value, field=field)
    if label not in (0, 1):
        _fail(f"{field} must be binary 0 or 1")
    return label


def _canonicalize_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[MRPCRow, ...]:
    canonical = []
    seen_ids = set()
    for position, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            _fail(f"validation row {position} must be an object")
        missing = [name for name in ("idx", "label", "sentence1", "sentence2") if name not in raw_row]
        if missing:
            _fail(f"validation row {position} is missing columns {missing}")
        sample_id = _exact_integer(raw_row["idx"], field=f"row {position} idx")
        if sample_id in seen_ids:
            _fail(f"validation idx {sample_id} is duplicated")
        seen_ids.add(sample_id)
        sentence1 = raw_row["sentence1"]
        sentence2 = raw_row["sentence2"]
        if not isinstance(sentence1, str) or not isinstance(sentence2, str):
            _fail(f"validation row {position} sentences must be strings")
        canonical.append(
            MRPCRow(
                idx=sample_id,
                label=_binary_label(raw_row["label"], field=f"row {position} label"),
                sentence1=sentence1,
                sentence2=sentence2,
            )
        )
    canonical.sort(key=lambda row: row.idx)
    return tuple(canonical)


def _normalize_ids(values: Any, *, field: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail(f"{field} must be an integer sequence")
    normalized = tuple(_exact_integer(value, field=f"{field}[{position}]") for position, value in enumerate(values))
    if len(set(normalized)) != len(normalized):
        _fail(f"{field} contains duplicate IDs")
    return normalized


def _label_histogram(rows: Iterable[MRPCRow]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for row in rows:
        counts[int(row.label)] += 1
    return counts


def _validate_fixture(fixture: MRPCFixture) -> None:
    if fixture.dataset_revision != MRPC_DATASET_REVISION:
        _fail("MRPC dataset revision does not match the pinned revision")
    if fixture.full_shuffle_seed != MRPC_FULL_SHUFFLE_SEED:
        _fail(f"MRPC full shuffle seed must be {MRPC_FULL_SHUFFLE_SEED}")
    if fixture.probe_seed != MRPC_PROBE_SEED:
        _fail(f"MRPC probe seed must be {MRPC_PROBE_SEED}")

    source_ids = tuple(row.idx for row in fixture.canonical_rows)
    if source_ids != tuple(sorted(source_ids)):
        _fail("MRPC canonical rows must use numeric idx order")
    if len(set(source_ids)) != len(source_ids):
        _fail("MRPC canonical rows contain duplicate IDs")
    if len(fixture.full_validation_ids) != len(source_ids) or set(fixture.full_validation_ids) != set(source_ids):
        _fail("full_validation_ids must be a permutation of canonical row IDs")
    if not set(fixture.probe_ids).issubset(source_ids):
        _fail("probe_ids reference missing canonical row IDs")


def build_mrpc_fixture(
    validation_rows: Iterable[Mapping[str, Any]],
    *,
    full_validation_ids: Sequence[int],
    probe_ids: Sequence[int],
    dataset_revision: str,
    full_shuffle_seed: int = MRPC_FULL_SHUFFLE_SEED,
    probe_seed: int = MRPC_PROBE_SEED,
) -> MRPCFixture:
    fixture = MRPCFixture(
        canonical_rows=_canonicalize_rows(validation_rows),
        full_validation_ids=_normalize_ids(
            full_validation_ids,
            field="full_validation_ids",
        ),
        probe_ids=_normalize_ids(probe_ids, field="probe_ids"),
        dataset_revision=str(dataset_revision or "").strip(),
        full_shuffle_seed=_exact_integer(
            full_shuffle_seed,
            field="full shuffle seed",
        ),
        probe_seed=_exact_integer(probe_seed, field="probe seed"),
    )
    _validate_fixture(fixture)
    return fixture


def load_mrpc_fixture(path: str | Path) -> MRPCFixture:
    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MRPCReproducibilityError(f"cannot load MRPC reproducibility fixture at {fixture_path}") from exc
    if not isinstance(payload, Mapping):
        _fail("MRPC reproducibility fixture must be an object")
    if set(payload) != _FIXTURE_FIELDS:
        missing = sorted(_FIXTURE_FIELDS.difference(payload))
        extra = sorted(set(payload).difference(_FIXTURE_FIELDS))
        _fail(f"MRPC fixture field set mismatch: missing={missing}, extra={extra}")
    if payload["schema_version"] != MRPC_FIXTURE_SCHEMA_VERSION:
        _fail("MRPC reproducibility fixture schema version is unsupported")
    if payload["dataset_repo"] != MRPC_DATASET_REPO:
        _fail("MRPC reproducibility fixture dataset repo mismatch")
    if payload["task"] != "mrpc" or payload["split"] != "validation":
        _fail("MRPC reproducibility fixture task/split mismatch")

    fixture = build_mrpc_fixture(
        payload["canonical_rows"],
        full_validation_ids=payload["full_validation_ids"],
        probe_ids=payload["probe_ids"],
        dataset_revision=payload["dataset_revision"],
        full_shuffle_seed=payload["full_shuffle_seed"],
        probe_seed=payload["probe_seed"],
    )
    if len(fixture.canonical_rows) != MRPC_FULL_EXAMPLE_COUNT:
        _fail(
            "MRPC fixture full row count mismatch: "
            f"expected {MRPC_FULL_EXAMPLE_COUNT}, got {len(fixture.canonical_rows)}"
        )
    if len(fixture.probe_ids) != MRPC_PROBE_EXAMPLE_COUNT:
        _fail(
            f"MRPC fixture probe row count mismatch: expected {MRPC_PROBE_EXAMPLE_COUNT}, got {len(fixture.probe_ids)}"
        )
    if fixture.label_histogram != MRPC_FULL_LABEL_HISTOGRAM:
        _fail("MRPC fixture full label histogram mismatch")
    if fixture.probe_label_histogram != MRPC_PROBE_LABEL_HISTOGRAM:
        _fail("MRPC fixture probe label histogram mismatch")
    return fixture


def _validation_split(dataset: Any) -> Any:
    try:
        return dataset["validation"]
    except Exception as exc:
        raise MRPCReproducibilityError("MRPC dataset is missing the validation split") from exc


def _select_rows(validation: Any, positions: Sequence[int], rows: Sequence[MRPCRow]) -> Any:
    if hasattr(validation, "select"):
        return validation.select(list(positions))
    return [row.as_dict() for row in rows]


def resolve_mrpc_validation_views(
    dataset: Any,
    fixture: MRPCFixture,
    *,
    expected_row_count: int | None = MRPC_FULL_EXAMPLE_COUNT,
) -> MRPCValidationViews:
    _validate_fixture(fixture)
    validation = _validation_split(dataset)
    try:
        physical_rows = list(validation)
    except Exception as exc:
        raise MRPCReproducibilityError("MRPC validation split cannot be enumerated") from exc
    canonical_rows = _canonicalize_rows(physical_rows)
    if expected_row_count is not None and len(canonical_rows) != int(expected_row_count):
        _fail(f"MRPC validation row count mismatch: expected {int(expected_row_count)}, got {len(canonical_rows)}")
    if canonical_rows != fixture.canonical_rows:
        _fail("MRPC validation raw rows do not match the reproducibility fixture")

    position_by_id = {}
    for position, raw_row in enumerate(physical_rows):
        if not isinstance(raw_row, Mapping) or "idx" not in raw_row:
            _fail(f"validation row {position} is missing idx")
        sample_id = _exact_integer(raw_row["idx"], field=f"row {position} idx")
        if sample_id in position_by_id:
            _fail(f"validation idx {sample_id} is duplicated")
        position_by_id[sample_id] = position

    rows_by_id = {row.idx: row for row in fixture.canonical_rows}

    def positions_for(ids: Sequence[int]) -> list[int]:
        missing = [sample_id for sample_id in ids if sample_id not in position_by_id]
        if missing:
            _fail(f"MRPC view references missing IDs {missing[:5]}")
        return [position_by_id[sample_id] for sample_id in ids]

    full_rows = [rows_by_id[sample_id] for sample_id in fixture.full_validation_ids]
    probe_rows = [rows_by_id[sample_id] for sample_id in fixture.probe_ids]
    return MRPCValidationViews(
        full_validation=_select_rows(
            validation,
            positions_for(fixture.full_validation_ids),
            full_rows,
        ),
        stability_probe=_select_rows(
            validation,
            positions_for(fixture.probe_ids),
            probe_rows,
        ),
    )


def resolve_mrpc_pretrained_revision_kwargs(
    *,
    fixture: MRPCFixture | None,
    data_path: str,
    model_id: str,
) -> tuple[dict[str, str], dict[str, str]]:
    if fixture is None:
        return {}, {}
    if str(data_path or "").strip().lower() != "mrpc":
        _fail("MRPC reproducibility fixture requires the MRPC task")
    if str(model_id or "").strip().lower() != MRPC_MODEL_ID:
        _fail(f"MRPC reproducibility requires model {MRPC_MODEL_ID}")
    return (
        {"revision": MRPC_MODEL_REVISION},
        {"revision": MRPC_TOKENIZER_REVISION},
    )


def _runtime_model_id(model: Any) -> str:
    config = getattr(model, "config", None)
    return str(getattr(config, "_name_or_path", "") or "").strip().lower()


def validate_mrpc_evaluation_setup(
    *,
    model: Any,
    tokenizer: Any,
    collator: Any,
    full_validation: Any,
    stability_probe: Any,
    batch_size: int,
) -> None:
    if _runtime_model_id(model) != MRPC_MODEL_ID:
        _fail(f"MRPC reproducibility requires model {MRPC_MODEL_ID}")
    config = getattr(model, "config", None)
    if str(getattr(config, "_commit_hash", "") or "").strip() != MRPC_MODEL_REVISION:
        _fail("MRPC model revision does not match the pinned revision")
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if not isinstance(init_kwargs, Mapping):
        _fail("MRPC tokenizer revision metadata is unavailable")
    tokenizer_model_id = (
        str(getattr(tokenizer, "name_or_path", "") or init_kwargs.get("name_or_path") or "").strip().lower()
    )
    if tokenizer_model_id != MRPC_MODEL_ID:
        _fail(f"MRPC reproducibility requires tokenizer {MRPC_MODEL_ID}")
    tokenizer_commit = str(init_kwargs.get("_commit_hash") or "").strip()
    if tokenizer_commit and tokenizer_commit != MRPC_TOKENIZER_REVISION:
        _fail("MRPC tokenizer revision does not match the pinned revision")
    if _exact_integer(batch_size, field="MRPC comparator batch size") != MRPC_COMPARATOR_BATCH_SIZE:
        _fail(f"MRPC comparator batch size must be {MRPC_COMPARATOR_BATCH_SIZE}")
    if (
        _exact_integer(
            getattr(config, "num_hidden_layers", None),
            field="MRPC model layer count",
        )
        != MRPC_NUM_LAYERS
    ):
        _fail(f"MRPC reproducibility requires {MRPC_NUM_LAYERS} model layers")
    if len(full_validation) != MRPC_FULL_EXAMPLE_COUNT:
        _fail("MRPC full validation size mismatch")
    if len(stability_probe) != MRPC_PROBE_EXAMPLE_COUNT:
        _fail("MRPC stability probe size mismatch")

    collator_type = type(collator)
    if collator_type.__name__ != "DataCollatorWithPadding" or not collator_type.__module__.startswith("transformers"):
        _fail("MRPC evaluation requires transformers.DataCollatorWithPadding")
    expected = {
        "padding": "max_length",
        "max_length": MRPC_MAX_LENGTH,
        "return_tensors": "pt",
        "pad_to_multiple_of": 8,
    }
    actual = {name: getattr(collator, name, None) for name in expected}
    if actual != expected:
        _fail("MRPC collator configuration mismatch")
