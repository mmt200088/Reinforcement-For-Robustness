#!/usr/bin/env python3
"""Build the frozen raw-row MRPC comparator fixture from local Parquet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from mrpc_reproducibility import (  # noqa: E402
    MRPC_DATASET_REVISION,
    MRPC_FULL_EXAMPLE_COUNT,
    MRPC_FULL_LABEL_HISTOGRAM,
    MRPC_FULL_SHUFFLE_SEED,
    MRPC_PROBE_EXAMPLE_COUNT,
    MRPC_PROBE_SEED,
    MRPCFixture,
    MRPCReproducibilityError,
    build_mrpc_fixture,
)


def _label_histogram(rows: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for position, row in enumerate(rows):
        value = row.get("label")
        if isinstance(value, bool) or value not in (0, 1, "0", "1"):
            raise MRPCReproducibilityError(f"MRPC validation row {position} has a non-binary label")
        counts[int(value)] += 1
    return counts


def build_mrpc_reproducibility_fixture_from_dataset(
    validation: Any,
    *,
    dataset_revision: str,
    full_shuffle_seed: int = MRPC_FULL_SHUFFLE_SEED,
    probe_seed: int = MRPC_PROBE_SEED,
    probe_size: int = MRPC_PROBE_EXAMPLE_COUNT,
    expected_row_count: int = MRPC_FULL_EXAMPLE_COUNT,
    expected_label_histogram: Mapping[int, int] = MRPC_FULL_LABEL_HISTOGRAM,
    split_fn: Callable[..., tuple[Sequence[int], Sequence[int]]],
) -> MRPCFixture:
    """Reproduce the historical shuffled-full and stratified-probe orders."""
    source_rows = list(validation)
    if len(source_rows) != int(expected_row_count):
        raise MRPCReproducibilityError(
            f"MRPC validation row count mismatch: expected {int(expected_row_count)}, got {len(source_rows)}"
        )
    expected_histogram = {int(key): int(value) for key, value in dict(expected_label_histogram).items()}
    actual_histogram = _label_histogram(source_rows)
    if actual_histogram != expected_histogram:
        raise MRPCReproducibilityError(
            f"MRPC validation label histogram mismatch: expected {expected_histogram}, got {actual_histogram}"
        )
    if not hasattr(validation, "shuffle"):
        raise MRPCReproducibilityError("MRPC validation dataset cannot reproduce the historical shuffle")

    historical_full = validation.shuffle(seed=int(full_shuffle_seed))
    historical_rows = list(historical_full)
    if len(historical_rows) != len(source_rows):
        raise MRPCReproducibilityError("MRPC historical full order changed the validation row count")
    full_validation_ids = [int(row["idx"]) for row in historical_rows]
    historical_labels = [int(row["label"]) for row in historical_rows]

    requested_probe_size = int(probe_size)
    if requested_probe_size <= 0 or requested_probe_size > len(historical_rows):
        raise MRPCReproducibilityError("MRPC probe size is outside the validation split")
    selected, _remainder = split_fn(
        list(range(len(historical_rows))),
        train_size=requested_probe_size,
        shuffle=True,
        random_state=int(probe_seed),
        stratify=historical_labels,
    )
    selected_positions = sorted(int(value) for value in selected)
    if (
        len(selected_positions) != requested_probe_size
        or len(set(selected_positions)) != requested_probe_size
        or any(position < 0 or position >= len(historical_rows) for position in selected_positions)
    ):
        raise MRPCReproducibilityError("MRPC historical stratified probe positions are invalid")
    probe_ids = [full_validation_ids[position] for position in selected_positions]
    return build_mrpc_fixture(
        source_rows,
        full_validation_ids=full_validation_ids,
        probe_ids=probe_ids,
        dataset_revision=dataset_revision,
        full_shuffle_seed=full_shuffle_seed,
        probe_seed=probe_seed,
    )


def write_mrpc_reproducibility_fixture(
    path: str | Path,
    fixture: MRPCFixture,
) -> None:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            fixture.as_payload(),
            handle,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the MRPC raw-row fixture while reproducing the historical full-validation and probe orders."
        )
    )
    parser.add_argument("--revision", default=MRPC_DATASET_REVISION)
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--full-shuffle-seed",
        type=int,
        default=MRPC_FULL_SHUFFLE_SEED,
    )
    parser.add_argument(
        "--probe-seed",
        type=int,
        default=MRPC_PROBE_SEED,
    )
    args = parser.parse_args()

    from datasets import Dataset
    from sklearn.model_selection import train_test_split

    source = Path(args.source_parquet)
    if not source.is_file():
        raise MRPCReproducibilityError(f"MRPC validation Parquet is unavailable: {source}")
    validation = Dataset.from_parquet(os.fspath(source))
    fixture = build_mrpc_reproducibility_fixture_from_dataset(
        validation,
        dataset_revision=args.revision,
        full_shuffle_seed=args.full_shuffle_seed,
        probe_seed=args.probe_seed,
        split_fn=train_test_split,
    )
    output = Path(args.output)
    write_mrpc_reproducibility_fixture(output, fixture)
    print(
        json.dumps(
            {
                "output": os.fspath(output),
                "full_example_count": len(fixture.canonical_rows),
                "probe_example_count": len(fixture.probe_ids),
                "label_histogram": fixture.label_histogram,
                "probe_label_histogram": fixture.probe_label_histogram,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
