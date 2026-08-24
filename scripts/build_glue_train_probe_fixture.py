#!/usr/bin/env python3
"""Build the fixed training-probe identities for supported GLUE tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from glue_data_protocol import (
    GLUE_DATASET_REPO,
    GLUE_DATASET_REVISION,
    SUPPORTED_DATASETS,
    build_train_probe,
    load_train_probe_fixture,
    write_train_probe_fixture,
)


def build_fixture(
    output: str | Path,
    *,
    load_dataset_fn: Callable[..., Any],
):
    identities = {}
    for dataset in SUPPORTED_DATASETS:
        dataset_dict = load_dataset_fn(
            GLUE_DATASET_REPO,
            dataset,
            revision=GLUE_DATASET_REVISION,
        )
        try:
            raw_train = dataset_dict["train"]
        except Exception as exc:
            raise RuntimeError(
                f"GLUE task {dataset} has no training split"
            ) from exc
        _, identities[dataset] = build_train_probe(
            raw_train,
            dataset=dataset,
        )
    write_train_probe_fixture(output, identities)
    return load_train_probe_fixture(output)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the pinned 256-example GLUE training probes.",
    )
    parser.add_argument(
        "--output",
        default="fixtures/reproducibility/glue_train_probe_v1.json",
    )
    args = parser.parse_args()

    from datasets import load_dataset

    fixture = build_fixture(args.output, load_dataset_fn=load_dataset)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "dataset_revision": fixture.dataset_revision,
        "tasks": {
            dataset: {
                "probe_size": len(fixture.identity_for(dataset).raw_ids),
                "ordered_identity_hash": fixture.identity_for(
                    dataset
                ).ordered_identity_hash,
            }
            for dataset in fixture.task_names
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
