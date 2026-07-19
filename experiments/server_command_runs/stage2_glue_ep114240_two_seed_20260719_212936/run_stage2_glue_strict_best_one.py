#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np


def _normalized(value):
    if dataclasses.is_dataclass(value):
        return _normalized(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalized(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        _normalized(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cfg_blocks(decoded):
    return {
        "block1": decoded.block1_cfgs,
        "block2": decoded.block2_cfgs,
        "block4": decoded.block4_cfgs,
        "block5": decoded.block5_cfgs,
    }


def _handler_blocks(handler):
    return {
        "block1": handler.block1_cfg_per_layer,
        "block2": handler.block2_cfg_per_layer,
        "block4": handler.block4_cfg_per_layer,
        "block5": handler.block5_cfg_per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-config", required=True)
    parser.add_argument("--template-zip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--local-mrpc-parquet-dir", required=True)
    parser.add_argument("--reference-test-arrow", required=True)
    args = parser.parse_args()

    from json_utils import read_json_file
    from Paean.action_grid import load_action_grid_config
    from blb_stage2_rl.action_space import load_max_sfs
    from blb_rl_bridge import BLBNoiseRLBridge
    from datasets import Dataset, DatasetDict
    import generate_glue_submission as glue

    action_path = Path(args.action_config).resolve()
    template_path = Path(args.template_zip).resolve()
    output_dir = Path(args.output_dir).resolve()
    parquet_dir = Path(args.local_mrpc_parquet_dir).resolve()
    reference_test_arrow = Path(args.reference_test_arrow).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(template_path) as archive:
        archive.extractall(output_dir)
    stale_zip = output_dir / "submission.zip"
    if stale_zip.exists():
        stale_zip.unlink()

    parquet_paths = {
        split: parquet_dir / f"{split}-00000-of-00001.parquet"
        for split in ("train", "validation", "test")
    }
    assert all(path.is_file() for path in parquet_paths.values())
    assert reference_test_arrow.is_file()
    local_dataset = DatasetDict({
        split: Dataset.from_parquet(str(path))
        for split, path in parquet_paths.items()
    })
    reference_test = Dataset.from_file(str(reference_test_arrow))
    assert len(local_dataset["test"]) == len(reference_test) == 1725
    assert local_dataset["test"].column_names == reference_test.column_names
    dataset_columns = reference_test.column_names
    dataset_mismatches = sum(
        local_dataset["test"][index] != reference_test[index]
        for index in range(len(reference_test))
    )
    assert dataset_mismatches == 0
    test_rows_sha256 = _sha256_bytes(
        b"".join(
            _canonical_bytes({key: reference_test[index][key] for key in dataset_columns}) + b"\n"
            for index in range(len(reference_test))
        )
    )

    payload = read_json_file(action_path, encoding="utf-8-sig")
    assert payload["schema_version"] == "fusion_count_fixed_action_v1"
    assert payload["profile"] == "mrpc"
    assert payload["num_layers"] == 12
    assert payload["gelu_degree"] == [4] * 12
    assert payload["attn_degree"] == [6] * 12
    assert payload["summary"]["step_count"] == 47
    assert payload["summary"]["total_fusion_count"] == 27
    assert payload["summary"]["boosted_option_count"] == 27

    choices = payload["group"]["choices_by_step"]
    b2 = [row for row in choices if row["block"] == 2]
    b4 = [row for row in choices if row["block"] == 4]
    b5 = [row for row in choices if row["block"] == 5]
    assert len(b2) == len(b4) == len(b5) == 12
    assert all(row["option_id"] == 1 and row["fusion_count"] == 1 for row in b2 + b5)
    assert [row["layer"] for row in b4 if row["option_id"] == 1] == [5, 6, 7]
    assert all(row["fusion_count"] == row["option_id"] for row in b4)

    grid = load_action_grid_config(
        str(action_path), num_layers_hint=12, profile="mrpc",
        gelu_degree=[4] * 12, attn_degree=[6] * 12,
    )
    assert grid.base_action_vec is not None and not isinstance(grid.base_action_vec, str)
    action_vec = np.asarray(grid.base_action_vec, dtype=int)
    assert action_vec.size == 877
    fusion_metadata = {
        "schema_version": "fusion_count_fixed_action_v1",
        "group": payload["group"],
    }
    decoded = glue._decode_blb_action_for_glue(
        action_vec=action_vec,
        fusion_metadata=fusion_metadata,
        profile="mrpc",
        gelu_degrees=[4] * 12,
        softmax_degrees=[6] * 12,
        max_sfs=load_max_sfs("mrpc"),
    )
    expected_blocks = _normalized(_cfg_blocks(decoded))
    expected_hash = _sha256_bytes(_canonical_bytes(expected_blocks))
    audit_path = output_dir / "install_audit.json"
    original_apply = BLBNoiseRLBridge.apply
    install_call_count = 0

    def audited_apply(self, **kwargs):
        nonlocal install_call_count
        install_call_count += 1
        supplied = _normalized({
            "block1": kwargs.get("block1_cfgs") or {},
            "block2": kwargs.get("block2_cfgs") or {},
            "block4": kwargs.get("block4_cfgs") or {},
            "block5": kwargs.get("block5_cfgs") or {},
        })
        supplied_hash = _sha256_bytes(_canonical_bytes(supplied))
        assert supplied_hash == expected_hash
        result = original_apply(self, **kwargs)
        installed = _normalized(_handler_blocks(self.handler))
        installed_hash = _sha256_bytes(_canonical_bytes(installed))
        assert installed_hash == expected_hash
        audit = {
            "schema_version": "stage2_glue_install_audit_v1",
            "seed": int(args.seed),
            "candidate_key": "b89b74a0b2e56e7931a8c1f1972348456451d2b9b64692573df8e28ac7366053",
            "action_config_sha256": _sha256_file(action_path),
            "action_vec_sha256": _sha256_bytes(_canonical_bytes(action_vec.tolist())),
            "expected_config_sha256": expected_hash,
            "supplied_to_bridge_sha256": supplied_hash,
            "installed_in_handler_sha256": installed_hash,
            "installed_layers": {
                str(layer): sorted(blocks)
                for layer, blocks in sorted(self.installed_layers().items())
            },
            "config": installed,
        }
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        return result

    BLBNoiseRLBridge.apply = audited_apply
    original_load_dataset = glue.load_dataset

    def audited_local_load_dataset(path, task_name):
        assert path == "nyu-mll/glue"
        assert task_name == "mrpc"
        return local_dataset

    glue.load_dataset = audited_local_load_dataset
    try:
        glue._seed_all_for_reproducibility(int(args.seed))
        glue._process_blb_task(
            task_name="mrpc",
            task_config=dict(glue.TASK_REGISTRY["mrpc"]),
            action_vec=action_vec,
            profile="mrpc",
            gelu_degrees=[4] * 12,
            softmax_degrees=[6] * 12,
            output_dir=str(output_dir),
            device=str(args.device),
            max_length=128,
            batch_size=int(args.batch_size),
            fusion_metadata=fusion_metadata,
        )
    finally:
        glue.load_dataset = original_load_dataset
        BLBNoiseRLBridge.apply = original_apply

    assert install_call_count == 1
    assert audit_path.is_file()
    mrpc_path = output_dir / "MRPC.tsv"
    rows = mrpc_path.read_text().splitlines()
    assert len(rows) == 1726
    assert rows[0] == "index\tprediction"
    labels = [line.split("\t", 1)[1] for line in rows[1:]]
    assert set(labels) <= {"0", "1"}
    assert glue.verify_outputs(str(output_dir))
    zip_path = Path(glue.create_submission_zip(str(output_dir)))
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(archive.namelist())
        assert members == sorted(glue.EXPECTED_LINES)
        assert archive.testzip() is None
        member_lines = {name: len(archive.read(name).splitlines()) for name in members}
    manifest = {
        "schema_version": "stage2_glue_submission_run_v1",
        "seed": int(args.seed),
        "candidate_key": "b89b74a0b2e56e7931a8c1f1972348456451d2b9b64692573df8e28ac7366053",
        "source_training_commit": "fa4ee9cbd27d6265238f8d1091b712e14ee86066",
        "submission_runtime_commit": "3889a9d0c215c2d4603ab4459fdf347599140cad",
        "action_config_sha256": _sha256_file(action_path),
        "template_zip_sha256": _sha256_file(template_path),
        "mrpc_tsv_sha256": _sha256_file(mrpc_path),
        "submission_zip_sha256": _sha256_file(zip_path),
        "mrpc_label_counts": {label: labels.count(label) for label in sorted(set(labels))},
        "zip_members": members,
        "zip_member_lines": member_lines,
        "install_audit_sha256": _sha256_file(audit_path),
        "install_config_sha256": expected_hash,
        "dataset": {
            "route": "verified_local_parquet",
            "test_rows": len(reference_test),
            "test_columns": dataset_columns,
            "test_row_mismatches_vs_cached_arrow": dataset_mismatches,
            "test_rows_canonical_sha256": test_rows_sha256,
            "reference_test_arrow": str(reference_test_arrow),
            "reference_test_arrow_sha256": _sha256_file(reference_test_arrow),
            "parquet_files": {
                split: {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "rows": len(local_dataset[split]),
                }
                for split, path in parquet_paths.items()
            },
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
