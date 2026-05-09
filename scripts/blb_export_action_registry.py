"""Export BLB Stage-2 action registry artifacts requested by the playbook."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blb_stage2_rl.action_space import (  # noqa: E402
    K_LEVELS,
    describe_action_vector,
    load_max_sfs,
    make_all_max_action_vector,
)


def _parse_degree_vector(raw: str | Sequence[int] | None, *, num_layers: int, default: int) -> List[int]:
    if raw is None or raw == "":
        return [int(default)] * int(num_layers)
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            values = json.loads(text)
        else:
            values = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    else:
        values = list(raw)
    out = [int(v) for v in values]
    if len(out) == 1:
        return out * int(num_layers)
    if len(out) != int(num_layers):
        raise ValueError(f"degree vector length {len(out)} must be 1 or num_layers={num_layers}")
    return out


def _scale_semantics(record: Dict[str, Any]) -> str:
    kind = str(record.get("kind", ""))
    if kind in ("F", "W", "M", "S"):
        return "encode/fresh increases current CKKS scale and controls simulator noise variance"
    if kind == "R":
        return "rescale action selects the target scale after this CKKS rescale point"
    if kind == "K":
        return "truncation action selects MPC/CKKS conversion fractional bits"
    return "see action_space field kind"


def _all_max_action_index(record: Dict[str, Any]) -> int:
    if record.get("value_type") == "truncation_k":
        return int(list(K_LEVELS).index(max(K_LEVELS)))
    return int(record.get("num_levels", 1)) - 1


def _registry_record(record: Dict[str, Any]) -> Dict[str, Any]:
    note = str(record.get("note", "") or "")
    return {
        "global_index": int(record["global_index"]),
        "layer": int(record["layer"]),
        "block_index": record.get("block_index"),
        "block": str(record["block"]),
        "field": str(record["field"]),
        "kind": str(record["kind"]),
        "operation": str(record["operation"]),
        "location": str(record["location"]),
        "config_name": str(record["config_name"]),
        "is_required": bool(record.get("effective", True)),
        "is_effective": bool(record.get("effective", True)),
        "ineffective_reason": note,
        "value_type": str(record["value_type"]),
        "N": record.get("N"),
        "max_sf": record.get("max_sf"),
        "num_levels": int(record["num_levels"]),
        "action_index": int(record["action_index"]),
        "action_values": [int(v) for v in record.get("level_values", [])],
        "level_values": [int(v) for v in record.get("level_values", [])],
        "decoded_value": int(record["value"]),
        "effective_value": record.get("effective_value"),
        "all_max_action_index": _all_max_action_index(record),
        "distribution": str(record.get("distribution", "")),
        "scale_semantics": _scale_semantics(record),
        "rotation_dependency": None,
        "source": "blb_stage2_rl.action_space.describe_action_vector",
    }


def _required_count_by_layer(records: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for record in records:
        if record["block"] == "first_input":
            continue
        if not record["is_required"]:
            continue
        layer = int(record["layer"])
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def _mismatch_markdown(
        *,
        profile: str,
        num_layers: int,
        expected_required_slots_per_layer: int,
        records: Sequence[Dict[str, Any]],
        ) -> str:
    counts = _required_count_by_layer(records)
    mismatched = {
        layer: count
        for layer, count in sorted(counts.items())
        if int(count) != int(expected_required_slots_per_layer)
    }
    status = "match" if not mismatched and len(counts) == int(num_layers) else "mismatch"
    lines = [
        "# BLB Action Registry Required-Slot Check",
        "",
        f"- profile: `{profile}`",
        f"- num_layers: `{int(num_layers)}`",
        f"- expected_required_slots_per_layer: `{int(expected_required_slots_per_layer)}`",
        f"- status: `{status}`",
        f"- effective_required_total: `{sum(counts.values())}`",
        "",
        "| layer | required_slots | status |",
        "|---:|---:|---|",
    ]
    for layer in range(int(num_layers)):
        count = int(counts.get(layer, 0))
        row_status = "ok" if count == int(expected_required_slots_per_layer) else "mismatch"
        lines.append(f"| {layer} | {count} | {row_status} |")
    if mismatched:
        lines.extend([
            "",
            "Safe handling: keep every current action field in the registry; mark non-required fields as compat or ineffective extras instead of deleting them.",
        ])
    return "\n".join(lines) + "\n"


def _mapping_markdown(records: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# BLB Action Index Mapping",
        "",
        "| idx | location | kind | values | all_max_idx | required |",
        "|---:|---|---|---|---:|---|",
    ]
    for record in records:
        values = ",".join(str(v) for v in record["action_values"])
        lines.append(
            f"| {record['global_index']} | `{record['location']}` | `{record['kind']}` | "
            f"`{values}` | {record['all_max_action_index']} | {str(record['is_required']).lower()} |"
        )
    return "\n".join(lines) + "\n"


def build_registry_payload(
        *,
        profile: str,
        num_layers: int,
        gelu_degree: str | Sequence[int] | None = None,
        attn_degree: str | Sequence[int] | None = None,
        expected_required_slots_per_layer: int = 59,
        ) -> Dict[str, Any]:
    gelu = _parse_degree_vector(gelu_degree, num_layers=num_layers, default=4)
    attn = _parse_degree_vector(attn_degree, num_layers=num_layers, default=4)
    action = make_all_max_action_vector(num_layers=num_layers)
    description = describe_action_vector(
        action,
        max_sfs=load_max_sfs(profile),
        num_layers=num_layers,
        gelu_degree=gelu,
        attn_degree=attn,
        profile=profile,
    )
    records = [_registry_record(record) for record in description["records"]]
    effective = [record for record in records if record["is_required"]]
    mismatch = _mismatch_markdown(
        profile=profile,
        num_layers=num_layers,
        expected_required_slots_per_layer=expected_required_slots_per_layer,
        records=records,
    )
    return {
        "schema": "blb_action_registry_export_v1",
        "profile": str(profile),
        "num_layers": int(num_layers),
        "gelu_degree": gelu,
        "attn_degree": attn,
        "expected_required_slots_per_layer": int(expected_required_slots_per_layer),
        "summary": {
            "slot_count": len(records),
            "required_slot_count": len(effective),
            "ineffective_or_compat_extra_count": len(records) - len(effective),
            "required_count_by_layer": _required_count_by_layer(records),
        },
        "slot_registry_full": records,
        "slot_registry_effective": effective,
        "required59_or_mismatch_markdown": mismatch,
        "action_index_mapping_markdown": _mapping_markdown(records),
    }


def write_registry_artifacts(payload: Dict[str, Any], output_dir: os.PathLike[str] | str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "slot_registry_full.json"
    effective_path = out / "slot_registry_effective.json"
    mismatch_path = out / "slot_registry_required59_or_mismatch.md"
    mapping_path = out / "action_index_mapping.md"
    full_path.write_text(
        json.dumps(payload["slot_registry_full"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    effective_path.write_text(
        json.dumps(payload["slot_registry_effective"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mismatch_path.write_text(payload["required59_or_mismatch_markdown"], encoding="utf-8")
    mapping_path.write_text(payload["action_index_mapping_markdown"], encoding="utf-8")
    return {
        "slot_registry_full": str(full_path),
        "slot_registry_effective": str(effective_path),
        "required59_or_mismatch": str(mismatch_path),
        "action_index_mapping": str(mapping_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--fixed-gelu", default="")
    parser.add_argument("--fixed-softmax", default="")
    parser.add_argument("--expected-required-slots-per-layer", type=int, default=59)
    parser.add_argument("--output-dir", default="reports/blb_opt/phase1_registry")
    args = parser.parse_args(argv)

    payload = build_registry_payload(
        profile=args.profile,
        num_layers=args.num_layers,
        gelu_degree=args.fixed_gelu,
        attn_degree=args.fixed_softmax,
        expected_required_slots_per_layer=args.expected_required_slots_per_layer,
    )
    paths = write_registry_artifacts(payload, args.output_dir)
    print(json.dumps(paths, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
