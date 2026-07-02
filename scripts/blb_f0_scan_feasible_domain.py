"""F0 optimizer-only feasible-domain scan for BLB Stage-2 actions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blb_stage2_rl.candidate_store import (  # noqa: E402
    action_hash,
    build_candidate_identity_context,
    candidate_key,
    effective_action_vector,
    f0_sort_key,
    raw_action_hash,
)

_DEFAULT_K_LEVELS_LEGACY_COMPAT = (8, 9, 11, 13, 10, 12)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_int_list(raw: str | None) -> List[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return [int(x.strip()) for x in str(raw).replace(";", ",").split(",") if x.strip()]


def _load_stage1_vectors(
        path: str | None,
        *,
        model: str = "bert-base",
        profile: str = "mrpc",
        ) -> Dict[str, Any]:
    if not path:
        return {"path": "", "content_hash": "", "gelu": None, "softmax": None}
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    content_hash = _file_sha256(p) if p.exists() else None
    payload = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    candidates = payload
    if isinstance(payload, Mapping):
        model_node = payload.get(str(model))
        if isinstance(model_node, Mapping) and isinstance(model_node.get(str(profile)), Mapping):
            candidates = model_node[str(profile)].get("stage1", model_node[str(profile)])
        elif isinstance(payload.get(str(profile)), Mapping):
            candidates = payload[str(profile)].get("stage1", payload[str(profile)])
        for key in ("stage1", "stage1_search_best", "best_stage1"):
            if isinstance(payload.get(key), Mapping):
                candidates = payload[key]
                break
    gelu = None
    softmax = None
    if isinstance(candidates, Mapping):
        gelu = candidates.get("gelu", candidates.get("gelu_degrees"))
        softmax = candidates.get("softmax", candidates.get("softmax_degrees"))
    return {
        "path": str(p),
        "content_hash": content_hash or "",
        "gelu": [int(x) for x in gelu] if isinstance(gelu, Sequence) and not isinstance(gelu, (str, bytes)) else gelu,
        "softmax": [int(x) for x in softmax] if isinstance(softmax, Sequence) and not isinstance(softmax, (str, bytes)) else softmax,
    }


def canonical_rescale_optimizer_hash(root: str | os.PathLike[str], profile: str) -> str:
    from scripts.blb_make_run_manifest import _canonical_rescale_optimizer_hash

    return str(_canonical_rescale_optimizer_hash(str(root), str(profile)) or "")


def _record_value(record: Mapping[str, Any], index: int) -> Any:
    values = record.get("action_values", record.get("level_values", []))
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and 0 <= int(index) < len(values):
        return values[int(index)]
    return index


def _normalize_eval(raw: Mapping[str, Any], action: Sequence[int], source: str) -> Dict[str, Any]:
    optimizer_valid = bool(raw.get("optimizer_valid", raw.get("valid", False)))
    bits = int(raw.get("total_bits_sum", 0) or 0)
    fusion = int(raw.get("fusion_count", raw.get("total_fusion_count", 0)) or 0)
    invalid = raw.get("invalid_chain", raw.get("invalid_chains"))
    record = {
        "action_indices": [int(x) for x in action],
        "source": str(source),
        "optimizer_valid": optimizer_valid,
        "valid": optimizer_valid,
        "total_bits_sum": bits,
        "fusion_count": fusion,
        "avg_k": float(raw.get("avg_k", 0.0) or 0.0),
        "invalid_chain": invalid,
        "invalid_summary": "" if invalid in (None, {}, "") else json.dumps(invalid, ensure_ascii=True, sort_keys=True),
        "q_bits": raw.get("q_bits", []),
        "q_head_bits": raw.get("q_head_bits"),
        "q_tail_bits": raw.get("q_tail_bits"),
    }
    record["rescale_cost"] = {"total_bits_sum": bits, "fusion_count": fusion, "rank_key": [bits, fusion]}
    record["f0_sort_key"] = [int(x) if float(x).is_integer() else float(x) for x in f0_sort_key(record)]
    record["raw_action_hash"] = str(raw.get("raw_action_hash") or raw_action_hash(action))
    record["action_hash"] = record["raw_action_hash"]
    record["action_vector_hash"] = record["action_hash"]
    record["effective_action_hash"] = str(raw.get("effective_action_hash") or record["raw_action_hash"])
    record["effective_action_indices"] = [
        int(x) for x in raw.get("effective_action_indices", action)
    ]
    record["candidate_key_basis"] = str(raw.get("candidate_key_basis") or "effective_action_hash + identity_context")
    record["candidate_key"] = str(raw.get("candidate_key") or _sha256_json({
        "effective_action_hash": record["effective_action_hash"],
        "candidate_key_basis": record["candidate_key_basis"],
        "source": source,
        "bits": bits,
        "fusion": fusion,
    }))
    return record


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _safe_allowed_k_indices() -> List[int]:
    allowed_values = {13, 12, 11}
    raw = str(os.environ.get("BLB_TRUNCATION_K_LEVELS", "") or "").strip()
    if raw:
        k_levels = tuple(int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip())
    else:
        k_levels = _DEFAULT_K_LEVELS_LEGACY_COMPAT
    if not k_levels:
        k_levels = _DEFAULT_K_LEVELS_LEGACY_COMPAT
    return [idx for idx, value in enumerate(k_levels) if int(value) in allowed_values]


def _smallest_cost_rows(rows: Iterable[Mapping[str, Any]], limit: int) -> List[Mapping[str, Any]]:
    return heapq.nsmallest(
        max(0, int(limit)),
        rows,
        key=lambda row: (int(row["total_bits_sum"]), int(row["fusion_count"])),
    )


def _build_per_slot_summary_rows(
        *,
        baseline_action: Sequence[int],
        records: Sequence[Mapping[str, Any]],
        rows: Iterable[Mapping[str, Any]],
        ) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    best_bits: List[int | None] = []
    best_fusion: List[int | None] = []
    for slot_idx, baseline_idx in enumerate(baseline_action):
        record = records[slot_idx] if slot_idx < len(records) else {}
        summaries.append({
            "slot_global_index": int(slot_idx),
            "layer": record.get("layer", ""),
            "block": record.get("block", ""),
            "field": record.get("field", ""),
            "kind": record.get("kind", ""),
            "baseline_index": int(baseline_idx),
            "candidate_count": 0,
            "valid_count": 0,
            "valid_rate": 1.0,
            "improving_valid_count": 0,
            "best_delta_total_bits": 0,
            "best_delta_fusion_count": 0,
        })
        best_bits.append(None)
        best_fusion.append(None)

    for row in rows:
        try:
            slot_idx = int(row["slot_global_index"])
        except Exception:
            continue
        if slot_idx < 0 or slot_idx >= len(summaries):
            continue
        summary = summaries[slot_idx]
        summary["candidate_count"] = int(summary["candidate_count"]) + 1
        if not bool(row.get("optimizer_valid")):
            continue
        delta_bits = int(row["delta_total_bits"])
        delta_fusion = int(row["delta_fusion_count"])
        summary["valid_count"] = int(summary["valid_count"]) + 1
        if delta_bits < 0 or delta_fusion < 0:
            summary["improving_valid_count"] = int(summary["improving_valid_count"]) + 1
        best_bits[slot_idx] = delta_bits if best_bits[slot_idx] is None else min(best_bits[slot_idx], delta_bits)
        best_fusion[slot_idx] = (
            delta_fusion if best_fusion[slot_idx] is None else min(best_fusion[slot_idx], delta_fusion)
        )

    for slot_idx, summary in enumerate(summaries):
        candidate_count = int(summary["candidate_count"])
        valid_count = int(summary["valid_count"])
        summary["valid_rate"] = float(valid_count / candidate_count) if candidate_count else 1.0
        summary["best_delta_total_bits"] = int(best_bits[slot_idx]) if best_bits[slot_idx] is not None else 0
        summary["best_delta_fusion_count"] = int(best_fusion[slot_idx]) if best_fusion[slot_idx] is not None else 0
    return summaries


def _build_mask(
        *,
        baseline_action: Sequence[int],
        action_dims: Sequence[int],
        records: Sequence[Mapping[str, Any]],
        per_slot_rows: Sequence[Mapping[str, Any]],
        source: str,
        ) -> Dict[str, Any]:
    by_slot: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_slot_rows:
        by_slot[int(row["slot_global_index"])].append(row)
    slots = []
    for idx, (baseline_idx, dim) in enumerate(zip(baseline_action, action_dims)):
        record = records[idx] if idx < len(records) else {}
        allowed = {int(baseline_idx)}
        reason = "baseline_only"
        if not bool(record.get("effective", True)):
            reason = "ineffective_compat_slot_baseline_only"
        elif str(record.get("kind")) == "K":
            for k_idx in _safe_allowed_k_indices():
                if 0 <= int(k_idx) < int(dim):
                    allowed.add(int(k_idx))
            reason = "safe_k_13_12_11"
        else:
            valid_rows = [
                row for row in by_slot.get(idx, [])
                if bool(row.get("optimizer_valid")) and int(row.get("candidate_index", baseline_idx)) != int(baseline_idx)
            ]
            for row in valid_rows:
                allowed.add(int(row["candidate_index"]))
            if valid_rows:
                reason = "single_slot_f0_valid"
        allowed_indices = sorted(int(x) for x in allowed if 0 <= int(x) < int(dim))
        slots.append({
            "global_index": int(idx),
            "allowed_indices": allowed_indices,
            "baseline_index": int(baseline_idx),
            "reason": reason,
            "kind": str(record.get("kind", "")),
            "block": str(record.get("block", "")),
            "field": str(record.get("field", "")),
            "allowed_values": [
                _record_value(record, int(action_idx))
                for action_idx in allowed_indices
            ],
        })
    return {
        "schema": "blb_action_mask_v1",
        "source": str(source),
        "action_width": int(len(baseline_action)),
        "baseline_action": [int(x) for x in baseline_action],
        "slots": slots,
    }


def _beam_scan(
        *,
        baseline_action: Sequence[int],
        baseline: Mapping[str, Any],
        mutations: Sequence[Mapping[str, Any]],
        evaluate_action: Callable[[Sequence[int], str], Mapping[str, Any]],
        beam_size: int,
        beam_depths: Sequence[int],
        mutation_limit: int = 64,
        ) -> List[Dict[str, Any]]:
    depths = sorted({int(d) for d in beam_depths if int(d) > 0})
    if not depths:
        return []
    max_depth = max(depths)
    mutation_pool = list(mutations)
    if int(mutation_limit) > 0:
        mutation_pool = mutation_pool[:int(mutation_limit)]
    beam = [{
        "action": [int(x) for x in baseline_action],
        "touched": [],
        "bits": int(baseline["total_bits_sum"]),
        "fusion": int(baseline["fusion_count"]),
    }]
    rows: List[Dict[str, Any]] = []
    for depth in range(1, max_depth + 1):
        attempted = 0
        invalid = 0
        next_beam = []
        for candidate in beam:
            touched = set(candidate["touched"])
            for mut in mutation_pool:
                slot = int(mut["slot_global_index"])
                if slot in touched:
                    continue
                attempted += 1
                action = list(candidate["action"])
                action[slot] = int(mut["candidate_index"])
                evaluated = _normalize_eval(evaluate_action(action, f"beam_depth_{depth}"), action, f"beam_depth_{depth}")
                if not bool(evaluated["optimizer_valid"]):
                    invalid += 1
                    continue
                next_beam.append({
                    "action": action,
                    "touched": list(touched | {slot}),
                    "bits": int(evaluated["total_bits_sum"]),
                    "fusion": int(evaluated["fusion_count"]),
                })
        next_beam.sort(key=lambda item: (int(item["bits"]), int(item["fusion"]), len(item["touched"])))
        beam = next_beam[:max(1, int(beam_size))]
        if depth in depths:
            best = beam[0] if beam else None
            rows.append({
                "depth": int(depth),
                "attempted_expansions": int(attempted),
                "valid_count": int(len(next_beam)),
                "invalid_rate": float(invalid / attempted) if attempted else 0.0,
                "best_total_bits_sum": None if best is None else int(best["bits"]),
                "best_fusion_count": None if best is None else int(best["fusion"]),
                "best_mutation_count": 0 if best is None else int(len(best["touched"])),
                "best_action_vector_hash": None if best is None else action_hash(best["action"]),
                "best_touched_slots": [] if best is None else [int(x) for x in best["touched"]],
                "mutation_pool_size": int(len(mutation_pool)),
            })
    return rows


def _masked_random_validity(
        *,
        mask: Mapping[str, Any],
        evaluate_action: Callable[[Sequence[int], str], Mapping[str, Any]],
        random_samples: int,
        random_seed: int,
        mutation_counts: Sequence[int] = (1, 2, 4, 8),
        ) -> Dict[str, Any]:
    rng = random.Random(int(random_seed))
    baseline = [int(x) for x in mask["baseline_action"]]
    mutable_slots = [
        slot for slot in mask["slots"]
        if any(int(idx) != int(slot["baseline_index"]) for idx in slot["allowed_indices"])
    ]
    by_count: Dict[str, Dict[str, Any]] = {}
    per_count = max(1, int(random_samples) // max(1, len(mutation_counts)))
    for mutation_count in mutation_counts:
        valid = 0
        attempted = 0
        invalid_examples = []
        valid_costs: List[Dict[str, Any]] = []
        for _ in range(per_count):
            action = list(baseline)
            chosen = rng.sample(mutable_slots, k=min(int(mutation_count), len(mutable_slots))) if mutable_slots else []
            for slot in chosen:
                non_base = [
                    int(idx) for idx in slot["allowed_indices"]
                    if int(idx) != int(slot["baseline_index"])
                ]
                if non_base:
                    action[int(slot["global_index"])] = rng.choice(non_base)
            evaluated = _normalize_eval(
                evaluate_action(action, f"masked_random_m{mutation_count}"),
                action,
                f"masked_random_m{mutation_count}",
            )
            attempted += 1
            if bool(evaluated["optimizer_valid"]):
                valid += 1
                valid_costs.append({
                    "total_bits_sum": int(evaluated["total_bits_sum"]),
                    "fusion_count": int(evaluated["fusion_count"]),
                    "raw_action_hash": str(evaluated.get("raw_action_hash", evaluated["action_hash"])),
                    "effective_action_hash": str(evaluated.get("effective_action_hash", evaluated["action_hash"])),
                    "action_hash": str(evaluated["action_hash"]),
                })
            elif len(invalid_examples) < 5:
                invalid_examples.append({
                    "action_hash": evaluated["action_hash"],
                    "invalid_summary": evaluated["invalid_summary"],
                })
        bits = [int(row["total_bits_sum"]) for row in valid_costs]
        fusion = [int(row["fusion_count"]) for row in valid_costs]
        best = _smallest_cost_rows(valid_costs, 5)
        by_count[str(int(mutation_count))] = {
            "mutation_count": int(mutation_count),
            "attempted": attempted,
            "valid": valid,
            "valid_rate": float(valid / attempted) if attempted else 0.0,
            "total_bits_min": int(min(bits)) if bits else None,
            "total_bits_mean": float(np.mean(bits)) if bits else None,
            "total_bits_max": int(max(bits)) if bits else None,
            "fusion_count_min": int(min(fusion)) if fusion else None,
            "fusion_count_mean": float(np.mean(fusion)) if fusion else None,
            "fusion_count_max": int(max(fusion)) if fusion else None,
            "best_action_hashes": [str(row["effective_action_hash"]) for row in best],
            "best_raw_action_hashes": [str(row["raw_action_hash"]) for row in best],
            "invalid_examples": invalid_examples,
        }
    return {
        "schema": "blb_masked_random_validity_v1",
        "random_seed": int(random_seed),
        "random_samples_requested": int(random_samples),
        "mutable_slot_count": int(len(mutable_slots)),
        "by_mutation_count": by_count,
    }


def _multi_random_scan(
        *,
        mask: Mapping[str, Any],
        evaluate_action: Callable[[Sequence[int], str], Mapping[str, Any]],
        random_samples: int,
        random_seed: int,
        mutation_counts: Sequence[int],
        ) -> Dict[str, Any]:
    rng = random.Random(int(random_seed))
    baseline = [int(x) for x in mask["baseline_action"]]
    mutable_slots = [
        slot for slot in mask["slots"]
        if any(int(idx) != int(slot["baseline_index"]) for idx in slot["allowed_indices"])
    ]
    rows: List[Dict[str, Any]] = []
    counts = [int(x) for x in mutation_counts if int(x) > 0]
    per_count = max(1, int(random_samples) // max(1, len(counts)))
    for mutation_count in counts:
        for _ in range(per_count):
            action = list(baseline)
            chosen = rng.sample(mutable_slots, k=min(int(mutation_count), len(mutable_slots))) if mutable_slots else []
            for slot in chosen:
                non_base = [
                    int(idx) for idx in slot["allowed_indices"]
                    if int(idx) != int(slot["baseline_index"])
                ]
                if non_base:
                    action[int(slot["global_index"])] = rng.choice(non_base)
            evaluated = _normalize_eval(
                evaluate_action(action, f"multi_random_m{mutation_count}"),
                action,
                f"multi_random_m{mutation_count}",
            )
            if not bool(evaluated["optimizer_valid"]):
                continue
            rows.append({
                "mutation_count": int(len(chosen)),
                "requested_mutation_count": int(mutation_count),
                "total_bits_sum": int(evaluated["total_bits_sum"]),
                "fusion_count": int(evaluated["fusion_count"]),
                "raw_action_hash": str(evaluated.get("raw_action_hash", evaluated["action_hash"])),
                "effective_action_hash": str(evaluated.get("effective_action_hash", evaluated["action_hash"])),
                "action_hash": str(evaluated["action_hash"]),
            })
    best_rows = _smallest_cost_rows(rows, 20)
    return {
        "schema": "blb_multi_random_f0_scan_v1",
        "random_seed": int(random_seed),
        "random_samples_requested": int(random_samples),
        "mutation_counts": counts,
        "mutable_slot_count": int(len(mutable_slots)),
        "valid_count": int(len(rows)),
        "best_valid": best_rows,
    }


def run_scan_core(
        *,
        baseline_action: Sequence[int],
        action_dims: Sequence[int],
        records: Sequence[Mapping[str, Any]],
        evaluate_action: Callable[[Sequence[int], str], Mapping[str, Any]],
        output_dir: str | os.PathLike[str],
        metadata: Mapping[str, Any],
        beam_size: int,
        beam_depths: Sequence[int],
        random_samples: int,
        random_seed: int,
        beam_mutation_limit: int = 64,
        multi_random_samples: int = 0,
        multi_mutation_counts: Sequence[int] = (),
        expected_baseline_bits: int | None = None,
        expected_baseline_fusion: int | None = None,
        ) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline_action = [int(x) for x in baseline_action]
    action_dims = [int(x) for x in action_dims]
    baseline = _normalize_eval(evaluate_action(baseline_action, "all_max_baseline"), baseline_action, "all_max_baseline")
    _write_json(out / "baseline_f0.json", baseline)
    if not bool(baseline["optimizer_valid"]):
        _write_json(out / "error_report.json", {
            "error": "baseline optimizer_valid is false",
            "baseline": baseline,
        })
        raise RuntimeError("baseline optimizer_valid is false; stop F0 scan")
    if expected_baseline_bits is not None and int(baseline["total_bits_sum"]) != int(expected_baseline_bits):
        raise RuntimeError(
            f"baseline total_bits_sum {baseline['total_bits_sum']} != expected {expected_baseline_bits}"
        )
    if expected_baseline_fusion is not None and int(baseline["fusion_count"]) != int(expected_baseline_fusion):
        raise RuntimeError(
            f"baseline fusion_count {baseline['fusion_count']} != expected {expected_baseline_fusion}"
        )

    rows: List[Dict[str, Any]] = []
    for slot_idx, baseline_idx in enumerate(baseline_action):
        record = records[slot_idx] if slot_idx < len(records) else {}
        candidates = list(range(0, int(baseline_idx)))
        for candidate_idx in candidates:
            action = list(baseline_action)
            action[slot_idx] = int(candidate_idx)
            evaluated = _normalize_eval(
                evaluate_action(action, f"single_slot_{slot_idx}_{candidate_idx}"),
                action,
                f"single_slot_{slot_idx}_{candidate_idx}",
            )
            row = {
                "slot_global_index": int(slot_idx),
                "layer": record.get("layer", ""),
                "block": record.get("block", ""),
                "field": record.get("field", ""),
                "kind": record.get("kind", ""),
                "distribution": record.get("distribution", ""),
                "operation": record.get("operation", ""),
                "effective": bool(record.get("effective", True)),
                "N": record.get("N", ""),
                "baseline_index": int(baseline_idx),
                "candidate_index": int(candidate_idx),
                "baseline_value": _record_value(record, int(baseline_idx)),
                "candidate_value": _record_value(record, int(candidate_idx)),
                "optimizer_valid": bool(evaluated["optimizer_valid"]),
                "total_bits_sum": int(evaluated["total_bits_sum"]),
                "fusion_count": int(evaluated["fusion_count"]),
                "delta_total_bits": int(evaluated["total_bits_sum"]) - int(baseline["total_bits_sum"]),
                "delta_fusion_count": int(evaluated["fusion_count"]) - int(baseline["fusion_count"]),
                "invalid_chain": evaluated["invalid_chain"],
                "invalid_summary": evaluated["invalid_summary"],
                "q_bits": evaluated.get("q_bits", []),
                "candidate_key": evaluated["candidate_key"],
                "optimizer_cost_irrelevant": str(record.get("kind", "")) == "K",
            }
            rows.append(row)
    _write_jsonl(out / "per_slot_scan.jsonl", rows)

    summary_rows = _build_per_slot_summary_rows(
        baseline_action=baseline_action,
        records=records,
        rows=rows,
    )
    summary_fields = [
        "slot_global_index", "layer", "block", "field", "kind", "baseline_index",
        "candidate_count", "valid_count", "valid_rate", "improving_valid_count",
        "best_delta_total_bits", "best_delta_fusion_count",
    ]
    _write_csv(out / "per_slot_summary.csv", summary_rows, summary_fields)
    md_lines = [
        "# Phase-1 F0 单槽位扫描摘要",
        "",
        "| slot | block | field | kind | baseline | candidates | valid | improving | best_delta_bits |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['slot_global_index']} | `{row['block']}` | `{row['field']}` | `{row['kind']}` | "
            f"{row['baseline_index']} | {row['candidate_count']} | {row['valid_count']} | "
            f"{row['improving_valid_count']} | {row['best_delta_total_bits']} |"
        )
    (out / "per_slot_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    invalid_counter = Counter(
        (str(row.get("block", "")), str(row.get("kind", "")))
        for row in rows if not bool(row.get("optimizer_valid"))
    )
    invalid_rows = [
        {"block": block, "kind": kind, "invalid_count": count}
        for (block, kind), count in sorted(invalid_counter.items())
    ]
    _write_csv(out / "invalid_by_block_kind.csv", invalid_rows, ["block", "kind", "invalid_count"])

    improving_mutations = [
        row for row in rows
        if bool(row["optimizer_valid"])
        and str(row.get("kind")) != "K"
        and (int(row["delta_total_bits"]) < 0 or int(row["delta_fusion_count"]) < 0)
    ]
    improving_mutations.sort(key=lambda r: (int(r["total_bits_sum"]), int(r["fusion_count"])))
    beam_rows = _beam_scan(
        baseline_action=baseline_action,
        baseline=baseline,
        mutations=improving_mutations,
        evaluate_action=evaluate_action,
        beam_size=int(beam_size),
        beam_depths=beam_depths,
        mutation_limit=int(beam_mutation_limit),
    )
    _write_jsonl(out / "beam_scan_results.jsonl", beam_rows)

    mask = _build_mask(
        baseline_action=baseline_action,
        action_dims=action_dims,
        records=records,
        per_slot_rows=rows,
        source="phase1b_f0_scan",
    )
    mask.update({
        "profile": metadata.get("profile", ""),
        "num_layers": metadata.get("num_layers", ""),
        "registry_hash": metadata.get("registry_hash", ""),
        "max_sfs_hash": metadata.get("max_sfs_hash", ""),
        "stage1_config_content_hash": metadata.get("stage1_config_content_hash", ""),
    })
    mask_hash = _sha256_json(mask)
    mask["mask_hash"] = mask_hash
    _write_json(out / "suggested_action_mask.json", mask)
    mask_md = [
        "# Phase-1 Suggested Action Mask",
        "",
        f"- action_width: `{mask['action_width']}`",
        f"- mask_hash: `{mask_hash}`",
        f"- source: `{mask['source']}`",
        "",
        "| slot | block | field | kind | baseline | allowed | reason |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for slot in mask["slots"]:
        allowed = ",".join(str(x) for x in slot["allowed_indices"])
        mask_md.append(
            f"| {slot['global_index']} | `{slot.get('block', '')}` | `{slot.get('field', '')}` | "
            f"`{slot.get('kind', '')}` | {slot['baseline_index']} | `{allowed}` | `{slot['reason']}` |"
        )
    (out / "suggested_action_mask.md").write_text("\n".join(mask_md) + "\n", encoding="utf-8")

    random_report = _masked_random_validity(
        mask=mask,
        evaluate_action=evaluate_action,
        random_samples=int(random_samples),
        random_seed=int(random_seed),
    )
    _write_json(out / "masked_random_validity.json", random_report)
    multi_random_report = _multi_random_scan(
        mask=mask,
        evaluate_action=evaluate_action,
        random_samples=int(multi_random_samples),
        random_seed=int(random_seed) + 1009,
        mutation_counts=multi_mutation_counts,
    ) if int(multi_random_samples) > 0 and list(multi_mutation_counts) else {
        "schema": "blb_multi_random_f0_scan_v1",
        "random_samples_requested": int(multi_random_samples),
        "mutation_counts": [int(x) for x in multi_mutation_counts],
        "valid_count": 0,
        "best_valid": [],
    }
    _write_json(out / "multi_random_summary.json", multi_random_report)
    _write_jsonl(out / "multi_random_best_valid.jsonl", multi_random_report.get("best_valid", []))

    manifest = {
        "schema": "blb_phase1_f0_scan_manifest_v1",
        "metadata": dict(metadata),
        "baseline_action_hash": action_hash(baseline_action),
        "baseline": {
            "optimizer_valid": bool(baseline["optimizer_valid"]),
            "total_bits_sum": int(baseline["total_bits_sum"]),
            "fusion_count": int(baseline["fusion_count"]),
            "avg_k": float(baseline["avg_k"]),
        },
        "beam_size": int(beam_size),
        "beam_depths": [int(x) for x in beam_depths],
        "beam_mutation_limit": int(beam_mutation_limit),
        "random_samples": int(random_samples),
        "random_seed": int(random_seed),
        "multi_random_samples": int(multi_random_samples),
        "multi_mutation_counts": [int(x) for x in multi_mutation_counts],
        "mask_hash": mask_hash,
    }
    _write_json(out / "manifest.json", manifest)
    return {
        "baseline": baseline,
        "per_slot_rows": rows,
        "summary_rows": summary_rows,
        "beam_rows": beam_rows,
        "mask": mask,
        "random_report": random_report,
        "multi_random_report": multi_random_report,
        "manifest": manifest,
    }


def _real_evaluator(
        *,
        profile: str,
        num_layers: int,
        rescale_optimizer_root: str,
        gelu_degree: Sequence[int],
        attn_degree: Sequence[int],
        identity_context: Mapping[str, Any],
        ) -> Callable[[Sequence[int], str], Mapping[str, Any]]:
    from blb_stage2_rl.action_space import (
        avg_truncation_k_in_action,
        describe_action_vector,
        load_max_sfs,
        make_all_max_action_vector,
    )
    from blb_stage2_rl.optimizer_cost import evaluate_action_for_cost
    from rescale_optimizer_bridge import InProcessInvoker, RescaleOptimizerBridge
    from scripts.blb_eval_action import (
        _optimizer_debug_from_outputs,
        _optimizer_outputs_summary,
        build_f0_candidate_record,
    )

    max_sfs = load_max_sfs(profile)
    bridge = RescaleOptimizerBridge(
        invoker=InProcessInvoker.from_profile(
            rescale_optimizer_root=rescale_optimizer_root,
            profile=profile,
        )
    )
    baseline = make_all_max_action_vector(num_layers=num_layers)
    baseline_desc = describe_action_vector(
        baseline,
        max_sfs=max_sfs,
        num_layers=num_layers,
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
        profile=profile,
    )

    def evaluate(action: Sequence[int], source: str) -> Mapping[str, Any]:
        action_arr = np.asarray(action, dtype=int)
        cost_eval = evaluate_action_for_cost(
            action_arr,
            profile=profile,
            num_layers=num_layers,
            max_sfs=max_sfs,
            rescale_bridge=bridge,
            gelu_degree=gelu_degree,
            attn_degree=attn_degree,
        )
        outputs = cost_eval.outputs
        signals = cost_eval.signals
        eff_action = effective_action_vector(action_arr, baseline_desc, baseline)
        record = build_f0_candidate_record(
            action_arr,
            source=source,
            signals=signals,
            baseline_total_bits=None,
            identity_context=identity_context,
            effective_action_indices=eff_action,
            optimizer_debug=_optimizer_debug_from_outputs(outputs),
            action_avg_k=avg_truncation_k_in_action(action_arr, num_layers),
        )
        record["optimizer_outputs"] = _optimizer_outputs_summary(outputs)
        return {
            "optimizer_valid": record["optimizer_valid"],
            "total_bits_sum": record["optimizer"]["total_bits_sum"],
            "fusion_count": record["optimizer"]["total_fusion_count"],
            "avg_k": record["mpc_truncation_term"]["avg_k"],
            "invalid_chain": record["optimizer"]["invalid_chains"],
            "q_bits": record["rescale_debug"]["optimizer_diagnostic_terms"]["q_bits"],
            "q_head_bits": record["rescale_debug"]["optimizer_diagnostic_terms"]["q_head_bits"],
            "q_tail_bits": record["rescale_debug"]["optimizer_diagnostic_terms"]["q_tail_bits"],
            "raw_action_hash": record["raw_action_hash"],
            "effective_action_hash": record["effective_action_hash"],
            "effective_action_indices": record["effective_action_indices"],
            "candidate_key_basis": record["candidate_key_basis"],
            "candidate_key": record.get("candidate_key") or candidate_key(
                action_arr,
                identity_context,
                effective_action_indices=eff_action,
            ),
        }

    return evaluate


def main(argv: Sequence[str] | None = None) -> int:
    from blb_stage2_rl.action_space import (
        action_dims_for_config,
        describe_action_vector,
        load_max_sfs,
        make_all_max_action_vector,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--model", default="bert-base")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--stage1-config", default="glue_final_configs_best_ppo.json")
    parser.add_argument("--fixed-gelu", default="")
    parser.add_argument("--fixed-softmax", default="")
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    parser.add_argument("--output-dir", default="reports/blb_opt/phase1b_f0_scan")
    parser.add_argument("--beam-size", type=int, default=32)
    parser.add_argument("--beam-depths", default="1,2,4,8")
    parser.add_argument("--beam-mutation-limit", type=int, default=64)
    parser.add_argument("--random-samples", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--multi-random-samples", type=int, default=500)
    parser.add_argument("--multi-mutation-counts", default="2,4,8,16,32")
    parser.add_argument("--registry-hash", default="")
    parser.add_argument("--max-sfs-hash", default="")
    parser.add_argument("--threshold-policy-hash", default="smoke_temporary")
    parser.add_argument("--expected-baseline-bits", type=int, default=14889)
    parser.add_argument("--expected-baseline-fusion", type=int, default=0)
    args = parser.parse_args(argv)

    stage1 = _load_stage1_vectors(args.stage1_config, model=args.model, profile=args.profile)
    fixed_gelu = _parse_int_list(args.fixed_gelu)
    fixed_softmax = _parse_int_list(args.fixed_softmax)
    gelu = fixed_gelu or stage1.get("gelu") or [4] * int(args.num_layers)
    softmax = fixed_softmax or stage1.get("softmax") or [4] * int(args.num_layers)
    if len(gelu) != int(args.num_layers) or len(softmax) != int(args.num_layers):
        raise ValueError("fixed/stage1 degree vectors must match --num-layers")
    rescale_hash = canonical_rescale_optimizer_hash(args.rescale_optimizer_root, args.profile)
    identity_context = build_candidate_identity_context(
        action_space_version="current-code-v1",
        registry_hash=args.registry_hash or "unknown",
        max_sfs_hash=args.max_sfs_hash or "unknown",
        stage1_config_content_hash=stage1.get("content_hash") or "unknown",
        stage1_gelu_degrees=gelu,
        stage1_softmax_degrees=softmax,
        profile=args.profile,
        dataset=args.profile,
        model=args.model,
        rescale_optimizer_mode="in_process_real",
        rescale_optimizer_root=args.rescale_optimizer_root,
        rescale_optimizer_canonical_hash=rescale_hash,
        decode_version="action_space_v1",
        metric_policy_version="mrpc-acc-f1-std-v1",
        threshold_policy_hash=args.threshold_policy_hash or "unknown",
        fidelity="F0_optimizer_only",
    )
    baseline_action = make_all_max_action_vector(args.num_layers)
    desc = describe_action_vector(
        baseline_action,
        max_sfs=load_max_sfs(args.profile),
        num_layers=args.num_layers,
        gelu_degree=gelu,
        attn_degree=softmax,
        profile=args.profile,
    )
    metadata = {
        "profile": args.profile,
        "num_layers": int(args.num_layers),
        "stage1_config_path": stage1.get("path", ""),
        "stage1_config_content_hash": stage1.get("content_hash", ""),
        "stage1_gelu_degrees": gelu,
        "stage1_softmax_degrees": softmax,
        "stage1_fixed_vectors_match_config": (
            (fixed_gelu is None or fixed_gelu == stage1.get("gelu"))
            and (fixed_softmax is None or fixed_softmax == stage1.get("softmax"))
        ),
        "rescale_optimizer_mode": "in_process_real",
        "rescale_optimizer_root": args.rescale_optimizer_root,
        "rescale_optimizer_canonical_hash": rescale_hash,
        "registry_hash": args.registry_hash,
        "max_sfs_hash": args.max_sfs_hash,
    }
    result = run_scan_core(
        baseline_action=baseline_action.tolist(),
        action_dims=action_dims_for_config(args.num_layers),
        records=list(desc["records"]),
        evaluate_action=_real_evaluator(
            profile=args.profile,
            num_layers=args.num_layers,
            rescale_optimizer_root=args.rescale_optimizer_root,
            gelu_degree=gelu,
            attn_degree=softmax,
            identity_context=identity_context,
        ),
        output_dir=args.output_dir,
        metadata=metadata,
        beam_size=args.beam_size,
        beam_depths=[int(x) for x in str(args.beam_depths).split(",") if x.strip()],
        beam_mutation_limit=args.beam_mutation_limit,
        random_samples=args.random_samples,
        random_seed=args.random_seed,
        multi_random_samples=args.multi_random_samples,
        multi_mutation_counts=[
            int(x) for x in str(args.multi_mutation_counts).split(",") if str(x).strip()
        ],
        expected_baseline_bits=args.expected_baseline_bits,
        expected_baseline_fusion=args.expected_baseline_fusion,
    )
    print(json.dumps({
        "output_dir": args.output_dir,
        "baseline": result["manifest"]["baseline"],
        "mask_hash": result["manifest"]["mask_hash"],
        "masked_random_validity": result["random_report"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
