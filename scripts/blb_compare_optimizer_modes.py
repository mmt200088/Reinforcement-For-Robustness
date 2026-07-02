"""Compare optimizer baseline and canonical cfg-derived action cost paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blb_stage2_rl.action_space import (  # noqa: E402
    action_dims_for_config,
    build_optimizer_requests,
    describe_action_vector,
    load_max_sfs,
    make_all_max_action_vector,
)
from blb_stage2_rl.candidate_store import (  # noqa: E402
    build_candidate_identity_context,
    effective_action_hash,
    effective_action_vector,
    raw_action_hash,
)
from blb_stage2_rl.optimizer_cost import evaluate_action_for_cost  # noqa: E402
from json_utils import write_json_file  # noqa: E402
from rescale_optimizer_bridge import (  # noqa: E402
    InProcessInvoker,
    RescaleOptimizerBridge,
    aggregate_optimizer_signals,
)
from scripts.blb_eval_action import _optimizer_outputs_summary  # noqa: E402
from scripts.blb_f0_scan_feasible_domain import (  # noqa: E402
    _load_stage1_vectors,
    _parse_int_list,
    canonical_rescale_optimizer_hash,
)


def _mutate_record(action: np.ndarray, record: Mapping[str, Any]) -> np.ndarray:
    out = np.asarray(action, dtype=int).copy()
    idx = int(record["global_index"])
    width = int(record.get("num_levels", 1) or 1)
    if width <= 1:
        return out
    out[idx] = 0 if int(out[idx]) != 0 else min(1, width - 1)
    return out


def _signals_dict(signals: Any) -> Dict[str, Any]:
    return {
        "optimizer_valid": not bool(getattr(signals, "any_invalid", False)),
        "total_bits_sum": int(getattr(signals, "total_bits_sum", 0) or 0),
        "fusion_count": int(getattr(signals, "total_fusion_count", 0) or 0),
        "valid_block_count": int(getattr(signals, "valid_block_count", 0) or 0),
        "invalid_block_count": int(getattr(signals, "invalid_block_count", 0) or 0),
        "invalid_chains": getattr(signals, "invalid_chains", {}) or {},
    }


def _t_new_sources(outputs: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, value in sorted(outputs.items()):
        raw = getattr(value, "raw", {}) or {}
        out[name] = str(raw.get("_t_new_source", ""))
    return out


def _request_flags(requests: Mapping[str, Any]) -> Dict[str, Any]:
    keys = sorted(str(k) for k in requests.keys())
    return {
        "request_count": int(len(keys)),
        "request_keys": keys,
        "sends_block1_mrpc_L0": "block1_mrpc_L0" in keys,
        "sends_first_input": any("first_input" in key for key in keys),
    }


def _evaluate_case(
        *,
        label: str,
        action: Sequence[int],
        eval_mode: str,
        profile: str,
        num_layers: int,
        max_sfs: Any,
        bridge: RescaleOptimizerBridge,
        gelu_degree: Sequence[int],
        attn_degree: Sequence[int],
        baseline_action: np.ndarray,
        baseline_desc: Mapping[str, Any],
        identity_context: Mapping[str, Any],
        ) -> Dict[str, Any]:
    action_arr = np.asarray(action, dtype=int).reshape(-1)
    cost_eval = evaluate_action_for_cost(
        action_arr,
        profile=profile,
        num_layers=num_layers,
        max_sfs=max_sfs,
        rescale_bridge=bridge,
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
    )
    requests = cost_eval.requests
    if eval_mode == "evaluate_baseline_blocks":
        outputs = bridge.evaluate_baseline_blocks(requests)
        signals = aggregate_optimizer_signals(outputs)
        optimizer_eval_mode = "evaluate_baseline_blocks"
    else:
        outputs = cost_eval.outputs
        signals = cost_eval.signals
        optimizer_eval_mode = "evaluate_blocks"
    effective_vec = effective_action_vector(action_arr, baseline_desc, baseline_action)
    return {
        "label": str(label),
        "raw_action_hash": raw_action_hash(action_arr),
        "effective_action_hash": effective_action_hash(action_arr, baseline_desc, baseline_action),
        "candidate_key_basis": "effective_action_hash + identity_context",
        "identity_context": dict(identity_context),
        "raw_action_indices": [int(x) for x in action_arr.tolist()],
        "effective_action_indices": [int(x) for x in effective_vec],
        "optimizer_eval_mode": optimizer_eval_mode,
        "t_new_source_per_config": _t_new_sources(outputs),
        **_request_flags(requests),
        "aggregate": _signals_dict(signals),
        "per_config": _optimizer_outputs_summary(outputs),
    }


def _same_cost(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    la = left["aggregate"]
    ra = right["aggregate"]
    return (
        bool(la["optimizer_valid"]) == bool(ra["optimizer_valid"])
        and int(la["total_bits_sum"]) == int(ra["total_bits_sum"])
        and int(la["fusion_count"]) == int(ra["fusion_count"])
        and left.get("per_config") == right.get("per_config")
    )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    cases = {case["label"]: case for case in payload["cases"]}
    invariants = payload["invariants"]
    lines = [
        "# Phase-1B Optimizer Mode Consistency",
        "",
        f"- profile: `{payload['profile']}`",
        f"- num_layers: `{payload['num_layers']}`",
        f"- rescale_optimizer_mode: `{payload['rescale_optimizer_mode']}`",
        f"- rescale_optimizer_hash: `{payload['rescale_optimizer_canonical_hash']}`",
        f"- stage1_config_hash: `{payload['stage1_config_content_hash']}`",
        "",
        "## Invariants",
        "",
        "| invariant | status |",
        "|---|---|",
    ]
    for name, ok in invariants.items():
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} |")
    lines.extend(["", "## Cases", "", "| case | mode | valid | bits | fusion | requests | raw_hash | effective_hash |", "|---|---|---:|---:|---:|---:|---|---|"])
    for label in payload["case_order"]:
        case = cases[label]
        agg = case["aggregate"]
        lines.append(
            f"| `{label}` | `{case['optimizer_eval_mode']}` | {str(agg['optimizer_valid']).lower()} | "
            f"{agg['total_bits_sum']} | {agg['fusion_count']} | {case['request_count']} | "
            f"`{case['raw_action_hash'][:12]}` | `{case['effective_action_hash'][:12]}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compare(argv: Sequence[str] | None = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--model", default="bert-base")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--stage1-config", default="glue_final_configs_best_ppo.json")
    parser.add_argument("--fixed-gelu", default="")
    parser.add_argument("--fixed-softmax", default="")
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    parser.add_argument("--output-dir", default="reports/blb_opt/phase1b_consistency")
    parser.add_argument("--registry-hash", default="")
    parser.add_argument("--max-sfs-hash", default="")
    args = parser.parse_args(argv)

    stage1 = _load_stage1_vectors(args.stage1_config, model=args.model, profile=args.profile)
    gelu = _parse_int_list(args.fixed_gelu) or stage1.get("gelu") or [4] * int(args.num_layers)
    attn = _parse_int_list(args.fixed_softmax) or stage1.get("softmax") or [4] * int(args.num_layers)
    gelu = [int(x) for x in gelu]
    attn = [int(x) for x in attn]
    max_sfs = load_max_sfs(args.profile)
    baseline = make_all_max_action_vector(args.num_layers)
    baseline_desc = describe_action_vector(
        baseline,
        max_sfs=max_sfs,
        num_layers=args.num_layers,
        gelu_degree=gelu,
        attn_degree=attn,
        profile=args.profile,
    )
    records = list(baseline_desc["records"])
    inactive_l0b1 = next(
        r for r in records
        if int(r.get("layer", -1)) == 0 and r.get("block") == "block1" and int(r.get("num_levels", 1)) > 1
    )
    inactive_first = next(r for r in records if r.get("block") == "first_input")
    effective_slot = next(
        r for r in records
        if bool(r.get("effective", True)) and int(r.get("num_levels", 1)) > 1 and r.get("kind") != "K"
    )

    rescale_hash = canonical_rescale_optimizer_hash(args.rescale_optimizer_root, args.profile)
    identity_context = build_candidate_identity_context(
        action_space_version="current-code-v1",
        registry_hash=args.registry_hash or "unknown",
        max_sfs_hash=args.max_sfs_hash or "unknown",
        stage1_config_content_hash=stage1.get("content_hash") or "unknown",
        stage1_gelu_degrees=gelu,
        stage1_softmax_degrees=attn,
        profile=args.profile,
        dataset=args.profile,
        model=args.model,
        rescale_optimizer_mode="in_process_real",
        rescale_optimizer_root=args.rescale_optimizer_root,
        rescale_optimizer_canonical_hash=rescale_hash,
        decode_version="action_space_v1",
        metric_policy_version="mrpc-acc-f1-std-v1",
        threshold_policy_hash="phase1b_diagnostic",
        fidelity="F0_optimizer_only",
    )
    bridge = RescaleOptimizerBridge(
        invoker=InProcessInvoker.from_profile(
            rescale_optimizer_root=args.rescale_optimizer_root,
            profile=args.profile,
        )
    )
    cases = [
        _evaluate_case(
            label="all_max_raw",
            action=baseline,
            eval_mode="evaluate_baseline_blocks",
            profile=args.profile,
            num_layers=args.num_layers,
            max_sfs=max_sfs,
            bridge=bridge,
            gelu_degree=gelu,
            attn_degree=attn,
            baseline_action=baseline,
            baseline_desc=baseline_desc,
            identity_context=identity_context,
        ),
        _evaluate_case(
            label="all_max_via_candidate_path",
            action=baseline,
            eval_mode="evaluate_blocks",
            profile=args.profile,
            num_layers=args.num_layers,
            max_sfs=max_sfs,
            bridge=bridge,
            gelu_degree=gelu,
            attn_degree=attn,
            baseline_action=baseline,
            baseline_desc=baseline_desc,
            identity_context=identity_context,
        ),
        _evaluate_case(
            label="inactive_l0b1_mutation",
            action=_mutate_record(baseline, inactive_l0b1),
            eval_mode="evaluate_blocks",
            profile=args.profile,
            num_layers=args.num_layers,
            max_sfs=max_sfs,
            bridge=bridge,
            gelu_degree=gelu,
            attn_degree=attn,
            baseline_action=baseline,
            baseline_desc=baseline_desc,
            identity_context=identity_context,
        ),
        _evaluate_case(
            label="inactive_first_input_mutation",
            action=_mutate_record(baseline, inactive_first),
            eval_mode="evaluate_blocks",
            profile=args.profile,
            num_layers=args.num_layers,
            max_sfs=max_sfs,
            bridge=bridge,
            gelu_degree=gelu,
            attn_degree=attn,
            baseline_action=baseline,
            baseline_desc=baseline_desc,
            identity_context=identity_context,
        ),
        _evaluate_case(
            label="effective_single_mutation",
            action=_mutate_record(baseline, effective_slot),
            eval_mode="evaluate_blocks",
            profile=args.profile,
            num_layers=args.num_layers,
            max_sfs=max_sfs,
            bridge=bridge,
            gelu_degree=gelu,
            attn_degree=attn,
            baseline_action=baseline,
            baseline_desc=baseline_desc,
            identity_context=identity_context,
        ),
    ]
    by_label = {case["label"]: case for case in cases}
    ref = by_label["all_max_via_candidate_path"]
    invariants = {
        "all_max_baseline_path_equals_candidate_path": _same_cost(by_label["all_max_raw"], ref),
        "inactive_l0b1_mutation_equals_all_max_candidate_path": _same_cost(by_label["inactive_l0b1_mutation"], ref),
        "inactive_first_input_mutation_equals_all_max_candidate_path": _same_cost(by_label["inactive_first_input_mutation"], ref),
        "effective_single_mutation_may_differ_from_all_max_candidate_path": True,
    }
    payload = {
        "schema": "blb_phase1b_optimizer_mode_comparison_v1",
        "profile": args.profile,
        "model": args.model,
        "num_layers": int(args.num_layers),
        "stage1_config_path": stage1.get("path", ""),
        "stage1_config_content_hash": stage1.get("content_hash", ""),
        "stage1_gelu_degrees": gelu,
        "stage1_softmax_degrees": attn,
        "rescale_optimizer_mode": "in_process_real",
        "rescale_optimizer_root": args.rescale_optimizer_root,
        "rescale_optimizer_canonical_hash": rescale_hash,
        "registry_hash": args.registry_hash,
        "max_sfs_hash": args.max_sfs_hash,
        "action_width": int(len(action_dims_for_config(args.num_layers))),
        "case_order": [case["label"] for case in cases],
        "cases": cases,
        "invariants": invariants,
        "all_invariants_pass": all(bool(v) for v in invariants.values()),
    }
    out = Path(args.output_dir)
    write_json_file(out / "optimizer_mode_comparison.json", payload)
    _write_markdown(out / "optimizer_mode_comparison.md", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    payload = run_compare(argv)
    print(json.dumps({
        "output_dir": "reports/blb_opt/phase1b_consistency",
        "all_invariants_pass": bool(payload["all_invariants_pass"]),
        "invariants": payload["invariants"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["all_invariants_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
