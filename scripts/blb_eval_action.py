"""Evaluate or record a BLB action candidate at playbook fidelity F0."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blb_stage2_rl.action_space import (  # noqa: E402
    action_vector_to_cfgs,
    avg_truncation_k_in_action,
    build_optimizer_requests,
    load_max_sfs,
    make_all_max_action_vector,
)
from blb_stage2_rl.candidate_store import (  # noqa: E402
    CandidateStore,
    action_hash,
    build_candidate_identity_context,
    candidate_key,
    candidate_rank_key,
    f0_sort_key,
    normalize_action_indices,
)
from blb_stage2_rl.optimizer_cost import evaluate_action_for_cost  # noqa: E402
from json_utils import read_json_file, write_json_file  # noqa: E402
from rescale_optimizer_bridge import (  # noqa: E402
    InProcessInvoker,
    RescaleOptimizerBridge,
    aggregate_optimizer_signals,
)


def _metric_from_signals(signals: Any, name: str, default: Any = None) -> Any:
    if isinstance(signals, Mapping):
        return signals.get(name, default)
    return getattr(signals, name, default)


def build_f0_candidate_record(
        action_indices: Any,
        *,
        source: str,
        signals: Any,
        baseline_total_bits: float | None = None,
        parent_hash: str | None = None,
        identity_context: Mapping[str, Any] | None = None,
        effective_action_indices: Any | None = None,
        optimizer_debug: Mapping[str, Any] | None = None,
        action_avg_k: float | None = None,
        ) -> dict:
    action = normalize_action_indices(action_indices)
    effective_action = (
        normalize_action_indices(effective_action_indices)
        if effective_action_indices is not None else list(action)
    )
    total_bits = float(_metric_from_signals(signals, "total_bits_sum", 0.0) or 0.0)
    if baseline_total_bits and float(baseline_total_bits) > 0.0:
        normalized_cost = total_bits / float(baseline_total_bits)
    else:
        normalized_cost = total_bits
    invalid_chains = _metric_from_signals(signals, "invalid_chains", {}) or {}
    optimizer_valid = not bool(_metric_from_signals(signals, "any_invalid", False))
    record = {
        "action_hash": action_hash(action),
        "action_vector_hash": action_hash(action),
        "action_indices": action,
        "raw_action_indices": list(action),
        "raw_action_hash": action_hash(action),
        "effective_action_indices": effective_action,
        "effective_action_hash": action_hash(effective_action),
        "candidate_key_basis": "effective_action_hash + identity_context",
        "source": str(source),
        "parent_hash": parent_hash,
        "fidelity": "F0",
        "valid": bool(optimizer_valid),
        "optimizer_valid": bool(optimizer_valid),
        "invalid_summary": json.dumps(invalid_chains, ensure_ascii=True, sort_keys=True),
        "acc_violation": 0.0,
        "stability_violation": 0.0,
        "normalized_cost": float(normalized_cost),
        "rescale_cost": {
            "optimizer_cost_terms": {
                "total_bits_sum": int(total_bits),
                "fusion_count": int(_metric_from_signals(signals, "total_fusion_count", 0) or 0),
            },
            "rank_key": [
                int(total_bits),
                int(_metric_from_signals(signals, "total_fusion_count", 0) or 0),
            ],
        },
        "rescale_debug": {
            "optimizer_validity_terms": {
                "invalid_chain": invalid_chains,
                "optimizer_valid": not bool(_metric_from_signals(signals, "any_invalid", False)),
                "any_invalid": bool(_metric_from_signals(signals, "any_invalid", False)),
            },
            "optimizer_diagnostic_terms": {
                "q_bits": list((optimizer_debug or {}).get("q_bits", [])),
                "q_head_bits": (optimizer_debug or {}).get("q_head_bits"),
                "q_tail_bits": (optimizer_debug or {}).get("q_tail_bits"),
            },
        },
        "mpc_truncation_cost_enabled": action_avg_k is not None,
        "mpc_truncation_term": {
            "avg_k": None if action_avg_k is None else float(action_avg_k),
        },
        "optimizer": {
            "total_bits_sum": int(total_bits),
            "total_fusion_count": int(_metric_from_signals(signals, "total_fusion_count", 0) or 0),
            "invalid_chains": invalid_chains,
        },
    }
    if identity_context is not None:
        record["identity_context"] = dict(identity_context)
        record["candidate_key"] = candidate_key(
            action,
            identity_context,
            effective_action_indices=effective_action,
        )
    record["rank_key"] = list(f0_sort_key(record))
    return record


def _load_action(path: str | None, *, num_layers: int) -> np.ndarray:
    if not path:
        return make_all_max_action_vector(num_layers=num_layers)
    payload = read_json_file(path)
    if isinstance(payload, Mapping):
        payload = payload.get("action_indices", payload.get("action", payload))
    return np.asarray(normalize_action_indices(payload), dtype=int)


def _parse_degree_arg(value: str | None, *, num_layers: int, name: str) -> int | list[int] | str:
    if value is None or str(value).strip() == "":
        return "unknown"
    payload = json.loads(str(value))
    if isinstance(payload, int):
        return int(payload)
    if isinstance(payload, list):
        out = [int(x) for x in payload]
        if len(out) != int(num_layers):
            raise ValueError(
                f"{name} degree vector length {len(out)} does not match num_layers={int(num_layers)}"
            )
        return out
    raise ValueError(f"{name} degree must be an int or a JSON int list")


def _optimizer_outputs_summary(outputs: Mapping[str, Any]) -> dict:
    summary = {}
    for name, out in sorted(outputs.items()):
        summary[name] = {
            "valid": bool(getattr(out, "valid", False)),
            "total_bits": int(getattr(out, "total_bits", 0) or 0),
            "fusion_count": int(getattr(out, "fusion_count", 0) or 0),
            "invalid_chain": getattr(out, "invalid_chain", None),
            "q_bits": _raw_chain_field(out, "q_bits"),
            "q_head_bits": _raw_chain_field(out, "q_head_bits"),
            "q_tail_bits": _raw_chain_field(out, "q_tail_bits"),
        }
    return summary


def _raw_chain_field(out: Any, name: str) -> Any:
    raw = getattr(out, "raw", {}) or {}
    result = raw.get("result") if isinstance(raw, Mapping) else {}
    chain = result.get("chain") if isinstance(result, Mapping) else {}
    if isinstance(chain, Mapping) and name in chain:
        return chain.get(name)
    return raw.get(name) if isinstance(raw, Mapping) else None


def _optimizer_debug_from_outputs(outputs: Mapping[str, Any]) -> dict:
    q_bits = []
    q_head_bits = []
    q_tail_bits = []
    for out in outputs.values():
        qb = _raw_chain_field(out, "q_bits")
        if qb is not None:
            q_bits.append(qb)
        qh = _raw_chain_field(out, "q_head_bits")
        if qh is not None:
            q_head_bits.append(qh)
        qt = _raw_chain_field(out, "q_tail_bits")
        if qt is not None:
            q_tail_bits.append(qt)
    return {
        "q_bits": q_bits,
        "q_head_bits": q_head_bits,
        "q_tail_bits": q_tail_bits,
    }


def run_f0_eval(
        *,
        profile: str,
        num_layers: int,
        action_json: str | None,
        output_dir: os.PathLike[str] | str,
        source: str,
        rescale_optimizer_root: str,
        baseline_total_bits: float | None,
        identity_context: Mapping[str, Any] | None = None,
        gelu_degree: int | list[int] | str = 4,
        attn_degree: int | list[int] | str = 4,
        ) -> dict:
    action = _load_action(action_json, num_layers=num_layers)
    bridge = RescaleOptimizerBridge(
        invoker=InProcessInvoker.from_profile(
            rescale_optimizer_root=rescale_optimizer_root,
            profile=profile,
        )
    )
    cost_eval = evaluate_action_for_cost(
        action,
        profile=profile,
        num_layers=num_layers,
        max_sfs=load_max_sfs(profile),
        rescale_bridge=bridge,
        gelu_degree=gelu_degree,
        attn_degree=attn_degree,
    )
    outputs = cost_eval.outputs
    signals = cost_eval.signals
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json_file(out / "optimizer_outputs.json", _optimizer_outputs_summary(outputs))
    record = build_f0_candidate_record(
        action,
        source=source,
        signals=signals,
        baseline_total_bits=baseline_total_bits,
        identity_context=identity_context,
        optimizer_debug=_optimizer_debug_from_outputs(outputs),
        action_avg_k=avg_truncation_k_in_action(action, num_layers),
    )
    record_path = CandidateStore(out / "candidates" / "candidate_store.jsonl").append(record)
    write_json_file(
        out / "rank_key.json",
        {
            "action_hash": record_path["action_hash"],
            "candidate_key": record_path.get("candidate_key"),
            "rank_key": record_path["rank_key"],
            "rescale_cost_rank_key": record_path["rescale_cost"]["rank_key"],
        },
    )
    return record_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--action-json", default="")
    parser.add_argument("--output-dir", default="reports/blb_opt/phase2_eval")
    parser.add_argument("--source", default="manual_f0")
    parser.add_argument("--rescale-optimizer-root", default="Rescale_optimizer")
    parser.add_argument("--baseline-total-bits", type=float, default=None)
    parser.add_argument("--registry-hash", default="")
    parser.add_argument("--max-sfs-hash", default="")
    parser.add_argument("--stage1-hash", default="")
    parser.add_argument("--stage1-config-content-hash", default="")
    parser.add_argument("--rescale-optimizer-hash", default="")
    parser.add_argument("--rescale-optimizer-canonical-hash", default="")
    parser.add_argument("--rescale-optimizer-mode", default="in_process_real")
    parser.add_argument("--decode-version", default="action_space_v1")
    parser.add_argument("--dataset", default="mrpc")
    parser.add_argument("--model", default="bert-base")
    parser.add_argument("--metric-policy-version", default="mrpc-acc-f1-std-v1")
    parser.add_argument("--threshold-policy-hash", default="")
    parser.add_argument("--stage1-gelu-degrees", default="")
    parser.add_argument("--stage1-softmax-degrees", default="")
    parser.add_argument("--fidelity", default="F0_optimizer_only")
    args = parser.parse_args(argv)
    stage1_gelu_degrees = _parse_degree_arg(
        args.stage1_gelu_degrees, num_layers=args.num_layers, name="stage1_gelu_degrees",
    )
    stage1_softmax_degrees = _parse_degree_arg(
        args.stage1_softmax_degrees, num_layers=args.num_layers, name="stage1_softmax_degrees",
    )
    identity_context = None
    stage1_config_content_hash = args.stage1_config_content_hash or args.stage1_hash
    rescale_canonical_hash = args.rescale_optimizer_canonical_hash or args.rescale_optimizer_hash
    if args.registry_hash or args.max_sfs_hash or stage1_config_content_hash:
        identity_context = build_candidate_identity_context(
            action_space_version="current-code-v1",
            registry_hash=args.registry_hash or "unknown",
            max_sfs_hash=args.max_sfs_hash or "unknown",
            stage1_hash=args.stage1_hash or "unknown",
            stage1_config_content_hash=stage1_config_content_hash or "unknown",
            stage1_degrees={"gelu": stage1_gelu_degrees, "softmax": stage1_softmax_degrees},
            stage1_gelu_degrees=stage1_gelu_degrees,
            stage1_softmax_degrees=stage1_softmax_degrees,
            profile=args.profile,
            rescale_optimizer_mode=args.rescale_optimizer_mode,
            rescale_optimizer_root=args.rescale_optimizer_root,
            rescale_optimizer_hash=args.rescale_optimizer_hash or "unknown",
            rescale_optimizer_canonical_hash=rescale_canonical_hash or "unknown",
            decode_version=args.decode_version,
            dataset=args.dataset,
            model=args.model,
            metric_policy_version=args.metric_policy_version,
            threshold_policy_hash=args.threshold_policy_hash or "unknown",
            fidelity=args.fidelity,
        )
    record = run_f0_eval(
        profile=args.profile,
        num_layers=args.num_layers,
        action_json=args.action_json or None,
        output_dir=args.output_dir,
        source=args.source,
        rescale_optimizer_root=args.rescale_optimizer_root,
        baseline_total_bits=args.baseline_total_bits,
        identity_context=identity_context,
        gelu_degree=stage1_gelu_degrees if stage1_gelu_degrees != "unknown" else 4,
        attn_degree=stage1_softmax_degrees if stage1_softmax_degrees != "unknown" else 4,
    )
    json.dump(record, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
