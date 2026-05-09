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
    build_optimizer_requests,
    load_max_sfs,
    make_all_max_action_vector,
)
from blb_stage2_rl.candidate_store import (  # noqa: E402
    CandidateStore,
    action_hash,
    candidate_rank_key,
    normalize_action_indices,
)
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
        ) -> dict:
    action = normalize_action_indices(action_indices)
    total_bits = float(_metric_from_signals(signals, "total_bits_sum", 0.0) or 0.0)
    if baseline_total_bits and float(baseline_total_bits) > 0.0:
        normalized_cost = total_bits / float(baseline_total_bits)
    else:
        normalized_cost = total_bits
    invalid_chains = _metric_from_signals(signals, "invalid_chains", {}) or {}
    record = {
        "action_hash": action_hash(action),
        "action_indices": action,
        "source": str(source),
        "parent_hash": parent_hash,
        "fidelity": "F0",
        "valid": not bool(_metric_from_signals(signals, "any_invalid", False)),
        "invalid_summary": json.dumps(invalid_chains, ensure_ascii=True, sort_keys=True),
        "acc_violation": 0.0,
        "stability_violation": 0.0,
        "normalized_cost": float(normalized_cost),
        "optimizer": {
            "total_bits_sum": int(total_bits),
            "total_fusion_count": int(_metric_from_signals(signals, "total_fusion_count", 0) or 0),
            "invalid_chains": invalid_chains,
        },
    }
    record["rank_key"] = list(candidate_rank_key(record))
    return record


def _load_action(path: str | None, *, num_layers: int) -> np.ndarray:
    if not path:
        return make_all_max_action_vector(num_layers=num_layers)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("action_indices", payload.get("action", payload))
    return np.asarray(normalize_action_indices(payload), dtype=int)


def _optimizer_outputs_summary(outputs: Mapping[str, Any]) -> dict:
    summary = {}
    for name, out in sorted(outputs.items()):
        summary[name] = {
            "valid": bool(getattr(out, "valid", False)),
            "total_bits": int(getattr(out, "total_bits", 0) or 0),
            "fusion_count": int(getattr(out, "fusion_count", 0) or 0),
            "invalid_chain": getattr(out, "invalid_chain", None),
        }
    return summary


def run_f0_eval(
        *,
        profile: str,
        num_layers: int,
        action_json: str | None,
        output_dir: os.PathLike[str] | str,
        source: str,
        rescale_optimizer_root: str,
        baseline_total_bits: float | None,
        ) -> dict:
    action = _load_action(action_json, num_layers=num_layers)
    decoded = action_vector_to_cfgs(
        action,
        load_max_sfs(profile),
        num_layers=num_layers,
    )
    requests = build_optimizer_requests(profile, decoded.cfgs_dict())
    bridge = RescaleOptimizerBridge(
        invoker=InProcessInvoker.from_profile(
            rescale_optimizer_root=rescale_optimizer_root,
            profile=profile,
        )
    )
    if np.array_equal(action, make_all_max_action_vector(num_layers=num_layers)):
        outputs = bridge.evaluate_baseline_blocks(requests)
    else:
        outputs = bridge.evaluate_blocks(requests)
    signals = aggregate_optimizer_signals(outputs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "optimizer_outputs.json").write_text(
        json.dumps(_optimizer_outputs_summary(outputs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record = build_f0_candidate_record(
        action,
        source=source,
        signals=signals,
        baseline_total_bits=baseline_total_bits,
    )
    record_path = CandidateStore(out / "candidates" / "candidate_store.jsonl").append(record)
    (out / "rank_key.json").write_text(
        json.dumps({"action_hash": record_path["action_hash"], "rank_key": record_path["rank_key"]}, indent=2) + "\n",
        encoding="utf-8",
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
    args = parser.parse_args(argv)
    record = run_f0_eval(
        profile=args.profile,
        num_layers=args.num_layers,
        action_json=args.action_json or None,
        output_dir=args.output_dir,
        source=args.source,
        rescale_optimizer_root=args.rescale_optimizer_root,
        baseline_total_bits=args.baseline_total_bits,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
