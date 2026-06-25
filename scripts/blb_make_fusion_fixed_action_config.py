#!/usr/bin/env python3
"""Convert a flat Stage-2 best-action JSON into fixed-fusion final-eval config.

Stage-2 fusion-count RL executes per-step actions as ``(fusion_option, K)``.
The persisted legacy ``action_vec`` is still useful, but by itself it loses the
semantic fusion option selection, especially for boosted options that carry
above-grid ``explicit_field_values``.  This helper reconstructs
``group.option_by_step`` by matching each block slice against the committed
fusion-count maps while leaving the K slot exactly as encoded in the flat vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
import sys

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "blb_stage2_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blb_stage2_rl.fusion_fixed_action import build_fusion_fixed_config


def _parse_int_list(raw: str, *, name: str, num_layers: int) -> List[int]:
    value = json.loads(raw)
    if not isinstance(value, list) or len(value) != int(num_layers):
        raise ValueError(f"{name} must be a JSON list with {num_layers} entries")
    return [int(x) for x in value]


def _load_action_vec(path: Path) -> List[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("action_vec"), list):
        raise ValueError(f"{path} must contain a top-level action_vec list")
    return [int(x) for x in payload["action_vec"]]


def build_config(
        *,
        action_vec: Sequence[int],
        profile: str,
        num_layers: int,
        gelu: Sequence[int],
        softmax: Sequence[int],
        source_path: str,
        ) -> Dict[str, Any]:
    """Thin wrapper over the shared reconstruction core (so the matcher lives in
    one place, reused by the runner / final-eval / GLUE boost-replay paths)."""
    return build_fusion_fixed_config(
        action_vec,
        profile=str(profile),
        num_layers=int(num_layers),
        gelu=gelu,
        softmax=softmax,
        source="inferred_from_flat_stage2_best_action",
        source_path=str(source_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="best_action_full.json or best_action_vec.json")
    parser.add_argument("--output", required=True, help="output fixed-fusion action-config JSON")
    parser.add_argument("--profile", default="mrpc")
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--gelu", required=True, help="JSON list of per-layer GELU degrees")
    parser.add_argument("--softmax", required=True, help="JSON list of per-layer Softmax degrees")
    args = parser.parse_args()

    input_path = Path(args.input)
    action_vec = _load_action_vec(input_path)
    cfg = build_config(
        action_vec=action_vec,
        profile=str(args.profile),
        num_layers=int(args.num_layers),
        gelu=_parse_int_list(args.gelu, name="gelu", num_layers=int(args.num_layers)),
        softmax=_parse_int_list(args.softmax, name="softmax", num_layers=int(args.num_layers)),
        source_path=str(input_path),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = cfg["summary"]
    print(
        f"wrote {output_path} "
        f"steps={summary['step_count']} fusion={summary['total_fusion_count']} "
        f"avg_k={summary['avg_k']:.4f} boosted_options={summary['boosted_option_count']}"
    )


if __name__ == "__main__":
    main()
