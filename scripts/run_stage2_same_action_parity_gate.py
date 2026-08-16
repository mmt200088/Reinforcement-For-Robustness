#!/usr/bin/env python3
"""Run a real Stage-2 fixed action through PPO and comparator adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from json_utils import write_json_file  # noqa: E402
from blb_stage2_rl.same_action_parity import (  # noqa: E402
    run_same_action_parity_gate,
)
import blb_stage2_rl.search_baseline_runner as search_runner  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Intercept one production comparator setup and prove that a fixed "
            "action has the exact PPO and comparator evaluation result."
        )
    )
    parser.add_argument("--evidence-output", required=True)
    parser.add_argument("--stream-index", type=int, default=0)
    parser.add_argument(
        "--action-json",
        default="",
        help="JSON action matrix; defaults to all-zero rows for every layer",
    )
    parser.add_argument(
        "--run-strict-validation",
        action="store_true",
        help="also execute the production canonical strict A/B/C validator",
    )
    parser.add_argument("target", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.target and args.target[0] == "--":
        args.target = args.target[1:]
    if not args.target:
        parser.error("a target Python program and its arguments are required")
    return args


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *arguments],
        text=True,
    ).strip()


def main() -> int:
    args = _parse_args()
    evidence_path = Path(args.evidence_output).expanduser().resolve()
    action_override = (
        json.loads(args.action_json)
        if str(args.action_json).strip() else None
    )
    original_runner = search_runner.run_layerwise_search_baseline
    original_argv = list(sys.argv)
    completed = False

    def parity_gate(**kwargs: Any) -> Any:
        nonlocal completed
        layerwise_env = kwargs["layerwise_env"]
        horizon = int(layerwise_env.horizon)
        action_matrix = (
            action_override
            if action_override is not None
            else [[0, 0] for _ in range(horizon)]
        )
        expected_trials = int(
            layerwise_env.base.env_cfg.num_trials_per_step
        )
        strict_validator = (
            kwargs.get("strict_validator")
            if args.run_strict_validation else None
        )
        if args.run_strict_validation and strict_validator is None:
            raise RuntimeError(
                "production comparator setup did not provide strict validation"
            )
        evidence = run_same_action_parity_gate(
            layerwise_env=layerwise_env,
            robust_reference=kwargs["robust_reference"],
            action_matrix=action_matrix,
            base_seed=int(kwargs["seed"]),
            stream_index=int(args.stream_index),
            expected_trials=expected_trials,
            device="cpu",
            strict_validator=strict_validator,
        )
        evidence.update({
            "source_commit": _git_value("rev-parse", "HEAD"),
            "source_tree": _git_value("rev-parse", "HEAD^{tree}"),
            "target_program": str(args.target[0]),
            "requested_backend": str(kwargs["backend"]),
            "strict_validation_requested": bool(
                args.run_strict_validation
            ),
        })
        write_json_file(
            evidence_path,
            evidence,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        completed = True
        raise SystemExit(0)

    search_runner.run_layerwise_search_baseline = parity_gate
    try:
        target_path = Path(args.target[0])
        if not target_path.is_absolute():
            target_path = REPO_ROOT / target_path
        sys.argv = [str(target_path), *args.target[1:]]
        try:
            runpy.run_path(str(target_path), run_name="__main__")
        except SystemExit as exc:
            if completed and exc.code in (None, 0):
                return 0
            raise
    finally:
        search_runner.run_layerwise_search_baseline = original_runner
        sys.argv = original_argv
    raise RuntimeError("target completed without reaching the parity gate")


if __name__ == "__main__":
    raise SystemExit(main())
