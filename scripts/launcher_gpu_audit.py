#!/usr/bin/env python3
"""Warn when launcher GPU flags leave visible GPUs unused.

This is a launcher/server-gate helper, not an RL algorithm component. It only
looks at user-facing flags and visible device count; it never changes the
training command or CUDA state.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Sequence


def _normalise(value: object) -> str:
    return str(value or "").strip()


def parse_device_spec(raw: object) -> List[str]:
    """Parse comma/list-like device specs into tokens.

    CUDA_VISIBLE_DEVICES may contain physical ids or UUIDs. Launcher flags use
    logical cuda ids after visibility filtering, so this function returns tokens
    only for counting. Recommendations are generated separately as logical ids.
    """
    text = _normalise(raw)
    if not text:
        return []
    lowered = text.lower()
    if lowered in {"none", "no", "off", "disabled", "void", "-1"}:
        return []
    if (text.startswith("(") and text.endswith(")")) or (
        text.startswith("[") and text.endswith("]")
    ):
        text = text[1:-1].strip()
    return [part.strip() for part in text.split(",") if part.strip()]


def _detect_nvidia_smi_devices() -> List[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def visible_devices(cuda_visible_devices: str | None) -> List[str]:
    """Return visible device tokens from CUDA_VISIBLE_DEVICES or nvidia-smi."""
    if cuda_visible_devices is None:
        env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_value is not None:
            return parse_device_spec(env_value)
        return _detect_nvidia_smi_devices()
    return parse_device_spec(cuda_visible_devices)


def _logical_device_list(count: int) -> str:
    return ",".join(str(i) for i in range(max(0, int(count))))


def _is_rl_stage(run_mode: str, stage_name: str) -> bool:
    return _normalise(run_mode).lower().replace("_", "-") == stage_name


def audit_launch(
    *,
    search_algorithm: str,
    run_mode: str,
    stage2_rl_variant: str,
    cuda_visible_devices: str | None = None,
    stage1_rl_devices: str = "",
    stage2_rl_devices: str = "",
    blb_v3_reward_devices: str = "",
    stage2_k_trials: int | str = 1,
) -> List[str]:
    """Return human-readable warnings for likely underused GPU launches."""
    if _normalise(search_algorithm).lower() != "rl":
        return []

    devices = visible_devices(cuda_visible_devices)
    visible_count = len(devices)
    if visible_count <= 1:
        return []

    logical = _logical_device_list(visible_count)
    warnings: List[str] = []
    if _is_rl_stage(run_mode, "stage1-only"):
        if not parse_device_spec(stage1_rl_devices):
            warnings.append(
                f"{visible_count} GPUs are visible, but Stage-1 RL has no "
                f"--stage1-rl-devices setting. Consider "
                f"--stage1-rl-devices {logical} so rollout collection uses all "
                "visible GPUs."
            )
        return warnings

    if not _is_rl_stage(run_mode, "stage2-only"):
        return warnings

    if _normalise(stage2_rl_variant).lower() != "blb_v3":
        return warnings

    stage2_devices = parse_device_spec(stage2_rl_devices)
    reward_devices = parse_device_spec(blb_v3_reward_devices)
    if not stage2_devices and not reward_devices:
        warnings.append(
            f"{visible_count} GPUs are visible, but Stage-2 BLB has neither "
            f"--stage2-rl-devices {logical} nor --blb-v3-reward-devices "
            f"{logical}. Pick the path that matches this run so terminal probe "
            "or episode collection does not silently fall back to one GPU."
        )
        return warnings

    if reward_devices:
        try:
            k_trials = int(stage2_k_trials)
        except (TypeError, ValueError):
            k_trials = 0
        if k_trials > 0 and k_trials < len(reward_devices):
            warnings.append(
                f"--stage2-k-trials {k_trials} provides fewer trials than the "
                f"{len(reward_devices)} reward devices in --blb-v3-reward-devices; "
                "some reward-probe GPUs will be idle unless this is intentional."
            )
    return warnings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-algorithm", required=True)
    parser.add_argument("--run-mode", required=True)
    parser.add_argument("--stage2-rl-variant", default="")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--stage1-rl-devices", default="")
    parser.add_argument("--stage2-rl-devices", default="")
    parser.add_argument("--blb-v3-reward-devices", default="")
    parser.add_argument("--stage2-k-trials", default="1")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return 2 when warnings are emitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    warnings = audit_launch(
        search_algorithm=args.search_algorithm,
        run_mode=args.run_mode,
        stage2_rl_variant=args.stage2_rl_variant,
        cuda_visible_devices=args.cuda_visible_devices,
        stage1_rl_devices=args.stage1_rl_devices,
        stage2_rl_devices=args.stage2_rl_devices,
        blb_v3_reward_devices=args.blb_v3_reward_devices,
        stage2_k_trials=args.stage2_k_trials,
    )
    for warning in warnings:
        print(f"[gpu-audit][WARN] {warning}", file=sys.stderr)
    return 2 if args.strict and warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
