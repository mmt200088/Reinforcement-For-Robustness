#!/usr/bin/env python3
"""Verify Stage-2 RL persistent output artifacts.

This checker is intentionally torch-free. It is meant for the post-launch /
post-smoke gate: point it at a persistent run slug or at the progress directory
and it verifies that the user-facing live artifacts plus debug artifacts exist
and contain enough rows to be useful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonl_utils import count_jsonl_with_required_fields  # noqa: E402

REQUIRED_EPISODE_FIELDS = (
    "episode",
    "total_reward",
    "terminal_reward",
    "valid_steps",
    "invalid_steps",
    "total_bits",
    "fusion_count",
    "terminal_priority",
    "terminal_loss_mean",
    "terminal_metric1_mean",
    "terminal_metric2_mean",
    "policy_rollout_wall_seconds",
    "per_step_optimizer_wall_seconds",
)

REQUIRED_PPO_UPDATE_FIELDS = (
    "update",
    "completed_episodes",
    "policy_loss",
    "value_loss",
    "entropy",
    "clip_fraction",
    "n_samples",
    "window_mean_return",
    "best_reward_so_far",
    "elapsed_sec",
)


def _resolve_progress_dir(args: argparse.Namespace) -> Path:
    if args.progress_dir:
        return Path(args.progress_dir)
    if not args.run_dir:
        raise SystemExit("Either --run-dir or --progress-dir is required")
    run_dir = Path(args.run_dir)
    candidates = [
        run_dir / "stage2_noise" / "progress",
        run_dir / "progress",
        run_dir,
    ]
    for p in candidates:
        if (p / "blb_stage2_status.json").is_file():
            return p
    return candidates[0]


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _check_file(
        path: Path,
        failures: list[str],
        successes: list[str],
        *,
        required: bool = True,
        label: str | None = None,
) -> None:
    name = label or str(path)
    if _nonempty(path):
        successes.append(name)
    elif required:
        failures.append(f"missing or empty: {name}")


def _parse_status(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        failures.append(f"cannot parse status JSON {path}: {exc}")
        return {}
    if payload.get("schema") not in (None, "blb_stage2_status_v1"):
        failures.append(f"unexpected status schema: {payload.get('schema')!r}")
    return payload if isinstance(payload, dict) else {}


def _detail_dir_candidates(progress: Path, args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.run_dir:
        run_dir = Path(args.run_dir)
        candidates.extend(
            [
                run_dir / "stage2_noise" / "details",
                run_dir / "details",
            ]
        )
    candidates.extend(
        [
            progress / "details",
            progress.parent / "details",
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _latest_detail_files(progress: Path, args: argparse.Namespace) -> tuple[int, list[Path]]:
    detail_dirs = _detail_dir_candidates(progress, args)
    existing_dirs = [p for p in detail_dirs if p.is_dir()]
    file_count = 0
    for details_dir in existing_dirs:
        file_count += sum(1 for p in details_dir.iterdir() if p.is_file())
    return file_count, existing_dirs


def verify(args: argparse.Namespace) -> int:
    progress = _resolve_progress_dir(args)
    failures: list[str] = []
    successes: list[str] = []

    status_path = progress / "blb_stage2_status.json"
    _check_file(status_path, failures, successes, label="blb_stage2_status.json")
    status = _parse_status(status_path, failures) if status_path.is_file() else {}

    completed = int(status.get("completed_episodes", 0) or 0)
    ppo_updates = int(status.get("ppo_update_count", 0) or 0)
    if completed < int(args.min_episodes):
        failures.append(
            f"completed_episodes {completed} < required {int(args.min_episodes)}"
        )
    if ppo_updates < int(args.min_ppo_updates):
        failures.append(
            f"ppo_update_count {ppo_updates} < required {int(args.min_ppo_updates)}"
        )

    required_files = [
        "blb_stage2_live_summary.md",
        "blb_stage2_training_curve.npz",
        "diagnostics/episodes.jsonl",
        "diagnostics/ppo_updates.jsonl",
        "diagnostics/diagnostics_summary.md",
    ]
    for rel in required_files:
        _check_file(progress / rel, failures, successes, label=rel)

    if args.require_png:
        for rel in ("blb_stage2_training_curve.png", "blb_stage2_entropy_curve.png"):
            _check_file(progress / rel, failures, successes, label=rel)

    episodes_path = progress / "diagnostics" / "episodes.jsonl"
    if episodes_path.is_file():
        try:
            n_episode_rows, field_failures = count_jsonl_with_required_fields(
                episodes_path,
                REQUIRED_EPISODE_FIELDS,
                label="episodes.jsonl",
            )
            successes.append(f"episodes.jsonl rows={n_episode_rows}")
            if n_episode_rows < int(args.min_episodes):
                failures.append(
                    f"episodes.jsonl rows {n_episode_rows} < required {int(args.min_episodes)}"
                )
            failures.extend(field_failures)
        except Exception as exc:
            failures.append(f"cannot parse diagnostics/episodes.jsonl: {exc}")

    ppo_path = progress / "diagnostics" / "ppo_updates.jsonl"
    if ppo_path.is_file():
        try:
            n_ppo_rows, field_failures = count_jsonl_with_required_fields(
                ppo_path,
                REQUIRED_PPO_UPDATE_FIELDS,
                label="ppo_updates.jsonl",
            )
            successes.append(f"ppo_updates.jsonl rows={n_ppo_rows}")
            if n_ppo_rows < int(args.min_ppo_updates):
                failures.append(
                    f"ppo_updates.jsonl rows {n_ppo_rows} < required {int(args.min_ppo_updates)}"
                )
            failures.extend(field_failures)
        except Exception as exc:
            failures.append(f"cannot parse diagnostics/ppo_updates.jsonl: {exc}")

    detail_file_count, detail_dirs = _latest_detail_files(progress, args)
    if args.require_details and not detail_file_count:
        failures.append("missing detail batch files under stage2_noise/details or progress/details")
    elif detail_file_count:
        detail_dir_list = ", ".join(str(p) for p in detail_dirs)
        successes.append(f"details files={detail_file_count} ({detail_dir_list})")

    print(f"progress_dir={progress}")
    print(f"completed_episodes={completed}")
    print(f"ppo_update_count={ppo_updates}")
    for item in successes:
        print(f"[OK] {item}")
    if failures:
        print("VERIFY_FAIL")
        for item in failures:
            print(f"[FAIL] {item}")
        return 1
    print("VERIFY_OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--progress-dir", default="")
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--min-ppo-updates", type=int, default=1)
    parser.add_argument("--require-png", action="store_true")
    parser.add_argument("--require-details", action="store_true", default=True)
    return verify(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
