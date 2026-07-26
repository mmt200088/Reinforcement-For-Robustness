#!/usr/bin/env python3
"""Strict scientific-equivalence and scaling gate for elastic RL runs.

Run this on the GPU server after matched Stage-1 or Stage-2 controls. The gate
compares logical training records and recursive checkpoint state exactly while
excluding only timing, process, device-assignment, health, and retry telemetry.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from blb_stage2_rl.candidate_store import CandidateStore
from jsonl_utils import iter_jsonl


_CHECKPOINT_NAMES = (
    "blb_stage2_rl_checkpoint_live.pt",
    "stage1_rl_checkpoint.pt",
)
_IGNORED_PATH_PARTS = {
    "pause_snapshots",
    "snapshots",
    "backup",
    "backups",
    "archive",
}
_EFFICIENCY_TELEMETRY_KEYS = {
    "created_at",
    "timestamp",
    "timestamps",
    "elapsed",
    "elapsed_sec",
    "elapsed_secs",
    "elapsed_seconds",
    "wall_seconds",
    "pid",
    "ppid",
    "process_id",
    "run_id",
    "structured_run_id",
    "run_dir",
    "run_path",
    "output_dir",
    "device",
    "devices",
    "device_id",
    "device_ids",
    "physical_device",
    "physical_devices",
    "cuda_visible_devices",
    "logical_device_spec",
    "terminal_probe_devices",
    "terminal_probe_trial_counts",
    "terminal_probe_trial_indices",
    "terminal_probe_trial_seeds",
    "worker",
    "worker_id",
    "worker_idx",
    "worker_index",
    "workers",
    "num_workers",
    "process_backend",
    "pool_generation",
    "retry_count",
    "retry_counts",
    "retry_rounds",
    "retried_trial_indices",
    "quarantine_events",
    "quarantined_devices",
    "quarantined_tokens",
    "healthy_tokens",
    "health_query_seconds",
    "recovery_action",
    "logical_generation",
    "candidate_store_size",
    "diagnostics_jsonl_sizes",
    "structured_jsonl_sizes",
    "detail_file_sizes",
    "details_file_sizes",
    "store_file_fingerprints",
    "cuda_rng_state_all",
    "cuda_rng_role_registry_version",
    "cuda_rng_state_by_role",
    "cuda_rng_active_role_count",
}
_EFFICIENCY_KEY_SUFFIXES = (
    "_wall_seconds",
    "_elapsed_seconds",
    "_elapsed_sec",
    "_timing_seconds",
    "_duration_seconds",
)
_EFFICIENCY_KEY_PREFIXES = (
    "per_worker_",
    "health_",
    "quarantine_",
)


@dataclass(frozen=True)
class RunArtifacts:
    root: Path
    checkpoint: Path
    diagnostic_episodes: Optional[Path]
    diagnostic_ppo: Optional[Path]
    candidate_store: Optional[Path]
    structured_steps: Optional[Path]
    structured_episodes: Optional[Path]
    structured_ppo: Optional[Path]
    structured_run_id: str


@dataclass(frozen=True)
class ComparisonResult:
    equal: bool
    diffs: Tuple[str, ...]
    compared: Mapping[str, Any]
    control: str
    candidate: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equal": bool(self.equal),
            "diffs": list(self.diffs),
            "compared": dict(self.compared),
            "control": self.control,
            "candidate": self.candidate,
        }


def _is_efficiency_telemetry_key(key: object) -> bool:
    normalized = str(key).strip().lower()
    if normalized in _EFFICIENCY_TELEMETRY_KEYS:
        return True
    if any(normalized.startswith(prefix) for prefix in _EFFICIENCY_KEY_PREFIXES):
        return True
    return any(normalized.endswith(suffix) for suffix in _EFFICIENCY_KEY_SUFFIXES)


def canonicalize(value: Any) -> Any:
    """Drop only fields that are expected to vary with execution efficiency."""
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(item)
            for key, item in value.items()
            if not _is_efficiency_telemetry_key(key)
        }
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(canonicalize(item) for item in value)
    return value


def _append_diff(
    diffs: List[str],
    message: str,
    *,
    max_diffs: int,
) -> None:
    if len(diffs) < int(max_diffs):
        diffs.append(str(message))


def _compare_values(
    control: Any,
    candidate: Any,
    *,
    path: str,
    diffs: List[str],
    max_diffs: int,
) -> None:
    if len(diffs) >= int(max_diffs):
        return

    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover - server has numpy.
        np = None
    try:
        import torch
    except ModuleNotFoundError:  # pragma: no cover - checkpoint gate needs torch.
        torch = None

    if torch is not None and (
        isinstance(control, torch.Tensor)
        or isinstance(candidate, torch.Tensor)
    ):
        if not (
            isinstance(control, torch.Tensor)
            and isinstance(candidate, torch.Tensor)
        ):
            _append_diff(
                diffs,
                f"{path}: tensor type differs",
                max_diffs=max_diffs,
            )
            return
        if (
            control.dtype != candidate.dtype
            or tuple(control.shape) != tuple(candidate.shape)
            or not torch.equal(control.cpu(), candidate.cpu())
        ):
            _append_diff(
                diffs,
                f"{path}: tensor differs "
                f"dtype={control.dtype}/{candidate.dtype} "
                f"shape={tuple(control.shape)}/{tuple(candidate.shape)}",
                max_diffs=max_diffs,
            )
        return

    if np is not None and (
        isinstance(control, np.ndarray)
        or isinstance(candidate, np.ndarray)
    ):
        if not (
            isinstance(control, np.ndarray)
            and isinstance(candidate, np.ndarray)
        ):
            _append_diff(
                diffs,
                f"{path}: ndarray type differs",
                max_diffs=max_diffs,
            )
            return
        if (
            control.dtype != candidate.dtype
            or control.shape != candidate.shape
            or not np.array_equal(control, candidate, equal_nan=True)
        ):
            _append_diff(
                diffs,
                f"{path}: ndarray differs",
                max_diffs=max_diffs,
            )
        return

    if isinstance(control, Mapping) or isinstance(candidate, Mapping):
        if not (
            isinstance(control, Mapping)
            and isinstance(candidate, Mapping)
        ):
            _append_diff(
                diffs,
                f"{path}: mapping type differs",
                max_diffs=max_diffs,
            )
            return
        control_keys = set(control)
        candidate_keys = set(candidate)
        for key in sorted(control_keys | candidate_keys, key=str):
            if key not in control or key not in candidate:
                _append_diff(
                    diffs,
                    f"{path}.{key}: key presence differs",
                    max_diffs=max_diffs,
                )
                if len(diffs) >= int(max_diffs):
                    return
                continue
            _compare_values(
                control[key],
                candidate[key],
                path=f"{path}.{key}",
                diffs=diffs,
                max_diffs=max_diffs,
            )
            if len(diffs) >= int(max_diffs):
                return
        return

    sequence_types = (list, tuple)
    if isinstance(control, sequence_types) or isinstance(candidate, sequence_types):
        if not (
            isinstance(control, sequence_types)
            and isinstance(candidate, sequence_types)
        ):
            _append_diff(
                diffs,
                f"{path}: sequence type differs",
                max_diffs=max_diffs,
            )
            return
        if len(control) != len(candidate):
            _append_diff(
                diffs,
                f"{path}: sequence length differs "
                f"{len(control)} != {len(candidate)}",
                max_diffs=max_diffs,
            )
            return
        for index, (control_item, candidate_item) in enumerate(
            zip(control, candidate)
        ):
            _compare_values(
                control_item,
                candidate_item,
                path=f"{path}[{index}]",
                diffs=diffs,
                max_diffs=max_diffs,
            )
            if len(diffs) >= int(max_diffs):
                return
        return

    if isinstance(control, float) or isinstance(candidate, float):
        try:
            control_float = float(control)
            candidate_float = float(candidate)
        except (TypeError, ValueError):
            pass
        else:
            if math.isnan(control_float) and math.isnan(candidate_float):
                return
            if control_float == candidate_float:
                return

    if control != candidate:
        _append_diff(
            diffs,
            f"{path}: {control!r} != {candidate!r}",
            max_diffs=max_diffs,
        )


def compare_scientific_values(
    control: Any,
    candidate: Any,
    *,
    path: str = "value",
    max_diffs: int = 100,
) -> Tuple[str, ...]:
    diffs: List[str] = []
    _compare_values(
        canonicalize(control),
        canonicalize(candidate),
        path=str(path),
        diffs=diffs,
        max_diffs=int(max_diffs),
    )
    return tuple(diffs)


def _usable_candidates(root: Path, name: str) -> List[Path]:
    return sorted(
        (
            path
            for path in root.rglob(name)
            if not any(part in _IGNORED_PATH_PARTS for part in path.parts)
        ),
        key=lambda path: (len(path.parts), str(path)),
    )


def _find_optional(root: Path, name: str) -> Optional[Path]:
    candidates = _usable_candidates(root, name)
    if not candidates:
        return None
    shortest_depth = len(candidates[0].parts)
    peers = [path for path in candidates if len(path.parts) == shortest_depth]
    if len(peers) > 1:
        raise RuntimeError(
            f"ambiguous {name} under {root}: "
            + ", ".join(str(path) for path in peers)
        )
    return candidates[0]


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping: {path}")
    return checkpoint


def _find_checkpoint(root: Path) -> Path:
    for name in _CHECKPOINT_NAMES:
        path = _find_optional(root, name)
        if path is not None:
            return path
    raise FileNotFoundError(f"no RL checkpoint found under {root}")


def _find_structured_dir(
    data_points_root: Path,
    *,
    stage: str,
    run_id: str,
) -> Path:
    stage_root = data_points_root / str(stage)
    candidates = sorted(
        path
        for path in stage_root.rglob(str(run_id))
        if path.is_dir() and path.name == str(run_id)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one structured run {run_id!r} under {stage_root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def discover_run_artifacts(
    root: Path | str,
    *,
    stage: str,
    data_points_root: Optional[Path | str] = None,
) -> RunArtifacts:
    run_root = Path(root).expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root is not a directory: {run_root}")
    checkpoint_path = _find_checkpoint(run_root)
    checkpoint = _load_checkpoint(checkpoint_path)
    structured_run_id = str(checkpoint.get("structured_run_id", "") or "")

    structured_steps = None
    structured_episodes = None
    structured_ppo = None
    if data_points_root is not None:
        if not structured_run_id:
            raise RuntimeError(
                f"checkpoint lacks structured_run_id: {checkpoint_path}"
            )
        structured_dir = _find_structured_dir(
            Path(data_points_root).expanduser().resolve(),
            stage=str(stage),
            run_id=structured_run_id,
        )
        structured_steps = structured_dir / "steps.jsonl"
        structured_episodes = structured_dir / "episodes.jsonl"
        structured_ppo = structured_dir / "ppo_updates.jsonl"
        for required in (
            structured_steps,
            structured_episodes,
            structured_ppo,
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"structured training artifact is missing: {required}"
                )

    return RunArtifacts(
        root=run_root,
        checkpoint=checkpoint_path,
        diagnostic_episodes=_find_optional(run_root, "episodes.jsonl"),
        diagnostic_ppo=_find_optional(run_root, "ppo_updates.jsonl"),
        candidate_store=_find_optional(run_root, "candidate_store.jsonl"),
        structured_steps=structured_steps,
        structured_episodes=structured_episodes,
        structured_ppo=structured_ppo,
        structured_run_id=structured_run_id,
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path, errors="raise")]


def _load_candidate_records(path: Path) -> List[Dict[str, Any]]:
    return [dict(row) for row in CandidateStore(path).iter_active_records()]


def _compare_optional_artifact(
    *,
    label: str,
    control_path: Optional[Path],
    candidate_path: Optional[Path],
    loader,
    diffs: List[str],
    compared: Dict[str, Any],
    max_diffs: int,
) -> None:
    if control_path is None or candidate_path is None:
        if control_path != candidate_path:
            _append_diff(
                diffs,
                f"{label}: artifact presence differs "
                f"{control_path} != {candidate_path}",
                max_diffs=max_diffs,
            )
        compared[label] = 0
        return
    control_value = loader(control_path)
    candidate_value = loader(candidate_path)
    compared[label] = len(control_value)
    artifact_diffs = compare_scientific_values(
        control_value,
        candidate_value,
        path=label,
        max_diffs=max(0, int(max_diffs) - len(diffs)),
    )
    diffs.extend(artifact_diffs)


def compare_runs(
    control: Path | str,
    candidate: Path | str,
    *,
    stage: str,
    data_points_root: Optional[Path | str] = None,
    max_diffs: int = 100,
) -> ComparisonResult:
    """Compare two complete RL runs after canonical telemetry exclusion."""
    stage_name = str(stage).strip().lower()
    if stage_name not in {"stage1", "stage2"}:
        raise ValueError("stage must be stage1 or stage2")
    control_artifacts = discover_run_artifacts(
        control,
        stage=stage_name,
        data_points_root=data_points_root,
    )
    candidate_artifacts = discover_run_artifacts(
        candidate,
        stage=stage_name,
        data_points_root=data_points_root,
    )
    diffs: List[str] = []
    compared: Dict[str, Any] = {}

    checkpoint_diffs = compare_scientific_values(
        _load_checkpoint(control_artifacts.checkpoint),
        _load_checkpoint(candidate_artifacts.checkpoint),
        path="checkpoint",
        max_diffs=max(0, int(max_diffs) - len(diffs)),
    )
    diffs.extend(checkpoint_diffs)
    compared["checkpoint"] = True

    artifacts = (
        (
            "diagnostic_episodes",
            control_artifacts.diagnostic_episodes,
            candidate_artifacts.diagnostic_episodes,
            _load_jsonl,
        ),
        (
            "diagnostic_ppo",
            control_artifacts.diagnostic_ppo,
            candidate_artifacts.diagnostic_ppo,
            _load_jsonl,
        ),
        (
            "candidate_records",
            control_artifacts.candidate_store,
            candidate_artifacts.candidate_store,
            _load_candidate_records,
        ),
        (
            "structured_steps",
            control_artifacts.structured_steps,
            candidate_artifacts.structured_steps,
            _load_jsonl,
        ),
        (
            "structured_episodes",
            control_artifacts.structured_episodes,
            candidate_artifacts.structured_episodes,
            _load_jsonl,
        ),
        (
            "structured_ppo",
            control_artifacts.structured_ppo,
            candidate_artifacts.structured_ppo,
            _load_jsonl,
        ),
    )
    for label, control_path, candidate_path, loader in artifacts:
        if len(diffs) >= int(max_diffs):
            break
        _compare_optional_artifact(
            label=label,
            control_path=control_path,
            candidate_path=candidate_path,
            loader=loader,
            diffs=diffs,
            compared=compared,
            max_diffs=int(max_diffs),
        )

    if stage_name == "stage2":
        for required_label in (
            "diagnostic_episodes",
            "diagnostic_ppo",
            "candidate_records",
        ):
            if int(compared.get(required_label, 0) or 0) <= 0:
                _append_diff(
                    diffs,
                    f"{required_label}: Stage-2 artifact is empty or missing",
                    max_diffs=max_diffs,
                )
    if data_points_root is not None:
        for required_label in (
            "structured_steps",
            "structured_episodes",
            "structured_ppo",
        ):
            if int(compared.get(required_label, 0) or 0) <= 0:
                _append_diff(
                    diffs,
                    f"{required_label}: structured artifact is empty",
                    max_diffs=max_diffs,
                )

    return ComparisonResult(
        equal=not diffs,
        diffs=tuple(diffs),
        compared=compared,
        control=str(control_artifacts.root),
        candidate=str(candidate_artifacts.root),
    )


def summarize_scaling(
    runs: Mapping[int, Tuple[Path | str, float]],
    *,
    stage: str,
    data_points_root: Optional[Path | str] = None,
    min_parallel_efficiency: float = 0.0,
    max_diffs: int = 100,
) -> Dict[str, Any]:
    """Build an exact-equivalence and wall-throughput scaling verdict."""
    normalized = {
        int(gpu_count): (Path(path).expanduser().resolve(), float(wall_seconds))
        for gpu_count, (path, wall_seconds) in runs.items()
    }
    if not normalized:
        raise ValueError("at least one scaling run is required")
    if any(gpu_count <= 0 for gpu_count in normalized):
        raise ValueError("GPU counts must be positive")
    if any(wall_seconds <= 0.0 for _path, wall_seconds in normalized.values()):
        raise ValueError("wall seconds must be positive")
    control_gpu_count = min(normalized)
    control_path, control_wall = normalized[control_gpu_count]
    control_artifacts = discover_run_artifacts(
        control_path,
        stage=stage,
        data_points_root=data_points_root,
    )
    if control_artifacts.diagnostic_episodes is None:
        raise FileNotFoundError(
            f"control diagnostic episodes are missing: {control_path}"
        )
    control_episode_count = len(
        _load_jsonl(control_artifacts.diagnostic_episodes)
    )
    if control_episode_count <= 0:
        raise RuntimeError("control run has no diagnostic episodes")
    control_throughput = (
        float(control_episode_count) * 3600.0 / float(control_wall)
    )

    run_summaries: Dict[str, Any] = {}
    exact_equivalence = True
    throughput_sequence: List[float] = []
    efficiency_pass = True
    for gpu_count in sorted(normalized):
        run_path, wall_seconds = normalized[gpu_count]
        if gpu_count == control_gpu_count:
            comparison = ComparisonResult(
                equal=True,
                diffs=(),
                compared={},
                control=str(control_path),
                candidate=str(run_path),
            )
            episode_count = control_episode_count
        else:
            comparison = compare_runs(
                control_path,
                run_path,
                stage=stage,
                data_points_root=data_points_root,
                max_diffs=max_diffs,
            )
            run_artifacts = discover_run_artifacts(
                run_path,
                stage=stage,
                data_points_root=data_points_root,
            )
            if run_artifacts.diagnostic_episodes is None:
                raise FileNotFoundError(
                    f"diagnostic episodes are missing: {run_path}"
                )
            episode_count = len(
                _load_jsonl(run_artifacts.diagnostic_episodes)
            )
        throughput = float(episode_count) * 3600.0 / float(wall_seconds)
        speedup = throughput / control_throughput
        theoretical = float(gpu_count) / float(control_gpu_count)
        parallel_efficiency = speedup / theoretical
        meets_efficiency = (
            parallel_efficiency + 1.0e-12
            >= float(min_parallel_efficiency)
        )
        exact_equivalence = exact_equivalence and comparison.equal
        efficiency_pass = efficiency_pass and meets_efficiency
        throughput_sequence.append(throughput)
        run_summaries[str(gpu_count)] = {
            "path": str(run_path),
            "wall_seconds": float(wall_seconds),
            "episodes": int(episode_count),
            "episodes_per_hour": float(throughput),
            "speedup": float(speedup),
            "theoretical_speedup": float(theoretical),
            "parallel_efficiency": float(parallel_efficiency),
            "meets_parallel_efficiency": bool(meets_efficiency),
            "exact_equivalence": bool(comparison.equal),
            "diffs": list(comparison.diffs),
            "compared": dict(comparison.compared),
        }

    monotonic_throughput = all(
        current + 1.0e-12 >= previous
        for previous, current in zip(
            throughput_sequence,
            throughput_sequence[1:],
        )
    )
    passed = (
        exact_equivalence
        and monotonic_throughput
        and efficiency_pass
    )
    return {
        "record_type": "elastic_rl_scaling_verdict_v1",
        "stage": str(stage),
        "control_gpu_count": int(control_gpu_count),
        "minimum_parallel_efficiency": float(min_parallel_efficiency),
        "exact_equivalence": bool(exact_equivalence),
        "monotonic_throughput": bool(monotonic_throughput),
        "parallel_efficiency_pass": bool(efficiency_pass),
        "passed": bool(passed),
        "runs": run_summaries,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _compare_command(args: argparse.Namespace) -> int:
    result = compare_runs(
        args.control,
        args.candidate,
        stage=args.stage,
        data_points_root=args.data_points_root,
        max_diffs=args.max_diffs,
    )
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        _write_json(Path(args.output), payload)
    return 0 if result.equal else 2


def _parse_key_values(
    values: Iterable[str],
    *,
    label: str,
) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for raw in values:
        key_text, separator, value = str(raw).partition("=")
        if not separator or not key_text.strip() or not value.strip():
            raise ValueError(f"{label} must use GPU_COUNT=VALUE, got {raw!r}")
        key = int(key_text)
        if key in parsed:
            raise ValueError(f"duplicate {label} GPU count: {key}")
        parsed[key] = value
    return parsed


def _read_wall_value(value: str) -> float:
    candidate = Path(str(value)).expanduser()
    if candidate.is_file():
        return float(candidate.read_text(encoding="utf-8").strip())
    return float(value)


def _scaling_command(args: argparse.Namespace) -> int:
    run_paths = _parse_key_values(args.run, label="--run")
    wall_values = _parse_key_values(args.wall, label="--wall")
    if set(run_paths) != set(wall_values):
        raise ValueError(
            "--run and --wall GPU counts differ: "
            f"{sorted(run_paths)} != {sorted(wall_values)}"
        )
    runs = {
        gpu_count: (run_paths[gpu_count], _read_wall_value(wall_values[gpu_count]))
        for gpu_count in run_paths
    }
    summary = summarize_scaling(
        runs,
        stage=args.stage,
        data_points_root=args.data_points_root,
        min_parallel_efficiency=args.min_parallel_efficiency,
        max_diffs=args.max_diffs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        _write_json(Path(args.output), summary)
    return 0 if summary["passed"] else 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare two complete RL run directories",
    )
    compare_parser.add_argument("--control", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument(
        "--stage",
        required=True,
        choices=("stage1", "stage2"),
    )
    compare_parser.add_argument("--data-points-root")
    compare_parser.add_argument("--max-diffs", type=int, default=100)
    compare_parser.add_argument("--output")
    compare_parser.set_defaults(handler=_compare_command)

    scaling_parser = subparsers.add_parser(
        "scaling",
        help="compare matched GPU-count runs and calculate throughput scaling",
    )
    scaling_parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="GPU_COUNT=RUN_DIR",
    )
    scaling_parser.add_argument(
        "--wall",
        action="append",
        required=True,
        metavar="GPU_COUNT=SECONDS_OR_FILE",
    )
    scaling_parser.add_argument(
        "--stage",
        required=True,
        choices=("stage1", "stage2"),
    )
    scaling_parser.add_argument("--data-points-root")
    scaling_parser.add_argument(
        "--min-parallel-efficiency",
        type=float,
        default=0.80,
    )
    scaling_parser.add_argument("--max-diffs", type=int, default=100)
    scaling_parser.add_argument("--output")
    scaling_parser.set_defaults(handler=_scaling_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
