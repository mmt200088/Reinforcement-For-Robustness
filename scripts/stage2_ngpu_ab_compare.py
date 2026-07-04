#!/usr/bin/env python3
"""Compare deterministic Stage-2 1-GPU vs N-GPU rollout runs.

This is intentionally torch-free. It treats wall-clock throughput as the speed
verdict and uses episode JSON only for equality/effect checks and diagnostics.
Timing/device fields are excluded from equality because those are the intended
differences between 1-GPU and N-GPU runs.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from jsonl_utils import iter_jsonl


TIMING_OR_DEVICE_KEYS = {
    "timestamp",
    "terminal_probe_wall_seconds",
    "terminal_probe_devices",
    "terminal_probe_trial_counts",
    "terminal_probe_trial_indices",
    "terminal_probe_speedup",
    "terminal_cost_eval_wall_seconds",
    "terminal_probe_install_wall_seconds",
    "terminal_probe_clear_wall_seconds",
    "per_step_optimizer_wall_seconds",
    "policy_rollout_wall_seconds",
    "rejection_optimizer_wall_seconds",
}

PPO_TIMING_KEYS = {
    "timestamp",
    "elapsed_sec",
}

DIAGNOSTIC_BOOKKEEPING_KEYS = {
    # Frontier event classification is diagnostic bookkeeping. It can differ
    # between historical 1-GPU and N-GPU artifacts even when rewards, metrics,
    # actions, hashes, and PPO-visible fields are identical.
    "terminal_pareto_event_kind",
    "terminal_pareto_frontier_removed",
}


def _find_jsonl(path: str, filename: str) -> str:
    if os.path.isfile(path):
        return path
    candidates = [
        os.path.join(path, filename),
        os.path.join(path, "diagnostics", filename),
        os.path.join(path, "progress", "diagnostics", filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    for root, _dirs, files in os.walk(path):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"could not find {filename} under {path}")


def _load_jsonl(path: str, *, filename: str, sort_key: str) -> List[Dict[str, Any]]:
    jsonl_path = _find_jsonl(path, filename)
    rows: List[Dict[str, Any]] = []
    previous_key: Optional[int] = None
    ordered = True
    for row in iter_jsonl(jsonl_path, errors="raise"):
        key = int(row.get(sort_key, 0) or 0)
        if previous_key is not None and key < previous_key:
            ordered = False
        previous_key = key
        rows.append(row)
    if ordered:
        return rows
    return sorted(rows, key=lambda row: int(row.get(sort_key, 0) or 0))


def _load_episodes(path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(path, filename="episodes.jsonl", sort_key="episode")


def _load_ppo_updates(path: str) -> List[Dict[str, Any]]:
    return _load_jsonl(path, filename="ppo_updates.jsonl", sort_key="update")


def _read_wall_seconds(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()
    return float(text)


def _timestamp_span(rows: Iterable[Mapping[str, Any]]) -> Optional[float]:
    count = 0
    min_ts: Optional[float] = None
    max_ts: Optional[float] = None
    for row in rows:
        raw_value = row.get("timestamp")
        if raw_value is None:
            continue
        value = float(raw_value)
        count += 1
        if min_ts is None or value < min_ts:
            min_ts = value
        if max_ts is None or value > max_ts:
            max_ts = value
    if count < 2 or min_ts is None or max_ts is None:
        return None
    return float(max_ts - min_ts)


def _canonical(
        row: Mapping[str, Any],
        *,
        strict_diagnostics: bool = False,
        excluded_keys: Optional[Iterable[str]] = None,
        ) -> Dict[str, Any]:
    excluded = set(TIMING_OR_DEVICE_KEYS if excluded_keys is None else excluded_keys)
    if not bool(strict_diagnostics):
        excluded.update(DIAGNOSTIC_BOOKKEEPING_KEYS)
    return _canonical_with_exclusions(row, excluded)


def _canonical_with_exclusions(
        row: Mapping[str, Any],
        excluded_keys: Iterable[str],
        ) -> Dict[str, Any]:
    return {
        str(key): value
        for key, value in row.items()
        if str(key) not in excluded_keys
    }


def _numbers_equal(a: Any, b: Any, *, atol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            return str(a) == str(b)
        return abs(float(a) - float(b)) <= float(atol)
    return a == b


def _diff_values(
        a: Any,
        b: Any,
        *,
        path: str,
        atol: float,
        limit: int,
        out: List[str],
        ) -> None:
    if len(out) >= int(limit):
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        keys = sorted(set(a) | set(b))
        for key in keys:
            if key not in a or key not in b:
                out.append(f"{path}.{key}: key presence differs")
                if len(out) >= int(limit):
                    return
                continue
            _diff_values(a[key], b[key], path=f"{path}.{key}", atol=atol, limit=limit, out=out)
            if len(out) >= int(limit):
                return
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length differs {len(a)} != {len(b)}")
            return
        for idx, (av, bv) in enumerate(zip(a, b)):
            _diff_values(av, bv, path=f"{path}[{idx}]", atol=atol, limit=limit, out=out)
            if len(out) >= int(limit):
                return
        return
    if not _numbers_equal(a, b, atol=atol):
        out.append(f"{path}: {a!r} != {b!r}")


def compare_rows(
        one: Sequence[Mapping[str, Any]],
        many: Sequence[Mapping[str, Any]],
        *,
        atol: float,
        limit: int,
        strict_diagnostics: bool = False,
        key_field: str = "episode",
        row_label: str = "episode",
        excluded_keys: Optional[Iterable[str]] = None,
        ) -> Tuple[bool, List[str]]:
    if len(one) != len(many):
        return False, [f"{row_label} count differs: {len(one)} != {len(many)}"]
    excluded = set(TIMING_OR_DEVICE_KEYS if excluded_keys is None else excluded_keys)
    if not bool(strict_diagnostics):
        excluded.update(DIAGNOSTIC_BOOKKEEPING_KEYS)
    diffs: List[str] = []
    for idx, (a_row, b_row) in enumerate(zip(one, many)):
        a_key = int(a_row.get(key_field, idx) or 0)
        b_key = int(b_row.get(key_field, idx) or 0)
        if a_key != b_key:
            diffs.append(f"row {idx}: {key_field} differs {a_key} != {b_key}")
            if len(diffs) >= int(limit):
                break
            continue
        _diff_values(
            _canonical_with_exclusions(a_row, excluded),
            _canonical_with_exclusions(b_row, excluded),
            path=f"{row_label}[{a_key}]",
            atol=float(atol),
            limit=int(limit),
            out=diffs,
        )
        if len(diffs) >= int(limit):
            break
    return not diffs, diffs


def _sum_float(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0.0) or 0.0) for row in rows))


def _mean_float(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    if not rows:
        return None
    return _sum_float(rows, key) / float(len(rows))


def _device_breakdown(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = collections.defaultdict(
        lambda: {
            "episodes": 0.0,
            "probe_s": 0.0,
            "policy_s": 0.0,
            "optimizer_s": 0.0,
            "component_s": 0.0,
        }
    )
    for row in rows:
        devices = row.get("terminal_probe_devices") or ["unknown"]
        device = str(devices[0] if devices else "unknown")
        probe_s = float(row.get("terminal_probe_wall_seconds", 0.0) or 0.0)
        policy_s = float(row.get("policy_rollout_wall_seconds", 0.0) or 0.0)
        optimizer_s = float(row.get("per_step_optimizer_wall_seconds", 0.0) or 0.0)
        out[device]["episodes"] += 1.0
        out[device]["probe_s"] += probe_s
        out[device]["policy_s"] += policy_s
        out[device]["optimizer_s"] += optimizer_s
        out[device]["component_s"] += probe_s + policy_s + optimizer_s
    return dict(out)


def _throughput(n: int, wall_s: Optional[float]) -> Optional[float]:
    if wall_s is None or wall_s <= 0.0:
        return None
    return float(n) * 3600.0 / float(wall_s)


def _fmt(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def _safe_div(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None:
        return None
    if float(denom) <= 0.0:
        return None
    return float(numer) / float(denom)


_TIMING_RE = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")

_NGPU_LOG_MARKERS = (
    "worker-local probe noise scopes active",
    "worker-local CUDA probe streams active",
    "policy_device=cpu",
)


def _parse_rollout_log(
        path: Optional[str],
        *,
        markers: Iterable[str] = (),
        ) -> Tuple[Dict[str, float], Dict[str, bool]]:
    marker_tuple = tuple(str(marker) for marker in markers)
    marker_flags = {marker: False for marker in marker_tuple}
    if not path:
        return {}, marker_flags
    if not os.path.isfile(path):
        return {}, marker_flags
    out: Dict[str, float] = collections.defaultdict(float)
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker_tuple:
                for marker in marker_tuple:
                    if not marker_flags[marker] and marker in line:
                        marker_flags[marker] = True
            if "stage2-rollout-timing" not in line:
                continue
            parsed: Dict[str, float] = {}
            for key, value in _TIMING_RE.findall(line):
                try:
                    parsed[str(key)] = float(value)
                except ValueError:
                    continue
            if not parsed:
                continue
            out["windows"] += 1.0
            out["episodes"] += float(parsed.get("episodes", 0.0) or 0.0)
            for key, value in parsed.items():
                if key in {"window_start", "episodes"}:
                    continue
                out[str(key)] += float(value)
    return dict(out), marker_flags


def _parse_rollout_timing_log(path: Optional[str]) -> Dict[str, float]:
    timing, _markers = _parse_rollout_log(path)
    return timing


def _timing_report_lines(label: str, summary: Mapping[str, float]) -> List[str]:
    if not summary:
        return [f"{label} rollout timing log: n/a"]
    windows = int(summary.get("windows", 0.0) or 0.0)
    episodes = int(summary.get("episodes", 0.0) or 0.0)
    lines = [f"{label} rollout timing log: windows={windows} episodes={episodes}"]
    for key in sorted(k for k in summary if k not in {"windows", "episodes"}):
        total = float(summary.get(key, 0.0) or 0.0)
        mean = total / max(float(windows), 1.0)
        lines.append(f"  {key}: total_s={total:.3f} mean_per_window_s={mean:.3f}")
    return lines


def build_report(args: argparse.Namespace) -> str:
    one = _load_episodes(args.one)
    many = _load_episodes(args.many)
    equality_ok, diffs = compare_rows(
        one,
        many,
        atol=float(args.atol),
        limit=int(args.max_diffs),
        strict_diagnostics=bool(args.strict_diagnostics),
    )
    _diag_ok, diag_diffs = compare_rows(
        one,
        many,
        atol=float(args.atol),
        limit=int(args.max_diffs),
        strict_diagnostics=True,
    )
    ppo_equality_ok: Optional[bool] = None
    ppo_diffs: List[str] = []
    one_ppo_count: Optional[int] = None
    many_ppo_count: Optional[int] = None
    one_ppo_path = getattr(args, "one_ppo", None)
    many_ppo_path = getattr(args, "many_ppo", None)
    if one_ppo_path or many_ppo_path:
        if not one_ppo_path or not many_ppo_path:
            ppo_equality_ok = False
            ppo_diffs = [
                "PPO update file presence differs: "
                f"one={bool(one_ppo_path)} many={bool(many_ppo_path)}"
            ]
        else:
            one_ppo = _load_ppo_updates(one_ppo_path)
            many_ppo = _load_ppo_updates(many_ppo_path)
            one_ppo_count = len(one_ppo)
            many_ppo_count = len(many_ppo)
            ppo_equality_ok, ppo_diffs = compare_rows(
                one_ppo,
                many_ppo,
                atol=float(args.atol),
                limit=int(args.max_diffs),
                strict_diagnostics=True,
                key_field="update",
                row_label="ppo_update",
                excluded_keys=PPO_TIMING_KEYS,
            )
    one_wall = _read_wall_seconds(args.one_wall)
    many_wall = _read_wall_seconds(args.many_wall)
    one_ts = _timestamp_span(one)
    many_ts = _timestamp_span(many)
    one_wall_for_speed = one_wall if one_wall is not None else one_ts
    many_wall_for_speed = many_wall if many_wall is not None else many_ts
    one_eph = _throughput(len(one), one_wall_for_speed)
    many_eph = _throughput(len(many), many_wall_for_speed)
    speedup = (
        None
        if one_eph is None or many_eph is None or one_eph <= 0.0
        else float(many_eph / one_eph)
    )
    many_dev = _device_breakdown(many)
    one_dev = _device_breakdown(one)
    many_probe_bound = max((v["probe_s"] for v in many_dev.values()), default=0.0)
    many_policy_bound = max((v["policy_s"] for v in many_dev.values()), default=0.0)
    many_component_bound = max((v["component_s"] for v in many_dev.values()), default=0.0)
    many_device_count = len([d for d in many_dev if d != "unknown"])
    many_episode_counts = [
        float(v["episodes"])
        for device, v in many_dev.items()
        if device != "unknown"
    ]
    many_episode_min = min(many_episode_counts) if many_episode_counts else None
    many_episode_max = max(many_episode_counts) if many_episode_counts else None
    speedup_fraction_of_devices = _safe_div(
        speedup,
        float(many_device_count) if many_device_count > 0 else None,
    )
    many_probe_bound_eph = _throughput(len(many), many_probe_bound)
    many_component_bound_eph = _throughput(len(many), many_component_bound)
    many_wall_over_probe_bound = _safe_div(many_wall_for_speed, many_probe_bound)
    many_wall_over_component_bound = _safe_div(many_wall_for_speed, many_component_bound)
    many_probe_ceiling_utilization = _safe_div(many_eph, many_probe_bound_eph)
    many_component_ceiling_utilization = _safe_div(many_eph, many_component_bound_eph)
    one_timing, _one_markers = _parse_rollout_log(getattr(args, "one_log", None))
    many_timing, many_markers = _parse_rollout_log(
        getattr(args, "many_log", None),
        markers=_NGPU_LOG_MARKERS,
    )
    one_probe_mean = _mean_float(one, "terminal_probe_wall_seconds")
    many_probe_mean = _mean_float(many, "terminal_probe_wall_seconds")
    one_policy_mean = _mean_float(one, "policy_rollout_wall_seconds")
    many_policy_mean = _mean_float(many, "policy_rollout_wall_seconds")
    scoped_noise_detected = many_markers.get("worker-local probe noise scopes active", False)
    worker_streams_detected = many_markers.get("worker-local CUDA probe streams active", False)
    cpu_policy_detected = many_markers.get("policy_device=cpu", False)
    lines = [
        "==== Stage-2 N-GPU A/B Verdict ====",
        f"1GPU episodes: {len(one)}",
        f"NGPU episodes: {len(many)}",
        f"quality/effect equality: {'PASS' if equality_ok else 'FAIL'}",
        f"strict diagnostic equality: {'PASS' if not diag_diffs else 'DIFF'}",
        (
            "PPO update equality: n/a"
            if ppo_equality_ok is None
            else f"PPO update equality: {'PASS' if ppo_equality_ok else 'FAIL'}"
        ),
        (
            "PPO update counts: n/a"
            if ppo_equality_ok is None
            else f"PPO update counts: 1GPU={one_ppo_count} NGPU={many_ppo_count}"
        ),
        f"wall source: {'wall file' if one_wall is not None and many_wall is not None else 'episode timestamps'}",
        f"1GPU wall_s: {_fmt(one_wall_for_speed)}",
        f"NGPU wall_s: {_fmt(many_wall_for_speed)}",
        f"1GPU episodes/hour: {_fmt(one_eph)}",
        f"NGPU episodes/hour: {_fmt(many_eph)}",
        f"speedup: {_fmt(speedup, 'x')}",
        f"NGPU distinct probe devices: {many_device_count}",
        f"speedup/device_count: {_fmt(speedup_fraction_of_devices)}",
        (
            "NGPU device episode balance min/max: n/a"
            if many_episode_min is None or many_episode_max is None
            else f"NGPU device episode balance min/max: {many_episode_min:.0f}/{many_episode_max:.0f}"
        ),
        f"NGPU probe critical-path lower bound_s: {many_probe_bound:.3f}",
        f"NGPU probe-bound ceiling episodes/hour: {_fmt(many_probe_bound_eph)}",
        f"NGPU wall/probe_bound ratio: {_fmt(many_wall_over_probe_bound)}",
        f"NGPU probe ceiling utilization: {_fmt(many_probe_ceiling_utilization)}",
        f"NGPU component critical-path lower bound_s: {many_component_bound:.3f}",
        f"NGPU component-bound ceiling episodes/hour: {_fmt(many_component_bound_eph)}",
        f"NGPU wall/component_bound ratio: {_fmt(many_wall_over_component_bound)}",
        f"NGPU component ceiling utilization: {_fmt(many_component_ceiling_utilization)}",
        f"NGPU policy diagnostic critical-path sum_s: {many_policy_bound:.3f}",
        f"1GPU terminal probe mean_s/episode: {_fmt(one_probe_mean)}",
        f"NGPU terminal probe mean_s/episode: {_fmt(many_probe_mean)}",
        f"NGPU/1GPU terminal probe mean ratio: {_fmt(_safe_div(many_probe_mean, one_probe_mean))}",
        f"1GPU policy rollout mean_s/episode: {_fmt(one_policy_mean)}",
        f"NGPU policy rollout mean_s/episode: {_fmt(many_policy_mean)}",
        f"NGPU/1GPU policy rollout mean ratio: {_fmt(_safe_div(many_policy_mean, one_policy_mean))}",
        f"NGPU worker-local probe noise scopes detected: {scoped_noise_detected}",
        f"NGPU worker-local CUDA probe streams detected: {worker_streams_detected}",
        f"NGPU cpu policy mode detected: {cpu_policy_detected}",
        "",
        "1GPU device breakdown:",
    ]
    for device, values in sorted(one_dev.items()):
        lines.append(f"  {device}: {values}")
    lines.append("NGPU device breakdown:")
    for device, values in sorted(many_dev.items()):
        lines.append(f"  {device}: {values}")
    lines.append("")
    lines.extend(_timing_report_lines("1GPU", one_timing))
    lines.extend(_timing_report_lines("NGPU", many_timing))
    if diffs:
        lines.append("")
        lines.append(f"first {len(diffs)} equality diffs:")
        lines.extend(f"  - {diff}" for diff in diffs)
    if diag_diffs:
        lines.append("")
        lines.append(f"first {len(diag_diffs)} diagnostic-only diffs:")
        lines.extend(f"  - {diff}" for diff in diag_diffs)
    if ppo_diffs:
        lines.append("")
        lines.append(f"first {len(ppo_diffs)} PPO update equality diffs:")
        lines.extend(f"  - {diff}" for diff in ppo_diffs)
    if args.require_equal and (not equality_ok or ppo_equality_ok is False):
        lines.append("")
        lines.append("[FATAL] equality requirement failed")
    if args.min_speedup is not None and speedup is not None:
        if speedup >= float(args.min_speedup):
            lines.append(f"[OK] speedup >= {float(args.min_speedup):.3f}x")
        else:
            lines.append(f"[WARN] speedup < {float(args.min_speedup):.3f}x")
            if args.require_speedup:
                lines.append("[FATAL] speedup requirement failed")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one", required=True, help="1-GPU run dir or episodes.jsonl")
    parser.add_argument("--many", required=True, help="N-GPU run dir or episodes.jsonl")
    parser.add_argument("--one-wall", default=None, help="optional wall seconds file for 1-GPU")
    parser.add_argument("--many-wall", default=None, help="optional wall seconds file for N-GPU")
    parser.add_argument("--one-ppo", default=None, help="optional 1-GPU diagnostics/ppo_updates.jsonl")
    parser.add_argument("--many-ppo", default=None, help="optional N-GPU diagnostics/ppo_updates.jsonl")
    parser.add_argument("--one-log", default=None, help="optional 1-GPU launch log with stage2-rollout-timing lines")
    parser.add_argument("--many-log", default=None, help="optional N-GPU launch log with stage2-rollout-timing lines")
    parser.add_argument("--out", default=None, help="optional verdict output path")
    parser.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for numeric equality")
    parser.add_argument("--max-diffs", type=int, default=20)
    parser.add_argument("--min-speedup", type=float, default=None)
    parser.add_argument("--require-equal", action="store_true")
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="exit nonzero when --min-speedup is set and measured speedup is lower",
    )
    parser.add_argument(
        "--strict-diagnostics",
        action="store_true",
        help="include diagnostic frontier bookkeeping fields in equality verdict",
    )
    args = parser.parse_args()
    report = build_report(args)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(report)
    print(report, end="")
    if args.require_equal and "[FATAL] equality requirement failed" in report:
        return 2
    if args.require_speedup and "[FATAL] speedup requirement failed" in report:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
