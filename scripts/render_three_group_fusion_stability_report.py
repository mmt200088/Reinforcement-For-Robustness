#!/usr/bin/env python3
"""Build a stability report for the three fixed Stage-2 fusion groups."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SEEDS = [20260721, 20261721, 20262721, 20263721, 20264721]
CANONICAL_INSTALL_PATH = (
    "BLBStage2SequentialEnv.evaluate_step -> commit_step -> "
    "BLBStage2Env.step(boosted_overrides)"
)
GROUP_SPECS = {
    "all_fusion0": {
        "label": "Noisy Stage-2 fusion-zero control",
        "pattern": {2: 0, 4: 0, 5: 0},
        "fusion_total": 0,
    },
    "block2_block5_all_layers_fusionmax": {
        "label": "B2/B5 all layers fusion max",
        "pattern": {2: 1, 4: 0, 5: 1},
        "fusion_total": 24,
    },
    "block2_block4_block5_all_layers_fusion1": {
        "label": "B2/B4/B5 all layers fusion 1",
        "pattern": {2: 1, 4: 1, 5: 1},
        "fusion_total": 36,
    },
}
COMPARISON_SPECS = {
    "b2b5_minus_control": (
        "block2_block5_all_layers_fusionmax",
        "all_fusion0",
    ),
    "b2b4b5_minus_control": (
        "block2_block4_block5_all_layers_fusion1",
        "all_fusion0",
    ),
    "b2b4b5_minus_b2b5": (
        "block2_block4_block5_all_layers_fusion1",
        "block2_block5_all_layers_fusionmax",
    ),
}
METRICS = ("loss", "metric1", "metric2")
METRIC_FIELDS = (
    "loss_mean",
    "loss_std",
    "metric1_mean",
    "metric1_std",
    "metric2_mean",
    "metric2_std",
    "loss_max",
    "metric1_min",
    "metric2_min",
)
GRAPH_KEYS = {
    1: "block1_mrpc",
    2: "block2_mrpc",
    4: "block4",
    5: "block5_n4",
}
SCHEMA_VERSION = "three-group-fusion-stability-v1"


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _failure(
    code: str,
    *,
    run_index: int | None = None,
    seed: Any = None,
    group: str | None = None,
    layer: Any = None,
    block: Any = None,
    detail: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code}
    if run_index is not None:
        result["run_index"] = run_index
    if seed is not None:
        result["seed"] = seed
    if group is not None:
        result["group"] = group
    if layer is not None:
        result["layer"] = layer
    if block is not None:
        result["block"] = block
    if detail is not None:
        result["detail"] = detail
    return result


def _gate(name: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "passed": not failures, "failures": failures}


def _run_seed(run: Any) -> Any:
    return run.get("seed") if isinstance(run, Mapping) else None


def _group_map(run: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(run, Mapping):
        return {}
    groups = run.get("group_results")
    if not isinstance(groups, list):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        if isinstance(group, Mapping) and isinstance(group.get("name"), str):
            result.setdefault(group["name"], group)
    return result


def _expected_schedule() -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    for layer in range(12):
        blocks = (2, 4, 5) if layer == 0 else (1, 2, 4, 5)
        schedule.extend((layer, block) for block in blocks)
    return schedule


EXPECTED_SCHEDULE = _expected_schedule()


def _metric_array(group: Any, metric: str) -> list[float]:
    if not isinstance(group, Mapping):
        return []
    trial_metrics = group.get("trial_metrics")
    if not isinstance(trial_metrics, Mapping):
        return []
    values = trial_metrics.get(metric)
    if not isinstance(values, list) or len(values) != 5:
        return []
    if not all(_is_finite_number(value) for value in values):
        return []
    return [float(value) for value in values]


def _finite_or_none(value: Any) -> int | float | None:
    return value if _is_finite_number(value) else None


def _stats(values: Iterable[Any], *, expected_count: int | None = None) -> dict[str, Any]:
    finite_values = [float(value) for value in values if _is_finite_number(value)]
    complete = expected_count is None or len(finite_values) == expected_count
    return {
        "count": len(finite_values),
        "values": finite_values,
        "mean": statistics.fmean(finite_values) if finite_values and complete else None,
        "std": statistics.pstdev(finite_values) if finite_values and complete else None,
    }


def _completeness_gate(run_payloads: Sequence[Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if len(run_payloads) != 5:
        failures.append(
            _failure("run_count", detail=f"expected 5, found {len(run_payloads)}")
        )

    actual_seeds = [_run_seed(run) for run in run_payloads]
    if actual_seeds != EXPECTED_SEEDS:
        failures.append(
            _failure(
                "seed_order",
                detail=f"expected {EXPECTED_SEEDS}, found {actual_seeds}",
            )
        )
    if len(actual_seeds) != len(set(map(str, actual_seeds))):
        failures.append(_failure("duplicate_seeds", detail=str(actual_seeds)))

    expected_names = set(GROUP_SPECS)
    expected_top_level = {
        "repeat": 5,
        "probe_size": 408,
        "stage1_gelu": [4] * 12,
        "stage1_softmax": [6] * 12,
        "shared_group_seed": True,
        "install_path": CANONICAL_INSTALL_PATH,
    }
    for run_index, run in enumerate(run_payloads):
        seed = _run_seed(run)
        if not isinstance(run, Mapping):
            failures.append(
                _failure("run_mapping", run_index=run_index, detail=type(run).__name__)
            )
            continue
        for field, expected in expected_top_level.items():
            if run.get(field) != expected:
                failures.append(
                    _failure(
                        "protocol_field",
                        run_index=run_index,
                        seed=seed,
                        detail=f"{field}: expected {expected!r}, found {run.get(field)!r}",
                    )
                )

        groups = run.get("group_results")
        if not isinstance(groups, list):
            failures.append(
                _failure("group_results_type", run_index=run_index, seed=seed)
            )
            continue
        names = [
            group.get("name")
            for group in groups
            if isinstance(group, Mapping) and isinstance(group.get("name"), str)
        ]
        if len(groups) != 3 or len(names) != 3 or set(names) != expected_names:
            failures.append(
                _failure(
                    "required_groups",
                    run_index=run_index,
                    seed=seed,
                    detail=f"expected {sorted(expected_names)}, found {names}",
                )
            )
        if len(names) != len(set(names)):
            failures.append(
                _failure("duplicate_groups", run_index=run_index, seed=seed, detail=str(names))
            )

        for group_name, group in _group_map(run).items():
            if group_name not in expected_names:
                continue
            trial_metrics = group.get("trial_metrics")
            for metric in METRICS:
                values = trial_metrics.get(metric) if isinstance(trial_metrics, Mapping) else None
                if not isinstance(values, list) or len(values) != 5:
                    failures.append(
                        _failure(
                            "trial_count",
                            run_index=run_index,
                            seed=seed,
                            group=group_name,
                            detail=f"{metric}: expected list length 5",
                        )
                    )
                    continue
                for trial_index, value in enumerate(values):
                    if not _is_finite_number(value):
                        failures.append(
                            _failure(
                                "non_finite",
                                run_index=run_index,
                                seed=seed,
                                group=group_name,
                                detail=f"{metric}[{trial_index}]={value!r}",
                            )
                        )
            metrics = group.get("metrics")
            if not isinstance(metrics, Mapping):
                failures.append(
                    _failure(
                        "metrics_mapping",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                    )
                )
            else:
                for field in METRIC_FIELDS:
                    if not _is_finite_number(metrics.get(field)):
                        failures.append(
                            _failure(
                                "non_finite",
                                run_index=run_index,
                                seed=seed,
                                group=group_name,
                                detail=f"metrics.{field}={metrics.get(field)!r}",
                            )
                        )
    return _gate("completeness", failures)


def _trial_metadata_gate(run_payloads: Sequence[Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    expected_indices = [list(range(5))]
    for run_index, run in enumerate(run_payloads):
        seed = _run_seed(run)
        group_seeds: list[Any] = []
        for group_name in GROUP_SPECS:
            group = _group_map(run).get(group_name)
            if group is None:
                continue
            probe = group.get("terminal_probe")
            if not isinstance(probe, Mapping):
                failures.append(
                    _failure(
                        "terminal_probe",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                        detail="missing mapping",
                    )
                )
                continue
            expected_fields = {
                "k": 5,
                "deterministic_probe_seed": seed,
                "per_worker_trial_indices": expected_indices,
            }
            for field, expected in expected_fields.items():
                if probe.get(field) != expected:
                    failures.append(
                        _failure(
                            "trial_metadata",
                            run_index=run_index,
                            seed=seed,
                            group=group_name,
                            detail=f"{field}: expected {expected!r}, found {probe.get(field)!r}",
                        )
                    )
            trial_seeds = probe.get("per_worker_trial_seeds")
            if (
                not isinstance(trial_seeds, list)
                or len(trial_seeds) != 1
                or not isinstance(trial_seeds[0], list)
                or len(trial_seeds[0]) != 5
            ):
                failures.append(
                    _failure(
                        "trial_seed_shape",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                    )
                )
            else:
                group_seeds.append(trial_seeds)
        if group_seeds and any(seeds != group_seeds[0] for seeds in group_seeds[1:]):
            failures.append(
                _failure(
                    "shared_trial_seeds",
                    run_index=run_index,
                    seed=seed,
                    detail="group trial seed lists differ",
                )
            )
    return _gate("trial_metadata", failures)


def _steps_install_gate(run_payloads: Sequence[Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for run_index, run in enumerate(run_payloads):
        seed = _run_seed(run)
        if isinstance(run, Mapping) and run.get("install_path") != CANONICAL_INSTALL_PATH:
            failures.append(
                _failure(
                    "install_path",
                    run_index=run_index,
                    seed=seed,
                    detail=str(run.get("install_path")),
                )
            )
        for group_name in GROUP_SPECS:
            group = _group_map(run).get(group_name)
            if group is None:
                continue
            if group.get("k_distribution") != {"13": 47}:
                failures.append(
                    _failure(
                        "k_distribution",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                        detail=str(group.get("k_distribution")),
                    )
                )
            if group.get("block5_graph_counts") != {"block5_n4": 12}:
                failures.append(
                    _failure(
                        "block5_graph_counts",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                        detail=str(group.get("block5_graph_counts")),
                    )
                )
            records = group.get("step_records")
            if not isinstance(records, list) or len(records) != 47:
                failures.append(
                    _failure(
                        "step_count",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                        detail=f"expected 47, found {len(records) if isinstance(records, list) else 'non-list'}",
                    )
                )
                continue
            for record_index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    failures.append(
                        _failure(
                            "step_mapping",
                            run_index=run_index,
                            seed=seed,
                            group=group_name,
                            detail=f"record {record_index}",
                        )
                    )
                    continue
                layer = record.get("layer_idx")
                block = record.get("block_idx")
                checks = {
                    "valid": record.get("valid") is True,
                    "k_value": record.get("k_value") == 13,
                    "k_index": record.get("k_index") == 3,
                    "model_uses_replan_config": record.get("model_uses_replan_config") is True,
                }
                application = record.get("replan_application")
                checks["applied_before_forward"] = (
                    isinstance(application, Mapping)
                    and application.get("applied_before_forward") is True
                )
                checks["nested_model_uses_replan_config"] = (
                    isinstance(application, Mapping)
                    and application.get("model_uses_replan_config") is True
                )
                for field, passed in checks.items():
                    if not passed:
                        failures.append(
                            _failure(
                                "step_evidence",
                                run_index=run_index,
                                seed=seed,
                                group=group_name,
                                layer=layer,
                                block=block,
                                detail=field,
                            )
                        )
    return _gate("steps_install", failures)


def _fusion_pattern_gate(run_payloads: Sequence[Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    expected_schedule_set = set(EXPECTED_SCHEDULE)
    for run_index, run in enumerate(run_payloads):
        seed = _run_seed(run)
        for group_name, spec in GROUP_SPECS.items():
            group = _group_map(run).get(group_name)
            if group is None:
                continue
            pattern = spec["pattern"]
            expected_by_block = {
                "1": 0,
                "2": pattern[2] * 12,
                "4": pattern[4] * 12,
                "5": pattern[5] * 12,
            }
            if group.get("fusion_by_block") != expected_by_block:
                failures.append(
                    _failure(
                        "fusion_by_block",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                        detail=f"expected {expected_by_block}, found {group.get('fusion_by_block')!r}",
                    )
                )
            if group.get("fusion_total") != spec["fusion_total"]:
                failures.append(
                    _failure(
                        "fusion_total",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                        detail=f"expected {spec['fusion_total']}, found {group.get('fusion_total')!r}",
                    )
                )
            records = group.get("step_records")
            if not isinstance(records, list):
                continue
            positions: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
            for record in records:
                if isinstance(record, Mapping):
                    position = (record.get("layer_idx"), record.get("block_idx"))
                    positions.setdefault(position, []).append(record)
            if set(positions) != expected_schedule_set or any(
                len(records_at_position) != 1 for records_at_position in positions.values()
            ):
                failures.append(
                    _failure(
                        "step_schedule",
                        run_index=run_index,
                        seed=seed,
                        group=group_name,
                        detail="expected one record at each of 47 layer/block positions",
                    )
                )
            for layer, block in EXPECTED_SCHEDULE:
                matching = positions.get((layer, block), [])
                if len(matching) != 1:
                    continue
                record = matching[0]
                expected_fusion = 0 if block == 1 else pattern[block]
                checks = {
                    "graph_key": (record.get("graph_key"), GRAPH_KEYS[block]),
                    "fusion_count_replan": (
                        record.get("fusion_count_replan"),
                        expected_fusion,
                    ),
                    "boosted": (record.get("boosted"), bool(expected_fusion)),
                }
                for field, (actual, expected) in checks.items():
                    if actual != expected or (
                        field == "boosted" and not isinstance(actual, bool)
                    ):
                        failures.append(
                            _failure(
                                "fusion_pattern",
                                run_index=run_index,
                                seed=seed,
                                group=group_name,
                                layer=layer,
                                block=block,
                                detail=f"{field}: expected {expected!r}, found {actual!r}",
                            )
                        )
    return _gate("fusion_pattern", failures)


def _per_run_summary(
    run_payloads: Sequence[Any], group_name: str
) -> list[dict[str, Any]]:
    per_runs: list[dict[str, Any]] = []
    for run_index, run in enumerate(run_payloads):
        group = _group_map(run).get(group_name)
        if group is None:
            per_runs.append(
                {
                    "run_index": run_index,
                    "seed": _run_seed(run),
                    "missing": True,
                    "metrics": {},
                    "trial_seeds": [],
                    "evidence": {},
                }
            )
            continue
        metrics = group.get("metrics")
        safe_metrics = {
            field: _finite_or_none(metrics.get(field))
            for field in METRIC_FIELDS
        } if isinstance(metrics, Mapping) else {}
        probe = group.get("terminal_probe")
        trial_seeds = (
            probe.get("per_worker_trial_seeds", [])
            if isinstance(probe, Mapping)
            else []
        )
        per_runs.append(
            {
                "run_index": run_index,
                "seed": _run_seed(run),
                "missing": False,
                "metrics": safe_metrics,
                "trial_seeds": trial_seeds,
                "evidence": {
                    "fusion_total": group.get("fusion_total"),
                    "fusion_by_block": group.get("fusion_by_block"),
                    "k_distribution": group.get("k_distribution"),
                    "block5_graph_counts": group.get("block5_graph_counts"),
                    "step_count": len(group.get("step_records", []))
                    if isinstance(group.get("step_records"), list)
                    else None,
                },
            }
        )
    return per_runs


def _group_summary(run_payloads: Sequence[Any], group_name: str) -> dict[str, Any]:
    spec = GROUP_SPECS[group_name]
    pooled_metrics: dict[str, Any] = {}
    run_mean_std: dict[str, Any] = {}
    for metric in METRICS:
        pooled = [
            value
            for run in run_payloads
            for value in _metric_array(_group_map(run).get(group_name), metric)
        ]
        pooled_metrics[metric] = _stats(pooled, expected_count=25)
        run_means: list[Any] = []
        for run in run_payloads:
            group = _group_map(run).get(group_name)
            metrics = group.get("metrics") if isinstance(group, Mapping) else None
            run_means.append(
                metrics.get(f"{metric}_mean") if isinstance(metrics, Mapping) else None
            )
        run_mean_std[metric] = _stats(run_means, expected_count=5)
    return {
        "label": spec["label"],
        "definition": {
            "B2": spec["pattern"][2],
            "B4": spec["pattern"][4],
            "B5": spec["pattern"][5],
            "K": 13,
        },
        "fusion_total": spec["fusion_total"],
        "fusion_by_block": {
            "1": 0,
            "2": spec["pattern"][2] * 12,
            "4": spec["pattern"][4] * 12,
            "5": spec["pattern"][5] * 12,
        },
        "per_runs": _per_run_summary(run_payloads, group_name),
        "pooled_metrics": pooled_metrics,
        "run_mean_std": run_mean_std,
    }


def _comparison_summary(
    run_payloads: Sequence[Any], treatment_name: str, reference_name: str
) -> dict[str, Any]:
    paired_deltas: dict[str, Any] = {}
    run_mean_deltas: dict[str, Any] = {}
    better_run_count: dict[str, int] = {}
    for metric in METRICS:
        raw_deltas: list[float] = []
        mean_deltas: list[float] = []
        for run in run_payloads:
            groups = _group_map(run)
            treatment = _metric_array(groups.get(treatment_name), metric)
            reference = _metric_array(groups.get(reference_name), metric)
            if len(treatment) == 5 and len(reference) == 5:
                raw_deltas.extend(
                    treatment_value - reference_value
                    for treatment_value, reference_value in zip(treatment, reference)
                )
                mean_deltas.append(
                    statistics.fmean(treatment) - statistics.fmean(reference)
                )
        paired_deltas[metric] = _stats(raw_deltas, expected_count=25)
        run_mean_deltas[metric] = _stats(mean_deltas, expected_count=5)
        better_run_count[metric] = sum(
            delta < 0 if metric == "loss" else delta > 0 for delta in mean_deltas
        )
    return {
        "treatment": treatment_name,
        "reference": reference_name,
        "paired_deltas": paired_deltas,
        "run_mean_deltas": run_mean_deltas,
        "better_run_count": better_run_count,
    }


def build_summary(*, run_payloads: Sequence[Any], source_commit: str) -> dict[str, Any]:
    """Validate evaluator payloads and aggregate paired stability statistics."""
    payloads = list(run_payloads) if isinstance(run_payloads, Sequence) else []
    gates = [
        _completeness_gate(payloads),
        _trial_metadata_gate(payloads),
        _steps_install_gate(payloads),
        _fusion_pattern_gate(payloads),
    ]
    groups = {
        group_name: _group_summary(payloads, group_name)
        for group_name in GROUP_SPECS
    }
    comparisons = {
        comparison_name: _comparison_summary(payloads, treatment, reference)
        for comparison_name, (treatment, reference) in COMPARISON_SPECS.items()
    }
    return {
        "source_commit": source_commit,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "stage1_gelu": [4] * 12,
            "stage1_softmax": [6] * 12,
        },
        "protocol": {
            "repeat": 5,
            "probe_size": 408,
            "shared_group_seed": True,
            "install_path": CANONICAL_INSTALL_PATH,
            "K": 13,
            "step_count": 47,
            "trials_per_run": 5,
        },
        "seeds": list(EXPECTED_SEEDS),
        "total_evaluations": 75,
        "groups": groups,
        "comparisons": comparisons,
        "gates": gates,
        "all_gates_pass": all(gate["passed"] for gate in gates),
    }


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _number(value: Any, digits: int = 8) -> str:
    if not _is_finite_number(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _mean_std(stats: Mapping[str, Any]) -> str:
    return f"{_number(stats.get('mean'))} +/- {_number(stats.get('std'))}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th>{_h(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_h(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(summary: Mapping[str, Any]) -> str:
    """Render a self-contained diagnostic HTML report."""
    protocol = summary.get("protocol", {})
    groups = summary.get("groups", {})
    comparisons = summary.get("comparisons", {})
    gates = summary.get("gates", [])
    status = "PASS" if summary.get("all_gates_pass") else "FAIL"

    protocol_rows = [
        ("Source commit", summary.get("source_commit", "")),
        ("Seeds", ", ".join(str(seed) for seed in summary.get("seeds", []))),
        ("Total evaluations", summary.get("total_evaluations", 0)),
        ("Probe size", protocol.get("probe_size", "n/a")),
        ("Repeat", protocol.get("repeat", "n/a")),
        ("K", f"K={protocol.get('K', 'n/a')}"),
        ("Install path", protocol.get("install_path", "")),
        ("Control", "Noisy Stage-2 fusion-zero control (not a plaintext baseline)"),
    ]

    group_sections: list[str] = []
    for group_name in GROUP_SPECS:
        group = groups.get(group_name, {}) if isinstance(groups, Mapping) else {}
        definition = group.get("definition", {})
        definition_text = ", ".join(
            (
                f"B2={definition.get('B2', 'n/a')}",
                f"B4={definition.get('B4', 'n/a')}",
                f"B5={definition.get('B5', 'n/a')}",
                f"K={definition.get('K', 'n/a')}",
            )
        )
        pooled = group.get("pooled_metrics", {})
        pooled_rows = [
            (
                metric,
                pooled.get(metric, {}).get("count", 0),
                _mean_std(pooled.get(metric, {})),
            )
            for metric in METRICS
        ]
        per_run_rows = []
        for run in group.get("per_runs", []):
            metrics = run.get("metrics", {})
            per_run_rows.append(
                (
                    run.get("seed", "n/a"),
                    _number(metrics.get("loss_mean")),
                    _number(metrics.get("loss_std")),
                    _number(metrics.get("loss_max")),
                    _number(metrics.get("metric1_mean")),
                    _number(metrics.get("metric1_std")),
                    _number(metrics.get("metric1_min")),
                    _number(metrics.get("metric2_mean")),
                    _number(metrics.get("metric2_std")),
                    _number(metrics.get("metric2_min")),
                )
            )
        group_sections.append(
            f"<section><h2>{_h(group_name)}</h2>"
            f"<p><strong>{_h(group.get('label', ''))}</strong><br>{_h(definition_text)}; "
            f"fusion total={_h(group.get('fusion_total', 'n/a'))}</p>"
            "<h3>Pooled metrics</h3>"
            + _table(("Metric", "Count", "mean +/- std"), pooled_rows)
            + "<h3>Per-run metrics and extrema</h3>"
            + _table(
                (
                    "Seed",
                    "Loss mean",
                    "Loss std",
                    "Loss max",
                    "Metric1 mean",
                    "Metric1 std",
                    "Metric1 min",
                    "Metric2 mean",
                    "Metric2 std",
                    "Metric2 min",
                ),
                per_run_rows,
            )
            + "</section>"
        )

    decision_rows = []
    for layer in range(12):
        for group_name, spec in GROUP_SPECS.items():
            decision_rows.append(
                (
                    layer,
                    group_name,
                    f"B2={spec['pattern'][2]}",
                    f"B4={spec['pattern'][4]}",
                    f"B5={spec['pattern'][5]}",
                    "K=13",
                )
            )

    comparison_rows = []
    for comparison_name in COMPARISON_SPECS:
        comparison = (
            comparisons.get(comparison_name, {})
            if isinstance(comparisons, Mapping)
            else {}
        )
        for metric in METRICS:
            paired = comparison.get("paired_deltas", {}).get(metric, {})
            run_delta = comparison.get("run_mean_deltas", {}).get(metric, {})
            comparison_rows.append(
                (
                    comparison_name,
                    metric,
                    paired.get("count", 0),
                    _mean_std(paired),
                    _mean_std(run_delta),
                    comparison.get("better_run_count", {}).get(metric, 0),
                )
            )

    gate_rows = []
    for gate in gates if isinstance(gates, list) else []:
        failures = gate.get("failures", []) if isinstance(gate, Mapping) else []
        failure_text = "none" if not failures else json.dumps(failures, sort_keys=True)
        gate_rows.append(
            (
                gate.get("name", "unknown"),
                "PASS" if gate.get("passed") else "FAIL",
                failure_text,
            )
        )

    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Three-group fusion stability report</title>
<style>
body { font-family: system-ui, sans-serif; color: #202428; margin: 0; background: #f4f5f6; }
main { max-width: 1240px; margin: 0 auto; padding: 28px; background: white; }
h1, h2, h3 { letter-spacing: 0; }
h1 { margin: 0 0 8px; font-size: 28px; }
h2 { margin-top: 32px; border-bottom: 2px solid #d7dadd; padding-bottom: 6px; }
h3 { font-size: 16px; margin-top: 22px; }
.status { font-weight: 700; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 20px; font-size: 13px; }
th, td { border: 1px solid #d7dadd; padding: 7px 8px; text-align: left; vertical-align: top; }
th { background: #eef0f1; }
tbody tr:nth-child(even) { background: #fafafa; }
td:last-child { overflow-wrap: anywhere; }
</style>
</head>
<body><main>""" + (
        f"<h1>Three-group fusion stability report</h1>"
        f"<p class=\"status\">Gate status: {_h(status)}</p>"
        "<h2>Protocol</h2>"
        + _table(("Field", "Value"), protocol_rows)
        + "".join(group_sections)
        + "<section><h2>12-layer fusion decisions</h2>"
        + _table(("Layer", "Group", "B2", "B4", "B5", "K"), decision_rows)
        + "</section><section><h2>Paired comparisons</h2>"
        + _table(
            (
                "Comparison",
                "Metric",
                "Paired count",
                "Paired mean +/- std",
                "Run-mean delta +/- std",
                "Better run count",
            ),
            comparison_rows,
        )
        + "</section><section><h2>Validation gates</h2>"
        + _table(("Gate", "Status", "Failures"), gate_rows)
        + "</section></main></body></html>"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the three-group Stage-2 fusion stability report."
    )
    parser.add_argument("--run-json", action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payloads: list[Any] = []
    for path_text in args.run_json:
        path = Path(path_text)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payloads.append(json.load(handle))
        except (OSError, json.JSONDecodeError) as error:
            payloads.append(
                {
                    "_load_error": f"{type(error).__name__}: {error}",
                    "_source_path": str(path),
                }
            )

    summary = build_summary(run_payloads=payloads, source_commit=args.source_commit)
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    with output_html.open("w", encoding="utf-8") as handle:
        handle.write(render_html(summary))
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
