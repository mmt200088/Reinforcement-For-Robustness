#!/usr/bin/env python3
"""Validate and render the three-group per-example MRPC logits report."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.render_three_group_fusion_stability_report import (
    EXPECTED_SEEDS,
    GROUP_SPECS,
    METRICS,
    build_summary,
)


SCHEMA_VERSION = "three-group-per-example-logits-v1"
PREDICTION_ROW_SCHEMA = "fusion-count-per-example-v1"
TRIAL_COUNT = 5
PRODUCTION_EXAMPLES = 408


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _failure(code: str, **context: Any) -> dict[str, Any]:
    result = {"code": code}
    result.update({key: value for key, value in context.items() if value is not None})
    return result


def _gate(name: str, failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {"name": name, "passed": not failures, "failures": failures}


def _group_map(run: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(run, Mapping) or not isinstance(run.get("group_results"), list):
        return {}
    return {
        group["name"]: group
        for group in run["group_results"]
        if isinstance(group, Mapping) and isinstance(group.get("name"), str)
    }


def _runs_by_seed(payloads: Sequence[Any]) -> dict[Any, Mapping[str, Any]]:
    return {
        run.get("seed"): run
        for run in payloads
        if isinstance(run, Mapping) and "seed" in run
    }


def _trial_seeds(group: Any) -> list[Any]:
    if not isinstance(group, Mapping):
        return []
    probe = group.get("terminal_probe")
    workers = probe.get("per_worker_trial_seeds") if isinstance(probe, Mapping) else None
    if (
        not isinstance(workers, list)
        or len(workers) != 1
        or not isinstance(workers[0], list)
    ):
        return []
    return list(workers[0])


def _metric_values(group: Any, metric: str) -> list[Any]:
    trial_metrics = group.get("trial_metrics") if isinstance(group, Mapping) else None
    values = trial_metrics.get(metric) if isinstance(trial_metrics, Mapping) else None
    return list(values) if isinstance(values, list) else []


def _row_context(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "seed": row.get("run_seed"),
        "group": row.get("group"),
        "trial_index": row.get("trial_index"),
        "dataset_idx": row.get("dataset_idx"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "positive_infinity"
        else:
            label = "negative_infinity"
        return {"non_finite": label}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def stable_cross_entropy(logits: Sequence[float], gold_label: int) -> float:
    """Return two-class cross entropy without overflowing exponentials."""
    values = [float(value) for value in logits]
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values)) - values[gold_label]


def _weighted_f1(gold: Sequence[int], predicted: Sequence[int]) -> float:
    total = len(gold)
    if total == 0:
        return 0.0
    weighted = 0.0
    for label in (0, 1):
        support = sum(actual == label for actual in gold)
        if support == 0:
            continue
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(gold, predicted)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(gold, predicted)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 0.0 if denominator == 0 else (2 * true_positive) / denominator
        weighted += support * f1
    return weighted / total


def _base_gate(base_summary: Mapping[str, Any]) -> dict[str, Any]:
    failures = []
    for gate in base_summary.get("gates", []):
        if isinstance(gate, Mapping) and not gate.get("passed"):
            failures.append(
                _failure(
                    "base_gate_failed",
                    base_gate=gate.get("name"),
                    failures=gate.get("failures", []),
                )
            )
    if not base_summary.get("all_gates_pass") and not failures:
        failures.append(_failure("base_summary_failed"))
    return _gate("base_three_group", failures)


def _index_rows(
    prediction_rows: Sequence[Any],
) -> tuple[
    dict[tuple[Any, str, int], list[tuple[int, Mapping[str, Any]]]],
    list[dict[str, Any]],
]:
    buckets: dict[tuple[Any, str, int], list[tuple[int, Mapping[str, Any]]]] = {}
    failures: list[dict[str, Any]] = []
    expected_seed_set = set(EXPECTED_SEEDS)
    expected_groups = set(GROUP_SPECS)
    for row_index, row in enumerate(prediction_rows):
        if not isinstance(row, Mapping):
            failures.append(
                _failure("row_mapping", row_index=row_index, detail=type(row).__name__)
            )
            continue
        context = _row_context(row, row_index)
        if row.get("schema_version") != PREDICTION_ROW_SCHEMA:
            failures.append(
                _failure(
                    "prediction_schema",
                    **context,
                    detail=f"expected {PREDICTION_ROW_SCHEMA!r}",
                )
            )
        seed = row.get("run_seed")
        group = row.get("group")
        trial_index = row.get("trial_index")
        valid_trial_index = (
            isinstance(trial_index, int)
            and not isinstance(trial_index, bool)
            and 0 <= trial_index < TRIAL_COUNT
        )
        if seed not in expected_seed_set:
            failures.append(_failure("unexpected_run_seed", **context))
        if group not in expected_groups:
            failures.append(_failure("unexpected_group", **context))
        if not valid_trial_index:
            failures.append(_failure("unexpected_trial_index", **context))
        if seed in expected_seed_set and group in expected_groups and valid_trial_index:
            buckets.setdefault((seed, group, trial_index), []).append((row_index, row))
    return buckets, failures


def _prediction_completeness_gate(
    prediction_rows: Sequence[Any],
    buckets: Mapping[tuple[Any, str, int], Sequence[Any]],
    index_failures: Sequence[dict[str, Any]],
    *,
    expected_examples: int,
    prediction_file_count: int,
    runs_by_seed: Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    failures = list(index_failures)
    expected_rows = 5 * len(GROUP_SPECS) * TRIAL_COUNT * expected_examples
    if prediction_file_count != 5:
        failures.append(
            _failure(
                "prediction_file_count",
                detail=f"expected 5, found {prediction_file_count}",
            )
        )
    if len(prediction_rows) != expected_rows:
        failures.append(
            _failure(
                "prediction_row_count",
                detail=f"expected {expected_rows}, found {len(prediction_rows)}",
            )
        )
    for seed in EXPECTED_SEEDS:
        run = runs_by_seed.get(seed)
        groups = _group_map(run)
        for group_name in GROUP_SPECS:
            seeds = _trial_seeds(groups.get(group_name))
            for trial_index in range(TRIAL_COUNT):
                trial_rows = buckets.get((seed, group_name, trial_index), [])
                if len(trial_rows) != expected_examples:
                    failures.append(
                        _failure(
                            "trial_row_count",
                            seed=seed,
                            group=group_name,
                            trial_index=trial_index,
                            detail=f"expected {expected_examples}, found {len(trial_rows)}",
                        )
                    )
                expected_seed = seeds[trial_index] if len(seeds) == TRIAL_COUNT else None
                for row_index, row in trial_rows:
                    if row.get("trial_seed") != expected_seed:
                        failures.append(
                            _failure(
                                "trial_seed_mismatch",
                                **_row_context(row, row_index),
                                detail=f"expected {expected_seed!r}, found {row.get('trial_seed')!r}",
                            )
                        )
    return _gate("prediction_completeness", failures)


def _identity_gate(
    buckets: Mapping[
        tuple[Any, str, int], Sequence[tuple[int, Mapping[str, Any]]]
    ],
    *,
    expected_examples: int,
) -> tuple[dict[str, Any], dict[Any, dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    canonical: dict[Any, dict[str, Any]] = {}
    expected_ids: set[Any] | None = None
    for seed in EXPECTED_SEEDS:
        for group_name in GROUP_SPECS:
            for trial_index in range(TRIAL_COUNT):
                trial_rows = buckets.get((seed, group_name, trial_index), [])
                seen: set[Any] = set()
                for row_index, row in trial_rows:
                    context = _row_context(row, row_index)
                    dataset_idx = row.get("dataset_idx")
                    if (
                        not isinstance(dataset_idx, int)
                        or isinstance(dataset_idx, bool)
                    ):
                        failures.append(_failure("dataset_idx_type", **context))
                        continue
                    if dataset_idx in seen:
                        failures.append(_failure("duplicate_dataset_idx", **context))
                    seen.add(dataset_idx)
                    identity = {
                        "input_ids": row.get("input_ids"),
                        "attention_mask": row.get("attention_mask"),
                        "token_type_ids": row.get("token_type_ids"),
                        "gold_label": row.get("gold_label"),
                    }
                    tensors_valid = all(
                        isinstance(identity[field], list)
                        for field in ("input_ids", "attention_mask")
                    ) and (
                        identity["token_type_ids"] is None
                        or isinstance(identity["token_type_ids"], list)
                    )
                    if not tensors_valid:
                        failures.append(_failure("input_tensor_type", **context))
                    if identity["gold_label"] not in (0, 1) or isinstance(
                        identity["gold_label"], bool
                    ):
                        failures.append(_failure("gold_label", **context))
                    previous = canonical.setdefault(dataset_idx, identity)
                    if previous != identity:
                        failures.append(_failure("unstable_input_identity", **context))
                if len(seen) == expected_examples:
                    if expected_ids is None:
                        expected_ids = seen
                    elif seen != expected_ids:
                        failures.append(
                            _failure(
                                "dataset_idx_set",
                                seed=seed,
                                group=group_name,
                                trial_index=trial_index,
                                detail="dataset IDs differ from the first complete trial",
                            )
                        )
                elif seen:
                    failures.append(
                        _failure(
                            "dataset_idx_count",
                            seed=seed,
                            group=group_name,
                            trial_index=trial_index,
                            detail=f"expected {expected_examples} unique IDs, found {len(seen)}",
                        )
                    )
    return _gate("input_identity", failures), canonical


def _logits_gate(
    prediction_rows: Sequence[Any],
) -> tuple[dict[str, Any], set[int]]:
    failures: list[dict[str, Any]] = []
    valid_row_indexes: set[int] = set()
    for row_index, row in enumerate(prediction_rows):
        if not isinstance(row, Mapping):
            continue
        context = _row_context(row, row_index)
        logits = row.get("logits")
        if not isinstance(logits, list) or len(logits) != 2:
            failures.append(_failure("logit_count", **context))
            continue
        if not all(_finite_number(value) for value in logits):
            failures.append(_failure("non_finite_logits", **context))
            continue
        predicted = row.get("predicted_label")
        expected_prediction = 0 if logits[0] >= logits[1] else 1
        if predicted != expected_prediction or isinstance(predicted, bool):
            failures.append(
                _failure(
                    "prediction_argmax",
                    **context,
                    detail=f"expected {expected_prediction}, found {predicted!r}",
                )
            )
            continue
        gold = row.get("gold_label")
        correct = row.get("correct")
        expected_correct = predicted == gold
        if not isinstance(correct, bool) or correct != expected_correct:
            failures.append(
                _failure(
                    "prediction_correctness",
                    **context,
                    detail=f"expected {expected_correct}, found {correct!r}",
                )
            )
            continue
        if gold not in (0, 1) or isinstance(gold, bool):
            failures.append(_failure("gold_label", **context))
            continue
        valid_row_indexes.add(row_index)
    return _gate("logits_prediction", failures), valid_row_indexes


def _trial_result(
    seed: Any,
    group_name: str,
    trial_index: int,
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    valid_row_indexes: set[int],
    expected_examples: int,
    raw_group: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    valid_rows = [row for row_index, row in rows if row_index in valid_row_indexes]
    result = {
        "seed": seed,
        "group": group_name,
        "trial_index": trial_index,
        "trial_seed": valid_rows[0].get("trial_seed") if valid_rows else None,
        "correct_count": sum(row.get("correct") is True for row in valid_rows),
        "incorrect_count": sum(row.get("correct") is False for row in valid_rows),
        "correct_dataset_indices": sorted(
            row["dataset_idx"]
            for row in valid_rows
            if row.get("correct") is True and isinstance(row.get("dataset_idx"), int)
        ),
        "incorrect_dataset_indices": sorted(
            row["dataset_idx"]
            for row in valid_rows
            if row.get("correct") is False and isinstance(row.get("dataset_idx"), int)
        ),
        "recomputed_loss": None,
        "recomputed_accuracy": None,
        "recomputed_weighted_f1": None,
    }
    if len(valid_rows) != expected_examples:
        return result, failures
    gold = [int(row["gold_label"]) for row in valid_rows]
    predicted = [int(row["predicted_label"]) for row in valid_rows]
    loss = statistics.fmean(
        stable_cross_entropy(row["logits"], int(row["gold_label"]))
        for row in valid_rows
    )
    accuracy = sum(actual == guess for actual, guess in zip(gold, predicted)) / len(gold)
    weighted_f1 = _weighted_f1(gold, predicted)
    result.update(
        {
            "recomputed_loss": loss,
            "recomputed_accuracy": accuracy,
            "recomputed_weighted_f1": weighted_f1,
        }
    )
    comparisons = (
        ("loss", loss, 1e-6),
        ("metric1", accuracy, 1e-12),
        ("metric2", weighted_f1, 1e-12),
    )
    for metric, recomputed, tolerance in comparisons:
        raw_values = _metric_values(raw_group, metric)
        raw = raw_values[trial_index] if len(raw_values) == TRIAL_COUNT else None
        if not _finite_number(raw) or not math.isclose(
            float(raw), recomputed, rel_tol=0.0, abs_tol=tolerance
        ):
            failures.append(
                _failure(
                    "recomputed_metric_mismatch",
                    seed=seed,
                    group=group_name,
                    trial_index=trial_index,
                    metric=metric,
                    detail=f"raw={raw!r}, recomputed={recomputed!r}, abs_tol={tolerance}",
                )
            )
    return result, failures


def _shared_trial_seeds_gate(run_payloads: Sequence[Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for run_index, run in enumerate(run_payloads):
        groups = _group_map(run)
        streams = {name: _trial_seeds(groups.get(name)) for name in GROUP_SPECS}
        reference = streams.get(next(iter(GROUP_SPECS)), [])
        if len(reference) != TRIAL_COUNT:
            failures.append(
                _failure(
                    "trial_seed_count",
                    run_index=run_index,
                    seed=run.get("seed") if isinstance(run, Mapping) else None,
                    detail=f"expected {TRIAL_COUNT}, found {len(reference)}",
                )
            )
        for group_name, stream in streams.items():
            if stream != reference:
                failures.append(
                    _failure(
                        "shared_trial_seed_mismatch",
                        run_index=run_index,
                        seed=run.get("seed") if isinstance(run, Mapping) else None,
                        group=group_name,
                        detail=f"expected {reference!r}, found {stream!r}",
                    )
                )
    return _gate("shared_trial_seeds", failures)


def _prior_equivalence_gate(
    run_payloads: Sequence[Any], prior_run_payloads: Sequence[Any]
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    if len(prior_run_payloads) != 5:
        failures.append(
            _failure(
                "prior_run_count",
                detail=f"expected 5, found {len(prior_run_payloads)}",
            )
        )
    current_by_seed = _runs_by_seed(run_payloads)
    prior_by_seed = _runs_by_seed(prior_run_payloads)
    for seed in EXPECTED_SEEDS:
        current_groups = _group_map(current_by_seed.get(seed))
        prior_groups = _group_map(prior_by_seed.get(seed))
        for group_name in GROUP_SPECS:
            for metric in METRICS:
                current = _metric_values(current_groups.get(group_name), metric)
                prior = _metric_values(prior_groups.get(group_name), metric)
                for trial_index in range(TRIAL_COUNT):
                    current_value = current[trial_index] if len(current) == TRIAL_COUNT else None
                    prior_value = prior[trial_index] if len(prior) == TRIAL_COUNT else None
                    if not (
                        _finite_number(current_value)
                        and _finite_number(prior_value)
                        and math.isclose(
                            float(current_value),
                            float(prior_value),
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                    ):
                        failures.append(
                            _failure(
                                "prior_trial_metric_mismatch",
                                seed=seed,
                                group=group_name,
                                trial_index=trial_index,
                                metric=metric,
                                detail=f"current={current_value!r}, prior={prior_value!r}",
                            )
                        )
    return _gate("prior_equivalence", failures)


def _input_aggregates(
    prediction_rows: Sequence[Any],
    canonical: Mapping[Any, Mapping[str, Any]],
    valid_row_indexes: set[int],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[Any, list[Mapping[str, Any]]]] = {
        group_name: {} for group_name in GROUP_SPECS
    }
    for row_index, row in enumerate(prediction_rows):
        if row_index not in valid_row_indexes or not isinstance(row, Mapping):
            continue
        group_name = row.get("group")
        dataset_idx = row.get("dataset_idx")
        if group_name in grouped and isinstance(dataset_idx, int):
            grouped[group_name].setdefault(dataset_idx, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for group_name in GROUP_SPECS:
        inputs: dict[str, Any] = {}
        for dataset_idx in sorted(grouped[group_name]):
            rows = grouped[group_name][dataset_idx]
            logits0 = [float(row["logits"][0]) for row in rows]
            logits1 = [float(row["logits"][1]) for row in rows]
            identity = canonical.get(dataset_idx, {})
            correct_count = sum(row["correct"] is True for row in rows)
            inputs[str(dataset_idx)] = {
                "dataset_idx": dataset_idx,
                "trial_count": len(rows),
                "correct_count": correct_count,
                "correct_rate": correct_count / len(rows),
                "mean_logits": [statistics.fmean(logits0), statistics.fmean(logits1)],
                "std_logits": [statistics.pstdev(logits0), statistics.pstdev(logits1)],
                "gold_label": identity.get("gold_label"),
                "input_ids": identity.get("input_ids"),
                "attention_mask": identity.get("attention_mask"),
                "token_type_ids": identity.get("token_type_ids"),
            }
        result[group_name] = {"inputs": inputs}
    return result


def _changed_examples(groups: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    all_ids = sorted(
        {
            int(dataset_idx)
            for group in groups.values()
            for dataset_idx in group.get("inputs", {})
        }
    )
    changed = []
    for dataset_idx in all_ids:
        rates = {
            group_name: groups[group_name].get("inputs", {}).get(
                str(dataset_idx), {}
            ).get("correct_rate")
            for group_name in GROUP_SPECS
        }
        finite_rates = [rate for rate in rates.values() if _finite_number(rate)]
        if len(finite_rates) == len(GROUP_SPECS) and max(finite_rates) != min(finite_rates):
            changed.append(
                {
                    "dataset_idx": dataset_idx,
                    "correct_rates": rates,
                    "rate_range": max(finite_rates) - min(finite_rates),
                }
            )
    changed.sort(key=lambda item: (-item["rate_range"], item["dataset_idx"]))
    return {
        "count": len(changed),
        "dataset_indices": [item["dataset_idx"] for item in changed],
        "examples": changed,
    }


def build_prediction_summary(
    *,
    run_payloads: Sequence[Any],
    prediction_rows: Sequence[Any],
    prior_run_payloads: Sequence[Any],
    source_commit: str,
    expected_examples: int = PRODUCTION_EXAMPLES,
    prediction_file_count: int = 5,
) -> dict[str, Any]:
    """Validate raw rows and aggregate per-trial and per-input diagnostics."""
    runs = list(run_payloads) if isinstance(run_payloads, Sequence) else []
    rows = list(prediction_rows) if isinstance(prediction_rows, Sequence) else []
    priors = list(prior_run_payloads) if isinstance(prior_run_payloads, Sequence) else []
    base_summary = build_summary(run_payloads=runs, source_commit=source_commit)
    runs_by_seed = _runs_by_seed(runs)
    buckets, index_failures = _index_rows(rows)
    completeness_gate = _prediction_completeness_gate(
        rows,
        buckets,
        index_failures,
        expected_examples=expected_examples,
        prediction_file_count=prediction_file_count,
        runs_by_seed=runs_by_seed,
    )
    identity_gate, canonical = _identity_gate(
        buckets, expected_examples=expected_examples
    )
    logits_gate, valid_row_indexes = _logits_gate(rows)
    trial_results: list[dict[str, Any]] = []
    recomputed_failures: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        groups = _group_map(runs_by_seed.get(seed))
        for group_name in GROUP_SPECS:
            raw_group = groups.get(group_name)
            for trial_index in range(TRIAL_COUNT):
                result, failures = _trial_result(
                    seed,
                    group_name,
                    trial_index,
                    buckets.get((seed, group_name, trial_index), []),
                    valid_row_indexes,
                    expected_examples,
                    raw_group,
                )
                trial_results.append(result)
                recomputed_failures.extend(failures)
    groups = _input_aggregates(rows, canonical, valid_row_indexes)
    for group_name, details in groups.items():
        base_group = base_summary.get("groups", {}).get(group_name, {})
        details.update(
            {
                "label": base_group.get("label"),
                "definition": base_group.get("definition"),
                "fusion_total": base_group.get("fusion_total"),
                "pooled_metrics": base_group.get("pooled_metrics"),
                "per_runs": base_group.get("per_runs"),
            }
        )
    gates = [
        _base_gate(base_summary),
        completeness_gate,
        identity_gate,
        logits_gate,
        _gate("recomputed_metrics", recomputed_failures),
        _shared_trial_seeds_gate(runs),
        _prior_equivalence_gate(runs, priors),
    ]
    expected_row_count = 5 * len(GROUP_SPECS) * TRIAL_COUNT * expected_examples
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "row_count": len(rows),
        "expected_row_count": expected_row_count,
        "prediction_file_count": prediction_file_count,
        "expected_prediction_file_count": 5,
        "expected_examples_per_trial": expected_examples,
        "protocol": base_summary.get("protocol", {}),
        "model": base_summary.get("model", {}),
        "expected_seeds": list(EXPECTED_SEEDS),
        "groups": groups,
        "trial_results": trial_results,
        "changed_examples": _changed_examples(groups),
        "base_summary": base_summary,
        "gates": gates,
        "all_gates_pass": all(gate["passed"] for gate in gates),
    }
    return _json_safe(summary)


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header_html = "".join(f"<th>{_h(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_h(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _number(value: Any) -> str:
    return f"{float(value):.10f}" if _finite_number(value) else "n/a"


def render_html(summary: Mapping[str, Any], prediction_rows: Sequence[Any]) -> str:
    """Render one standalone report with a bounded client-side detail table."""
    gates = summary.get("gates", [])
    gate_rows = [
        (
            gate.get("name"),
            "PASS" if gate.get("passed") else "FAIL",
            "none" if not gate.get("failures") else json.dumps(gate["failures"], sort_keys=True),
        )
        for gate in gates
        if isinstance(gate, Mapping)
    ]
    protocol = summary.get("protocol", {})
    model = summary.get("model", {})
    protocol_rows = [
        ("Gate status", "PASS" if summary.get("all_gates_pass") else "FAIL"),
        ("Source commit", summary.get("source_commit", "")),
        ("Rows", f"{summary.get('row_count', 0)} / {summary.get('expected_row_count', 0)}"),
        ("Prediction files", f"{summary.get('prediction_file_count', 0)} / 5"),
        ("Seeds", ", ".join(str(seed) for seed in summary.get("expected_seeds", []))),
        ("Trials per group/run", protocol.get("trials_per_run", TRIAL_COUNT)),
        ("Examples per trial", summary.get("expected_examples_per_trial")),
        ("Stage-1 GELU", model.get("stage1_gelu", [])),
        ("Stage-1 Softmax", model.get("stage1_softmax", [])),
        ("K", protocol.get("K")),
        ("Install path", protocol.get("install_path", "")),
    ]
    group_rows = []
    for group_name in GROUP_SPECS:
        group = summary.get("groups", {}).get(group_name, {})
        pooled = group.get("pooled_metrics", {}) or {}
        group_rows.append(
            (
                group_name,
                group.get("definition"),
                group.get("fusion_total"),
                _number((pooled.get("loss") or {}).get("mean")),
                _number((pooled.get("metric1") or {}).get("mean")),
                _number((pooled.get("metric2") or {}).get("mean")),
            )
        )
    trial_rows = [
        (
            trial.get("seed"),
            trial.get("group"),
            trial.get("trial_index"),
            trial.get("trial_seed"),
            trial.get("correct_count"),
            trial.get("incorrect_count"),
            trial.get("correct_dataset_indices"),
            trial.get("incorrect_dataset_indices"),
            _number(trial.get("recomputed_loss")),
            _number(trial.get("recomputed_accuracy")),
            _number(trial.get("recomputed_weighted_f1")),
        )
        for trial in summary.get("trial_results", [])
        if isinstance(trial, Mapping)
    ]
    input_rows = []
    for group_name in GROUP_SPECS:
        inputs = summary.get("groups", {}).get(group_name, {}).get("inputs", {})
        for dataset_idx, aggregate in inputs.items():
            input_rows.append(
                (
                    group_name,
                    dataset_idx,
                    aggregate.get("gold_label"),
                    aggregate.get("correct_count"),
                    _number(aggregate.get("correct_rate")),
                    aggregate.get("mean_logits"),
                    aggregate.get("std_logits"),
                    aggregate.get("input_ids"),
                    aggregate.get("attention_mask"),
                    aggregate.get("token_type_ids"),
                )
            )
    changed = summary.get("changed_examples", {})
    embedded = json.dumps(
        _json_safe(list(prediction_rows)),
        allow_nan=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    static_content = (
        "<h1>Stage-2 three-group per-example logits report</h1>"
        + "<h2>Protocol and overall summary</h2>"
        + _table(("Field", "Value"), protocol_rows)
        + "<h2>Validation gates</h2>"
        + _table(("Gate", "Status", "Failures"), gate_rows)
        + "<h2>Group summaries and configurations</h2>"
        + _table(("Group", "Configuration", "Fusion total", "Loss", "Accuracy", "Weighted F1"), group_rows)
        + "<h2>Trial summaries</h2>"
        + _table(
            (
                "Seed",
                "Group",
                "Trial",
                "Trial seed",
                "Correct",
                "Incorrect",
                "Correct dataset IDs",
                "Incorrect dataset IDs",
                "Loss",
                "Accuracy",
                "Weighted F1",
            ),
            trial_rows,
        )
        + "<h2>Changed examples across groups</h2>"
        + f"<p>{_h(changed.get('count', 0))} changed inputs: {_h(changed.get('dataset_indices', []))}</p>"
        + "<h2>Per-input aggregate</h2>"
        + _table(
            (
                "Group",
                "Dataset ID",
                "Gold",
                "Correct count",
                "Correct rate",
                "Mean logits",
                "Std logits",
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ),
            input_rows,
        )
    )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stage-2 per-example logits report</title>
<style>
body { margin: 0; color: #1f2428; background: #f2f3f4; font-family: system-ui, sans-serif; }
main { max-width: 1480px; margin: 0 auto; padding: 24px; background: #fff; }
h1 { font-size: 28px; } h2 { margin-top: 30px; border-bottom: 2px solid #d5d9dd; padding-bottom: 6px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 20px; font-size: 12px; }
th, td { border: 1px solid #d5d9dd; padding: 6px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }
th { background: #eceff1; } tbody tr:nth-child(even) { background: #fafafa; }
.filters { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 8px; margin: 12px 0; }
label { display: grid; gap: 4px; font-size: 12px; font-weight: 600; }
select, input, button { min-height: 34px; border: 1px solid #aeb5bb; background: #fff; padding: 5px 7px; }
.pagination { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
@media (max-width: 800px) { .filters { grid-template-columns: 1fr 1fr; } main { padding: 12px; } }
</style></head><body><main>""" + static_content + """
<h2>Per-example prediction rows</h2>
<div class="filters">
<label>Seed<select id="seed-filter"><option value="">All</option></select></label>
<label>Group<select id="group-filter"><option value="">All</option></select></label>
<label>Trial<select id="trial-filter"><option value="">All</option></select></label>
<label>Correct<select id="correct-filter"><option value="">All</option><option value="true">Correct</option><option value="false">Incorrect</option></select></label>
<label>Dataset ID<input id="dataset-idx-filter" inputmode="numeric"></label>
</div>
<div class="pagination"><button id="previous-page" type="button">Previous</button><span id="page-status"></span><button id="next-page" type="button">Next</button></div>
<table><thead><tr><th>Seed</th><th>Group</th><th>Trial</th><th>Dataset ID</th><th>input_ids</th><th>attention_mask</th><th>token_type_ids</th><th>Gold</th><th>Predicted</th><th>Correct</th><th>Logits</th></tr></thead><tbody id="prediction-table-body"></tbody></table>
<script type="application/json" id="prediction-data">""" + embedded + """</script>
<script>
(() => {
  'use strict';
  const PAGE_SIZE = 100;
  const predictionRows = JSON.parse(document.getElementById('prediction-data').textContent);
  const controls = {
    seed: document.getElementById('seed-filter'),
    group: document.getElementById('group-filter'),
    trial: document.getElementById('trial-filter'),
    correct: document.getElementById('correct-filter'),
    datasetIdx: document.getElementById('dataset-idx-filter')
  };
  const body = document.getElementById('prediction-table-body');
  const pageStatus = document.getElementById('page-status');
  let filteredRows = predictionRows;
  let pageStart = 0;
  function addOptions(select, values) {
    Array.from(values).sort((a, b) => String(a).localeCompare(String(b), undefined, {numeric: true})).forEach(value => {
      const option = document.createElement('option'); option.value = String(value); option.textContent = String(value); select.appendChild(option);
    });
  }
  addOptions(controls.seed, new Set(predictionRows.reduce((values, row) => values.concat([row.run_seed]), [])));
  addOptions(controls.group, new Set(predictionRows.reduce((values, row) => values.concat([row.group]), [])));
  addOptions(controls.trial, new Set(predictionRows.reduce((values, row) => values.concat([row.trial_index]), [])));
  function appendCell(tr, value) { const td = document.createElement('td'); td.textContent = Array.isArray(value) ? JSON.stringify(value) : String(value ?? ''); tr.appendChild(td); }
  function renderPage() {
    body.replaceChildren();
    const pageRows = filteredRows.slice(pageStart, pageStart + PAGE_SIZE);
    pageRows.forEach(row => {
      const tr = document.createElement('tr');
      [row.run_seed, row.group, row.trial_index, row.dataset_idx, row.input_ids, row.attention_mask, row.token_type_ids, row.gold_label, row.predicted_label, row.correct, row.logits].forEach(value => appendCell(tr, value));
      body.appendChild(tr);
    });
    const first = filteredRows.length ? pageStart + 1 : 0;
    pageStatus.textContent = `${first}-${Math.min(pageStart + PAGE_SIZE, filteredRows.length)} of ${filteredRows.length}`;
    document.getElementById('previous-page').disabled = pageStart === 0;
    document.getElementById('next-page').disabled = pageStart + PAGE_SIZE >= filteredRows.length;
  }
  function applyFilters() {
    const datasetNeedle = controls.datasetIdx.value.trim();
    filteredRows = predictionRows.filter(row =>
      (!controls.seed.value || String(row.run_seed) === controls.seed.value) &&
      (!controls.group.value || row.group === controls.group.value) &&
      (!controls.trial.value || String(row.trial_index) === controls.trial.value) &&
      (!controls.correct.value || String(row.correct) === controls.correct.value) &&
      (!datasetNeedle || String(row.dataset_idx) === datasetNeedle)
    );
    pageStart = 0; renderPage();
  }
  Object.values(controls).forEach(control => control.addEventListener('input', applyFilters));
  document.getElementById('previous-page').addEventListener('click', () => { pageStart = Math.max(0, pageStart - PAGE_SIZE); renderPage(); });
  document.getElementById('next-page').addEventListener('click', () => { if (pageStart + PAGE_SIZE < filteredRows.length) pageStart += PAGE_SIZE; renderPage(); });
  renderPage();
})();
</script></main></body></html>"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the strict Stage-2 three-group per-example logits report."
    )
    parser.add_argument("--run-json", action="append", required=True)
    parser.add_argument("--prediction-jsonl", action="append", required=True)
    parser.add_argument("--prior-run-json", action="append", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    return parser


def _load_json(path_text: str) -> Any:
    path = Path(path_text)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return {
            "_load_error": f"{type(error).__name__}: {error}",
            "_source_path": str(path),
        }


def _load_jsonl(paths: Sequence[str]) -> list[Any]:
    rows: list[Any] = []
    for path_text in paths:
        path = Path(path_text)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        rows.append(
                            {
                                "_load_error": f"JSONDecodeError: {error}",
                                "_source_path": str(path),
                                "_line_number": line_number,
                            }
                        )
        except OSError as error:
            rows.append(
                {
                    "_load_error": f"{type(error).__name__}: {error}",
                    "_source_path": str(path),
                }
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_payloads = [_load_json(path) for path in args.run_json]
    prior_payloads = [_load_json(path) for path in args.prior_run_json]
    prediction_rows = _load_jsonl(args.prediction_jsonl)
    summary = build_prediction_summary(
        run_payloads=run_payloads,
        prediction_rows=prediction_rows,
        prior_run_payloads=prior_payloads,
        source_commit=args.source_commit,
        prediction_file_count=len(args.prediction_jsonl),
    )
    output_json = Path(args.output_json)
    output_html = Path(args.output_html)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    with output_html.open("w", encoding="utf-8") as handle:
        handle.write(render_html(summary, prediction_rows))
    return 0 if summary["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
