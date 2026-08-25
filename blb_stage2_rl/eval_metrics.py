"""Shared evaluation metric helpers for Stage-2 inference paths.

The Stage-2 codebase has several inference entrypoints: online RL probes,
multi-GPU probe workers, Paean final-eval, and fixed-action experiments.  These
helpers keep the metric aggregation semantics identical across those paths:
batch losses are weighted by sample count, metric2 is weighted F1, and
repeat evaluations report population mean/std over complete trials.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
from rfr.preparation.data.protocol import dataset_from_profile


def logits_to_classes(logits: Any) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        return (arr > 0.5).astype(int)
    return np.argmax(arr, axis=1).astype(int)


def uses_weighted_f1_metric2(metric_profile: str) -> bool:
    dataset_from_profile(metric_profile)
    return True


def accuracy_from_labels(labels: Any, preds: Any) -> float:
    labels_arr = np.asarray(labels).reshape(-1)
    preds_arr = np.asarray(preds).reshape(-1)
    if labels_arr.size == 0:
        return 0.0
    return float(np.count_nonzero(preds_arr == labels_arr) / labels_arr.size)


def _is_zero_one_array(arr: np.ndarray) -> bool:
    if arr.dtype.kind not in "biu":
        return False
    return int(arr.min()) >= 0 and int(arr.max()) <= 1


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = (2.0 * tp) + fp + fn
    return (2.0 * tp / denom) if denom > 0.0 else 0.0


def _binary_zero_one_weighted_f1(labels_arr: np.ndarray, preds_arr: np.ndarray) -> float:
    label_pos = labels_arr == 1
    pred_pos = preds_arr == 1
    total = float(labels_arr.size)
    support_pos = float(np.count_nonzero(label_pos))
    pred_pos_count = float(np.count_nonzero(pred_pos))
    tp = float(np.count_nonzero(pred_pos & label_pos))
    fp = pred_pos_count - tp
    fn = support_pos - tp
    tn = total - tp - fp - fn
    pos_f1 = _f1_from_counts(tp, fp, fn)
    neg_f1 = _f1_from_counts(tn, fn, fp)
    return float((((total - support_pos) / total) * neg_f1)
                 + ((support_pos / total) * pos_f1))



def weighted_f1_from_labels(labels: Any, preds: Any) -> float:
    labels_arr = np.asarray(labels).reshape(-1)
    preds_arr = np.asarray(preds).reshape(-1)
    if labels_arr.size == 0:
        return 0.0
    if _is_zero_one_array(labels_arr) and _is_zero_one_array(preds_arr):
        return _binary_zero_one_weighted_f1(labels_arr, preds_arr)
    classes = np.union1d(preds_arr, labels_arr)
    total = float(labels_arr.size)
    out = 0.0
    for cls in classes:
        pred_pos = preds_arr == cls
        label_pos = labels_arr == cls
        support = float(np.sum(label_pos))
        if support <= 0.0:
            continue
        tp = float(np.sum(pred_pos & label_pos))
        fp = float(np.sum(pred_pos & ~label_pos))
        fn = float(np.sum(~pred_pos & label_pos))
        f1 = _f1_from_counts(tp, fp, fn)
        out += (support / total) * f1
    return float(out)



def metric_pair_for_dataset(
        dataset_key: str,
        labels: Any,
        logits_or_predictions: Any,
        *,
        predictions_are_classes: bool = False,
        ) -> Tuple[float, float]:
    dataset_from_profile(dataset_key)
    labels_arr = np.asarray(labels).reshape(-1)
    pred_classes = (
        np.asarray(logits_or_predictions).reshape(-1).astype(int)
        if predictions_are_classes
        else logits_to_classes(logits_or_predictions)
    )
    metric1 = accuracy_from_labels(labels_arr, pred_classes)
    metric2 = weighted_f1_from_labels(labels_arr, pred_classes)
    return metric1, metric2


def probe_batch_sample_count(labels: Any) -> int:
    if hasattr(labels, "dim") and callable(labels.dim):
        if int(labels.dim()) == 0:
            return 1
        return int(labels.shape[0])
    labels_arr = np.asarray(labels)
    if labels_arr.ndim == 0:
        return 1
    return int(labels_arr.shape[0])


def _probe_count_weights(counts: Sequence[int]) -> np.ndarray:
    return np.asarray([max(0, int(c)) for c in counts], dtype=float)


def _sample_weighted_mean_with_weights(
        values: Sequence[float],
        weights: np.ndarray,
        weight_sum: float,
        ) -> float:
    if weights.size == 0 or weight_sum <= 0.0:
        return float(np.mean(values)) if values else float("nan")
    values_arr = np.asarray(values, dtype=float)
    return float(np.dot(values_arr, weights) / weight_sum)


def sample_weighted_mean(values: Sequence[float], counts: Sequence[int]) -> float:
    weights = _probe_count_weights(counts)
    return _sample_weighted_mean_with_weights(
        values,
        weights,
        float(weights.sum()),
    )


def weighted_probe_batch_means(
        losses: Sequence[float],
        m1s: Sequence[float],
        m2s: Sequence[float],
        counts: Sequence[int],
        ) -> Tuple[float, float, float]:
    weights = _probe_count_weights(counts)
    weight_sum = float(weights.sum())
    return (
        _sample_weighted_mean_with_weights(losses, weights, weight_sum),
        _sample_weighted_mean_with_weights(m1s, weights, weight_sum),
        _sample_weighted_mean_with_weights(m2s, weights, weight_sum),
    )


def _flatten_probe_arrays(values: Sequence[np.ndarray]) -> np.ndarray:
    if len(values) == 1:
        return np.asarray(values[0]).reshape(-1)
    return np.concatenate([np.asarray(value).reshape(-1) for value in values])


def finalize_probe_trial_metrics(
        losses: Sequence[float],
        m1s: Sequence[float],
        m2s: Sequence[float],
        counts: Sequence[int],
        *,
        metric_profile: str,
        is_regression: bool,
        preds: Optional[Sequence[np.ndarray]] = None,
        labels: Optional[Sequence[np.ndarray]] = None,
        ) -> Optional[Tuple[float, float, float]]:
    if not losses:
        return None
    dataset_from_profile(metric_profile)
    if is_regression:
        raise ValueError("regression metrics are unsupported")
    loss, m1, m2 = weighted_probe_batch_means(losses, m1s, m2s, counts)
    if preds and labels:
        all_preds = _flatten_probe_arrays(preds)
        all_labels = _flatten_probe_arrays(labels)
        m2 = weighted_f1_from_labels(all_labels, all_preds)
    return float(loss), float(m1), float(m2)


def summarize_eval_trials(trials: Sequence[Mapping[str, Any]]) -> dict:
    losses = np.asarray([float(t["loss"]) for t in trials], dtype=float)
    ps = np.asarray([float(t["p"]) for t in trials], dtype=float)
    ss = np.asarray([float(t["s"]) for t in trials], dtype=float)
    times = np.asarray([float(t.get("time_ms", 0.0)) for t in trials], dtype=float)
    return {
        "n": int(len(trials)),
        "loss_mean": float(losses.mean()),
        "loss_std": float(losses.std(ddof=0)),
        "loss_min": float(losses.min()),
        "loss_max": float(losses.max()),
        "loss_range": float(losses.max() - losses.min()),
        "p_mean": float(ps.mean()),
        "p_std": float(ps.std(ddof=0)),
        "p_min": float(ps.min()),
        "p_max": float(ps.max()),
        "p_range": float(ps.max() - ps.min()),
        "s_mean": float(ss.mean()),
        "s_std": float(ss.std(ddof=0)),
        "s_min": float(ss.min()),
        "s_max": float(ss.max()),
        "s_range": float(ss.max() - ss.min()),
        "time_mean_ms": float(times.mean()),
        "time_std_ms": float(times.std(ddof=0)),
    }


def summarize_selected_vs_random_results(
        selected_results: Sequence[Mapping[str, Any]],
        random_results: Sequence[Mapping[str, Any]],
        *,
        num_metrics: int,
        ) -> dict:
    """Build the canonical selected-vs-random comparison summary."""
    if not selected_results and not random_results:
        return {}

    def _new_stats() -> dict[str, Any]:
        return {
            "n": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": None,
            "max": None,
        }

    def _add_stats(stat: dict[str, Any], value: float) -> None:
        value_f = float(value)
        stat["n"] += 1
        stat["sum"] += value_f
        stat["sum_sq"] += value_f * value_f
        if stat["min"] is None or value_f < stat["min"]:
            stat["min"] = value_f
        if stat["max"] is None or value_f > stat["max"]:
            stat["max"] = value_f

    def _finish_stats(stat: Mapping[str, Any]) -> dict[str, float | int]:
        count = int(stat["n"])
        if count <= 0:
            return {"n": 0}
        mean = float(stat["sum"]) / float(count)
        variance = float(stat["sum_sq"]) / float(count) - mean * mean
        if np.isfinite(variance) and variance < 0.0:
            variance = 0.0
        return {
            "n": count,
            "mean": float(mean),
            "std": float(variance ** 0.5),
            "min": float(stat["min"]),
            "max": float(stat["max"]),
        }

    anchor = selected_results[0] if selected_results else None
    random_count = len(random_results)
    summary: dict[str, Any] = {
        "selected_count": len(selected_results),
        "random_count": random_count,
    }
    if anchor is not None:
        summary["selected_anchor"] = {
            "name": str(anchor.get("name", "selected")),
            "loss_mean": float(anchor.get("loss", 0.0)),
            "loss_std": float(anchor.get("loss_std", 0.0)),
            "metric1_mean": float(anchor.get("p", 0.0)),
            "metric1_std": float(anchor.get("p_std", 0.0)),
            "metric2_mean": float(anchor.get("s", 0.0)),
            "metric2_std": float(anchor.get("s_std", 0.0)),
            "total_bits_sum": int(anchor.get("total_bits_sum", 0)),
            "total_fusion_count": int(anchor.get("total_fusion_count", 0)),
            "avg_truncation_k": float(anchor.get("avg_truncation_k", 0.0)),
        }
    if random_results:
        stat_fields = (
            ("loss_mean", "loss"),
            ("loss_std", "loss_std"),
            ("metric1_mean", "p"),
            ("metric1_std", "p_std"),
            ("metric2_mean", "s"),
            ("metric2_std", "s_std"),
        )
        stats_by_name = {name: _new_stats() for name, _field in stat_fields}
        metric1_rank = 0
        loss_rank = 0
        metric2_rank = 0
        anchor_p = float(anchor.get("p", 0.0)) if anchor is not None else 0.0
        anchor_loss = float(anchor.get("loss", 0.0)) if anchor is not None else 0.0
        anchor_s = float(anchor.get("s", 0.0)) if anchor is not None else 0.0

        for row in random_results:
            for name, field in stat_fields:
                _add_stats(stats_by_name[name], float(row.get(field, 0.0)))
            if anchor is None:
                continue
            if float(row.get("p", 0.0)) < anchor_p:
                metric1_rank += 1
            if float(row.get("loss", 0.0)) > anchor_loss:
                loss_rank += 1
            if num_metrics > 1 and float(row.get("s", 0.0)) < anchor_s:
                metric2_rank += 1

        summary["random_stats"] = {
            name: _finish_stats(stats_by_name[name])
            for name, _field in stat_fields
        }

        if anchor is not None:

            def _rank_dict(rank: int) -> dict[str, float | int] | None:
                count = int(random_count)
                if count <= 0:
                    return None
                rank_i = int(rank)
                return {
                    "rank_better_than_selected": rank_i,
                    "out_of": count,
                    "percentile": float(rank_i) / float(count),
                }

            summary["anchor_rank_vs_random"] = {
                "metric1_higher_better": _rank_dict(metric1_rank),
                "loss_lower_better": _rank_dict(loss_rank),
            }
            if num_metrics > 1:
                summary["anchor_rank_vs_random"]["metric2_higher_better"] = (
                    _rank_dict(metric2_rank)
                )
    return summary


def pack_repeat_evaluation(
        trials: Sequence[Mapping[str, Any]],
        *,
        evaluation_mode: Optional[str] = None,
        ) -> dict:
    """Build the canonical repeated-evaluation payload.

    Keep the trial numbering, numeric coercion, and population-stat semantics in
    this helper so Paean final eval, fixed-action experiments, and future report
    writers do not drift in their JSON shape.
    """
    normalized_trials = [
        {
            "trial": int(idx + 1),
            "loss": float(trial["loss"]),
            "p": float(trial["p"]),
            "s": float(trial["s"]),
            "time_ms": float(trial.get("time_ms", 0.0)),
        }
        for idx, trial in enumerate(trials)
    ]
    stats = summarize_eval_trials(normalized_trials)
    if evaluation_mode is not None:
        stats["evaluation_mode"] = str(evaluation_mode)
    return {
        "trials": normalized_trials,
        "stats": stats,
    }
