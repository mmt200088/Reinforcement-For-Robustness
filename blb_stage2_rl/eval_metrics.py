"""Shared evaluation metric helpers for Stage-2 inference paths.

The Stage-2 codebase has several inference entrypoints: online RL probes,
multi-GPU probe workers, Paean final-eval, and fixed-action experiments.  These
helpers keep the metric aggregation semantics identical across those paths:
batch losses are weighted by sample count, MRPC/QQP metric2 is weighted F1, and
repeat evaluations report population mean/std over complete trials.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


def logits_to_classes(logits: Any) -> np.ndarray:
    arr = np.asarray(logits)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        return (arr > 0.5).astype(int)
    return np.argmax(arr, axis=1).astype(int)


def uses_weighted_f1_metric2(metric_profile: str) -> bool:
    normalized = str(metric_profile or "").lower()
    return "mrpc" in normalized or "qqp" in normalized


def accuracy_from_labels(labels: Any, preds: Any) -> float:
    labels_arr = np.asarray(labels).reshape(-1)
    preds_arr = np.asarray(preds).reshape(-1)
    if labels_arr.size == 0:
        return 0.0
    return float(np.mean(preds_arr == labels_arr))


def weighted_f1_from_labels(labels: Any, preds: Any) -> float:
    labels_arr = np.asarray(labels).reshape(-1)
    preds_arr = np.asarray(preds).reshape(-1)
    if labels_arr.size == 0:
        return 0.0
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
        denom = (2.0 * tp) + fp + fn
        f1 = (2.0 * tp / denom) if denom > 0.0 else 0.0
        out += (support / total) * f1
    return float(out)


def matthews_corrcoef_from_labels(labels: Any, preds: Any) -> float:
    labels_arr = np.asarray(labels).reshape(-1)
    preds_arr = np.asarray(preds).reshape(-1)
    if labels_arr.size == 0:
        return 0.0
    classes = np.union1d(labels_arr, preds_arr)
    if classes.size <= 1:
        return 0.0
    if classes.size == 2:
        neg, pos = classes[0], classes[1]
        tp = float(np.sum((preds_arr == pos) & (labels_arr == pos)))
        tn = float(np.sum((preds_arr == neg) & (labels_arr == neg)))
        fp = float(np.sum((preds_arr == pos) & (labels_arr == neg)))
        fn = float(np.sum((preds_arr == neg) & (labels_arr == pos)))
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float(((tp * tn) - (fp * fn)) / denom) if denom > 0.0 else 0.0
    # Multi-class MCC. Kept here for completeness; current GLUE use is binary.
    conf = np.zeros((classes.size, classes.size), dtype=float)
    cls_to_idx = {cls: i for i, cls in enumerate(classes.tolist())}
    for y_true, y_pred in zip(labels_arr, preds_arr):
        conf[cls_to_idx[y_true], cls_to_idx[y_pred]] += 1.0
    t_sum = conf.sum(axis=1)
    p_sum = conf.sum(axis=0)
    n_correct = np.trace(conf)
    n_samples = conf.sum()
    cov_ytyp = (n_correct * n_samples) - np.dot(t_sum, p_sum)
    cov_ypyp = (n_samples ** 2) - np.dot(p_sum, p_sum)
    cov_ytyt = (n_samples ** 2) - np.dot(t_sum, t_sum)
    denom = math.sqrt(cov_ytyt * cov_ypyp)
    return float(cov_ytyp / denom) if denom > 0.0 else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: Any, y: Any) -> float:
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    if x_arr.size < 2 or y_arr.size < 2:
        return 0.0
    x_c = x_arr - x_arr.mean()
    y_c = y_arr - y_arr.mean()
    denom = math.sqrt(float(np.dot(x_c, x_c) * np.dot(y_c, y_c)))
    return float(np.dot(x_c, y_c) / denom) if denom > 0.0 else 0.0


def spearman_corr(x: Any, y: Any) -> float:
    return pearson_corr(_average_ranks(np.asarray(x, dtype=float).reshape(-1)),
                        _average_ranks(np.asarray(y, dtype=float).reshape(-1)))


def metric_pair_for_dataset(
        dataset_key: str,
        labels: Any,
        logits_or_predictions: Any,
        *,
        predictions_are_classes: bool = False,
        ) -> Tuple[float, float]:
    ds = str(dataset_key or "").lower()
    labels_arr = np.asarray(labels).reshape(-1)
    if ds == "stsb":
        preds = np.asarray(logits_or_predictions, dtype=float).reshape(-1)
        return pearson_corr(preds, labels_arr), spearman_corr(preds, labels_arr)
    pred_classes = (
        np.asarray(logits_or_predictions).reshape(-1).astype(int)
        if predictions_are_classes
        else logits_to_classes(logits_or_predictions)
    )
    if ds == "cola":
        metric1 = matthews_corrcoef_from_labels(labels_arr, pred_classes)
        return metric1, metric1
    metric1 = accuracy_from_labels(labels_arr, pred_classes)
    if uses_weighted_f1_metric2(ds):
        metric2 = weighted_f1_from_labels(labels_arr, pred_classes)
    else:
        metric2 = metric1
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


def sample_weighted_mean(values: Sequence[float], counts: Sequence[int]) -> float:
    weights = np.asarray([max(0, int(c)) for c in counts], dtype=float)
    if weights.size == 0 or float(weights.sum()) <= 0.0:
        return float(np.mean(values)) if values else float("nan")
    return float(np.average(np.asarray(values, dtype=float), weights=weights))


def weighted_probe_batch_means(
        losses: Sequence[float],
        m1s: Sequence[float],
        m2s: Sequence[float],
        counts: Sequence[int],
        ) -> Tuple[float, float, float]:
    return (
        sample_weighted_mean(losses, counts),
        sample_weighted_mean(m1s, counts),
        sample_weighted_mean(m2s, counts),
    )


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
    loss, m1, m2 = weighted_probe_batch_means(losses, m1s, m2s, counts)
    if is_regression and preds and labels:
        all_preds = np.concatenate([np.asarray(p).reshape(-1) for p in preds])
        all_labels = np.concatenate([np.asarray(l).reshape(-1) for l in labels])
        m1, m2 = metric_pair_for_dataset(metric_profile, all_labels, all_preds)
    elif (
            not is_regression
            and uses_weighted_f1_metric2(metric_profile)
            and preds
            and labels
            ):
        all_preds = np.concatenate([np.asarray(p).reshape(-1) for p in preds])
        all_labels = np.concatenate([np.asarray(l).reshape(-1) for l in labels])
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
