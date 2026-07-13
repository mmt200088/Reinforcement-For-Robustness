"""Shared post-install model inference evaluation helpers.

These helpers cover the part after a BLB/Stage-1 configuration has already
been installed on the model: run the model forward, aggregate loss/logits, and
compute metrics.  Keep this module as the shared seam for RL probes, multi-GPU
probe workers, Paean final/fixed eval, and fixed-action experiments.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .eval_metrics import (
    accuracy_from_labels,
    finalize_probe_trial_metrics,
    logits_to_classes,
    metric_pair_for_dataset,
    probe_batch_sample_count,
    sample_weighted_mean,
    uses_weighted_f1_metric2,
)


@dataclass(frozen=True)
class InstalledModelEvalResult:
    loss: float
    metric1: float
    metric2: float
    time_ms: float
    labels: np.ndarray
    logits: np.ndarray


def normalize_labels_for_metrics(labels: Any) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        arr = labels.detach()
        if arr.device.type != "cpu":
            arr = arr.cpu()
        return np.asarray(arr.numpy()).reshape(-1)
    return np.asarray(labels).reshape(-1)


def normalize_logits_for_metrics(logits: Any, expected_batch_size: int) -> np.ndarray:
    if isinstance(logits, torch.Tensor):
        arr = logits.detach()
        if arr.device.type != "cpu":
            arr = arr.cpu()
        logits_arr = np.asarray(arr.numpy())
    else:
        logits_arr = np.asarray(logits)
    if logits_arr.ndim == 0:
        logits_arr = logits_arr.reshape(1)
    if logits_arr.shape[0] != expected_batch_size:
        logits_arr = logits_arr.reshape(expected_batch_size, -1)
    elif expected_batch_size == 1 and logits_arr.ndim == 1:
        logits_arr = logits_arr.reshape(1, -1)
    if logits_arr.ndim == 2 and logits_arr.shape[1] == 1:
        return logits_arr.reshape(-1)
    return logits_arr


def normalize_logits_tensor_for_metrics(
        logits: torch.Tensor,
        expected_batch_size: int,
        ) -> torch.Tensor:
    logits_t = logits.detach()
    if logits_t.dim() == 0:
        logits_t = logits_t.reshape(1)
    if logits_t.shape[0] != expected_batch_size:
        logits_t = logits_t.reshape(expected_batch_size, -1)
    elif expected_batch_size == 1 and logits_t.dim() == 1:
        logits_t = logits_t.reshape(1, -1)
    if logits_t.dim() == 2 and logits_t.shape[1] == 1:
        return logits_t.reshape(-1)
    return logits_t


def output_loss_and_logits(outputs: Any) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    loss = getattr(outputs, "loss", None)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        logits = outputs[1]
    return loss, logits


def concatenate_logits_for_metrics(batch_logits: Sequence[Any]) -> np.ndarray:
    if not batch_logits:
        return np.asarray([])
    if all(isinstance(x, torch.Tensor) for x in batch_logits):
        tensors = [x.detach() for x in batch_logits]
        first_device = tensors[0].device
        if all(t.device == first_device for t in tensors):
            if all(t.dim() == 1 for t in tensors):
                packed = torch.cat([t.reshape(-1) for t in tensors], dim=0)
            else:
                packed = torch.cat(
                    [t.reshape(-1, 1) if t.dim() == 1 else t for t in tensors],
                    dim=0,
                )
            return packed.detach().cpu().numpy()
    arrays = []
    for x in batch_logits:
        if isinstance(x, torch.Tensor):
            arrays.append(np.asarray(x.detach().cpu().numpy()))
        else:
            arrays.append(np.asarray(x))
    if all(arr.ndim == 1 for arr in arrays):
        return np.concatenate([arr.reshape(-1) for arr in arrays], axis=0)
    return np.concatenate(
        [arr.reshape(-1, 1) if arr.ndim == 1 else arr for arr in arrays],
        axis=0,
    )


def concatenate_labels_for_metrics(batch_labels: Sequence[Any]) -> np.ndarray:
    if not batch_labels:
        return np.asarray([])
    if all(isinstance(x, torch.Tensor) for x in batch_labels):
        tensors = [x.detach().reshape(-1) for x in batch_labels]
        first_device = tensors[0].device
        if all(t.device == first_device for t in tensors):
            return torch.cat(tensors, dim=0).detach().cpu().numpy()
    arrays = [
        np.asarray(x.detach().cpu().numpy()).reshape(-1)
        if isinstance(x, torch.Tensor)
        else np.asarray(x).reshape(-1)
        for x in batch_labels
    ]
    return np.concatenate(arrays, axis=0)


def tensor_scalar_values_to_float_list(values: Sequence[torch.Tensor]) -> List[float]:
    if not values:
        return []
    stacked = torch.stack([v.reshape(()) for v in values], dim=0)
    return [float(x) for x in stacked.detach().cpu().numpy().reshape(-1)]


def tensor_scalar_sequences_to_float_lists(
        *sequences: Sequence[torch.Tensor],
        ) -> Tuple[List[float], ...]:
    if not sequences:
        return tuple()
    lengths = [len(values) for values in sequences]
    if sum(lengths) == 0:
        return tuple([] for _ in sequences)
    stacked = torch.stack(
        [v.reshape(()) for values in sequences for v in values],
        dim=0,
    )
    flat = stacked.detach().cpu().numpy().reshape(-1)
    out: List[List[float]] = []
    cursor = 0
    for length in lengths:
        next_cursor = cursor + length
        out.append([float(x) for x in flat[cursor:next_cursor]])
        cursor = next_cursor
    return tuple(out)


def tensor_values_to_numpy_arrays(values: Sequence[Any]) -> List[np.ndarray]:
    if not values:
        return []
    if all(isinstance(x, torch.Tensor) for x in values):
        tensors = [x.detach().reshape(-1) for x in values]
        first_device = tensors[0].device
        if all(t.device == first_device for t in tensors):
            return [torch.cat(tensors, dim=0).detach().cpu().numpy()]
    return [
        np.asarray(x.detach().cpu().numpy())
        if isinstance(x, torch.Tensor)
        else np.asarray(x)
        for x in values
    ]


def run_installed_model_on_dataloader(
        model: torch.nn.Module,
        dataloader: Any,
        *,
        device: torch.device,
        metric_profile: str,
        use_train: bool = False,
        split_name: Optional[str] = None,
        mnli_metric2_fn: Optional[Callable[[], float]] = None,
        loss_average: str = "sample",
        ) -> InstalledModelEvalResult:
    """Run a full installed-model eval loop on an existing dataloader.

    The caller is responsible for installing any function/noise cfg before this
    function is called and clearing it afterwards when needed.
    """
    del split_name  # Kept in the signature so call sites can pass their context.
    model.eval()
    loss_tensors: List[torch.Tensor] = []
    loss_counts: List[int] = []
    batch_logits: List[Any] = []
    batch_labels: List[Any] = []
    t0 = time.time()
    with torch.inference_mode():
        for batch in dataloader:
            raw_labels = batch["labels"]
            if isinstance(raw_labels, torch.Tensor):
                labels = raw_labels.detach().reshape(-1)
                expected_batch_size = int(labels.numel())
            else:
                labels = normalize_labels_for_metrics(raw_labels)
                expected_batch_size = int(labels.size)
            moved = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**moved)
            loss_t, logits_t = output_loss_and_logits(outputs)
            if loss_t is not None:
                loss_tensors.append(loss_t.detach().reshape(()))
                loss_counts.append(expected_batch_size)
            if isinstance(logits_t, torch.Tensor):
                logits = normalize_logits_tensor_for_metrics(
                    logits_t,
                    expected_batch_size=expected_batch_size,
                )
            else:
                logits = normalize_logits_for_metrics(
                    logits_t,
                    expected_batch_size=expected_batch_size,
                )
            batch_logits.append(logits)
            batch_labels.append(labels)

    loss_values = tensor_scalar_values_to_float_list(loss_tensors)
    if not loss_values:
        avg_loss = 0.0
    elif str(loss_average).lower() == "batch":
        avg_loss = float(sum(loss_values) / len(loss_values))
    else:
        avg_loss = sample_weighted_mean(loss_values, loss_counts)
    all_logits = concatenate_logits_for_metrics(batch_logits)
    all_labels = concatenate_labels_for_metrics(batch_labels)
    n_batches = max(1, len(dataloader))
    avg_time = (time.time() - t0) * 1000.0 / n_batches

    ds = str(metric_profile or "").lower()
    if ds == "mnli":
        pred_classes = logits_to_classes(all_logits)
        metric1 = accuracy_from_labels(all_labels, pred_classes)
        metric2 = (
            float(mnli_metric2_fn())
            if (not use_train and mnli_metric2_fn is not None)
            else metric1
        )
    else:
        metric1, metric2 = metric_pair_for_dataset(ds, all_labels, all_logits)
    return InstalledModelEvalResult(
        loss=float(avg_loss),
        metric1=float(metric1),
        metric2=float(metric2),
        time_ms=float(avg_time),
        labels=all_labels,
        logits=all_logits,
    )


def logits_to_classes_tensor(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        return (logits > 0.5).long()
    return logits.argmax(dim=-1)


def compute_probe_batch_metrics(
        logits: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_regression: bool,
        ) -> Tuple[float, float, float]:
    """Return per-batch (loss, metric1, metric2) for reward probes."""
    if is_regression:
        preds = logits.view(-1).float()
        targets = labels.view(-1).float()
        loss = float(torch.nn.functional.mse_loss(preds, targets).item())
        return loss, -loss, -loss
    loss_t = torch.nn.functional.cross_entropy(
        logits.float(), labels.long(), reduction="mean",
    )
    preds = logits_to_classes_tensor(logits.detach())
    acc = float((preds.detach().long() == labels.detach().long()).float().mean().item())
    return float(loss_t.item()), acc, acc


def probe_batch_to_model_kwargs(batch: Any) -> dict:
    kwargs = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": batch.labels,
    }
    token_type_ids = getattr(batch, "token_type_ids", None)
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids
    return kwargs


def run_installed_probe_trial(
        model: torch.nn.Module,
        probe_batches: Sequence[Any],
        *,
        is_regression: bool,
        metric_profile: str,
        restore_training: bool = True,
        forward_context: Optional[Any] = None,
        ) -> Tuple[float, float, float]:
    """Run one already-installed noisy probe trial over probe batches.

    ``forward_context`` scopes only model forwards. Metric kernels and the
    batched device-to-host synchronization run after it exits, so callers can
    protect a shared noise generator without serializing metric aggregation.
    """
    trial_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    loss_tensors: List[torch.Tensor] = []
    m1_tensors: List[torch.Tensor] = []
    m2_tensors: List[torch.Tensor] = []
    counts: List[int] = []
    trial_pred_tensors: List[Any] = []
    trial_label_tensors: List[Any] = []
    was_training = bool(model.training)
    need_prediction_arrays = (
        bool(is_regression) or uses_weighted_f1_metric2(metric_profile)
    )
    if was_training:
        model.eval()
    forward_ctx = (
        forward_context
        if forward_context is not None
        else contextlib.nullcontext()
    )
    try:
        with forward_ctx:
            with torch.inference_mode():
                for batch in probe_batches:
                    outputs = model(**probe_batch_to_model_kwargs(batch))
                    _loss_t, logits = output_loss_and_logits(outputs)
                    trial_outputs.append((logits, batch.labels))
    finally:
        if restore_training and was_training:
            model.train()

    with torch.inference_mode():
        for logits, labels in trial_outputs:
            counts.append(probe_batch_sample_count(labels))
            if is_regression:
                preds = logits.view(-1).float()
                targets = labels.view(-1).float()
                loss_t = torch.nn.functional.mse_loss(preds, targets)
                m1_t = -loss_t
                m2_t = -loss_t
                if need_prediction_arrays:
                    trial_pred_tensors.append(preds.detach())
            else:
                loss_t = torch.nn.functional.cross_entropy(
                    logits.float(), labels.long(), reduction="mean",
                )
                preds = logits_to_classes_tensor(logits.detach())
                detached_labels = labels.detach()
                m1_t = (preds.long() == detached_labels.long()).float().mean()
                m2_t = m1_t
                if need_prediction_arrays:
                    trial_pred_tensors.append(preds.detach())
            loss_tensors.append(loss_t.detach().reshape(()))
            m1_tensors.append(m1_t.detach().reshape(()))
            m2_tensors.append(m2_t.detach().reshape(()))
            if need_prediction_arrays:
                trial_label_tensors.append(labels.detach().reshape(-1))

    losses, m1s, m2s = tensor_scalar_sequences_to_float_lists(
        loss_tensors,
        m1_tensors,
        m2_tensors,
    )
    trial_preds = (
        tensor_values_to_numpy_arrays(trial_pred_tensors)
        if need_prediction_arrays
        else None
    )
    trial_labels = (
        tensor_values_to_numpy_arrays(trial_label_tensors)
        if need_prediction_arrays
        else None
    )
    trial_metrics = finalize_probe_trial_metrics(
        losses,
        m1s,
        m2s,
        counts,
        metric_profile=metric_profile,
        is_regression=bool(is_regression),
        preds=trial_preds,
        labels=trial_labels,
    )
    if trial_metrics is None:
        return (float("nan"), float("nan"), float("nan"))
    return trial_metrics
