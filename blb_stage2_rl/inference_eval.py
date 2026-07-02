"""Shared post-install model inference evaluation helpers.

These helpers cover the part after a BLB/Stage-1 configuration has already
been installed on the model: run the model forward, aggregate loss/logits, and
compute metrics.  Keep this module as the shared seam for RL probes, multi-GPU
probe workers, Paean final/fixed eval, and fixed-action experiments.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .eval_metrics import (
    finalize_probe_trial_metrics,
    logits_to_classes,
    metric_pair_for_dataset,
    probe_batch_sample_count,
    sample_weighted_mean,
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


def output_loss_and_logits(outputs: Any) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    loss = getattr(outputs, "loss", None)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        logits = outputs[1]
    return loss, logits


def concatenate_logits_for_metrics(batch_logits: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(x) for x in batch_logits]
    if not arrays:
        return np.asarray([])
    if all(arr.ndim == 1 for arr in arrays):
        return np.concatenate([arr.reshape(-1) for arr in arrays], axis=0)
    return np.concatenate(
        [arr.reshape(-1, 1) if arr.ndim == 1 else arr for arr in arrays],
        axis=0,
    )


def run_installed_model_on_dataloader(
        model: torch.nn.Module,
        dataloader: Any,
        *,
        device: torch.device,
        metric_profile: str,
        use_train: bool = False,
        split_name: Optional[str] = None,
        mnli_metric2_fn: Optional[Callable[[], float]] = None,
        ) -> InstalledModelEvalResult:
    """Run a full installed-model eval loop on an existing dataloader.

    The caller is responsible for installing any function/noise cfg before this
    function is called and clearing it afterwards when needed.
    """
    del split_name  # Kept in the signature so call sites can pass their context.
    model.eval()
    loss_values: List[float] = []
    loss_counts: List[int] = []
    batch_logits: List[np.ndarray] = []
    batch_labels: List[np.ndarray] = []
    t0 = time.time()
    with torch.inference_mode():
        for batch in dataloader:
            labels = normalize_labels_for_metrics(batch["labels"])
            moved = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**moved)
            loss_t, logits_t = output_loss_and_logits(outputs)
            if loss_t is not None:
                loss_values.append(float(loss_t.detach().item()))
                loss_counts.append(int(labels.size))
            logits = normalize_logits_for_metrics(
                logits_t,
                expected_batch_size=int(labels.size),
            )
            batch_logits.append(logits)
            batch_labels.append(labels)

    avg_loss = sample_weighted_mean(loss_values, loss_counts) if loss_values else 0.0
    n_batches = max(1, len(dataloader))
    avg_time = (time.time() - t0) * 1000.0 / n_batches
    all_logits = concatenate_logits_for_metrics(batch_logits)
    all_labels = (
        np.concatenate([np.asarray(x).reshape(-1) for x in batch_labels], axis=0)
        if batch_labels else np.asarray([])
    )

    ds = str(metric_profile or "").lower()
    if ds == "mnli":
        pred_classes = logits_to_classes(all_logits)
        metric1 = float(np.mean(pred_classes == all_labels)) if all_labels.size else 0.0
        metric2 = float(mnli_metric2_fn()) if (not use_train and mnli_metric2_fn is not None) else metric1
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
        ) -> Tuple[float, float, float]:
    """Run one already-installed noisy probe trial over probe batches."""
    losses: List[float] = []
    m1s: List[float] = []
    m2s: List[float] = []
    counts: List[int] = []
    trial_preds: List[np.ndarray] = []
    trial_labels: List[np.ndarray] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for batch in probe_batches:
                outputs = model(**probe_batch_to_model_kwargs(batch))
                _loss_t, logits = output_loss_and_logits(outputs)
                loss, m1, m2 = compute_probe_batch_metrics(
                    logits,
                    batch.labels,
                    is_regression=bool(is_regression),
                )
                losses.append(loss)
                m1s.append(m1)
                m2s.append(m2)
                counts.append(probe_batch_sample_count(batch.labels))
                preds = (
                    logits.view(-1).detach().cpu().numpy()
                    if is_regression
                    else logits_to_classes_tensor(logits.detach()).detach().cpu().numpy()
                )
                trial_preds.append(preds)
                trial_labels.append(batch.labels.detach().cpu().numpy())
    finally:
        if restore_training and was_training:
            model.train()

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
