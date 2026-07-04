#!/usr/bin/env python3
"""Repeat a fixed Stage-1 GELU/Softmax plaintext eval and check determinism."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from function_handler import ReversibleLayerHandler
from json_utils import write_json_file
from layer_importance_evaluator import LayerImportanceEvaluator


_TASK_FIELDS = {
    "sst2": ("sentence", None),
    "rte": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
}


def _load_glue_validation(task: str):
    from datasets import load_dataset

    return load_dataset("glue", task, split="validation")


def _noise_state_counts(handler: ReversibleLayerHandler) -> dict:
    projection = getattr(handler, "original_projection_noise", {})
    return {
        "input_noise_layers": len(getattr(handler, "original_input_noise", {})),
        "softmax_value_noise_layers": len(getattr(handler, "original_softmax_value_noise", {})),
        "projection_noise_layers": sum(len(v) for v in projection.values()),
        "block1_noise_layers": len(getattr(handler, "blb_block1_state", {})),
        "block2_noise_layers": len(getattr(handler, "blb_block2_state", {})),
        "block3_noise_layers": len(getattr(handler, "blb_block3_state", {})),
        "block4_noise_layers": len(getattr(handler, "blb_block4_state", {})),
        "block5_noise_layers": len(getattr(handler, "blb_block5_state", {})),
        "first_input_noise_layers": len(getattr(handler, "blb_first_input_noise_state", {})),
    }


def _all_zero(values: dict) -> bool:
    return all(int(v) == 0 for v in values.values())


def _build_dataloader(tokenizer, task: str, batch_size: int, max_samples: int | None):
    field_a, field_b = _TASK_FIELDS[task]
    dataset = _load_glue_validation(task)
    if max_samples is not None:
        dataset = dataset.select(range(min(int(max_samples), len(dataset))))

    def _collate(batch):
        texts_a = [row[field_a] for row in batch]
        if field_b is None:
            enc = tokenizer(texts_a, padding=True, truncation=True, max_length=256, return_tensors="pt")
        else:
            texts_b = [row[field_b] for row in batch]
            enc = tokenizer(texts_a, texts_b, padding=True, truncation=True, max_length=256, return_tensors="pt")
        labels = torch.tensor([int(row["label"]) for row in batch], dtype=torch.long)
        enc["labels"] = labels
        return enc

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    ), len(dataset)


def _evaluate(model, dataloader, device: torch.device) -> dict:
    total = 0
    loss_sum_t = torch.zeros((), dtype=torch.float64, device=device)
    correct_t = torch.zeros((), dtype=torch.long, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        model.eval()
        for batch in dataloader:
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            logits = model(**batch).logits
            loss_sum_t += loss_fn(logits, labels).detach().to(dtype=torch.float64)
            preds = torch.argmax(logits, dim=-1)
            correct_t += (preds == labels).sum()
            total += int(labels.numel())
    loss_sum = float(loss_sum_t.detach().cpu().item())
    correct = int(correct_t.detach().cpu().item())
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(_TASK_FIELDS), default="sst2")
    parser.add_argument("--model-name", default="textattack/bert-base-uncased-SST-2")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260529)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name).to(device)
    model.eval()
    handler = ReversibleLayerHandler(model)
    n_layers = int(model.config.num_hidden_layers)
    fixed_gelu = [4] * n_layers
    fixed_softmax = [6] * n_layers

    dummy_self = SimpleNamespace(
        model=model,
        reversible_handler=handler,
        layers_attribute="bert.encoder.layer",
    )
    LayerImportanceEvaluator.apply_configuration(dummy_self, fixed_gelu, fixed_softmax)
    noise_state_after_install = _noise_state_counts(handler)

    dataloader, dataset_size = _build_dataloader(
        tokenizer, args.task, args.batch_size, args.max_samples
    )
    runs = []
    for idx in range(int(args.repeats)):
        metrics = _evaluate(model, dataloader, device)
        metrics["repeat_index"] = idx
        runs.append(metrics)

    first = runs[0]
    max_loss_abs_diff = max(abs(float(run["loss"]) - float(first["loss"])) for run in runs)
    max_accuracy_abs_diff = max(abs(float(run["accuracy"]) - float(first["accuracy"])) for run in runs)
    identical_metrics = (
        max_loss_abs_diff == 0.0
        and max_accuracy_abs_diff == 0.0
        and all(int(run["total"]) == int(first["total"]) for run in runs)
    )

    summary = {
        "task": args.task,
        "model_name": args.model_name,
        "device": str(device),
        "dataset_split": "validation",
        "dataset_size": int(dataset_size),
        "batch_size": int(args.batch_size),
        "repeats": int(args.repeats),
        "gelu_config": fixed_gelu,
        "softmax_config": fixed_softmax,
        "noise_state_after_stage1_install": noise_state_after_install,
        "no_stage2_noise_hooks_installed": _all_zero(noise_state_after_install),
        "runs": runs,
        "max_loss_abs_diff": max_loss_abs_diff,
        "max_accuracy_abs_diff": max_accuracy_abs_diff,
        "identical_metrics": bool(identical_metrics),
        "finite_metrics": all(
            math.isfinite(float(run["loss"])) and math.isfinite(float(run["accuracy"]))
            for run in runs
        ),
    }

    out = Path(args.output_json)
    write_json_file(out, summary)
    json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    if not summary["no_stage2_noise_hooks_installed"] or not summary["identical_metrics"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
