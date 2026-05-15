"""BERT-base MRPC layer-output Gaussian noise experiments.

This sidecar script is intentionally independent from the BLB/RL launcher. It
loads a fine-tuned MRPC classifier, injects Gaussian noise after Transformer
layer outputs through forward hooks, writes metrics, and renders paper-style
figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_MODEL = "textattack/bert-base-uncased-MRPC"
DEFAULT_TASK = "mrpc"
DEFAULT_SPLIT = "validation"
DEFAULT_OUTPUT_DIR = Path("reports/transformer_noise_mrpc")
DEFAULT_SEED = 20260514


def _nice_float(value: float) -> float:
    return float(f"{value:.12g}")


def build_sigma_grid(
    start: float = 1e-10,
    stop: float = 1e1,
    dense_start: float = 1e-4,
) -> List[float]:
    """Return a sorted unique sigma grid with dense decade multiples.

    The grid always includes the log-decade endpoints from ``start`` to
    ``stop``. From ``dense_start`` onward it also includes 1..9 multiples in
    each decade, matching the requested finer spacing near the degradation
    region.
    """
    if start <= 0 or stop <= 0:
        raise ValueError("sigma bounds must be positive")
    if start > stop:
        raise ValueError("sigma start must not exceed stop")

    values = set()
    min_exp = math.floor(math.log10(start))
    max_exp = math.ceil(math.log10(stop))
    for exp in range(min_exp, max_exp + 1):
        value = 10.0 ** exp
        if start <= value <= stop:
            values.add(_nice_float(value))

    dense_exp = math.floor(math.log10(max(dense_start, start)))
    for exp in range(dense_exp, max_exp + 1):
        decade = 10.0 ** exp
        for multiplier in range(1, 10):
            value = multiplier * decade
            if start <= value <= stop:
                values.add(_nice_float(value))

    values.add(_nice_float(start))
    values.add(_nice_float(stop))
    return sorted(values)


def parse_sigma_values(raw: Optional[str]) -> List[float]:
    if raw is None or raw.strip() == "":
        return build_sigma_grid()
    values = [_nice_float(float(item.strip())) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--sigmas must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("--sigmas values must be positive")
    return sorted(set(values))


def inject_noise_into_layer_output(
    output: Any,
    add_noise: Callable[[Any], Any],
) -> Any:
    """Perturb only the hidden-state tensor while preserving BERT output shape."""
    if isinstance(output, tuple):
        if not output:
            return output
        return (add_noise(output[0]),) + output[1:]
    if isinstance(output, list):
        if not output:
            return output
        return [add_noise(output[0])] + output[1:]
    return add_noise(output)


def aggregate_metric_trials(trials: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not trials:
        raise ValueError("at least one trial is required")
    values = [float(trial["acc"]) for trial in trials]
    return {"acc_mean": statistics.fmean(values)}


def select_mild_drop_sigma(
    rows: Sequence[Mapping[str, float]],
    baseline_acc: float,
    target_drop: float = 0.02,
) -> float:
    """Pick a sigma that mildly affects performance for layer-wise probing."""
    if not rows:
        raise ValueError("cannot select a sigma from empty experiment rows")
    candidates = []
    for row in rows:
        acc_drop = max(0.0, float(baseline_acc) - float(row["acc_mean"]))
        candidates.append((abs(acc_drop - target_drop), -acc_drop, float(row["sigma"])))
    candidates.sort()
    return candidates[0][2]


def accuracy_metric(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
    if len(labels) != len(preds):
        raise ValueError("labels and predictions must have the same length")
    if not labels:
        raise ValueError("cannot compute metrics for an empty prediction set")

    correct = sum(1 for label, pred in zip(labels, preds) if int(label) == int(pred))
    return {"acc": correct / len(labels)}


def resolve_dotted_attr(root: Any, dotted_path: str) -> Any:
    obj = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def choose_device(torch_module: Any, requested: str) -> Any:
    if requested != "auto":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def set_trial_seed(torch_module: Any, seed: int) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


@contextmanager
def temporary_layer_output_noise(
    model: Any,
    layer_indices: Sequence[int],
    sigma: float,
    torch_module: Any,
    layers_attr: str,
):
    """Install forward hooks that add N(0, sigma^2) noise to layer outputs."""
    handles = []
    if sigma > 0.0 and layer_indices:
        layers = list(resolve_dotted_attr(model, layers_attr))
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= len(layers):
                raise IndexError(f"layer index {layer_idx} is outside 0..{len(layers) - 1}")

            def hook(_module: Any, _inputs: Any, output: Any, *, _sigma: float = sigma) -> Any:
                def add_noise(tensor: Any) -> Any:
                    return tensor + torch_module.randn_like(tensor) * _sigma

                return inject_noise_into_layer_output(output, add_noise)

            handles.append(layers[layer_idx].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_mrpc_dataset(
    dataset_name: str,
    task: str,
    split: str,
    dataset_cache_dir: Optional[str],
) -> Any:
    from datasets import DatasetDict, load_dataset, load_from_disk

    if dataset_cache_dir:
        cache_path = Path(dataset_cache_dir).expanduser()
        candidates = [
            cache_path,
            cache_path / task,
            cache_path / "glue" / task,
            cache_path / "nyu-mll" / "glue" / task,
        ]
        for candidate in candidates:
            if candidate.exists():
                loaded = load_from_disk(str(candidate))
                if isinstance(loaded, DatasetDict):
                    return loaded[split]
                return loaded
    return load_dataset(dataset_name, task, split=split)


def build_dataloader(
    tokenizer: Any,
    dataset: Any,
    batch_size: int,
    max_length: int,
    max_samples: Optional[int],
) -> Any:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def tokenize(batch: Mapping[str, Sequence[str]]) -> Mapping[str, Any]:
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    keep_columns = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in tokenized.column_names:
        keep_columns.append("token_type_ids")
    tokenized = tokenized.remove_columns(
        [name for name in tokenized.column_names if name not in keep_columns]
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)


def evaluate_condition(
    model: Any,
    dataloader: Any,
    device: Any,
    torch_module: Any,
    layers_attr: str,
    layer_indices: Sequence[int],
    sigma: float,
    seed: int,
) -> Dict[str, float]:
    set_trial_seed(torch_module, seed)
    model.eval()
    labels_all: List[int] = []
    preds_all: List[int] = []
    started = time.time()
    with temporary_layer_output_noise(
        model=model,
        layer_indices=layer_indices,
        sigma=sigma,
        torch_module=torch_module,
        layers_attr=layers_attr,
    ):
        with torch_module.no_grad():
            for batch in dataloader:
                labels = batch.pop("labels", batch.pop("label", None))
                if labels is None:
                    raise KeyError("dataloader batch does not contain labels")
                labels = labels.to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                labels_all.extend(labels.detach().cpu().tolist())
                preds_all.extend(preds.detach().cpu().tolist())
    metrics = accuracy_metric(labels_all, preds_all)
    metrics["n_samples"] = float(len(labels_all))
    metrics["elapsed_sec"] = time.time() - started
    return metrics


def run_repeated_condition(
    model: Any,
    dataloader: Any,
    device: Any,
    torch_module: Any,
    layers_attr: str,
    layer_indices: Sequence[int],
    sigma: float,
    repeats: int,
    seed: int,
) -> Dict[str, Any]:
    trials = []
    for repeat_idx in range(repeats):
        trial_seed = seed + repeat_idx * 1009
        metrics = evaluate_condition(
            model=model,
            dataloader=dataloader,
            device=device,
            torch_module=torch_module,
            layers_attr=layers_attr,
            layer_indices=layer_indices,
            sigma=sigma,
            seed=trial_seed,
        )
        trials.append(metrics)
    summary = aggregate_metric_trials(trials)
    summary["sigma"] = float(sigma)
    return summary


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def save_results(output_dir: Path, results: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "bert_mrpc_layer_noise_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2, sort_keys=True)

    exp1_rows = results["experiment1"]
    write_csv(
        output_dir / "noise_magnitude_results.csv",
        exp1_rows,
        ["sigma", "acc_mean"],
    )
    exp2_rows = results["experiment2"]["rows"]
    write_csv(
        output_dir / "layer_position_results.csv",
        exp2_rows,
        ["layer", "sigma", "acc_mean"],
    )
    return result_path


def _require_times_new_roman() -> None:
    from matplotlib import font_manager

    has_font = any(font.name == "Times New Roman" for font in font_manager.fontManager.ttflist)
    if not has_font:
        raise RuntimeError("Times New Roman font is required for these figures but was not found.")


def configure_matplotlib_style(plt: Any) -> None:
    _require_times_new_roman()
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 0.75,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _score_ylim_percent(values: Iterable[float], *, min_span: float = 3.0) -> List[float]:
    vals = list(values)
    lo = max(0.0, min(vals) - 0.8)
    hi = min(100.0, max(vals) + 0.8)
    if hi - lo < min_span:
        center = (hi + lo) / 2
        lo = max(0.0, center - min_span / 2)
        hi = min(100.0, center + min_span / 2)
    return [lo, hi]


def stretched_log_positions(
    sigmas: Sequence[float],
    pivot_sigma: float = 1e-1,
    left_exponent: float = 1.7,
    right_exponent: float = 0.65,
) -> List[float]:
    """Map sigma to a stretched log axis with 1e-1 centered."""
    if not sigmas:
        return []
    logs = [math.log10(float(sigma)) for sigma in sigmas]
    lo = min(logs)
    hi = max(logs)
    if hi == lo:
        return [0.0 for _ in logs]
    pivot = min(max(math.log10(pivot_sigma), lo), hi)
    positions = []
    for value in logs:
        if value <= pivot:
            denom = pivot - lo
            norm = 0.0 if denom == 0 else (value - lo) / denom
            positions.append(0.5 * (norm ** left_exponent))
        else:
            denom = hi - pivot
            norm = 0.0 if denom == 0 else (value - pivot) / denom
            positions.append(0.5 + 0.5 * (norm ** right_exponent))
    return positions


def log_tick_positions(sigmas: Sequence[float]) -> tuple[List[float], List[str]]:
    logs = [math.log10(float(sigma)) for sigma in sigmas]
    min_exp = math.ceil(min(logs))
    max_exp = math.floor(max(logs))
    preferred_exps = [-10, -6, -3, -1, 0, 1]
    exps = [exp for exp in preferred_exps if min_exp <= exp <= max_exp]
    for exp in (min_exp, max_exp):
        if exp not in exps:
            exps.append(exp)
    exps = sorted(set(exps))
    tick_sigmas = [10.0 ** exp for exp in exps]
    tick_positions = stretched_log_positions(tick_sigmas)
    tick_labels = [rf"$10^{{{exp}}}$" for exp in exps]
    return tick_positions, tick_labels


def _add_zero_slot_to_stretched_positions(
    positions: Sequence[float],
    zero_slot_width: float = 0.08,
) -> List[float]:
    adjusted = []
    for position in positions:
        if position <= 0.5:
            adjusted.append(zero_slot_width + (0.5 - zero_slot_width) * (position / 0.5))
        else:
            adjusted.append(position)
    return adjusted


def noise_magnitude_accuracy_curve(
    results: Mapping[str, Any],
) -> tuple[List[float], List[float], List[float], List[str]]:
    rows = results["experiment1"]
    sigmas = [float(row["sigma"]) for row in rows]
    positive_positions = _add_zero_slot_to_stretched_positions(stretched_log_positions(sigmas))
    tick_positions, tick_labels = log_tick_positions(sigmas)
    tick_positions = [0.0] + _add_zero_slot_to_stretched_positions(tick_positions)
    tick_labels = ["0"] + tick_labels
    values = [100.0 * float(results["baseline"]["acc"])]
    values.extend(100.0 * float(row["acc_mean"]) for row in rows)
    return [0.0] + positive_positions, values, tick_positions, tick_labels


def _finish_paper_axes(ax: Any, metric_values: Sequence[float]) -> None:
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color("black")
    ax.tick_params(width=0.75, length=3.2)
    ax.set_ylim(_score_ylim_percent(metric_values))
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    if span > 35:
        tick_step = 10.0
    elif span > 12:
        tick_step = 5.0
    elif span > 5:
        tick_step = 1.0
    else:
        tick_step = 0.5
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, which="major", axis="y", color="#E0E0E0", linestyle="--", linewidth=0.55)
    for tick_label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        tick_label.set_fontweight("bold")


def plot_noise_magnitude_accuracy(results: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib_style(plt)
    x_positions, values, tick_positions, tick_labels = noise_magnitude_accuracy_curve(results)

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.plot(x_positions, values, marker="o", markersize=2.5, linewidth=1.25, color="#D55E00")
    ax.set_xlabel("Gaussian Noise Std. Dev.", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Accuracy vs. Uniform Layer-Output Noise", fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(-0.025, 1.025)
    ax.grid(True, which="major", axis="x", color="#D0D0D0", linestyle="--", linewidth=0.55)
    _finish_paper_axes(ax, values)
    fig.savefig(output_dir / "noise_magnitude_accuracy.pdf")
    fig.savefig(output_dir / "noise_magnitude_accuracy.png", dpi=600)
    plt.close(fig)


def layer_position_accuracy_bars(results: Mapping[str, Any]) -> tuple[List[str], List[float]]:
    rows = results["experiment2"]["rows"]
    labels = [str(int(row["layer"])) for row in rows]
    values = [100.0 * float(row["acc_mean"]) for row in rows]
    return labels, values


def plot_layer_position_accuracy(results: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    configure_matplotlib_style(plt)
    labels, bar_values = layer_position_accuracy_bars(results)
    x = np.arange(len(labels))
    values = np.array(bar_values)
    baseline_value = 100.0 * float(results["baseline"]["acc"])

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.bar(x, values, 0.58, color="#0072B2", edgecolor="black", linewidth=0.35)
    ax.axhline(baseline_value, color="#7A7A7A", linestyle="--", linewidth=1.0, label="Clean")
    ax.set_xlabel("Perturbed Transformer Layer", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(f"Accuracy by Noise Injection Layer (std. dev. = {results['experiment2']['sigma']:.2g})",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _finish_paper_axes(ax, values.tolist() + [baseline_value])
    legend = ax.legend(frameon=False, loc="upper right", fontsize=7.0, handlelength=1.5)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.savefig(output_dir / "layer_position_accuracy.pdf")
    fig.savefig(output_dir / "layer_position_accuracy.png", dpi=600)
    plt.close(fig)


def remove_stale_figure_outputs(output_dir: Path) -> None:
    stale_names = [
        "bert_mrpc_noise_sensitivity_combined.pdf",
        "bert_mrpc_noise_sensitivity_combined.png",
        "layer_position_f1.pdf",
        "layer_position_f1.png",
        "layer_position_sensitivity.pdf",
        "layer_position_sensitivity.png",
        "noise_magnitude_f1.pdf",
        "noise_magnitude_f1.png",
        "noise_magnitude_sensitivity.pdf",
        "noise_magnitude_sensitivity.png",
    ]
    for name in stale_names:
        path = output_dir / name
        if path.exists():
            path.unlink()


def render_plots(results: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_figure_outputs(output_dir)
    plot_noise_magnitude_accuracy(results, output_dir)
    plot_layer_position_accuracy(results, output_dir)


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(torch, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    dataset = load_mrpc_dataset(
        dataset_name=args.dataset_name,
        task=args.task,
        split=args.split,
        dataset_cache_dir=args.dataset_cache_dir,
    )
    dataloader = build_dataloader(
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    layers = list(resolve_dotted_attr(model, args.layers_attr))
    layer_count = len(layers)
    sigma_grid = parse_sigma_values(args.sigmas)

    baseline = evaluate_condition(
        model=model,
        dataloader=dataloader,
        device=device,
        torch_module=torch,
        layers_attr=args.layers_attr,
        layer_indices=[],
        sigma=0.0,
        seed=args.seed,
    )
    baseline = {
        "acc": baseline["acc"],
        "n_samples": int(baseline["n_samples"]),
        "elapsed_sec": baseline["elapsed_sec"],
    }

    all_layers = list(range(layer_count))
    experiment1 = []
    for sigma in sigma_grid:
        row = run_repeated_condition(
            model=model,
            dataloader=dataloader,
            device=device,
            torch_module=torch,
            layers_attr=args.layers_attr,
            layer_indices=all_layers,
            sigma=sigma,
            repeats=args.repeats,
            seed=args.seed + int(round(-math.log10(sigma) * 1000)),
        )
        experiment1.append(row)
        print(
            f"[exp1] sigma={sigma:.3g} acc_mean={row['acc_mean']:.4f}",
            flush=True,
        )

    if args.layer_sigma == "auto":
        layer_sigma = select_mild_drop_sigma(
            experiment1,
            baseline_acc=baseline["acc"],
            target_drop=args.target_drop,
        )
    else:
        layer_sigma = float(args.layer_sigma)

    experiment2_rows = []
    for layer_idx in range(layer_count):
        row = run_repeated_condition(
            model=model,
            dataloader=dataloader,
            device=device,
            torch_module=torch,
            layers_attr=args.layers_attr,
            layer_indices=[layer_idx],
            sigma=layer_sigma,
            repeats=args.repeats,
            seed=args.seed + 50000 + layer_idx * 100,
        )
        row["layer"] = layer_idx
        experiment2_rows.append(row)
        print(
            f"[exp2] layer={layer_idx} sigma={layer_sigma:.3g} acc_mean={row['acc_mean']:.4f}",
            flush=True,
        )

    return {
        "metadata": {
            "model": args.model,
            "dataset_name": args.dataset_name,
            "task": args.task,
            "split": args.split,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_samples": args.max_samples,
            "repeats": args.repeats,
            "seed": args.seed,
            "device": str(device),
            "layers_attr": args.layers_attr,
            "layer_count": layer_count,
            "metric": "MRPC accuracy",
        },
        "baseline": baseline,
        "experiment1": experiment1,
        "experiment2": {
            "sigma": layer_sigma,
            "selection": {
                "mode": args.layer_sigma,
                "target_drop": args.target_drop,
            },
            "rows": experiment2_rows,
        },
    }


def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and plot BERT-base MRPC layer-output Gaussian noise experiments."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default="glue")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--layers-attr", default="bert.encoder.layer")
    parser.add_argument(
        "--sigmas",
        default=None,
        help="Comma-separated sigma list. Defaults to 1e-10..10 with dense points from 1e-4.",
    )
    parser.add_argument(
        "--layer-sigma",
        default="0.6",
        help="Fixed sigma for layer-wise experiment, or 'auto' to choose a mild-drop value from experiment 1.",
    )
    parser.add_argument("--target-drop", type=float, default=0.02)
    parser.add_argument(
        "--plot-only",
        default=None,
        help="Path to an existing results JSON. If set, skip evaluation and only render figures.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    output_dir = Path(args.output_dir)
    _require_times_new_roman()

    if args.plot_only:
        results = load_results(Path(args.plot_only))
        render_plots(results, output_dir)
        return 0

    results = run_experiment(args)
    result_path = save_results(output_dir, results)
    render_plots(results, output_dir)
    print(f"[done] wrote {result_path}", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
