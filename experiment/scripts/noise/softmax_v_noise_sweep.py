#!/usr/bin/env python
"""Sweep fresh noise on attention softmax and V activations.

This experiment fixes GELU degree to 4 and Softmax degree to 6 for every layer.
The only scanned variables are:
  1. attention softmax fresh-noise scaling factor
  2. attention V-activation fresh-noise scaling factor

The attention product is evaluated as:
    (softmax + e1) @ (V + e2)

By default, existing x and W noise terms are held at their current maximum
scaling factors, which is the minimum-noise setting in the current RL/GA search
space. Each grid point is evaluated 5 times on validation_full and plotted with
the mean metrics unless the CLI overrides those defaults.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment.scripts.noise.noise_scaling_sweep import (
    ALL_TASKS,
    TASK_REGISTRY,
    build_evaluator,
    clean_number,
    maybe_limit_eval_split,
    set_global_seed,
    summarize_series,
)
from function_handler import (
    SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS,
    get_input_noise_variance,
)
from layer_importance_evaluator import LayerImportanceEvaluator


DEFAULT_OUTPUT_DIR = os.path.join(
    "experiment",
    "outputs",
    "noise",
    "softmax_v_sweep",
)
DEFAULT_GELU_DEGREE = 4
DEFAULT_SOFTMAX_DEGREE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D sweep for fresh noise on attention softmax and V activations."
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_split", type=str, default="validation_full")
    parser.add_argument("--repeat_n", type=int, default=5)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scaling_factors",
        type=int,
        nargs="+",
        default=list(SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS),
        help="Scaling factors scanned for both softmax and V noise.",
    )
    parser.add_argument(
        "--base_noise_mode",
        type=str,
        choices=("max", "none"),
        default="max",
        help="max keeps x/W noise at current maximum scaling factors; none disables x/W noise.",
    )
    parser.add_argument(
        "--approx_base_config",
        type=str,
        default="glue_final_configs_best_ppo.json",
        help="Only used to initialize the shared evaluator path layout.",
    )
    return parser.parse_args()


def resolve_tasks(raw_tasks: Optional[Sequence[str]]) -> List[str]:
    if raw_tasks is None:
        return list(ALL_TASKS)
    tasks = []
    for task in raw_tasks:
        key = str(task).strip().lower()
        if key not in TASK_REGISTRY:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: {', '.join(ALL_TASKS)}"
            )
        tasks.append(key)
    return tasks


def resolve_scaling_factors(raw_values: Sequence[int]) -> Tuple[int, ...]:
    allowed = set(SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS)
    values = tuple(int(v) for v in raw_values)
    invalid = sorted(set(values) - allowed)
    if invalid:
        raise ValueError(
            f"Unsupported scaling factors: {invalid}. "
            f"Allowed values: {list(SOFTMAX_VALUE_NOISE_ALLOWED_SCALING_FACTORS)}"
        )
    return tuple(sorted(dict.fromkeys(values)))


def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def make_trial_seed(
    base_seed: int,
    dataset_idx: int,
    softmax_idx: int,
    value_idx: int,
    trial_idx: int,
) -> int:
    return int(
        base_seed
        + dataset_idx * 1_000_000
        + softmax_idx * 10_000
        + value_idx * 100
        + trial_idx
    )


def fixed_approx_config(total_layers: int) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.full(total_layers, DEFAULT_GELU_DEGREE, dtype=int),
        np.full(total_layers, DEFAULT_SOFTMAX_DEGREE, dtype=int),
    )


def base_noise_config(evaluator: LayerImportanceEvaluator, mode: str) -> Dict[str, np.ndarray]:
    if mode == "none":
        return {}
    return evaluator._get_max_noise_configuration()


def to_list_config(config: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
    return {key: np.asarray(value, dtype=int).tolist() for key, value in config.items()}


def summarize_series_with_variance(values: Sequence[float]) -> dict:
    summary = summarize_series(values)
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    summary["variance"] = None if finite.size == 0 else float(np.var(finite))
    return summary


def evaluate_grid_point(
    evaluator: LayerImportanceEvaluator,
    fixed_gelu: np.ndarray,
    fixed_softmax: np.ndarray,
    base_noise: Dict[str, np.ndarray],
    softmax_factor: int,
    value_factor: int,
    repeat_n: int,
    split_name: str,
    dataset_idx: int,
    softmax_idx: int,
    value_idx: int,
    seed: int,
) -> dict:
    losses = []
    metric1_values = []
    metric2_values = []
    times = []
    trial_seeds = []
    total_layers = evaluator.total_layers
    softmax_noise = np.full(total_layers, int(softmax_factor), dtype=int)
    value_noise = np.full(total_layers, int(value_factor), dtype=int)

    for trial_idx in range(max(1, int(repeat_n))):
        trial_seed = make_trial_seed(
            base_seed=seed,
            dataset_idx=dataset_idx,
            softmax_idx=softmax_idx,
            value_idx=value_idx,
            trial_idx=trial_idx,
        )
        trial_seeds.append(trial_seed)
        set_global_seed(trial_seed)
        loss, metric1, metric2, elapsed_ms = evaluator.evaluate_model_with_softmax_value_noise(
            fixed_gelu,
            fixed_softmax,
            softmax_noise_scaling_factors=softmax_noise,
            value_noise_scaling_factors=value_noise,
            use_train=False,
            split=split_name,
            **base_noise,
        )
        losses.append(float(loss))
        metric1_values.append(float(metric1))
        metric2_values.append(float(metric2))
        times.append(float(elapsed_ms))

    return {
        "softmax_scaling_factor": int(softmax_factor),
        "v_scaling_factor": int(value_factor),
        "softmax_variance_fresh": clean_number(
            get_input_noise_variance(int(softmax_factor), distribution="fresh")
        ),
        "v_variance_fresh": clean_number(
            get_input_noise_variance(int(value_factor), distribution="fresh")
        ),
        "trial_seeds": trial_seeds,
        "loss": summarize_series_with_variance(losses),
        "m1": summarize_series_with_variance(metric1_values),
        "m2": summarize_series_with_variance(metric2_values),
        "time_ms": summarize_series_with_variance(times),
    }


def run_task_grid(
    evaluator: LayerImportanceEvaluator,
    dataset_key: str,
    dataset_idx: int,
    scaling_factors: Sequence[int],
    repeat_n: int,
    split_name: str,
    seed: int,
    base_noise_mode: str,
) -> dict:
    metric_names = evaluator.get_metric_short_names()
    fixed_gelu, fixed_softmax = fixed_approx_config(evaluator.total_layers)
    base_noise = base_noise_config(evaluator, base_noise_mode)
    records = []

    print(
        f"[{dataset_key.upper()}] Fixed GELU={DEFAULT_GELU_DEGREE}, "
        f"Softmax={DEFAULT_SOFTMAX_DEGREE}; base_noise_mode={base_noise_mode}",
        flush=True,
    )
    for softmax_idx, softmax_factor in enumerate(scaling_factors):
        for value_idx, value_factor in enumerate(scaling_factors):
            record = evaluate_grid_point(
                evaluator=evaluator,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                base_noise=base_noise,
                softmax_factor=int(softmax_factor),
                value_factor=int(value_factor),
                repeat_n=repeat_n,
                split_name=split_name,
                dataset_idx=dataset_idx,
                softmax_idx=softmax_idx,
                value_idx=value_idx,
                seed=seed,
            )
            records.append(record)
            loss_mean = record["loss"]["mean"]
            m1_mean = record["m1"]["mean"]
            m2_mean = record["m2"]["mean"]
            print(
                f"  softmax_sf={softmax_factor:>2}, v_sf={value_factor:>2} "
                f"=> loss={loss_mean:.6f}, m1={m1_mean:.6f}, m2={m2_mean:.6f}",
                flush=True,
            )

    return {
        "dataset": dataset_key,
        "eval_split": split_name,
        "repeat_n": int(repeat_n),
        "metric_short_names": list(metric_names),
        "fixed_approx_config": {
            "gelu": fixed_gelu.tolist(),
            "softmax": fixed_softmax.tolist(),
        },
        "scaling_factors": list(map(int, scaling_factors)),
        "noise_distribution": "fresh",
        "base_noise_mode": base_noise_mode,
        "base_noise_config": to_list_config(base_noise),
        "records": records,
    }


def write_csv(summary: dict, output_path: str) -> None:
    fieldnames = [
        "dataset",
        "softmax_scaling_factor",
        "v_scaling_factor",
        "softmax_variance_fresh",
        "v_variance_fresh",
        "loss_mean",
        "loss_std",
        "loss_variance",
        "loss_min",
        "loss_max",
        "m1_mean",
        "m1_std",
        "m1_variance",
        "m1_min",
        "m1_max",
        "m2_mean",
        "m2_std",
        "m2_variance",
        "m2_min",
        "m2_max",
        "time_ms_mean",
        "time_ms_variance",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in summary["records"]:
            writer.writerow(
                {
                    "dataset": summary["dataset"],
                    "softmax_scaling_factor": record["softmax_scaling_factor"],
                    "v_scaling_factor": record["v_scaling_factor"],
                    "softmax_variance_fresh": record["softmax_variance_fresh"],
                    "v_variance_fresh": record["v_variance_fresh"],
                    "loss_mean": record["loss"]["mean"],
                    "loss_std": record["loss"]["std"],
                    "loss_variance": record["loss"]["variance"],
                    "loss_min": record["loss"]["min"],
                    "loss_max": record["loss"]["max"],
                    "m1_mean": record["m1"]["mean"],
                    "m1_std": record["m1"]["std"],
                    "m1_variance": record["m1"]["variance"],
                    "m1_min": record["m1"]["min"],
                    "m1_max": record["m1"]["max"],
                    "m2_mean": record["m2"]["mean"],
                    "m2_std": record["m2"]["std"],
                    "m2_variance": record["m2"]["variance"],
                    "m2_min": record["m2"]["min"],
                    "m2_max": record["m2"]["max"],
                    "time_ms_mean": record["time_ms"]["mean"],
                    "time_ms_variance": record["time_ms"]["variance"],
                }
            )


def _best_record(records: Sequence[dict], metric: str, maximize: bool) -> Optional[dict]:
    finite = [
        record
        for record in records
        if record.get(metric, {}).get("mean") is not None
        and np.isfinite(float(record[metric]["mean"]))
    ]
    if not finite:
        return None
    return sorted(
        finite,
        key=lambda record: float(record[metric]["mean"]),
        reverse=bool(maximize),
    )[0]


def write_text_report(summary: dict, output_path: str) -> None:
    records = summary["records"]
    best_loss = _best_record(records, "loss", maximize=False)
    best_m1 = _best_record(records, "m1", maximize=True)
    best_m2 = _best_record(records, "m2", maximize=True)

    def fmt_best(label: str, record: Optional[dict], metric: str) -> str:
        if record is None:
            return f"{label}: N/A"
        return (
            f"{label}: softmax_sf={record['softmax_scaling_factor']}, "
            f"v_sf={record['v_scaling_factor']}, "
            f"{metric}_mean={record[metric]['mean']:.6f}"
        )

    lines = [
        f"Dataset: {summary['dataset'].upper()}",
        f"Eval split: {summary['eval_split']}",
        f"Repeats per point: {summary['repeat_n']}",
        "Fixed approximation: GELU degree 4, Softmax degree 6 on all layers",
        f"Base x/W noise mode: {summary['base_noise_mode']}",
        "Scanned noise distribution: fresh",
        f"Scaling factors: {summary['scaling_factors']}",
        f"Metric names: {summary['metric_short_names']}",
        "",
        "Best points:",
        fmt_best("Lowest loss", best_loss, "loss"),
        fmt_best("Highest m1", best_m1, "m1"),
        fmt_best("Highest m2", best_m2, "m2"),
        "",
        "Grid records:",
        (
            "softmax_sf\tv_sf\t"
            "loss_mean\tloss_variance\tloss_std\t"
            "m1_mean\tm1_variance\tm1_std\t"
            "m2_mean\tm2_variance\tm2_std"
        ),
    ]
    for record in records:
        lines.append(
            f"{record['softmax_scaling_factor']}\t"
            f"{record['v_scaling_factor']}\t"
            f"{record['loss']['mean']:.8f}\t"
            f"{record['loss']['variance']:.8f}\t"
            f"{record['loss']['std']:.8f}\t"
            f"{record['m1']['mean']:.8f}\t"
            f"{record['m1']['variance']:.8f}\t"
            f"{record['m1']['std']:.8f}\t"
            f"{record['m2']['mean']:.8f}\t"
            f"{record['m2']['variance']:.8f}\t"
            f"{record['m2']['std']:.8f}"
        )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _metric_matrix(
    records: Sequence[dict],
    scaling_factors: Sequence[int],
    metric: str,
    stat: str = "mean",
) -> np.ndarray:
    index = {int(value): idx for idx, value in enumerate(scaling_factors)}
    matrix = np.full((len(scaling_factors), len(scaling_factors)), np.nan, dtype=float)
    for record in records:
        row = index[int(record["v_scaling_factor"])]
        col = index[int(record["softmax_scaling_factor"])]
        value = record[metric][stat]
        matrix[row, col] = np.nan if value is None else float(value)
    return matrix


def plot_3d_surfaces(summary: dict, output_path: str) -> None:
    factors = np.asarray(summary["scaling_factors"], dtype=float)
    softmax_grid, value_grid = np.meshgrid(factors, factors)
    metric_names = summary.get("metric_short_names") or []
    metric_labels = {
        "loss": "Loss",
        "m1": f"m1 ({metric_names[0]})" if metric_names else "m1",
        "m2": f"m2 ({metric_names[1]})" if len(metric_names) > 1 else "m2",
    }
    cmaps = {"loss": "viridis", "m1": "plasma", "m2": "cividis"}

    fig = plt.figure(figsize=(18, 5.8))
    fig.suptitle(
        f"Softmax/V Fresh-Noise Scaling Sweep ({summary['dataset'].upper()})",
        fontsize=14,
        fontweight="bold",
    )

    for panel_idx, metric in enumerate(("loss", "m1", "m2"), start=1):
        ax = fig.add_subplot(1, 3, panel_idx, projection="3d")
        z_values = _metric_matrix(summary["records"], summary["scaling_factors"], metric)
        if len(factors) >= 2:
            surface = ax.plot_surface(
                softmax_grid,
                value_grid,
                z_values,
                cmap=cmaps[metric],
                edgecolor="none",
                linewidth=0,
                antialiased=True,
                alpha=0.92,
            )
            fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08)
        else:
            ax.scatter(
                softmax_grid.ravel(),
                value_grid.ravel(),
                z_values.ravel(),
                c=z_values.ravel(),
                cmap=cmaps[metric],
                s=48,
            )
        ax.set_title(metric_labels[metric])
        ax.set_xlabel("Softmax scaling factor")
        ax.set_ylabel("V scaling factor")
        ax.set_zlabel(metric_labels[metric])
        ax.view_init(elev=26, azim=-132)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _grid_edges(values: Sequence[int]) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.size == 1:
        return np.asarray([centers[0] - 1.0, centers[0] + 1.0], dtype=float)
    mids = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def plot_2d_heatmaps(summary: dict, output_path: str) -> None:
    factors = summary["scaling_factors"]
    x_edges = _grid_edges(factors)
    y_edges = _grid_edges(factors)
    metric_names = summary.get("metric_short_names") or []
    metric_labels = {
        "loss": "Loss",
        "m1": f"m1 ({metric_names[0]})" if metric_names else "m1",
        "m2": f"m2 ({metric_names[1]})" if len(metric_names) > 1 else "m2",
    }
    cmaps = {"loss": "viridis", "m1": "plasma", "m2": "cividis"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.4), constrained_layout=True)
    fig.suptitle(
        f"Softmax/V Fresh-Noise Sweep Mean and Variance ({summary['dataset'].upper()})",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, stat in enumerate(("mean", "variance")):
        for col_idx, metric in enumerate(("loss", "m1", "m2")):
            ax = axes[row_idx, col_idx]
            matrix = _metric_matrix(summary["records"], factors, metric, stat=stat)
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                matrix,
                cmap=cmaps[metric],
                shading="flat",
            )
            if len(factors) >= 3:
                x_centers, y_centers = np.meshgrid(
                    np.asarray(factors, dtype=float),
                    np.asarray(factors, dtype=float),
                )
                ax.contour(
                    x_centers,
                    y_centers,
                    matrix,
                    colors="white",
                    linewidths=0.7,
                    alpha=0.65,
                )
            if len(factors) <= 5:
                for value_row, value_factor in enumerate(factors):
                    for softmax_col, softmax_factor in enumerate(factors):
                        value = matrix[value_row, softmax_col]
                        if np.isfinite(value):
                            ax.text(
                                softmax_factor,
                                value_factor,
                                f"{value:.4g}",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="white",
                            )
            stat_label = "Mean" if stat == "mean" else "Variance"
            ax.set_title(f"{metric_labels[metric]} {stat_label}")
            ax.set_xlabel("Softmax scaling factor")
            ax.set_ylabel("V scaling factor")
            ax.set_xticks(factors)
            ax.set_yticks(factors)
            fig.colorbar(mesh, ax=ax, shrink=0.82)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_2d_change_curves(summary: dict, output_path: str) -> None:
    factors = np.asarray(summary["scaling_factors"], dtype=int)
    metric_names = summary.get("metric_short_names") or []
    metric_labels = {
        "loss": "Loss",
        "m1": f"m1 ({metric_names[0]})" if metric_names else "m1",
        "m2": f"m2 ({metric_names[1]})" if len(metric_names) > 1 else "m2",
    }
    records_by_v = {
        int(v_factor): sorted(
            [
                record
                for record in summary["records"]
                if int(record["v_scaling_factor"]) == int(v_factor)
            ],
            key=lambda record: int(record["softmax_scaling_factor"]),
        )
        for v_factor in factors
    }
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(factors.min()), vmax=float(factors.max()))

    fig, axes = plt.subplots(3, 2, figsize=(15.5, 13.2), constrained_layout=True)
    fig.suptitle(
        f"2D Change Curves by V Scaling Factor ({summary['dataset'].upper()})",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, metric in enumerate(("loss", "m1", "m2")):
        for col_idx, stat in enumerate(("mean", "variance")):
            ax = axes[row_idx, col_idx]
            for value_factor, records in records_by_v.items():
                x_values = [int(record["softmax_scaling_factor"]) for record in records]
                y_values = [
                    np.nan
                    if record[metric][stat] is None
                    else float(record[metric][stat])
                    for record in records
                ]
                ax.plot(
                    x_values,
                    y_values,
                    color=cmap(norm(value_factor)),
                    linewidth=1.4,
                    alpha=0.92,
                )
            stat_label = "Mean" if stat == "mean" else "Variance"
            ax.set_title(f"{metric_labels[metric]} {stat_label}")
            ax.set_xlabel("Softmax scaling factor")
            ax.set_ylabel(f"{metric_labels[metric]} {stat_label.lower()}")
            ax.set_xticks(factors)
            ax.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.72, label="V scaling factor")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main() -> int:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    scaling_factors = resolve_scaling_factors(args.scaling_factors)
    repeat_n = max(1, int(args.repeat_n))
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[Device] CUDA unavailable; falling back to CPU.", flush=True)
        device = "cpu"

    print("============================================================", flush=True)
    print("  Softmax/V Fresh-Noise Sweep", flush=True)
    print("============================================================", flush=True)
    print(f"  Tasks          : {tasks}", flush=True)
    print(f"  Output dir     : {args.output_dir}", flush=True)
    print(f"  Device         : {device}", flush=True)
    print(f"  Eval split     : {args.eval_split}", flush=True)
    print(f"  Repeat n       : {repeat_n}", flush=True)
    print(f"  Scaling factors: {list(scaling_factors)}", flush=True)
    print(f"  Base x/W noise : {args.base_noise_mode}", flush=True)
    print("============================================================", flush=True)

    for dataset_idx, task_name in enumerate(tasks):
        evaluator = None
        try:
            print(f"\n=== Running {task_name.upper()} ===", flush=True)
            set_global_seed(args.seed + dataset_idx)
            evaluator = build_evaluator(
                task_name=task_name,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
                approx_base_config_path=os.path.abspath(args.approx_base_config),
                eval_split=args.eval_split,
                seed=args.seed + dataset_idx,
            )
            maybe_limit_eval_split(
                evaluator,
                split_name=args.eval_split,
                max_eval_samples=int(args.max_eval_samples),
                seed=args.seed + dataset_idx,
            )
            summary = run_task_grid(
                evaluator=evaluator,
                dataset_key=task_name,
                dataset_idx=dataset_idx,
                scaling_factors=scaling_factors,
                repeat_n=repeat_n,
                split_name=args.eval_split,
                seed=args.seed,
                base_noise_mode=args.base_noise_mode,
            )

            json_path = os.path.join(
                args.output_dir,
                f"softmax_v_noise_sweep_{task_name}.json",
            )
            csv_path = os.path.join(
                args.output_dir,
                f"softmax_v_noise_sweep_{task_name}.csv",
            )
            text_path = os.path.join(
                args.output_dir,
                f"softmax_v_noise_sweep_{task_name}.txt",
            )
            plot_path = os.path.join(
                args.output_dir,
                f"softmax_v_noise_sweep_{task_name}_3d.png",
            )
            heatmap_path = os.path.join(
                args.output_dir,
                f"softmax_v_noise_sweep_{task_name}_2d.png",
            )
            curve_path = os.path.join(
                args.output_dir,
                f"softmax_v_noise_sweep_{task_name}_2d_curves.png",
            )
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False, default=json_default)
            write_csv(summary, csv_path)
            write_text_report(summary, text_path)
            plot_3d_surfaces(summary, plot_path)
            plot_2d_heatmaps(summary, heatmap_path)
            plot_2d_change_curves(summary, curve_path)
            print(f"[Saved] {json_path}", flush=True)
            print(f"[Saved] {csv_path}", flush=True)
            print(f"[Saved] {text_path}", flush=True)
            print(f"[Saved] {plot_path}", flush=True)
            print(f"[Saved] {heatmap_path}", flush=True)
            print(f"[Saved] {curve_path}", flush=True)
        finally:
            if evaluator is not None:
                try:
                    evaluator.clear_softmax_value_noise_configuration()
                    evaluator.clear_weight_noise_configuration()
                    evaluator.clear_input_noise_configuration()
                except Exception:
                    pass
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll softmax/V noise sweep experiments finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
