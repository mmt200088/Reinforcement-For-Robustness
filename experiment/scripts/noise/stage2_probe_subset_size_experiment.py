#!/usr/bin/env python
"""
Stage-2 RL probe-subset-size experiment on MRPC.

This experiment evaluates how the stability-probe subset size affects repeated
noise-sampling results under two configuration families:
1. Max-action config for GELU / Softmax / noise.
2. A fixed random-control config for GELU / Softmax / noise.

For each family, the experiment runs:
- probe_size in {32, 64, 128, 256}
- K in {5, 10}

This yields 16 groups in total. Each group stores every trial result, plus
mean/std summaries, text reports, CSV tables, JSON output, and plots.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment.scripts.noise.noise_scaling_sweep import (
    build_evaluator,
    maybe_limit_eval_split,
)
from layer_importance_evaluator import (
    GELU_MAP,
    SOFTMAX_MAP,
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
)


DATASET_KEY = "mrpc"
DEFAULT_OUTPUT_DIR = os.path.join(
    "experiment",
    "outputs",
    "noise",
    "stage2_probe_subset_size",
    DATASET_KEY,
)
DEFAULT_PROBE_SIZES = (32, 64, 128, 256)
DEFAULT_K_VALUES = (5, 10)
FAMILY_ORDER = ("max_action", "random_control")
FAMILY_LABELS = {
    "max_action": "Max-Action",
    "random_control": "Random-Control",
}
FAMILY_COLORS = {
    "max_action": "#d55e00",
    "random_control": "#0072b2",
}
MAX_ACTION_GELU_DEGREE = 4
MAX_ACTION_SOFTMAX_DEGREE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-2 RL probe subset size experiment on MRPC."
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe_seed", type=int, default=42)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument(
        "--probe_sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROBE_SIZES),
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--approx_base_config",
        type=str,
        default="glue_configs_best_ppo.json",
    )
    parser.add_argument(
        "--noise_base_config",
        type=str,
        default="glue_noise_configs_best_ppo.json",
    )
    return parser.parse_args()


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type for JSON serialization: {type(obj)!r}")


def _make_max_action_config(total_layers: int) -> Dict[str, object]:
    # For this experiment we define "max action" in degree/scaling space:
    # GELU degree=4, Softmax degree=6, and all noise terms at max scaling.
    gelu_degree = int(MAX_ACTION_GELU_DEGREE)
    softmax_degree = int(MAX_ACTION_SOFTMAX_DEGREE)
    return {
        "family": "max_action",
        "label": FAMILY_LABELS["max_action"],
        "gelu": np.full(total_layers, gelu_degree, dtype=int),
        "softmax": np.full(total_layers, softmax_degree, dtype=int),
        "noise": {
            "input_noise_scaling_factors": np.full(
                total_layers, max(INPUT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int
            ),
            "wq_noise_scaling_factors": np.full(
                total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int
            ),
            "wk_noise_scaling_factors": np.full(
                total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int
            ),
            "wv_noise_scaling_factors": np.full(
                total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int
            ),
            "wo_noise_scaling_factors": np.full(
                total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int
            ),
            "wffn1_noise_scaling_factors": np.full(
                total_layers, max(WFFN1_NOISE_ALLOWED_SCALING_FACTORS), dtype=int
            ),
            "wffn2_noise_scaling_factors": np.full(
                total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int
            ),
        },
    }


def _make_random_control_config(total_layers: int, seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    # Random control excludes GELU degree 0.
    gelu_choices = np.asarray(
        sorted(v for v in set(GELU_MAP.values()) if int(v) != 0),
        dtype=int,
    )
    softmax_choices = np.asarray(sorted(set(SOFTMAX_MAP.values())), dtype=int)
    return {
        "family": "random_control",
        "label": FAMILY_LABELS["random_control"],
        "gelu": rng.choice(gelu_choices, size=total_layers).astype(int),
        "softmax": rng.choice(softmax_choices, size=total_layers).astype(int),
        "noise": {
            "input_noise_scaling_factors": rng.choice(
                np.asarray(INPUT_NOISE_ALLOWED_SCALING_FACTORS, dtype=int),
                size=total_layers,
            ).astype(int),
            "wq_noise_scaling_factors": rng.choice(
                np.asarray(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS, dtype=int),
                size=total_layers,
            ).astype(int),
            "wk_noise_scaling_factors": rng.choice(
                np.asarray(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS, dtype=int),
                size=total_layers,
            ).astype(int),
            "wv_noise_scaling_factors": rng.choice(
                np.asarray(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS, dtype=int),
                size=total_layers,
            ).astype(int),
            "wo_noise_scaling_factors": rng.choice(
                np.asarray(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS, dtype=int),
                size=total_layers,
            ).astype(int),
            "wffn1_noise_scaling_factors": rng.choice(
                np.asarray(WFFN1_NOISE_ALLOWED_SCALING_FACTORS, dtype=int),
                size=total_layers,
            ).astype(int),
            "wffn2_noise_scaling_factors": rng.choice(
                np.asarray(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS, dtype=int),
                size=total_layers,
            ).astype(int),
        },
    }


def _compute_config_costs(evaluator, gelu, softmax, noise_cfg):
    approx_total_cost, gelu_cost, softmax_cost = evaluator.get_simulated_cost(
        np.asarray(gelu, dtype=int),
        np.asarray(softmax, dtype=int),
    )
    noise_total_cost, noise_breakdown = evaluator.get_noise_simulated_cost(**noise_cfg)
    return {
        "approx_total_cost": float(approx_total_cost),
        "gelu_cost": float(gelu_cost),
        "softmax_cost": float(softmax_cost),
        "noise_total_cost": float(noise_total_cost),
        "noise_breakdown": {
            key: float(value) for key, value in noise_breakdown.items()
        },
    }


def _run_group(
    evaluator,
    family_config: Dict[str, object],
    probe_size: int,
    k_value: int,
    probe_seed: int,
):
    noise_cfg = {
        key: np.asarray(value, dtype=int).copy()
        for key, value in family_config["noise"].items()
    }
    stats = evaluator.evaluate_model_with_attention_noise_segmented(
        np.asarray(family_config["gelu"], dtype=int),
        np.asarray(family_config["softmax"], dtype=int),
        segments=int(k_value),
        use_train=False,
        split="validation_full",
        probe_size_override=int(probe_size),
        probe_seed=int(probe_seed),
        **noise_cfg,
    )
    costs = _compute_config_costs(
        evaluator,
        family_config["gelu"],
        family_config["softmax"],
        noise_cfg,
    )
    return {
        "family": family_config["family"],
        "label": family_config["label"],
        "probe_size_requested": int(probe_size),
        "probe_size_observed": int(stats["probe_size"]),
        "k_value": int(k_value),
        "probe_seed": int(probe_seed),
        "evaluation_mode": stats["evaluation_mode"],
        "split_name": stats["split_name"],
        "gelu": np.asarray(family_config["gelu"], dtype=int).copy(),
        "softmax": np.asarray(family_config["softmax"], dtype=int).copy(),
        "noise": noise_cfg,
        "costs": costs,
        "summary": {
            "loss_mean": float(stats["loss_mean"]),
            "loss_std": float(stats["loss_std"]),
            "loss_min": float(stats["loss_min"]),
            "loss_max": float(stats["loss_max"]),
            "p_mean": float(stats["p_mean"]),
            "p_std": float(stats["p_std"]),
            "p_min": float(stats["p_min"]),
            "p_max": float(stats["p_max"]),
            "s_mean": float(stats["s_mean"]),
            "s_std": float(stats["s_std"]),
            "s_min": float(stats["s_min"]),
            "s_max": float(stats["s_max"]),
            "time_mean_ms": float(stats["time_mean_ms"]),
            "time_std_ms": float(stats["time_std_ms"]),
        },
        "trials": [
            {
                "trial": int(idx + 1),
                "loss": float(item["loss"]),
                "p": float(item["p"]),
                "s": float(item["s"]),
                "time_ms": float(item["time_ms"]),
            }
            for idx, item in enumerate(stats["trials"])
        ],
    }


def _write_summary_csv(results: Sequence[Dict[str, object]], output_path: str) -> None:
    fieldnames = [
        "family",
        "label",
        "probe_size_requested",
        "probe_size_observed",
        "k_value",
        "loss_mean",
        "loss_std",
        "p_mean",
        "p_std",
        "s_mean",
        "s_std",
        "time_mean_ms",
        "time_std_ms",
        "approx_total_cost",
        "gelu_cost",
        "softmax_cost",
        "noise_total_cost",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = {
                "family": item["family"],
                "label": item["label"],
                "probe_size_requested": item["probe_size_requested"],
                "probe_size_observed": item["probe_size_observed"],
                "k_value": item["k_value"],
                "loss_mean": item["summary"]["loss_mean"],
                "loss_std": item["summary"]["loss_std"],
                "p_mean": item["summary"]["p_mean"],
                "p_std": item["summary"]["p_std"],
                "s_mean": item["summary"]["s_mean"],
                "s_std": item["summary"]["s_std"],
                "time_mean_ms": item["summary"]["time_mean_ms"],
                "time_std_ms": item["summary"]["time_std_ms"],
                "approx_total_cost": item["costs"]["approx_total_cost"],
                "gelu_cost": item["costs"]["gelu_cost"],
                "softmax_cost": item["costs"]["softmax_cost"],
                "noise_total_cost": item["costs"]["noise_total_cost"],
            }
            writer.writerow(row)


def _write_trials_csv(results: Sequence[Dict[str, object]], output_path: str) -> None:
    fieldnames = [
        "family",
        "label",
        "probe_size_requested",
        "k_value",
        "trial",
        "loss",
        "p",
        "s",
        "time_ms",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            for trial in item["trials"]:
                writer.writerow(
                    {
                        "family": item["family"],
                        "label": item["label"],
                        "probe_size_requested": item["probe_size_requested"],
                        "k_value": item["k_value"],
                        "trial": trial["trial"],
                        "loss": trial["loss"],
                        "p": trial["p"],
                        "s": trial["s"],
                        "time_ms": trial["time_ms"],
                    }
                )


def _format_config_values(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def _write_text_report(
    results: Sequence[Dict[str, object]],
    metric_short_names: Sequence[str],
    output_path: str,
) -> None:
    grouped_configs = {}
    for item in results:
        grouped_configs.setdefault(item["family"], item)

    lines: List[str] = []
    lines.append("Stage-2 RL Probe Subset Size Experiment")
    lines.append("=" * 72)
    lines.append(f"Dataset: {DATASET_KEY.upper()}")
    lines.append(f"Groups: {len(results)}")
    lines.append("")
    lines.append("Configuration Families")
    lines.append("-" * 72)
    for family in FAMILY_ORDER:
        item = grouped_configs[family]
        lines.append(f"[{item['label']}]")
        lines.append(f"GELU    : {_format_config_values(item['gelu'])}")
        lines.append(f"Softmax : {_format_config_values(item['softmax'])}")
        for noise_key, noise_values in item["noise"].items():
            lines.append(f"{noise_key:<28}: {_format_config_values(noise_values)}")
        lines.append(
            "Approx cost / GELU / Softmax / Noise total: "
            f"{item['costs']['approx_total_cost']:.2f} / "
            f"{item['costs']['gelu_cost']:.2f} / "
            f"{item['costs']['softmax_cost']:.2f} / "
            f"{item['costs']['noise_total_cost']:.2f}"
        )
        lines.append("")

    lines.append("Group Summary")
    lines.append("-" * 72)
    header = (
        f"{'Family':<16} {'Probe':>6} {'K':>4} "
        f"{'Loss(mean±std)':>24} "
        f"{metric_short_names[0] + '(mean±std)':>24} "
        f"{metric_short_names[1] + '(mean±std)':>24}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for item in results:
        lines.append(
            f"{item['label']:<16} "
            f"{item['probe_size_requested']:>6} "
            f"{item['k_value']:>4} "
            f"{item['summary']['loss_mean']:>10.4f}±{item['summary']['loss_std']:<10.4f} "
            f"{item['summary']['p_mean']:>10.4f}±{item['summary']['p_std']:<10.4f} "
            f"{item['summary']['s_mean']:>10.4f}±{item['summary']['s_std']:<10.4f}"
        )
    lines.append("")

    lines.append("Per-Group Trial Details")
    lines.append("-" * 72)
    for item in results:
        lines.append(
            f"[{item['label']}] probe={item['probe_size_requested']} "
            f"K={item['k_value']} observed_probe={item['probe_size_observed']}"
        )
        lines.append(
            f"Summary: loss={item['summary']['loss_mean']:.6f}±{item['summary']['loss_std']:.6f}, "
            f"{metric_short_names[0]}={item['summary']['p_mean']:.6f}±{item['summary']['p_std']:.6f}, "
            f"{metric_short_names[1]}={item['summary']['s_mean']:.6f}±{item['summary']['s_std']:.6f}, "
            f"time_ms={item['summary']['time_mean_ms']:.3f}±{item['summary']['time_std_ms']:.3f}"
        )
        for trial in item["trials"]:
            lines.append(
                f"  Trial {trial['trial']:>2}: "
                f"loss={trial['loss']:.6f}, "
                f"{metric_short_names[0]}={trial['p']:.6f}, "
                f"{metric_short_names[1]}={trial['s']:.6f}, "
                f"time_ms={trial['time_ms']:.3f}"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _plot_raw_and_errorbars(
    results: Sequence[Dict[str, object]],
    metric_short_names: Sequence[str],
    output_path: str,
) -> None:
    metric_specs = [
        ("loss", "loss_mean", "loss_std", "Loss"),
        ("p", "p_mean", "p_std", metric_short_names[0]),
        ("s", "s_mean", "s_std", metric_short_names[1]),
        ("time_ms", "time_mean_ms", "time_std_ms", "Time (ms)"),
    ]
    labels = [
        f"{'MAX' if item['family'] == 'max_action' else 'RND'}\nP{item['probe_size_requested']}\nK{item['k_value']}"
        for item in results
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()
    x = np.arange(len(results))

    for ax, (trial_key, mean_key, std_key, ylabel) in zip(axes, metric_specs):
        for idx, item in enumerate(results):
            color = FAMILY_COLORS[item["family"]]
            trial_values = np.asarray(
                [trial[trial_key] for trial in item["trials"]],
                dtype=float,
            )
            jitter = np.linspace(-0.14, 0.14, len(trial_values))
            ax.scatter(
                np.full_like(trial_values, idx, dtype=float) + jitter,
                trial_values,
                color=color,
                alpha=0.45,
                s=28,
                zorder=2,
            )
            ax.errorbar(
                idx,
                item["summary"][mean_key],
                yerr=item["summary"][std_key],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=4,
                markersize=7,
                zorder=3,
            )
        ax.axvline(7.5, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(f"{ylabel}: raw trials + mean±std")

    fig.suptitle("Stage-2 RL Probe Subset Size Experiment on MRPC", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    results: Sequence[Dict[str, object]],
    metric_short_names: Sequence[str],
    probe_sizes: Sequence[int],
    k_values: Sequence[int],
    output_path: str,
) -> None:
    metric_specs = [
        ("loss_mean", "loss_std", "Loss"),
        ("p_mean", "p_std", metric_short_names[0]),
        ("s_mean", "s_std", metric_short_names[1]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for row_idx, family in enumerate(FAMILY_ORDER):
        family_results = {
            (int(item["probe_size_requested"]), int(item["k_value"])): item
            for item in results
            if item["family"] == family
        }
        for col_idx, (mean_key, std_key, title) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            matrix = np.zeros((len(k_values), len(probe_sizes)), dtype=float)
            for i, k_value in enumerate(k_values):
                for j, probe_size in enumerate(probe_sizes):
                    matrix[i, j] = family_results[(probe_size, k_value)]["summary"][mean_key]
            im = ax.imshow(matrix, aspect="auto", cmap="viridis")
            for i, k_value in enumerate(k_values):
                for j, probe_size in enumerate(probe_sizes):
                    item = family_results[(probe_size, k_value)]
                    ax.text(
                        j,
                        i,
                        f"{item['summary'][mean_key]:.4f}\n±{item['summary'][std_key]:.4f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
            ax.set_xticks(range(len(probe_sizes)))
            ax.set_xticklabels([str(v) for v in probe_sizes])
            ax.set_yticks(range(len(k_values)))
            ax.set_yticklabels([str(v) for v in k_values])
            ax.set_xlabel("Probe Size")
            ax.set_ylabel("K")
            ax.set_title(f"{FAMILY_LABELS[family]} - {title}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Mean ± Std Heatmaps", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    probe_sizes = [int(v) for v in args.probe_sizes]
    k_values = [int(v) for v in args.k_values]
    output_dir = _ensure_dir(os.path.normpath(args.output_dir))

    print("=" * 72, flush=True)
    print("Stage-2 RL Probe Subset Size Experiment", flush=True)
    print("=" * 72, flush=True)
    print(f"Dataset       : {DATASET_KEY}", flush=True)
    print(f"Probe sizes   : {probe_sizes}", flush=True)
    print(f"K values      : {k_values}", flush=True)
    print(f"Output dir    : {output_dir}", flush=True)
    resolved_device = args.device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"
        print("[Warning] CUDA unavailable, falling back to CPU.", flush=True)
    print(f"Device        : {resolved_device}", flush=True)
    print("=" * 72, flush=True)

    evaluator = None
    try:
        evaluator = build_evaluator(
            task_name=DATASET_KEY,
            device=resolved_device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            output_dir=output_dir,
            approx_base_config_path=os.path.abspath(args.approx_base_config),
            noise_base_config_path=os.path.abspath(args.noise_base_config),
            eval_split="validation_full",
            seed=args.seed,
        )
        maybe_limit_eval_split(
            evaluator=evaluator,
            split_name="validation_full",
            max_eval_samples=int(args.max_eval_samples),
            seed=args.seed,
        )

        family_configs = [
            _make_max_action_config(evaluator.total_layers),
            _make_random_control_config(evaluator.total_layers, args.seed),
        ]
        metric_short_names = evaluator.get_metric_short_names()
        if len(metric_short_names) < 2:
            metric_short_names = list(metric_short_names) + ["metric2"]

        results: List[Dict[str, object]] = []
        for family_config in family_configs:
            print(f"\n[Family] {family_config['label']}", flush=True)
            for probe_size in probe_sizes:
                for k_value in k_values:
                    print(
                        f"  -> probe_size={probe_size}, K={k_value}",
                        flush=True,
                    )
                    result = _run_group(
                        evaluator=evaluator,
                        family_config=family_config,
                        probe_size=probe_size,
                        k_value=k_value,
                        probe_seed=args.probe_seed,
                    )
                    results.append(result)
                    print(
                        "     "
                        f"loss={result['summary']['loss_mean']:.4f}±{result['summary']['loss_std']:.4f}, "
                        f"{metric_short_names[0]}={result['summary']['p_mean']:.4f}±{result['summary']['p_std']:.4f}, "
                        f"{metric_short_names[1]}={result['summary']['s_mean']:.4f}±{result['summary']['s_std']:.4f}",
                        flush=True,
                    )

        results.sort(
            key=lambda item: (
                FAMILY_ORDER.index(item["family"]),
                int(item["probe_size_requested"]),
                int(item["k_value"]),
            )
        )

        report = {
            "dataset": DATASET_KEY,
            "probe_sizes": probe_sizes,
            "k_values": k_values,
            "seed": int(args.seed),
            "probe_seed": int(args.probe_seed),
            "metric_short_names": list(metric_short_names[:2]),
            "results": results,
        }

        json_path = os.path.join(output_dir, "stage2_probe_subset_size_results.json")
        txt_path = os.path.join(output_dir, "stage2_probe_subset_size_results.txt")
        summary_csv_path = os.path.join(output_dir, "stage2_probe_subset_size_summary.csv")
        trials_csv_path = os.path.join(output_dir, "stage2_probe_subset_size_trials.csv")
        plot_trials_path = os.path.join(output_dir, "stage2_probe_subset_size_trials.png")
        plot_heatmap_path = os.path.join(output_dir, "stage2_probe_subset_size_heatmaps.png")

        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=_json_default)
        _write_text_report(results, metric_short_names[:2], txt_path)
        _write_summary_csv(results, summary_csv_path)
        _write_trials_csv(results, trials_csv_path)
        _plot_raw_and_errorbars(results, metric_short_names[:2], plot_trials_path)
        _plot_heatmaps(
            results,
            metric_short_names[:2],
            probe_sizes=probe_sizes,
            k_values=k_values,
            output_path=plot_heatmap_path,
        )

        print("\nOutputs", flush=True)
        print(f"  JSON : {json_path}", flush=True)
        print(f"  TXT  : {txt_path}", flush=True)
        print(f"  CSV1 : {summary_csv_path}", flush=True)
        print(f"  CSV2 : {trials_csv_path}", flush=True)
        print(f"  PNG1 : {plot_trials_path}", flush=True)
        print(f"  PNG2 : {plot_heatmap_path}", flush=True)
    finally:
        if evaluator is not None:
            try:
                evaluator.clear_weight_noise_configuration()
                evaluator.clear_input_noise_configuration()
                evaluator.reversible_handler.restore_all()
            except Exception:
                pass
            del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
