#!/usr/bin/env python3
"""Plot activation/weight modal-magnitude heatmaps for MRPC BERT-base.

The source statistics are histogram CSVs.  For each layer/group, this script
finds the most-populated non-zero absolute-value bin and represents that
bin by its geometric center.  The plotted unit is log10(mode magnitude), so
one colorbar unit corresponds to one order of magnitude.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_ACTIVATION_CSV = Path(
    "/var/tmp/root-home/Reinforcement-For-Robustness/"
    "Model_analysis/all_analysis_approx/mrpc/mrpc_magnitude_stats.csv"
)
DEFAULT_WEIGHT_CSV = Path(
    "/var/tmp/root-home/Reinforcement-For-Robustness/"
    "Model_analysis/model_statistics/weight_hist_out/BERT-Base-MRPC_magnitude_hist.csv"
)


ACTIVATION_GROUPS = [
    ("Embed out", ["after_embed"]),
    ("Q/K proj act", ["query_proj", "key_proj"]),
    ("V proj act", ["value_proj"]),
    ("Attn-O act", ["attn_output"]),
    ("Post-attn LN act", ["post_attn_ln"]),
    ("FFN1/GELU input act", ["gelu_input"]),
    ("FFN2 out act", ["ffn2_output"]),
    ("Post-FFN LN act", ["post_ffn_ln"]),
]

PARAM_GROUPS = [
    ("Q/K weight", ["Attn/Q/weight", "Attn/K/weight"]),
    ("V weight", ["Attn/V/weight"]),
    ("Attn-O weight", ["Attn/O/weight"]),
    ("FFN1 weight", ["FFN/Intermediate/weight"]),
    ("FFN2 weight", ["FFN/OutputDense/weight"]),
    ("LayerNorm weight", ["LayerNorm/weight"]),
    ("Q/K bias", ["Attn/Q/bias", "Attn/K/bias"]),
    ("V bias", ["Attn/V/bias"]),
    ("Attn-O bias", ["Attn/O/bias"]),
    ("FFN1 bias", ["FFN/Intermediate/bias"]),
    ("FFN2 bias", ["FFN/OutputDense/bias"]),
    ("LayerNorm bias", ["LayerNorm/bias"]),
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def parse_bins(fieldnames: list[str]) -> list[tuple[float, float, str, str]]:
    bins = []
    for field in fieldnames:
        if not field.startswith("pct_("):
            continue
        inner = field[len("pct_(") : -1]
        lo_str, hi_str = inner.split(",")
        hi_str = hi_str.rstrip("]")
        lo = float(lo_str)
        hi = float(hi_str)
        bins.append((lo, hi, field, f"({lo_str},{hi_str}]"))
    return bins


def row_count(row: dict[str, str]) -> float:
    if "count" in row:
        return float(row["count"])
    return float(row["n_total"])


def aggregate_mode(
    rows: list[dict[str, str]],
    bins: list[tuple[float, float, str, str]],
) -> dict[str, float | str]:
    if not rows:
        raise ValueError("No rows to aggregate.")
    total = sum(row_count(r) for r in rows)
    if total <= 0:
        raise ValueError("Aggregated rows have zero count.")

    bin_counts = []
    for lo, hi, field, label in bins:
        count = sum(row_count(r) * float(r.get(field, 0.0)) / 100.0 for r in rows)
        bin_counts.append((count, lo, hi, label))
    count, lo, hi, label = max(bin_counts, key=lambda x: x[0])
    center = math.sqrt(lo * hi)
    return {
        "mode_bin": label,
        "mode_mass_pct": 100.0 * count / total,
        "mode_magnitude": center,
        "log10_mode_magnitude": math.log10(center),
        "total": total,
    }


def aggregate_bin_percentages(
    rows: list[dict[str, str]],
    bins: list[tuple[float, float, str, str]],
) -> np.ndarray:
    if not rows:
        raise ValueError("No rows to aggregate.")
    total = sum(row_count(r) for r in rows)
    if total <= 0:
        raise ValueError("Aggregated rows have zero count.")
    values = []
    for _lo, _hi, field, _label in bins:
        count = sum(row_count(r) * float(r.get(field, 0.0)) / 100.0 for r in rows)
        values.append(100.0 * count / total)
    return np.array(values, dtype=np.float64)


def activation_rows_for_group(
    rows: list[dict[str, str]],
    probes: list[str],
    layer: str,
) -> list[dict[str, str]]:
    selected = []
    for r in rows:
        if r["probe"] not in probes:
            continue
        if r["layer"] == layer or r["layer"] == "all":
            selected.append(r)
    return selected


def param_rows_for_group(
    rows: list[dict[str, str]],
    categories: list[str],
    layer: str,
) -> list[dict[str, str]]:
    scope = f"L{layer}"
    return [r for r in rows if r["scope"] == scope and r["category"] in categories]


def build_grouped_modes(
    activation_rows: list[dict[str, str]],
    activation_bins,
    param_rows: list[dict[str, str]],
    param_bins,
) -> list[dict[str, float | str]]:
    out = []
    for source, groups, rows, bins in [
        ("activation", ACTIVATION_GROUPS, activation_rows, activation_bins),
        ("parameter", PARAM_GROUPS, param_rows, param_bins),
    ]:
        for group_name, keys in groups:
            for layer in range(12):
                layer_str = str(layer)
                if source == "activation":
                    selected = activation_rows_for_group(rows, keys, layer_str)
                else:
                    selected = param_rows_for_group(rows, keys, layer_str)
                if not selected:
                    continue
                mode = aggregate_mode(selected, bins)
                out.append(
                    {
                        "source": source,
                        "group": group_name,
                        "layer": layer,
                        **mode,
                    }
                )
    return out


def build_distribution_profiles(
    activation_rows: list[dict[str, str]],
    activation_bins,
    param_rows: list[dict[str, str]],
    param_bins,
) -> tuple[list[str], list[str], np.ndarray]:
    labels: list[str] = []
    bin_labels = [label for *_rest, label in activation_bins]
    matrices = []

    for group_name, keys in ACTIVATION_GROUPS:
        selected = [
            r
            for r in activation_rows
            if r["probe"] in keys and (r["layer"] == "all" or r["layer"].isdigit())
        ]
        labels.append(f"A: {group_name}")
        matrices.append(aggregate_bin_percentages(selected, activation_bins))

    # Weight histograms have fewer high-magnitude bins than activation histograms.
    # Align them onto activation bins so the x-axis uses one common unit.
    weight_by_label = {label: i for i, (*_, label) in enumerate(param_bins)}
    for group_name, categories in PARAM_GROUPS:
        selected = [
            r
            for r in param_rows
            if r["scope"].startswith("L") and r["scope"][1:].isdigit() and r["category"] in categories
        ]
        raw = aggregate_bin_percentages(selected, param_bins)
        aligned = np.zeros(len(activation_bins), dtype=np.float64)
        for i, (*_, label) in enumerate(activation_bins):
            if label in weight_by_label:
                aligned[i] = raw[weight_by_label[label]]
        labels.append(f"P: {group_name}")
        matrices.append(aligned)

    return labels, bin_labels, np.vstack(matrices)


def matrix_from_modes(
    modes: list[dict[str, float | str]],
    groups: list[str],
    value_key: str,
) -> np.ndarray:
    mat = np.full((len(groups), 12), np.nan, dtype=np.float64)
    by_key = {(m["group"], int(m["layer"])): m for m in modes}
    for i, group in enumerate(groups):
        for layer in range(12):
            if (group, layer) in by_key:
                mat[i, layer] = float(by_key[(group, layer)][value_key])
    return mat


def plot_mode_order_heatmap(modes: list[dict[str, float | str]], out_path: Path) -> None:
    act_groups = [g for g, _ in ACTIVATION_GROUPS]
    param_groups = [g for g, _ in PARAM_GROUPS]
    groups = act_groups + param_groups
    mat = matrix_from_modes(modes, groups, "log10_mode_magnitude")

    fig, ax = plt.subplots(figsize=(13.5, 8.2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="turbo", vmin=-3.5, vmax=0.5)
    ax.set_xticks(range(12), labels=[str(i) for i in range(12)])
    ax.set_yticks(range(len(groups)), labels=groups)
    ax.set_xlabel("BERT layer")
    ax.set_ylabel("Grouped activation / parameter")
    ax.set_title("Modal magnitude by layer: activation + model parameters")
    ax.axhline(len(act_groups) - 0.5, color="white", linewidth=2.0)
    ax.text(
        11.65,
        (len(act_groups) - 1) / 2,
        "activations",
        va="center",
        ha="left",
        fontsize=9,
        color="#333333",
    )
    ax.text(
        11.65,
        len(act_groups) + (len(param_groups) - 1) / 2,
        "parameters",
        va="center",
        ha="left",
        fontsize=9,
        color="#333333",
    )
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(mode |value|); one unit = 10x magnitude")
    cbar.set_ticks([-3.5, -2.5, -1.5, -0.5, 0.5])
    cbar.set_ticklabels(["3e-4", "3e-3", "3e-2", "3e-1", "3"])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_distribution_profile_heatmap(
    activation_rows: list[dict[str, str]],
    activation_bins,
    param_rows: list[dict[str, str]],
    param_bins,
    out_path: Path,
) -> None:
    labels, bin_labels, mat = build_distribution_profiles(
        activation_rows,
        activation_bins,
        param_rows,
        param_bins,
    )
    center_logs = [
        (math.log10(lo) + math.log10(hi)) / 2.0 for lo, hi, *_ in activation_bins
    ]
    fig, ax = plt.subplots(figsize=(13.8, 8.6), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=85.0)
    ax.set_xticks(range(len(bin_labels)), labels=[f"{x:.1f}" for x in center_logs], rotation=0)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Magnitude bin center: log10(|value|)")
    ax.set_ylabel("Grouped activation / parameter")
    ax.set_title("Full magnitude distribution profile, averaged over layers")
    ax.axhline(len(ACTIVATION_GROUPS) - 0.5, color="white", linewidth=2.0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= 8.0:
                color = "black" if mat[i, j] > 58 else "white"
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.3, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Values in magnitude bin (%)")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_mode_mass_heatmap(modes: list[dict[str, float | str]], out_path: Path) -> None:
    act_groups = [g for g, _ in ACTIVATION_GROUPS]
    param_groups = [g for g, _ in PARAM_GROUPS]
    groups = act_groups + param_groups
    mat = matrix_from_modes(modes, groups, "mode_mass_pct")

    fig, ax = plt.subplots(figsize=(13.5, 8.2), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=100.0)
    ax.set_xticks(range(12), labels=[str(i) for i in range(12)])
    ax.set_yticks(range(len(groups)), labels=groups)
    ax.set_xlabel("BERT layer")
    ax.set_ylabel("Grouped activation / parameter")
    ax.set_title("How concentrated each distribution is in its modal magnitude bin")
    ax.axhline(len(act_groups) - 0.5, color="white", linewidth=2.0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                color = "black" if mat[i, j] > 65 else "white"
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mass in modal bin (%)")
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def write_summary(path: Path, modes: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "source",
            "group",
            "layer",
            "mode_bin",
            "mode_mass_pct",
            "mode_magnitude",
            "log10_mode_magnitude",
            "total",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in modes:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation_csv", type=Path, default=DEFAULT_ACTIVATION_CSV)
    parser.add_argument("--weight_csv", type=Path, default=DEFAULT_WEIGHT_CSV)
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/activation_weight_mode_heatmaps"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    act_fields, act_rows = read_csv(args.activation_csv)
    weight_fields, weight_rows = read_csv(args.weight_csv)
    act_bins = parse_bins(act_fields)
    weight_bins = parse_bins(weight_fields)
    modes = build_grouped_modes(act_rows, act_bins, weight_rows, weight_bins)

    write_summary(args.output_dir / "activation_weight_mode_summary.csv", modes)
    plot_mode_order_heatmap(modes, args.output_dir / "activation_weight_mode_order_heatmap.png")
    plot_mode_mass_heatmap(modes, args.output_dir / "activation_weight_mode_mass_heatmap.png")
    plot_distribution_profile_heatmap(
        act_rows,
        act_bins,
        weight_rows,
        weight_bins,
        args.output_dir / "activation_weight_distribution_profile_heatmap.png",
    )
    print(f"Saved mode heatmaps to {args.output_dir}")


if __name__ == "__main__":
    main()
