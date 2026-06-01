#!/usr/bin/env python3
"""Score noise-vs-accuracy tradeoffs from MRPC noise sweep outputs.

This script reads the CSV/JSON artifacts produced by
``relative_vs_absolute_noise_mrpc.py`` and computes two useful metrics:

1. APNU: accuracy-preserving noise utility

       APNU_gamma(B_abs) = B_abs * (metric(B_abs) / metric_clean) ** gamma

   where ``B_abs`` is an absolute noise budget, implemented as
   ``P80(|noise|)``.  For relative-noise experiments, ``B_abs`` is the actual
   absolute-noise percentile induced by the relative perturbation.  For
   absolute-noise experiments, ``B_abs`` is the corresponding absolute error
   percentile/threshold.  ``gamma`` controls how harshly accuracy/F1 drops are
   penalized.  Larger is better.

2. MNB@delta: maximum noise budget while preserving performance

       max B_abs  subject to  metric(B_abs) >= metric_clean - delta

   This is analogous to robust accuracy / robustness-radius reporting in
   robustness papers: instead of asking accuracy at one radius, ask the largest
   radius tolerated under a metric-drop budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "relative": "Relative",
    "absolute_bulk_p80_x": "Absolute A=tau*P80(|x|)",
    "absolute_matched_abs_p": "Absolute matched |noise| p80",
    "absolute_global_re": "Absolute same global re",
    "relative_dist_q20": "Relative dist Q20 per node",
    "absolute_global_q20": "Absolute global Q20",
    "absolute_matched_rel_p80": "Absolute matched relative P80",
}


def read_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                out[k] = v if k == "method" else float(v)
            rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def trapezoid_auc(xs: list[float], ys: list[float], x_max: float) -> float:
    if not xs or x_max <= 0:
        return 0.0
    order = np.argsort(xs)
    x = np.array(xs, dtype=float)[order]
    y = np.array(ys, dtype=float)[order]
    if x[0] > 0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0])
    if x[-1] < x_max:
        x = np.append(x, x_max)
        y = np.append(y, y[-1])
    return float(np.trapz(y, x) / x_max)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("experiments/relative_vs_absolute_noise_mrpc_summary.csv"),
    )
    p.add_argument(
        "--specs_json",
        type=Path,
        default=Path("experiments/relative_vs_absolute_noise_mrpc_specs.json"),
    )
    p.add_argument(
        "--activation_json",
        type=Path,
        default=None,
    )
    p.add_argument("--output_dir", type=Path, default=Path("experiments"))
    p.add_argument("--output_prefix", type=str, default="noise_accuracy_tradeoff")
    p.add_argument("--clean_method", type=str, default=None)
    p.add_argument("--gamma", type=float, default=8.0)
    p.add_argument(
        "--legacy_tau_scores",
        action="store_true",
        help="Also emit legacy tau-contract scores for debugging; primary scores use absolute P80(|noise|).",
    )
    p.add_argument("--deltas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.summary_csv)
    specs = {}
    if args.specs_json and args.specs_json.exists():
        specs = {(s["method"], float(s["tau"])): s for s in json.load(args.specs_json.open())}
    act_p80 = 0.0
    if args.activation_json and args.activation_json.exists():
        act = json.load(args.activation_json.open())
        act_p80 = float(act.get("p80", 0.0))

    clean_candidates = [r for r in rows if r["tau"] == 0.0]
    if args.clean_method:
        clean = next(r for r in clean_candidates if r["method"] == args.clean_method)
    elif any(r["method"] == "relative" for r in clean_candidates):
        clean = next(r for r in clean_candidates if r["method"] == "relative")
    else:
        clean = clean_candidates[0]
    clean_f1 = clean["f1_mean"]
    clean_acc = clean["accuracy_mean"]

    point_rows = []
    for r in rows:
        spec = specs.get((r["method"], r["tau"]), {})
        abs_noise_p80 = float(
            spec.get(
                "abs_p_coverage",
                r.get("abs_p80_noise", r.get("abs_p_coverage", 0.0)),
            )
        )
        norm_abs_p80 = abs_noise_p80 / act_p80 if act_p80 else 0.0
        f1_ret = r["f1_mean"] / clean_f1
        acc_ret = r["accuracy_mean"] / clean_acc
        row = {
            "method": r["method"],
            "display": DISPLAY.get(r["method"], r["method"]),
            "tau_generation_setting": r["tau"],
            "absolute_noise_p80": abs_noise_p80,
            "absolute_noise_p80_over_activation_p80": norm_abs_p80,
            "f1_mean": r["f1_mean"],
            "accuracy_mean": r["accuracy_mean"],
            "f1_retention": f1_ret,
            "accuracy_retention": acc_ret,
            f"apnu_abs_f1_g{args.gamma:g}": abs_noise_p80 * (f1_ret ** args.gamma),
            f"apnu_abs_acc_g{args.gamma:g}": abs_noise_p80 * (acc_ret ** args.gamma),
        }
        if args.legacy_tau_scores:
            row[f"legacy_apnu_tau_f1_g{args.gamma:g}"] = r["tau"] * (f1_ret ** args.gamma)
            row[f"legacy_apnu_tau_acc_g{args.gamma:g}"] = r["tau"] * (acc_ret ** args.gamma)
        point_rows.append(row)

    point_path = args.output_dir / f"{args.output_prefix}_point_scores.csv"
    write_csv(point_path, point_rows)

    method_rows = []
    methods = sorted({r["method"] for r in point_rows})
    for method in methods:
        vals = [r for r in point_rows if r["method"] == method]
        vals_nozero = [r for r in vals if r["absolute_noise_p80"] > 0]
        summary = {"method": method, "display": DISPLAY.get(method, method)}
        for metric_col, clean_metric in (("f1_mean", clean_f1), ("accuracy_mean", clean_acc)):
            metric_tag = "f1" if metric_col == "f1_mean" else "acc"
            score_col = f"apnu_abs_{metric_tag}_g{args.gamma:g}"
            best = max(vals_nozero, key=lambda x: x[score_col])
            summary[f"best_abs_{metric_tag}_score"] = best[score_col]
            summary[f"best_abs_{metric_tag}_tau_generation_setting"] = best["tau_generation_setting"]
            summary[f"best_abs_{metric_tag}_budget"] = best["absolute_noise_p80"]
            summary[f"best_abs_{metric_tag}_metric"] = best[metric_col]
            xs = [v["absolute_noise_p80"] for v in vals]
            ys = [(v[metric_col] / clean_metric) ** args.gamma for v in vals]
            summary[f"auc_abs_{metric_tag}_retention_g{args.gamma:g}"] = trapezoid_auc(
                xs, ys, max(xs)
            )

            if args.legacy_tau_scores:
                legacy_col = f"legacy_apnu_tau_{metric_tag}_g{args.gamma:g}"
                legacy_best = max(vals_nozero, key=lambda x: x[legacy_col])
                summary[f"legacy_best_tau_{metric_tag}_score"] = legacy_best[legacy_col]
                summary[f"legacy_best_tau_{metric_tag}_tau"] = legacy_best["tau_generation_setting"]

        for delta in args.deltas:
            for metric_col, clean_metric in (("f1_mean", clean_f1), ("accuracy_mean", clean_acc)):
                ok = [v for v in vals if v[metric_col] >= clean_metric - delta]
                metric_tag = "f1" if metric_col == "f1_mean" else "acc"
                dtag = str(delta).replace(".", "p")
                summary[f"mnb_abs_{metric_tag}_drop_{dtag}"] = (
                    max(v["absolute_noise_p80"] for v in ok) if ok else 0.0
                )
                summary[f"mnb_tau_generation_{metric_tag}_drop_{dtag}"] = (
                    max(v["tau_generation_setting"] for v in ok) if ok else 0.0
                )
        method_rows.append(summary)

    summary_path = args.output_dir / f"{args.output_prefix}_method_summary.csv"
    write_csv(summary_path, method_rows)

    # Compact plot: APNU per point using the absolute P80(|noise|) budget and F1.
    score_col = f"apnu_abs_f1_g{args.gamma:g}"
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    for method in methods:
        vals = [r for r in point_rows if r["method"] == method and r["absolute_noise_p80"] > 0]
        vals.sort(key=lambda r: r["absolute_noise_p80"])
        ax.plot(
            [r["absolute_noise_p80"] for r in vals],
            [r[score_col] for r in vals],
            marker="o",
            linewidth=2,
            label=DISPLAY.get(method, method),
        )
    ax.set_title(f"APNU score: absolute-noise reward with F1 penalty (gamma={args.gamma:g})")
    ax.set_xlabel("Absolute noise budget: P80(|noise|)")
    ax.set_ylabel("APNU = P80(|noise|) * (F1 / clean F1)^gamma")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plot_path = args.output_dir / f"{args.output_prefix}_apnu.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    # Method-level view: this makes the headline comparison easier to read.
    bar_path = args.output_dir / f"{args.output_prefix}_summary_bars.png"
    preferred_order = [
        "absolute_global_q20",
        "absolute_matched_rel_p80",
        "relative_dist_q20",
        "absolute_bulk_p80_x",
        "absolute_matched_abs_p",
        "absolute_global_re",
        "relative",
    ]
    ordered_rows = sorted(
        method_rows,
        key=lambda r: preferred_order.index(r["method"]) if r["method"] in preferred_order else 999,
    )
    labels = [r["display"] for r in ordered_rows]
    colors = ["#d62728", "#9467bd", "#1f77b4", "#ff7f0e", "#8c564b", "#2ca02c", "#7f7f7f"]
    colors = colors[: len(labels)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    axes[0].bar(labels, [r["best_abs_f1_score"] for r in ordered_rows], color=colors)
    axes[0].set_title(f"Best APNU-F1 (gamma={args.gamma:g})")
    axes[0].set_ylabel("Best APNU")
    axes[1].bar(labels, [r["mnb_abs_f1_drop_0p005"] for r in ordered_rows], color=colors)
    axes[1].set_title("Max absolute noise with F1 drop <= 0.005")
    axes[1].set_ylabel("P80(|noise|)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(bar_path, dpi=220)
    plt.close(fig)

    print(f"Clean method={clean['method']}, F1={clean_f1:.6f}, accuracy={clean_acc:.6f}")
    if act_p80:
        print(f"Activation P80={act_p80:.6f}")
    print("Primary budget is absolute_noise_p80 = P80(|noise|), not tau percent.")
    print(f"Saved point scores to {point_path}")
    print(f"Saved method summary to {summary_path}")
    print(f"Saved plot to {plot_path}")
    print(f"Saved method summary plot to {bar_path}")


if __name__ == "__main__":
    main()
