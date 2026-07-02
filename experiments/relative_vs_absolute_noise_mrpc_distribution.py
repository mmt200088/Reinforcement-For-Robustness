#!/usr/bin/env python3
"""MRPC/BERT-base noise experiment using distribution-derived relative budgets.

Corrected interpretation of ``re=(coverage, tau)``:

For each intermediate node/layer, read the magnitude distribution from
``mrpc_magnitude_stats.csv``.  Let

    q = Q_(1 - coverage)(|x|)

For ``re=(0.8, 0.1)``, this is Q20(|x|).  The relative-error noise budget for
that node is

    B_node = tau * q

Then Gaussian noise is injected with sigma chosen so that

    P(|N(0, sigma_node)| < B_node) = coverage.

For coverage=0.8, sigma_node = B_node / Phi^-1(0.9).

The absolute baseline uses one pooled/global q from all hooked intermediate
nodes and applies a single global absolute Gaussian noise scale everywhere.
Final tradeoff plots use the actual absolute noise amount, approximated by
the pooled P80(|noise|), not the relative tau percentage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from csv_field_utils import write_csv_rows_with_inferred_fields


MODEL_NAME = "textattack/bert-base-uncased-MRPC"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "mrpc"
DEFAULT_LEVELS = tuple(round(i * 0.005, 3) for i in range(0, 101)) + tuple(
    round(i * 0.05, 2) for i in range(11, 21)
)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_MAG_STATS = (
    "/var/tmp/root-home/Reinforcement-For-Robustness/"
    "Model_analysis/all_analysis_approx/mrpc/mrpc_magnitude_stats.csv"
)


@dataclass(frozen=True)
class HookPoint:
    name: str
    module: nn.Module
    probe: str
    layer: str

    @property
    def key(self) -> str:
        return f"{self.probe}:{self.layer}"


@dataclass(frozen=True)
class NoiseSpec:
    method: str
    tau: float
    coverage: float
    sigma_abs: float
    abs_p_coverage: float
    budget_by_key: dict[str, float]
    sigma_by_key: dict[str, float]


def normal_two_sided_z(coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("--coverage must be in (0, 1)")
    p = torch.tensor((1.0 + coverage) / 2.0, dtype=torch.float64)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(p).item())


def quantile_percent_for_coverage(coverage: float) -> int:
    return int(round((1.0 - coverage) * 100.0))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloader(tokenizer, batch_size: int, max_length: int, max_samples: int):
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    def tok(batch):
        return tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    ds = ds.map(tok, batched=True)
    keep = ["input_ids", "attention_mask", "token_type_ids", "label"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format(type="torch", columns=keep)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def iter_hook_points(model) -> Iterable[HookPoint]:
    yield HookPoint("bert.embeddings", model.bert.embeddings, "after_embed", "all")
    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        li = str(layer_idx)
        prefix = f"bert.encoder.layer.{layer_idx}"
        yield HookPoint(f"{prefix}.attention.self.query", layer.attention.self.query, "query_proj", li)
        yield HookPoint(f"{prefix}.attention.self.key", layer.attention.self.key, "key_proj", li)
        yield HookPoint(f"{prefix}.attention.self.value", layer.attention.self.value, "value_proj", li)
        yield HookPoint(f"{prefix}.attention.output.dense", layer.attention.output.dense, "attn_output", li)
        yield HookPoint(f"{prefix}.attention.output.LayerNorm", layer.attention.output.LayerNorm, "post_attn_ln", li)
        yield HookPoint(f"{prefix}.intermediate.dense", layer.intermediate.dense, "gelu_input", li)
        yield HookPoint(f"{prefix}.output.dense", layer.output.dense, "ffn2_output", li)
        yield HookPoint(f"{prefix}.output.LayerNorm", layer.output.LayerNorm, "post_ffn_ln", li)


def _parse_mag_bins(fieldnames: list[str]) -> list[tuple[float, float, str]]:
    out = []
    for field in fieldnames:
        if not field.startswith("pct_("):
            continue
        inner = field[len("pct_(") : -1]
        lo, hi = inner.split(",")
        hi = hi.rstrip("]")
        out.append((float(lo), float(hi), field))
    return out


def _quantile_from_hist(row: dict[str, str], bins, quantile: float) -> float:
    target = quantile * 100.0
    zero = float(row["pct_zero"])
    if target <= zero:
        return 0.0
    cum = zero
    for lo, hi, field in bins:
        pct = float(row[field])
        if pct <= 0:
            continue
        if cum + pct >= target:
            frac = (target - cum) / pct
            return math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        cum += pct
    return bins[-1][1]


def _pooled_quantile_from_rows(rows: list[dict[str, str]], bins, quantile: float) -> float:
    total = sum(float(r["count"]) for r in rows)
    if total <= 0:
        raise ValueError("No histogram counts available for pooled quantile.")
    target = quantile * total
    zero_count = sum(float(r["count"]) * float(r["pct_zero"]) / 100.0 for r in rows)
    if target <= zero_count:
        return 0.0
    cum = zero_count
    for lo, hi, field in bins:
        bin_count = sum(float(r["count"]) * float(r[field]) / 100.0 for r in rows)
        if bin_count <= 0:
            continue
        if cum + bin_count >= target:
            frac = (target - cum) / bin_count
            return math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
        cum += bin_count
    return bins[-1][1]


def load_distribution_budgets(
    magnitude_stats_csv: Path,
    hook_keys: list[tuple[str, str]],
    coverage: float,
) -> tuple[dict[str, dict], float, list[dict]]:
    q = 1.0 - coverage
    q_percent = quantile_percent_for_coverage(coverage)
    with magnitude_stats_csv.open() as f:
        reader = csv.DictReader(f)
        bins = _parse_mag_bins(reader.fieldnames or [])
        rows = list(reader)

    by_probe_layer = {(r["probe"], r["layer"]): r for r in rows}
    selected_rows = []
    info: dict[str, dict] = {}
    for probe, layer in hook_keys:
        row = by_probe_layer.get((probe, layer))
        if row is None:
            raise KeyError(f"Missing magnitude stats row for probe={probe} layer={layer}")
        selected_rows.append(row)
        q_val = _quantile_from_hist(row, bins, q)
        key = f"{probe}:{layer}"
        info[key] = {
            "probe": probe,
            "layer": layer,
            "count": float(row["count"]),
            f"q{q_percent}_abs": q_val,
        }
    pooled_q = _pooled_quantile_from_rows(selected_rows, bins, q)
    return info, pooled_q, selected_rows


def approx_pooled_noise_pcoverage(
    sigmas: np.ndarray,
    counts: np.ndarray,
    coverage: float,
    seed: int,
    sample_size: int = 2_000_000,
) -> float:
    if np.max(sigmas) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    probs = counts / counts.sum()
    idx = rng.choice(len(sigmas), size=sample_size, p=probs)
    samples = np.abs(rng.standard_normal(sample_size)) * sigmas[idx]
    return float(np.quantile(samples, coverage))


def make_noise_specs(
    dist_info: dict[str, dict],
    pooled_q: float,
    levels: list[float],
    coverage: float,
    seed: int,
) -> list[NoiseSpec]:
    z = normal_two_sided_z(coverage)
    q_percent = quantile_percent_for_coverage(coverage)
    keys = list(dist_info)
    counts = np.array([dist_info[k]["count"] for k in keys], dtype=np.float64)
    qvals = np.array([dist_info[k][f"q{q_percent}_abs"] for k in keys])
    specs: list[NoiseSpec] = []

    for tau in levels:
        # Layer/probe-specific relative-error budget from each node's Q20.
        budgets = tau * qvals
        sigmas = budgets / z if z > 0 else budgets
        budget_by_key = {k: float(b) for k, b in zip(keys, budgets)}
        sigma_by_key = {k: float(s) for k, s in zip(keys, sigmas)}
        rel_abs_p = approx_pooled_noise_pcoverage(sigmas, counts, coverage, seed)
        specs.append(
            NoiseSpec(
                method="relative_dist_q20",
                tau=tau,
                coverage=coverage,
                sigma_abs=0.0,
                abs_p_coverage=rel_abs_p,
                budget_by_key=budget_by_key,
                sigma_by_key=sigma_by_key,
            )
        )

        # One absolute error for every intermediate node from the pooled Q20.
        abs_budget = tau * pooled_q
        abs_sigma = abs_budget / z if z > 0 else abs_budget
        specs.append(
            NoiseSpec(
                method="absolute_global_q20",
                tau=tau,
                coverage=coverage,
                sigma_abs=abs_sigma,
                abs_p_coverage=abs_budget,
                budget_by_key={},
                sigma_by_key={},
            )
        )

        # Diagnostic: same pooled absolute-noise percentile as relative_dist_q20.
        matched_sigma = rel_abs_p / z if z > 0 else rel_abs_p
        specs.append(
            NoiseSpec(
                method="absolute_matched_rel_p80",
                tau=tau,
                coverage=coverage,
                sigma_abs=matched_sigma,
                abs_p_coverage=rel_abs_p,
                budget_by_key={},
                sigma_by_key={},
            )
        )
    return specs


class NoisyActivation:
    def __init__(self, spec: NoiseSpec, key: str) -> None:
        self.spec = spec
        self.key = key

    def __call__(self, _module: nn.Module, _inputs, output):
        if not torch.is_tensor(output) or self.spec.tau == 0.0:
            return output
        if self.spec.method == "relative_dist_q20":
            sigma = self.spec.sigma_by_key[self.key]
        else:
            sigma = self.spec.sigma_abs
        if sigma == 0.0:
            return output
        return output + torch.randn_like(output) * sigma


def install_noise_hooks(model, spec: NoiseSpec):
    handles = []
    for hp in iter_hook_points(model):
        handles.append(hp.module.register_forward_hook(NoisyActivation(spec, hp.key)))
    return handles


def remove_hooks(handles) -> None:
    for h in handles:
        h.remove()


def evaluate(model, dataloader, device: torch.device, desc: str) -> dict[str, float]:
    preds: list[int] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            y = batch.pop("label").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()
            preds.extend(np.argmax(logits, axis=-1).tolist())
            labels.extend(y.tolist())
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "n": float(len(labels)),
    }


def summarize_by_method(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["tau"]), []).append(row)
    out = []
    for (method, tau), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        f1 = np.array([v["f1"] for v in vals], dtype=np.float64)
        acc = np.array([v["accuracy"] for v in vals], dtype=np.float64)
        abs_p = np.array([v["abs_p_coverage"] for v in vals], dtype=np.float64)
        out.append(
            {
                "method": method,
                "tau": tau,
                "abs_p80_noise": float(abs_p.mean()),
                "f1_mean": float(f1.mean()),
                "f1_std": float(f1.std(ddof=0)),
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=0)),
                "runs": len(vals),
            }
        )
    return out


def plot_results(summary: list[dict], baseline: dict, out_path: Path) -> None:
    display = {
        "relative_dist_q20": ("Relative dist Q20 per node", "#1f77b4", "o"),
        "absolute_global_q20": ("Absolute global Q20", "#d62728", "s"),
        "absolute_matched_rel_p80": ("Absolute matched relative P80", "#9467bd", "D"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    all_f1 = [baseline["f1"]]
    all_acc = [baseline["accuracy"]]
    for method, (label, color, marker) in display.items():
        vals = [r for r in summary if r["method"] == method]
        vals.sort(key=lambda r: r["abs_p80_noise"])
        x = np.array([r["abs_p80_noise"] for r in vals])
        y_f1 = np.array([r["f1_mean"] for r in vals])
        e_f1 = np.array([r["f1_std"] for r in vals])
        y_acc = np.array([r["accuracy_mean"] for r in vals])
        e_acc = np.array([r["accuracy_std"] for r in vals])
        all_f1.extend(y_f1.tolist())
        all_acc.extend(y_acc.tolist())
        axes[0].errorbar(
            x,
            y_f1,
            yerr=e_f1,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=3.4,
            elinewidth=0.8,
            capsize=1.3,
        )
        axes[1].errorbar(
            x,
            y_acc,
            yerr=e_acc,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=3.4,
            elinewidth=0.8,
            capsize=1.3,
        )
    axes[0].axhline(baseline["f1"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[1].axhline(baseline["accuracy"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[0].set_title("MRPC F1 vs actual absolute noise")
    axes[1].set_title("MRPC accuracy vs actual absolute noise")
    for ax in axes:
        ax.set_xlabel("Actual absolute noise budget: pooled P80(|noise|)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylim(max(0.0, min(all_f1) - 0.02), min(1.0, max(all_f1) + 0.01))
    axes[1].set_ylim(max(0.0, min(all_acc) - 0.02), min(1.0, max(all_acc) + 0.01))
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle("Distribution-derived relative error vs one global absolute error", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_preservation_zoom(summary: list[dict], baseline: dict, out_path: Path) -> None:
    display = {
        "relative_dist_q20": ("Relative dist Q20 per node", "#1f77b4", "o"),
        "absolute_global_q20": ("Absolute global Q20", "#d62728", "s"),
        "absolute_matched_rel_p80": ("Absolute matched relative P80", "#9467bd", "D"),
    }
    focus = [
        r
        for r in summary
        if r["f1_mean"] >= baseline["f1"] - 0.05
        and r["accuracy_mean"] >= baseline["accuracy"] - 0.05
    ]
    x_max = max(r["abs_p80_noise"] for r in focus) if focus else 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for method, (label, color, marker) in display.items():
        vals = [
            r
            for r in summary
            if r["method"] == method and r["abs_p80_noise"] <= x_max * 1.05
        ]
        vals.sort(key=lambda r: r["abs_p80_noise"])
        x = np.array([r["abs_p80_noise"] for r in vals])
        y_f1 = np.array([r["f1_mean"] for r in vals])
        e_f1 = np.array([r["f1_std"] for r in vals])
        y_acc = np.array([r["accuracy_mean"] for r in vals])
        e_acc = np.array([r["accuracy_std"] for r in vals])
        axes[0].errorbar(
            x,
            y_f1,
            yerr=e_f1,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=3.4,
            elinewidth=0.8,
            capsize=1.3,
        )
        axes[1].errorbar(
            x,
            y_acc,
            yerr=e_acc,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=3.4,
            elinewidth=0.8,
            capsize=1.3,
        )
    axes[0].axhline(baseline["f1"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[1].axhline(baseline["accuracy"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[0].axhline(baseline["f1"] - 0.005, color="#999999", linestyle=":", linewidth=1.1, label="clean - 0.005")
    axes[1].axhline(baseline["accuracy"] - 0.005, color="#999999", linestyle=":", linewidth=1.1, label="clean - 0.005")
    axes[0].set_title("F1 in the accuracy-preserving region")
    axes[1].set_title("Accuracy in the accuracy-preserving region")
    axes[0].set_ylim(max(0.0, baseline["f1"] - 0.065), min(1.0, baseline["f1"] + 0.01))
    axes[1].set_ylim(max(0.0, baseline["accuracy"] - 0.065), min(1.0, baseline["accuracy"] + 0.01))
    for ax in axes:
        ax.set_xlim(left=-0.002, right=x_max * 1.05)
        ax.set_xlabel("Actual absolute noise budget: pooled P80(|noise|)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("Accuracy")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_drop_results(summary: list[dict], baseline: dict, out_path: Path) -> None:
    display = {
        "relative_dist_q20": ("Relative dist Q20 per node", "#1f77b4", "o"),
        "absolute_global_q20": ("Absolute global Q20", "#d62728", "s"),
        "absolute_matched_rel_p80": ("Absolute matched relative P80", "#9467bd", "D"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for method, (label, color, marker) in display.items():
        vals = [r for r in summary if r["method"] == method]
        vals.sort(key=lambda r: r["abs_p80_noise"])
        x = np.array([r["abs_p80_noise"] for r in vals])
        f1_drop = np.array([baseline["f1"] - r["f1_mean"] for r in vals])
        acc_drop = np.array([baseline["accuracy"] - r["accuracy_mean"] for r in vals])
        axes[0].plot(x, f1_drop, label=label, color=color, marker=marker, linewidth=1.7, markersize=3.4)
        axes[1].plot(x, acc_drop, label=label, color=color, marker=marker, linewidth=1.7, markersize=3.4)
    axes[0].axhline(0.005, color="#555555", linestyle="--", linewidth=1.2, label="drop = 0.005")
    axes[1].axhline(0.005, color="#555555", linestyle="--", linewidth=1.2, label="drop = 0.005")
    axes[0].set_title("F1 drop from clean")
    axes[1].set_title("Accuracy drop from clean")
    for ax in axes:
        ax.set_xlabel("Actual absolute noise budget: pooled P80(|noise|)")
        ax.set_ylabel("Clean metric - noisy metric")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tau_metric_results(summary: list[dict], baseline: dict, out_path: Path) -> None:
    display = {
        "relative_dist_q20": ("Relative dist Q20 per node", "#1f77b4", "o"),
        "absolute_global_q20": ("Absolute global Q20", "#d62728", "s"),
        "absolute_matched_rel_p80": ("Absolute matched relative P80", "#9467bd", "D"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    all_f1 = [baseline["f1"]]
    all_acc = [baseline["accuracy"]]
    for method, (label, color, marker) in display.items():
        vals = [r for r in summary if r["method"] == method]
        vals.sort(key=lambda r: r["tau"])
        x = np.array([100 * r["tau"] for r in vals])
        y_f1 = np.array([r["f1_mean"] for r in vals])
        y_acc = np.array([r["accuracy_mean"] for r in vals])
        all_f1.extend(y_f1.tolist())
        all_acc.extend(y_acc.tolist())
        axes[0].plot(x, y_f1, label=label, color=color, marker=marker, linewidth=1.7, markersize=3.4)
        axes[1].plot(x, y_acc, label=label, color=color, marker=marker, linewidth=1.7, markersize=3.4)
    axes[0].axhline(baseline["f1"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[1].axhline(baseline["accuracy"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[0].set_title("F1 vs generation tau")
    axes[1].set_title("Accuracy vs generation tau")
    for ax in axes:
        ax.set_xlabel("Generation setting tau (%) in re=(0.8, tau)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylim(max(0.0, min(all_f1) - 0.02), min(1.0, max(all_f1) + 0.01))
    axes[1].set_ylim(max(0.0, min(all_acc) - 0.02), min(1.0, max(all_acc) + 0.01))
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("Accuracy")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tau_advantage(summary: list[dict], out_path: Path) -> None:
    by_method_tau = {(r["method"], r["tau"]): r for r in summary}
    taus = sorted(
        tau
        for method, tau in by_method_tau
        if method == "relative_dist_q20" and tau > 0.0
    )
    xs = []
    budget_ratio = []
    f1_vs_matched = []
    f1_vs_global = []
    for tau in taus:
        rel = by_method_tau.get(("relative_dist_q20", tau))
        glob = by_method_tau.get(("absolute_global_q20", tau))
        matched = by_method_tau.get(("absolute_matched_rel_p80", tau))
        if not rel or not glob or not matched or glob["abs_p80_noise"] == 0:
            continue
        xs.append(100 * tau)
        budget_ratio.append(rel["abs_p80_noise"] / glob["abs_p80_noise"])
        f1_vs_matched.append(rel["f1_mean"] - matched["f1_mean"])
        f1_vs_global.append(rel["f1_mean"] - glob["f1_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].plot(xs, budget_ratio, marker="o", color="#1f77b4", linewidth=1.7, markersize=3.4)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1.1)
    axes[0].set_title("Noise budget gain at the same tau")
    axes[0].set_ylabel("Relative P80(|noise|) / global-absolute P80(|noise|)")
    axes[1].plot(xs, f1_vs_matched, marker="o", color="#1f77b4", linewidth=1.7, markersize=3.4, label="Relative - matched absolute")
    axes[1].plot(xs, f1_vs_global, marker="s", color="#d62728", linewidth=1.7, markersize=3.4, label="Relative - global absolute")
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=1.1)
    axes[1].set_title("F1 difference at the same tau")
    axes[1].set_ylabel("F1 difference")
    for ax in axes:
        ax.set_xlabel("Generation setting tau (%) in re=(0.8, tau)")
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_q20_heatmap(dist_info: dict[str, dict], out_path: Path) -> None:
    probes = [
        "query_proj",
        "key_proj",
        "value_proj",
        "attn_output",
        "post_attn_ln",
        "gelu_input",
        "ffn2_output",
        "post_ffn_ln",
    ]
    layers = [str(i) for i in range(12)]
    mat = np.zeros((len(probes), len(layers)), dtype=np.float64)
    for i, probe in enumerate(probes):
        for j, layer in enumerate(layers):
            node = dist_info[f"{probe}:{layer}"]
            q_key = next(k for k in node if k.startswith("q") and k.endswith("_abs"))
            mat[i, j] = max(float(node[q_key]), 1e-12)
    embed = dist_info.get("after_embed:all", {})
    embed_q = None
    if embed:
        q_key = next(k for k in embed if k.startswith("q") and k.endswith("_abs"))
        embed_q = float(embed[q_key])

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    im = ax.imshow(np.log10(mat), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(layers)), labels=layers)
    ax.set_yticks(range(len(probes)), labels=probes)
    ax.set_xlabel("BERT layer")
    ax.set_ylabel("Hooked intermediate node")
    title = "Per-layer Q20(|x|) used for re=(0.8, tau)"
    if embed_q is not None:
        title += f"; embedding Q20={embed_q:.3g}"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 Q20(|x|)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_tau_results(summary: list[dict], baseline: dict, out_path: Path) -> None:
    display = {
        "relative_dist_q20": ("Relative dist Q20 per node", "#1f77b4", "o"),
        "absolute_global_q20": ("Absolute global Q20", "#d62728", "s"),
        "absolute_matched_rel_p80": ("Absolute matched relative P80", "#9467bd", "D"),
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for method, (label, color, marker) in display.items():
        vals = [r for r in summary if r["method"] == method]
        vals.sort(key=lambda r: r["tau"])
        ax.plot(
            [100 * r["tau"] for r in vals],
            [r["abs_p80_noise"] for r in vals],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=3.4,
        )
    ax.set_title("Actual absolute noise induced by each tau setting")
    ax.set_xlabel("Generation setting tau (%) in re=(0.8, tau)")
    ax.set_ylabel("Pooled P80(|noise|)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=Path("relative_vs_absolute_noise_mrpc_distribution_out"))
    p.add_argument("--magnitude_stats_csv", type=Path, default=Path(DEFAULT_MAG_STATS))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--calibration_seed", type=int, default=20260519)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataloader = prepare_dataloader(tokenizer, args.batch_size, args.max_length, args.max_samples)
    set_seed(args.calibration_seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    hook_keys = [(hp.probe, hp.layer) for hp in iter_hook_points(model)]
    dist_info, pooled_q, _rows = load_distribution_budgets(
        args.magnitude_stats_csv, hook_keys, args.coverage
    )
    q_percent = quantile_percent_for_coverage(args.coverage)
    with (args.output_dir / "distribution_q20_budgets.json").open("w") as f:
        json.dump(
            {
                "coverage": args.coverage,
                "quantile_percent": q_percent,
                f"pooled_q{q_percent}": pooled_q,
                "nodes": dist_info,
            },
            f,
            indent=2,
        )
    print(f"Loaded distribution stats: pooled Q{q_percent}={pooled_q:.6g}")

    baseline = evaluate(model, dataloader, device, desc="clean baseline")
    with (args.output_dir / "baseline.json").open("w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Clean baseline: f1={baseline['f1']:.4f}, acc={baseline['accuracy']:.4f}, n={int(baseline['n'])}")

    specs = make_noise_specs(dist_info, pooled_q, list(args.levels), args.coverage, args.calibration_seed)
    with (args.output_dir / "noise_specs.json").open("w") as f:
        json.dump(
            [
                {
                    "method": s.method,
                    "tau": s.tau,
                    "coverage": s.coverage,
                    "sigma_abs": s.sigma_abs,
                    "abs_p_coverage": s.abs_p_coverage,
                    "budget_by_key": s.budget_by_key,
                    "sigma_by_key": s.sigma_by_key,
                }
                for s in specs
            ],
            f,
            indent=2,
        )

    rows = []
    for spec in specs:
        if spec.tau == 0.0 and spec.method != "relative_dist_q20":
            continue
        for seed in args.seeds:
            set_seed(seed)
            handles = install_noise_hooks(model, spec)
            metrics = evaluate(model, dataloader, device, desc=f"{spec.method} tau={spec.tau:.3f} seed={seed}")
            remove_hooks(handles)
            row = {
                "method": spec.method,
                "tau": spec.tau,
                "coverage": spec.coverage,
                "seed": seed,
                "sigma_abs": spec.sigma_abs,
                "abs_p_coverage": spec.abs_p_coverage,
                **metrics,
            }
            rows.append(row)
            print(
                f"{spec.method:26s} tau={spec.tau:.3f} seed={seed} "
                f"P80|noise|={spec.abs_p_coverage:.6g} "
                f"f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f}"
            )

    summary = summarize_by_method(rows)
    write_csv_rows_with_inferred_fields(args.output_dir / "per_seed_results.csv", rows)
    write_csv_rows_with_inferred_fields(args.output_dir / "summary_results.csv", summary)
    plot_results(summary, baseline, args.output_dir / "relative_vs_absolute_noise_mrpc_distribution.png")
    plot_preservation_zoom(
        summary,
        baseline,
        args.output_dir / "relative_vs_absolute_noise_mrpc_distribution_preserve_zoom.png",
    )
    plot_drop_results(summary, baseline, args.output_dir / "relative_vs_absolute_noise_mrpc_distribution_drop.png")
    plot_tau_metric_results(summary, baseline, args.output_dir / "relative_vs_absolute_noise_mrpc_distribution_tau_metrics.png")
    plot_tau_advantage(summary, args.output_dir / "relative_vs_absolute_noise_mrpc_distribution_tau_advantage.png")
    plot_q20_heatmap(dist_info, args.output_dir / "relative_vs_absolute_noise_mrpc_distribution_q20_heatmap.png")
    plot_tau_results(summary, baseline, args.output_dir / "noise_budget_by_tau_distribution.png")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
