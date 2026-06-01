#!/usr/bin/env python
"""Compare relative-vs-absolute Gaussian perturbations on MRPC/BERT-base.

The experiment is designed around a relative-error contract:

    re = (coverage, tau) means P(|noise| / |x| < tau) >= coverage.

For Gaussian relative noise, noise = N(0, sigma_rel * |x|), so
sigma_rel = tau / Phi^{-1}((1 + coverage) / 2).

The absolute-noise baselines use noise = N(0, sigma_abs), shared by all
intermediate values.  The main "absolute_bulk_p80_x" baseline sets the
absolute contract to A = tau * P80(|activation|), i.e. it uses one global
absolute scale derived from the same 80% activation mass as re=(0.8, tau).
This is the most direct absolute analogue to the relative contract and shows
the failure mode: small activations are over-perturbed while large activations
are under-perturbed.

A secondary "absolute_matched_abs_p" curve is intentionally conservative: it
matches the relative-noise run's realised 80th-percentile absolute
perturbation.  This is useful as a sanity check, but it is not the right curve
to prove relative error is the better metric because it can make the absolute
baseline much smaller than the bulk activation scale.

An extra "absolute_global_re" curve is also reported: it chooses sigma_abs so
that the clean activation population satisfies the same average relative-error
contract.  This usually becomes very small because tiny activations dominate a
single global absolute threshold, illustrating why absolute error is a poor
global metric for these intermediates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

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


MODEL_NAME = "textattack/bert-base-uncased-MRPC"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "mrpc"
DEFAULT_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
DEFAULT_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class NoiseSpec:
    method: str
    tau: float
    coverage: float
    sigma_rel: float
    sigma_abs: float
    abs_p_coverage: float


class ActivationMagnitudeSampler:
    """Collect a bounded sample of clean intermediate magnitudes."""

    def __init__(self, per_tensor: int, max_values: int, seed: int) -> None:
        self.per_tensor = per_tensor
        self.max_values = max_values
        self.rng = np.random.default_rng(seed)
        self.parts: list[np.ndarray] = []
        self.total_seen = 0

    def add(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().float().abs().reshape(-1).cpu().numpy()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            return
        if flat.size > self.per_tensor:
            idx = self.rng.choice(flat.size, size=self.per_tensor, replace=False)
            flat = flat[idx]
        self.parts.append(flat.astype(np.float32, copy=False))
        self.total_seen += int(flat.size)

        # Keep memory bounded with a simple downsample when the list grows large.
        if self.total_seen > self.max_values * 2:
            merged = np.concatenate(self.parts)
            if merged.size > self.max_values:
                idx = self.rng.choice(merged.size, size=self.max_values, replace=False)
                merged = merged[idx]
            self.parts = [merged.astype(np.float32, copy=False)]
            self.total_seen = int(merged.size)

    def values(self) -> np.ndarray:
        if not self.parts:
            return np.empty((0,), dtype=np.float32)
        merged = np.concatenate(self.parts)
        if merged.size > self.max_values:
            idx = self.rng.choice(merged.size, size=self.max_values, replace=False)
            merged = merged[idx]
        return merged.astype(np.float32, copy=False)


class NoisyActivation:
    """Forward hook object that perturbs tensor outputs."""

    def __init__(self, spec: NoiseSpec, eps: float = 0.0) -> None:
        self.spec = spec
        self.eps = eps

    def __call__(self, _module: nn.Module, _inputs, output):
        if not torch.is_tensor(output) or self.spec.tau == 0.0:
            return output
        if self.spec.method == "relative":
            scale = output.detach().abs().clamp_min(self.eps) * self.spec.sigma_rel
        else:
            scale = torch.full_like(output, self.spec.sigma_abs)
        return output + torch.randn_like(output) * scale


def normal_two_sided_z(coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("--coverage must be in (0, 1)")
    p = torch.tensor((1.0 + coverage) / 2.0, dtype=torch.float64)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(p).item())


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


def iter_noise_modules(model) -> Iterable[tuple[str, nn.Module]]:
    yield "bert.embeddings", model.bert.embeddings
    for layer_idx, layer in enumerate(model.bert.encoder.layer):
        prefix = f"bert.encoder.layer.{layer_idx}"
        yield f"{prefix}.attention.self.query", layer.attention.self.query
        yield f"{prefix}.attention.self.key", layer.attention.self.key
        yield f"{prefix}.attention.self.value", layer.attention.self.value
        yield f"{prefix}.attention.output.dense", layer.attention.output.dense
        yield f"{prefix}.attention.output.LayerNorm", layer.attention.output.LayerNorm
        yield f"{prefix}.intermediate.dense", layer.intermediate.dense
        yield f"{prefix}.output.dense", layer.output.dense
        yield f"{prefix}.output.LayerNorm", layer.output.LayerNorm


def install_sampling_hooks(model, sampler: ActivationMagnitudeSampler):
    handles = []
    for _name, module in iter_noise_modules(model):
        handles.append(
            module.register_forward_hook(
                lambda _m, _i, out, s=sampler: s.add(out)
                if torch.is_tensor(out)
                else None
            )
        )
    return handles


def install_noise_hooks(model, spec: NoiseSpec):
    handles = []
    hook = NoisyActivation(spec)
    for _name, module in iter_noise_modules(model):
        handles.append(module.register_forward_hook(hook))
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


def collect_activation_magnitudes(
    model,
    dataloader,
    device: torch.device,
    per_tensor: int,
    max_values: int,
    seed: int,
) -> np.ndarray:
    sampler = ActivationMagnitudeSampler(per_tensor, max_values, seed)
    handles = install_sampling_hooks(model, sampler)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="calibrate clean magnitudes"):
            batch = {k: v.to(device) for k, v in batch.items() if k != "label"}
            model(**batch)
    remove_hooks(handles)
    values = sampler.values()
    if values.size == 0:
        raise RuntimeError("No activation magnitudes were collected.")
    return values


def solve_abs_sigma_for_global_relative_contract(
    abs_values: np.ndarray,
    coverage: float,
    tau: float,
    max_iter: int = 60,
) -> float:
    """Find sigma so mean_i P(|N(0,sigma)| < tau*|x_i|) == coverage."""
    if tau == 0.0:
        return 0.0
    x = torch.from_numpy(abs_values.astype(np.float64))
    # Exact zeros can never satisfy a positive relative bound under absolute
    # Gaussian noise.  Keeping them would make the contract impossible whenever
    # their fraction is too high; use the smallest positive observed magnitude
    # only to make the diagnostic curve finite.
    positive = x[x > 0]
    if positive.numel() == 0:
        return 0.0
    floor = torch.quantile(positive, 0.001).item()
    x = torch.clamp(x, min=max(floor, 1e-12))

    def achieved(sigma: float) -> float:
        z = tau * x / (math.sqrt(2.0) * sigma)
        return float(torch.erf(z).mean().item())

    lo = 1e-18
    hi = max(float(torch.quantile(x, 0.95).item()) * tau, 1e-12)
    while achieved(hi) > coverage:
        hi *= 2.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        if achieved(mid) >= coverage:
            lo = mid
        else:
            hi = mid
    return lo


def make_noise_specs(
    abs_values: np.ndarray,
    levels: list[float],
    coverage: float,
    seed: int,
) -> list[NoiseSpec]:
    z = normal_two_sided_z(coverage)
    rng = np.random.default_rng(seed)
    specs: list[NoiseSpec] = []
    half_normal = np.abs(rng.standard_normal(size=min(abs_values.size, 2_000_000)))
    sampled_abs = abs_values
    if sampled_abs.size > half_normal.size:
        idx = rng.choice(sampled_abs.size, size=half_normal.size, replace=False)
        sampled_abs = sampled_abs[idx]

    for tau in levels:
        sigma_rel = 0.0 if tau == 0.0 else tau / z
        rel_abs_noise = sampled_abs * sigma_rel * half_normal[: sampled_abs.size]
        abs_p = 0.0 if tau == 0.0 else float(np.quantile(rel_abs_noise, coverage))
        sigma_abs_matched = 0.0 if tau == 0.0 else abs_p / z
        bulk_abs_bound = 0.0 if tau == 0.0 else tau * float(np.quantile(abs_values, coverage))
        sigma_abs_bulk = 0.0 if tau == 0.0 else bulk_abs_bound / z
        sigma_abs_global_re = solve_abs_sigma_for_global_relative_contract(
            abs_values, coverage=coverage, tau=tau
        )
        specs.append(
            NoiseSpec(
                method="relative",
                tau=tau,
                coverage=coverage,
                sigma_rel=sigma_rel,
                sigma_abs=0.0,
                abs_p_coverage=abs_p,
            )
        )
        specs.append(
            NoiseSpec(
                method="absolute_bulk_p80_x",
                tau=tau,
                coverage=coverage,
                sigma_rel=0.0,
                sigma_abs=sigma_abs_bulk,
                abs_p_coverage=bulk_abs_bound,
            )
        )
        specs.append(
            NoiseSpec(
                method="absolute_matched_abs_p",
                tau=tau,
                coverage=coverage,
                sigma_rel=0.0,
                sigma_abs=sigma_abs_matched,
                abs_p_coverage=abs_p,
            )
        )
        specs.append(
            NoiseSpec(
                method="absolute_global_re",
                tau=tau,
                coverage=coverage,
                sigma_rel=0.0,
                sigma_abs=sigma_abs_global_re,
                abs_p_coverage=sigma_abs_global_re * z,
            )
        )
    return specs


def summarize_by_method(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["tau"]), []).append(row)
    summary = []
    for (method, tau), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        f1 = np.array([v["f1"] for v in vals], dtype=np.float64)
        acc = np.array([v["accuracy"] for v in vals], dtype=np.float64)
        sigma_abs = np.array([v["sigma_abs"] for v in vals], dtype=np.float64)
        sigma_rel = np.array([v["sigma_rel"] for v in vals], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "tau": tau,
                "f1_mean": float(f1.mean()),
                "f1_std": float(f1.std(ddof=0)),
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=0)),
                "sigma_abs": float(sigma_abs.mean()),
                "sigma_rel": float(sigma_rel.mean()),
                "runs": len(vals),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary: list[dict], baseline: dict, out_path: Path) -> None:
    display = {
        "relative": ("Relative Gaussian", "#1f77b4", "o"),
        "absolute_bulk_p80_x": ("Absolute, A=tau*P80(|x|)", "#d62728", "s"),
        "absolute_matched_abs_p": ("Absolute, matched |noise| p80", "#9467bd", "D"),
        "absolute_global_re": ("Absolute, same global re", "#2ca02c", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)

    for method, (label, color, marker) in display.items():
        vals = [r for r in summary if r["method"] == method]
        vals.sort(key=lambda r: r["tau"])
        if not vals:
            continue
        x = np.array([100.0 * r["tau"] for r in vals])
        y_f1 = np.array([r["f1_mean"] for r in vals])
        e_f1 = np.array([r["f1_std"] for r in vals])
        y_acc = np.array([r["accuracy_mean"] for r in vals])
        e_acc = np.array([r["accuracy_std"] for r in vals])
        axes[0].errorbar(x, y_f1, yerr=e_f1, label=label, color=color, marker=marker, linewidth=2)
        axes[1].errorbar(x, y_acc, yerr=e_acc, label=label, color=color, marker=marker, linewidth=2)

    axes[0].axhline(baseline["f1"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[1].axhline(baseline["accuracy"], color="#555555", linestyle="--", linewidth=1.2, label="Clean baseline")
    axes[0].axvline(10.0, color="#777777", linestyle=":", linewidth=1.0)
    axes[1].axvline(10.0, color="#777777", linestyle=":", linewidth=1.0)

    axes[0].set_title("MRPC F1 under activation noise")
    axes[1].set_title("MRPC accuracy under activation noise")
    for ax in axes:
        ax.set_xlabel("Relative-error threshold tau (%) in re=(0.8, tau)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("Accuracy")
    fig.suptitle("Relative error allocates noise by activation scale; absolute error uses one global scale", fontsize=13)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_sigma(summary: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for method, color, marker in [
        ("absolute_bulk_p80_x", "#d62728", "s"),
        ("absolute_matched_abs_p", "#9467bd", "D"),
        ("absolute_global_re", "#2ca02c", "^"),
    ]:
        vals = [r for r in summary if r["method"] == method]
        vals.sort(key=lambda r: r["tau"])
        ax.plot(
            [100.0 * r["tau"] for r in vals],
            [r["sigma_abs"] for r in vals],
            marker=marker,
            linewidth=2,
            color=color,
            label=method,
        )
    ax.set_xlabel("Relative-error threshold tau (%)")
    ax.set_ylabel("Calibrated absolute Gaussian sigma")
    ax.set_title("How one absolute sigma is chosen for each relative-noise level")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=Path, default=Path("relative_vs_absolute_noise_mrpc_out"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=0, help="0 means full MRPC validation")
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--calibration_seed", type=int, default=20260519)
    p.add_argument("--sample_per_tensor", type=int, default=2048)
    p.add_argument("--max_calibration_values", type=int, default=2_000_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataloader = prepare_dataloader(tokenizer, args.batch_size, args.max_length, args.max_samples)

    set_seed(args.calibration_seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    baseline = evaluate(model, dataloader, device, desc="clean baseline")
    print(f"Clean baseline: f1={baseline['f1']:.4f}, acc={baseline['accuracy']:.4f}, n={int(baseline['n'])}")

    abs_values = collect_activation_magnitudes(
        model,
        dataloader,
        device,
        per_tensor=args.sample_per_tensor,
        max_values=args.max_calibration_values,
        seed=args.calibration_seed,
    )
    act_summary = {
        "count": int(abs_values.size),
        "min": float(np.min(abs_values)),
        "p01": float(np.quantile(abs_values, 0.01)),
        "p10": float(np.quantile(abs_values, 0.10)),
        "p50": float(np.quantile(abs_values, 0.50)),
        "p80": float(np.quantile(abs_values, 0.80)),
        "p90": float(np.quantile(abs_values, 0.90)),
        "p99": float(np.quantile(abs_values, 0.99)),
        "max": float(np.max(abs_values)),
    }
    print("Activation |x| summary:", json.dumps(act_summary, indent=2))

    specs = make_noise_specs(abs_values, list(args.levels), args.coverage, args.calibration_seed)
    with (args.output_dir / "noise_specs.json").open("w") as f:
        json.dump([spec.__dict__ for spec in specs], f, indent=2)
    with (args.output_dir / "activation_magnitude_summary.json").open("w") as f:
        json.dump(act_summary, f, indent=2)

    rows = []
    for spec in specs:
        if spec.tau == 0.0 and spec.method != "relative":
            # tau=0 is identical for all methods; keep just one clean noisy row.
            continue
        for seed in args.seeds:
            set_seed(seed)
            handles = install_noise_hooks(model, spec)
            metrics = evaluate(
                model,
                dataloader,
                device,
                desc=f"{spec.method} tau={spec.tau:.3f} seed={seed}",
            )
            remove_hooks(handles)
            row = {
                "method": spec.method,
                "tau": spec.tau,
                "coverage": spec.coverage,
                "seed": seed,
                "sigma_rel": spec.sigma_rel,
                "sigma_abs": spec.sigma_abs,
                "abs_p_coverage": spec.abs_p_coverage,
                **metrics,
            }
            rows.append(row)
            print(
                f"{spec.method:24s} tau={spec.tau:.3f} seed={seed} "
                f"f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f} "
                f"sigma_rel={spec.sigma_rel:.6g} sigma_abs={spec.sigma_abs:.6g}"
            )

    summary = summarize_by_method(rows)
    write_csv(args.output_dir / "per_seed_results.csv", rows)
    write_csv(args.output_dir / "summary_results.csv", summary)
    with (args.output_dir / "baseline.json").open("w") as f:
        json.dump(baseline, f, indent=2)

    plot_results(summary, baseline, args.output_dir / "relative_vs_absolute_noise_mrpc.png")
    plot_sigma(summary, args.output_dir / "absolute_sigma_calibration.png")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
