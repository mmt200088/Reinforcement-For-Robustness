"""
诊断脚本：在 BERT-base + MRPC 上估算 CertAcc 与 ratio = γ / (2β_sim)，
判断当前 noise upper-bound 配置是否能给出非空 certified set。

用法:
    python scripts/diagnose_certacc.py \
        --noise-mode simple \
        --c-sigma 6.0 \
        --num-stages 144 \
        --delta-bits 30 \
        --N 32768 --h 64

输出:
    - 全数据集 ratio 的 5/25/50/75/95 分位数
    - CertAcc (按当前 β_sim 计算)
    - ratio 诊断分级 (>=100 / [1,100] / <1)
    - histogram (PNG)

说明:
    `β_sim(x)` 的 closed-form 取自 accuracy_preservation_render.md §3 公式 (7)-(8):

        β_sim(x) = c_σ · L_input(x) · σ_ct_total / Δ_working

    其中:
      σ_ct_total = √(num_stages) · √(N·h/12)         (sub-Gaussian 累加)
      L_input(x)  = ‖∂F/∂embedding‖_2 (random-direction sample, K=4)
      Δ_working   = 2^delta_bits

    这里把"per-stage L_ℓ + 加权和"近似成"input-level total L + RMS 累加"，
    便宜但保守。要更精确可在 --noise-mode=per_layer 下逐层 power-iteration。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    from datasets import load_dataset
except ImportError as e:
    print(f"[fatal] missing dependency: {e}", file=sys.stderr)
    print("install:  pip install transformers datasets", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. β_sim 的几种估算方式
# ---------------------------------------------------------------------------


@dataclass
class NoiseModelParams:
    """β_sim 的 closed-form 参数 (对应 accuracy_preservation_render.md §3-§4)。"""
    c_sigma: float = 6.0           # sub-Gaussian → ℓ_∞ 高置信因子 (δ_0 ≈ 1e-9)
    N: int = 32768                  # ring dim (默认 2^15)
    h: int = 64                     # secret-key Hamming weight (HWT(64))
    num_stages: int = 144           # 总 noise-injection 次数 (跨所有 transformer layer)
    delta_bits: int = 30            # 工作 scale Δ_working = 2^delta_bits
    sf_pt_min: int = 15             # 最小明文 sf (用于 CTPT encoding 项)
    include_ctpt_term: bool = True  # 是否加 CTPT encoding 误差项

    @property
    def sigma_per_stage(self) -> float:
        """sub-Gaussian σ for one rescale / KS injection (ct-domain)."""
        return math.sqrt(self.N * self.h / 12.0)

    @property
    def sigma_ct_total(self) -> float:
        """RMS 累加: σ_total = √(num_stages) · σ_per_stage."""
        return math.sqrt(self.num_stages) * self.sigma_per_stage

    @property
    def ctpt_term(self) -> float:
        """每 stage 的 CTPT encoding 误差贡献 (ct-domain), 假设每 stage 一个 CTPT。"""
        return self.sigma_per_stage * (2.0 ** (-self.sf_pt_min))

    def beta_logit_per_unit_sensitivity(self) -> float:
        """β_sim(x) = L_input(x) · this_value 中的"this_value"."""
        # ct-domain 总 noise (含 sub-Gaussian 累加 + CTPT encoding)
        sigma_ct = self.c_sigma * self.sigma_ct_total
        if self.include_ctpt_term:
            sigma_ct += self.c_sigma * math.sqrt(self.num_stages) * self.ctpt_term
        # 转到 logit space: 除以工作 scale Δ_working
        delta_working = 2.0 ** self.delta_bits
        return sigma_ct / delta_working


# ---------------------------------------------------------------------------
# 2. Input-level Jacobian norm via random-direction sampling
# ---------------------------------------------------------------------------


def estimate_input_jacobian_norm(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    num_directions: int = 4,
    eps: float = 1e-3,
) -> float:
    """随机方向 finite-difference 估计 ‖∂F/∂embedding‖_2 → L_input(x)。

    返回 L_input(x): 输入 embedding 的单位扰动对 logit ‖·‖_∞ 的最坏放大。
    """
    device = input_ids.device
    # 取 embedding output (要求模型暴露 .bert.embeddings)
    bert = getattr(model, "bert", None) or getattr(model, "roberta", None)
    if bert is None:
        raise RuntimeError(
            "model 没有 .bert / .roberta 属性，无法做 embedding-level perturbation. "
            "请改用 forward-hook 版或直接对 input_ids embedding 输出 perturb."
        )

    with torch.no_grad():
        embeds = bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )  # [B, L, H]
        # baseline forward (用 inputs_embeds bypass embedding lookup)
        out0 = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
        ).logits  # [B, K]

    L = []
    for _ in range(num_directions):
        v = torch.randn_like(embeds)
        v = v / (v.norm() + 1e-12)
        with torch.no_grad():
            out_p = model(
                inputs_embeds=embeds + eps * v,
                attention_mask=attention_mask,
            ).logits
        # ‖logit change‖_∞ / ‖perturbation‖_2  ≈  L_∞,2 of input Jacobian
        delta_logit = (out_p - out0).abs().max().item()
        L.append(delta_logit / eps)
    return max(L)  # worst-case across sampled directions


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------


def run(
    model_name: str,
    task_name: str,
    split: str,
    max_samples: Optional[int],
    batch_size: int,
    max_length: int,
    noise_params: NoiseModelParams,
    num_jacobian_dirs: int,
    device: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[load] tokenizer + model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()

    print(f"[load] dataset: glue/{task_name}/{split}")
    ds = load_dataset("glue", task_name, split=split)

    # MRPC 字段: sentence1, sentence2, label
    text_keys = {
        "mrpc": ("sentence1", "sentence2"),
        "rte": ("sentence1", "sentence2"),
        "stsb": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "wnli": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "cola": ("sentence", None),
    }
    keys = text_keys.get(task_name, ("sentence1", "sentence2"))

    def encode(ex):
        if keys[1] is None:
            return tokenizer(
                ex[keys[0]], padding="max_length", truncation=True, max_length=max_length
            )
        return tokenizer(
            ex[keys[0]],
            ex[keys[1]],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    ds = ds.map(encode, batched=False)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "label"],
    )
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"[run] total samples = {len(ds)}")
    print(f"[noise-model] {asdict(noise_params)}")
    beta_per_unit = noise_params.beta_logit_per_unit_sensitivity()
    print(f"[noise-model] β / L_input(x) = {beta_per_unit:.3e}")
    print(f"             (i.e., β_sim(x) = L_input(x) · {beta_per_unit:.3e})")
    print(f"             σ_ct_total = {noise_params.c_sigma * noise_params.sigma_ct_total:.3e}")
    print(f"             Δ_working  = 2^{noise_params.delta_bits} = {2.0 ** noise_params.delta_bits:.3e}")

    # 主循环
    records = []
    t0 = time.time()
    correct_plain = 0

    loader = DataLoader(ds, batch_size=batch_size)
    sample_idx = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits  # [B, K]

        # 明文 margin γ(x, y) per sample
        for i in range(logits.size(0)):
            z = logits[i]                         # [K]
            y = int(labels[i].item())
            z_y = float(z[y].item())
            mask = torch.ones_like(z, dtype=torch.bool)
            mask[y] = False
            z_other = float(z[mask].max().item())
            gamma = z_y - z_other                 # 可能 < 0 (明文就分错)
            pred = int(z.argmax().item())
            if pred == y:
                correct_plain += 1

            # estimate L_input(x)
            L_input = estimate_input_jacobian_norm(
                model,
                input_ids[i:i + 1],
                attention_mask[i:i + 1],
                token_type_ids[i:i + 1],
                num_directions=num_jacobian_dirs,
            )
            beta_sim = L_input * beta_per_unit
            ratio = gamma / (2 * beta_sim) if beta_sim > 0 else float("inf")

            records.append({
                "idx": sample_idx,
                "label": y,
                "pred": pred,
                "z_y": z_y,
                "z_max_other": z_other,
                "gamma": gamma,
                "L_input": L_input,
                "beta_sim": beta_sim,
                "ratio": ratio,
                "certified": (gamma > 2 * beta_sim) and (pred == y),
                "plain_correct": (pred == y),
            })
            sample_idx += 1

        if sample_idx % 50 == 0 or sample_idx == len(ds):
            elapsed = time.time() - t0
            print(
                f"[run] processed {sample_idx}/{len(ds)} "
                f"[{elapsed:.1f}s, {sample_idx/elapsed:.2f} samples/s]"
            )

    # ----- 汇总 -----
    arr_gamma = np.array([r["gamma"] for r in records])
    arr_beta = np.array([r["beta_sim"] for r in records])
    arr_ratio = np.array([r["ratio"] for r in records if math.isfinite(r["ratio"])])
    arr_L = np.array([r["L_input"] for r in records])

    n = len(records)
    cert_count = sum(1 for r in records if r["certified"])
    cert_acc = cert_count / n
    plain_acc = correct_plain / n

    print()
    print("=" * 72)
    print(f" Diagnose CertAcc on {model_name} / glue.{task_name} / {split}")
    print("=" * 72)
    print(f"  total samples       : {n}")
    print(f"  plaintext accuracy  : {plain_acc:.4f}")
    print(f"  CertAcc             : {cert_acc:.4f}  ({cert_count} / {n})")
    print()
    print(f"  γ(x,y) percentiles  :")
    for p in (5, 25, 50, 75, 95):
        print(f"     p{p:02d} = {np.percentile(arr_gamma, p):+.4f}")
    print(f"  β_sim(x) percentiles:")
    for p in (5, 25, 50, 75, 95):
        print(f"     p{p:02d} = {np.percentile(arr_beta, p):.4e}")
    print(f"  L_input(x) median   : {np.median(arr_L):.3e}")
    print(f"  ratio = γ / (2·β):")
    for p in (5, 25, 50, 75, 95):
        print(f"     p{p:02d} = {np.percentile(arr_ratio, p):.3e}")
    print()

    # ----- 诊断分级 -----
    median_ratio = float(np.median(arr_ratio))
    print(f"  median(ratio) = {median_ratio:.3e}")
    if median_ratio >= 100:
        verdict = "✅ STRONG: β_sim 远小于 γ，CertAcc≈plain_acc，v2 (IBP) framing 完美适用"
    elif median_ratio >= 1:
        verdict = "⚠️  MARGINAL: β_sim 与 γ 同量级，CertAcc 介于 0 和 plain_acc，v2 可用但需精细化 β"
    else:
        verdict = "❌ WEAK: β_sim 大于 γ/2，CertAcc 接近 0；v2 vacuous，建议切到 v1 (sub-Gaussian) 或 X2 framing"
    print(f"  verdict             : {verdict}")
    print("=" * 72)

    # ----- 写盘 -----
    out_json = os.path.join(output_dir, f"diagnose_{task_name}_{split}.json")
    summary = {
        "model": model_name,
        "task": task_name,
        "split": split,
        "n": n,
        "plain_accuracy": plain_acc,
        "cert_accuracy": cert_acc,
        "median_ratio": median_ratio,
        "verdict": verdict,
        "noise_params": asdict(noise_params),
        "noise_params_derived": {
            "sigma_per_stage_ct": noise_params.sigma_per_stage,
            "sigma_ct_total": noise_params.c_sigma * noise_params.sigma_ct_total,
            "delta_working": 2.0 ** noise_params.delta_bits,
            "beta_per_unit_L": beta_per_unit,
        },
        "percentiles": {
            "gamma": {p: float(np.percentile(arr_gamma, p)) for p in (5, 25, 50, 75, 95)},
            "beta": {p: float(np.percentile(arr_beta, p)) for p in (5, 25, 50, 75, 95)},
            "ratio": {p: float(np.percentile(arr_ratio, p)) for p in (5, 25, 50, 75, 95)},
        },
        "records": records,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote: {out_json}")

    # ----- 直方图 -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(arr_gamma, bins=40, alpha=0.6, label="γ(x,y)")
        axes[0].hist(2 * arr_beta, bins=40, alpha=0.6, label="2·β_sim(x)")
        axes[0].set_xlabel("logit-space magnitude")
        axes[0].set_ylabel("count")
        axes[0].set_title(f"γ vs 2β_sim (CertAcc = {cert_acc:.3f})")
        axes[0].legend()

        axes[1].hist(np.log10(arr_ratio + 1e-30), bins=40)
        axes[1].axvline(0, color="r", linestyle="--", label="ratio=1")
        axes[1].axvline(2, color="g", linestyle="--", label="ratio=100")
        axes[1].set_xlabel("log10(ratio)")
        axes[1].set_ylabel("count")
        axes[1].set_title(f"ratio = γ / (2·β_sim), median = {median_ratio:.2e}")
        axes[1].legend()
        plt.tight_layout()
        out_png = os.path.join(output_dir, f"diagnose_{task_name}_{split}.png")
        plt.savefig(out_png, dpi=120)
        print(f"  wrote: {out_png}")
    except ImportError:
        print("  (matplotlib 未安装, 跳过 histogram)")

    return summary


def main():
    p = argparse.ArgumentParser(description="Diagnose CertAcc for BERT on a GLUE task")
    p.add_argument("--model", default="textattack/bert-base-uncased-MRPC",
                   help="HuggingFace model name (default: BERT-MRPC)")
    p.add_argument("--task", default="mrpc", choices=[
        "mrpc", "rte", "stsb", "qqp", "mnli", "qnli", "wnli", "sst2", "cola"
    ])
    p.add_argument("--split", default="validation")
    p.add_argument("--max-samples", type=int, default=200,
                   help="只跑前 N 条; -1 表示全部")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default="./diagnose_certacc_output")

    # noise-model parameters
    p.add_argument("--c-sigma", type=float, default=6.0,
                   help="sub-Gaussian → ℓ_∞ 高置信因子 (6 ≈ 1e-9 confidence)")
    p.add_argument("--N", type=int, default=32768, help="ring dim (default 2^15)")
    p.add_argument("--h", type=int, default=64, help="secret HW (default HWT(64))")
    p.add_argument("--num-stages", type=int, default=144,
                   help="总 noise-injection 次数 across all transformer layers")
    p.add_argument("--delta-bits", type=int, default=30,
                   help="working scale Δ_working = 2^delta_bits")
    p.add_argument("--sf-pt-min", type=int, default=15,
                   help="min plaintext scaling factor (mask sf)")
    p.add_argument("--no-ctpt-term", action="store_true",
                   help="不加 CTPT encoding 项 (默认加)")

    p.add_argument("--num-jacobian-dirs", type=int, default=4,
                   help="random direction Jacobian estimate samples (per x)")

    args = p.parse_args()

    if args.max_samples == -1:
        args.max_samples = None

    noise_params = NoiseModelParams(
        c_sigma=args.c_sigma,
        N=args.N,
        h=args.h,
        num_stages=args.num_stages,
        delta_bits=args.delta_bits,
        sf_pt_min=args.sf_pt_min,
        include_ctpt_term=not args.no_ctpt_term,
    )

    run(
        model_name=args.model,
        task_name=args.task,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        noise_params=noise_params,
        num_jacobian_dirs=args.num_jacobian_dirs,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
