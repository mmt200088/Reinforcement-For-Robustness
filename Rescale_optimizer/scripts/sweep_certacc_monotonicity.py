"""
单调性扫描: CertAcc(λ · β_sim) vs λ.

读取 diagnose_certacc.py 已经写出的 JSON (含每条样本的 γ, β_sim, plain_correct),
扫一个噪声放大因子 λ ∈ [λ_min, λ_max]:

    β_λ(x) := λ · β_sim(x)
    CertAcc(λ) = #{x : pred(x)=y(x)  ∧  γ(x,y) > 2·β_λ(x)} / N

定理 Corollary 1 (cost-model monotonicity) 预言:
    λ ↗  ⇒  β_λ ↗  ⇒  CertAcc(λ) ↘  (单调非增)

期望图形:
    CertAcc 在 λ ≪ ratio_median 时 = plain_acc (平台)
    在 λ ≈ ratio_median 时 开始下降
    在 λ ≫ ratio_max 时 → 0
    全程单调非增 (绝不上升).

用法:
    python scripts/sweep_certacc_monotonicity.py \
        --in  diagnose_certacc_output/diagnose_mrpc_validation.json \
        --out diagnose_certacc_output/sweep_mrpc.png

可与 --num-points 100 配合得到平滑曲线.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Dict, Any

import numpy as np


def cert_acc_at(records: List[Dict[str, Any]], lam: float) -> float:
    n = len(records)
    if n == 0:
        return 0.0
    correct = sum(
        1
        for r in records
        if r["plain_correct"] and r["gamma"] > 2.0 * lam * r["beta_sim"]
    )
    return correct / n


def main():
    p = argparse.ArgumentParser(description="Sweep CertAcc(λ·β_sim) to demonstrate monotonicity")
    p.add_argument(
        "--in", dest="in_json",
        default="diagnose_certacc_output/diagnose_mrpc_validation.json",
        help="diagnose_certacc.py 输出的 JSON",
    )
    p.add_argument(
        "--out", dest="out_png",
        default=None,
        help="输出 PNG (默认: 与输入同目录, 文件名前缀 sweep_)",
    )
    p.add_argument("--lam-min", type=float, default=1e-3,
                   help="λ 扫描下界")
    p.add_argument("--lam-max", type=float, default=1e10,
                   help="λ 扫描上界 (默认远超过 ratio_median, 让 CertAcc → 0)")
    p.add_argument("--num-points", type=int, default=200,
                   help="λ 采样点数 (log-spaced)")
    p.add_argument("--mark-physical", action="store_true",
                   help="在图上标几个物理意义的 λ:")
    args = p.parse_args()

    with open(args.in_json) as f:
        data = json.load(f)

    records = data["records"]
    n = len(records)
    plain_acc = data["plain_accuracy"]
    cert_acc_at_one = data["cert_accuracy"]
    median_ratio = data["median_ratio"]

    print(f"[load] {args.in_json}")
    print(f"       n = {n}, plain_acc = {plain_acc:.4f}, "
          f"CertAcc(λ=1) = {cert_acc_at_one:.4f}, "
          f"median_ratio = {median_ratio:.3e}")

    # ----- 扫描 -----
    lams = np.geomspace(args.lam_min, args.lam_max, args.num_points)
    accs = np.array([cert_acc_at(records, float(lam)) for lam in lams])

    # ----- 单调性检查 (严格非增, 允许 ε 数值漂移) -----
    diffs = np.diff(accs)
    if np.all(diffs <= 1e-12):
        mono_status = "✅ STRICTLY MONOTONIC (CertAcc(λ) is non-increasing in λ, as predicted by Corollary 1)"
    else:
        # 数值上 CertAcc 是离散阶梯, 永远不该上升; 如果上升了说明有 bug
        bad = np.where(diffs > 1e-12)[0]
        mono_status = (f"❌ NON-MONOTONIC at {len(bad)} points "
                       f"(first violation: λ={lams[bad[0]]:.3e}, "
                       f"acc {accs[bad[0]]:.4f} → {accs[bad[0]+1]:.4f}). "
                       f"BUG in noise model or β_sim computation.")
    print(f"[check] {mono_status}")

    # 关键 λ 值
    def acc_at(lam):
        return cert_acc_at(records, lam)
    print()
    print(f"  CertAcc(λ=1e-3)   = {acc_at(1e-3):.4f}   ← 极小噪声,  CertAcc ≈ plain_acc")
    print(f"  CertAcc(λ=1)     = {acc_at(1.0):.4f}    ← 当前优化器配置 (论文主结果)")
    print(f"  CertAcc(λ=1e3)   = {acc_at(1e3):.4f}    ← 噪声 ×1000")
    print(f"  CertAcc(λ=1e6)   = {acc_at(1e6):.4f}    ← 噪声 ×10⁶ (接近 median ratio)")
    print(f"  CertAcc(λ={median_ratio:.1e}) = {acc_at(median_ratio):.4f}   ← 噪声 = median(γ/2β), 应该 ≈ 0.5 · plain_acc")
    print(f"  CertAcc(λ=1e10)  = {acc_at(1e10):.4f}    ← 极端噪声,  CertAcc → 0")

    # ----- 画图 -----
    out_png = args.out_png
    if out_png is None:
        d = os.path.dirname(args.in_json) or "."
        base = os.path.basename(args.in_json).replace(".json", "")
        out_png = os.path.join(d, f"sweep_{base}.png")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lams, accs, "b-", linewidth=2.0, label="CertAcc(λ · β_sim)")
        ax.axhline(plain_acc, color="g", linestyle=":", label=f"plain_acc = {plain_acc:.3f}")
        ax.axhline(0, color="k", linestyle=":", alpha=0.3)
        ax.axvline(1.0, color="r", linestyle="--",
                   label=f"λ=1 (current optimizer config), CertAcc = {cert_acc_at_one:.3f}")
        ax.axvline(median_ratio, color="orange", linestyle="--",
                   label=f"λ = median(γ/2β) = {median_ratio:.2e}")

        if args.mark_physical:
            # 把 c_sigma×10 / num_stages×100 / Δ÷2^10 这些物理事件标到图上
            np_ = data["noise_params"]
            # 物理 λ 含义示例
            ax.axvline(10, color="gray", linestyle=":", alpha=0.5)
            ax.text(10, 0.05, "c_σ×10\n(δ→1e-87)", rotation=90, fontsize=8, color="gray")
            ax.axvline(100, color="gray", linestyle=":", alpha=0.5)
            ax.text(100, 0.05, "num_stages×10⁴", rotation=90, fontsize=8, color="gray")
            ax.axvline(2 ** 10, color="gray", linestyle=":", alpha=0.5)
            ax.text(2 ** 10, 0.05, "Δ÷2¹⁰", rotation=90, fontsize=8, color="gray")

        ax.set_xscale("log")
        ax.set_xlabel("λ (noise multiplier; β_λ = λ · β_sim)")
        ax.set_ylabel("Certified Accuracy")
        ax.set_title(
            f"Monotonicity sweep: CertAcc(λ · β_sim) on {data['task']}/{data['split']}\n"
            f"({mono_status.split()[0]} {mono_status.split()[1]} per Corollary 1)"
        )
        ax.set_ylim(-0.02, max(plain_acc + 0.05, 1.0))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f"\n  wrote: {out_png}")
    except ImportError:
        print("  (matplotlib 未安装, 跳过画图)")

    # 同时把数据 dump 出来供论文表格使用
    out_csv = out_png.replace(".png", "_curve.csv")
    with open(out_csv, "w") as f:
        f.write("lambda,cert_acc\n")
        for lam, acc in zip(lams, accs):
            f.write(f"{lam},{acc}\n")
    print(f"  wrote: {out_csv}")


if __name__ == "__main__":
    main()
