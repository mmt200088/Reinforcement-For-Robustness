import os
import re
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageClassification, AutoModelForSequenceClassification

# =========================
# 1) 需要统计的模型（按架构：BERT / GPT-2 / ViT / OPT）
# =========================
MODEL_IDS = {

    # ViT (image classification)
    "ViT-Base": "google/vit-base-patch16-224",
    "ViT-Large": "google/vit-large-patch16-224",

    # GPT-2
    "GPT2": "openai-community/gpt2",
    "GPT2-Medium": "openai-community/gpt2-medium",
    "GPT2-Large": "openai-community/gpt2-large",
    "GPT2-XL": "openai-community/gpt2-xl",

    # OPT (causal LM)
    "OPT-1.3B": "facebook/opt-1.3b",

    # BERT-Base (GLUE)
    "BERT-Base-QNLI": "textattack/bert-base-uncased-QNLI",
    "BERT-Base-RTE":  "textattack/bert-base-uncased-RTE",
    "BERT-Base-MRPC": "textattack/bert-base-uncased-MRPC",
    # BERT-Large
    "BERT-Large": "google-bert/bert-large-uncased",
    "BERT-Large-CoLA": "yoshitomo-matsubara/bert-large-uncased-cola",
    "BERT-Large-SST2": "assemblyai/bert-large-uncased-sst2",
}

# Encoder-only models (no classification head); load with AutoModel
ENCODER_ONLY_IDS = {"google-bert/bert-large-uncased"}

# Image models (ViT etc.); load with AutoModelForImageClassification
IMAGE_MODEL_IDS = {"google/vit-base-patch16-224", "google/vit-large-patch16-224"}

# Causal LM (OPT etc.); load with AutoModelForCausalLM
CAUSAL_LM_IDS = {"facebook/opt-1.3b"}

OUT_DIR = "weight_hist_out"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# 按数量级分桶：最小从 1e-8 开始（你可改成 1e-10 等）
MIN_EDGE = 1e-8

# 每次分块处理多少元素（避免一次性占用过大显存/内存）
CHUNK_ELEMS = 5_000_000

# =========================
# 2) 类别规则 + 层号解析（按架构：BERT / GPT-2）
# =========================
# BERT (base & large): bert.encoder.layer.<i>.*
BERT_CATEGORY_PATTERNS = [
    (r"\.attention\.self\.query\.", "Attn/Q"),
    (r"\.attention\.self\.key\.",   "Attn/K"),
    (r"\.attention\.self\.value\.", "Attn/V"),
    (r"\.attention\.output\.dense\.", "Attn/O"),
    (r"\.intermediate\.dense\.", "FFN/Intermediate"),
    (r"\.output\.dense\.",       "FFN/OutputDense"),
    (r"\.LayerNorm\.", "LayerNorm"),
    (r"^bert\.embeddings\.", "Embeddings"),
    (r"^bert\.pooler\.", "Pooler"),
    (r"^classifier\.", "Classifier"),
]
BERT_LAYER_RE = re.compile(r"bert\.encoder\.layer\.(\d+)\.")

# GPT-2: transformer.h.<i>.*
GPT2_CATEGORY_PATTERNS = [
    (r"\.attn\.c_attn\.", "Attn/QKV"),
    (r"\.attn\.c_proj\.", "Attn/O"),
    (r"\.mlp\.c_fc\.", "FFN/Intermediate"),
    (r"\.mlp\.c_proj\.", "FFN/Output"),
    (r"\.ln_1\.", "LayerNorm/ln_1"),
    (r"\.ln_2\.", "LayerNorm/ln_2"),
    (r"^transformer\.wte\.", "Embeddings/wte"),
    (r"^transformer\.wpe\.", "Embeddings/wpe"),
    (r"^transformer\.ln_f\.", "LayerNorm/ln_f"),
    (r"^score\.", "Classifier"),
    (r"\.lm_head\.", "Classifier"),
]
GPT2_LAYER_RE = re.compile(r"transformer\.h\.(\d+)\.")

# ViT: vit.encoder.layer.<i>.* (same encoder layout as BERT but with vit prefix)
VIT_CATEGORY_PATTERNS = [
    (r"\.attention\.attention\.query\.", "Attn/Q"),
    (r"\.attention\.attention\.key\.", "Attn/K"),
    (r"\.attention\.attention\.value\.", "Attn/V"),
    (r"\.attention\.output\.dense\.", "Attn/O"),
    (r"\.intermediate\.dense\.", "FFN/Intermediate"),
    (r"\.output\.dense\.", "FFN/OutputDense"),
    (r"\.layernorm_before\.", "LayerNorm/before"),
    (r"\.layernorm_after\.", "LayerNorm/after"),
    (r"^vit\.embeddings\.", "Embeddings"),
    (r"^vit\.layernorm\.", "LayerNorm/final"),
    (r"^classifier\.", "Classifier"),
]
VIT_LAYER_RE = re.compile(r"vit\.encoder\.layer\.(\d+)\.")

# OPT: model.decoder.layers.<i>.* (decoder-only, q_proj/k_proj/v_proj/out_proj, fc1/fc2)
OPT_CATEGORY_PATTERNS = [
    (r"\.self_attn\.q_proj\.", "Attn/Q"),
    (r"\.self_attn\.k_proj\.", "Attn/K"),
    (r"\.self_attn\.v_proj\.", "Attn/V"),
    (r"\.self_attn\.out_proj\.", "Attn/O"),
    (r"\.fc1\.", "FFN/Intermediate"),
    (r"\.fc2\.", "FFN/Output"),
    (r"\.self_attn_layer_norm\.", "LayerNorm/self_attn"),
    (r"decoder\.layers\.\d+\.final_layer_norm\.", "LayerNorm/final"),
    (r"^model\.decoder\.embed_tokens\.", "Embeddings"),
    (r"^model\.decoder\.embed_positions\.", "Embeddings/positions"),
    (r"^model\.decoder\.final_layer_norm\.", "LayerNorm/decoder_final"),
    (r"^model\.decoder\.project_in\.", "Embeddings/project_in"),
    (r"^model\.decoder\.project_out\.", "Embeddings/project_out"),
    (r"\.lm_head\.", "Classifier"),
]
OPT_LAYER_RE = re.compile(r"model\.decoder\.layers\.(\d+)\.")

ARCH_CONFIG = {
    "bert": (BERT_LAYER_RE, BERT_CATEGORY_PATTERNS),
    "gpt2": (GPT2_LAYER_RE, GPT2_CATEGORY_PATTERNS),
    "vit": (VIT_LAYER_RE, VIT_CATEGORY_PATTERNS),
    "opt": (OPT_LAYER_RE, OPT_CATEGORY_PATTERNS),
}

def parse_layer(name: str, layer_re: re.Pattern) -> str:
    m = layer_re.search(name)
    if m:
        return f"L{int(m.group(1))}"
    return "NO_LAYER"

def categorize(name: str, category_patterns: List[Tuple[str, str]]) -> str:
    suffix = "weight" if name.endswith(".weight") else ("bias" if name.endswith(".bias") else "other")
    for pat, cat in category_patterns:
        if re.search(pat, name):
            return f"{cat}/{suffix}"
    return f"Other/{suffix}"

def get_arch_config(model) -> Tuple[re.Pattern, List[Tuple[str, str]]]:
    """从 model.config 或参数名推断架构，返回 (layer_re, category_patterns)。"""
    model_type = getattr(model.config, "model_type", None)
    if model_type and model_type in ARCH_CONFIG:
        return ARCH_CONFIG[model_type]
    for pname, _ in model.named_parameters():
        if "bert.encoder.layer." in pname:
            return ARCH_CONFIG["bert"]
        if "transformer.h." in pname:
            return ARCH_CONFIG["gpt2"]
        if "vit.encoder.layer." in pname:
            return ARCH_CONFIG["vit"]
        if "model.decoder.layers." in pname:
            return ARCH_CONFIG["opt"]
        break
    return ARCH_CONFIG["bert"]

# =========================
# 3) 统计结构
# =========================
# Outlier thresholds: count and report |w| > these
OUTLIER_THRESHOLDS = (1.0, 10.0)

@dataclass
class AggStats:
    key: str
    n_total: int = 0
    n_zero: int = 0
    n_abs_gt_1: int = 0
    n_abs_gt_10: int = 0
    min_val: float = float("inf")
    max_val: float = float("-inf")
    max_abs: float = 0.0
    # counts_per_bin 不含 0（0 单独统计在 n_zero）
    counts_per_bin: np.ndarray = None  # shape=(B,)

    def ensure_bins(self, B: int):
        if self.counts_per_bin is None:
            self.counts_per_bin = np.zeros(B, dtype=np.int64)

    def update_minmax(self, t: torch.Tensor):
        # 精确 min/max/maxabs（对整个张量）
        # 这里在 DEVICE 上做，避免来回拷贝
        x = t.detach().to(device=DEVICE, dtype=DTYPE)
        self.n_total += x.numel()
        self.min_val = min(self.min_val, float(x.min().item()))
        self.max_val = max(self.max_val, float(x.max().item()))
        self.max_abs = max(self.max_abs, float(x.abs().max().item()))

    def add_hist_counts(self, zero_cnt: int, bin_counts: np.ndarray, n_abs_gt_1: int = 0, n_abs_gt_10: int = 0):
        self.n_zero += int(zero_cnt)
        self.counts_per_bin += bin_counts
        self.n_abs_gt_1 += int(n_abs_gt_1)
        self.n_abs_gt_10 += int(n_abs_gt_10)

# =========================
# 4) 构造“数量级 bins”（10 的幂）
# =========================
def make_magnitude_bins(max_abs: float, min_edge: float = MIN_EDGE) -> np.ndarray:
    """
    返回 edges: [1e-8, 1e-7, 1e-6, ..., 1eK]，保证最后一个 edge > max_abs
    """
    max_abs = max(float(max_abs), min_edge * 10.0)
    emin = int(math.floor(math.log10(min_edge)))
    emax = int(math.ceil(math.log10(max_abs))) + 1  # +1 确保覆盖最大值
    edges = np.array([10.0 ** e for e in range(emin, emax + 1)], dtype=np.float64)
    # edges 长度 = B+1
    return edges

def bin_labels(edges: np.ndarray) -> List[str]:
    """
    edges: [1e-8,1e-7,...]
    bins 表示 (edges[i], edges[i+1]]（我们用 bucketize/right=False + clamp 实现近似）
    """
    labs = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i+1]
        labs.append(f"({a:.0e},{b:.0e}]")
    return labs

# =========================
# 5) 对一个张量做全量直方图（不采样）
# =========================
def hist_abs_tensor(t: torch.Tensor, edges: np.ndarray, chunk_elems: int = CHUNK_ELEMS) -> Tuple[int, np.ndarray, int, int]:
    """
    统计 |t| 的：
      - zero_cnt: 等于0的个数
      - bin_counts: 每个数量级 bin 的计数（不含0）
      - n_abs_gt_1, n_abs_gt_10: 满足 |w|>1 和 |w|>10 的个数
    """
    x = t.detach().to(device=DEVICE, dtype=DTYPE).view(-1)
    n = x.numel()

    edges_t = torch.from_numpy(edges).to(device=DEVICE, dtype=DTYPE)
    B = len(edges) - 1
    bin_counts = np.zeros(B, dtype=np.int64)
    zero_cnt = 0
    n_abs_gt_1 = 0
    n_abs_gt_10 = 0

    for start in range(0, n, chunk_elems):
        seg = x[start:start+chunk_elems].abs()

        # 0 单独统计
        zero_cnt += int((seg == 0).sum().item())
        # outliers
        n_abs_gt_1 += int((seg > 1.0).sum().item())
        n_abs_gt_10 += int((seg > 10.0).sum().item())

        seg = seg[seg > 0]
        if seg.numel() == 0:
            continue

        # clip到 edges 范围内（避免数值落在最后边界外）
        seg = torch.clamp(seg, min=float(edges[0]), max=float(edges[-1]) * (1 - 1e-7))

        # bucketize：返回 1..len(edges)-1，减1得到 0..B-1
        idx = torch.bucketize(seg, edges_t, right=False) - 1
        idx = torch.clamp(idx, 0, B - 1)

        bc = torch.bincount(idx, minlength=B).to(torch.int64).detach().cpu().numpy()
        bin_counts += bc

    return zero_cnt, bin_counts, n_abs_gt_1, n_abs_gt_10

# =========================
# 6) 输出 CSV + 画图（百分比柱状图）
# =========================
def save_csv(model_name: str, stats: Dict[str, AggStats], edges: np.ndarray):
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{model_name}_magnitude_hist.csv")
    labels = bin_labels(edges)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["model", "scope", "category", "n_total", "pct_zero", "n_abs_gt_1", "pct_abs_gt_1", "n_abs_gt_10", "pct_abs_gt_10", "min", "max", "maxabs"] + \
                 [f"pct_{lab}" for lab in labels]
        w.writerow(header)

        for key in sorted(stats.keys()):
            scope, cat = key.split("|", 1)
            s = stats[key]
            total = s.n_total if s.n_total > 0 else 1

            pct_zero = 100.0 * s.n_zero / total
            pct_gt_1 = 100.0 * s.n_abs_gt_1 / total
            pct_gt_10 = 100.0 * s.n_abs_gt_10 / total
            pct_bins = 100.0 * (s.counts_per_bin.astype(np.float64) / total)

            w.writerow([model_name, scope, cat, s.n_total, pct_zero, s.n_abs_gt_1, pct_gt_1, s.n_abs_gt_10, pct_gt_10, s.min_val, s.max_val, s.max_abs, *pct_bins.tolist()])

    print(f"[{model_name}] CSV saved to: {csv_path}")

def plot_all_category_bars(model_name: str, stats: Dict[str, AggStats], edges: np.ndarray):
    """
    对 ALL|category 画柱状图：y=百分比
    """
    labels = ["0"] + bin_labels(edges)
    x = np.arange(len(labels))

    all_keys = [k for k in stats.keys() if k.startswith("ALL|")]
    for key in sorted(all_keys):
        cat = key.split("|", 1)[1]
        s = stats[key]
        if s.n_total == 0:
            continue

        pct_zero = 100.0 * s.n_zero / s.n_total
        pct_bins = 100.0 * (s.counts_per_bin.astype(np.float64) / s.n_total)
        y = np.concatenate([[pct_zero], pct_bins])

        plt.figure(figsize=(12, 4))
        plt.bar(x, y)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("percentage (%)")
        plt.title(f"{model_name} ALL {cat}  |w| magnitude histogram")
        plt.tight_layout()

        out_path = os.path.join(PLOTS_DIR, f"{model_name}_ALL_{cat.replace('/','_')}_magnitude_bar.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

def save_outlier_summary(summary: List[Dict]) -> None:
    """Write per-model outlier counts and percentages to CSV."""
    csv_path = os.path.join(OUT_DIR, "outlier_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_total", "n_abs_gt_1", "pct_abs_gt_1", "n_abs_gt_10", "pct_abs_gt_10"])
        for row in summary:
            w.writerow([row["model"], row["n_total"], row["n_abs_gt_1"], row["pct_abs_gt_1"], row["n_abs_gt_10"], row["pct_abs_gt_10"]])
    print(f"Outlier summary CSV saved to: {csv_path}")


def plot_outlier_summary(summary: List[Dict]) -> None:
    """Bar chart: per model, percentage of weights with |w|>1 and |w|>10."""
    if not summary:
        return
    models = [r["model"] for r in summary]
    pct_gt_1 = [r["pct_abs_gt_1"] for r in summary]
    pct_gt_10 = [r["pct_abs_gt_10"] for r in summary]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.8), 5))
    bars1 = ax.bar(x - width / 2, pct_gt_1, width, label="|w| > 1 (%)")
    bars2 = ax.bar(x + width / 2, pct_gt_10, width, label="|w| > 10 (%)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Model")
    ax.set_title("Weight outliers: |w| > 1 and |w| > 10 (all params)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "outlier_summary_pct.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Outlier summary plot saved to: {out_path}")


def main():
    print(f"Using DEVICE={DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)
    outlier_summary: List[Dict] = []

    for model_name, model_id in MODEL_IDS.items():
        print(f"\n=== Loading {model_name}: {model_id} ===")
        if model_id in ENCODER_ONLY_IDS:
            model = AutoModel.from_pretrained(model_id).to(DEVICE)
        elif model_id in IMAGE_MODEL_IDS:
            model = AutoModelForImageClassification.from_pretrained(model_id).to(DEVICE)
        elif model_id in CAUSAL_LM_IDS:
            model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE)
        model.eval()

        layer_re, category_patterns = get_arch_config(model)
        print(f"[{model_name}] arch config: layer_re={layer_re.pattern[:40]}...")

        # -------- pass 1: 扫一遍拿 global maxabs，顺便精确 min/max/maxabs（按 ALL|cat 和 Lk|cat）--------
        pass1: Dict[str, AggStats] = {}
        global_maxabs = 0.0

        def keys_for_param(pname: str) -> List[str]:
            layer = parse_layer(pname, layer_re)
            cat = categorize(pname, category_patterns)
            return [f"ALL|{cat}", f"{layer}|{cat}"]

        with torch.no_grad():
            for pname, p in model.named_parameters():
                # 更新global maxabs
                global_maxabs = max(global_maxabs, float(p.detach().to(DEVICE, dtype=DTYPE).abs().max().item()))

                for key in keys_for_param(pname):
                    if key not in pass1:
                        pass1[key] = AggStats(key=key)
                    pass1[key].update_minmax(p)

        edges = make_magnitude_bins(global_maxabs, min_edge=MIN_EDGE)
        B = len(edges) - 1
        print(f"[{model_name}] max|w|={global_maxabs:.6g}, bins={B}, edges[{edges[0]:.0e}..{edges[-1]:.0e}]")

        # -------- pass 2: 全量直方图（每个参数算一次 hist，再累加到 ALL 与 layer）+ outlier 计数 --------
        stats: Dict[str, AggStats] = {}
        for k, s1 in pass1.items():
            s = AggStats(key=k, n_total=s1.n_total, n_zero=0, min_val=s1.min_val, max_val=s1.max_val, max_abs=s1.max_abs)
            s.ensure_bins(B)
            stats[k] = s

        with torch.no_grad():
            for pname, p in model.named_parameters():
                zero_cnt, bin_counts, n_gt_1, n_gt_10 = hist_abs_tensor(p, edges, chunk_elems=CHUNK_ELEMS)
                for key in keys_for_param(pname):
                    stats[key].add_hist_counts(zero_cnt, bin_counts, n_gt_1, n_gt_10)

        save_csv(model_name, stats, edges)
        plot_all_category_bars(model_name, stats, edges)

        # -------- 汇总该模型全局 outlier，用于最后画总图 --------
        all_keys = [k for k in stats.keys() if k.startswith("ALL|")]
        total_params = sum(stats[k].n_total for k in all_keys)
        total_gt_1 = sum(stats[k].n_abs_gt_1 for k in all_keys)
        total_gt_10 = sum(stats[k].n_abs_gt_10 for k in all_keys)
        outlier_summary.append({
            "model": model_name,
            "n_total": total_params,
            "n_abs_gt_1": total_gt_1,
            "pct_abs_gt_1": 100.0 * total_gt_1 / total_params if total_params else 0,
            "n_abs_gt_10": total_gt_10,
            "pct_abs_gt_10": 100.0 * total_gt_10 / total_params if total_params else 0,
        })

    # -------- 写入 outlier 汇总 CSV 并画图 --------
    save_outlier_summary(outlier_summary)
    plot_outlier_summary(outlier_summary)
    print("\nAll done.")

if __name__ == "__main__":
    main()