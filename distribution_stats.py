#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect per-layer weight and activation distribution stats:
abs-max, p99, p99.9, p99.99.

- Weight stats: runs once (model parameters).
- Activation stats: runs during forward passes on sample inputs using forward hooks.

Outputs:
  - weight_stats.csv
  - act_stats.csv

Usage example:
  python collect_layer_stats.py \
    --base_model textattack/bert-base-uncased-STS-B \
    --task stsb \
    --max_batches 20 \
    --batch_size 8 \
    --seq_len 128 \
    --out_dir stats_out
"""

import os
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------
# Utilities
# ---------------------------

PCTS = [99.0, 99.9, 99.99]

def _to_numpy_abs_flat(x: torch.Tensor) -> np.ndarray:
    # Move to CPU, float32, abs, flatten
    return x.detach().to(dtype=torch.float32).abs().reshape(-1).cpu().numpy()

def _percentiles_from_np(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"max": float("nan"), "p99": float("nan"), "p99.9": float("nan"), "p99.99": float("nan")}
    vals = np.percentile(arr, PCTS).tolist()
    return {
        "max": float(np.max(arr)),
        "p99": float(vals[0]),
        "p99.9": float(vals[1]),
        "p99.99": float(vals[2]),
    }

def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")

def _shape_str(t: torch.Tensor) -> str:
    return "x".join(map(str, list(t.shape)))

def safe_numel(t: torch.Tensor) -> int:
    try:
        return int(t.numel())
    except Exception:
        return -1


# ---------------------------
# Activation collector with streaming stats
# ---------------------------

@dataclass
class RunningStats:
    """
    We want percentiles. Exact percentiles require storing all values, which is too big.
    Practical approach:
      - reservoir sampling or histograms.
    Here: use reservoir sampling per layer tensor-kind (in/out).
    """
    max_abs: float = 0.0
    seen: int = 0
    reservoir_size: int = 200_000
    # store float32 abs values
    reservoir: Optional[np.ndarray] = None

    def update(self, x_abs_flat: np.ndarray):
        if x_abs_flat.size == 0:
            return
        # update max
        cur_max = float(np.max(x_abs_flat))
        if cur_max > self.max_abs:
            self.max_abs = cur_max

        # init reservoir
        if self.reservoir is None:
            take = min(self.reservoir_size, x_abs_flat.size)
            self.reservoir = np.empty((self.reservoir_size,), dtype=np.float32)
            self.reservoir[:take] = x_abs_flat[:take].astype(np.float32, copy=False)
            self.seen = x_abs_flat.size
            self._filled = take
            return

        # reservoir sampling
        n = x_abs_flat.size
        self.seen += n
        # Fill remaining if not full yet
        if getattr(self, "_filled", 0) < self.reservoir_size:
            space = self.reservoir_size - self._filled
            take = min(space, n)
            self.reservoir[self._filled:self._filled + take] = x_abs_flat[:take].astype(np.float32, copy=False)
            self._filled += take
            x_abs_flat = x_abs_flat[take:]
            n = x_abs_flat.size
            if n == 0:
                return

        # Now reservoir is full; do sampling
        # For each new item with global index i, replace with prob reservoir_size/i
        # Vectorized approx: sample random indices and thresholds
        # We'll do a chunked loop to keep it simple and stable.
        start_global = self.seen - n  # index of first of this chunk in 1..seen (conceptually)
        for j in range(n):
            i = start_global + j + 1  # 1-based
            r = np.random.randint(0, i)
            if r < self.reservoir_size:
                self.reservoir[r] = float(x_abs_flat[j])

    def finalize(self) -> Dict[str, float]:
        if self.reservoir is None:
            return {"max": float("nan"), "p99": float("nan"), "p99.9": float("nan"), "p99.99": float("nan")}
        filled = getattr(self, "_filled", self.reservoir_size)
        sample = self.reservoir[:filled]
        vals = np.percentile(sample, PCTS).tolist()
        return {
            "max": float(self.max_abs),
            "p99": float(vals[0]),
            "p99.9": float(vals[1]),
            "p99.99": float(vals[2]),
        }


class ActivationStatsCollector:
    def __init__(
        self,
        model: nn.Module,
        module_filter: Tuple[type, ...] = (nn.Linear,),
        record_input: bool = True,
        record_output: bool = True,
        reservoir_size: int = 200_000,
    ):
        self.model = model
        self.module_filter = module_filter
        self.record_input = record_input
        self.record_output = record_output
        self.reservoir_size = reservoir_size

        # key: (module_name, kind) where kind in {"in", "out"}
        self.stats: Dict[Tuple[str, str], RunningStats] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _get_stats(self, name: str, kind: str) -> RunningStats:
        key = (name, kind)
        if key not in self.stats:
            self.stats[key] = RunningStats(reservoir_size=self.reservoir_size)
        return self.stats[key]

    def _hook(self, name: str):
        def fn(module: nn.Module, inputs, output):
            # inputs is a tuple
            if self.record_input and len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
                arr = _to_numpy_abs_flat(inputs[0])
                self._get_stats(name, "in").update(arr)

            if self.record_output:
                out_t = None
                if isinstance(output, torch.Tensor):
                    out_t = output
                elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                    out_t = output[0]
                if out_t is not None:
                    arr = _to_numpy_abs_flat(out_t)
                    self._get_stats(name, "out").update(arr)
        return fn

    def install(self):
        for name, m in self.model.named_modules():
            if isinstance(m, self.module_filter):
                h = m.register_forward_hook(self._hook(name))
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for (name, kind), rs in self.stats.items():
            fin = rs.finalize()
            rows.append({
                "module": name,
                "kind": kind,  # in/out
                **fin,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["module", "kind"]).reset_index(drop=True)
        return df


# ---------------------------
# Weight stats
# ---------------------------

def collect_weight_stats(
    model: nn.Module,
    include_embeddings: bool = True,
    only_linears: bool = False,
) -> pd.DataFrame:
    rows = []
    for name, m in model.named_modules():
        if only_linears and not isinstance(m, nn.Linear):
            continue
        if isinstance(m, nn.Linear):
            w = m.weight
            arr = _to_numpy_abs_flat(w)
            st = _percentiles_from_np(arr)
            rows.append({
                "module": name,
                "type": "Linear.weight",
                "shape": _shape_str(w),
                "dtype": _dtype_name(w),
                **st,
            })
            if m.bias is not None:
                b = m.bias
                arrb = _to_numpy_abs_flat(b)
                stb = _percentiles_from_np(arrb)
                rows.append({
                    "module": name,
                    "type": "Linear.bias",
                    "shape": _shape_str(b),
                    "dtype": _dtype_name(b),
                    **stb,
                })
        elif include_embeddings and isinstance(m, nn.Embedding):
            w = m.weight
            arr = _to_numpy_abs_flat(w)
            st = _percentiles_from_np(arr)
            rows.append({
                "module": name,
                "type": "Embedding.weight",
                "shape": _shape_str(w),
                "dtype": _dtype_name(w),
                **st,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["module", "type"]).reset_index(drop=True)
    return df

def collect_all_matrix_param_stats(model: nn.Module) -> pd.DataFrame:
    rows = []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 只统计“矩阵类参数”：Linear/Conv/Embedding 等权重都会进来
        if p.ndim >= 2:
            arr = _to_numpy_abs_flat(p)
            st = _percentiles_from_np(arr)
            rows.append({
                "param": pname,
                "ndim": int(p.ndim),
                "shape": "x".join(map(str, p.shape)),
                "dtype": str(p.dtype).replace("torch.", ""),
                **st,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["param"]).reset_index(drop=True)
    return df

def collect_qkv_per_head_stats(model: nn.Module) -> pd.DataFrame:
    """
    对常见命名包含 query/key/value 的权重做 per-head 切片统计。
    适用于 BERT 类：query.weight 形状 [hidden, hidden]（out,in）
    head 切片：沿 out 维度切成 num_heads 份，每份 size=head_dim。
    """
    cfg = getattr(model, "config", None)
    if cfg is None or not hasattr(cfg, "num_attention_heads") or not hasattr(cfg, "hidden_size"):
        return pd.DataFrame([])

    H = int(cfg.hidden_size)
    nh = int(cfg.num_attention_heads)
    assert H % nh == 0, "hidden_size must be divisible by num_attention_heads"
    hd = H // nh

    rows = []

    # 兼容不同命名：query/key/value 或 q_proj/k_proj/v_proj 或 in_proj_weight / qkv
    for pname, p in model.named_parameters():
        if p.ndim != 2:
            continue

        name_lower = pname.lower()

        def add_heads(tag: str, mat: torch.Tensor):
            # mat assumed shape [out, in]
            if mat.shape[0] % nh != 0:
                return
            head_dim = mat.shape[0] // nh
            for h in range(nh):
                sl = mat[h*head_dim:(h+1)*head_dim, :]
                arr = _to_numpy_abs_flat(sl)
                st = _percentiles_from_np(arr)
                rows.append({
                    "param": pname,
                    "tag": tag,
                    "head": h,
                    "shape": "x".join(map(str, sl.shape)),
                    **st,
                })

        # (1) BERT-style：独立的 query/key/value
        if any(k in name_lower for k in [".query.weight", ".key.weight", ".value.weight",
                                         ".q_proj.weight", ".k_proj.weight", ".v_proj.weight"]):
            # tag = q/k/v
            if ".query." in name_lower or ".q_proj." in name_lower:
                add_heads("q", p)
            elif ".key." in name_lower or ".k_proj." in name_lower:
                add_heads("k", p)
            elif ".value." in name_lower or ".v_proj." in name_lower:
                add_heads("v", p)

        # (2) fused qkv：形状可能是 [3H, H]（out,in），把 out 维拆成 q/k/v 三段
        elif any(k in name_lower for k in ["qkv.weight", "in_proj_weight"]):
            if p.shape[0] % 3 == 0:
                out = p.shape[0]
                one = out // 3
                q, k, v = p[:one, :], p[one:2*one, :], p[2*one:, :]
                add_heads("q", q)
                add_heads("k", k)
                add_heads("v", v)

        # (3) O 投影：也可以按 head 切（但注意它会混合 heads，解释时要小心）
        # 如果你仍想看：沿输入维度切更有意义（每个 head 对应输入的一段）
        elif any(k in name_lower for k in [".attention.output.dense.weight", ".o_proj.weight"]):
            # mat shape [H, H]：按输入维度切分（每个 head 的输入 chunk）
            if p.shape == (H, H):
                for h in range(nh):
                    sl = p[:, h*hd:(h+1)*hd]  # 切 input channels
                    arr = _to_numpy_abs_flat(sl)
                    st = _percentiles_from_np(arr)
                    rows.append({
                        "param": pname,
                        "tag": "o_inchunk",
                        "head": h,
                        "shape": "x".join(map(str, sl.shape)),
                        **st,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["param", "tag", "head"]).reset_index(drop=True)
    return df


# ---------------------------
# Data: simple sample texts
# ---------------------------

def default_text_pairs():
    # For STS-B-like regression, provide sentence pairs.
    return [
        ("A man is playing a guitar.", "A person plays a musical instrument."),
        ("Two dogs are running through a field.", "Animals are outdoors and moving."),
        ("A woman is reading a book.", "Someone is cooking in a kitchen."),
        ("The cat sits on the mat.", "A feline is resting on a rug."),
        ("He is driving a car.", "A person is operating a vehicle."),
        ("Kids are playing in the park.", "Children are outdoors having fun."),
        ("A plane is taking off.", "An aircraft is landing."),
        ("People are eating at a restaurant.", "Customers are dining out."),
    ]


def make_batches(tokenizer, texts, batch_size: int, seq_len: int, is_pair: bool):
    # Yield dicts of tokenized inputs
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        if is_pair:
            s1 = [a for a, b in batch]
            s2 = [b for a, b in batch]
            enc = tokenizer(
                s1, s2,
                padding=True,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt"
            )
        else:
            s1 = [a for a in batch]
            enc = tokenizer(
                s1,
                padding=True,
                truncation=True,
                max_length=seq_len,
                return_tensors="pt"
            )
        yield enc


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--task", type=str, default="stsb", choices=["stsb", "single"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--max_batches", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default="stats_out")
    ap.add_argument("--include_embeddings", action="store_true")
    ap.add_argument("--only_linears", action="store_true")
    ap.add_argument("--collect_activations", action="store_true")
    ap.add_argument("--reservoir_size", type=int, default=200_000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Build model similar to your snippet
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1 if args.task == "stsb" else 2,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)} if args.device.startswith("cuda") else None,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.eval()

    # If device_map isn't used (e.g., cpu), move
    if args.device and "cuda" not in str(next(model.parameters()).device):
        model.to(args.device)

    # 1) Weight stats
    wdf = collect_weight_stats(
        model,
        include_embeddings=args.include_embeddings,
        only_linears=args.only_linears,
    )
    w_path = os.path.join(args.out_dir, "weight_stats.csv")
    wdf.to_csv(w_path, index=False)
    print(f"[OK] wrote {w_path} ({len(wdf)} rows)")

    # 2) Activation stats (optional)
    if args.collect_activations:
        module_filter = (nn.Linear,)  # you can extend: (nn.Linear, nn.LayerNorm)
        collector = ActivationStatsCollector(
            model,
            module_filter=module_filter,
            record_input=True,
            record_output=True,
            reservoir_size=args.reservoir_size,
        )
        collector.install()

        is_pair = (args.task == "stsb")
        texts = default_text_pairs() if is_pair else [t[0] for t in default_text_pairs()]
        # Repeat to have enough batches
        texts = (texts * max(1, args.max_batches * args.batch_size // max(1, len(texts)) + 1))[:args.max_batches * args.batch_size]

        with torch.no_grad():
            bcount = 0
            for enc in make_batches(tokenizer, texts, args.batch_size, args.seq_len, is_pair=is_pair):
                enc = {k: v.to(args.device) for k, v in enc.items()}
                _ = model(**enc)
                bcount += 1
                if bcount >= args.max_batches:
                    break

        collector.remove()
        adf = collector.to_dataframe()
        a_path = os.path.join(args.out_dir, "act_stats.csv")
        adf.to_csv(a_path, index=False)
        print(f"[OK] wrote {a_path} ({len(adf)} rows)")

    print("[DONE]")


if __name__ == "__main__":
    main()
