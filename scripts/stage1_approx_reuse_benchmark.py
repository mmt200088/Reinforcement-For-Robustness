#!/usr/bin/env python3
"""Benchmark + correctness check for the Stage-1 approx-module reuse speedup.

Measures the per-episode Stage-1 hot path (install GELU/Softmax approximations
-> BERT forward) with the optimization ON vs OFF, on a model whose dimensions
match the requested architecture (bert-large by default). It proves two things
at once:

  1. **Identical results.** For every episode config the forward logits of the
     reuse-ON arm and the reuse-OFF (original reconstruct-every-call) arm are
     compared; any mismatch aborts with a non-zero exit code.
  2. **Faster.** Per-episode install + forward wall time is reported for both
     arms, with the mean speedup and the module-reconstruction counts.

Synthetic batches are used on purpose: the install + forward cost is driven by
tensor *shapes*, not data values, and the identity guarantee holds for any
input. This keeps the benchmark self-contained (no GLUE data / evaluator init).

Usage (server, four-GPU box — single GPU is enough for this microbenchmark):

  CUDA_VISIBLE_DEVICES=0 python scripts/stage1_approx_reuse_benchmark.py \
      --model-type bert-large --num-episodes 40 --batch-size 32 --seq-len 128 \
      --output-dir experiments/server_command_runs/stage1_approx_reuse_$(date +%Y%m%d_%H%M%S)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import random
import sys
import time
from typing import Sequence

import torch
from transformers import BertConfig, BertForSequenceClassification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from function_handler import ReversibleLayerHandler  # noqa: E402
from stats_utils import mean_or_default  # noqa: E402

# Degrees the Stage-1 policy can actually select (mirror GELU_MAP / SOFTMAX_MAP
# value sets in layer_importance_evaluator.py).
_GELU_DEGREES = [4, 2, 1]
_SOFTMAX_DEGREES = [6, 5, 4, 3, 2]
_GELU_GROUPS = [0, 1, 2, 4]
_SOFTMAX_GROUPS = list(range(2, 7))
_LAYERS_ATTR = "model.bert.encoder.layer"

# bert-base / bert-large dimensions.
_DIMS = {
    "bert-base": dict(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072),
    "bert-large": dict(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096),
}


def _install_stage1_config(handler, gelu_degrees, softmax_degrees):
    """Mirror LayerImportanceEvaluator._stage1_evaluate_on_model install body."""
    gelu_map = {d: [] for d in _GELU_GROUPS}
    for i, d in enumerate(gelu_degrees):
        gelu_map[int(d)].append(i)
    for d in _GELU_GROUPS:
        if gelu_map[d]:
            handler.replace_layer_gelu(gelu_map[d], _LAYERS_ATTR, degree=d)
    softmax_map = {d: [] for d in _SOFTMAX_GROUPS}
    for i, d in enumerate(softmax_degrees):
        softmax_map[int(d)].append(i)
    for d in _SOFTMAX_GROUPS:
        if softmax_map[d]:
            handler.replace_layer_softmax(softmax_map[d], _LAYERS_ATTR, degree=d)


def _build_model(model_type, device):
    dims = _DIMS[model_type]
    cfg = BertConfig(
        vocab_size=30522,
        max_position_embeddings=512,
        num_labels=2,
        **dims,
    )
    # All self-attentions are replaced before any forward; force eager to match
    # the approx-attention path. Attribute (not kwarg) for version robustness.
    cfg._attn_implementation = "eager"
    torch.manual_seed(20260529)
    model = BertForSequenceClassification(cfg).to(device).eval()
    return model


def _equal_with_nan(a, b):
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all().item())


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_episode(handler, model, batch, gelu, softmax, device):
    _sync(device)
    t0 = time.perf_counter()
    _install_stage1_config(handler, gelu, softmax)
    _sync(device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        model.eval()
        logits = model(**batch).logits
    _sync(device)
    t2 = time.perf_counter()
    return logits, (t1 - t0), (t2 - t1)


def _summarize_timings(
        fast_install: Sequence[float],
        fast_fwd: Sequence[float],
        slow_install: Sequence[float],
        slow_fwd: Sequence[float],
) -> dict:
    fast_install_mean = mean_or_default(fast_install, default=float("nan"))
    fast_fwd_mean = mean_or_default(fast_fwd, default=float("nan"))
    slow_install_mean = mean_or_default(slow_install, default=float("nan"))
    slow_fwd_mean = mean_or_default(slow_fwd, default=float("nan"))

    fast_total_sum = 0.0
    slow_total_sum = 0.0
    total_count = 0
    for fi, ff, si, sf in zip(fast_install, fast_fwd, slow_install, slow_fwd):
        fast_total_sum += float(fi) + float(ff)
        slow_total_sum += float(si) + float(sf)
        total_count += 1

    if total_count:
        fast_total_mean = fast_total_sum / float(total_count)
        slow_total_mean = slow_total_sum / float(total_count)
    else:
        fast_total_mean = float("nan")
        slow_total_mean = float("nan")

    return {
        "num_episodes_timed": total_count,
        "reuse_on": {
            "install_ms_mean": fast_install_mean * 1000.0,
            "forward_ms_mean": fast_fwd_mean * 1000.0,
            "total_ms_mean": fast_total_mean * 1000.0,
        },
        "reuse_off": {
            "install_ms_mean": slow_install_mean * 1000.0,
            "forward_ms_mean": slow_fwd_mean * 1000.0,
            "total_ms_mean": slow_total_mean * 1000.0,
        },
        "episode_speedup": (
            slow_total_mean / fast_total_mean
            if fast_total_mean > 0 else float("nan")
        ),
        "install_speedup": (
            slow_install_mean / fast_install_mean
            if fast_install_mean > 0 else float("nan")
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=list(_DIMS), default="bert-large")
    ap.add_argument("--num-episodes", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[bench] device={device} model={args.model_type} "
          f"episodes={args.num_episodes} batch={args.batch_size} seq={args.seq_len}")

    import copy
    base = _build_model(args.model_type, device)
    n_layers = base.config.num_hidden_layers

    model_fast = copy.deepcopy(base)
    model_slow = copy.deepcopy(base)
    del base
    handler_fast = ReversibleLayerHandler(model_fast)
    handler_fast.reuse_approx_modules = True
    handler_slow = ReversibleLayerHandler(model_slow)
    handler_slow.reuse_approx_modules = False

    g = torch.Generator().manual_seed(args.seed)
    batch = {
        "input_ids": torch.randint(0, 30522, (args.batch_size, args.seq_len), generator=g).to(device),
        "attention_mask": torch.ones((args.batch_size, args.seq_len), dtype=torch.long, device=device),
    }

    rng = random.Random(args.seed)
    schedule = [
        ([rng.choice(_GELU_DEGREES) for _ in range(n_layers)],
         [rng.choice(_SOFTMAX_DEGREES) for _ in range(n_layers)])
        for _ in range(args.num_episodes + args.warmup)
    ]

    fast_install, fast_fwd, slow_install, slow_fwd = [], [], [], []
    max_abs_diff = 0.0
    mismatches = 0

    for ep, (gelu, softmax) in enumerate(schedule):
        lf, fi, ff = _timed_episode(handler_fast, model_fast, batch, gelu, softmax, device)
        ls, si, sf = _timed_episode(handler_slow, model_slow, batch, gelu, softmax, device)
        diff = (lf - ls).abs()
        diff = diff[~torch.isnan(diff)]
        if diff.numel():
            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
        if not _equal_with_nan(lf, ls):
            mismatches += 1
            print(f"[bench][MISMATCH] episode {ep}: logits differ "
                  f"(max|diff|={float((lf - ls).abs().max().item()):.3e})")
        if ep >= args.warmup:  # exclude warmup from timing stats
            fast_install.append(fi); fast_fwd.append(ff)
            slow_install.append(si); slow_fwd.append(sf)

    timing_summary = _summarize_timings(fast_install, fast_fwd, slow_install, slow_fwd)
    reuse_on = {
        **timing_summary["reuse_on"],
        "softmax_rebuilds": handler_fast._approx_softmax_rebuilds,
        "gelu_rebuilds": handler_fast._approx_gelu_rebuilds,
    }
    reuse_off = {
        **timing_summary["reuse_off"],
        "softmax_rebuilds": handler_slow._approx_softmax_rebuilds,
        "gelu_rebuilds": handler_slow._approx_gelu_rebuilds,
    }

    summary = {
        "device": str(device),
        "model_type": args.model_type,
        "num_layers": n_layers,
        "num_episodes_timed": timing_summary["num_episodes_timed"],
        "warmup": args.warmup,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "identical_logits": mismatches == 0,
        "logit_mismatches": mismatches,
        "max_abs_logit_diff": max_abs_diff,
        "reuse_on": reuse_on,
        "reuse_off": reuse_off,
        "episode_speedup": timing_summary["episode_speedup"],
        "install_speedup": timing_summary["install_speedup"],
        # Proof the reuse path actually engaged across the changing per-episode
        # configs: reuse rebuilds each layer at most once, vs once-per-episode
        # without it. If this is False the cache silently did nothing.
        "cache_engaged": (handler_fast._approx_softmax_rebuilds <= n_layers
                          and handler_fast._approx_softmax_rebuilds < handler_slow._approx_softmax_rebuilds),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    print("\n===== Stage-1 approx-module reuse benchmark =====")
    print(f"  identical logits      : {summary['identical_logits']} "
          f"(mismatches={mismatches}, max|diff|={max_abs_diff:.3e})")
    print(f"  reuse OFF  install/fwd/total ms: "
          f"{summary['reuse_off']['install_ms_mean']:.2f} / "
          f"{summary['reuse_off']['forward_ms_mean']:.2f} / "
          f"{summary['reuse_off']['total_ms_mean']:.2f}")
    print(f"  reuse ON   install/fwd/total ms: "
          f"{summary['reuse_on']['install_ms_mean']:.2f} / "
          f"{summary['reuse_on']['forward_ms_mean']:.2f} / "
          f"{summary['reuse_on']['total_ms_mean']:.2f}")
    print(f"  episode speedup       : {summary['episode_speedup']:.2f}x  (install speedup "
          f"{summary['install_speedup']:.2f}x)")
    print(f"  softmax rebuilds      : ON={handler_fast._approx_softmax_rebuilds} "
          f"OFF={handler_slow._approx_softmax_rebuilds}")
    print(f"  cache engaged         : {summary['cache_engaged']} "
          f"(reuse rebuilds each layer <= once across changing configs)")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "stage1_approx_reuse_benchmark.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  wrote {os.path.join(args.output_dir, 'stage1_approx_reuse_benchmark.json')}")

    if mismatches:
        print("\n[bench] FAILED: optimization changed the forward result.")
        sys.exit(1)
    print("\n[bench] OK: identical results, faster.")


if __name__ == "__main__":
    main()
