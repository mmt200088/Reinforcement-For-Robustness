#!/usr/bin/env python
"""
Per-computation distribution analysis with *function substitution* for
softmax/GELU, referencing the polynomial/Taylor approximations defined in
``function_handler.py``.

与 ``analyze_all_distribution_new.py`` 的区别：

  1. 先做函数替代 (function substitution)，再做 magnitude 统计。
     - Softmax 里的 ``exp`` 用 Taylor-iterated-squaring 近似：
           exp_approx(x) = (1 + x / 2**d) ** (2**d),   x < lower_bound => 0
     - GELU 用 Bumblebee 分段多项式近似 (degree ∈ {0..4})，系数见
       ``function_handler.py`` 的 ``GELU_COEEF``。

  2. 每一个计算之后都统计 —— 在近似路径里加入额外探针：

     Softmax 近似链:
         attn_scores  -->  softmax_x_shifted  (x − max x)
                      -->  softmax_exp_raw    ((1 + x/2^d)^(2^d), 无 mask)
                      -->  softmax_exp_out    (x < lower_bound 被 clip 为 0)
                      -->  softmax_sum_exp    (Σ exp, 分母)
                      -->  attn_probs         (exp_out / sum_exp, 原有探针)

     GELU 近似链:
         gelu_input   -->  gelu_poly_neg      (负分支多项式 y = poly(x, neg))
                      -->  gelu_poly_pos      (正分支多项式 y = poly(x, pos))
                      -->  gelu_output        (piecewise 组合, 原有探针)

  3. Softmax / GELU 的近似 degree 按层可配置，通过 ``--approx_config`` 指向
     一个 JSON 文件。未配置的层回退到 default；若 default 为 null 则该层
     使用原始函数 (不做近似)。

Config schema (JSON)
--------------------
::

    {
      "softmax": {
        "default_degree": 4,                 // 1..6  (Exp_bound keys); null => off
        "per_layer":      { "0": 2, "11": 3 } // 覆盖 default
      },
      "gelu": {
        "default_degree": 2,                 // 0..4  (GELU_COEEF keys); null => off
        "per_layer":      { "0": 0, "5": 4 }
      }
    }

Usage
-----
::

    cd /var/tmp/root-home/Reinforcement-For-Robustness/Model_analysis

    mkdir -p all_analysis_approx/wnli
    
    python analyze_all_distribution_approx.py \
    --config configs/approx_per_dataset.json \
    --stage stage1 \
    --tasks wnli \
    --output_dir all_analysis_approx/wnli

    tail -f all_analysis_approx/wnli/wnli_run.log
"""

import argparse
import copy
import csv
import json
import os
import queue
import sys
import threading

# ``function_handler.py`` lives in the parent ``Reinforcement-For-Robustness/``
# directory, while this script sits under ``Model_analysis/``. Make the parent
# directory importable regardless of where the user launches Python from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_PARENT_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ---- Reuse building blocks from the plain analyzer ------------------------
import analyze_all_distribution_new as base
from analyze_all_distribution_new import (
    GLOBAL_PROBES,
    LayerWiseCollector,
    MAG_EDGES,
    MAG_NBINS,
    TASK_REGISTRY,
    _ActWrapper,
    _accum_max_linear,
    _enqueue,
    _make_hook,
    _make_linear_hook,
    _make_ln_internals_pre_hook,
    _make_pre_hook,
    _mag_bin_labels,
    _ORIG_MATMUL,
    _prepare_bert_data,
    _prepare_gpt2_data,
    _stats_worker,
    plot_heatmap,
    plot_magnitude_heatmap,
    plot_magnitude_per_layer,
    plot_outlier_overview,
    plot_overview,
    plot_probe_histograms,
    plot_probe_magnitude_bar,
    remove_hooks,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# ---- Approximation constants/utilities (reuse; do NOT modify function_handler) ----
from function_handler import Exp_bound, GELU_COEEF, polynomial


# ============================================================================
# Extend base module's probe tables with the approximation-internal probes
# ============================================================================

#
# Probe inventory for the approximation internals
# ----------------------------------------------
# Softmax exp chain (iterated-squaring Taylor):
#   softmax_x_shifted      x - x.max()
#   softmax_exp_scaled     x_shifted / 2^n                  (scaled input)
#   softmax_exp_base       1 + x_shifted / 2^n              (Taylor base)
#   softmax_exp_sq1..sq6   base^(2^i) after each squaring   (layer fills up to i=n)
#   softmax_exp_raw        final exp approx, = sq{n}        (kept for compatibility)
#   softmax_exp_out        exp_raw after lower_bound clip
#   softmax_sum_exp        Σ exp_out over last dim
#
# GELU polynomial (piecewise, Bumblebee):
#   gelu_x2, gelu_x3, gelu_x4   pure powers of gelu_input (x itself = gelu_input)
#   gelu_neg_t0..t4             per-term c_i * x^i  for the negative branch
#   gelu_pos_t0..t4             per-term c_i * x^i  for the positive branch
#   gelu_poly_neg / _pos        branch polynomial = Σ_i c_i x^i (kept)
#
# For a layer whose degree is d, only the first (d+1) term-probes and the
# first d squaring-probes are populated; the rest stay empty.
_SOFTMAX_EXP_SQ_MAX = 6    # Exp_bound supports degrees 1..6
_GELU_TERM_MAX     = 4     # GELU_COEEF supports degrees 0..4 → up to c4*x^4

APPROX_PROBES = [
    "softmax_x_shifted",
    "softmax_exp_scaled",
    "softmax_exp_base",
    *[f"softmax_exp_sq{i}" for i in range(1, _SOFTMAX_EXP_SQ_MAX + 1)],
    "softmax_exp_raw",
    "softmax_exp_out",
    "softmax_sum_exp",
    *[f"gelu_x{i}" for i in range(2, _GELU_TERM_MAX + 1)],
    *[f"gelu_neg_t{i}" for i in range(0, _GELU_TERM_MAX + 1)],
    *[f"gelu_pos_t{i}" for i in range(0, _GELU_TERM_MAX + 1)],
    "gelu_poly_neg",
    "gelu_poly_pos",
]

_PROBE_DISPLAY_ADD = {
    "softmax_x_shifted":   "Softmax x−max",
    "softmax_exp_scaled":  "x/2^n",
    "softmax_exp_base":    "1 + x/2^n",
    **{f"softmax_exp_sq{i}": f"(1+x/2^n)^{2**i}"
       for i in range(1, _SOFTMAX_EXP_SQ_MAX + 1)},
    "softmax_exp_raw":     "Approx exp (pre-mask)",
    "softmax_exp_out":     "Approx exp (masked)",
    "softmax_sum_exp":     "Σ exp (denominator)",
    **{f"gelu_x{i}": f"x^{i}" for i in range(2, _GELU_TERM_MAX + 1)},
    **{f"gelu_neg_t{i}": f"GELU neg  c{i}·x^{i}"
       for i in range(0, _GELU_TERM_MAX + 1)},
    **{f"gelu_pos_t{i}": f"GELU pos  c{i}·x^{i}"
       for i in range(0, _GELU_TERM_MAX + 1)},
    "gelu_poly_neg":       "GELU poly (neg branch)",
    "gelu_poly_pos":       "GELU poly (pos branch)",
}

# Magnitude histograms (log-binned) give the real picture; these linear-bin
# ranges are just for the readable "distribution" PNGs.  Values outside the
# range are clipped to the edge bins.
_PROBE_HIST_RANGE_ADD = {
    "softmax_x_shifted":  (-50.0,   1.0, 300),
    "softmax_exp_scaled": (-15.0,   1.0, 300),
    "softmax_exp_base":   (-15.0,   2.0, 300),
    "softmax_exp_sq1":    (0.0,   100.0, 300),
    "softmax_exp_sq2":    (0.0,  1000.0, 300),
    "softmax_exp_sq3":    (0.0, 10000.0, 300),
    "softmax_exp_sq4":    (0.0, 10000.0, 300),
    "softmax_exp_sq5":    (0.0, 10000.0, 300),
    "softmax_exp_sq6":    (0.0, 10000.0, 300),
    "softmax_exp_raw":    (0.0,  1000.0, 300),
    "softmax_exp_out":    (0.0,     1.5, 300),
    "softmax_sum_exp":    (0.0,    50.0, 300),
    "gelu_x2":            (0.0,   250.0, 300),
    "gelu_x3":            (-1500.0, 1500.0, 300),
    "gelu_x4":            (0.0, 20000.0, 300),
    "gelu_neg_t0":        (-1.0,     1.0, 200),
    "gelu_pos_t0":        (-1.0,     1.0, 200),
    "gelu_neg_t1":        (-20.0,   20.0, 300),
    "gelu_pos_t1":        (-20.0,   20.0, 300),
    "gelu_neg_t2":        (-150.0, 150.0, 300),
    "gelu_pos_t2":        (-150.0, 150.0, 300),
    "gelu_neg_t3":        (-500.0, 500.0, 300),
    "gelu_pos_t3":        (-500.0, 500.0, 300),
    "gelu_neg_t4":        (-500.0, 500.0, 300),
    "gelu_pos_t4":        (-500.0, 500.0, 300),
    "gelu_poly_neg":      (-2.0,     2.0, 300),
    "gelu_poly_pos":      (-2.0,    15.0, 300),
}

_PROBE_COLORS_ADD = {
    "softmax_x_shifted":  "#FFD700",
    "softmax_exp_scaled": "#FFE4B5",
    "softmax_exp_base":   "#FFDAB9",
    "softmax_exp_sq1":    "#FFA07A",
    "softmax_exp_sq2":    "#FF8C69",
    "softmax_exp_sq3":    "#FF7F50",
    "softmax_exp_sq4":    "#FF6347",
    "softmax_exp_sq5":    "#FF4500",
    "softmax_exp_sq6":    "#E9967A",
    "softmax_exp_raw":    "#FF6347",
    "softmax_exp_out":    "#DC143C",
    "softmax_sum_exp":    "#B22222",
    "gelu_x2":            "#B0E0E6",
    "gelu_x3":            "#87CEEB",
    "gelu_x4":            "#6495ED",
    "gelu_neg_t0":        "#E6E6FA",
    "gelu_neg_t1":        "#D8BFD8",
    "gelu_neg_t2":        "#DDA0DD",
    "gelu_neg_t3":        "#BA55D3",
    "gelu_neg_t4":        "#9370DB",
    "gelu_pos_t0":        "#F0FFF0",
    "gelu_pos_t1":        "#98FB98",
    "gelu_pos_t2":        "#90EE90",
    "gelu_pos_t3":        "#3CB371",
    "gelu_pos_t4":        "#2E8B57",
    "gelu_poly_neg":      "#4169E1",
    "gelu_poly_pos":      "#6A5ACD",
}

for _p in APPROX_PROBES:
    if _p not in base.PROBE_POINTS:
        base.PROBE_POINTS.append(_p)
base.PROBE_DISPLAY.update(_PROBE_DISPLAY_ADD)
base.PROBE_HIST_RANGE.update(_PROBE_HIST_RANGE_ADD)
base.PROBE_COLORS.update(_PROBE_COLORS_ADD)

PROBE_POINTS = base.PROBE_POINTS


# ============================================================================
# Approximation config handling
# ============================================================================

DEFAULT_APPROX_CONFIG = {
    "softmax": {"default_degree": 4, "per_layer": {}},
    "gelu":    {"default_degree": 2, "per_layer": {}},
}


def _is_legacy_flat_format(cfg):
    """Legacy schema: top-level 'softmax'/'gelu' dicts with default_degree/per_layer."""
    for k in ("softmax", "gelu"):
        if (
            k in cfg and isinstance(cfg[k], dict)
            and ("default_degree" in cfg[k] or "per_layer" in cfg[k])
        ):
            return True
    return False


def _array_to_per_layer(arr, section_name):
    if not isinstance(arr, list):
        raise ValueError(
            f"[approx_config] '{section_name}' must be a list of per-layer degrees, "
            f"got {type(arr).__name__}"
        )
    return {str(i): (None if v is None else int(v)) for i, v in enumerate(arr)}


def load_approx_config(path, task_name=None, stage="stage1"):
    """Load approximation config.

    Supports two formats:

    1. Dataset-keyed (preferred; written as arrays)::

        {
          "wnli": {
            "stage1": {
              "gelu":    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
              "softmax": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            }
          },
          "sst2": { "stage1": { ... } }
        }

       When this format is used, ``task_name`` must be supplied; the array
       index ``i`` is the transformer-layer index.  Use ``null`` in an array
       slot to skip approximation for that single layer.

    2. Legacy flat (still supported)::

        {
          "softmax": {"default_degree": 4, "per_layer": {"0": 2}},
          "gelu":    {"default_degree": 2, "per_layer": {}}
        }
    """
    if path is None:
        return copy.deepcopy(DEFAULT_APPROX_CONFIG)
    with open(path, "r") as f:
        cfg = json.load(f)

    if _is_legacy_flat_format(cfg):
        out = copy.deepcopy(DEFAULT_APPROX_CONFIG)
        for section in ("softmax", "gelu"):
            if section in cfg:
                if "default_degree" in cfg[section]:
                    out[section]["default_degree"] = cfg[section]["default_degree"]
                if "per_layer" in cfg[section]:
                    out[section]["per_layer"] = cfg[section]["per_layer"]
        return out

    # Dataset-keyed format
    datasets = [k for k in cfg.keys() if not k.startswith("_")]
    if task_name is None:
        raise ValueError(
            f"[approx_config] Dataset-keyed config requires a task name. "
            f"Available datasets in '{path}': {datasets}"
        )
    if task_name not in cfg:
        raise ValueError(
            f"[approx_config] Task '{task_name}' not found in '{path}'. "
            f"Available: {datasets}"
        )
    task_cfg = cfg[task_name]
    if stage not in task_cfg:
        raise ValueError(
            f"[approx_config] Stage '{stage}' not found for task '{task_name}'. "
            f"Available: {list(task_cfg.keys())}"
        )
    stage_cfg = task_cfg[stage]

    out = {
        "softmax": {"default_degree": None, "per_layer": {}},
        "gelu":    {"default_degree": None, "per_layer": {}},
    }
    for section in ("softmax", "gelu"):
        if section in stage_cfg:
            out[section]["per_layer"] = _array_to_per_layer(stage_cfg[section], section)
    return out


def _resolve_degree(cfg_section, layer_idx):
    """Return ``int`` degree for this layer, or ``None`` if this layer is
    configured to skip approximation (use the real op)."""
    per_layer = cfg_section.get("per_layer") or {}
    if str(layer_idx) in per_layer:
        val = per_layer[str(layer_idx)]
    elif layer_idx in per_layer:
        val = per_layer[layer_idx]
    else:
        val = cfg_section.get("default_degree")
    if val is None:
        return None
    return int(val)


def resolve_softmax_degree(cfg, layer_idx):
    d = _resolve_degree(cfg["softmax"], layer_idx)
    if d is None:
        return None
    if d not in Exp_bound:
        raise ValueError(
            f"[approx_config] softmax degree {d} not in Exp_bound={sorted(Exp_bound)}"
        )
    return d


def _enqueue_mask_filtered(probe, layer, tensor, q, accum_ma=None,
                           threshold=1e30):
    """Enqueue stats after dropping BERT's padding-mask sentinel values.

    HuggingFace BERT builds ``extended_attention_mask`` as
    ``(1 - mask) * torch.finfo(dtype).min`` (-3.4028e+38 for fp32) and adds it
    to ``QK^T / sqrt(d_k)`` before softmax.  This sentinel then propagates
    unchanged into ``softmax_x_shifted`` (= ``x - x.max()``) and explodes to
    ``inf`` when raised to the 2^d power in ``softmax_exp_raw``.  Those
    entries are semantically "masked out" (the approximation clips them to 0
    via ``lower_bound``), but they still dominate the min/mean/std/max
    aggregates of the pre-mask probes and make the output unreadable.

    This helper removes any value with ``|x| > threshold`` or non-finite
    (inf / NaN) before handing the tensor to the plain ``_enqueue``.
    """
    t = tensor.reshape(-1)
    keep = torch.isfinite(t) & (t.abs() < threshold)
    if not torch.any(keep):
        return
    _enqueue(probe, layer, t[keep], q, accum_ma=accum_ma)


def resolve_gelu_degree(cfg, layer_idx):
    d = _resolve_degree(cfg["gelu"], layer_idx)
    if d is None:
        return None
    if d not in GELU_COEEF:
        raise ValueError(
            f"[approx_config] gelu degree {d} not in GELU_COEEF={sorted(GELU_COEEF)}"
        )
    return d


# ============================================================================
# Approximation primitives (mirror function_handler.py, plus per-step stats)
# ============================================================================


def _approx_exp(x, degree):
    """Exp approximation via iterated-squaring Taylor, identical to
    ``BertSelfAttentionWithAproximation.approximation_exponential``.

    Kept as a reference single-call variant.  The instrumented softmax below
    inlines the iterative form so that every intermediate can be probed.
    """
    return torch.pow(1 + x / (2 ** degree), 2 ** degree)


def _make_approx_softmax_with_stats(degree, lower_bound, layer_idx, q):
    """Return a softmax function (signature ``fn(x, dim=-1)``) that is
    numerically equivalent to ``BertSelfAttentionWithAproximation.approximation_softmax``
    but exposes every step of the iterated-squaring exp approximation as a
    probe:

        x_shifted       = x − max(x)                       → softmax_x_shifted
        scaled          = x_shifted / 2^degree             → softmax_exp_scaled
        base            = 1 + scaled                       → softmax_exp_base
        sq_i            = base^(2^i),  i = 1 .. degree     → softmax_exp_sq{i}
        exp_raw         = sq_{degree}                      → softmax_exp_raw
        exp_out         = exp_raw · 1[x_shifted ≥ lb]      → softmax_exp_out
        sum_exp         = Σ_j exp_out_j                    → softmax_sum_exp

    All derivatives of ``x_shifted`` go through ``_enqueue_mask_filtered`` so
    the padding-mask sentinel (torch.finfo.min ≈ −3.4e38) does not poison the
    distribution statistics.
    """

    two_d = 2 ** degree

    def _softmax_fn(x, dim=-1):
        # Assumes dim == -1 (every HF attention uses -1). We keep the parameter
        # for signature compatibility with ``F.softmax``.
        x_max = x.max(dim=-1, keepdim=True)[0]
        x_shifted = x - x_max + 1e-9
        _enqueue_mask_filtered("softmax_x_shifted", layer_idx, x_shifted.detach(), q)

        scaled = x_shifted / two_d
        _enqueue_mask_filtered("softmax_exp_scaled", layer_idx, scaled.detach(), q)

        base = 1.0 + scaled
        _enqueue_mask_filtered("softmax_exp_base", layer_idx, base.detach(), q)

        current = base
        for i in range(1, degree + 1):
            current = current * current
            _enqueue_mask_filtered(
                f"softmax_exp_sq{i}", layer_idx, current.detach(), q
            )

        exp_raw = current  # = base^(2^degree)
        _enqueue_mask_filtered("softmax_exp_raw", layer_idx, exp_raw.detach(), q)

        exp_out = torch.where(
            x_shifted < lower_bound, torch.zeros_like(x_shifted), exp_raw
        )
        _enqueue("softmax_exp_out", layer_idx, exp_out.detach(), q)

        sum_exp = torch.sum(exp_out, dim=-1, keepdim=True) + 1e-9
        _enqueue("softmax_sum_exp", layer_idx, sum_exp.detach().squeeze(-1), q)

        return exp_out / sum_exp

    return _softmax_fn


class StatsPolynomialGELU(nn.Module):
    """Bumblebee piecewise polynomial GELU (matches ``PolynomialGELU``) with
    per-term statistic capture.

    For each branch (negative / positive) the polynomial
    ``Σ_i c_i · x^i`` is expanded term by term and every intermediate is
    probed:

        x^i               → ``gelu_x{i}``    (shared, i = 2..degree)
        c_i · x^i (neg)   → ``gelu_neg_t{i}``  (i = 0..degree)
        c_i · x^i (pos)   → ``gelu_pos_t{i}``  (i = 0..degree)
        Σ_i c_i·x^i (neg) → ``gelu_poly_neg``
        Σ_i c_i·x^i (pos) → ``gelu_poly_pos``

    Results on the full ``x`` tensor are recorded; the branch-selection mask
    is applied only for the returned output, exactly as in
    ``function_handler.PolynomialGELU``.
    ``gelu_input`` / ``gelu_output`` are captured by the surrounding
    ``_ActWrapper``.
    """

    def __init__(self, degree, layer_idx, q):
        super().__init__()
        self.coeff = GELU_COEEF[degree]
        self.degree = degree
        self.layer_idx = layer_idx
        self.q = q

    def _branch_poly(self, x, sign, branch_name, powers):
        """Evaluate ``Σ_i c_i · x^i`` for the requested branch, emitting a
        probe for each term.

        ``powers`` is a list with ``powers[i] = x^i`` for ``i = 1..degree``
        (``powers[0]`` is unused since ``c_0`` multiplies the implicit ``x^0``).
        """
        coeffs = self.coeff[sign]
        result = None
        for i, c in enumerate(coeffs):
            c_val = float(c)
            if i == 0:
                term = torch.full_like(x, c_val)
            else:
                term = c_val * powers[i]
            _enqueue(f"gelu_{branch_name}_t{i}", self.layer_idx, term.detach(), self.q)
            result = term if result is None else result + term
        return result

    def _compute_powers(self, x):
        """Return ``[None, x, x^2, x^3, ..., x^degree]`` and enqueue
        ``gelu_x{i}`` for ``i >= 2``.  ``x^1`` itself is already captured by
        the ``_ActWrapper`` as ``gelu_input`` and is not re-emitted."""
        powers = [None, x]
        for i in range(2, self.degree + 1):
            xi = x.pow(i)
            powers.append(xi)
            _enqueue(f"gelu_x{i}", self.layer_idx, xi.detach(), self.q)
        return powers

    def forward(self, x):
        if self.degree == 0:
            # degree-0 collapses to a single polynomial across the entire
            # input, identical to PolynomialGELU(degree=0).
            powers = [None, x]
            y = self._branch_poly(x, 1, "neg", powers)
            _enqueue("gelu_poly_neg", self.layer_idx, y.detach(), self.q)
            return y

        powers = self._compute_powers(x)
        y1 = self._branch_poly(x, 1, "neg", powers)
        y2 = self._branch_poly(x, 0, "pos", powers)
        _enqueue("gelu_poly_neg", self.layer_idx, y1.detach(), self.q)
        _enqueue("gelu_poly_pos", self.layer_idx, y2.detach(), self.q)

        mask_low  = x < -2.7
        mask_neg  = (x >= -2.7) & (x < 0)
        mask_pos  = (x >= 0) & (x <= 2.7)
        mask_high = x > 2.7

        out = torch.where(mask_low, torch.zeros_like(x), torch.zeros_like(x))
        out = torch.where(mask_neg, y1, out)
        out = torch.where(mask_pos, y2, out)
        out = torch.where(mask_high, x, out)
        return out


# ============================================================================
# Attention wrapper with pluggable softmax
# ============================================================================


def _make_attn_wrapper_with_softmax(fwd, li, q, softmax_fn):
    """Like ``base._make_attn_wrapper`` but uses ``softmax_fn`` (may be the
    approximation or ``None`` meaning "use real softmax").

    Captures: ``qkt_raw``, ``attn_scores``, ``attn_probs``, ``attn_context``
    (identical to the base helper).  When ``softmax_fn`` is provided, the four
    additional approximation-internal probes are emitted from inside it.
    """

    def _wrapped(*args, **kwargs):
        captured = {}
        mm_count = [0]
        _real_mm = torch.matmul
        _real_sm = torch.nn.functional.softmax

        def _cap_mm(a, b):
            result = _real_mm(a, b)
            mm_count[0] += 1
            if mm_count[0] == 1:
                captured["qkt"] = result.detach()
                captured["qkt_acc"] = _ORIG_MATMUL(
                    a.detach().abs(), b.detach().abs()
                ).max().item()
            elif mm_count[0] == 2:
                captured["context"] = result.detach()
                captured["ctx_acc"] = _ORIG_MATMUL(
                    a.detach().abs(), b.detach().abs()
                ).max().item()
            return result

        def _cap_sm(inp, dim=None, **kw):
            captured["scores"] = inp.detach()
            if softmax_fn is None:
                result = _real_sm(inp, dim=dim, **kw)
            else:
                result = softmax_fn(inp, dim=dim)
            captured["probs"] = result.detach()
            return result

        torch.matmul = _cap_mm
        torch.nn.functional.softmax = _cap_sm
        try:
            outputs = fwd(*args, **kwargs)
        finally:
            torch.matmul = _real_mm
            torch.nn.functional.softmax = _real_sm

        if "qkt" in captured:
            _enqueue("qkt_raw", li, captured["qkt"], q,
                     accum_ma=captured.get("qkt_acc"))
        if "scores" in captured:
            _enqueue_mask_filtered("attn_scores", li, captured["scores"], q)
        if "probs" in captured:
            _enqueue("attn_probs", li, captured["probs"], q)
        if "context" in captured:
            _enqueue("attn_context", li, captured["context"], q,
                     accum_ma=captured.get("ctx_acc"))
        else:
            _enqueue("attn_context", li, outputs[0].detach(), q)
        return outputs

    return _wrapped


# ============================================================================
# BERT / GPT-2 hook installers with approximation
# ============================================================================


def _install_bert_hooks_approx(model, q, approx_cfg):
    handles = []
    restore = []

    # Global: input_ids
    def _ids_hook(_mod, inp):
        if inp[0] is not None:
            _enqueue("input_ids", 0, inp[0].detach().float(), q)

    handles.append(
        model.bert.embeddings.word_embeddings.register_forward_pre_hook(_ids_hook)
    )
    handles.append(
        model.bert.embeddings.register_forward_hook(
            lambda _m, _i, o: _enqueue("after_embed", 0, o.detach(), q)
        )
    )

    for i, layer in enumerate(model.bert.encoder.layer):
        sa = layer.attention.self

        # Q / K / V
        for probe, mod in [
            ("query_proj", sa.query),
            ("key_proj", sa.key),
            ("value_proj", sa.value),
        ]:
            handles.append(mod.register_forward_hook(_make_linear_hook(probe, i, q)))

        # --- Attention wrapper, optionally with approx softmax ---
        sm_degree = resolve_softmax_degree(approx_cfg, i)
        if sm_degree is None:
            sm_fn = None
        else:
            sm_fn = _make_approx_softmax_with_stats(
                degree=sm_degree,
                lower_bound=Exp_bound[sm_degree],
                layer_idx=i,
                q=q,
            )
        orig_fwd = sa.forward
        sa.forward = _make_attn_wrapper_with_softmax(orig_fwd, i, q, sm_fn)
        restore.append(("fwd", sa, orig_fwd))

        # Linear projections
        for probe, mod in [
            ("attn_output", layer.attention.output.dense),
            ("gelu_input", layer.intermediate.dense),
            ("ffn2_output", layer.output.dense),
        ]:
            handles.append(mod.register_forward_hook(_make_linear_hook(probe, i, q)))

        # LayerNorm outputs
        for probe, mod in [
            ("post_attn_ln", layer.attention.output.LayerNorm),
            ("post_ffn_ln", layer.output.LayerNorm),
        ]:
            handles.append(mod.register_forward_hook(_make_hook(probe, i, q)))
        handles.append(
            layer.attention.output.LayerNorm.register_forward_pre_hook(
                _make_ln_internals_pre_hook("ln1", i, q)
            )
        )
        handles.append(
            layer.output.LayerNorm.register_forward_pre_hook(
                _make_ln_internals_pre_hook("ln2", i, q)
            )
        )

        # --- GELU: optionally replace with StatsPolynomialGELU ---
        gelu_degree = resolve_gelu_degree(approx_cfg, i)
        orig_act = layer.intermediate.intermediate_act_fn
        if gelu_degree is None:
            inner_act = orig_act
        else:
            inner_act = StatsPolynomialGELU(
                degree=gelu_degree, layer_idx=i, q=q
            ).to(
                device=layer.intermediate.dense.weight.device,
                dtype=layer.intermediate.dense.weight.dtype,
            )
        layer.intermediate.intermediate_act_fn = _ActWrapper(inner_act, i, q)
        restore.append(("bert_act", layer.intermediate, orig_act))

    return handles, restore


def _install_gpt2_hooks_approx(model, q, approx_cfg):
    handles = []
    restore = []
    n_embd = model.config.n_embd

    def _ids_hook(_mod, inp):
        if inp[0] is not None:
            _enqueue("input_ids", 0, inp[0].detach().float(), q)

    handles.append(model.transformer.wte.register_forward_pre_hook(_ids_hook))
    handles.append(
        model.transformer.drop.register_forward_hook(
            lambda _m, _i, o: _enqueue("after_embed", 0, o.detach(), q)
        )
    )

    num_layers = model.config.n_layer
    for i in range(num_layers):
        block = model.transformer.h[i]
        attn = block.attn

        # Q / K / V — split from combined c_attn + accum_max
        def _make_qkv_hook(li, ne):
            def hook(mod, inp, out):
                qp, kp, vp = out.split(ne, dim=-1)
                x = inp[0].detach()
                w_abs = mod.weight.detach().abs()
                acc = _ORIG_MATMUL(x.abs(), w_abs)
                if mod.bias is not None:
                    acc = acc + mod.bias.detach().abs()
                qa, ka, va = acc.split(ne, dim=-1)
                _enqueue("query_proj", li, qp.detach(), q, accum_ma=qa.max().item())
                _enqueue("key_proj", li, kp.detach(), q, accum_ma=ka.max().item())
                _enqueue("value_proj", li, vp.detach(), q, accum_ma=va.max().item())

            return hook

        handles.append(attn.c_attn.register_forward_hook(_make_qkv_hook(i, n_embd)))

        # Attention internals (with optional approx softmax)
        sm_degree = resolve_softmax_degree(approx_cfg, i)
        if sm_degree is None:
            sm_fn = None
        else:
            sm_fn = _make_approx_softmax_with_stats(
                degree=sm_degree,
                lower_bound=Exp_bound[sm_degree],
                layer_idx=i,
                q=q,
            )
        orig_fwd = attn.forward
        attn.forward = _make_attn_wrapper_with_softmax(orig_fwd, i, q, sm_fn)
        restore.append(("fwd", attn, orig_fwd))

        handles.append(attn.c_proj.register_forward_hook(
            _make_linear_hook("attn_output", i, q)))

        # LayerNorm probes
        handles.append(block.ln_1.register_forward_pre_hook(
            _make_pre_hook("ln1_input", i, q)))
        handles.append(block.ln_1.register_forward_hook(
            _make_hook("ln1_output", i, q)))
        handles.append(block.ln_1.register_forward_pre_hook(
            _make_ln_internals_pre_hook("ln1", i, q)))

        def _make_ln2_pre_hook(li):
            def hook(_mod, inp):
                t = inp[0].detach()
                _enqueue("ln2_input", li, t, q)
                _enqueue("post_attn_ln", li, t, q)
            return hook

        handles.append(block.ln_2.register_forward_pre_hook(_make_ln2_pre_hook(i)))
        handles.append(block.ln_2.register_forward_hook(
            _make_hook("ln2_output", i, q)))
        handles.append(block.ln_2.register_forward_pre_hook(
            _make_ln_internals_pre_hook("ln2", i, q)))

        # FFN1
        handles.append(block.mlp.c_fc.register_forward_hook(
            _make_linear_hook("gelu_input", i, q)))

        # --- GELU: optionally replace ---
        gelu_degree = resolve_gelu_degree(approx_cfg, i)
        orig_act = block.mlp.act
        if gelu_degree is None:
            new_act = orig_act
        else:
            new_act = StatsPolynomialGELU(
                degree=gelu_degree, layer_idx=i, q=q
            ).to(
                device=block.mlp.c_fc.weight.device,
                dtype=block.mlp.c_fc.weight.dtype,
            )
        # Always wrap so that gelu_output is captured consistently.
        block.mlp.act = _ActWrapper(new_act, i, q)
        restore.append(("gpt2_act", block.mlp, orig_act))

        # FFN2
        handles.append(block.mlp.c_proj.register_forward_hook(
            _make_linear_hook("ffn2_output", i, q)))

        def _make_block_hook(li):
            def hook(_mod, _inp, out):
                _enqueue("post_ffn_ln", li, out[0].detach(), q)
            return hook

        handles.append(block.register_forward_hook(_make_block_hook(i)))

    return handles, restore


def install_hooks_approx(model, arch, q, approx_cfg):
    if arch == "bert":
        return _install_bert_hooks_approx(model, q, approx_cfg)
    if arch == "gpt2":
        return _install_gpt2_hooks_approx(model, q, approx_cfg)
    raise ValueError(f"Unknown arch: {arch}")


# ============================================================================
# Per-task driver (mirrors base.process_task, with approx config recording)
# ============================================================================


def _layer_probes():
    return [p for p in PROBE_POINTS if p not in GLOBAL_PROBES]


def _dump_approx_summary(cfg, num_layers, fout):
    fout.write("\nApproximation config (effective, per-layer):\n")
    fout.write(f'  {"Layer":<6}{"softmax_deg":>14}{"softmax_lb":>14}{"gelu_deg":>12}\n')
    fout.write(f'  {"-" * 46}\n')
    for li in range(num_layers):
        sd = resolve_softmax_degree(cfg, li)
        gd = resolve_gelu_degree(cfg, li)
        sd_str = "—" if sd is None else str(sd)
        lb_str = "—" if sd is None else f"{Exp_bound[sd]}"
        gd_str = "—" if gd is None else str(gd)
        fout.write(f'  L{li:<5}{sd_str:>14}{lb_str:>14}{gd_str:>12}\n')


def process_task_approx(task_name, cfg, approx_cfg, output_dir, device,
                        max_length=128, batch_size=32, max_samples=0):
    arch = cfg["arch"]
    num_layers = cfg["num_layers"]
    print(f'\n{"=" * 70}')
    print(f'  Task: {task_name.upper()}  (arch={arch}, layers={num_layers})')
    print(f'  Approx config:')
    print(f'    softmax default_degree = {approx_cfg["softmax"].get("default_degree")}')
    print(f'    softmax per_layer      = {approx_cfg["softmax"].get("per_layer", {})}')
    print(f'    gelu    default_degree = {approx_cfg["gelu"].get("default_degree")}')
    print(f'    gelu    per_layer      = {approx_cfg["gelu"].get("per_layer", {})}')
    print(f'{"=" * 70}')

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    if arch == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"],
            num_labels=cfg["num_labels"],
            pad_token_id=tokenizer.pad_token_id,
            trust_remote_code=True,
        )
    elif arch == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.to(device).eval()
    print(f'  Model : {cfg["model_name"]}')

    if arch == "bert":
        dl = _prepare_bert_data(cfg, tokenizer, max_length, batch_size, max_samples)
    else:
        dl = _prepare_gpt2_data(cfg, tokenizer, max_length, batch_size, max_samples)

    collector = LayerWiseCollector(num_layers=num_layers)
    stats_queue = queue.Queue(maxsize=4096)
    lock = threading.Lock()
    worker = threading.Thread(
        target=_stats_worker, args=(stats_queue, collector, lock), daemon=True
    )
    worker.start()

    handles, restore = install_hooks_approx(model, arch, stats_queue, approx_cfg)

    print("  Running forward pass (with function substitution) …")
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"  {task_name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    stats_queue.put(None)
    stats_queue.join()
    worker.join()
    remove_hooks(handles, restore)

    # ---- Persist the effective approximation config beside outputs ----
    eff_cfg_path = os.path.join(output_dir, f"{task_name}_approx_config.json")
    eff_cfg = {
        "model": cfg["model_name"],
        "arch": arch,
        "num_layers": num_layers,
        "config": approx_cfg,
        "resolved_per_layer": [
            {
                "layer": li,
                "softmax_degree": resolve_softmax_degree(approx_cfg, li),
                "softmax_lower_bound": (
                    None if resolve_softmax_degree(approx_cfg, li) is None
                    else Exp_bound[resolve_softmax_degree(approx_cfg, li)]
                ),
                "gelu_degree": resolve_gelu_degree(approx_cfg, li),
            }
            for li in range(num_layers)
        ],
    }
    with open(eff_cfg_path, "w") as fout:
        json.dump(eff_cfg, fout, indent=2)
    print(f"  Saved: {eff_cfg_path}")

    # ---- Text summary ----
    txt_path = os.path.join(output_dir, f"{task_name}_all_stats.txt")
    with open(txt_path, "w") as fout:
        fout.write(f'Model: {cfg["model_name"]}  Arch: {arch}  Layers: {num_layers}\n')
        _dump_approx_summary(approx_cfg, num_layers, fout)
        for probe in PROBE_POINTS:
            is_global = probe in GLOBAL_PROBES
            layers_range = [0] if is_global else range(num_layers)
            header = f'\nProbe: {probe} ({base.PROBE_DISPLAY[probe]})'
            if is_global:
                header += "  [global]"
            col_hdr = (f'  {"Layer":<8}{"Count":>14}{"Mean":>12}{"Std":>12}'
                       f'{"Min":>12}{"Max":>12}{"AccumMax":>14}')
            sep = f'  {"-" * 82}'
            fout.write(header + "\n" + col_hdr + "\n" + sep + "\n")
            print(header); print(col_hdr); print(sep)
            for li in layers_range:
                s = collector.stats(probe, li)
                if s:
                    prefix = "  All " if is_global else f"  L{li:<6}"
                    acc_str = (f'{s["accum_max"]:>14.4f}'
                               if "accum_max" in s else f'{"—":>14}')
                    line = (f'{prefix}{s["count"]:>14,}{s["mean"]:>12.4f}'
                            f'{s["std"]:>12.4f}{s["min"]:>12.4f}'
                            f'{s["max"]:>12.4f}{acc_str}')
                    fout.write(line + "\n")
                    print(line)
    print(f"  Saved: {txt_path}")

    # ---- CSV ----
    csv_path = os.path.join(output_dir, f"{task_name}_all_stats.csv")
    with open(csv_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["task", "arch", "model", "probe", "probe_display",
                         "layer", "count", "mean", "std", "min", "max",
                         "accum_max", "softmax_deg", "gelu_deg"])
        for probe in PROBE_POINTS:
            is_global = probe in GLOBAL_PROBES
            layers_range = [0] if is_global else range(num_layers)
            for li in layers_range:
                s = collector.stats(probe, li)
                if s:
                    sd = resolve_softmax_degree(approx_cfg, li) if not is_global else ""
                    gd = resolve_gelu_degree(approx_cfg, li) if not is_global else ""
                    writer.writerow([
                        task_name, arch, cfg["model_name"],
                        probe, base.PROBE_DISPLAY[probe],
                        "all" if is_global else li,
                        s["count"],
                        f'{s["mean"]:.6f}', f'{s["std"]:.6f}',
                        f'{s["min"]:.6f}', f'{s["max"]:.6f}',
                        f'{s["accum_max"]:.6f}' if "accum_max" in s else "",
                        "" if sd is None else sd,
                        "" if gd is None else gd,
                    ])
    print(f"  Saved: {csv_path}")

    # ---- Magnitude text summary ----
    mag_txt_path = os.path.join(output_dir, f"{task_name}_magnitude_stats.txt")
    mag_bin_labs = _mag_bin_labels()
    with open(mag_txt_path, "w") as fout:
        fout.write(f'Model: {cfg["model_name"]}  Arch: {arch}  Layers: {num_layers}\n')
        fout.write(f'Magnitude bins: {MAG_NBINS} bins from '
                   f'{MAG_EDGES[0]:.0e} to {MAG_EDGES[-1]:.0e}\n')
        _dump_approx_summary(approx_cfg, num_layers, fout)
        for probe in PROBE_POINTS:
            is_global = probe in GLOBAL_PROBES
            layers_range = [0] if is_global else range(num_layers)
            header = f'\nProbe: {probe} ({base.PROBE_DISPLAY[probe]})'
            if is_global:
                header += "  [global]"
            col_hdr = (f'  {"Layer":<8}{"Count":>14}{"pct_zero":>10}'
                       f'{"pct>1":>10}{"pct>10":>10}'
                       f'  | magnitude bin percentages ...')
            sep = f'  {"-" * 60}'
            fout.write(header + "\n" + col_hdr + "\n" + sep + "\n")
            print(header); print(col_hdr); print(sep)
            for li in layers_range:
                ms = collector.mag_stats(probe, li)
                if ms:
                    prefix = "  All " if is_global else f"  L{li:<6}"
                    bins_str = "  ".join(f"{v:6.2f}" for v in ms["pct_bins"])
                    line = (f'{prefix}{ms["count"]:>14,}'
                            f'{ms["pct_zero"]:>10.3f}'
                            f'{ms["pct_gt1"]:>10.3f}'
                            f'{ms["pct_gt10"]:>10.3f}'
                            f'  | {bins_str}')
                    fout.write(line + "\n")
                    print(line)
            agg = collector.mag_stats_aggregated(probe, num_layers)
            if agg and not is_global:
                bins_str = "  ".join(f"{v:6.2f}" for v in agg["pct_bins"])
                line = (f'  {"AGG":<6}{agg["count"]:>14,}'
                        f'{agg["pct_zero"]:>10.3f}'
                        f'{agg["pct_gt1"]:>10.3f}'
                        f'{agg["pct_gt10"]:>10.3f}'
                        f'  | {bins_str}')
                fout.write(line + "\n")
                print(line)
    print(f"  Saved: {mag_txt_path}")

    # ---- Magnitude CSV ----
    mag_csv_path = os.path.join(output_dir, f"{task_name}_magnitude_stats.csv")
    with open(mag_csv_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            ["task", "arch", "model", "probe", "probe_display",
             "layer", "softmax_deg", "gelu_deg",
             "count", "n_zero", "pct_zero",
             "n_gt1", "pct_gt1", "n_gt10", "pct_gt10"]
            + [f"pct_{lab}" for lab in mag_bin_labs]
        )
        for probe in PROBE_POINTS:
            is_global = probe in GLOBAL_PROBES
            layers_range = [0] if is_global else range(num_layers)
            for li in layers_range:
                ms = collector.mag_stats(probe, li)
                if ms:
                    sd = resolve_softmax_degree(approx_cfg, li) if not is_global else ""
                    gd = resolve_gelu_degree(approx_cfg, li) if not is_global else ""
                    writer.writerow([
                        task_name, arch, cfg["model_name"],
                        probe, base.PROBE_DISPLAY[probe],
                        "all" if is_global else li,
                        "" if sd is None else sd,
                        "" if gd is None else gd,
                        ms["count"], ms["n_zero"],
                        f'{ms["pct_zero"]:.4f}',
                        ms["n_gt1"], f'{ms["pct_gt1"]:.4f}',
                        ms["n_gt10"], f'{ms["pct_gt10"]:.4f}',
                    ] + [f"{v:.4f}" for v in ms["pct_bins"]])
            if not is_global:
                agg = collector.mag_stats_aggregated(probe, num_layers)
                if agg:
                    writer.writerow([
                        task_name, arch, cfg["model_name"],
                        probe, base.PROBE_DISPLAY[probe], "AGG",
                        "", "",
                        agg["count"], agg["n_zero"],
                        f'{agg["pct_zero"]:.4f}',
                        agg["n_gt1"], f'{agg["pct_gt1"]:.4f}',
                        agg["n_gt10"], f'{agg["pct_gt10"]:.4f}',
                    ] + [f"{v:.4f}" for v in agg["pct_bins"]])
    print(f"  Saved: {mag_csv_path}")

    # ---- Plots ----
    for probe in PROBE_POINTS:
        plot_probe_histograms(collector, probe, task_name, output_dir, num_layers)
    plot_overview(collector, task_name, output_dir, num_layers)
    plot_heatmap(collector, task_name, output_dir, num_layers)

    for probe in PROBE_POINTS:
        plot_probe_magnitude_bar(collector, probe, task_name, output_dir, num_layers)
        plot_magnitude_per_layer(collector, probe, task_name, output_dir, num_layers)
    plot_magnitude_heatmap(collector, task_name, output_dir, num_layers)
    plot_outlier_overview(collector, task_name, output_dir, num_layers)

    del model, collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# CLI
# ============================================================================


def main():
    bert_tasks = [k for k, v in TASK_REGISTRY.items() if v["arch"] == "bert"]
    gpt2_tasks = [k for k, v in TASK_REGISTRY.items() if v["arch"] == "gpt2"]

    parser = argparse.ArgumentParser(
        description="Per-computation distribution analysis with function substitution"
    )
    parser.add_argument("--output_dir", type=str, default="all_analysis_approx")
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help=(f"Tasks to run.  BERT: {bert_tasks}  GPT-2: {gpt2_tasks}  "
              f"Default: all BERT tasks"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples per task, 0 = all")
    parser.add_argument("--approx_config", type=str, default=None,
                        help="Path to approximation config JSON. "
                             "Supports dataset-keyed (recommended) or legacy "
                             "flat format; see load_approx_config docstring.")
    parser.add_argument("--stage", type=str, default="stage1",
                        help="Stage name inside a dataset-keyed config "
                             "(default: stage1)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    tasks = args.tasks or bert_tasks
    print(f"Tasks         : {tasks}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Device        : {device}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Max length    : {args.max_length}")
    print(f'Max samples   : {"all" if args.max_samples == 0 else args.max_samples}')
    print(f"Approx config : {args.approx_config or '<default>'}")
    print(f"Stage         : {args.stage}")

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f'\n[Warning] Unknown task "{task_name}", skipping')
            continue
        # Load approx config per-task (dataset-keyed format reads the right
        # section; legacy flat format just returns the same dict every time).
        approx_cfg = load_approx_config(
            args.approx_config, task_name=task_name, stage=args.stage
        )
        process_task_approx(task_name, TASK_REGISTRY[task_name], approx_cfg,
                            args.output_dir, device,
                            args.max_length, args.batch_size, args.max_samples)

    print(f'\n{"=" * 70}')
    print(f"  All done!  Results saved to {args.output_dir}/")
    print(f'{"=" * 70}')


if __name__ == "__main__":
    main()
