#!/usr/bin/env python
"""
Analyze all intermediate tensor distributions across BERT and GPT-2 layers
on representative datasets — layer-wise statistics only.

Supported architectures:
  BERT  — fine-tuned on GLUE classification tasks (post-norm transformer)
  GPT-2 — pretrained causal LM on WikiText-2      (pre-norm transformer)

Probe points (in transformer-block order):
  Global (not per-layer):
    1.  input_ids     — Raw token indices
    2.  after_embed   — Embedding output
  Per-layer:
    3.  query_proj    — XW_q + b_q
    4.  key_proj      — XW_k + b_k
    5.  value_proj    — XW_v + b_v
    6.  qkt_raw       — QK^T (raw, before /√d_k and mask)
    7.  attn_scores   — QK^T/√d_k + mask  (pre-softmax)
    8.  attn_probs    — Softmax(scores)
    9.  attn_context  — attn_probs × V (heads concatenated)
   10.  attn_output   — context × W_O
   11.  post_attn_ln  — after 1st sub-block (BERT: res+LN; GPT-2: res only)
   12.  gelu_input    — FFN1 dense output (GELU input)
   13.  gelu_output   — GELU(FFN1)
   14.  ffn2_output   — FFN2 dense output
   15.  post_ffn_ln   — after 2nd sub-block (BERT: res+LN; GPT-2: res only)

Usage:
    nohup python analyze_all_distribution.py --tasks sst2 gpt2_wt2 --max_samples 5000 > all_analysis/run.log 2>&1 &
    nohup python analyze_all_distribution.py --tasks gpt2_wt2 gpt2m_wt2 --output_dir all_analysis > all_analysis/run.log 2>&1 &

    # GPT-2 only (quick test)
    nohup python analyze_all_distribution.py --tasks gpt2_wt2 --max_samples 2000 --output_dir all_analysis > all_analysis/run.log 2>&1 &

    # GPT-2 Medium (24 layers; consider smaller batch/length for memory)
    nohup python analyze_all_distribution.py --tasks gpt2m_wt2 --batch_size 16 --max_length 512 --output_dir all_analysis > all_analysis/run.log 2>&1 &

    # Mix BERT + GPT-2
    nohup python analyze_all_distribution.py --tasks sst2 gpt2_wt2 --max_samples 5000 --output_dir all_analysis > all_analysis/run.log 2>&1 &

    
"""

import os
import csv
import argparse
import queue
import threading
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== Task Registry ====================

TASK_REGISTRY = {
    # ---- BERT (GLUE classification, post-norm, 12 layers) ----
    'cola': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-CoLA',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'cola',
        'num_labels': 2, 'input_cols': ('sentence',),
    },
    'sst2': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-SST-2',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'sst2',
        'num_labels': 2, 'input_cols': ('sentence',),
    },
    'mrpc': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-MRPC',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'mrpc',
        'num_labels': 2, 'input_cols': ('sentence1', 'sentence2'),
    },
    'stsb': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-STS-B',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'stsb',
        'num_labels': 1, 'input_cols': ('sentence1', 'sentence2'),
    },
    'mnli': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-MNLI',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'mnli',
        'num_labels': 3, 'input_cols': ('premise', 'hypothesis'),
    },
    'qnli': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-QNLI',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'qnli',
        'num_labels': 2, 'input_cols': ('question', 'sentence'),
    },
    'rte': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-RTE',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'rte',
        'num_labels': 2, 'input_cols': ('sentence1', 'sentence2'),
    },
    'wnli': {
        'arch': 'bert', 'num_layers': 12,
        'model_name': 'textattack/bert-base-uncased-WNLI',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'wnli',
        'num_labels': 2, 'input_cols': ('sentence1', 'sentence2'),
    },
    # ---- BERT-Large (GLUE classification, post-norm, 24 layers) ----
    'bl_cola': {
        'arch': 'bert', 'num_layers': 24,
        'model_name': 'yoshitomo-matsubara/bert-large-uncased-cola',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'cola',
        'num_labels': 2, 'input_cols': ('sentence',),
    },
    'bl_sst2': {
        'arch': 'bert', 'num_layers': 24,
        'model_name': 'assemblyai/bert-large-uncased-sst2',
        'dataset_name': 'nyu-mll/glue', 'dataset_config': 'sst2',
        'num_labels': 2, 'input_cols': ('sentence',),
    },
    # ---- GPT-2 (causal LM on WikiText-2, pre-norm) ----
    'gpt2_wt2': {
        'arch': 'gpt2', 'num_layers': 12,
        'model_name': 'gpt2',
        'dataset_name': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
    },
    'gpt2m_wt2': {
        'arch': 'gpt2', 'num_layers': 24,
        'model_name': 'gpt2-medium',
        'dataset_name': 'wikitext', 'dataset_config': 'wikitext-2-raw-v1',
    },
}

# ==================== Probe Configuration ====================

PROBE_POINTS = [
    'input_ids',
    'after_embed',
    'query_proj',
    'key_proj',
    'value_proj',
    'qkt_raw',
    'attn_scores',
    'attn_probs',
    'attn_context',
    'attn_output',
    'post_attn_ln',
    'gelu_input',
    'gelu_output',
    'ffn2_output',
    'post_ffn_ln',
    # Pre-norm LayerNorm input/output (GPT-2 only; BERT leaves these empty)
    'ln1_input',
    'ln1_output',
    'ln2_input',
    'ln2_output',
]

GLOBAL_PROBES = {'input_ids', 'after_embed'}

PROBE_DISPLAY = {
    'input_ids':    'Input IDs',
    'after_embed':  'After Embedding',
    'query_proj':   'XW_q + b_q',
    'key_proj':     'XW_k + b_k',
    'value_proj':   'XW_v + b_v',
    'qkt_raw':      'QKᵀ (raw)',
    'attn_scores':  'QKᵀ/√dₖ + mask',
    'attn_probs':   'Softmax(scores)',
    'attn_context': 'Softmax × V',
    'attn_output':  'Attn × Wₒ',
    'post_attn_ln': 'Post-Attn sub-block',
    'gelu_input':   'FFN1 (GELU in)',
    'gelu_output':  'GELU output',
    'ffn2_output':  'FFN2 output',
    'post_ffn_ln':  'Post-FFN sub-block',
    'ln1_input':    'LN₁ input (pre-norm)',
    'ln1_output':   'LN₁ output (pre-norm)',
    'ln2_input':    'LN₂ input (pre-norm)',
    'ln2_output':   'LN₂ output (pre-norm)',
}

PROBE_HIST_RANGE = {
    'input_ids':    (0.0, 50300.0, 300),
    'after_embed':  (-5.0, 5.0, 300),
    'query_proj':   (-5.0, 5.0, 300),
    'key_proj':     (-5.0, 5.0, 300),
    'value_proj':   (-5.0, 5.0, 300),
    'qkt_raw':      (-100.0, 100.0, 300),
    'attn_scores':  (-50.0, 50.0, 300),
    'attn_probs':   (0.0, 1.0, 200),
    'attn_context': (-5.0, 5.0, 300),
    'attn_output':  (-5.0, 5.0, 300),
    'post_attn_ln': (-5.0, 5.0, 300),
    'gelu_input':   (-15.0, 15.0, 300),
    'gelu_output':  (-15.0, 15.0, 300),
    'ffn2_output':  (-15.0, 15.0, 300),
    'post_ffn_ln':  (-5.0, 5.0, 300),
    'ln1_input':    (-15.0, 15.0, 300),
    'ln1_output':   (-5.0, 5.0, 300),
    'ln2_input':    (-15.0, 15.0, 300),
    'ln2_output':   (-5.0, 5.0, 300),
}

PROBE_COLORS = {
    'input_ids':    '#95a5a6',
    'after_embed':  '#7f8c8d',
    'query_proj':   '#e74c3c',
    'key_proj':     '#c0392b',
    'value_proj':   '#d35400',
    'qkt_raw':      '#e67e22',
    'attn_scores':  '#f39c12',
    'attn_probs':   '#f1c40f',
    'attn_context': '#2ecc71',
    'attn_output':  '#27ae60',
    'post_attn_ln': '#1abc9c',
    'gelu_input':   '#3498db',
    'gelu_output':  '#2980b9',
    'ffn2_output':  '#9b59b6',
    'post_ffn_ln':  '#8e44ad',
    'ln1_input':    '#16a085',
    'ln1_output':   '#1abc9c',
    'ln2_input':    '#2c3e50',
    'ln2_output':   '#34495e',
}

# ==================== Layer-wise Statistics Collector ====================


class LayerWiseCollector:

    def __init__(self, probe_names=PROBE_POINTS, num_layers=12):
        self.probe_names = probe_names
        self.num_layers = num_layers
        self._d = {}
        for p in probe_names:
            hmin, hmax, hbins = PROBE_HIST_RANGE[p]
            self._d[p] = {
                'edges': np.linspace(hmin, hmax, hbins + 1),
                'hist': [np.zeros(hbins, dtype=np.int64) for _ in range(num_layers)],
                'n':  [0] * num_layers,
                's':  [0.0] * num_layers,
                'sq': [0.0] * num_layers,
                'mi': [float('inf')] * num_layers,
                'ma': [float('-inf')] * num_layers,
            }

    def merge(self, probe, layer, bs):
        d = self._d[probe]
        d['hist'][layer] += bs['hist']
        d['n'][layer]  += bs['n']
        d['s'][layer]  += bs['s']
        d['sq'][layer] += bs['sq']
        d['mi'][layer] = min(d['mi'][layer], bs['mi'])
        d['ma'][layer] = max(d['ma'][layer], bs['ma'])

    def stats(self, probe, layer):
        d = self._d[probe]
        n = d['n'][layer]
        if n == 0:
            return None
        mean = d['s'][layer] / n
        var = d['sq'][layer] / n - mean ** 2
        return {
            'count': n,
            'mean': mean,
            'std': np.sqrt(max(var, 0)),
            'min': d['mi'][layer],
            'max': d['ma'][layer],
        }

    def histogram(self, probe, layer):
        d = self._d[probe]
        centers = (d['edges'][:-1] + d['edges'][1:]) / 2
        return centers, d['hist'][layer]


# ==================== GPU Stats Computation ====================


@torch.no_grad()
def _gpu_stats(tensor, hmin, hmax, hbins):
    flat = tensor.reshape(-1).float()
    hist = torch.histc(flat, bins=hbins, min=hmin, max=hmax)
    flat_d = flat.double()
    return {
        'hist': hist.long().cpu().numpy(),
        'n':  flat.numel(),
        's':  flat_d.sum().cpu().item(),
        'sq': flat_d.pow(2).sum().cpu().item(),
        'mi': flat.min().cpu().item(),
        'ma': flat.max().cpu().item(),
    }


def _enqueue(probe, layer, tensor, q):
    hmin, hmax, hbins = PROBE_HIST_RANGE[probe]
    q.put((probe, layer, _gpu_stats(tensor, hmin, hmax, hbins)))


# ==================== Shared Hook Helpers ====================


def _make_hook(probe, layer, q):
    def hook(_mod, _inp, out):
        _enqueue(probe, layer, out.detach(), q)
    return hook


def _make_pre_hook(probe, layer, q):
    def hook(_mod, inp):
        _enqueue(probe, layer, inp[0].detach(), q)
    return hook


class _ActWrapper(nn.Module):
    """Wraps an activation function to capture GELU output."""

    def __init__(self, orig, layer, q):
        super().__init__()
        self.orig = orig
        self.layer = layer
        self.q = q

    def forward(self, x):
        y = self.orig(x)
        _enqueue('gelu_output', self.layer, y.detach(), self.q)
        return y


def _make_attn_wrapper(fwd, li, q):
    """Wrap a self-attention forward to capture qkt_raw / attn_scores / probs / context.

    Works for both BERT (BertSelfAttention) and GPT-2 (GPT2Attention) because
    both internally use ``torch.matmul`` for QK^T and probs@V, and
    ``F.softmax`` for the attention softmax.

    The 1st matmul is QK^T, the 2nd is probs@V (the true context = convex
    combination of value vectors).  For BERT the module returns the context
    directly; for GPT-2 the module continues with merge_heads + c_proj +
    dropout, so we must capture the 2nd matmul explicitly.
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
                captured['qkt'] = result.detach()
            elif mm_count[0] == 2:
                captured['context'] = result.detach()
            return result

        def _cap_sm(inp, dim=None, **kw):
            captured['scores'] = inp.detach()
            result = _real_sm(inp, dim=dim, **kw)
            captured['probs'] = result.detach()
            return result

        torch.matmul = _cap_mm
        torch.nn.functional.softmax = _cap_sm
        try:
            outputs = fwd(*args, **kwargs)
        finally:
            torch.matmul = _real_mm
            torch.nn.functional.softmax = _real_sm

        if 'qkt' in captured:
            _enqueue('qkt_raw', li, captured['qkt'], q)
        if 'scores' in captured:
            _enqueue('attn_scores', li, captured['scores'], q)
        if 'probs' in captured:
            _enqueue('attn_probs', li, captured['probs'], q)
        if 'context' in captured:
            _enqueue('attn_context', li, captured['context'], q)
        else:
            _enqueue('attn_context', li, outputs[0].detach(), q)
        return outputs
    return _wrapped


# ==================== BERT Hook Installation ====================


def _install_bert_hooks(model, q):
    handles = []
    restore = []

    # Global: input_ids
    def _ids_hook(_mod, inp):
        if inp[0] is not None:
            _enqueue('input_ids', 0, inp[0].detach().float(), q)
    handles.append(
        model.bert.embeddings.word_embeddings.register_forward_pre_hook(_ids_hook))

    # Global: after_embed
    handles.append(
        model.bert.embeddings.register_forward_hook(
            lambda _m, _i, o: _enqueue('after_embed', 0, o.detach(), q)))

    for i, layer in enumerate(model.bert.encoder.layer):
        sa = layer.attention.self

        # Q / K / V
        for probe, mod in [
            ('query_proj', sa.query),
            ('key_proj',   sa.key),
            ('value_proj', sa.value),
        ]:
            handles.append(mod.register_forward_hook(_make_hook(probe, i, q)))

        # Attention internals
        orig_fwd = sa.forward
        sa.forward = _make_attn_wrapper(orig_fwd, i, q)
        restore.append(('fwd', sa, orig_fwd))

        # Post-attention hooks
        for probe, mod in [
            ('attn_output',  layer.attention.output.dense),
            ('post_attn_ln', layer.attention.output.LayerNorm),
            ('gelu_input',   layer.intermediate.dense),
            ('ffn2_output',  layer.output.dense),
            ('post_ffn_ln',  layer.output.LayerNorm),
        ]:
            handles.append(mod.register_forward_hook(_make_hook(probe, i, q)))

        # GELU output
        orig_act = layer.intermediate.intermediate_act_fn
        layer.intermediate.intermediate_act_fn = _ActWrapper(orig_act, i, q)
        restore.append(('bert_act', layer.intermediate, orig_act))

    return handles, restore


# ==================== GPT-2 Hook Installation ====================


def _install_gpt2_hooks(model, q):
    """GPT-2 uses pre-norm: LN → Attn → Res, then LN → MLP → Res.

    Module layout per block ``model.transformer.h[i]``:
      .ln_1          — LayerNorm before attention
      .attn.c_attn   — combined Q/K/V projection  (Conv1D, out_dim = 3*n_embd)
      .attn.c_proj   — output projection W_O       (Conv1D)
      .ln_2          — LayerNorm before MLP
      .mlp.c_fc      — FFN1                        (Conv1D)
      .mlp.act       — GELU activation
      .mlp.c_proj    — FFN2                        (Conv1D)
    """
    handles = []
    restore = []
    n_embd = model.config.n_embd

    # Global: input_ids
    def _ids_hook(_mod, inp):
        if inp[0] is not None:
            _enqueue('input_ids', 0, inp[0].detach().float(), q)
    handles.append(
        model.transformer.wte.register_forward_pre_hook(_ids_hook))

    # Global: after_embed (output of embedding dropout = wte + wpe)
    handles.append(
        model.transformer.drop.register_forward_hook(
            lambda _m, _i, o: _enqueue('after_embed', 0, o.detach(), q)))

    num_layers = model.config.n_layer
    for i in range(num_layers):
        block = model.transformer.h[i]
        attn = block.attn

        # Q / K / V  — split from combined c_attn output
        def _make_qkv_hook(li, ne):
            def hook(_mod, _inp, out):
                qp, kp, vp = out.split(ne, dim=-1)
                _enqueue('query_proj', li, qp.detach(), q)
                _enqueue('key_proj', li, kp.detach(), q)
                _enqueue('value_proj', li, vp.detach(), q)
            return hook
        handles.append(attn.c_attn.register_forward_hook(_make_qkv_hook(i, n_embd)))

        # Attention internals (qkt_raw, attn_scores, attn_probs, attn_context)
        orig_fwd = attn.forward
        attn.forward = _make_attn_wrapper(orig_fwd, i, q)
        restore.append(('fwd', attn, orig_fwd))

        # attn_output = output of W_O projection
        handles.append(attn.c_proj.register_forward_hook(
            _make_hook('attn_output', i, q)))

        # ---- LayerNorm probes (pre-norm architecture) ----
        # ln_1: input = block input (prev residual); output = normalized → attn
        handles.append(block.ln_1.register_forward_pre_hook(
            _make_pre_hook('ln1_input', i, q)))
        handles.append(block.ln_1.register_forward_hook(
            _make_hook('ln1_output', i, q)))

        # ln_2: input = residual + attn_output; output = normalized → MLP
        # Also fill post_attn_ln with the same tensor (ln_2 input) for compat
        def _make_ln2_pre_hook(li):
            def hook(_mod, inp):
                t = inp[0].detach()
                _enqueue('ln2_input', li, t, q)
                _enqueue('post_attn_ln', li, t, q)
            return hook
        handles.append(block.ln_2.register_forward_pre_hook(
            _make_ln2_pre_hook(i)))
        handles.append(block.ln_2.register_forward_hook(
            _make_hook('ln2_output', i, q)))

        # gelu_input = FFN1 output
        handles.append(block.mlp.c_fc.register_forward_hook(
            _make_hook('gelu_input', i, q)))

        # gelu_output
        act = block.mlp.act
        if isinstance(act, nn.Module):
            handles.append(act.register_forward_hook(
                _make_hook('gelu_output', i, q)))
        else:
            block.mlp.act = _ActWrapper(act, i, q)
            restore.append(('gpt2_act', block.mlp, act))

        # ffn2_output = FFN2 output
        handles.append(block.mlp.c_proj.register_forward_hook(
            _make_hook('ffn2_output', i, q)))

        # post_ffn_ln: block output = residual + MLP output
        def _make_block_hook(li):
            def hook(_mod, _inp, out):
                _enqueue('post_ffn_ln', li, out[0].detach(), q)
            return hook
        handles.append(block.register_forward_hook(_make_block_hook(i)))

    return handles, restore


# ==================== Unified Hook Dispatch ====================


def install_hooks(model, arch, q):
    if arch == 'bert':
        return _install_bert_hooks(model, q)
    if arch == 'gpt2':
        return _install_gpt2_hooks(model, q)
    raise ValueError(f'Unknown architecture: {arch}')


def remove_hooks(handles, restore):
    for h in handles:
        h.remove()
    for tag, *rest in restore:
        if tag == 'fwd':
            mod, orig = rest
            mod.forward = orig
        elif tag == 'bert_act':
            parent, orig = rest
            parent.intermediate_act_fn = orig
        elif tag == 'gpt2_act':
            parent, orig = rest
            parent.act = orig


# ==================== Stats Worker Thread ====================


def _stats_worker(q, collector, lock):
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            break
        probe, layer, bs = item
        with lock:
            collector.merge(probe, layer, bs)
        q.task_done()


# ==================== Data Preparation ====================


def _prepare_bert_data(cfg, tokenizer, max_length, batch_size, max_samples):
    data = load_dataset(cfg['dataset_name'], cfg['dataset_config'])
    split = data['train']
    if 0 < max_samples < len(split):
        split = split.select(range(max_samples))
    print(f'  Samples: {len(split)}')

    input_cols = cfg['input_cols']

    def _tok(examples):
        if len(input_cols) == 1:
            return tokenizer(examples[input_cols[0]],
                             truncation=True, padding=False, max_length=max_length)
        return tokenizer(examples[input_cols[0]], examples[input_cols[1]],
                         truncation=True, padding=False, max_length=max_length)

    ds = split.map(_tok, batched=True)
    cols = ['input_ids', 'attention_mask']
    if 'token_type_ids' in ds.column_names:
        cols.append('token_type_ids')
    ds.set_format(type='torch', columns=cols)

    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer, padding='max_length',
            max_length=max_length, return_tensors='pt',
            pad_to_multiple_of=8,
        ),
    )
    return dl


def _prepare_gpt2_data(cfg, tokenizer, max_length, batch_size, max_samples):
    data = load_dataset(cfg['dataset_name'], cfg['dataset_config'])
    split = data['train']

    def _tok(examples):
        return tokenizer(examples['text'], truncation=False)

    tokenized = split.map(_tok, batched=True, remove_columns=split.column_names)

    block_size = max_length

    def _group(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concatenated['input_ids']) // block_size) * block_size
        return {
            k: [vals[i:i + block_size] for i in range(0, total, block_size)]
            for k, vals in concatenated.items()
        }

    lm_ds = tokenized.map(_group, batched=True)
    if 0 < max_samples < len(lm_ds):
        lm_ds = lm_ds.select(range(max_samples))
    print(f'  Samples (chunks of {block_size}): {len(lm_ds)}')

    lm_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dl = DataLoader(lm_ds, batch_size=batch_size, shuffle=False,
                    collate_fn=default_data_collator)
    return dl


# ==================== Plotting ====================


def _layer_probes():
    return [p for p in PROBE_POINTS if p not in GLOBAL_PROBES]


def plot_probe_histograms(collector, probe, task_name, output_dir, num_layers):
    is_global = probe in GLOBAL_PROBES
    display = PROBE_DISPLAY[probe]
    color = PROBE_COLORS[probe]

    if is_global:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(f'{display} Distribution — {task_name.upper()}',
                     fontsize=16, fontweight='bold')
        centers, hist = collector.histogram(probe, 0)
        bw = centers[1] - centers[0]
        ax.bar(centers, hist.astype(float), width=bw,
               color=color, alpha=0.8, edgecolor='none')
        s = collector.stats(probe, 0)
        if s:
            ax.set_title(f'μ={s["mean"]:.4f}, σ={s["std"]:.4f}, '
                         f'min={s["min"]:.4f}, max={s["max"]:.4f}',
                         fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    else:
        ncols = 3
        nrows = (num_layers + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5.5))
        fig.suptitle(f'{display} Distribution — {task_name.upper()}',
                     fontsize=18, fontweight='bold', y=0.98)
        for li in range(num_layers):
            r, c = divmod(li, ncols)
            ax = axes[r][c] if nrows > 1 else axes[c]
            centers, hist = collector.histogram(probe, li)
            bw = centers[1] - centers[0]
            ax.bar(centers, hist.astype(float), width=bw,
                   color=color, alpha=0.8, edgecolor='none')
            s = collector.stats(probe, li)
            title = f'Layer {li}'
            if s:
                title += f'  (μ={s["mean"]:.4f}, σ={s["std"]:.4f})'
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        for idx in range(num_layers, nrows * ncols):
            r, c = divmod(idx, ncols)
            (axes[r][c] if nrows > 1 else axes[c]).set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fp = os.path.join(output_dir, f'{task_name}_{probe}_hist.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fp}')


def plot_overview(collector, task_name, output_dir, num_layers):
    lp = _layer_probes()
    n = len(lp)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.5))
    fig.suptitle(f'Per-Layer Intermediate Distributions — {task_name.upper()}',
                 fontsize=18, fontweight='bold', y=0.99)
    layers = np.arange(num_layers)

    for idx, probe in enumerate(lp):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        means, stds, mins, maxs = [], [], [], []
        for li in range(num_layers):
            s = collector.stats(probe, li)
            if s:
                means.append(s['mean']); stds.append(s['std'])
                mins.append(s['min']); maxs.append(s['max'])
            else:
                means.append(0); stds.append(0)
                mins.append(0); maxs.append(0)
        means, stds = np.array(means), np.array(stds)
        mins, maxs = np.array(mins), np.array(maxs)
        color = PROBE_COLORS[probe]

        ax.plot(layers, means, 'o-', color=color, lw=2, ms=4, label='mean')
        ax.fill_between(layers, means - stds, means + stds,
                        color=color, alpha=0.2, label='±1σ')
        ax.plot(layers, mins, '--', color=color, alpha=0.4, lw=1, label='min')
        ax.plot(layers, maxs, '--', color=color, alpha=0.4, lw=1, label='max')
        ax.set_title(PROBE_DISPLAY[probe], fontsize=12, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_xticks(layers)
        ax.legend(fontsize=7, loc='best')
        ax.grid(axis='y', alpha=0.3)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fp = os.path.join(output_dir, f'{task_name}_overview.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fp}')


def plot_heatmap(collector, task_name, output_dir, num_layers):
    lp = _layer_probes()
    nprobes = len(lp)
    fig_w = max(14, num_layers * 1.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, max(5, nprobes * 0.55)))
    fig.suptitle(f'Layer-wise Statistics Heatmap — {task_name.upper()}',
                 fontsize=16, fontweight='bold')

    mean_data = np.zeros((nprobes, num_layers))
    std_data  = np.zeros_like(mean_data)
    for pi, probe in enumerate(lp):
        for li in range(num_layers):
            s = collector.stats(probe, li)
            if s:
                mean_data[pi, li] = s['mean']
                std_data[pi, li]  = s['std']

    ylabels = [PROBE_DISPLAY[p] for p in lp]
    xlabels = [f'L{i}' for i in range(num_layers)]
    fontsize_cell = 5.5 if num_layers <= 12 else 4.0

    vabs = max(abs(mean_data.min()), abs(mean_data.max())) or 1.0
    im1 = ax1.imshow(mean_data, aspect='auto', cmap='RdBu_r',
                     vmin=-vabs, vmax=vabs, interpolation='nearest')
    ax1.set_title('Mean', fontsize=13)
    ax1.set_yticks(range(nprobes))
    ax1.set_yticklabels(ylabels, fontsize=9)
    ax1.set_xticks(range(num_layers))
    ax1.set_xticklabels(xlabels, fontsize=8)
    fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02)
    for pi in range(nprobes):
        for li in range(num_layers):
            ax1.text(li, pi, f'{mean_data[pi, li]:.2f}',
                     ha='center', va='center', fontsize=fontsize_cell)

    im2 = ax2.imshow(std_data, aspect='auto', cmap='Oranges',
                     interpolation='nearest')
    ax2.set_title('Std Dev', fontsize=13)
    ax2.set_yticks(range(nprobes))
    ax2.set_yticklabels(ylabels, fontsize=9)
    ax2.set_xticks(range(num_layers))
    ax2.set_xticklabels(xlabels, fontsize=8)
    fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02)
    for pi in range(nprobes):
        for li in range(num_layers):
            ax2.text(li, pi, f'{std_data[pi, li]:.2f}',
                     ha='center', va='center', fontsize=fontsize_cell)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(output_dir, f'{task_name}_heatmap.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {fp}')


# ==================== Per-Task Processing ====================


def process_task(task_name, cfg, output_dir, device,
                 max_length=128, batch_size=32, max_samples=0):
    arch = cfg['arch']
    num_layers = cfg['num_layers']
    print(f'\n{"=" * 70}')
    print(f'  Task: {task_name.upper()}  (arch={arch}, layers={num_layers})')
    print(f'{"=" * 70}')

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or '[PAD]'

    if arch == 'bert':
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg['model_name'],
            num_labels=cfg['num_labels'],
            pad_token_id=tokenizer.pad_token_id,
            trust_remote_code=True,
        )
    elif arch == 'gpt2':
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_name'],
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        raise ValueError(f'Unknown arch: {arch}')

    model.to(device).eval()
    print(f'  Model : {cfg["model_name"]}')

    # ---- Prepare data ----
    if arch == 'bert':
        dl = _prepare_bert_data(cfg, tokenizer, max_length, batch_size, max_samples)
    else:
        dl = _prepare_gpt2_data(cfg, tokenizer, max_length, batch_size, max_samples)

    # ---- Collect ----
    collector = LayerWiseCollector(num_layers=num_layers)
    stats_queue = queue.Queue(maxsize=4096)
    lock = threading.Lock()
    worker = threading.Thread(target=_stats_worker,
                              args=(stats_queue, collector, lock), daemon=True)
    worker.start()

    handles, restore = install_hooks(model, arch, stats_queue)

    print('  Running forward pass …')
    with torch.no_grad():
        for batch in tqdm(dl, desc=f'  {task_name}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    stats_queue.put(None)
    stats_queue.join()
    worker.join()
    remove_hooks(handles, restore)

    # ---- Console + text summary ----
    txt_path = os.path.join(output_dir, f'{task_name}_all_stats.txt')
    with open(txt_path, 'w') as fout:
        fout.write(f'Model: {cfg["model_name"]}  Arch: {arch}  Layers: {num_layers}\n')
        for probe in PROBE_POINTS:
            is_global = probe in GLOBAL_PROBES
            layers_range = [0] if is_global else range(num_layers)
            header = f'\nProbe: {probe} ({PROBE_DISPLAY[probe]})'
            if is_global:
                header += '  [global]'
            col_hdr = (f'  {"Layer":<8}{"Count":>14}{"Mean":>12}{"Std":>12}'
                       f'{"Min":>12}{"Max":>12}')
            sep = f'  {"-" * 70}'
            fout.write(header + '\n' + col_hdr + '\n' + sep + '\n')
            print(header)
            print(col_hdr)
            print(sep)
            for li in layers_range:
                s = collector.stats(probe, li)
                if s:
                    prefix = '  All ' if is_global else f'  L{li:<6}'
                    line = (f'{prefix}{s["count"]:>14,}{s["mean"]:>12.4f}'
                            f'{s["std"]:>12.4f}{s["min"]:>12.4f}{s["max"]:>12.4f}')
                    fout.write(line + '\n')
                    print(line)
    print(f'  Saved: {txt_path}')

    # ---- CSV ----
    csv_path = os.path.join(output_dir, f'{task_name}_all_stats.csv')
    with open(csv_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['task', 'arch', 'model', 'probe', 'probe_display',
                         'layer', 'count', 'mean', 'std', 'min', 'max'])
        for probe in PROBE_POINTS:
            is_global = probe in GLOBAL_PROBES
            layers_range = [0] if is_global else range(num_layers)
            for li in layers_range:
                s = collector.stats(probe, li)
                if s:
                    writer.writerow([
                        task_name, arch, cfg['model_name'],
                        probe, PROBE_DISPLAY[probe],
                        'all' if is_global else li,
                        s['count'],
                        f'{s["mean"]:.6f}', f'{s["std"]:.6f}',
                        f'{s["min"]:.6f}', f'{s["max"]:.6f}',
                    ])
    print(f'  Saved: {csv_path}')

    # ---- Plots ----
    for probe in PROBE_POINTS:
        plot_probe_histograms(collector, probe, task_name, output_dir, num_layers)
    plot_overview(collector, task_name, output_dir, num_layers)
    plot_heatmap(collector, task_name, output_dir, num_layers)

    del model, collector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Main ====================


def main():
    bert_tasks = [k for k, v in TASK_REGISTRY.items() if v['arch'] == 'bert']
    gpt2_tasks = [k for k, v in TASK_REGISTRY.items() if v['arch'] == 'gpt2']

    parser = argparse.ArgumentParser(
        description='Analyze intermediate distributions in BERT / GPT-2')
    parser.add_argument('--output_dir', type=str, default='all_analysis')
    parser.add_argument(
        '--tasks', type=str, nargs='+', default=None,
        help=(f'Tasks to run.  BERT: {bert_tasks}  GPT-2: {gpt2_tasks}  '
              f'Default: all BERT tasks'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Sequence length (default 128; GPT-2 supports up to 1024)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=0,
                        help='Max samples per task, 0 = all')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('[Warning] CUDA not available, falling back to CPU')
        device = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)
    tasks = args.tasks or bert_tasks

    print(f'Tasks      : {tasks}')
    print(f'Output dir : {args.output_dir}')
    print(f'Device     : {device}')
    print(f'Batch size : {args.batch_size}')
    print(f'Max length : {args.max_length}')
    print(f'Max samples: {"all" if args.max_samples == 0 else args.max_samples}')

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f'\n[Warning] Unknown task "{task_name}", skipping')
            continue
        process_task(task_name, TASK_REGISTRY[task_name], args.output_dir,
                     device, args.max_length, args.batch_size, args.max_samples)

    print(f'\n{"=" * 70}')
    print(f'  All done!  Results saved to {args.output_dir}/')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
