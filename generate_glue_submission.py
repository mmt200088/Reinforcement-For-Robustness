#!/usr/bin/env python
"""
================================================================================
GLUE 基准测试提交文件生成器
================================================================================

根据优化后的 GELU/Softmax 近似配置 以及 噪声 Scaling Factor 配置，在 GLUE
测试集上运行推理并生成可提交到 https://gluebenchmark.com/ 的 TSV 文件。

支持四种组合模式：
  1. 纯基线 (--no_approx --no_noise)     : 原始 GELU + exp，无噪声
  2. 仅近似 (--config X --no_noise)       : GELU/Softmax 多项式近似，无噪声
  3. 仅噪声 (--no_approx --noise_config Y): 原始函数，注入噪声
  4. 近似+噪声 (--config X --noise_config Y): 同时使用两阶段优化结果

支持的 GLUE 任务：cola, sst2, mrpc, stsb, mnli, qnli, rte, wnli
(QQP 和 AX 自动以占位符填充)

================================================================================
命令行参数说明
================================================================================

必选（二选一）：
  --config PATH         GELU/Softmax 近似配置 JSON 文件路径
                        (与 --no_approx 互斥；不使用 --no_approx 时必须提供)
  --no_approx           跳过 GELU/Softmax 多项式近似，使用原始函数

噪声相关：
  --noise_config PATH   噪声 scaling factor 配置 JSON 文件路径
                        (提供此参数即启用噪声注入)
  --no_noise            显式禁用噪声注入（默认行为，可省略；与 --noise_config 互斥）

可选：
  --output_dir DIR      输出目录 (默认: glue_submission)
  --tasks TASK [TASK..] 仅运行指定任务 (默认: 配置文件中的所有任务)
  --device DEVICE       推理设备 (默认: cuda)
  --max_length N        最大序列长度 (默认: 128)
  --batch_size N        推理批大小 (默认: 16)

================================================================================
配置文件格式
================================================================================

统一的合并 JSON（推荐，glue_final_configs_best_{ppo|genetic}.json）结构：
  顶层按模型变体分节 ("bert-base" / "bert-large" / "gpt-2")，
  每个任务下同时包含 "stage1" 和 "stage2" 两个子块。
  --config 和 --noise_config 都接收同一份合并 JSON；
  本脚本会自动抽取 stage1 用于近似、stage2 用于噪声注入。

  示例：
  {
      "bert-base": {
          "qnli": {
              "stage1": {
                  "gelu":    [1, 1, 1, 1, 2, 4, 4, 4, 4, 1, 1, 1],
                  "softmax": [2, 3, 4, 4, 3, 2, 2, 4, 3, 5, 5, 5]
              },
              "stage2": {
                  "x":     [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
                  "wq":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                  "wk":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                  "wv":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                  "wo":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                  "wffn1": [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
                  "wffn2": [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
              }
          },
          ...
      }
  }

取值范围：
  GELU degree ∈ {0, 1, 2, 4}，其中 0 表示 ReLU；Softmax degree ∈ {2, 3, 4, 5, 6}。
  x ∈ {22, 24, 26, 28, 30}；wq/wk/wv/wo/wffn2 ∈ {14, 16, 18, 20, 22}；
  wffn1 ∈ {16, 18, 20, 22, 24}。噪声数值越大 → 隐私保护越强。

兼容性：加载器也能识别旧的分离式 JSON
  （仅含 "gelu"/"softmax" 或仅含 "x"/"wq"/... 的任务字典）——
  这种情况下它按原样返回，不做 stage1/stage2 抽取。

================================================================================
使用示例
================================================================================

# 1) 纯基线：原始模型，无近似无噪声
python generate_glue_submission.py --no_approx --output_dir glue_baseline

# 2) 仅 GELU/Softmax 近似（无噪声）
python generate_glue_submission.py --config glue_final_configs_best_ppo.json --output_dir glue_approx

# 3) 仅噪声注入（无近似）
python generate_glue_submission.py --no_approx --noise_config glue_final_configs_best_ppo.json --output_dir glue_noise_only

# 4) 近似 + 噪声（完整两阶段优化）同一份合并 JSON 同时用作 --config 和 --noise_config
python generate_glue_submission.py --config glue_final_configs_best_ppo.json --noise_config glue_final_configs_best_ppo.json --output_dir glue_full

# 5) 指定部分任务
python generate_glue_submission.py --config glue_final_configs_best_ppo.json --noise_config glue_final_configs_best_ppo.json --tasks qnli sst2 mrpc

# 6) 近似 + GA 最优噪声组合
python generate_glue_submission.py --config glue_final_configs_best_genetic.json --noise_config glue_final_configs_best_genetic.json --output_dir glue_ga_full

================================================================================
输出说明
================================================================================

生成的文件位于 --output_dir 指定目录下：
  CoLA.tsv, SST-2.tsv, MRPC.tsv, STS-B.tsv,
  MNLI-m.tsv, MNLI-mm.tsv, QNLI.tsv, RTE.tsv, WNLI.tsv,
  QQP.tsv (占位符), AX.tsv (占位符),
  submission.zip (包含上述所有 TSV)

将 submission.zip 上传到 https://gluebenchmark.com/ 即可获得评测结果。
================================================================================
"""

import json
import os
import sys
import argparse
import zipfile
import re
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from json_utils import read_json_file

try:
    from datasets import load_dataset
    _DATASETS_IMPORT_ERROR = None
except ImportError as exc:
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    DataCollatorWithPadding = None
    _TRANSFORMERS_IMPORT_ERROR = exc
from function_handler import (
    ReversibleLayerHandler,
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
)

sys.setrecursionlimit(50000)

NOISE_KEYS = ('x', 'wq', 'wk', 'wv', 'wo', 'wffn1', 'wffn2')

NOISE_HANDLER_MAP = {
    'x':     ('replace_layer_input_noise',            'fresh'),
    'wq':    ('replace_layer_query_noise',             'encoding'),
    'wk':    ('replace_layer_key_noise',               'encoding'),
    'wv':    ('replace_layer_value_noise',             'encoding'),
    'wo':    ('replace_layer_attention_output_noise',   'encoding'),
    'wffn1': ('replace_layer_ffn1_noise',              'encoding'),
    'wffn2': ('replace_layer_ffn2_noise',              'encoding'),
}
NOISE_ALLOWED_SCALING_FACTORS = {
    'x': tuple(INPUT_NOISE_ALLOWED_SCALING_FACTORS),
    'wq': tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
    'wk': tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
    'wv': tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
    'wo': tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
    'wffn1': tuple(WFFN1_NOISE_ALLOWED_SCALING_FACTORS),
    'wffn2': tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
}

# ==================== GLUE Task Registry ====================
TASK_REGISTRY = {
    'cola': {
        'model_name': 'textattack/bert-base-uncased-CoLA',
        'glue_name': 'cola',
        'num_labels': 2,
        'input_cols': ('sentence',),
        'output_file': 'CoLA.tsv',
        'label_map': {0: '0', 1: '1'},
        'task_type': 'classification',
        'test_split': 'test',
    },
    'sst2': {
        'model_name': 'textattack/bert-base-uncased-SST-2',
        'glue_name': 'sst2',
        'num_labels': 2,
        'input_cols': ('sentence',),
        'output_file': 'SST-2.tsv',
        'label_map': {0: '0', 1: '1'},
        'task_type': 'classification',
        'test_split': 'test',
    },
    'mrpc': {
        'model_name': 'textattack/bert-base-uncased-MRPC',
        'glue_name': 'mrpc',
        'num_labels': 2,
        'input_cols': ('sentence1', 'sentence2'),
        'output_file': 'MRPC.tsv',
        'label_map': {0: '0', 1: '1'},
        'task_type': 'classification',
        'test_split': 'test',
    },
    'stsb': {
        'model_name': 'textattack/bert-base-uncased-STS-B',
        'glue_name': 'stsb',
        'num_labels': 1,
        'input_cols': ('sentence1', 'sentence2'),
        'output_file': 'STS-B.tsv',
        'label_map': None,
        'task_type': 'regression',
        'test_split': 'test',
    },
    'mnli': {
        'model_name': 'textattack/bert-base-uncased-MNLI',
        'glue_name': 'mnli',
        'num_labels': 3,
        'input_cols': ('premise', 'hypothesis'),
        'output_file': ['MNLI-m.tsv', 'MNLI-mm.tsv'],
        'label_map': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'task_type': 'classification',
        'test_split': ['test_matched', 'test_mismatched'],
    },
    'qnli': {
        'model_name': 'textattack/bert-base-uncased-QNLI',
        'glue_name': 'qnli',
        'num_labels': 2,
        'input_cols': ('question', 'sentence'),
        'output_file': 'QNLI.tsv',
        'label_map': {0: 'entailment', 1: 'not_entailment'},
        'task_type': 'classification',
        'test_split': 'test',
    },
    'rte': {
        'model_name': 'textattack/bert-base-uncased-RTE',
        'glue_name': 'rte',
        'num_labels': 2,
        'input_cols': ('sentence1', 'sentence2'),
        'output_file': 'RTE.tsv',
        'label_map': {0: 'entailment', 1: 'not_entailment'},
        'task_type': 'classification',
        'test_split': 'test',
    },
    'wnli': {
        'model_name': 'textattack/bert-base-uncased-WNLI',
        'glue_name': 'wnli',
        'num_labels': 2,
        'input_cols': ('sentence1', 'sentence2'),
        'output_file': 'WNLI.tsv',
        'label_map': {0: '0', 1: '1'},
        'task_type': 'classification',
        'test_split': 'test',
    },
    'qqp': {
        'model_name': 'textattack/bert-base-uncased-QQP',
        'glue_name': 'qqp',
        'num_labels': 2,
        'input_cols': ('question1', 'question2'),
        'output_file': 'QQP.tsv',
        'label_map': {0: '0', 1: '1'},
        'task_type': 'classification',
        'test_split': 'test',
    },
    # AX is the GLUE diagnostic set for NLI; predictions are produced by the
    # MNLI model on the dedicated `ax` dataset (test split only).
    'ax': {
        'model_name': 'textattack/bert-base-uncased-MNLI',
        'glue_name': 'ax',
        'num_labels': 3,
        'input_cols': ('premise', 'hypothesis'),
        'output_file': 'AX.tsv',
        'label_map': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
        'task_type': 'classification',
        'test_split': 'test',
    },
}

# ---- bert-large model name overrides (yoshitomo-matsubara family) ----
# GLUE tasks without a reliable bert-large checkpoint are omitted on purpose
# (mnli/ax/wnli/qqp). Unsupported tasks are skipped with a warning at runtime.
BERT_LARGE_MODEL_NAMES = {
    'cola': 'yoshitomo-matsubara/bert-large-uncased-cola',
    'sst2': 'yoshitomo-matsubara/bert-large-uncased-sst2',
    'mrpc': 'yoshitomo-matsubara/bert-large-uncased-mrpc',
    'stsb': 'yoshitomo-matsubara/bert-large-uncased-stsb',
    'qqp': 'yoshitomo-matsubara/bert-large-uncased-qqp',
    'qnli': 'yoshitomo-matsubara/bert-large-uncased-qnli',
    'rte':  'yoshitomo-matsubara/bert-large-uncased-rte',
}

# ---- gpt-2 model name overrides ----
# PavanNeerudu/gpt2-finetuned-<task> is a family of GPT2ForSequenceClassification
# checkpoints already fine-tuned on each GLUE training set (correct head shape,
# backbone converged). We use these directly so RL only needs to optimize the
# approximation / noise schedule on top of a frozen, task-competent backbone.
# AX uses the MNLI checkpoint per GLUE convention (AX is the diagnostic set for NLI).
GPT2_MODEL_NAMES = {
    'cola': 'PavanNeerudu/gpt2-finetuned-cola',
    'sst2': 'PavanNeerudu/gpt2-finetuned-sst2',
    'mrpc': 'PavanNeerudu/gpt2-finetuned-mrpc',
    'stsb': 'PavanNeerudu/gpt2-finetuned-stsb',
    'qqp':  'PavanNeerudu/gpt2-finetuned-qqp',
    'qnli': 'PavanNeerudu/gpt2-finetuned-qnli',
    'rte':  'PavanNeerudu/gpt2-finetuned-rte',
    'wnli': 'PavanNeerudu/gpt2-finetuned-wnli',
    'mnli': 'PavanNeerudu/gpt2-finetuned-mnli',
    'ax':   'PavanNeerudu/gpt2-finetuned-mnli',
}


def _unwrap_variant_config(cfg_map, model_type, cfg_path, stage_key=None):
    """
    Accept the merged schema
        {"bert-base": {task: {"stage1": {...}, "stage2": {...}}}, ...}
    the per-stage variant schema
        {"bert-base": {task: {...}}, "bert-large": {task: {...}}}
    and the legacy flat schema
        {task: {...}}  (implicitly bert-base only)
    and return the task-level dict for the selected `model_type`. When the
    merged schema is detected and `stage_key` ("stage1" / "stage2") is
    supplied, the corresponding sub-dict is extracted for every task.
    """
    if not isinstance(cfg_map, dict):
        raise ValueError(f"Config file '{cfg_path}' is not a JSON object.")
    has_variant_keys = any(k in cfg_map for k in ('bert-base', 'bert-large', 'gpt-2'))
    if has_variant_keys:
        if model_type not in cfg_map:
            raise KeyError(
                f"Config file '{cfg_path}' has no '{model_type}' section "
                f"(found keys: {sorted(cfg_map.keys())})."
            )
        section = cfg_map[model_type]
        if not isinstance(section, dict):
            raise ValueError(
                f"Config file '{cfg_path}' section '{model_type}' is not a dict."
            )
    else:
        # Legacy flat schema — only valid for bert-base.
        if model_type != 'bert-base':
            raise KeyError(
                f"Config file '{cfg_path}' uses the legacy flat schema which only "
                f"supports bert-base; add a '{model_type}' section to use it with "
                f"--model_type {model_type}."
            )
        section = cfg_map

    if stage_key is not None:
        sample_task = next(
            (t for t, v in section.items() if isinstance(v, dict) and t != "_comment"),
            None,
        )
        if sample_task is not None and stage_key in section[sample_task]:
            extracted = {}
            for task, per_task in section.items():
                if task == "_comment" or not isinstance(per_task, dict):
                    continue
                if stage_key not in per_task:
                    raise KeyError(
                        f"Config file '{cfg_path}' task '{task}' missing "
                        f"'{stage_key}' section."
                    )
                extracted[task] = per_task[stage_key]
            return extracted
    return section


EXPECTED_LINES = {
    'AX.tsv': 1105,
    'CoLA.tsv': 1064,
    'MNLI-mm.tsv': 9848,
    'MNLI-m.tsv': 9797,
    'MRPC.tsv': 1726,
    'QNLI.tsv': 5464,
    'QQP.tsv': 390966,
    'RTE.tsv': 3001,
    'SST-2.tsv': 1822,
    'STS-B.tsv': 1380,
    'WNLI.tsv': 147,
}

PLACEHOLDER_DEFAULTS = {
    'AX.tsv': 'entailment',
    'CoLA.tsv': '0',
    'MNLI-mm.tsv': 'entailment',
    'MNLI-m.tsv': 'entailment',
    'MRPC.tsv': '0',
    'QNLI.tsv': 'entailment',
    'QQP.tsv': '0',
    'RTE.tsv': 'entailment',
    'SST-2.tsv': '0',
    'STS-B.tsv': '0.000',
    'WNLI.tsv': '0',
}

EXPECTED_LABELS = {
    'AX.tsv': {'entailment', 'neutral', 'contradiction'},
    'CoLA.tsv': {'0', '1'},
    'MNLI-mm.tsv': {'entailment', 'neutral', 'contradiction'},
    'MNLI-m.tsv': {'entailment', 'neutral', 'contradiction'},
    'MRPC.tsv': {'0', '1'},
    'QNLI.tsv': {'entailment', 'not_entailment'},
    'QQP.tsv': {'0', '1'},
    'RTE.tsv': {'entailment', 'not_entailment'},
    'SST-2.tsv': {'0', '1'},
    'WNLI.tsv': {'0', '1'},
}

TEXTATTACK_MNLI_LABEL_MAP = {
    0: 'contradiction',
    1: 'entailment',
    2: 'neutral',
}


def detect_layer_attribute(model):
    candidates = ['bert.encoder.layer', 'model.layers', 'transformer.h', 'roberta.encoder.layer']
    for path in candidates:
        try:
            obj = model
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if len(obj) > 0:
                return path
        except Exception:
            continue
    return 'bert.encoder.layer'


def apply_approx_configuration(handler, layers_attribute, gelu_degrees, softmax_degrees):
    handler_layer_name = "model." + layers_attribute
    gelu_map = {d: [] for d in [0, 1, 2, 4]}
    for idx, deg in enumerate(gelu_degrees):
        deg_int = int(deg)
        if deg_int not in gelu_map:
            raise ValueError(f"Unsupported Stage-1 GELU degree: {deg_int}")
        gelu_map[deg_int].append(idx)
    for d in [0, 1, 2, 4]:
        if gelu_map[d]:
            handler.replace_layer_gelu(gelu_map[d], handler_layer_name, degree=d)

    softmax_map = {d: [] for d in range(2, 7)}
    for idx, deg in enumerate(softmax_degrees):
        deg_int = int(deg)
        if deg_int not in softmax_map:
            raise ValueError(f"Unsupported Stage-1 Softmax degree: {deg_int}")
        softmax_map[deg_int].append(idx)
    for d in range(2, 7):
        if softmax_map[d]:
            handler.replace_layer_softmax(softmax_map[d], handler_layer_name, degree=d)


def apply_noise_configuration(handler, layers_attribute, noise_config):
    handler_layer_name = "model." + layers_attribute

    for noise_key in NOISE_KEYS:
        method_name, distribution = NOISE_HANDLER_MAP[noise_key]
        factors = noise_config[noise_key]
        allowed = NOISE_ALLOWED_SCALING_FACTORS[noise_key]

        sf_map = {sf: [] for sf in allowed}
        for idx, sf in enumerate(factors):
            sf_int = int(sf)
            if sf_int not in sf_map:
                raise ValueError(
                    f"Noise key '{noise_key}' layer {idx}: scaling factor {sf_int} "
                    f"not in allowed set {list(allowed)}"
                )
            sf_map[sf_int].append(idx)

        replace_fn = getattr(handler, method_name)
        for sf in allowed:
            if sf_map[sf]:
                replace_fn(sf_map[sf], handler_layer_name,
                           scaling_factor=sf, distribution=distribution)


def _extract_applied_noise_state(handler, noise_key, num_layers):
    scaling_factors = [None] * num_layers
    distributions = [None] * num_layers

    if noise_key == 'x':
        source = getattr(handler, "original_input_noise", {})
        for idx, meta in source.items():
            layer_idx = int(idx)
            scaling_factors[layer_idx] = int(meta.get("scaling_factor"))
            distributions[layer_idx] = str(meta.get("distribution"))
        return scaling_factors, distributions

    gpt2_slot_map = {
        'wq': 'query',
        'wk': 'key',
        'wv': 'value',
    }
    if getattr(handler, "_arch", None) == "gpt2" and noise_key in gpt2_slot_map:
        qkv_state = getattr(handler, "_gpt2_qkv_state", {})
        for idx, per_layer in qkv_state.items():
            slot = gpt2_slot_map[noise_key]
            if slot not in per_layer:
                continue
            scaling_factor, distribution = per_layer[slot]
            layer_idx = int(idx)
            scaling_factors[layer_idx] = int(scaling_factor)
            distributions[layer_idx] = str(distribution)
        return scaling_factors, distributions

    projection_key_map = {
        'wq': 'query',
        'wk': 'key',
        'wv': 'value',
        'wo': 'wo',
        'wffn1': 'wffn1',
        'wffn2': 'wffn2',
    }
    projection_store = getattr(handler, "original_projection_noise", {})
    source = projection_store.get(projection_key_map[noise_key], {})
    for idx, meta in source.items():
        layer_idx = int(idx)
        scaling_factors[layer_idx] = int(meta.get("scaling_factor"))
        distributions[layer_idx] = str(meta.get("distribution"))
    return scaling_factors, distributions


def verify_noise_configuration(handler, noise_config, num_layers):
    for noise_key in NOISE_KEYS:
        expected_factors = [int(v) for v in noise_config[noise_key]]
        expected_distribution = NOISE_HANDLER_MAP[noise_key][1]
        actual_factors, actual_distributions = _extract_applied_noise_state(
            handler, noise_key, num_layers
        )

        mismatches = []
        for idx, expected_factor in enumerate(expected_factors):
            if actual_factors[idx] != expected_factor:
                mismatches.append(
                    f"layer {idx}: expected sf={expected_factor}, got {actual_factors[idx]}"
                )
                continue
            if actual_distributions[idx] != expected_distribution:
                mismatches.append(
                    f"layer {idx}: expected dist={expected_distribution}, got {actual_distributions[idx]}"
                )

        if mismatches:
            raise RuntimeError(
                f"Noise verification failed for '{noise_key}': "
                + "; ".join(mismatches[:5])
            )

        unique_factors = sorted(set(expected_factors))
        print(
            f"  [Verify] Noise {noise_key:5s}: {num_layers}/{num_layers} layers, "
            f"scaling={unique_factors}, distribution={expected_distribution}"
        )


def run_inference(model, dataloader, device):
    model.eval()
    model.to(device)
    all_logits = []

    use_cuda = torch.cuda.is_available() and 'cuda' in str(device)
    if use_cuda:
        try:
            dummy = next(iter(dataloader))
            dummy = {k: v.to(device, non_blocking=True) for k, v in dummy.items()}
            with torch.no_grad():
                _ = model(**dummy)
            torch.cuda.synchronize()
        except StopIteration:
            pass

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Inference"):
            batch = {k: v.to(device, non_blocking=use_cuda) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().float().cpu().numpy()
            all_logits.append(logits)

    if not all_logits:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(all_logits, axis=0)


def _is_generic_model_label(raw_label):
    return re.fullmatch(r"(?:label[_-]?)?\d+", str(raw_label).strip().lower()) is not None


def _extract_generic_label_id(raw_label):
    match = re.fullmatch(r"(?:label[_-]?)?(\d+)", str(raw_label).strip().lower())
    if match is None:
        return None
    return int(match.group(1))


def _generic_class_index_to_glue_label(task_name, class_idx, task_config):
    """Convert a numeric model class index to GLUE's submission label string."""
    model_name = str(task_config.get('model_name', '')).lower()
    if task_name in ('mnli', 'ax') and 'textattack/' in model_name:
        return TEXTATTACK_MNLI_LABEL_MAP[int(class_idx)]

    label_map = task_config.get('label_map')
    if label_map is not None:
        return str(label_map[int(class_idx)])

    return str(class_idx)


def _normalize_glue_label(task_name, raw_label, class_idx=None, task_config=None):
    """Map a model-specific label string to the GLUE-submission label string.

    Different textattack BERT checkpoints use different id2label conventions
    (e.g. MNLI uses {0:contradiction,1:entailment,2:neutral} on some forks
    but the opposite ordering on others). To stay robust we always read
    `model.config.id2label`, then normalize the *string* to the GLUE format.
    """
    s = str(raw_label).strip().lower().replace('-', '_').replace(' ', '_')
    generic_label_id = _extract_generic_label_id(s)
    if generic_label_id is not None:
        resolved_idx = int(class_idx) if class_idx is not None else generic_label_id
        if task_config is not None:
            return _generic_class_index_to_glue_label(task_name, resolved_idx, task_config)
        return str(generic_label_id)

    # NLI-style tasks
    if task_name in ('mnli', 'ax'):
        if 'contradict' in s:
            return 'contradiction'
        if 'neutral' in s:
            return 'neutral'
        if 'entail' in s:
            return 'entailment'
        return s
    if task_name in ('qnli', 'rte'):
        if 'not' in s and 'entail' in s:
            return 'not_entailment'
        if 'entail' in s:
            return 'entailment'
        return s
    # Binary 0/1 tasks (cola, sst2, mrpc, wnli)
    positive_markers = ('1', 'positive', 'acceptable', 'equivalent',
                        'duplicate', 'entailment', 'true', 'pos')
    negative_markers = ('0', 'negative', 'unacceptable', 'not_equivalent',
                        'not_duplicate', 'not_entailment', 'false', 'neg')
    if s in positive_markers or any(m in s for m in ('acceptable', 'positive', 'equivalent', 'duplicate')) and 'un' not in s and 'not' not in s:
        return '1'
    if s in negative_markers:
        return '0'
    # Last resort: trust the original character
    if s.startswith('1'):
        return '1'
    if s.startswith('0'):
        return '0'
    return s


def logits_to_predictions(logits, task_config, task_name, id2label):
    if task_config['task_type'] == 'regression':
        preds = logits.squeeze()
        if np.ndim(preds) == 0:
            preds = np.array([preds])
        return [f"{np.clip(p, 0.0, 5.0):.3f}" for p in preds]

    if len(logits.shape) == 1:
        pred_classes = (logits > 0.5).astype(int)
    else:
        pred_classes = np.argmax(logits, axis=1)

    if id2label is None:
        return [
            _generic_class_index_to_glue_label(task_name, int(c), task_config)
            for c in pred_classes
        ]

    predictions = []
    for c in pred_classes:
        class_idx = int(c)
        raw_label = id2label.get(class_idx)
        if raw_label is None:
            raw_label = id2label.get(str(class_idx))
        if raw_label is None or _is_generic_model_label(raw_label):
            predictions.append(
                _generic_class_index_to_glue_label(task_name, class_idx, task_config)
            )
        else:
            predictions.append(
                _normalize_glue_label(
                    task_name,
                    raw_label,
                    class_idx=class_idx,
                    task_config=task_config,
                )
            )
    return predictions


def write_tsv(filepath, predictions):
    with open(filepath, 'w') as f:
        f.write("index\tprediction\n")
        for idx, pred in enumerate(predictions):
            f.write(f"{idx}\t{pred}\n")
    print(f"  -> {os.path.basename(filepath)}: {len(predictions) + 1} lines (incl. header)")


def generate_placeholder(filepath, num_predictions, default_label="0"):
    with open(filepath, 'w') as f:
        f.write("index\tprediction\n")
        for idx in range(num_predictions):
            f.write(f"{idx}\t{default_label}\n")
    print(f"  -> [Placeholder] {os.path.basename(filepath)}: {num_predictions + 1} lines")


def _validate_prediction_value(filename, value):
    if filename == 'STS-B.tsv':
        try:
            score = float(value)
        except ValueError:
            return False, "STS-B prediction is not a float"
        if not 0.0 <= score <= 5.0:
            return False, "STS-B prediction outside [0, 5]"
        return True, None

    allowed = EXPECTED_LABELS.get(filename)
    if allowed is not None and value not in allowed:
        return False, f"prediction '{value}' not in {sorted(allowed)}"
    return True, None


def validate_tsv_file(filepath, filename, expected_lines):
    errors = []
    try:
        with open(filepath, 'r') as f:
            lines = [line.rstrip('\n').rstrip('\r') for line in f]
    except OSError as exc:
        return 0, [f"failed to read file: {exc}"]

    actual_lines = len(lines)
    if actual_lines != expected_lines:
        errors.append(f"line count {actual_lines}, expected {expected_lines}")

    if not lines:
        errors.append("empty file")
        return actual_lines, errors

    if lines[0] != "index\tprediction":
        errors.append("header must be exactly 'index\\tprediction'")

    for expected_idx, line in enumerate(lines[1:]):
        parts = line.split('\t')
        if len(parts) != 2:
            errors.append(f"row {expected_idx}: expected 2 tab-separated columns")
            break
        row_idx, prediction = parts
        if row_idx != str(expected_idx):
            errors.append(f"row {expected_idx}: index column is '{row_idx}'")
            break
        ok, message = _validate_prediction_value(filename, prediction)
        if not ok:
            errors.append(f"row {expected_idx}: {message}")
            break

    return actual_lines, errors


def tokenize_and_prepare(dataset_split, tokenizer, input_cols, max_length):
    def tokenize_fn(examples):
        if len(input_cols) == 1:
            return tokenizer(
                examples[input_cols[0]],
                truncation=True, padding=False, max_length=max_length,
            )
        else:
            return tokenizer(
                examples[input_cols[0]], examples[input_cols[1]],
                truncation=True, padding=False, max_length=max_length,
            )

    tokenized = dataset_split.map(tokenize_fn, batched=True)

    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.column_names:
        columns.append("token_type_ids")
    tokenized.set_format(type="torch", columns=columns)
    return tokenized


def process_task(task_name, task_config, gelu_degrees, softmax_degrees,
                 noise_config, output_dir, device, max_length=128, batch_size=16,
                 no_approx=False, no_noise=False):
    if _DATASETS_IMPORT_ERROR is not None:
        raise ImportError(
            "The 'datasets' package is required for GLUE submission generation. "
            "Install it in the active Python environment."
        ) from _DATASETS_IMPORT_ERROR
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "The 'transformers' package is required for GLUE submission generation. "
            "Install it in the active Python environment."
        ) from _TRANSFORMERS_IMPORT_ERROR

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name.upper()}")

    if no_approx:
        print(f"  Approx:  OFF (original GELU + exp)")
    else:
        print(f"  GELU:    {gelu_degrees}")
        print(f"  Softmax: {softmax_degrees}")

    if no_noise:
        print(f"  Noise:   OFF (no noise injection)")
    else:
        for nk in NOISE_KEYS:
            print(f"  Noise {nk:5s}: {noise_config[nk]}")

    print(f"{'=' * 60}")

    model_name = task_config['model_name']
    print(f"  Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=task_config['num_labels'],
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )
    model.to(device)

    model_id2label = None
    if hasattr(model.config, 'id2label') and model.config.id2label:
        model_id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        print(f"  Model label mapping (used for predictions): {model_id2label}")
    else:
        print(f"  [Warning] model.config.id2label missing; falling back to TASK_REGISTRY label_map")

    need_handler = (not no_approx) or (not no_noise)

    if need_handler:
        handler = ReversibleLayerHandler(model)
        layers_attr = detect_layer_attribute(model)
        num_layers = len(eval('model.' + layers_attr))
        print(f"  Layers: {layers_attr} ({num_layers} layers)")

        if not no_approx:
            assert len(gelu_degrees) == num_layers, \
                f"GELU config length ({len(gelu_degrees)}) != model layers ({num_layers})"
            assert len(softmax_degrees) == num_layers, \
                f"Softmax config length ({len(softmax_degrees)}) != model layers ({num_layers})"
            apply_approx_configuration(handler, layers_attr, gelu_degrees, softmax_degrees)

        if not no_noise:
            for nk in NOISE_KEYS:
                assert len(noise_config[nk]) == num_layers, \
                    f"Noise '{nk}' config length ({len(noise_config[nk])}) != model layers ({num_layers})"
            apply_noise_configuration(handler, layers_attr, noise_config)

        # Critical: replace_layer_softmax instantiates a *new* attention
        # module (BertSelfAttentionWithAproximation) on CPU. Without re-
        # moving the model to the target device, inference would fail with
        # a CPU/CUDA device mismatch. Also re-asserts dtype/device for any
        # buffers added by noise wrappers.
        model.to(device)
        model.eval()
        # Sanity check: confirm every parameter is on the requested device.
        bad = [n for n, p in model.named_parameters() if str(p.device) != str(torch.device(device))]
        if bad:
            print(f"  [Warning] {len(bad)} params not on {device}, re-moving. First: {bad[0]}")
            model.to(device)
        # Verify the configuration actually took effect on the live model.
        if not no_approx:
            from function_handler import PolynomialGELU, BertSelfAttentionWithAproximation
            layers_obj = eval('model.' + layers_attr)
            is_gpt2_layers = (layers_attr == 'transformer.h')
            if is_gpt2_layers:
                applied_gelu = sum(
                    1 for L in layers_obj
                    if isinstance(getattr(getattr(L, 'mlp', None), 'act', None), PolynomialGELU)
                )
                applied_relu = sum(
                    1 for L in layers_obj
                    if isinstance(getattr(getattr(L, 'mlp', None), 'act', None), torch.nn.ReLU)
                )
                applied_sm = 0  # softmax approximation not supported on GPT-2
                print(f"  [Verify] PolynomialGELU layers: {applied_gelu}/{len(layers_obj)}, "
                      f"ReLU layers: {applied_relu}/{len(layers_obj)}, "
                      f"ApproxSoftmax layers: N/A (GPT-2, Stage 1 disabled)")
            else:
                applied_gelu = sum(
                    1 for L in layers_obj
                    if isinstance(L.intermediate.intermediate_act_fn, PolynomialGELU)
                )
                applied_relu = sum(
                    1 for L in layers_obj
                    if isinstance(L.intermediate.intermediate_act_fn, torch.nn.ReLU)
                )
                applied_sm = sum(
                    1 for L in layers_obj
                    if isinstance(L.attention.self, BertSelfAttentionWithAproximation)
                )
                print(f"  [Verify] PolynomialGELU layers: {applied_gelu}/{len(layers_obj)}, "
                      f"ReLU layers: {applied_relu}/{len(layers_obj)}, "
                      f"ApproxSoftmax layers: {applied_sm}/{len(layers_obj)}")
        if not no_noise:
            verify_noise_configuration(handler, noise_config, num_layers)
    else:
        handler = None
        print(f"  Using original model (no approximation, no noise)")

    print(f"  Loading GLUE test set: {task_config['glue_name']}")
    data = load_dataset("nyu-mll/glue", task_config['glue_name'])

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )

    test_splits = task_config['test_split']
    output_files = task_config['output_file']

    if isinstance(test_splits, list):
        for split_name, out_file in zip(test_splits, output_files):
            print(f"  Processing split: {split_name}")
            test_data = tokenize_and_prepare(
                data[split_name], tokenizer, task_config['input_cols'], max_length
            )
            dataloader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator
            )
            logits = run_inference(model, dataloader, device)
            predictions = logits_to_predictions(logits, task_config, task_name, model_id2label)
            write_tsv(os.path.join(output_dir, out_file), predictions)
    else:
        test_data = tokenize_and_prepare(
            data[test_splits], tokenizer, task_config['input_cols'], max_length
        )
        dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )
        logits = run_inference(model, dataloader, device)
        predictions = logits_to_predictions(logits, task_config, task_name, model_id2label)
        write_tsv(os.path.join(output_dir, output_files), predictions)

    # Aggressive cleanup: handler holds a deepcopy backup_model that doubles VRAM.
    if handler is not None:
        try:
            handler.backup_model = None
        except Exception:
            pass
        del handler
    model.to('cpu')
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_outputs(output_dir):
    print(f"\n{'=' * 60}")
    print("Verification: TSV format, labels, and line counts")
    print(f"{'=' * 60}")
    all_ok = True
    for filename, expected in sorted(EXPECTED_LINES.items()):
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            actual, errors = validate_tsv_file(filepath, filename, expected)
            if not errors:
                status = "OK"
            else:
                status = f"INVALID ({'; '.join(errors[:2])})"
                all_ok = False
            print(f"  {filename:15s}: {actual:>8d} lines  [{status}]")
        else:
            print(f"  {filename:15s}: MISSING")
            all_ok = False
    return all_ok


def create_submission_zip(output_dir):
    zip_path = os.path.join(output_dir, "submission.zip")
    print(f"\nCreating: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for filename in EXPECTED_LINES:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                zf.write(filepath, filename)
            else:
                print(f"  [Warning] {filename} not found, skipping in ZIP")
    return zip_path


def fill_missing_submission_files(output_dir):
    print(f"\n{'=' * 60}")
    print("Filling missing submission files with placeholders")
    print(f"{'=' * 60}")
    created_files = []
    for filename, expected in sorted(EXPECTED_LINES.items()):
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            continue
        default_label = PLACEHOLDER_DEFAULTS.get(filename, "0")
        generate_placeholder(filepath, expected - 1, default_label=default_label)
        created_files.append(filename)
    return created_files


def remove_stale_submission_files(output_dir):
    removed = []
    for filename in list(EXPECTED_LINES) + ["submission.zip"]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            removed.append(filename)
    if removed:
        print(
            f"[Output] Removed {len(removed)} stale submission file(s) from "
            f"{output_dir}: {', '.join(sorted(removed))}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate GLUE benchmark submission files from optimized configurations"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config with GELU/Softmax configurations per task")
    parser.add_argument("--noise_config", type=str, default=None,
                        help="Path to JSON config with noise scaling factor configurations per task")
    parser.add_argument("--blb_action_config", type=str, default=None,
                        help="Path to BLB action JSON (slot-form schema_version=blb_v3_slots_human_v1, "
                             "or any schema accepted by Paean.action_grid.load_action_grid_config). "
                             "When set, the script switches to the BLB Stage-2 submission path: the "
                             "task chosen via --blb_task is run with the decoded BLB action installed "
                             "(via BLBNoiseRLBridge); every other GLUE task runs the textattack baseline "
                             "(original GELU + exp, no noise).")
    parser.add_argument("--blb_task", type=str, default=None,
                        help="Which GLUE task the BLB action was trained for (e.g. mrpc). "
                             "Required when --blb_action_config is set.")
    parser.add_argument("--blb_seed", type=int, default=42,
                        help="Random seed for reproducible BLB-noise sampling (default: 42).")
    parser.add_argument("--output_dir", type=str, default="run",
                        help="Output sub-directory name; final path will be "
                             "glue_submission/<output_dir> (default: run)")
    parser.add_argument("--tasks", type=str, nargs='+', default=None,
                        help="Specific tasks to run (default: all tasks in config)")
    parser.add_argument("--no_approx", action="store_true",
                        help="Skip GELU/Softmax approximation, use original functions (baseline)")
    parser.add_argument("--no_noise", action="store_true",
                        help="Explicitly disable noise injection (default if --noise_config not given)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for inference: auto (default, prefers cuda), "
                             "cuda, cuda:N, or cpu. CPU fallback only if no GPU.")
    parser.add_argument("--allow_cpu", action="store_true",
                        help="Permit silent fallback to CPU when CUDA is unavailable. "
                             "Without this flag the script aborts if --device is cuda* "
                             "and no GPU is visible (recommended for GLUE inference).")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max sequence length (default: 128)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    parser.add_argument("--model_type", type=str, default="bert-base",
                        choices=["bert-base", "bert-large", "gpt-2"],
                        help="Pretrained backbone to use for submission: "
                             "bert-base (textattack/bert-base-uncased-*, all GLUE tasks); "
                             "bert-large (yoshitomo-matsubara/bert-large-uncased-*, "
                             "supports cola/sst2/mrpc/stsb/qnli/rte only; mnli/wnli/ax/qqp "
                             "will be skipped or filled with placeholders); "
                             "gpt-2 (openai-community/gpt2, 12 layers, uses freshly "
                             "initialized classification head — fine-tune before use).")
    args = parser.parse_args()

    # BLB action path (new): switches the whole submission to the BLB Stage-2
    # noise pipeline for the named task, runs every other task on baseline.
    if args.blb_action_config is not None:
        if args.blb_task is None:
            parser.error("--blb_task is required when --blb_action_config is set")
        if args.config is not None or args.noise_config is not None:
            parser.error(
                "--blb_action_config is mutually exclusive with --config / --noise_config "
                "(the BLB pipeline derives its own GELU/Softmax degrees from the JSON)."
            )
        normalized_output_dir = os.path.normpath(args.output_dir)
        already_under_glue_submission = (
            normalized_output_dir == "glue_submission"
            or normalized_output_dir.startswith(f"glue_submission{os.sep}")
        )
        if os.path.isabs(args.output_dir) or already_under_glue_submission:
            final_output_dir = normalized_output_dir
        else:
            final_output_dir = os.path.join("glue_submission", normalized_output_dir)
        summary = generate_blb_glue_submission(
            action_config_path=args.blb_action_config,
            blb_task=args.blb_task,
            model_type=args.model_type,
            output_dir=final_output_dir,
            seed=int(args.blb_seed),
            device=args.device,
            allow_cpu=bool(args.allow_cpu),
            max_length=int(args.max_length),
            batch_size=int(args.batch_size),
        )
        print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
        if summary.get("failures"):
            print(f"\nFailures: {summary['failures']}")
            sys.exit(1)
        return

    if not args.no_approx and args.config is None:
        parser.error("--config is required when not using --no_approx")

    if args.no_noise and args.noise_config is not None:
        parser.error("--no_noise and --noise_config are mutually exclusive")

    use_noise = args.noise_config is not None and not args.no_noise

    # GPU-first device resolution.
    requested = args.device.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"[Device] auto -> cuda ({torch.cuda.get_device_name(0)})")
        elif args.allow_cpu:
            device = "cpu"
            print("[Device] auto -> cpu (CUDA unavailable, --allow_cpu set)")
        else:
            print("[Error] CUDA is not available and --allow_cpu was not set. "
                  "GLUE inference on CPU is extremely slow; aborting. "
                  "Re-run with --allow_cpu to override.")
            sys.exit(1)
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            if args.allow_cpu:
                print(f"[Warning] {requested} requested but CUDA unavailable; "
                      f"falling back to CPU (--allow_cpu).")
                device = "cpu"
            else:
                print(f"[Error] {requested} requested but CUDA unavailable. "
                      f"Re-run with --allow_cpu to fall back to CPU.")
                sys.exit(1)
        else:
            device = requested
            try:
                print(f"[Device] using {device} ({torch.cuda.get_device_name(device)})")
            except Exception:
                print(f"[Device] using {device}")
    else:
        device = requested
        print(f"[Device] using {device}")

    model_type = args.model_type

    approx_configs = {}
    if args.config is not None:
        with open(args.config, 'r') as f:
            raw = json.load(f)
        raw.pop("_comment", None)
        approx_configs = _unwrap_variant_config(raw, model_type, args.config, stage_key="stage1")
        approx_configs.pop("_comment", None)

    noise_configs = {}
    if use_noise:
        with open(args.noise_config, 'r') as f:
            raw = json.load(f)
        raw.pop("_comment", None)
        noise_configs = _unwrap_variant_config(raw, model_type, args.noise_config, stage_key="stage2")
        noise_configs.pop("_comment", None)

    # All outputs are rooted under ./glue_submission/<sub>
    normalized_output_dir = os.path.normpath(args.output_dir)
    already_under_glue_submission = (
        normalized_output_dir == "glue_submission"
        or normalized_output_dir.startswith(f"glue_submission{os.sep}")
    )
    if os.path.isabs(args.output_dir) or already_under_glue_submission:
        final_output_dir = normalized_output_dir
    else:
        final_output_dir = os.path.join("glue_submission", normalized_output_dir)
    args.output_dir = final_output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    remove_stale_submission_files(args.output_dir)

    if args.tasks:
        tasks_to_run = args.tasks
    else:
        candidate_tasks = set()
        if approx_configs:
            candidate_tasks.update(t for t in approx_configs if t in TASK_REGISTRY)
        if noise_configs:
            candidate_tasks.update(t for t in noise_configs if t in TASK_REGISTRY)
        if 'mnli' in candidate_tasks:
            candidate_tasks.add('ax')
        if candidate_tasks:
            tasks_to_run = sorted(candidate_tasks)
        else:
            tasks_to_run = list(TASK_REGISTRY.keys())

    approx_str = "OFF (baseline)" if args.no_approx else f"ON (config: {args.config})"
    noise_str = f"ON (config: {args.noise_config})" if use_noise else "OFF"

    print(f"Model type:        {model_type}")
    print(f"Approximation:     {approx_str}")
    print(f"Noise injection:   {noise_str}")
    print(f"Tasks to process:  {tasks_to_run}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Device:            {device}")

    task_failures = []
    task_skips = []

    for task_name in tasks_to_run:
        if task_name not in TASK_REGISTRY:
            print(f"\n[Warning] Unknown task '{task_name}', skipping")
            task_skips.append((task_name, "unknown_task"))
            continue

        task_cfg = TASK_REGISTRY[task_name]

        # Swap in the bert-large checkpoint when requested; skip tasks without
        # a reliable bert-large checkpoint (mnli/wnli/ax — AX is generated from
        # the MNLI model which is unavailable for bert-large).
        if model_type == "bert-large":
            if task_name not in BERT_LARGE_MODEL_NAMES:
                print(f"\n[Warning] Task '{task_name}' has no bert-large checkpoint, "
                      f"skipping (will fall back to placeholder if applicable).")
                task_skips.append((task_name, "unsupported_model_checkpoint"))
                continue
            task_cfg = dict(task_cfg)
            task_cfg['model_name'] = BERT_LARGE_MODEL_NAMES[task_name]
        elif model_type == "gpt-2":
            if task_name not in GPT2_MODEL_NAMES:
                print(f"\n[Warning] Task '{task_name}' has no gpt-2 checkpoint, skipping.")
                task_skips.append((task_name, "unsupported_model_checkpoint"))
                continue
            task_cfg = dict(task_cfg)
            task_cfg['model_name'] = GPT2_MODEL_NAMES[task_name]

        if args.no_approx:
            gelu = None
            softmax = None
        else:
            approx_task_name = task_name
            if approx_task_name == 'ax' and approx_task_name not in approx_configs and 'mnli' in approx_configs:
                approx_task_name = 'mnli'
            if approx_task_name not in approx_configs:
                print(f"\n[Warning] No approx config for task '{task_name}' in {args.config}, skipping")
                task_skips.append((task_name, "missing_approx_config"))
                continue
            gelu = approx_configs[approx_task_name]['gelu']
            softmax = approx_configs[approx_task_name]['softmax']

        if use_noise:
            noise_task_name = task_name
            if noise_task_name == 'ax' and noise_task_name not in noise_configs and 'mnli' in noise_configs:
                noise_task_name = 'mnli'
            if noise_task_name not in noise_configs:
                print(f"\n[Warning] No noise config for task '{task_name}' in {args.noise_config}, skipping")
                task_skips.append((task_name, "missing_noise_config"))
                continue
            task_noise = noise_configs[noise_task_name]
            missing_keys = [k for k in NOISE_KEYS if k not in task_noise]
            if missing_keys:
                print(f"\n[Error] Noise config for '{task_name}' missing keys: {missing_keys}")
                task_skips.append((task_name, f"missing_noise_keys:{','.join(missing_keys)}"))
                continue
        else:
            task_noise = None
 
        try:
            process_task(
                task_name, task_cfg, gelu, softmax, task_noise,
                args.output_dir, device, args.max_length, args.batch_size,
                no_approx=args.no_approx, no_noise=not use_noise,
            )
        except Exception as exc:
            task_failures.append((task_name, type(exc).__name__, str(exc)))
            print(f"\n[Error] Task '{task_name}' failed; continuing with remaining tasks.")
            print(f"        {type(exc).__name__}: {exc}")
            exc.__traceback__ = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("Generating placeholder files")
    print(f"{'=' * 60}")
    placeholder_files = fill_missing_submission_files(args.output_dir)

    all_ok = verify_outputs(args.output_dir)
    create_submission_zip(args.output_dir)

    if placeholder_files:
        print("\nPlaceholder files created:")
        for filename in sorted(placeholder_files):
            print(f"  - {filename}")
        all_ok = False

    if task_skips:
        print("\nTasks skipped before inference:")
        for task_name, reason in task_skips:
            print(f"  - {task_name}: {reason}")
        all_ok = False

    if task_failures:
        print("\nTask failures encountered during generation:")
        for task_name, exc_type, message in task_failures:
            print(f"  - {task_name}: {exc_type}: {message}")
        all_ok = False

    if all_ok:
        print("\nAll checks passed. Ready to submit to https://gluebenchmark.com/")
    else:
        print("\nSome checks failed or placeholders were used. "
              "Please verify the output files before submitting.")


def _seed_all_for_reproducibility(seed: int) -> None:
    import random as _random

    seed = int(seed) & 0xFFFFFFFF
    _random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _decode_blb_action_for_glue(
        *,
        action_vec,
        fusion_metadata,
        profile: str,
        gelu_degrees,
        softmax_degrees,
        max_sfs,
        ):
    """Decode a Stage-2 action into installable BLB cfgs for GLUE submission,
    REPLAYING the precision boost and applying the Rescale_optimizer override.

    This is the parity guarantee: the flat ``action_vec`` cannot carry the boost
    (above-grid SFs live in the option's ``explicit_field_values``), so for a
    fusion run we decode the ``fusion_count_fixed_action_v1`` metadata via the SAME
    methods the validation-set final eval uses (``BLBActionFinalEvaluationModule``),
    then apply ``apply_optimizer_output_to_cfg`` (fused rescales → None) the same
    way. Reusing the methods (via a lightweight shim, the pattern the unit tests
    use) guarantees byte-for-byte agreement with final eval + the training probe —
    no second, drift-prone copy of the install pipeline.

    ``fusion_metadata is None`` ⇒ legacy per-slot / non-fusion path (index decode,
    no override), preserved for back-compat.
    """
    import numpy as _np

    if fusion_metadata is None:
        from blb_stage2_rl.action_space import action_vector_to_cfgs as _avc
        return _avc(
            action_vec=_np.asarray(action_vec, dtype=int),
            max_sfs=max_sfs,
            num_layers=int(len(gelu_degrees)),
            gelu_degree=_np.asarray(gelu_degrees, dtype=int),
            attn_degree=_np.asarray(softmax_degrees, dtype=int),
        )

    from Paean.blb_action_eval import BLBActionFinalEvaluationModule
    from rescale_optimizer_bridge import InProcessInvoker, RescaleOptimizerBridge

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Rescale_optimizer")
    invoker = InProcessInvoker.from_profile(rescale_optimizer_root=root, profile=str(profile))
    bridge = RescaleOptimizerBridge(invoker=invoker)

    # Lightweight shim so the final-eval methods run without a full evaluator:
    # decode uses no self-state; the override loop needs only rescale_bridge and
    # the rotation-name-map resolver (evaluator=None ⇒ {} rotation map, matching a
    # final-eval evaluator that has no rotation map configured).
    shim = BLBActionFinalEvaluationModule.__new__(BLBActionFinalEvaluationModule)
    shim.evaluator = None
    shim.rescale_bridge = bridge
    shim.rescale_invoker_kind = "in_process"
    shim.rescale_optimizer_root = root

    num_layers = int(len(gelu_degrees))
    decoded = shim._decode_action_candidate(
        action_vec=_np.asarray(action_vec, dtype=int),
        metadata=fusion_metadata,
        max_sfs=max_sfs,
        num_layers=num_layers,
        gelu=_np.asarray(gelu_degrees, dtype=int),
        softmax=_np.asarray(softmax_degrees, dtype=int),
        profile=str(profile),
    )
    cfgs_dict = decoded.cfgs_dict()
    opt_outputs, _opt_signals = shim._optimizer_outputs(str(profile), cfgs_dict)
    shim._apply_optimizer_outputs_to_decoded(
        profile=str(profile),
        decoded=decoded,
        cfgs_dict=cfgs_dict,
        opt_outputs=opt_outputs,
    )
    return decoded


def _process_blb_task(
        *,
        task_name: str,
        task_config: dict,
        action_vec: np.ndarray,
        profile: str,
        gelu_degrees,
        softmax_degrees,
        output_dir: str,
        device,
        max_length: int = 128,
        batch_size: int = 16,
        fusion_metadata=None,
        max_sfs=None,
        ) -> None:
    """Mirror of ``process_task`` for BLB action vectors.

    Installs BLB Stage-2 noise via :class:`BLBNoiseRLBridge` after running
    ``Rescale_optimizer`` over the decoded cfgs. Uses ``apply_approx_configuration``
    for the GELU/Softmax polynomial degrees that came alongside the BLB action.
    """
    if _DATASETS_IMPORT_ERROR is not None:
        raise ImportError(
            "The 'datasets' package is required for GLUE submission generation."
        ) from _DATASETS_IMPORT_ERROR
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "The 'transformers' package is required for GLUE submission generation."
        ) from _TRANSFORMERS_IMPORT_ERROR

    from blb_rl_bridge import BLBNoiseRLBridge

    if max_sfs is None:
        raise ValueError(
            "BLB GLUE submission requires the calibrated Stage-2 max_sfs table"
        )

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name.upper()} (BLB action path)")
    print(f"  GELU:    {list(gelu_degrees)}")
    print(f"  Softmax: {list(softmax_degrees)}")
    print(f"  BLB profile: {profile}, action_length={int(np.asarray(action_vec).size)}")
    print(f"{'=' * 60}")

    model_name = task_config['model_name']
    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=task_config['num_labels'],
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )
    model.to(device)

    model_id2label = None
    if hasattr(model.config, 'id2label') and model.config.id2label:
        model_id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        print(f"  Model label mapping: {model_id2label}")
    else:
        print("  [Warning] model.config.id2label missing; falling back to TASK_REGISTRY label_map")

    handler = ReversibleLayerHandler(model)
    layers_attr = detect_layer_attribute(model)
    num_layers = len(eval('model.' + layers_attr))
    print(f"  Layers: {layers_attr} ({num_layers} layers)")

    # Stage 1 approximation (BLB action only encodes Stage-2 noise; GELU /
    # Softmax degrees come from the accompanying stage1 ladder).
    assert len(gelu_degrees) == num_layers, (
        f"GELU config length ({len(gelu_degrees)}) != model layers ({num_layers})"
    )
    assert len(softmax_degrees) == num_layers, (
        f"Softmax config length ({len(softmax_degrees)}) != model layers ({num_layers})"
    )
    apply_approx_configuration(handler, layers_attr, list(gelu_degrees), list(softmax_degrees))

    # Decode action → cfgs → BLBNoiseRLBridge.apply, REPLAYING the precision boost
    # and applying the Rescale_optimizer override (fused rescales → None) for a
    # fusion run, exactly like Paean.blb_action_eval's validation-set final eval and
    # the training terminal probe. The fusion path reuses the final-eval methods
    # (single source of truth) so the submitted config == the RL-selected config.
    # Non-fusion (fusion_metadata None) keeps the legacy index decode.
    decoded = _decode_blb_action_for_glue(
        action_vec=np.asarray(action_vec, dtype=int),
        fusion_metadata=fusion_metadata,
        profile=str(profile),
        gelu_degrees=gelu_degrees,
        softmax_degrees=softmax_degrees,
        max_sfs=max_sfs,
    )
    noise_bridge = BLBNoiseRLBridge(handler, layers_attribute="model." + layers_attr)
    noise_bridge.apply(
        block1_cfgs=decoded.block1_cfgs,
        block2_cfgs=decoded.block2_cfgs,
        block3_cfgs=decoded.block3_cfgs,
        block4_cfgs=decoded.block4_cfgs,
        block5_cfgs=decoded.block5_cfgs,
    )

    model.to(device)
    model.eval()

    print(f"  Loading GLUE test set: {task_config['glue_name']}")
    data = load_dataset("nyu-mll/glue", task_config['glue_name'])
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
    test_splits = task_config['test_split']
    output_files = task_config['output_file']
    try:
        if isinstance(test_splits, list):
            for split_name, out_file in zip(test_splits, output_files):
                print(f"  Processing split: {split_name}")
                test_data = tokenize_and_prepare(
                    data[split_name], tokenizer, task_config['input_cols'], max_length,
                )
                dataloader = DataLoader(
                    test_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator,
                )
                logits = run_inference(model, dataloader, device)
                predictions = logits_to_predictions(logits, task_config, task_name, model_id2label)
                write_tsv(os.path.join(output_dir, out_file), predictions)
        else:
            test_data = tokenize_and_prepare(
                data[test_splits], tokenizer, task_config['input_cols'], max_length,
            )
            dataloader = DataLoader(
                test_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator,
            )
            logits = run_inference(model, dataloader, device)
            predictions = logits_to_predictions(logits, task_config, task_name, model_id2label)
            write_tsv(os.path.join(output_dir, output_files), predictions)
    finally:
        try:
            noise_bridge.clear()
        except Exception:
            pass
        try:
            handler.backup_model = None
        except Exception:
            pass
        del handler
        model.to('cpu')
        del model
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_blb_glue_submission(
        *,
        action_config_path: str,
        blb_task: str,
        model_type: str = "bert-base",
        output_dir: str = "glue_submission",
        seed: int = 42,
        profile: str = "",
        gelu_degree=None,
        softmax_degree=None,
        device: str = "auto",
        allow_cpu: bool = False,
        max_length: int = 128,
        batch_size: int = 16,
        log_fn=None,
        calibrated_action_context=None,
        ) -> dict:
    """Generate a GLUE submission zip for a BLB Stage-2 action.

    The ``blb_task`` task uses the BLB-decoded action installed via
    :class:`BLBNoiseRLBridge`. Every other task runs the textattack baseline
    (original GELU + exp, no noise) so the submission zip is complete.
    """
    log_fn = log_fn or print
    if not os.path.isfile(action_config_path):
        raise FileNotFoundError(f"BLB action config not found: {action_config_path}")
    payload = read_json_file(action_config_path, encoding="utf-8-sig")

    # 加大精度 handoff: a fusion_count_fixed_action_v1 config carries the per-step
    # fusion (option, K) selection. Pass it through so the BLB task replays the
    # boosted config + Rescale_optimizer override (parity with final eval). A flat
    # slot/index config (legacy / per-slot run) leaves this None.
    fusion_metadata = None
    if str(payload.get("schema_version", "")) == "fusion_count_fixed_action_v1":
        _group = payload.get("group")
        if isinstance(_group, dict):
            fusion_metadata = {
                "schema_version": "fusion_count_fixed_action_v1",
                "group": _group,
            }
            log_fn("[BLB GLUE] fusion_count_fixed_action_v1 detected → replaying boosted config")

    # Resolve effective profile + Stage 1 degrees from caller args; fall back
    # to the JSON file if the caller did not supply them.
    profile = str(profile or payload.get("profile") or blb_task)
    gelu_list = list(gelu_degree) if gelu_degree is not None else (
        list(payload.get("gelu_degree") or [])
    )
    softmax_list = list(softmax_degree) if softmax_degree is not None else (
        list(payload.get("attn_degree") or payload.get("softmax_degree") or [])
    )
    if not gelu_list or not softmax_list:
        raise ValueError(
            "BLB GLUE submission requires Stage-1 GELU + Softmax degrees. "
            "Either pass gelu_degree/softmax_degree or include them under "
            "'gelu_degree'/'attn_degree' in the action config JSON."
        )

    # Resolve the BLB action vec via the canonical loader (handles slot-form,
    # base+overrides, and flat action_vec schemas).
    from Paean.action_grid import load_action_grid_config
    num_layers_hint = int(payload.get("num_layers") or len(gelu_list))
    from blb_stage2_rl.baseline_bootstrap import (
        load_calibrated_stage2_action_context,
        validate_calibrated_stage2_action_context,
    )
    action_context = calibrated_action_context
    if action_context is None:
        action_context = load_calibrated_stage2_action_context(
            rescale_optimizer_root=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Rescale_optimizer",
            ),
            dataset=str(profile),
            num_layers=num_layers_hint,
            gelu_per_layer=gelu_list,
            softmax_per_layer=softmax_list,
            snap_sf_to_noise_table=False,
        )
    validate_calibrated_stage2_action_context(
        action_context,
        dataset=str(profile),
        num_layers=num_layers_hint,
        gelu_per_layer=gelu_list,
        softmax_per_layer=softmax_list,
        snap_sf_to_noise_table=False,
    )
    embedded_context = payload.get("calibrated_action_context")
    if isinstance(embedded_context, dict):
        comparable_keys = (
            "schema_version",
            "dataset",
            "num_layers",
            "gelu_per_layer",
            "softmax_per_layer",
            "archive_sha256",
            "snap_sf_to_noise_table",
        )
        provenance = dict(action_context.provenance)
        mismatches = {
            key: {
                "action_config": embedded_context.get(key),
                "current_context": provenance.get(key),
            }
            for key in comparable_keys
            if embedded_context.get(key) != provenance.get(key)
        }
        if mismatches:
            raise ValueError(
                "BLB action config calibrated context mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
    grid_cfg = load_action_grid_config(
        action_config_path,
        num_layers_hint=num_layers_hint,
        profile=str(profile),
        gelu_degree=gelu_list,
        attn_degree=softmax_list,
        max_sfs=action_context.max_sfs,
    )
    if grid_cfg.base_action_vec is None:
        raise ValueError(
            "BLB GLUE submission requires a concrete action vector in the JSON "
            "(slots/base+overrides/action_vec). Got config without a base."
        )
    if isinstance(grid_cfg.base_action_vec, str):
        from blb_stage2_rl.action_space import (
            make_all_max_action_vector as _all_max,
            make_all_min_action_vector as _all_min,
        )
        text = grid_cfg.base_action_vec.lower()
        if text in ("max", "all-max", "all_max", "blb-baseline", "blb_baseline",
                    "rescale-baseline", "rescale_baseline"):
            action_vec = _all_max(num_layers_hint).astype(int)
        elif text in ("min", "all-min", "all_min"):
            action_vec = _all_min(num_layers_hint).astype(int)
        else:
            raise ValueError(f"BLB GLUE: unsupported base action sentinel '{text}'")
    else:
        action_vec = np.asarray(grid_cfg.base_action_vec, dtype=int)

    # Resolve device with the same policy as ``main``: GPU-first, optional CPU.
    requested = str(device or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            device_resolved = "cuda"
        elif allow_cpu:
            device_resolved = "cpu"
            log_fn("[Device] auto -> cpu (CUDA unavailable, allow_cpu=True)")
        else:
            raise RuntimeError(
                "CUDA unavailable and allow_cpu=False; GLUE inference on CPU is "
                "extremely slow. Re-call with allow_cpu=True to override."
            )
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            if allow_cpu:
                log_fn(f"[Warning] {requested} requested but CUDA unavailable; falling back to CPU.")
                device_resolved = "cpu"
            else:
                raise RuntimeError(
                    f"{requested} requested but CUDA unavailable. Set allow_cpu=True to override."
                )
        else:
            device_resolved = requested
    else:
        device_resolved = requested

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    remove_stale_submission_files(output_dir)

    blb_task = str(blb_task).strip().lower()
    if blb_task not in TASK_REGISTRY:
        raise KeyError(f"Unknown BLB task '{blb_task}'; valid tasks: {sorted(TASK_REGISTRY)}")

    log_fn(f"[BLB GLUE] model_type={model_type}, blb_task={blb_task}, profile={profile}")
    log_fn(f"[BLB GLUE] output_dir={output_dir}, device={device_resolved}, seed={seed}")
    log_fn(f"[BLB GLUE] action_config={action_config_path}")

    _seed_all_for_reproducibility(int(seed))

    failures = []
    skips = []
    # 1) Inference on the BLB-trained task with the BLB action installed.
    blb_cfg = dict(TASK_REGISTRY[blb_task])
    if model_type == "bert-large" and blb_task in BERT_LARGE_MODEL_NAMES:
        blb_cfg['model_name'] = BERT_LARGE_MODEL_NAMES[blb_task]
    elif model_type == "gpt-2" and blb_task in GPT2_MODEL_NAMES:
        blb_cfg['model_name'] = GPT2_MODEL_NAMES[blb_task]
    try:
        _process_blb_task(
            task_name=blb_task,
            task_config=blb_cfg,
            action_vec=action_vec,
            profile=str(profile),
            gelu_degrees=gelu_list,
            softmax_degrees=softmax_list,
            output_dir=output_dir,
            device=device_resolved,
            max_length=int(max_length),
            batch_size=int(batch_size),
            fusion_metadata=fusion_metadata,
            max_sfs=action_context.max_sfs,
        )
    except Exception as exc:
        failures.append((blb_task, type(exc).__name__, str(exc)))
        log_fn(f"[Error] BLB task '{blb_task}' failed: {type(exc).__name__}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2) Baseline inference for every other GLUE task (original GELU + exp,
    # no noise). MNLI implies AX (AX uses the MNLI model on the diagnostic set).
    other_tasks = [
        name for name in sorted(TASK_REGISTRY.keys())
        if name != blb_task
    ]
    # de-dup mnli/ax pair: AX always uses the MNLI checkpoint, so we still need
    # to enumerate it separately for the test split.
    for task_name in other_tasks:
        task_cfg = dict(TASK_REGISTRY[task_name])
        if model_type == "bert-large":
            if task_name not in BERT_LARGE_MODEL_NAMES:
                log_fn(f"[Warning] '{task_name}' has no bert-large checkpoint; "
                       "will be filled with placeholders.")
                skips.append((task_name, "unsupported_model_checkpoint"))
                continue
            task_cfg['model_name'] = BERT_LARGE_MODEL_NAMES[task_name]
        elif model_type == "gpt-2":
            if task_name not in GPT2_MODEL_NAMES:
                log_fn(f"[Warning] '{task_name}' has no gpt-2 checkpoint; will be filled with placeholders.")
                skips.append((task_name, "unsupported_model_checkpoint"))
                continue
            task_cfg['model_name'] = GPT2_MODEL_NAMES[task_name]
        try:
            process_task(
                task_name, task_cfg, gelu_degrees=None, softmax_degrees=None,
                noise_config=None, output_dir=output_dir, device=device_resolved,
                max_length=int(max_length), batch_size=int(batch_size),
                no_approx=True, no_noise=True,
            )
        except Exception as exc:
            failures.append((task_name, type(exc).__name__, str(exc)))
            log_fn(f"[Error] Baseline task '{task_name}' failed: {type(exc).__name__}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    placeholder_files = fill_missing_submission_files(output_dir)
    ok = verify_outputs(output_dir)
    zip_path = create_submission_zip(output_dir)

    summary = {
        "zip_path": zip_path,
        "output_dir": output_dir,
        "blb_task": blb_task,
        "model_type": model_type,
        "profile": str(profile),
        "seed": int(seed),
        "failures": [
            {"task": name, "exc_type": etype, "message": msg}
            for name, etype, msg in failures
        ],
        "skipped": [{"task": name, "reason": reason} for name, reason in skips],
        "placeholder_files": list(sorted(placeholder_files or [])),
        "verify_ok": bool(ok),
        "calibrated_action_context": dict(action_context.provenance),
    }
    return summary


if __name__ == "__main__":
    main()
