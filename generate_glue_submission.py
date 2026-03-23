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

▶ GELU/Softmax 近似配置 (--config):
  每个任务包含 "gelu" 和 "softmax" 两个数组，长度 = 模型层数 (BERT-base: 12)。
  GELU degree 取值 {1, 2, 4}；Softmax degree 取值 {2, 3, 4, 5, 6}。

  示例 (glue_configs_best_ppo.json):
  {
      "qnli": {
          "gelu":    [1, 1, 1, 1, 2, 4, 4, 4, 4, 1, 1, 1],
          "softmax": [2, 3, 4, 4, 3, 2, 2, 4, 3, 5, 5, 5]
      },
      ...
  }

▶ 噪声 Scaling Factor 配置 (--noise_config):
  每个任务包含 7 个数组 (x, wq, wk, wv, wo, wffn1, wffn2)，长度 = 模型层数。
  x 取值 {20, 22, 24, 26, 28, 30}；wq/wk/wv/wo/wffn1/wffn2 取值 {10, 12, 14, 16, 18, 20, 22}。
  数值越大 → 噪声越大 → 隐私保护越强。

  示例 (glue_noise_configs_best_ppo.json):
  {
      "qnli": {
          "x":     [30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
          "wq":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
          "wk":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
          "wv":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
          "wo":    [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
          "wffn1": [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22],
          "wffn2": [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22]
      },
      ...
  }

================================================================================
使用示例
================================================================================

# 1) 纯基线：原始模型，无近似无噪声
python generate_glue_submission.py --no_approx --output_dir glue_baseline

# 2) 仅 GELU/Softmax 近似（无噪声）
python generate_glue_submission.py --config glue_configs_best_ppo.json --output_dir glue_approx

# 3) 仅噪声注入（无近似）
python generate_glue_submission.py --no_approx --noise_config glue_noise_configs_best_ppo.json --output_dir glue_noise_only

# 4) 近似 + 噪声（完整两阶段优化）
python generate_glue_submission.py --config glue_configs_best_ppo.json --noise_config glue_noise_configs_best_ppo.json --output_dir glue_full

# 5) 指定部分任务
python generate_glue_submission.py --config glue_configs_best_ppo.json --noise_config glue_noise_configs_best_ppo.json --tasks qnli sst2 mrpc

# 6) 近似 + 最高 scaling factor 噪声（使用全部最大值的噪声配置文件）
python generate_glue_submission.py --config glue_configs_best_ppo.json --noise_config glue_noise_configs_best_ppo.json --output_dir glue_max_noise

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
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from function_handler import (
    ReversibleLayerHandler,
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
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
}

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
    gelu_map = {d: [] for d in [1, 2, 4]}
    for idx, deg in enumerate(gelu_degrees):
        if deg in gelu_map:
            gelu_map[deg].append(idx)
    for d in [1, 2, 4]:
        if gelu_map[d]:
            handler.replace_layer_gelu(gelu_map[d], handler_layer_name, degree=d)

    softmax_map = {d: [] for d in range(2, 7)}
    for idx, deg in enumerate(softmax_degrees):
        if deg in softmax_map:
            softmax_map[deg].append(idx)
    for d in range(2, 7):
        if softmax_map[d]:
            handler.replace_layer_softmax(softmax_map[d], handler_layer_name, degree=d)


def apply_noise_configuration(handler, layers_attribute, noise_config):
    handler_layer_name = "model." + layers_attribute

    for noise_key in NOISE_KEYS:
        method_name, distribution = NOISE_HANDLER_MAP[noise_key]
        factors = noise_config[noise_key]

        allowed = (INPUT_NOISE_ALLOWED_SCALING_FACTORS if noise_key == 'x'
                   else WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)

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


def run_inference(model, dataloader, device):
    model.eval()
    model.to(device)
    all_logits = []

    if torch.cuda.is_available():
        dummy = next(iter(dataloader))
        dummy = {k: v.to(device) for k, v in dummy.items()}
        with torch.no_grad():
            _ = model(**dummy)
        torch.cuda.synchronize()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


def logits_to_predictions(logits, task_config):
    if task_config['task_type'] == 'regression':
        preds = logits.squeeze()
        if np.ndim(preds) == 0:
            preds = np.array([preds])
        return [f"{np.clip(p, 0.0, 5.0):.3f}" for p in preds]
    else:
        if len(logits.shape) == 1:
            pred_classes = (logits > 0.5).astype(int)
        else:
            pred_classes = np.argmax(logits, axis=1)
        label_map = task_config['label_map']
        return [str(label_map[int(c)]) for c in pred_classes]


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

    if hasattr(model.config, 'id2label'):
        print(f"  Model label mapping: {model.config.id2label}")

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
            predictions = logits_to_predictions(logits, task_config)
            write_tsv(os.path.join(output_dir, out_file), predictions)
    else:
        test_data = tokenize_and_prepare(
            data[test_splits], tokenizer, task_config['input_cols'], max_length
        )
        dataloader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )
        logits = run_inference(model, dataloader, device)
        predictions = logits_to_predictions(logits, task_config)
        write_tsv(os.path.join(output_dir, output_files), predictions)

    del model
    if handler is not None:
        del handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_outputs(output_dir):
    print(f"\n{'=' * 60}")
    print("Verification: TSV line counts")
    print(f"{'=' * 60}")
    all_ok = True
    for filename, expected in sorted(EXPECTED_LINES.items()):
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                actual = sum(1 for _ in f)
            if actual == expected:
                status = "OK"
            else:
                status = f"MISMATCH (expected {expected})"
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate GLUE benchmark submission files from optimized configurations"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config with GELU/Softmax configurations per task")
    parser.add_argument("--noise_config", type=str, default=None,
                        help="Path to JSON config with noise scaling factor configurations per task")
    parser.add_argument("--output_dir", type=str, default="glue_submission",
                        help="Output directory for TSV files (default: glue_submission)")
    parser.add_argument("--tasks", type=str, nargs='+', default=None,
                        help="Specific tasks to run (default: all tasks in config)")
    parser.add_argument("--no_approx", action="store_true",
                        help="Skip GELU/Softmax approximation, use original functions (baseline)")
    parser.add_argument("--no_noise", action="store_true",
                        help="Explicitly disable noise injection (default if --noise_config not given)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max sequence length (default: 128)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    args = parser.parse_args()

    if not args.no_approx and args.config is None:
        parser.error("--config is required when not using --no_approx")

    if args.no_noise and args.noise_config is not None:
        parser.error("--no_noise and --noise_config are mutually exclusive")

    use_noise = args.noise_config is not None and not args.no_noise

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    approx_configs = {}
    if args.config is not None:
        with open(args.config, 'r') as f:
            approx_configs = json.load(f)
        approx_configs.pop("_comment", None)

    noise_configs = {}
    if use_noise:
        with open(args.noise_config, 'r') as f:
            noise_configs = json.load(f)
        noise_configs.pop("_comment", None)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tasks:
        tasks_to_run = args.tasks
    else:
        candidate_tasks = set()
        if approx_configs:
            candidate_tasks.update(t for t in approx_configs if t in TASK_REGISTRY)
        if noise_configs:
            candidate_tasks.update(t for t in noise_configs if t in TASK_REGISTRY)
        if candidate_tasks:
            tasks_to_run = sorted(candidate_tasks)
        else:
            tasks_to_run = list(TASK_REGISTRY.keys())

    approx_str = "OFF (baseline)" if args.no_approx else f"ON (config: {args.config})"
    noise_str = f"ON (config: {args.noise_config})" if use_noise else "OFF"

    print(f"Approximation:     {approx_str}")
    print(f"Noise injection:   {noise_str}")
    print(f"Tasks to process:  {tasks_to_run}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Device:            {device}")

    for task_name in tasks_to_run:
        if task_name not in TASK_REGISTRY:
            print(f"\n[Warning] Unknown task '{task_name}', skipping")
            continue

        task_cfg = TASK_REGISTRY[task_name]

        if args.no_approx:
            gelu = None
            softmax = None
        else:
            if task_name not in approx_configs:
                print(f"\n[Warning] No approx config for task '{task_name}' in {args.config}, skipping")
                continue
            gelu = approx_configs[task_name]['gelu']
            softmax = approx_configs[task_name]['softmax']

        if use_noise:
            if task_name not in noise_configs:
                print(f"\n[Warning] No noise config for task '{task_name}' in {args.noise_config}, skipping")
                continue
            task_noise = noise_configs[task_name]
            missing_keys = [k for k in NOISE_KEYS if k not in task_noise]
            if missing_keys:
                print(f"\n[Error] Noise config for '{task_name}' missing keys: {missing_keys}")
                continue
        else:
            task_noise = None

        process_task(
            task_name, task_cfg, gelu, softmax, task_noise,
            args.output_dir, device, args.max_length, args.batch_size,
            no_approx=args.no_approx, no_noise=not use_noise,
        )

    print(f"\n{'=' * 60}")
    print("Generating placeholder files (QQP, AX)")
    print(f"{'=' * 60}")

    qqp_path = os.path.join(args.output_dir, "QQP.tsv")
    if not os.path.exists(qqp_path):
        generate_placeholder(qqp_path, EXPECTED_LINES['QQP.tsv'] - 1, default_label="0")

    ax_path = os.path.join(args.output_dir, "AX.tsv")
    if not os.path.exists(ax_path):
        generate_placeholder(ax_path, EXPECTED_LINES['AX.tsv'] - 1, default_label="entailment")

    all_ok = verify_outputs(args.output_dir)
    create_submission_zip(args.output_dir)

    if all_ok:
        print("\nAll checks passed. Ready to submit to https://gluebenchmark.com/")
    else:
        print("\nSome checks failed. Please verify the output files before submitting.")


if __name__ == "__main__":
    main()
