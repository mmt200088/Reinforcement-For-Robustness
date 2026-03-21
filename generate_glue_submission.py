#!/usr/bin/env python
"""
Generate GLUE benchmark submission files from optimized GELU/Softmax configurations.

Usage:
    # With approximation (from config)
    python generate_glue_submission.py --config glue_configs.json --output_dir glue_submission
    python generate_glue_submission.py --config glue_configs.json --tasks qnli sst2

    # Without approximation (original GELU + exp, baseline)
    python generate_glue_submission.py --config glue_configs.json --no_approx --output_dir glue_baseline
    python generate_glue_submission.py --no_approx --tasks qnli sst2 --output_dir glue_baseline
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
from function_handler import ReversibleLayerHandler

sys.setrecursionlimit(50000)

# ==================== GLUE Task Registry ====================
# Each task: model name, GLUE dataset key, num_labels, input columns,
# output filename(s), label mapping for submission, test split name(s).
#
# Label maps follow GLUE submission conventions:
#   MNLI/AX: entailment / neutral / contradiction
#   QNLI/RTE: entailment / not_entailment
#   CoLA/SST-2/MRPC/WNLI/QQP: 0 / 1
#   STS-B: float in [0, 5]

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


def apply_configuration(handler, layers_attribute, gelu_degrees, softmax_degrees):
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
                 output_dir, device, max_length=128, batch_size=16,
                 no_approx=False):
    print(f"\n{'=' * 60}")
    print(f"Task: {task_name.upper()}")
    if no_approx:
        print(f"  Mode:    NO APPROXIMATION (original GELU + exp)")
    else:
        print(f"  GELU:    {gelu_degrees}")
        print(f"  Softmax: {softmax_degrees}")
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

    if not no_approx:
        handler = ReversibleLayerHandler(model)
        layers_attr = detect_layer_attribute(model)
        num_layers = len(eval('model.' + layers_attr))
        print(f"  Layers: {layers_attr} ({num_layers} layers)")

        assert len(gelu_degrees) == num_layers, \
            f"GELU config length ({len(gelu_degrees)}) != model layers ({num_layers})"
        assert len(softmax_degrees) == num_layers, \
            f"Softmax config length ({len(softmax_degrees)}) != model layers ({num_layers})"

        apply_configuration(handler, layers_attr, gelu_degrees, softmax_degrees)
    else:
        handler = None
        print(f"  Skipping approximation, using original model")

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
    parser.add_argument("--output_dir", type=str, default="glue_submission",
                        help="Output directory for TSV files (default: glue_submission)")
    parser.add_argument("--tasks", type=str, nargs='+', default=None,
                        help="Specific tasks to run (default: all tasks in config)")
    parser.add_argument("--no_approx", action="store_true",
                        help="Skip approximation, use original GELU + exp (baseline)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (default: cuda)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max sequence length (default: 128)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    args = parser.parse_args()

    if not args.no_approx and args.config is None:
        parser.error("--config is required when not using --no_approx")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    configs = {}
    if args.config is not None:
        with open(args.config, 'r') as f:
            configs = json.load(f)
        configs.pop("_comment", None)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.tasks:
        tasks_to_run = args.tasks
    elif configs:
        tasks_to_run = [t for t in configs if t in TASK_REGISTRY]
    else:
        tasks_to_run = list(TASK_REGISTRY.keys())

    mode_str = "NO APPROXIMATION (baseline)" if args.no_approx else "WITH APPROXIMATION"
    print(f"Mode:              {mode_str}")
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
            if task_name not in configs:
                print(f"\n[Warning] No config for task '{task_name}' in {args.config}, skipping")
                continue
            gelu = configs[task_name]['gelu']
            softmax = configs[task_name]['softmax']

        process_task(
            task_name, task_cfg, gelu, softmax,
            args.output_dir, device, args.max_length, args.batch_size,
            no_approx=args.no_approx,
        )

    # Generate placeholder files
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
