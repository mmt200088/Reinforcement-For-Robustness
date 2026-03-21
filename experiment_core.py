"""
Shared evaluation core for Transformer approximation error non-accumulation experiments.

Provides model loading, config application, validation-set evaluation, and Bootstrap
resampling utilities used by all experiment scripts.
"""

import sys
import copy
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from function_handler import ReversibleLayerHandler

sys.setrecursionlimit(50000)

NUM_LAYERS = 12
GELU_FULL = 4
GELU_LOW = 1
SOFTMAX_FULL = 6
SOFTMAX_LOW = 2

BASELINE_CONFIG = {"gelu": [GELU_FULL] * NUM_LAYERS, "softmax": [SOFTMAX_FULL] * NUM_LAYERS}

TASK_REGISTRY = {
    'cola': {
        'model_name': 'textattack/bert-base-uncased-CoLA',
        'glue_name': 'cola',
        'num_labels': 2,
        'input_cols': ('sentence',),
        'task_type': 'classification',
        'val_split': 'validation',
        'primary_metric': 'mcc',
        'all_metrics': ['mcc'],
        'metric_names': ['MCC'],
    },
    'sst2': {
        'model_name': 'textattack/bert-base-uncased-SST-2',
        'glue_name': 'sst2',
        'num_labels': 2,
        'input_cols': ('sentence',),
        'task_type': 'classification',
        'val_split': 'validation',
        'primary_metric': 'accuracy',
        'all_metrics': ['accuracy'],
        'metric_names': ['Accuracy'],
    },
    'mrpc': {
        'model_name': 'textattack/bert-base-uncased-MRPC',
        'glue_name': 'mrpc',
        'num_labels': 2,
        'input_cols': ('sentence1', 'sentence2'),
        'task_type': 'classification',
        'val_split': 'validation',
        'primary_metric': 'accuracy',
        'all_metrics': ['accuracy', 'f1'],
        'metric_names': ['Accuracy', 'F1'],
    },
    'stsb': {
        'model_name': 'textattack/bert-base-uncased-STS-B',
        'glue_name': 'stsb',
        'num_labels': 1,
        'input_cols': ('sentence1', 'sentence2'),
        'task_type': 'regression',
        'val_split': 'validation',
        'primary_metric': 'pearson',
        'all_metrics': ['pearson', 'spearman'],
        'metric_names': ['Pearson', 'Spearman'],
    },
    'mnli': {
        'model_name': 'textattack/bert-base-uncased-MNLI',
        'glue_name': 'mnli',
        'num_labels': 3,
        'input_cols': ('premise', 'hypothesis'),
        'task_type': 'classification',
        'val_split': 'validation_matched',
        'primary_metric': 'accuracy',
        'all_metrics': ['accuracy'],
        'metric_names': ['Matched Acc.'],
    },
    'qnli': {
        'model_name': 'textattack/bert-base-uncased-QNLI',
        'glue_name': 'qnli',
        'num_labels': 2,
        'input_cols': ('question', 'sentence'),
        'task_type': 'classification',
        'val_split': 'validation',
        'primary_metric': 'accuracy',
        'all_metrics': ['accuracy'],
        'metric_names': ['Accuracy'],
    },
    'rte': {
        'model_name': 'textattack/bert-base-uncased-RTE',
        'glue_name': 'rte',
        'num_labels': 2,
        'input_cols': ('sentence1', 'sentence2'),
        'task_type': 'classification',
        'val_split': 'validation',
        'primary_metric': 'accuracy',
        'all_metrics': ['accuracy'],
        'metric_names': ['Accuracy'],
    },
    'wnli': {
        'model_name': 'textattack/bert-base-uncased-WNLI',
        'glue_name': 'wnli',
        'num_labels': 2,
        'input_cols': ('sentence1', 'sentence2'),
        'task_type': 'classification',
        'val_split': 'validation',
        'primary_metric': 'accuracy',
        'all_metrics': ['accuracy'],
        'metric_names': ['Accuracy'],
    },
}

ALL_TASKS = ["sst2", "cola", "mrpc", "stsb", "mnli", "qnli", "rte", "wnli"]

TASK_GROUPS = {
    "single_sentence": ["cola", "sst2"],
    "similarity_paraphrase": ["mrpc", "stsb"],
    "nli": ["mnli", "qnli", "rte", "wnli"],
}


def detect_layer_attribute(model):
    for path in ['bert.encoder.layer', 'model.layers', 'transformer.h', 'roberta.encoder.layer']:
        try:
            obj = model
            for attr in path.split('.'):
                obj = getattr(obj, attr)
            if len(obj) > 0:
                return path
        except Exception:
            continue
    return 'bert.encoder.layer'


def apply_config(model, handler, layers_attr, gelu_degrees, softmax_degrees):
    """Apply GELU/Softmax approximation config to model via handler."""
    handler_layer_name = "model." + layers_attr
    gelu_map = {d: [] for d in [0, 1, 2, 4]}
    for idx, deg in enumerate(gelu_degrees):
        if deg in gelu_map:
            gelu_map[deg].append(idx)
    for d in [0, 1, 2, 4]:
        if gelu_map[d]:
            handler.replace_layer_gelu(gelu_map[d], handler_layer_name, degree=d)

    softmax_map = {d: [] for d in range(2, 7)}
    for idx, deg in enumerate(softmax_degrees):
        if deg in softmax_map:
            softmax_map[deg].append(idx)
    for d in range(2, 7):
        if softmax_map[d]:
            handler.replace_layer_softmax(softmax_map[d], handler_layer_name, degree=d)


def load_model_and_data(task_name, device='cuda', max_length=128, batch_size=16):
    """
    Load task model, tokenizer, and validation DataLoader.
    Returns (model, handler, layers_attr, dataloader, labels, task_cfg).
    """
    task_cfg = TASK_REGISTRY[task_name]
    model_name = task_cfg['model_name']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=task_cfg['num_labels'],
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )
    model.to(device)

    handler = ReversibleLayerHandler(model)
    layers_attr = detect_layer_attribute(model)

    data = load_dataset("nyu-mll/glue", task_cfg['glue_name'])
    val_split = data[task_cfg['val_split']]

    labels = np.array(val_split['label'])

    input_cols = task_cfg['input_cols']

    def tokenize_fn(examples):
        if len(input_cols) == 1:
            return tokenizer(examples[input_cols[0]], truncation=True, padding=False, max_length=max_length)
        return tokenizer(examples[input_cols[0]], examples[input_cols[1]], truncation=True, padding=False, max_length=max_length)

    tokenized = val_split.map(tokenize_fn, batched=True)

    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.column_names:
        columns.append("token_type_ids")
    columns.append("label")
    tokenized.set_format(type="torch", columns=columns)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=max_length,
        return_tensors="pt", pad_to_multiple_of=8,
    )

    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    return model, handler, layers_attr, dataloader, labels, task_cfg


def run_inference_with_labels(model, dataloader, device):
    """Run inference, return (all_logits, all_labels) as numpy arrays."""
    model.eval()
    model.to(device)
    all_logits, all_labels = [], []

    if torch.cuda.is_available():
        dummy = next(iter(dataloader))
        dummy_input = {k: v.to(device) for k, v in dummy.items() if k != 'labels'}
        with torch.no_grad():
            _ = model(**dummy_input)
        torch.cuda.synchronize()

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.detach().cpu().numpy()
            all_logits.append(logits)
            all_labels.extend(labels)

    return np.concatenate(all_logits, axis=0), np.array(all_labels)


def compute_metrics(logits, labels, task_name):
    """
    Compute all metrics for a task. Returns dict {metric_name: value}.
    The primary metric is always the first one.
    """
    task_cfg = TASK_REGISTRY[task_name]
    results = {}

    if task_cfg['task_type'] == 'regression':
        preds = logits.squeeze()
        if np.ndim(preds) == 0:
            preds = np.array([preds])
        results['pearson'] = pearsonr(preds, labels)[0]
        results['spearman'] = spearmanr(preds, labels)[0]
    else:
        if len(logits.shape) == 1:
            pred_classes = (logits > 0.5).astype(int)
        else:
            pred_classes = np.argmax(logits, axis=1)

        if task_name == 'cola':
            results['mcc'] = matthews_corrcoef(labels, pred_classes)
        elif task_name == 'mrpc':
            results['accuracy'] = accuracy_score(labels, pred_classes)
            results['f1'] = f1_score(labels, pred_classes, average='weighted')
        else:
            results['accuracy'] = accuracy_score(labels, pred_classes)

    return results


def evaluate_config(model, handler, layers_attr, dataloader, labels, task_name, gelu_degrees, softmax_degrees, device='cuda'):
    """
    Apply config, run inference, compute metrics. Restores model after.
    Returns dict of metrics.
    """
    handler.restore_all()
    model = handler.model
    model.to(device)

    apply_config(model, handler, layers_attr, gelu_degrees, softmax_degrees, )

    logits, _ = run_inference_with_labels(model, dataloader, device)
    metrics = compute_metrics(logits, labels, task_name)
    return metrics


def get_logits_for_config(model, handler, layers_attr, dataloader, gelu_degrees, softmax_degrees, device='cuda'):
    """Apply config, run inference, return raw logits. Restores model after."""
    handler.restore_all()
    model = handler.model
    model.to(device)
    apply_config(model, handler, layers_attr, gelu_degrees, softmax_degrees)
    logits, labels = run_inference_with_labels(model, dataloader, device)
    return logits, labels


def bootstrap_metric(logits, labels, task_name, n_bootstrap=100, seed=42):
    """
    Bootstrap resample predictions to get metric distribution.
    Returns dict {metric_name: np.array of bootstrap samples}.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    task_cfg = TASK_REGISTRY[task_name]
    metric_keys = task_cfg['all_metrics']
    distributions = {k: [] for k in metric_keys}

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_logits = logits[indices]
        boot_labels = labels[indices]
        m = compute_metrics(boot_logits, boot_labels, task_name)
        for k in metric_keys:
            distributions[k].append(m[k])

    return {k: np.array(v) for k, v in distributions.items()}


def get_primary_metric(task_name):
    return TASK_REGISTRY[task_name]['primary_metric']


def config_label(gelu, softmax):
    """Short string label for a configuration."""
    g_str = ",".join(str(x) for x in gelu)
    s_str = ",".join(str(x) for x in softmax)
    return f"G[{g_str}]_S[{s_str}]"


def load_ppo_configs(path="glue_configs_best_ppo.json"):
    with open(path, 'r') as f:
        configs = json.load(f)
    configs.pop("_comment", None)
    return configs
