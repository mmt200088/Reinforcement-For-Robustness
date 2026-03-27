#!/usr/bin/env python
"""
噪声 Scaling Factor 扫描实验脚本

一、脚本用途
本脚本用于系统性评估第二阶段噪声注入中 7 类噪声对象的 scaling factor 变化对模型推理结果的影响。
7 类噪声对象分别为：
1. x      : Transformer 层输入噪声
2. wq     : Attention Query 权重噪声
3. wk     : Attention Key 权重噪声
4. wv     : Attention Value 权重噪声
5. wo     : Attention 输出投影权重噪声
6. wffn1  : FFN 第一层权重噪声
7. wffn2  : FFN 第二层权重噪声

二、默认实验设计
1. 默认数据集：
   mnli / sst2 / mrpc / stsb / qnli / cola / rte / wnli
2. 默认评估切分：
   validation_full（验证集全集）
3. 默认重复次数：
   每个扫描点重复 50 次
4. 默认“当前 x 和 6 个 W 配置”来源：
   glue_noise_configs_best_ppo.json
5. 默认固定的 GELU / Softmax 配置来源：
   glue_configs_best_ppo.json
6. 默认扫描方式：
   对目标噪声项的 12 层统一设置为同一个 scaling factor，
   其余 6 类噪声保持当前配置不变。

三、默认 scaling factor 扫描范围
1. x 使用：
   {22, 24, 26, 28, 30}
2. wq / wk / wv / wo / wffn2 使用：
   {14, 16, 18, 20, 22}
3. wffn1 使用：
   {16, 18, 20, 22, 24}

四、图像含义
每个数据集输出一张总图，图中每条曲线对应一个噪声对象（x / wq / wk / ...）。
同一条曲线在每个 scaling factor 点上都会做重复实验，并展示三层信息：
1. 实线：该点多次实验的均值
2. 深色半透明阴影：mean ± std
3. 浅色半透明阴影：min ~ max 包络

五、输出文件
默认输出目录：
experiment_results/noise_scaling_sweep

每个数据集会生成：
1. noise_scaling_sweep_<dataset>.png
2. noise_scaling_sweep_<dataset>.json
3. runtime_<dataset>.log

六、前台运行示例
1. 跑全部数据集：
   python experiment_noise_scaling_sweep.py
2. 只跑单个数据集：
   python experiment_noise_scaling_sweep.py --tasks sst2
3. 指定多个数据集：
   python experiment_noise_scaling_sweep.py --tasks sst2 mrpc cola
4. 指定输出目录：
   python experiment_noise_scaling_sweep.py --output_dir experiment_results/noise_scaling_sweep_v2
5. 指定评估切分：
   python experiment_noise_scaling_sweep.py --tasks sst2 --eval_split validation_full
6. 指定重复次数：
   python experiment_noise_scaling_sweep.py --tasks sst2 --repeat_n 100
7. 开发调试用小样本模式：
   python experiment_noise_scaling_sweep.py --tasks sst2 --repeat_n 2 --max_eval_samples 32
8. 使用手动噪声配置作为“当前配置”：
   python experiment_noise_scaling_sweep.py --tasks mrpc --noise_base_source manual --manual_noise_config "{\"x\":[30,30,30,30,30,30,30,30,30,30,30,30],\"wq\":[22,22,22,22,22,22,22,22,22,22,22,22],\"wk\":[22,22,22,22,22,22,22,22,22,22,22,22],\"wv\":[22,22,22,22,22,22,22,22,22,22,22,22],\"wo\":[22,22,22,22,22,22,22,22,22,22,22,22],\"wffn1\":[22,22,22,22,22,22,22,22,22,22,22,22],\"wffn2\":[22,22,22,22,22,22,22,22,22,22,22,22]}"

七、后台运行方式
推荐直接使用配套的 run_noise_scaling_sweep.sh：
1. 后台运行：
   bash run_noise_scaling_sweep.sh
2. 指定单任务后台运行：
   bash run_noise_scaling_sweep.sh --tasks sst2
3. 前台运行：
   bash run_noise_scaling_sweep.sh --foreground --tasks sst2 --repeat_n 2 --max_eval_samples 32

八、查看日志与停止任务
如果使用 run_noise_scaling_sweep.sh 后台运行：
1. 查看日志：
   tail -f experiment_results/noise_scaling_sweep/run.log
2. 查看 PID：
   cat experiment_results/noise_scaling_sweep/pid.txt
3. 停止任务：
   kill -9 $(cat experiment_results/noise_scaling_sweep/pid.txt)

九、注意事项
1. 正式实验建议保持默认 repeat_n=50 或更高。
2. repeat_n < 50 时，本脚本会给出提示，但仍允许执行，便于开发调试。
3. 本脚本默认固定 GELU / Softmax，仅扫描噪声 scaling factor。
4. 本脚本默认使用 validation_full；如果你切换到 train，请注意运行时间会更长。
5. x 的 scaling factor 与 6 个 W 的 scaling factor 数值区间不同，这是当前项目代码中的既定设计。
"""

import argparse
import gc
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from function_handler import (
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
    get_input_noise_variance,
)
from layer_importance_evaluator import LayerImportanceEvaluator


TASK_REGISTRY = {
    "mnli": {
        "model_name": "textattack/bert-base-uncased-MNLI",
        "glue_name": "mnli",
        "num_labels": 3,
        "input_cols": ("premise", "hypothesis"),
        "validation_split": "validation_matched",
        "validation_split_mm": "validation_mismatched",
    },
    "sst2": {
        "model_name": "textattack/bert-base-uncased-SST-2",
        "glue_name": "sst2",
        "num_labels": 2,
        "input_cols": ("sentence",),
        "validation_split": "validation",
        "validation_split_mm": None,
    },
    "mrpc": {
        "model_name": "textattack/bert-base-uncased-MRPC",
        "glue_name": "mrpc",
        "num_labels": 2,
        "input_cols": ("sentence1", "sentence2"),
        "validation_split": "validation",
        "validation_split_mm": None,
    },
    "stsb": {
        "model_name": "textattack/bert-base-uncased-STS-B",
        "glue_name": "stsb",
        "num_labels": 1,
        "input_cols": ("sentence1", "sentence2"),
        "validation_split": "validation",
        "validation_split_mm": None,
    },
    "qnli": {
        "model_name": "textattack/bert-base-uncased-QNLI",
        "glue_name": "qnli",
        "num_labels": 2,
        "input_cols": ("question", "sentence"),
        "validation_split": "validation",
        "validation_split_mm": None,
    },
    "cola": {
        "model_name": "textattack/bert-base-uncased-CoLA",
        "glue_name": "cola",
        "num_labels": 2,
        "input_cols": ("sentence",),
        "validation_split": "validation",
        "validation_split_mm": None,
    },
    "rte": {
        "model_name": "textattack/bert-base-uncased-RTE",
        "glue_name": "rte",
        "num_labels": 2,
        "input_cols": ("sentence1", "sentence2"),
        "validation_split": "validation",
        "validation_split_mm": None,
    },
    "wnli": {
        "model_name": "textattack/bert-base-uncased-WNLI",
        "glue_name": "wnli",
        "num_labels": 2,
        "input_cols": ("sentence1", "sentence2"),
        "validation_split": "validation",
        "validation_split_mm": None,
    },
}

ALL_TASKS = ["mnli", "sst2", "mrpc", "stsb", "qnli", "cola", "rte", "wnli"]

NOISE_TARGETS = {
    "x": {
        "full_key": "input_noise_scaling_factors",
        "display_name": "x (Input)",
        "allowed_values": tuple(INPUT_NOISE_ALLOWED_SCALING_FACTORS),
        "distribution": "fresh",
    },
    "wq": {
        "full_key": "wq_noise_scaling_factors",
        "display_name": "Wq",
        "allowed_values": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
        "distribution": "encoding",
    },
    "wk": {
        "full_key": "wk_noise_scaling_factors",
        "display_name": "Wk",
        "allowed_values": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
        "distribution": "encoding",
    },
    "wv": {
        "full_key": "wv_noise_scaling_factors",
        "display_name": "Wv",
        "allowed_values": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
        "distribution": "encoding",
    },
    "wo": {
        "full_key": "wo_noise_scaling_factors",
        "display_name": "Wo",
        "allowed_values": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
        "distribution": "encoding",
    },
    "wffn1": {
        "full_key": "wffn1_noise_scaling_factors",
        "display_name": "Wffn1",
        "allowed_values": tuple(WFFN1_NOISE_ALLOWED_SCALING_FACTORS),
        "distribution": "encoding",
    },
    "wffn2": {
        "full_key": "wffn2_noise_scaling_factors",
        "display_name": "Wffn2",
        "allowed_values": tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS),
        "distribution": "encoding",
    },
}

TARGET_ORDER = ["x", "wq", "wk", "wv", "wo", "wffn1", "wffn2"]
TARGET_COLORS = {
    "x": "#E45756",
    "wq": "#4C78A8",
    "wk": "#72B7B2",
    "wv": "#54A24B",
    "wo": "#EECA3B",
    "wffn1": "#F58518",
    "wffn2": "#B279A2",
}

APPROX_GELU_ALLOWED = (0, 1, 2, 4)
APPROX_SOFTMAX_ALLOWED = (2, 3, 4, 5, 6)
UNION_FACTOR_TICKS = sorted(
    set(INPUT_NOISE_ALLOWED_SCALING_FACTORS).union(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep x and 6 weight-noise scaling factors across GLUE datasets."
    )
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("experiment_results", "noise_scaling_sweep"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_split", type=str, default="validation_full")
    parser.add_argument("--repeat_n", type=int, default=50)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise_base_source",
        type=str,
        default="json",
        choices=["json", "manual"],
    )
    parser.add_argument(
        "--noise_base_config",
        type=str,
        default="glue_noise_configs_best_ppo.json",
    )
    parser.add_argument("--manual_noise_config", type=str, default="")
    parser.add_argument(
        "--approx_base_config",
        type=str,
        default="glue_configs_best_ppo.json",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_noise_config(raw_value: str) -> Optional[dict]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    return json.loads(text)


def clean_number(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def summarize_series(values: Sequence[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    output = {
        "mean": None,
        "min": None,
        "max": None,
        "std": None,
        "trial_count": int(arr.size),
        "raw_values": [clean_number(v) for v in arr.tolist()],
    }
    if finite.size == 0:
        return output
    output["mean"] = float(np.mean(finite))
    output["min"] = float(np.min(finite))
    output["max"] = float(np.max(finite))
    output["std"] = float(np.std(finite))
    return output


def normalize_array(
    values: Sequence[int],
    total_layers: int,
    allowed_values: Sequence[int],
    label: str,
) -> np.ndarray:
    arr = np.asarray(list(values), dtype=int).flatten()
    if arr.shape != (total_layers,):
        raise ValueError(
            f"{label} length mismatch: expected ({total_layers},), got {arr.shape}"
        )
    invalid = sorted(set(arr.tolist()) - set(allowed_values))
    if invalid:
        raise ValueError(
            f"{label} has unsupported values {invalid}. Allowed values: {list(allowed_values)}"
        )
    return arr


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.pop("_comment", None)
    return data


def resolve_tasks(raw_tasks: Optional[Sequence[str]]) -> List[str]:
    if raw_tasks is None:
        return list(ALL_TASKS)
    resolved = []
    for task in raw_tasks:
        key = str(task).strip().lower()
        if key not in TASK_REGISTRY:
            raise ValueError(
                f"Unsupported task '{task}'. Supported tasks: {', '.join(ALL_TASKS)}"
            )
        resolved.append(key)
    return resolved


def sample_hf_dataset(dataset, max_samples: int, seed: int):
    if dataset is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False))
    return dataset.select(indices.tolist())


def tokenize_split(dataset, tokenizer, input_cols: Tuple[str, ...], max_length: int):
    if dataset is None:
        return None

    def tokenize_fn(examples):
        if len(input_cols) == 1:
            return tokenizer(
                examples[input_cols[0]],
                truncation=True,
                padding=False,
                max_length=max_length,
            )
        return tokenizer(
            examples[input_cols[0]],
            examples[input_cols[1]],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    if "label" in tokenized.column_names:
        tokenized = tokenized.rename_column("label", "labels")

    columns = ["input_ids", "attention_mask"]
    if "token_type_ids" in tokenized.column_names:
        columns.append("token_type_ids")
    if "labels" in tokenized.column_names:
        columns.append("labels")
    tokenized.set_format(type="torch", columns=columns)
    return tokenized


def build_evaluator(
    task_name: str,
    device: str,
    max_length: int,
    batch_size: int,
    output_dir: str,
    approx_base_config_path: str,
    noise_base_config_path: str,
    eval_split: str,
    seed: int,
):
    task_cfg = TASK_REGISTRY[task_name]
    model_name = task_cfg["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=task_cfg["num_labels"],
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True,
    )
    model.to(device)

    data = load_dataset("nyu-mll/glue", task_cfg["glue_name"])
    raw_train = data["train"]
    if eval_split != "train":
        raw_train = sample_hf_dataset(
            raw_train,
            min(len(raw_train), max(batch_size * 2, 64)),
            seed + 11,
        )

    raw_validation = data[task_cfg["validation_split"]]
    raw_validation_mm = (
        data[task_cfg["validation_split_mm"]]
        if task_cfg["validation_split_mm"] is not None
        else None
    )

    train_data = tokenize_split(raw_train, tokenizer, task_cfg["input_cols"], max_length)
    validation_data = tokenize_split(
        raw_validation, tokenizer, task_cfg["input_cols"], max_length
    )
    validation_data_mm = tokenize_split(
        raw_validation_mm, tokenizer, task_cfg["input_cols"], max_length
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )

    original_cwd = os.getcwd()
    runtime_init_dir = os.path.join(output_dir, "_runtime", task_name)
    os.makedirs(runtime_init_dir, exist_ok=True)
    try:
        os.chdir(runtime_init_dir)
        evaluator = LayerImportanceEvaluator(
            model=model,
            train_data=train_data,
            test_data=validation_data,
            data_collator=data_collator,
            device=device,
            data_path=task_name,
            test_data_mm=validation_data_mm,
            final_eval_config_source="json",
            final_eval_config_path=approx_base_config_path,
            noise_eval_config_source="json",
            noise_eval_config_path=noise_base_config_path,
            skip_stage1_rl=True,
            skip_stage1_final_eval=True,
            skip_noise_rl=True,
            skip_noise_final_eval=True,
        )
    finally:
        os.chdir(original_cwd)

    evaluator.log_file = os.path.join(output_dir, f"runtime_{task_name}.log")
    evaluator.dataloader_train = evaluator.dataloaders.get("train")
    evaluator.dataloader_test = evaluator.dataloaders.get("validation_full")
    evaluator.dataloader_test_mm = evaluator.dataloaders_mm.get("validation_full")
    return evaluator


def maybe_limit_eval_split(
    evaluator: LayerImportanceEvaluator,
    split_name: str,
    max_eval_samples: int,
    seed: int,
) -> None:
    if max_eval_samples <= 0:
        return
    if not evaluator.has_dataset_split(split_name):
        available = sorted(evaluator.dataloaders.keys())
        raise ValueError(
            f"Eval split '{split_name}' is unavailable. Available splits: {available}"
        )

    dataset = evaluator.dataset_splits[split_name]
    dataset_mm = evaluator.dataset_splits_mm.get(split_name)
    limited = evaluator._sample_dataset_by_size(dataset, max_eval_samples, seed + 101)
    limited_mm = None
    if dataset_mm is not None:
        limited_mm = evaluator._sample_dataset_by_size(
            dataset_mm,
            max_eval_samples,
            seed + 151,
        )

    evaluator._register_dataset_split(split_name, limited, limited_mm)
    if split_name == "train":
        evaluator.dataloader_train = evaluator.dataloaders.get("train")
    if split_name == "validation_full":
        evaluator.dataloader_test = evaluator.dataloaders.get("validation_full")
        evaluator.dataloader_test_mm = evaluator.dataloaders_mm.get("validation_full")

    print(
        f"[Sample Limit] Split '{split_name}' limited to {len(limited)} samples"
        + (
            f" / mismatched {len(limited_mm)} samples"
            if limited_mm is not None
            else ""
        ),
        flush=True,
    )


def resolve_fixed_approx_config(
    dataset_key: str,
    config_path: str,
    total_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    config_map = load_json(config_path)
    if dataset_key not in config_map:
        raise KeyError(
            f"Dataset '{dataset_key}' not found in approx config file '{config_path}'."
        )
    cfg = config_map[dataset_key]
    gelu = normalize_array(
        cfg["gelu"],
        total_layers=total_layers,
        allowed_values=APPROX_GELU_ALLOWED,
        label=f"{dataset_key}.gelu",
    )
    softmax = normalize_array(
        cfg["softmax"],
        total_layers=total_layers,
        allowed_values=APPROX_SOFTMAX_ALLOWED,
        label=f"{dataset_key}.softmax",
    )
    return gelu, softmax


def resolve_base_noise_config(
    dataset_key: str,
    total_layers: int,
    source: str,
    config_path: str,
    manual_noise_config: Optional[dict],
) -> Dict[str, np.ndarray]:
    if source == "json":
        config_map = load_json(config_path)
        if dataset_key not in config_map:
            raise KeyError(
                f"Dataset '{dataset_key}' not found in noise config file '{config_path}'."
            )
        raw_cfg = config_map[dataset_key]
    elif source == "manual":
        if manual_noise_config is None:
            raise ValueError("noise_base_source=manual 时必须提供 --manual_noise_config。")
        raw_cfg = manual_noise_config
    else:
        raise ValueError(f"Unsupported noise_base_source: {source}")

    resolved = {}
    for short_key in TARGET_ORDER:
        meta = NOISE_TARGETS[short_key]
        candidate = None
        if short_key in raw_cfg:
            candidate = raw_cfg[short_key]
        elif meta["full_key"] in raw_cfg:
            candidate = raw_cfg[meta["full_key"]]
        if candidate is None:
            raise KeyError(
                f"Noise config for dataset '{dataset_key}' is missing '{short_key}' / '{meta['full_key']}'."
            )
        resolved[meta["full_key"]] = normalize_array(
            candidate,
            total_layers=total_layers,
            allowed_values=meta["allowed_values"],
            label=f"{dataset_key}.{short_key}",
        )
    return resolved


def to_short_noise_config(noise_config: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
    output = {}
    for short_key, meta in NOISE_TARGETS.items():
        output[short_key] = noise_config[meta["full_key"]].astype(int).tolist()
    return output


def build_sweep_noise_config(
    base_noise_config: Dict[str, np.ndarray],
    target_key: str,
    factor: int,
    total_layers: int,
) -> Dict[str, np.ndarray]:
    updated = {
        key: np.asarray(value, dtype=int).copy()
        for key, value in base_noise_config.items()
    }
    updated[NOISE_TARGETS[target_key]["full_key"]] = np.full(
        total_layers, int(factor), dtype=int
    )
    return updated


def make_trial_seed(
    base_seed: int,
    dataset_idx: int,
    target_idx: int,
    factor_idx: int,
    trial_idx: int,
    namespace: int = 0,
) -> int:
    return int(
        base_seed
        + namespace * 1_000_000
        + dataset_idx * 100_000
        + target_idx * 10_000
        + factor_idx * 100
        + trial_idx
    )


def evaluate_noise_config_trials(
    evaluator: LayerImportanceEvaluator,
    fixed_gelu: np.ndarray,
    fixed_softmax: np.ndarray,
    noise_config: Dict[str, np.ndarray],
    repeat_n: int,
    split_name: str,
    dataset_idx: int,
    target_idx: int,
    factor_idx: int,
    base_seed: int,
    namespace: int = 0,
) -> dict:
    trial_seeds = []
    losses = []
    primary = []
    secondary = []
    times = []

    for trial_idx in range(repeat_n):
        trial_seed = make_trial_seed(
            base_seed=base_seed,
            dataset_idx=dataset_idx,
            target_idx=target_idx,
            factor_idx=factor_idx,
            trial_idx=trial_idx,
            namespace=namespace,
        )
        trial_seeds.append(int(trial_seed))
        set_global_seed(trial_seed)
        loss, metric1, metric2, elapsed_ms = evaluator.evaluate_model_with_attention_noise(
            fixed_gelu,
            fixed_softmax,
            use_train=False,
            split=split_name,
            **noise_config,
        )
        losses.append(float(loss))
        primary.append(float(metric1))
        secondary.append(float(metric2))
        times.append(float(elapsed_ms))

        evaluator.clear_weight_noise_configuration()
        evaluator.clear_input_noise_configuration()
        evaluator.apply_configuration(fixed_gelu, fixed_softmax)

    return {
        "trial_seeds": trial_seeds,
        "loss": summarize_series(losses),
        "primary_metric": summarize_series(primary),
        "secondary_metric": summarize_series(secondary),
        "time_ms": summarize_series(times),
    }


def extract_metric_mean(target_records: List[dict], metric_name: str) -> np.ndarray:
    values = []
    for record in target_records:
        value = record[metric_name]["mean"]
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def extract_metric_min(target_records: List[dict], metric_name: str) -> np.ndarray:
    values = []
    for record in target_records:
        value = record[metric_name]["min"]
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def extract_metric_max(target_records: List[dict], metric_name: str) -> np.ndarray:
    values = []
    for record in target_records:
        value = record[metric_name]["max"]
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def extract_metric_std(target_records: List[dict], metric_name: str) -> np.ndarray:
    values = []
    for record in target_records:
        value = record[metric_name]["std"]
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def draw_metric_panel(
    ax,
    target_results: Dict[str, List[dict]],
    metric_name: str,
    ylabel: str,
    reference_mean: Optional[float],
):
    for target_key in TARGET_ORDER:
        records = target_results[target_key]
        factors = np.asarray([entry["scaling_factor"] for entry in records], dtype=float)
        means = extract_metric_mean(records, metric_name)
        mins = extract_metric_min(records, metric_name)
        maxs = extract_metric_max(records, metric_name)
        stds = extract_metric_std(records, metric_name)
        inner_low = np.maximum(means - stds, mins)
        inner_high = np.minimum(means + stds, maxs)

        color = TARGET_COLORS[target_key]
        ax.fill_between(factors, mins, maxs, color=color, alpha=0.10)
        ax.fill_between(factors, inner_low, inner_high, color=color, alpha=0.24)
        ax.plot(
            factors,
            means,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=NOISE_TARGETS[target_key]["display_name"],
        )

    if reference_mean is not None and np.isfinite(reference_mean):
        ax.axhline(
            float(reference_mean),
            color="#222222",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )

    ax.set_xticks(UNION_FACTOR_TICKS)
    ax.set_xlabel("Scaling Factor")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def build_reference_text(
    dataset_key: str,
    split_name: str,
    repeat_n: int,
    noise_source: str,
    noise_config_path: str,
    approx_config_path: str,
    base_reference: dict,
    primary_metric_name: str,
    secondary_metric_name: Optional[str],
) -> str:
    lines = [
        f"Dataset: {dataset_key.upper()}",
        f"Eval split: {split_name}",
        f"Repeats / point: {repeat_n}",
        f"Noise base source: {noise_source}",
        f"Noise base file: {os.path.basename(noise_config_path)}",
        f"Fixed GELU/Softmax: {os.path.basename(approx_config_path)}",
        "",
        "Reference current-config mean:",
        f"Loss = {base_reference['loss']['mean']:.4f}"
        if base_reference["loss"]["mean"] is not None
        else "Loss = N/A",
        f"{primary_metric_name} = {base_reference['primary_metric']['mean']:.4f}"
        if base_reference["primary_metric"]["mean"] is not None
        else f"{primary_metric_name} = N/A",
    ]
    if secondary_metric_name is not None:
        lines.append(
            f"{secondary_metric_name} = {base_reference['secondary_metric']['mean']:.4f}"
            if base_reference["secondary_metric"]["mean"] is not None
            else f"{secondary_metric_name} = N/A"
        )
    lines.extend(
        [
            "",
            "Shading:",
            "dark = mean ± std",
            "light = min ~ max",
            "",
            "Factor ranges:",
            f"x: {list(INPUT_NOISE_ALLOWED_SCALING_FACTORS)}",
            f"W (wq/wk/wv/wo/wffn2): {list(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)}",
            f"Wffn1: {list(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)}",
        ]
    )
    return "\n".join(lines)


def plot_task_results(
    dataset_key: str,
    split_name: str,
    repeat_n: int,
    noise_source: str,
    noise_config_path: str,
    approx_config_path: str,
    metric_short_names: Sequence[str],
    num_metrics: int,
    target_results: Dict[str, List[dict]],
    base_reference: dict,
    output_path: str,
) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Noise Scaling Sweep ({dataset_key.upper()})",
        fontsize=15,
        fontweight="bold",
    )

    ax_loss = axes[0, 0]
    ax_primary = axes[0, 1]
    ax_bottom_left = axes[1, 0]
    ax_info = axes[1, 1]

    draw_metric_panel(
        ax_loss,
        target_results,
        metric_name="loss",
        ylabel="Loss",
        reference_mean=base_reference["loss"]["mean"],
    )
    ax_loss.set_title("Loss vs Scaling Factor")

    draw_metric_panel(
        ax_primary,
        target_results,
        metric_name="primary_metric",
        ylabel=metric_short_names[0],
        reference_mean=base_reference["primary_metric"]["mean"],
    )
    ax_primary.set_title(f"{metric_short_names[0]} vs Scaling Factor")

    if num_metrics > 1:
        draw_metric_panel(
            ax_bottom_left,
            target_results,
            metric_name="secondary_metric",
            ylabel=metric_short_names[1],
            reference_mean=base_reference["secondary_metric"]["mean"],
        )
        ax_bottom_left.set_title(f"{metric_short_names[1]} vs Scaling Factor")
    else:
        ax_bottom_left.axis("off")
        ax_bottom_left.text(
            0.02,
            0.95,
            "统计含义\n\n"
            "每条曲线代表一个噪声对象。\n"
            "横轴为 scaling factor。\n"
            "实线是均值，深色阴影是 mean±std，浅色阴影是 min~max。\n"
            "黑色虚线表示当前整套噪声配置的重复评估均值。",
            va="top",
            ha="left",
            fontsize=10,
        )

    ax_info.axis("off")
    info_text = build_reference_text(
        dataset_key=dataset_key,
        split_name=split_name,
        repeat_n=repeat_n,
        noise_source=noise_source,
        noise_config_path=noise_config_path,
        approx_config_path=approx_config_path,
        base_reference=base_reference,
        primary_metric_name=metric_short_names[0],
        secondary_metric_name=metric_short_names[1] if num_metrics > 1 else None,
    )
    ax_info.text(
        0.02,
        0.95,
        info_text,
        va="top",
        ha="left",
        fontsize=10,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TARGET_COLORS[target_key],
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=NOISE_TARGETS[target_key]["display_name"],
        )
        for target_key in TARGET_ORDER
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#222222",
                linestyle="--",
                linewidth=1.4,
                label="Current Config Mean",
            ),
            Patch(facecolor="#666666", alpha=0.24, label="Mean ± Std"),
            Patch(facecolor="#666666", alpha=0.10, label="Min ~ Max"),
        ]
    )
    ax_info.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=9,
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_task_sweep(
    evaluator: LayerImportanceEvaluator,
    dataset_key: str,
    dataset_idx: int,
    fixed_gelu: np.ndarray,
    fixed_softmax: np.ndarray,
    base_noise_config: Dict[str, np.ndarray],
    split_name: str,
    repeat_n: int,
    seed: int,
) -> Tuple[dict, Dict[str, List[dict]]]:
    metric_short_names = evaluator.get_metric_short_names()
    num_metrics = evaluator.get_num_metrics()
    base_reference = evaluate_noise_config_trials(
        evaluator=evaluator,
        fixed_gelu=fixed_gelu,
        fixed_softmax=fixed_softmax,
        noise_config=base_noise_config,
        repeat_n=repeat_n,
        split_name=split_name,
        dataset_idx=dataset_idx,
        target_idx=0,
        factor_idx=0,
        base_seed=seed,
        namespace=9,
    )

    target_results = {}
    for target_idx, target_key in enumerate(TARGET_ORDER):
        meta = NOISE_TARGETS[target_key]
        base_values = base_noise_config[meta["full_key"]]
        target_results[target_key] = []
        print(
            f"  [Target] {dataset_key.upper()} - {target_key}: "
            f"base unique values = {sorted(set(base_values.tolist()))}",
            flush=True,
        )

        for factor_idx, factor in enumerate(meta["allowed_values"]):
            sweep_config = build_sweep_noise_config(
                base_noise_config=base_noise_config,
                target_key=target_key,
                factor=factor,
                total_layers=evaluator.total_layers,
            )
            total_cost, breakdown = evaluator.get_noise_simulated_cost(**sweep_config)
            trial_summary = evaluate_noise_config_trials(
                evaluator=evaluator,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                noise_config=sweep_config,
                repeat_n=repeat_n,
                split_name=split_name,
                dataset_idx=dataset_idx,
                target_idx=target_idx,
                factor_idx=factor_idx,
                base_seed=seed,
                namespace=1,
            )
            record = {
                "scaling_factor": int(factor),
                "variance": clean_number(
                    get_input_noise_variance(
                        int(factor),
                        distribution=meta["distribution"],
                    )
                ),
                "simulated_total_cost": float(total_cost),
                "simulated_cost_breakdown": {
                    key: float(value) for key, value in breakdown.items()
                },
                "noise_config": to_short_noise_config(sweep_config),
                "trial_seeds": trial_summary["trial_seeds"],
                "loss": trial_summary["loss"],
                "primary_metric": trial_summary["primary_metric"],
                "secondary_metric": (
                    trial_summary["secondary_metric"] if num_metrics > 1 else None
                ),
                "time_ms": trial_summary["time_ms"],
            }
            target_results[target_key].append(record)

            msg = (
                f"    factor={factor}: loss={record['loss']['mean']:.4f}"
                if record["loss"]["mean"] is not None
                else f"    factor={factor}: loss=N/A"
            )
            if record["primary_metric"]["mean"] is not None:
                msg += f", {metric_short_names[0]}={record['primary_metric']['mean']:.4f}"
            else:
                msg += f", {metric_short_names[0]}=N/A"
            if num_metrics > 1:
                secondary_summary = record["secondary_metric"]
                if secondary_summary is not None and secondary_summary["mean"] is not None:
                    msg += f", {metric_short_names[1]}={secondary_summary['mean']:.4f}"
                else:
                    msg += f", {metric_short_names[1]}=N/A"
            print(msg, flush=True)

    return base_reference, target_results


def build_task_json_summary(
    dataset_key: str,
    split_name: str,
    repeat_n: int,
    max_eval_samples: int,
    noise_base_source: str,
    noise_base_config_path: str,
    approx_base_config_path: str,
    fixed_gelu: np.ndarray,
    fixed_softmax: np.ndarray,
    base_noise_config: Dict[str, np.ndarray],
    base_reference: dict,
    metric_short_names: Sequence[str],
    num_metrics: int,
    target_results: Dict[str, List[dict]],
) -> dict:
    targets_summary = {}
    for target_key in TARGET_ORDER:
        meta = NOISE_TARGETS[target_key]
        base_values = base_noise_config[meta["full_key"]]
        targets_summary[target_key] = {
            "display_name": meta["display_name"],
            "base_layer_values": base_values.astype(int).tolist(),
            "base_unique_values": sorted(set(base_values.astype(int).tolist())),
            "allowed_scaling_factors": list(meta["allowed_values"]),
            "distribution": meta["distribution"],
            "records": target_results[target_key],
        }

    return {
        "dataset": dataset_key,
        "eval_split": split_name,
        "repeat_n": int(repeat_n),
        "max_eval_samples": int(max_eval_samples),
        "noise_base_source": noise_base_source,
        "noise_base_config_path": noise_base_config_path,
        "approx_base_config_path": approx_base_config_path,
        "metric_short_names": list(metric_short_names),
        "num_metrics": int(num_metrics),
        "fixed_approx_config": {
            "gelu": fixed_gelu.astype(int).tolist(),
            "softmax": fixed_softmax.astype(int).tolist(),
        },
        "base_noise_config": to_short_noise_config(base_noise_config),
        "current_config_reference": {
            "trial_seeds": base_reference["trial_seeds"],
            "loss": base_reference["loss"],
            "primary_metric": base_reference["primary_metric"],
            "secondary_metric": (
                base_reference["secondary_metric"] if num_metrics > 1 else None
            ),
            "time_ms": base_reference["time_ms"],
        },
        "targets": targets_summary,
    }


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args.tasks)
    output_dir = os.path.abspath(args.output_dir)
    noise_base_config_path = os.path.abspath(args.noise_base_config)
    approx_base_config_path = os.path.abspath(args.approx_base_config)
    manual_noise_config = parse_noise_config(args.manual_noise_config)

    if args.repeat_n <= 0:
        raise ValueError("--repeat_n must be a positive integer.")
    if args.repeat_n < 50:
        print(
            "[Warning] repeat_n < 50，当前更适合作为开发/冒烟测试，不建议作为正式实验结果。",
            flush=True,
        )
    if args.max_eval_samples < 0:
        raise ValueError("--max_eval_samples must be >= 0.")

    resolved_device = args.device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA unavailable, falling back to CPU.", flush=True)
        resolved_device = "cpu"

    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(args.seed)

    print("=" * 68, flush=True)
    print("Noise Scaling Sweep Experiment", flush=True)
    print("=" * 68, flush=True)
    print(f"Tasks               : {tasks}", flush=True)
    print(f"Output dir          : {output_dir}", flush=True)
    print(f"Device              : {resolved_device}", flush=True)
    print(f"Batch size          : {args.batch_size}", flush=True)
    print(f"Max length          : {args.max_length}", flush=True)
    print(f"Eval split          : {args.eval_split}", flush=True)
    print(f"Repeat per point    : {args.repeat_n}", flush=True)
    print(f"Max eval samples    : {args.max_eval_samples}", flush=True)
    print(f"Noise base source   : {args.noise_base_source}", flush=True)
    print(f"Noise base config   : {noise_base_config_path}", flush=True)
    print(f"Approx base config  : {approx_base_config_path}", flush=True)
    print("=" * 68, flush=True)

    for dataset_idx, task_name in enumerate(tasks):
        print(f"\n[Dataset] Starting {task_name.upper()} ...", flush=True)
        evaluator = None
        try:
            evaluator = build_evaluator(
                task_name=task_name,
                device=resolved_device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                output_dir=output_dir,
                approx_base_config_path=approx_base_config_path,
                noise_base_config_path=noise_base_config_path,
                eval_split=args.eval_split,
                seed=args.seed,
            )

            if not evaluator.has_dataset_split(args.eval_split):
                available = sorted(evaluator.dataloaders.keys())
                raise ValueError(
                    f"Eval split '{args.eval_split}' is unavailable for task '{task_name}'. "
                    f"Available splits: {available}"
                )

            maybe_limit_eval_split(
                evaluator=evaluator,
                split_name=args.eval_split,
                max_eval_samples=args.max_eval_samples,
                seed=args.seed + dataset_idx,
            )

            fixed_gelu, fixed_softmax = resolve_fixed_approx_config(
                dataset_key=task_name,
                config_path=approx_base_config_path,
                total_layers=evaluator.total_layers,
            )
            base_noise_config = resolve_base_noise_config(
                dataset_key=task_name,
                total_layers=evaluator.total_layers,
                source=args.noise_base_source,
                config_path=noise_base_config_path,
                manual_noise_config=manual_noise_config,
            )

            evaluator.apply_configuration(fixed_gelu, fixed_softmax)

            base_reference, target_results = run_task_sweep(
                evaluator=evaluator,
                dataset_key=task_name,
                dataset_idx=dataset_idx,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                base_noise_config=base_noise_config,
                split_name=args.eval_split,
                repeat_n=args.repeat_n,
                seed=args.seed,
            )

            metric_short_names = evaluator.get_metric_short_names()
            num_metrics = evaluator.get_num_metrics()
            summary = build_task_json_summary(
                dataset_key=task_name,
                split_name=args.eval_split,
                repeat_n=args.repeat_n,
                max_eval_samples=args.max_eval_samples,
                noise_base_source=args.noise_base_source,
                noise_base_config_path=noise_base_config_path,
                approx_base_config_path=approx_base_config_path,
                fixed_gelu=fixed_gelu,
                fixed_softmax=fixed_softmax,
                base_noise_config=base_noise_config,
                base_reference=base_reference,
                metric_short_names=metric_short_names,
                num_metrics=num_metrics,
                target_results=target_results,
            )

            json_path = os.path.join(output_dir, f"noise_scaling_sweep_{task_name}.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)

            plot_path = os.path.join(output_dir, f"noise_scaling_sweep_{task_name}.png")
            plot_task_results(
                dataset_key=task_name,
                split_name=args.eval_split,
                repeat_n=args.repeat_n,
                noise_source=args.noise_base_source,
                noise_config_path=noise_base_config_path,
                approx_config_path=approx_base_config_path,
                metric_short_names=metric_short_names,
                num_metrics=num_metrics,
                target_results=target_results,
                base_reference=base_reference,
                output_path=plot_path,
            )

            print(f"[Dataset] Finished {task_name.upper()}", flush=True)
            print(f"  JSON: {json_path}", flush=True)
            print(f"  Plot: {plot_path}", flush=True)
        finally:
            if evaluator is not None:
                try:
                    evaluator.clear_weight_noise_configuration()
                    evaluator.clear_input_noise_configuration()
                    evaluator.reversible_handler.restore_all()
                except Exception:
                    pass
                del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll noise scaling sweep experiments finished.", flush=True)


if __name__ == "__main__":
    main()
