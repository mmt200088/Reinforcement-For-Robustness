"""
rl_tune_general.py — 通用 RL 策略的命令行入口脚本
=================================================
训练和搜索完全分离:
  train   多任务 round-robin 训练通用策略 + Critic → 输出到持久化目录
  search  利用已训练的通用策略做离线 rollout 搜索最优配置（网络冻结不变）

持久化目录结构:
  rl_results/persistent/general-rl/{model_type}/{taskset}/{accuracy_slug}/
  - 同一数据集 + 同一准确度区间 → 同一目录（自动续训练）
  - 不同区间 → 不同目录

用法:
  # 训练（输出到持久化目录，可续训练）
  python rl_tune_general.py train --model_type bert-base --data_path mrpc,cola,rte,stsb ...
  # 搜索（手动指定持久化目录或策略文件路径）
  python rl_tune_general.py search --data_path mrpc --general_policy_dir rl_results/persistent/general-rl/bert-base/cola_mrpc_rte_stsb/default
  python rl_tune_general.py search --data_path mrpc --general_stage1_policy path/to/policy.pt
"""

import os
import sys
import json
import numpy as np
from typing import List, Optional, Union

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer,
    DataCollatorWithPadding,
)
from runtime_error_reporter import run_fire_entrypoint

sys.path.append(os.path.join(os.getcwd(), "./importance-aware-sparse-tuning-IST-paper/peft/src/"))


# ---------------------------------------------------------------------------
# Helpers (shared with rl_tune_genetic.py)
# ---------------------------------------------------------------------------

def parse_degree_config(raw_value):
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, (list, tuple)):
        return [int(item) for item in raw_value]
    text = str(raw_value).strip()
    if not text:
        return None
    if text.startswith("["):
        return [int(item) for item in json.loads(text)]
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_noise_config(raw_value):
    if raw_value is None or raw_value == "":
        return None
    if isinstance(raw_value, dict):
        return raw_value
    text = str(raw_value).strip()
    if not text:
        return None
    return json.loads(text)


def parse_bool_flag(raw_value, flag_name):
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return False
    text = str(raw_value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on"):
        return True
    if text in ("0", "false", "f", "no", "n", "off", ""):
        return False
    raise ValueError(
        f"Invalid boolean value for {flag_name}: {raw_value!r}. "
        "Expected one of: true/false/1/0/yes/no."
    )


def parse_positive_int(raw_value, flag_name):
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid positive integer for {flag_name}: {raw_value!r}."
        ) from None
    if value <= 0:
        raise ValueError(
            f"Invalid positive integer for {flag_name}: {raw_value!r}."
        )
    return value


# ---------------------------------------------------------------------------
# Dataset → BASE_MODEL mapping (mirrors llama_7B_LayerImportance.sh)
# ---------------------------------------------------------------------------
_BERT_BASE_MAP = {
    "wnli": "textattack/bert-base-uncased-WNLI",
    "rte":  "textattack/bert-base-uncased-RTE",
    "cola": "textattack/bert-base-uncased-CoLA",
    "qnli": "textattack/bert-base-uncased-QNLI",
    "mrpc": "textattack/bert-base-uncased-MRPC",
    "sst2": "textattack/bert-base-uncased-SST-2",
    "stsb": "textattack/bert-base-uncased-STS-B",
}
_BERT_LARGE_MAP = {
    "mrpc": "yoshitomo-matsubara/bert-large-uncased-mrpc",
    "cola": "yoshitomo-matsubara/bert-large-uncased-cola",
    "stsb": "yoshitomo-matsubara/bert-large-uncased-stsb",
    "rte":  "yoshitomo-matsubara/bert-large-uncased-rte",
    "sst2": "yoshitomo-matsubara/bert-large-uncased-sst2",
    "qnli": "yoshitomo-matsubara/bert-large-uncased-qnli",
}
_GPT2_MAP = {
    "cola": "PavanNeerudu/gpt2-finetuned-cola",
    "sst2": "PavanNeerudu/gpt2-finetuned-sst2",
    "mrpc": "PavanNeerudu/gpt2-finetuned-mrpc",
    "stsb": "PavanNeerudu/gpt2-finetuned-stsb",
    "qnli": "PavanNeerudu/gpt2-finetuned-qnli",
    "rte":  "PavanNeerudu/gpt2-finetuned-rte",
    "wnli": "PavanNeerudu/gpt2-finetuned-wnli",
}


def resolve_base_model(model_type: str, dataset: str) -> str:
    """Resolve (model_type, dataset) → HuggingFace checkpoint name."""
    mt = model_type.lower().replace("_", "-")
    ds = dataset.lower()
    if mt in ("bert-base", "bertbase"):
        m = _BERT_BASE_MAP.get(ds)
    elif mt in ("bert-large", "bertlarge"):
        m = _BERT_LARGE_MAP.get(ds)
    elif mt in ("gpt-2", "gpt2"):
        m = _GPT2_MAP.get(ds)
    else:
        raise ValueError(f"Unsupported model-type: {model_type}")
    if m is None:
        raise ValueError(
            f"model-type={model_type} 不支持数据集 {dataset}."
        )
    return m


# ---------------------------------------------------------------------------
# 创建 evaluator 的辅助函数
# ---------------------------------------------------------------------------

def _build_evaluator(
    base_model: str,
    data_path: str,
    batch_size: int,
    rl_lr: float,
    degree: int,
    run_output_dir: str,
    stage1_rl_episodes: int = 51000,
    stage2_rl_episodes: int = 40000,
    val_set_size: int = 120,
    cutoff_len: int = 256,
    lora_r: int = 32,
    lora_alpha: int = 64,
    load_8bit: bool = False,
):
    """创建并返回 (model, evaluator, data_collator)。"""
    if 'llama' in base_model and 'llama3' not in base_model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    config = AutoConfig.from_pretrained(base_model)
    _dp = data_path.lower()
    if _dp == "stsb":
        _num_labels = 1
    elif _dp == "mnli":
        _num_labels = 3
    else:
        _num_labels = 2

    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=_num_labels,
        device_map=device_map,
        trust_remote_code=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    for _param in model.parameters():
        _param.requires_grad_(False)
    model.eval()
    model.to("cuda")

    # Tokenize
    def tokenize_fn(examples):
        dp_lower = data_path.lower()
        if dp_lower in ("sst2", "cola"):
            return tokenizer(
                examples["sentence"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        elif dp_lower == "qnli":
            return tokenizer(
                examples["question"], examples["sentence"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        elif dp_lower == "mnli":
            return tokenizer(
                examples["premise"], examples["hypothesis"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        else:
            return tokenizer(
                examples["sentence1"], examples["sentence2"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )

    data = load_dataset("nyu-mll/glue", data_path)

    val_data_mm = None
    is_mnli = data_path.lower() == "mnli"
    if is_mnli:
        train_data = data["train"].shuffle().map(tokenize_fn)
        val_data = data["validation_matched"].shuffle().map(tokenize_fn)
        val_data_mm = data["validation_mismatched"].shuffle().map(tokenize_fn)
        train_data = train_data.rename_column("label", "labels")
        val_data = val_data.rename_column("label", "labels")
        val_data_mm = val_data_mm.rename_column("label", "labels")
        columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        train_data.set_format(type="torch", columns=columns)
        val_data.set_format(type="torch", columns=columns)
        val_data_mm.set_format(type="torch", columns=columns)
    else:
        train_data = data["train"].shuffle().map(tokenize_fn)
        val_data = data["validation"].shuffle().map(tokenize_fn)
        train_data = train_data.rename_column("label", "labels")
        val_data = val_data.rename_column("label", "labels")
        columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
        train_data.set_format(type="torch", columns=columns)
        val_data.set_format(type="torch", columns=columns)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )

    from layer_importance_evaluator import LayerImportanceEvaluator

    evaluator = LayerImportanceEvaluator(
        model=model,
        train_data=train_data,
        test_data=val_data,
        data_collator=data_collator,
        batch_size=batch_size,
        rl_lr=rl_lr,
        degree=degree,
        stage1_rl_episodes=stage1_rl_episodes,
        stage2_rl_episodes=stage2_rl_episodes,
        run_output_dir=run_output_dir,
        final_eval_config_source="search",
        final_eval_config_path="",
        skip_noise_rl=False,
        noise_eval_config_source="search",
        noise_eval_config_path="",
        skip_stage1_rl=False,
        skip_stage1_final_eval=True,
        skip_noise_final_eval=True,
        data_path=data_path,
        test_data_mm=val_data_mm,
        search_algorithm="general-rl",
    )
    model.config.use_cache = False
    model.config.is_decoder = False

    return model, evaluator, data_collator


# ===========================================================================
# 子命令: train — 多任务 round-robin 训练
# ===========================================================================

def train(
    # 基础参数
    base_model: str = "",
    data_path: str = "mrpc,cola,rte,stsb",
    model_type: str = "bert-base",
    output_dir: str = "./general_rl_output",
    batch_size: int = 16,
    rl_lr: float = 1e-4,
    degree: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    # 通用策略训练参数
    total_rounds: int = 50,
    # ppo_update_interval 同时决定：PPO 更新间隔 / 每轮每任务的 episode 数 / details 分块大小(=3×)
    ppo_update_interval: int = 120,
    general_lr: float = 3e-5,
    # Stage-1/Stage-2 选项
    skip_stage2: bool = False,
    stage1_output: str = "",
    stage2_output: str = "",
    # Stage-1 固定配置（用于 Stage-2 训练）
    stage1_config_json: str = "",
    # 准确度容忍泛化
    accuracy_tolerances: str = "",
    accuracy_tolerance_range: str = "",
    # 续训练
    resume_from: str = "",
    # 设备
    device: str = "cuda",
):
    """多任务 round-robin 训练通用策略 + Critic。

    必须提供多个数据集用逗号分隔: --data_path mrpc,cola,rte,stsb

    准确度容忍泛化有两种模式:
      --accuracy_tolerances 0.005,0.01,0.02  离散采样
      --accuracy_tolerance_range 0.005,0.02  连续均匀采样 [0.5%, 2%]
    连续模式优先级高于离散模式。
    """
    skip_stage2 = parse_bool_flag(skip_stage2, "skip_stage2")
    batch_size = parse_positive_int(batch_size, "batch_size")
    total_rounds = parse_positive_int(total_rounds, "total_rounds")
    ppo_update_interval = parse_positive_int(
        ppo_update_interval, "ppo_update_interval"
    )
    # 覆盖 PPO 更新间隔及其派生常量, 必须在构建 evaluator / 启动 multi_task_train 之前
    import layer_importance_evaluator as _lie
    _lie.set_ppo_update_interval(ppo_update_interval)
    print(
        f"[PPO] ppo_update_interval={_lie.PPO_UPDATE_INTERVAL} "
        f"(batch={_lie.PPO_BATCH_SIZE} steps, details chunk={_lie.STEP_INFO_CHUNK_SIZE} episodes)"
    )
    # 每轮每任务的 episode 数与 PPO 更新间隔保持相等（确保 buffer 不跨任务混合）
    episodes_per_task_per_round = ppo_update_interval

    # 解析准确度容忍列表（离散模式）
    parsed_tolerances = None
    if accuracy_tolerances and str(accuracy_tolerances).strip():
        _raw = str(accuracy_tolerances).strip()
        try:
            parsed_tolerances = [float(x.strip()) for x in _raw.split(",") if x.strip()]
        except ValueError:
            raise ValueError(
                f"--accuracy_tolerances 格式错误: {_raw!r}. "
                "请提供逗号分隔的浮点数, 如 '0.005,0.01,0.02'"
            )
        for t in parsed_tolerances:
            if t <= 0 or t >= 1:
                raise ValueError(
                    f"--accuracy_tolerances 中的值必须在 (0, 1) 区间, "
                    f"当前值 {t} 不合法. 例如 0.01 表示 1%."
                )
        print(f"准确度容忍泛化(离散): {[f'{t*100:.1f}%' for t in parsed_tolerances]}")

    # 解析准确度容忍范围（连续模式）
    parsed_tolerance_range = None
    if accuracy_tolerance_range and str(accuracy_tolerance_range).strip():
        _raw_range = str(accuracy_tolerance_range).strip()
        try:
            _parts = [float(x.strip()) for x in _raw_range.split(",") if x.strip()]
        except ValueError:
            raise ValueError(
                f"--accuracy_tolerance_range 格式错误: {_raw_range!r}. "
                "请提供两个逗号分隔的浮点数, 如 '0.005,0.02'"
            )
        if len(_parts) != 2:
            raise ValueError(
                f"--accuracy_tolerance_range 需要恰好两个值 (min,max), "
                f"当前提供了 {len(_parts)} 个值."
            )
        if _parts[0] <= 0 or _parts[1] >= 1 or _parts[0] >= _parts[1]:
            raise ValueError(
                f"--accuracy_tolerance_range 要求 0 < min < max < 1, "
                f"当前值 ({_parts[0]}, {_parts[1]}) 不合法."
            )
        parsed_tolerance_range = tuple(_parts)
        print(f"准确度容忍泛化(连续): [{_parts[0]*100:.1f}%, {_parts[1]*100:.1f}%]")

    resume_run_dir = resume_from.strip() if resume_from else ""
    os.makedirs(output_dir, exist_ok=True)

    from general_policy_module import (
        prepare_stage1_task,
        multi_task_train_stage1,
        prepare_stage2_task,
        multi_task_train_stage2,
        GENERAL_STAGE1_TRAIN_CHECKPOINT,
        GENERAL_STAGE2_TRAIN_CHECKPOINT,
    )

    # ---- 续训练 checkpoint 路径检测 ----
    # 优先使用显式 resume_from；否则自动检测 output_dir 中的 checkpoint
    general_s1_resume_path = None
    general_s2_resume_path = None
    _search_dirs = []
    if resume_run_dir:
        _search_dirs.append(resume_run_dir)
    # 持久化目录模式下 output_dir == resume_run_dir，
    # 但如果 resume_from 未指定，也尝试在 output_dir 中查找 checkpoint
    if output_dir and output_dir not in _search_dirs:
        _search_dirs.append(output_dir)
    for _dir in _search_dirs:
        if general_s1_resume_path is None:
            _s1_ckpt = os.path.join(_dir, GENERAL_STAGE1_TRAIN_CHECKPOINT)
            if os.path.isfile(_s1_ckpt):
                general_s1_resume_path = _s1_ckpt
                print(f"[续训练] 检测到 Stage-1 checkpoint: {_s1_ckpt}")
        if general_s2_resume_path is None:
            _s2_ckpt = os.path.join(_dir, GENERAL_STAGE2_TRAIN_CHECKPOINT)
            if os.path.isfile(_s2_ckpt):
                general_s2_resume_path = _s2_ckpt
                print(f"[续训练] 检测到 Stage-2 checkpoint: {_s2_ckpt}")

    task_names = [t.strip().lower() for t in data_path.split(",") if t.strip()]
    if len(task_names) < 1:
        raise ValueError("--data_path 至少需要提供一个数据集名称。")

    # ---- Stage-1 训练 ----
    s1_output = stage1_output or os.path.join(output_dir, "general_stage1_policy.pt")
    print(f"\n{'='*60}")
    print(f"[Stage-1] 准备多任务训练...")
    print(f"[Stage-1] 输出路径: {s1_output}")
    print(f"{'='*60}")

    s1_tasks = {}
    evaluators = {}
    for name in task_names:
        bm = base_model if base_model else resolve_base_model(model_type, name)
        print(f"  准备任务 {name} (base_model={bm}) ...")
        _, ev, _ = _build_evaluator(
            base_model=bm,
            data_path=name,
            batch_size=batch_size,
            rl_lr=rl_lr,
            degree=degree,
            run_output_dir=os.path.join(output_dir, f"task_{name}"),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        evaluators[name] = ev
        s1_tasks[name] = prepare_stage1_task(ev, use_train=True)
        print(f"  任务 {name} 准备完毕. task_context={s1_tasks[name]['task_context']}")

    multi_task_train_stage1(
        tasks=s1_tasks,
        output_path=s1_output,
        total_rounds=total_rounds,
        episodes_per_task_per_round=episodes_per_task_per_round,
        lr=general_lr,
        log_fn=print,
        device=device,
        resume_checkpoint_path=general_s1_resume_path,
        accuracy_tolerances=parsed_tolerances,
        accuracy_tolerance_range=parsed_tolerance_range,
    )
    print(f"\n[Stage-1] 训练完成, 策略已保存至: {s1_output}")

    # ---- Stage-2 训练 ----
    if skip_stage2:
        print("[Stage-2] 已跳过 (--skip_stage2)")
        return {"stage1_policy_path": s1_output}

    s2_output = stage2_output or os.path.join(output_dir, "general_stage2_noise_policy.pt")
    print(f"\n{'='*60}")
    print(f"[Stage-2] 准备多任务噪声策略训练...")
    print(f"[Stage-2] 输出路径: {s2_output}")
    print(f"{'='*60}")

    # 加载 Stage-1 配置（每个任务的 fixed_gelu / fixed_softmax）
    stage1_configs = {}
    if stage1_config_json:
        with open(stage1_config_json, "r") as f:
            stage1_configs = json.load(f)
        print(f"[Stage-2] 从 JSON 加载 Stage-1 配置: {stage1_config_json}")
    else:
        # 用通用策略 offline 推断每个任务的 Stage-1 配置
        from general_policy_module import offline_find_best_config_stage1
        print(f"[Stage-2] 使用 Stage-1 通用策略离线推断各任务的 GELU/Softmax 配置...")
        for name in task_names:
            result = offline_find_best_config_stage1(
                evaluator=evaluators[name],
                general_policy_path=s1_output,
                num_rollouts=500,
                greedy=False,
                device=device,
                log_fn=print,
            )
            stage1_configs[name] = {
                "gelu": result["best_config"]["gelu"].tolist(),
                "softmax": result["best_config"]["softmax"].tolist(),
            }
            print(f"  任务 {name}: best_reward={result['best_reward']:.4f}")

        # 保存推断结果
        s1_config_path = os.path.join(output_dir, "stage1_configs_from_general.json")
        with open(s1_config_path, "w") as f:
            json.dump(stage1_configs, f, indent=2)
        print(f"[Stage-2] Stage-1 配置已保存至: {s1_config_path}")

    s2_tasks = {}
    for name in task_names:
        cfg = stage1_configs[name]
        fixed_gelu = np.array(cfg["gelu"], dtype=int)
        fixed_softmax = np.array(cfg["softmax"], dtype=int)
        s2_tasks[name] = prepare_stage2_task(
            evaluators[name], fixed_gelu, fixed_softmax,
        )
        print(f"  任务 {name} Stage-2 准备完毕.")

    multi_task_train_stage2(
        tasks=s2_tasks,
        output_path=s2_output,
        total_rounds=total_rounds,
        episodes_per_task_per_round=episodes_per_task_per_round,
        lr=general_lr,
        log_fn=print,
        device=device,
        resume_checkpoint_path=general_s2_resume_path,
        accuracy_tolerances=parsed_tolerances,
        accuracy_tolerance_range=parsed_tolerance_range,
    )
    print(f"\n[Stage-2] 训练完成, 噪声策略已保存至: {s2_output}")

    return {
        "stage1_policy_path": s1_output,
        "stage2_policy_path": s2_output,
    }


# ===========================================================================
# 子命令: search — 离线 rollout 搜索（网络冻结不变）
# ===========================================================================

def search(
    # 基础参数
    base_model: str = "",
    data_path: str = "mrpc",
    model_type: str = "bert-base",
    output_dir: str = "./general_rl_search_output",
    batch_size: int = 16,
    rl_lr: float = 1e-4,
    degree: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    # 通用策略路径（二选一：显式指定文件 或 指定持久化目录自动推导）
    general_stage1_policy: str = "",
    general_stage2_policy: str = "",
    general_policy_dir: str = "",
    # 搜索参数
    num_rollouts: int = 500,
    greedy: bool = False,
    # Stage-1 固定配置（用于 Stage-2 搜索，可选）
    fixed_gelu: str = "",
    fixed_softmax: str = "",
    # 控制
    skip_stage2: bool = False,
    # 准确度容忍
    accuracy_tolerance: float = 0.0,
    # 最终评估
    skip_final_eval: bool = False,
    final_eval_random_seed: int = 42,
    final_eval_permutation_trials: int = 10,
    final_eval_cost_equivalent_trials: int = 10,
    final_eval_budget_equivalent_trials: int = 10,
    noise_eval_repeat_n: int = 1,
    # 设备
    device: str = "cuda",
):
    """使用已训练的通用策略对单个任务做离线 rollout 搜索最优配置。

    策略网络在搜索过程中完全冻结，不做任何梯度更新。

    策略来源（二选一）:
      --general_policy_dir PATH    指定训练持久化目录，自动推导策略文件
      --general_stage1_policy PATH 显式指定 Stage-1 策略文件

    --data_path 只需提供单个数据集名称。
    """
    skip_stage2 = parse_bool_flag(skip_stage2, "skip_stage2")
    skip_final_eval = parse_bool_flag(skip_final_eval, "skip_final_eval")
    greedy = parse_bool_flag(greedy, "greedy")
    batch_size = parse_positive_int(batch_size, "batch_size")
    num_rollouts = parse_positive_int(num_rollouts, "num_rollouts")

    # ---- 从 general_policy_dir 自动推导策略文件路径 ----
    if general_policy_dir:
        general_policy_dir = general_policy_dir.strip()
        if not os.path.isdir(general_policy_dir):
            raise ValueError(
                f"--general_policy_dir 指定的目录不存在: {general_policy_dir}"
            )
        if not general_stage1_policy:
            _auto_s1 = os.path.join(general_policy_dir, "general_stage1_policy.pt")
            if os.path.isfile(_auto_s1):
                general_stage1_policy = _auto_s1
                print(f"[搜索] 自动推导 Stage-1 策略: {_auto_s1}")
            else:
                raise ValueError(
                    f"在 --general_policy_dir 中未找到 Stage-1 策略文件: {_auto_s1}"
                )
        if not general_stage2_policy:
            _auto_s2 = os.path.join(general_policy_dir, "general_stage2_noise_policy.pt")
            if os.path.isfile(_auto_s2):
                general_stage2_policy = _auto_s2
                print(f"[搜索] 自动推导 Stage-2 策略: {_auto_s2}")

    if not general_stage1_policy:
        raise ValueError(
            "必须提供 --general_stage1_policy 或 --general_policy_dir 以指定策略来源。"
        )

    # 解析准确度容忍
    _infer_tol = None
    if accuracy_tolerance and float(accuracy_tolerance) > 0:
        _infer_tol = float(accuracy_tolerance)
        if _infer_tol >= 1:
            raise ValueError(
                f"--accuracy_tolerance 必须在 (0, 1) 区间, "
                f"当前值 {_infer_tol} 不合法. 例如 0.01 表示 1%."
            )

    os.makedirs(output_dir, exist_ok=True)

    task_name = data_path.strip().lower()
    if "," in task_name:
        raise ValueError("--data_path 搜索模式只支持单个数据集名称。")

    bm = base_model if base_model else resolve_base_model(model_type, task_name)
    print(f"[通用RL搜索] 任务: {task_name}, base_model={bm}")
    print(f"[通用RL搜索] rollouts={num_rollouts}, greedy={greedy}")
    if general_policy_dir:
        print(f"[通用RL搜索] 策略目录: {general_policy_dir}")
    print(f"[通用RL搜索] Stage-1 策略: {general_stage1_policy}")
    if general_stage2_policy:
        print(f"[通用RL搜索] Stage-2 策略: {general_stage2_policy}")
    if _infer_tol is not None:
        print(f"[通用RL搜索] 准确度容忍: {_infer_tol*100:.1f}%")

    _, evaluator, _ = _build_evaluator(
        base_model=bm,
        data_path=task_name,
        batch_size=batch_size,
        rl_lr=rl_lr,
        degree=degree,
        run_output_dir=output_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    from general_policy_module import (
        offline_find_best_config_stage1,
        offline_find_best_config_stage2,
        prepare_stage2_task,
    )
    import noise_rl_module_v2 as _s2

    # ---- Stage-1 离线推断 ----
    print(f"\n{'='*60}")
    print(f"[Stage-1] 离线 rollout: policy={general_stage1_policy}")
    print(f"{'='*60}")

    s1_result = offline_find_best_config_stage1(
        evaluator=evaluator,
        general_policy_path=general_stage1_policy,
        num_rollouts=num_rollouts,
        greedy=greedy,
        device=device,
        log_fn=print,
        tolerance=_infer_tol,
    )
    best_gelu = s1_result["best_config"]["gelu"]
    best_softmax = s1_result["best_config"]["softmax"]
    print(f"[Stage-1] 最优 reward={s1_result['best_reward']:.4f}")
    print(f"[Stage-1] GELU={best_gelu.tolist()}")
    print(f"[Stage-1] Softmax={best_softmax.tolist()}")

    # 保存 Stage-1 结果
    s1_result_path = os.path.join(output_dir, "stage1_infer_result.json")
    with open(s1_result_path, "w") as f:
        json.dump({
            "task": task_name,
            "best_reward": float(s1_result["best_reward"]),
            "best_gelu": best_gelu.tolist(),
            "best_softmax": best_softmax.tolist(),
            "num_rollouts": num_rollouts,
            "greedy": greedy,
        }, f, indent=2)
    print(f"[Stage-1] 结果已保存至: {s1_result_path}")

    result = {"stage1_result": s1_result_path}

    # ---- Stage-2 离线推断 ----
    if skip_stage2 or not general_stage2_policy:
        if not general_stage2_policy:
            print("[Stage-2] 未指定 --general_stage2_policy, 跳过 Stage-2 推断.")
        else:
            print("[Stage-2] 已跳过 (--skip_stage2)")
        return result

    # 使用 Stage-1 结果或手动指定的 fixed config
    fg = parse_degree_config(fixed_gelu) if fixed_gelu else best_gelu
    fs = parse_degree_config(fixed_softmax) if fixed_softmax else best_softmax
    fg = np.asarray(fg, dtype=int)
    fs = np.asarray(fs, dtype=int)

    print(f"\n{'='*60}")
    print(f"[Stage-2] 离线 rollout: policy={general_stage2_policy}")
    print(f"[Stage-2] fixed_gelu={fg.tolist()}")
    print(f"[Stage-2] fixed_softmax={fs.tolist()}")
    print(f"{'='*60}")

    s2_task = prepare_stage2_task(evaluator, fg, fs)
    noise_env = _s2._NoiseOptEnv(**s2_task["env_kwargs"])

    s2_result = offline_find_best_config_stage2(
        evaluator=evaluator,
        general_policy_path=general_stage2_policy,
        fixed_gelu=fg,
        fixed_softmax=fs,
        noise_env=noise_env,
        num_rollouts=num_rollouts,
        greedy=greedy,
        device=device,
        log_fn=print,
    )
    print(f"[Stage-2] 最优 noise reward={s2_result['best_reward']:.4f}")

    # 保存 Stage-2 结果
    s2_result_path = os.path.join(output_dir, "stage2_infer_result.json")
    best_nc = s2_result.get("best_noise_config", {})
    serializable_nc = {}
    for k, v in best_nc.items():
        if isinstance(v, np.ndarray):
            serializable_nc[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            serializable_nc[k] = v.item()
        else:
            serializable_nc[k] = v
    with open(s2_result_path, "w") as f:
        json.dump({
            "task": task_name,
            "best_reward": float(s2_result["best_reward"]),
            "best_noise_config": serializable_nc,
            "fixed_gelu": fg.tolist(),
            "fixed_softmax": fs.tolist(),
            "num_rollouts": num_rollouts,
            "greedy": greedy,
        }, f, indent=2)
    print(f"[Stage-2] 结果已保存至: {s2_result_path}")
    result["stage2_result"] = s2_result_path

    return result


# infer 保留为 search 的别名，兼容旧脚本调用
infer = search


# ===========================================================================
# 入口
# ===========================================================================

if __name__ == "__main__":
    run_fire_entrypoint(
        fire,
        {
            "train": train,
            "search": search,
            "infer": infer,  # 兼容别名
        },
        program_name="rl_tune_general.py",
    )
