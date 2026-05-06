import os
import sys
import json
import math
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
from runtime_error_reporter import run_fire_entrypoint
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "./importance-aware-sparse-tuning-IST-paper/peft/src/"))
# from peft import (  # noqa: E402
#     LoraConfig,
#     DoraConfig,
#     BottleneckConfig,
#     PrefixTuningConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, LlamaTokenizer, DataCollatorWithPadding, AutoModel  # noqa: F402


def seed_everything(seed: int) -> int:
    seed = int(seed)
    transformers.set_seed(seed)
    try:
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed



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


def parse_optional_positive_int(raw_value, flag_name):
    if raw_value is None or raw_value == "":
        return None
    return parse_positive_int(raw_value, flag_name)


def _load_prior_ga_search_results(
    run_dir,
    stage1_filename="ga_search_results.json",
    stage2_filename="noise_ga_search_results.json",
    algorithm_label="GA",
):
    """从 run_dir 读取之前 GA 运行保存的 stage1 / stage2 搜索结果。

    GA 会把最好配置写到:
      - {run_dir}/stage1/ga_search_results.json  → payload['best_config'] = {gelu, softmax, ...}
      - {run_dir}/stage2_noise/noise_ga_search_results.json → payload['best_config'] = {*_scaling_factors, ...}

    返回 ``(stage1_search_best, stage2_search_best)``，缺失项为 None。
    不存在文件或解析失败时只打印警告并返回 None，不抛错（final_eval 仍可回退到 json/manual）。
    """
    import numpy as np

    stage1_best = None
    stage2_best = None
    if not run_dir:
        return stage1_best, stage2_best

    stage1_path = os.path.join(run_dir, "stage1", stage1_filename)
    if os.path.isfile(stage1_path):
        try:
            with open(stage1_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            cfg = payload.get("best_config")
            if cfg and "gelu" in cfg and "softmax" in cfg:
                stage1_best = {
                    "gelu": np.asarray(cfg["gelu"], dtype=int),
                    "softmax": np.asarray(cfg["softmax"], dtype=int),
                }
                print(f"[final_eval_only] 加载 Stage-1 {algorithm_label} 搜索结果: {stage1_path}")
        except Exception as exc:  # pragma: no cover - diagnostics path
            print(f"[final_eval_only][警告] 读取 {stage1_path} 失败: {exc}")

    stage2_path = os.path.join(run_dir, "stage2_noise", stage2_filename)
    if os.path.isfile(stage2_path):
        try:
            with open(stage2_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            cfg = payload.get("best_config") or payload.get("best_noise_config")
            if cfg:
                stage2_best = {
                    key: np.asarray(value, dtype=int)
                    for key, value in cfg.items()
                    if isinstance(key, str) and key.endswith("scaling_factors")
                }
                if stage2_best:
                    print(f"[final_eval_only] 加载 Stage-2 {algorithm_label} 搜索结果: {stage2_path}")
                else:
                    stage2_best = None
        except Exception as exc:  # pragma: no cover - diagnostics path
            print(f"[final_eval_only][警告] 读取 {stage2_path} 失败: {exc}")

    return stage1_best, stage2_best


def infer_model_total_layers(model):
    config = getattr(model, "config", None)
    for attr_name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(config, attr_name, None)
        if value is not None:
            return parse_positive_int(value, f"model.config.{attr_name}")
    raise ValueError("Unable to infer total layer count from model.config.")


def stage1_ga_population_size(total_layers):
    return max(32, int(total_layers) * 2)


def stage2_ga_population_size(total_layers):
    return max(32, int(total_layers))


def resolve_ga_generation_budget(
    *,
    generation_raw,
    generation_specified,
    generation_flag_name,
    legacy_episode_raw,
    legacy_episode_specified,
    legacy_episode_flag_name,
    population_size,
):
    generation_value = parse_optional_positive_int(generation_raw, generation_flag_name)
    generation_is_explicit = bool(generation_specified) or generation_value is not None
    legacy_episode_value = parse_positive_int(legacy_episode_raw, legacy_episode_flag_name)

    if generation_is_explicit and legacy_episode_specified:
        raise ValueError(
            f"Do not specify both {generation_flag_name} and {legacy_episode_flag_name}."
        )

    if generation_is_explicit:
        if generation_value is None:
            raise ValueError(f"{generation_flag_name} must be provided as a positive integer.")
        return generation_value, True, generation_flag_name

    resolved_generations = max(
        1,
        math.ceil(legacy_episode_value / max(int(population_size), 1)),
    )
    source = legacy_episode_flag_name if legacy_episode_specified else "default"
    return resolved_generations, bool(legacy_episode_specified), source


def normalize_ga_skip_budget_flags(
    *,
    skip_stage1_rl,
    skip_noise_rl,
    stage1_ga_generations,
    stage2_ga_generations,
    stage1_budget_specified,
    stage2_budget_specified,
    stage1_rl_episodes,
    stage2_rl_episodes,
):
    if skip_stage1_rl and stage1_budget_specified:
        raise ValueError(
            "skip_stage1_rl=True 时不能再显式设置 Stage-1 GA 搜索预算。"
            "请移除 --stage1-search-generations，或取消 --skip-stage1-search。"
        )
    if skip_noise_rl and stage2_budget_specified:
        raise ValueError(
            "skip_noise_rl=True 时不能再显式设置 Stage-2 GA 搜索预算。"
            "请移除 --stage2-search-generations，或取消 --skip-noise-search。"
        )

    stage1_budget_source = "skipped" if skip_stage1_rl else None
    stage2_budget_source = "skipped" if skip_noise_rl else None
    if skip_stage1_rl:
        stage1_rl_episodes = 51000
        stage1_budget_specified = False
    if skip_noise_rl:
        stage2_rl_episodes = 40000
        stage2_budget_specified = False

    return {
        "stage1_ga_generations": int(stage1_ga_generations),
        "stage2_ga_generations": int(stage2_ga_generations),
        "stage1_budget_specified": bool(stage1_budget_specified),
        "stage2_budget_specified": bool(stage2_budget_specified),
        "stage1_rl_episodes": int(stage1_rl_episodes),
        "stage2_rl_episodes": int(stage2_rl_episodes),
        "stage1_budget_source": stage1_budget_source,
        "stage2_budget_source": stage2_budget_source,
    }


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        adapter_name: str = "lora",
        load_8bit: bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = True,
        use_adapterp: bool = False,
        target_modules: List[str] = None,
        # Dora hyperparams
        Wdecompose_target_modules: List[str] = None,
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # Sparse tuning hyperparams
        use_ist: bool = False,
        use_rst: bool = False,
        search_backend: str = "ga",
        rl_lr: float = 1e-4, 
        degree: int = 4,  # degree of polynomial for approximation
        stage1_ga_generations: Optional[int] = None,
        stage2_ga_generations: Optional[int] = None,
        stage1_ga_generations_specified: bool = False,
        stage2_ga_generations_specified: bool = False,
        stage1_rl_episodes: int = 51000,
        stage2_rl_episodes: int = 40000,
        stage1_rl_episodes_specified: bool = False,
        stage2_rl_episodes_specified: bool = False,
        final_eval_config_source: str = "search",  # search | json | manual | max
        final_eval_config_path: str = "glue_final_configs_best_genetic.json",
        manual_stage1_gelu: str = "",
        manual_stage1_softmax: str = "",
        manual_stage2_noise: str = "",
        stage2_fixed_config_source: str = "",
        stage2_fixed_config_path: str = "",
        stage2_manual_gelu: str = "",
        stage2_manual_softmax: str = "",
        final_eval_random_seed: int = 42,
        final_eval_permutation_trials: int = 10,
        final_eval_cost_equivalent_trials: int = 10,
        final_eval_budget_equivalent_trials: int = 10,
        final_eval_stage1_budget_trials: int = 10,
        final_eval_stage2_budget_trials: int = 10,
        final_eval_repeat_n: int = 1,
        final_eval_preset: str = "default",
        final_eval_output_root: str = "",
        final_eval_run_name: str = "",
        final_eval_random_enabled: bool = False,
        final_eval_action_config: str = "",
        final_eval_action_ranges: str = "",
        final_eval_action_fixed: str = "",
        skip_noise_rl: bool = False,
        skip_stage1_rl: bool = False,
        skip_final_eval: bool = False,
        final_eval_only: bool = False,
        resume_run_dir: str = "",
        # accuracy constraint params
        stage1_accuracy_tolerance: float = None,
        stage2_limit_tolerance: float = None,
        stage2_stability_tolerance: float = None,
        stage2_k_trials: int = None,
        stage2_probe_size: int = None,
        # Stage-2 RL variant (新版 BLB v3 / 旧版 v2 二选一；默认新版)
        stage2_rl_variant: str = "blb_v3",
        blb_v3_rescale_invoker_kind: str = "heuristic",
        blb_v3_subprocess_optimizer_root: str = "",
        blb_v3_subprocess_cli_module: str = "rescale_optimizer.replan",
        blb_v3_rollout_size: int = None,
        blb_v3_eval_interval: int = None,
        blb_v3_save_interval: int = None,
        blb_v3_calibrate_baseline_samples: int = None,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    skip_noise_rl = parse_bool_flag(skip_noise_rl, "skip_noise_rl")
    skip_stage1_rl = parse_bool_flag(skip_stage1_rl, "skip_stage1_rl")
    skip_final_eval = parse_bool_flag(skip_final_eval, "skip_final_eval")
    final_eval_only = parse_bool_flag(final_eval_only, "final_eval_only")
    final_eval_random_enabled = parse_bool_flag(
        final_eval_random_enabled, "final_eval_random_enabled"
    )
    search_backend = str(search_backend or "ga").strip().lower().replace("-", "_")
    if search_backend in ("ga", "genetic"):
        search_backend = "ga"
    elif search_backend in ("greedy", "greedy_search"):
        search_backend = "greedy"
    else:
        raise ValueError(
            f"Unsupported search_backend={search_backend!r}. Expected 'ga' or 'greedy'."
        )
    # --final_eval_only 语义：只跑 final eval，不跑任何搜索阶段。
    # 这等价于同时设置 skip_stage1_rl=True & skip_noise_rl=True & skip_final_eval=False，
    # 但额外支持从 resume_run_dir（或 output_dir）加载之前保存的搜索结果作为 search 源。
    # 由于不进入任何搜索循环，优雅停止 handler 不会被安装，任何训练 checkpoint 也不会被读写。
    if final_eval_only:
        if skip_final_eval:
            raise ValueError(
                "final_eval_only=True 与 skip_final_eval=True 冲突：无可执行项。"
            )
        if not skip_stage1_rl:
            print("[final_eval_only] 自动设置 skip_stage1_rl=True")
            skip_stage1_rl = True
        if not skip_noise_rl:
            print("[final_eval_only] 自动设置 skip_noise_rl=True")
            skip_noise_rl = True
    stage1_ga_generations_specified = parse_bool_flag(
        stage1_ga_generations_specified, "stage1_ga_generations_specified"
    )
    stage2_ga_generations_specified = parse_bool_flag(
        stage2_ga_generations_specified, "stage2_ga_generations_specified"
    )
    stage1_rl_episodes_specified = parse_bool_flag(
        stage1_rl_episodes_specified, "stage1_rl_episodes_specified"
    )
    stage2_rl_episodes_specified = parse_bool_flag(
        stage2_rl_episodes_specified, "stage2_rl_episodes_specified"
    )
    batch_size = parse_positive_int(batch_size, "batch_size")
    micro_batch_size = parse_positive_int(micro_batch_size, "micro_batch_size")
    stage1_ga_generations = parse_optional_positive_int(
        stage1_ga_generations, "stage1_ga_generations"
    )
    stage2_ga_generations = parse_optional_positive_int(
        stage2_ga_generations, "stage2_ga_generations"
    )
    stage1_rl_episodes = parse_positive_int(
        stage1_rl_episodes, "stage1_rl_episodes"
    )
    stage2_rl_episodes = parse_positive_int(
        stage2_rl_episodes, "stage2_rl_episodes"
    )

    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"Wdecompose_target_modules: {Wdecompose_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"search_backend: {search_backend}\n"
        f"target_modules: {target_modules}\n"
        f"final_eval_config_source: {final_eval_config_source}\n"
        f"final_eval_config_path: {final_eval_config_path}\n"
        f"manual_stage1_gelu: {manual_stage1_gelu}\n"
        f"manual_stage1_softmax: {manual_stage1_softmax}\n"
        f"manual_stage2_noise: {manual_stage2_noise}\n"
        f"stage1_ga_generations: {stage1_ga_generations}\n"
        f"stage2_ga_generations: {stage2_ga_generations}\n"
        f"stage1_ga_generations_specified: {stage1_ga_generations_specified}\n"
        f"stage2_ga_generations_specified: {stage2_ga_generations_specified}\n"
        f"stage1_rl_episodes: {stage1_rl_episodes}\n"
        f"stage2_rl_episodes: {stage2_rl_episodes}\n"
        f"stage1_rl_episodes_specified: {stage1_rl_episodes_specified}\n"
        f"stage2_rl_episodes_specified: {stage2_rl_episodes_specified}\n"
        f"skip_noise_rl: {skip_noise_rl}\n"
        f"final_eval_repeat_n: {final_eval_repeat_n}\n"
        f"final_eval_preset: {final_eval_preset}\n"
        f"final_eval_output_root: {final_eval_output_root}\n"
        f"final_eval_run_name: {final_eval_run_name}\n"
        f"final_eval_random_enabled: {final_eval_random_enabled}\n"
        f"final_eval_action_config: {final_eval_action_config}\n"
        f"final_eval_action_ranges: {final_eval_action_ranges}\n"
        f"final_eval_action_fixed: {final_eval_action_fixed}\n"
        f"skip_stage1_rl: {skip_stage1_rl}\n"
        f"skip_final_eval: {skip_final_eval}\n"
        f"final_eval_only: {final_eval_only}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"resume_run_dir: {resume_run_dir}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    run_output_dir = str(output_dir or "").strip()
    trainer_output_dir = (
        os.path.join(run_output_dir, "trainer_output")
        if run_output_dir
        else "./inference_output"
    )
    os.makedirs(trainer_output_dir, exist_ok=True)
    seed_everything(final_eval_random_seed)

    # device_map = "gpu"
    ddp = True  # Distributed Data Parallelism disabled

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if 'llama' in base_model and 'llama3' not in base_model:
        # Due to the name of transformers' LlamaTokenizer, we have to do this
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    if load_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=quantization_config,
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        # config.use_causal_lm = False  # Key: disable causal mask for MRPC.
        _dp = data_path.lower()
        if _dp == "stsb":
            _num_labels = 1
        elif _dp == "mnli":
            _num_labels = 3
        else:
            _num_labels = 2
        print(f"Auto-detected num_labels={_num_labels} for dataset '{data_path}'")

        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=_num_labels,
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            # device_map ="cpu",
            trust_remote_code=True,
            # pad_token_id=tokenizer.eos_token_id
            pad_token_id=tokenizer.pad_token_id,
        )

    # ---------------------------------------------------------------
    # Freeze the backbone. The downstream pipeline (layer_importance_
    # evaluator.py + noise_rl_module_v2.py) only uses this HF model for
    # **inference** to compute rewards — the PPO policy/value networks
    # are the only thing being trained. Explicitly disabling grads on
    # every parameter and pinning the model to eval() makes that
    # contract bulletproof: no amount of noise-wrapping, function-
    # replacement or stray autograd call can push an update into the
    # pretrained weights, and dropout/train-mode side-effects cannot
    # add variance to the reward signal mid-episode.
    # ---------------------------------------------------------------
    for _param in model.parameters():
        _param.requires_grad_(False)
    model.eval()

    model.to("cuda")

    total_layers = infer_model_total_layers(model)
    stage1_population_size = stage1_ga_population_size(total_layers)
    stage2_population_size = stage2_ga_population_size(total_layers)
    stage1_ga_generations, stage1_budget_specified, stage1_budget_source = (
        resolve_ga_generation_budget(
            generation_raw=stage1_ga_generations,
            generation_specified=stage1_ga_generations_specified,
            generation_flag_name="stage1_ga_generations",
            legacy_episode_raw=stage1_rl_episodes,
            legacy_episode_specified=stage1_rl_episodes_specified,
            legacy_episode_flag_name="stage1_rl_episodes",
            population_size=stage1_population_size,
        )
    )
    stage2_ga_generations, stage2_budget_specified, stage2_budget_source = (
        resolve_ga_generation_budget(
            generation_raw=stage2_ga_generations,
            generation_specified=stage2_ga_generations_specified,
            generation_flag_name="stage2_ga_generations",
            legacy_episode_raw=stage2_rl_episodes,
            legacy_episode_specified=stage2_rl_episodes_specified,
            legacy_episode_flag_name="stage2_rl_episodes",
            population_size=stage2_population_size,
        )
    )
    stage1_rl_episodes = int(stage1_ga_generations * stage1_population_size)
    stage2_rl_episodes = int(stage2_ga_generations * stage2_population_size)
    stage1_rl_episodes_specified = stage1_budget_specified
    stage2_rl_episodes_specified = stage2_budget_specified
    skip_budget_normalized = normalize_ga_skip_budget_flags(
        skip_stage1_rl=skip_stage1_rl,
        skip_noise_rl=skip_noise_rl,
        stage1_ga_generations=stage1_ga_generations,
        stage2_ga_generations=stage2_ga_generations,
        stage1_budget_specified=stage1_budget_specified,
        stage2_budget_specified=stage2_budget_specified,
        stage1_rl_episodes=stage1_rl_episodes,
        stage2_rl_episodes=stage2_rl_episodes,
    )
    stage1_rl_episodes = skip_budget_normalized["stage1_rl_episodes"]
    stage2_rl_episodes = skip_budget_normalized["stage2_rl_episodes"]
    stage1_rl_episodes_specified = skip_budget_normalized["stage1_budget_specified"]
    stage2_rl_episodes_specified = skip_budget_normalized["stage2_budget_specified"]
    if skip_budget_normalized["stage1_budget_source"] is not None:
        stage1_budget_source = skip_budget_normalized["stage1_budget_source"]
    if skip_budget_normalized["stage2_budget_source"] is not None:
        stage2_budget_source = skip_budget_normalized["stage2_budget_source"]

    _budget_label = "Greedy" if search_backend == "greedy" else "GA"
    print(
        f"Resolved {_budget_label} search budgets:\n"
        f"total_layers: {total_layers}\n"
        f"stage1_ga_population_size: {stage1_population_size}\n"
        f"stage2_ga_population_size: {stage2_population_size}\n"
        f"stage1_ga_generations_resolved: {stage1_ga_generations} (source={stage1_budget_source})\n"
        f"stage2_ga_generations_resolved: {stage2_ga_generations} (source={stage2_budget_source})\n"
        f"stage1_eval_budget: {stage1_rl_episodes}\n"
        f"stage2_eval_budget: {stage2_rl_episodes}\n"
    )


    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    # Tokenize helper.
    def tokenize_fn(examples):
        _dp = data_path.lower()
        if _dp in ("sst2", "cola"):
            tokenized = tokenizer(
                examples["sentence"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        elif _dp == "qnli":
            tokenized = tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        elif _dp == "mnli":
            tokenized = tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        else:  # mrpc, stsb, rte, wnli
            tokenized = tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True, padding=False, max_length=128, return_tensors=None,
            )
        return tokenized

    # def generate_and_tokenize_prompt(data_point):
    #     full_prompt = generate_prompt(data_point)
    #     tokenized_full_prompt = tokenize(full_prompt)
    #     if not train_on_inputs:
    #         user_prompt = generate_prompt({**data_point, "output": ""})
    #         tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    #         user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #         tokenized_full_prompt["labels"] = [
    #                                               -100
    #                                           ] * user_prompt_len + tokenized_full_prompt["labels"][
    #                                                                 user_prompt_len:
    #                                                                 ]  # could be sped up, probably
    #     return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # if adapter_name == "lora":
    #     config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    # elif adapter_name == "dora":
    #     dora_simple = True
    #     config = DoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #         dora_simple=dora_simple,
    #         Wdecompose_target_modules=Wdecompose_target_modules
    #     )
    # elif adapter_name == "bottleneck":
    #     config = BottleneckConfig(
    #         bottleneck_size=bottleneck_size,
    #         non_linearity=non_linearity,
    #         adapter_dropout=adapter_dropout,
    #         use_parallel_adapter=use_parallel_adapter,
    #         use_adapterp=use_adapterp,
    #         target_modules=target_modules,
    #         scaling=scaling,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    # elif adapter_name == "prefix-tuning":
    #     config = PrefixTuningConfig(
    #         num_virtual_tokens=num_virtual_tokens,
    #         task_type="CAUSAL_LM",
    #     )
    # model = get_peft_model(model, config)

    if adapter_name == "prefix-tuning":
        model.to("cuda") 
    
    print(model)
    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        # glue tasks: "stsb", "mnli", "sst2", "cola", "qnli", "rte", "wnli", "mrpc"
        data = load_dataset("nyu-mll/glue", data_path)



    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            # adapters_weights = torch.load(checkpoint_name)
            # model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # if val_set_size > 0:
    #     train_val = data["train"].train_test_split(
    #         test_size=val_set_size, shuffle=True, seed=42
    #     )
    #     train_data = (
    #         train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    #     val_data = (
    #         train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     )
    # else:
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    
    # MNLI needs special handling: matched and mismatched validation splits.
    val_data_mm = None  # MNLI mismatched validation split.
    
    if val_set_size > 0:
        is_mnli = data_path.lower() == 'mnli'
        
        if is_mnli:
            print(f"Loading MNLI dataset (matched + mismatched validation sets)")
            train_data = data["train"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            val_data = data["validation_matched"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            val_data_mm = data["validation_mismatched"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            
            print(f"After tokenize matched: {val_data[0]}")
            train_data = train_data.rename_column("label", "labels")
            val_data = val_data.rename_column("label", "labels")
            val_data_mm = val_data_mm.rename_column("label", "labels")
            
            columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
            train_data.set_format(type="torch", columns=columns)
            val_data.set_format(type="torch", columns=columns)
            val_data_mm.set_format(type="torch", columns=columns)
            
            print(f"Train data size: {len(train_data)}")
            print(f"Validation matched size: {len(val_data)}")
            print(f"Validation mismatched size: {len(val_data_mm)}")
        else:
            print(f"Loading dataset: {data['validation']}")
            train_data = data["train"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            val_data = data["validation"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
            # The current RL flow does not use the official test split.
            # test_data = data["test"].shuffle().map(tokenize_fn)
            
            print(f"After tokenize: {val_data[0]}")
            # add label
            train_data = train_data.rename_column("label", "labels")
            val_data = val_data.rename_column("label", "labels")
            
            print(f"After add label: {val_data[0]}")
            
            # Set PyTorch tensor format.
            columns = ["input_ids", "attention_mask", "token_type_ids", "labels"]
            train_data.set_format(type="torch", columns=columns)
            val_data.set_format(type="torch", columns=columns)

            print(f"After format: {val_data}")
            
            print(f"Train data size: {len(train_data)}")
            print(f"Validation data size: {len(val_data)}") 
            # print(f"Test data size: {len(test_data)}")
            
    else:
        train_data = data["train"].shuffle(seed=final_eval_random_seed).map(tokenize_fn)
        val_data = None

    # data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    # for Binary classification task
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding= "max_length",
        max_length=128,     # Effective when padding="max_length"
        return_tensors="pt", # Return PyTorch tensors
        pad_to_multiple_of=8   # Return attention masks
    )
    
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True
    parsed_manual_stage1_gelu = parse_degree_config(manual_stage1_gelu)
    parsed_manual_stage1_softmax = parse_degree_config(manual_stage1_softmax)
    parsed_manual_stage2_noise = parse_noise_config(manual_stage2_noise)
    trainer_callbacks = []

    if use_ist:
        from layer_importance_evaluator import LayerImportanceEvaluator
        from genetic_search_module import (
            GA_STAGE1_CHECKPOINT_FILENAME,
            GA_STAGE2_CHECKPOINT_FILENAME,
            GeneticUnifiedFinalEvaluationModule,
            build_stage1_context,
            build_stage2_final_eval_context_without_search,
            resolve_stage1_selected_config,
            run_stage1_genetic_search,
            run_stage2_noise_genetic_search,
        )
        if search_backend == "greedy":
            from greedy_search_module import (
                GREEDY_STAGE1_RESULT_FILENAME,
                GREEDY_STAGE2_RESULT_FILENAME,
                GreedyUnifiedFinalEvaluationModule,
                run_stage1_greedy_search,
                run_stage2_noise_greedy_search,
            )

            SearchFinalEvaluationModule = GreedyUnifiedFinalEvaluationModule
            run_stage1_search = run_stage1_greedy_search
            run_stage2_noise_search = run_stage2_noise_greedy_search
            stage1_result_filename = GREEDY_STAGE1_RESULT_FILENAME
            stage2_result_filename = GREEDY_STAGE2_RESULT_FILENAME
            backend_label = "Greedy"
        else:
            SearchFinalEvaluationModule = GeneticUnifiedFinalEvaluationModule
            run_stage1_search = run_stage1_genetic_search
            run_stage2_noise_search = run_stage2_noise_genetic_search
            stage1_result_filename = "ga_search_results.json"
            stage2_result_filename = "noise_ga_search_results.json"
            backend_label = "GA"

        print(f"{backend_label} search to evaluate layer sensitivity to approximation")
        # Pass data_path so evaluator can detect dataset type and metrics.
        importance_evaluator = LayerImportanceEvaluator(
            model=model, 
            train_data=train_data, 
            # Keep the historical argument name; we pass validation data here.
            test_data=val_data, 
            data_collator=data_collator, 
            batch_size=batch_size,
            rl_lr=rl_lr, 
            degree=degree,
            stage1_rl_episodes=stage1_rl_episodes,
            stage2_rl_episodes=stage2_rl_episodes,
            stage1_rl_episodes_specified=stage1_rl_episodes_specified,
            stage2_rl_episodes_specified=stage2_rl_episodes_specified,
            run_output_dir=run_output_dir,
            final_eval_config_source=final_eval_config_source,
            final_eval_config_path=final_eval_config_path,
            manual_stage1_gelu=parsed_manual_stage1_gelu,
            manual_stage1_softmax=parsed_manual_stage1_softmax,
            manual_stage2_noise=parsed_manual_stage2_noise,
            final_eval_random_seed=final_eval_random_seed,
            final_eval_permutation_trials=final_eval_permutation_trials,
            final_eval_cost_equivalent_trials=final_eval_cost_equivalent_trials,
            final_eval_budget_equivalent_trials=final_eval_budget_equivalent_trials,
            final_eval_stage1_budget_trials=final_eval_stage1_budget_trials,
            final_eval_stage2_budget_trials=final_eval_stage2_budget_trials,
            final_eval_repeat_n=final_eval_repeat_n,
            final_eval_preset=final_eval_preset,
            final_eval_output_root=final_eval_output_root,
            final_eval_run_name=final_eval_run_name,
            final_eval_random_enabled=final_eval_random_enabled,
            final_eval_action_config=final_eval_action_config,
            final_eval_action_ranges=final_eval_action_ranges,
            final_eval_action_fixed=final_eval_action_fixed,
            skip_noise_rl=skip_noise_rl,
            skip_stage1_rl=skip_stage1_rl,
            skip_final_eval=skip_final_eval,
            final_eval_only=final_eval_only,
            resume_run_dir=resume_run_dir,
            data_path=data_path,
            test_data_mm=val_data_mm,
            search_algorithm=search_backend,
            stage1_accuracy_tolerance=stage1_accuracy_tolerance,
            stage2_limit_tolerance=stage2_limit_tolerance,
            stage2_stability_tolerance=stage2_stability_tolerance,
            stage2_k_trials=stage2_k_trials,
            stage2_probe_size=stage2_probe_size,
            stage2_rl_variant=stage2_rl_variant,
            blb_v3_rescale_invoker_kind=blb_v3_rescale_invoker_kind,
            blb_v3_subprocess_optimizer_root=(
                blb_v3_subprocess_optimizer_root
                if blb_v3_subprocess_optimizer_root not in (None, "") else None
            ),
            blb_v3_subprocess_cli_module=blb_v3_subprocess_cli_module,
            blb_v3_rollout_size=blb_v3_rollout_size,
            blb_v3_eval_interval=blb_v3_eval_interval,
            blb_v3_save_interval=blb_v3_save_interval,
            blb_v3_calibrate_baseline_samples=blb_v3_calibrate_baseline_samples,
        )
        importance_evaluator.stage1_ga_generations = int(stage1_ga_generations)
        importance_evaluator.stage2_ga_generations = int(stage2_ga_generations)
        importance_evaluator.stage1_ga_generations_specified = bool(stage1_budget_specified)
        importance_evaluator.stage2_ga_generations_specified = bool(stage2_budget_specified)
        importance_evaluator.stage1_ga_population_size = int(stage1_population_size)
        importance_evaluator.stage2_ga_population_size = int(stage2_population_size)
        importance_evaluator.log(
            f"{backend_label} budgets resolved to validation-full search units: "
            f"stage1_generations={stage1_ga_generations} "
            f"(population={stage1_population_size}), "
            f"stage2_generations={stage2_ga_generations} "
            f"(population={stage2_population_size})."
        )
        model.config.use_cache = False
        model.config.is_decoder = False

        # ---- Resolve GA resume checkpoint paths ----
        ga_stage1_resume_path = None
        ga_stage2_resume_path = None
        if search_backend == "ga" and resume_run_dir:
            _s1_path = os.path.join(resume_run_dir, "stage1", GA_STAGE1_CHECKPOINT_FILENAME)
            if os.path.isfile(_s1_path):
                ga_stage1_resume_path = _s1_path
                print(f"[GA Resume] Found Stage-1 checkpoint: {_s1_path}")
            _s2_path = os.path.join(resume_run_dir, "stage2_noise", "progress", GA_STAGE2_CHECKPOINT_FILENAME)
            if os.path.isfile(_s2_path):
                ga_stage2_resume_path = _s2_path
                print(f"[GA Resume] Found Stage-2 checkpoint: {_s2_path}")

        # ---- final_eval_only: 预先加载之前 GA 搜索得到的最优配置 ----
        # 仅当 final_eval_only=True 时触发；其它情形维持原行为 None。
        # 不读取/写入任何 checkpoint 文件，因此不会影响续训练；
        # 同时此分支进入 final_eval 时会跳过两个搜索阶段，所以也不会安装 graceful-stop 信号，
        # 无法与优雅停止的 STOP_RL 状态发生冲突。
        prior_stage1_search_best = None
        prior_stage2_search_best = None
        if final_eval_only:
            _prior_load_dir = resume_run_dir or run_output_dir
            prior_stage1_search_best, prior_stage2_search_best = _load_prior_ga_search_results(
                _prior_load_dir,
                stage1_filename=stage1_result_filename,
                stage2_filename=stage2_result_filename,
                algorithm_label=backend_label,
            )
            if prior_stage1_search_best is None and final_eval_config_source == "search":
                print(
                    "[final_eval_only][提示] 未在 "
                    f"{_prior_load_dir} 找到 Stage-1 {backend_label} 搜索结果。"
                    "final_eval 将按照 final_eval_config_source 的回退规则选择配置。"
                )
            if prior_stage2_search_best is None and final_eval_config_source == "search":
                print(
                    "[final_eval_only][提示] 未在 "
                    f"{_prior_load_dir} 找到 Stage-2 {backend_label} 搜索结果。"
                    "final_eval 将按照 final_eval_config_source 的回退规则选择配置。"
                )

        stage1_search_result = None
        stage1_context = None
        if not importance_evaluator.skip_stage1_rl:
            stage1_search_result = run_stage1_search(
                importance_evaluator,
                random_seed=final_eval_random_seed,
                resume_checkpoint_path=ga_stage1_resume_path,
            )
            stage1_context = stage1_search_result["context"]
        else:
            importance_evaluator.log(f"Stage-1 {backend_label} search skipped by flag.")

        if stage1_context is None:
            stage1_context = build_stage1_context(
                importance_evaluator,
                log_fn=importance_evaluator.log,
                include_distribution=False,
                constraint_ratio=getattr(importance_evaluator, "error_threshold", None),
            )

        stage1_search_best = (
            stage1_search_result["best_config"]
            if stage1_search_result is not None
            else prior_stage1_search_best
        )

        fixed_gelu = None
        fixed_softmax = None
        fixed_label = None
        fixed_source = None
        noise_stage_result = None
        final_eval_result = None
        stage2_log_previous = None
        stage2_logging_active = False
        need_stage2_scope = (not importance_evaluator.skip_noise_rl) or (
            not importance_evaluator.skip_final_eval
        )
        if need_stage2_scope:
            if hasattr(importance_evaluator, "activate_noise_logging"):
                stage2_log_previous = importance_evaluator.activate_noise_logging()
            else:
                stage2_log_previous = getattr(importance_evaluator, "active_log_file", None)
                noise_log_file = getattr(importance_evaluator, "noise_log_file", None)
                if noise_log_file:
                    os.makedirs(os.path.dirname(noise_log_file), exist_ok=True)
                    importance_evaluator.active_log_file = noise_log_file
            stage2_logging_active = True
        try:
            if need_stage2_scope:
                fixed_gelu, fixed_softmax, fixed_label, fixed_source = resolve_stage1_selected_config(
                    evaluator=importance_evaluator,
                    search_best_config=stage1_search_best,
                    config_source=importance_evaluator.final_eval_config_source,
                    config_path=importance_evaluator.final_eval_config_path,
                    manual_gelu=importance_evaluator.manual_stage1_gelu,
                    manual_softmax=importance_evaluator.manual_stage1_softmax,
                )
                importance_evaluator.log(
                    f"Stage-2 fixed Stage-1 config source={fixed_source}, label={fixed_label}"
                )

            if not importance_evaluator.skip_noise_rl:
                noise_stage_result = run_stage2_noise_search(
                    importance_evaluator,
                    fixed_gelu=fixed_gelu,
                    fixed_softmax=fixed_softmax,
                    fixed_label=fixed_label,
                    fixed_source=fixed_source,
                    random_seed=final_eval_random_seed,
                    resume_checkpoint_path=ga_stage2_resume_path,
                )
            else:
                importance_evaluator.log(f"Stage-2 noise {backend_label} search skipped by flag.")

            if not importance_evaluator.skip_final_eval:
                if noise_stage_result is not None:
                    baseline_noise_config = noise_stage_result["baseline_noise_config"]
                    baseline_tot_c = noise_stage_result["baseline_tot_c"]
                    stage2_search_best = noise_stage_result["best_noise_config"]
                    limit_loss = noise_stage_result["limit_loss"]
                    limit_p = noise_stage_result["limit_p"]
                    limit_s = noise_stage_result["limit_s"]
                else:
                    noise_final_eval_context = build_stage2_final_eval_context_without_search(
                        importance_evaluator
                    )
                    baseline_noise_config = noise_final_eval_context["baseline_noise_config"]
                    baseline_tot_c = noise_final_eval_context["baseline_tot_c"]
                    stage2_search_best = prior_stage2_search_best
                    limit_loss = noise_final_eval_context["limit_loss"]
                    limit_p = noise_final_eval_context["limit_p"]
                    limit_s = noise_final_eval_context["limit_s"]

                if (
                    importance_evaluator.final_eval_only
                    and importance_evaluator._should_run_blb_action_final_eval(stage2_search_best)
                ):
                    final_eval_result = importance_evaluator.run_unified_final_eval(
                        stage1_search_best=stage1_search_best,
                        stage2_search_best=stage2_search_best,
                        baseline_stage1_gelu=stage1_context.base_gelu,
                        baseline_stage1_softmax=stage1_context.base_softmax,
                        baseline_noise_tot_c=baseline_tot_c,
                        limit_loss=limit_loss,
                        limit_p=limit_p,
                        limit_s=limit_s,
                    )
                elif importance_evaluator.final_eval_only:
                    final_eval_runner = SearchFinalEvaluationModule(
                        evaluator=importance_evaluator,
                        config_source=final_eval_config_source,
                        config_path=final_eval_config_path,
                        manual_stage1_gelu=parsed_manual_stage1_gelu,
                        manual_stage1_softmax=parsed_manual_stage1_softmax,
                        manual_stage2_noise=parsed_manual_stage2_noise,
                        random_seed=final_eval_random_seed,
                        permutation_trials=final_eval_permutation_trials,
                        cost_equivalent_trials=final_eval_cost_equivalent_trials,
                        budget_equivalent_trials=final_eval_budget_equivalent_trials,
                        stage1_budget_trials=final_eval_stage1_budget_trials,
                        stage2_budget_trials=final_eval_stage2_budget_trials,
                        repeat_n=importance_evaluator.final_eval_repeat_n,
                        results_dir=importance_evaluator.final_eval_dir,
                    )
                    final_eval_result = final_eval_runner.run(
                        search_best_stage1=stage1_search_best,
                        search_best_stage2=stage2_search_best,
                        baseline_stage1_gelu=stage1_context.base_gelu,
                        baseline_stage1_softmax=stage1_context.base_softmax,
                        baseline_noise_tot_c=baseline_tot_c,
                        limit_loss=limit_loss,
                        limit_p=limit_p,
                        limit_s=limit_s,
                    )
                else:
                    from final_eval.embedded import run_embedded_final_eval

                    final_eval_result = run_embedded_final_eval(
                        evaluator=importance_evaluator,
                        search_best_stage1=stage1_search_best,
                        search_best_stage2=stage2_search_best,
                        baseline_stage1_gelu=stage1_context.base_gelu,
                        baseline_stage1_softmax=stage1_context.base_softmax,
                        baseline_noise_tot_c=baseline_tot_c,
                        limit_loss=limit_loss,
                        limit_p=limit_p,
                        limit_s=limit_s,
                        preset_name=importance_evaluator.final_eval_preset,
                        output_root=importance_evaluator.final_eval_output_root,
                        run_name=importance_evaluator.final_eval_run_name,
                        module_cls=SearchFinalEvaluationModule,
                    )
            else:
                importance_evaluator.log("Unified final evaluation skipped by flag.")
        finally:
            if stage2_logging_active:
                if hasattr(importance_evaluator, "restore_log_file"):
                    importance_evaluator.restore_log_file(stage2_log_previous)
                elif stage2_log_previous is not None:
                    importance_evaluator.active_log_file = stage2_log_previous

        return {
            "stage1_search_status": (
                stage1_search_result["status"]
                if stage1_search_result is not None
                else "skipped"
            ),
            "stage1_search_path": (
                stage1_search_result["result_path"]
                if stage1_search_result is not None
                else None
            ),
            "stage2_noise_search_status": (
                noise_stage_result["status"]
                if noise_stage_result is not None
                else "skipped"
            ),
            "stage2_noise_search_path": (
                noise_stage_result["result_path"]
                if noise_stage_result is not None
                else None
            ),
            "final_eval_path": (
                final_eval_result["summary_path"]
                if final_eval_result is not None
                else None
            ),
        }
    # elif use_rst:
    #     from rst import RSTCallback
    #     print('Random Sparse Tuning activated')
    #     rst_callback = RSTCallback(model)
    #     trainer_callbacks.append(rst_callback)
    else:
        print('No sparse tuning activated')
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            output_dir=trainer_output_dir,
            per_device_eval_batch_size=batch_size,  # 推理批次大小
            disable_tqdm=False,  # Optional progress bar control
            # per_device_train_batch_size=micro_batch_size,
            # gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=100,
            # num_train_epochs=num_epochs,
            # learning_rate=learning_rate,
            # weight_decay=weight_decay,
            # # fp16=True,
            # fp16=False,
            # fp16_full_eval=False,
            # logging_steps=10,
            # optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            # save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            # save_steps=save_step,
            # output_dir=output_dir,
            # save_total_limit=3,
            # load_best_model_at_end=True if val_set_size > 0 else False,
            # ddp_find_unused_parameters=False if ddp else None,
            # group_by_length=group_by_length,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=data_collator,
        callbacks=trainer_callbacks
    )

    model.config.use_cache = False
    model.config.is_decoder = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    print( "Model compile started")
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    
    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # model.save_pretrained(output_dir)

    print("Model compile finished")
    print("Starting evaluation...")

    # trainer.predict(test_dataset=val_data, metric_key_prefix="predict")

    for _ in range(1):
        print(f"Round {_} of evaluation")
        
        print(val_data[0])  # Should be list[int]
        print(val_data[0])   # Should be consistent

        eval_results = trainer.evaluate(eval_dataset=val_data)
        final_loss = eval_results["eval_loss"] if "eval_loss" in eval_results else None
        print(f"Round {_}, Final evaluation loss: {final_loss}")

    
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


if __name__ == "__main__":
    run_fire_entrypoint(
        fire,
        train,
        program_name="rl_tune_genetic.py",
    )

