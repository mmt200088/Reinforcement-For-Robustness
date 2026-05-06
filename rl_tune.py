import os
import sys
import json
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
        rl_lr: float = 1e-4, 
        stage1_rl_lr: float = None,
        stage2_rl_lr: float = None,
        degree: int = 4,  # degree of polynomial for approximation
        stage1_rl_episodes: int = 51000,
        stage2_rl_episodes: int = 40000,
        stage1_rl_episodes_specified: bool = False,
        stage2_rl_episodes_specified: bool = False,
        ppo_update_interval: int = 120,  # PPO 更新间隔（episode 数）；同时决定 batch 大小与 details 分块大小
        final_eval_config_source: str = "search",  # search | json | manual | max
        final_eval_config_path: str = "glue_final_configs_best_ppo.json",
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
        blb_v3_rollout_size: int = None,
        blb_v3_eval_interval: int = None,
        blb_v3_save_interval: int = None,
        blb_v3_calibrate_baseline_samples: int = None,
        blb_v3_inproc_rescale_optimizer_root: str = "",
        final_eval_require_rescale_optimizer: bool = False,
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
    final_eval_require_rescale_optimizer = parse_bool_flag(
        final_eval_require_rescale_optimizer, "final_eval_require_rescale_optimizer"
    )
    # --final_eval_only 语义：只跑 final eval，不跑任何 RL 搜索阶段。
    # 等价于自动设置 skip_stage1_rl=True & skip_noise_rl=True & skip_final_eval=False，
    # 同时尝试从 resume_run_dir / output_dir 下读取之前搜索得到的最优配置作为 final-eval 输入。
    # 该路径不会安装 graceful-stop 信号、不读写 RL 训练 checkpoint，因此不影响优雅停止与续训。
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
    stage1_rl_episodes_specified = parse_bool_flag(
        stage1_rl_episodes_specified, "stage1_rl_episodes_specified"
    )
    stage2_rl_episodes_specified = parse_bool_flag(
        stage2_rl_episodes_specified, "stage2_rl_episodes_specified"
    )
    batch_size = parse_positive_int(batch_size, "batch_size")
    micro_batch_size = parse_positive_int(micro_batch_size, "micro_batch_size")
    stage1_rl_episodes = parse_positive_int(
        stage1_rl_episodes, "stage1_rl_episodes"
    )
    stage2_rl_episodes = parse_positive_int(
        stage2_rl_episodes, "stage2_rl_episodes"
    )
    ppo_update_interval = parse_positive_int(
        ppo_update_interval, "ppo_update_interval"
    )
    # 在创建 LayerImportanceEvaluator 之前覆盖 PPO 更新间隔及其派生常量
    import layer_importance_evaluator as _lie
    _lie.set_ppo_update_interval(ppo_update_interval)
    print(
        f"[PPO] ppo_update_interval={_lie.PPO_UPDATE_INTERVAL} "
        f"(batch={_lie.PPO_BATCH_SIZE} steps, details chunk={_lie.STEP_INFO_CHUNK_SIZE} episodes)"
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
        f"rl_lr: {rl_lr}\n"
        f"stage1_rl_lr: {stage1_rl_lr}\n"
        f"stage2_rl_lr: {stage2_rl_lr}\n"
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
        f"target_modules: {target_modules}\n"
        f"final_eval_config_source: {final_eval_config_source}\n"
        f"final_eval_config_path: {final_eval_config_path}\n"
        f"manual_stage1_gelu: {manual_stage1_gelu}\n"
        f"manual_stage1_softmax: {manual_stage1_softmax}\n"
        f"manual_stage2_noise: {manual_stage2_noise}\n"
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
        f"final_eval_require_rescale_optimizer: {final_eval_require_rescale_optimizer}\n"
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
        f"stage2_rl_variant: {stage2_rl_variant}\n"
        f"blb_v3_inproc_rescale_optimizer_root: {blb_v3_inproc_rescale_optimizer_root}\n"
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
        print('Reinforcement Learning to evaluate layer sensitivity to approximation')
        # Pass data_path so evaluator can detect dataset type and metrics.
        importance_evaluator = LayerImportanceEvaluator(
            model=model, 
            train_data=train_data, 
            # Keep the historical argument name; we pass validation data here.
            test_data=val_data, 
            data_collator=data_collator, 
            batch_size=batch_size,
            rl_lr=rl_lr, 
            stage1_rl_lr=stage1_rl_lr,
            stage2_rl_lr=stage2_rl_lr,
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
            final_eval_require_rescale_optimizer=final_eval_require_rescale_optimizer,
            skip_noise_rl=skip_noise_rl,
            skip_stage1_rl=skip_stage1_rl,
            skip_final_eval=skip_final_eval,
            final_eval_only=final_eval_only,
            resume_run_dir=resume_run_dir,
            data_path=data_path,
            test_data_mm=val_data_mm,
            stage1_accuracy_tolerance=stage1_accuracy_tolerance,
            stage2_limit_tolerance=stage2_limit_tolerance,
            stage2_stability_tolerance=stage2_stability_tolerance,
            stage2_k_trials=stage2_k_trials,
            stage2_probe_size=stage2_probe_size,
            stage2_rl_variant=stage2_rl_variant,
            blb_v3_inproc_rescale_optimizer_root=(
                blb_v3_inproc_rescale_optimizer_root
                if blb_v3_inproc_rescale_optimizer_root not in (None, "") else None
            ),
            blb_v3_rollout_size=blb_v3_rollout_size,
            blb_v3_eval_interval=blb_v3_eval_interval,
            blb_v3_save_interval=blb_v3_save_interval,
            blb_v3_calibrate_baseline_samples=blb_v3_calibrate_baseline_samples,
        )
        trainer_callbacks.append(importance_evaluator)
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
        program_name="rl_tune.py",
    )

