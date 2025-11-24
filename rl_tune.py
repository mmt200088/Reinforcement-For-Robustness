import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
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
        degree: int = 4,  # degree of polynomial for approximation
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
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

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
        # config.use_causal_lm = False  # 关键！关闭因果掩码 for mrpc
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=1,  # Assuming binary classification
            # load_in_8bit=False,
            # torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            # device_map ="cpu",
            trust_remote_code=True,
            # pad_token_id=tokenizer.eos_token_id
            pad_token_id=tokenizer.pad_token_id  
        )
    
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

    # Tokenize函数
    def tokenize_fn(examples):
        # print(f"examples keys: {examples.keys()}")
        tokenized = tokenizer(
            # examples["question"], 
            # examples["sentence"],

            examples["sentence1"], 
            examples["sentence2"],
            truncation=True,
            padding=False,
            max_length=128,
            return_tensors= None
        )
        
        # tokenized["labels"] = tokenized["input_ids"].copy()
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
        # model.to('cuda')
    
    print(model)
    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        # glue tasks: "stsb", "mnli", "sst2", "cola", "qnli", "rte", "wnli", "mrpc"
        data = load_dataset("glue", data_path)



    
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
    
    if val_set_size > 0:

        print(f"Loading dataset: {data['validation']}")
        train_data = data["train"].shuffle().map(tokenize_fn)
        val_data = data["validation"].shuffle().map(tokenize_fn)
        test_data = data["test"].shuffle().map(tokenize_fn)
        
        print(f"After tokenize: {val_data[0]}")
        # add label
        train_data = train_data.rename_column("label", "labels")
        val_data = val_data.rename_column("label", "labels")
        
        print(f"After add label: {val_data[0]}")
        
        # 设置PyTorch格式
        columns = ["input_ids", "attention_mask",  "token_type_ids", "labels"]
        train_data.set_format(type="torch", columns=columns)
        val_data.set_format(type="torch", columns=columns)

        print(f"After format: {val_data}")
        
        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}") 
        # print(f"Test data size: {len(test_data)}")
    else:
        train_data = data["train"].shuffle().map(tokenize_fn)
        val_data = None

    # data_collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    # for Binary classification task
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding= "max_length",
        max_length=128,     # 当padding="max_length"时生效
        return_tensors="pt", # 返回PyTorch张量
        pad_to_multiple_of=8   # 返回注意力掩码
    )
    
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True
    trainer_callbacks = []

    if use_ist:
        from layer_importance_evaluator import LayerImportanceEvaluator
        print('Reinforcement Learning to evaluate layer sensitivity to approximation')
        importance_evaluator = LayerImportanceEvaluator(model=model, train_data = train_data, test_data=val_data, data_collator=data_collator, rl_lr=rl_lr, degree=degree)
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
            output_dir="./inference_output",
            per_device_eval_batch_size=16,  # 推理批次大小
            disable_tqdm=False,  # 可选进度条控制
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
        
        print(val_data[0])  # 应为list[int]
        print(val_data[0])   # 应一致

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
    fire.Fire(train)
