This is a Repository for Transformer robustness evaluation using Reinforcement Learning.

Please Ignore the LLM-Adapters, EzPC, and importance-aware-sparse-tuning-IST-paper in root directory. Sorry, but the code is DIRTY now!

## How to Start

### Allocate enough memory for Docker container
    mount -o remount,size=64G /dev/shm
### Activate python enviroment first
    conda activate llm_ist
### Go into the .sh located directory (/root/ppml/MoE-Privacy)
    cd /root/ppml/MoE-Privacy 
    cd /var/tmp/root-home/Reinforcement-For-Robustness
### Execute the running scripts 
    bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree]

lora_r: parameter for lora, ignore (we just use the Lora Framework to inference...), just set is to 32.  
lora_alpha: parameter for lora, ignore, just set is to 64.  
logfile_path: the log file output path, you can change it when the learning rate varies.  
rl_lr: PPO learning-rate control. If `rl_lr < 1`, it is used directly as the PPO optimizer LR. For backward compatibility, legacy values like `20` / `40` are interpreted as `20e-6` / `40e-6`.  
degree: parameter for early debug, now deprecated. Just set it to 2.  

example: `bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

#### Optional FINAL EVALUATION modes
The original 5-argument command is still fully supported. You can now optionally append named arguments after the first 5 positional arguments.

Use RL-learned configuration for final evaluation (default behavior):
`bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

Use configuration from a JSON file for final evaluation:
`bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 --final-eval-source json --final-eval-config glue_configs_best_ppo.json`

Use manually specified layer-wise configuration for final evaluation:
`bash llama_7B_LayerImportance.sh 32 64 output_manual.log 20 2 --final-eval-source manual --manual-gelu "[1,1,1,4,1,1,1,1,1,1,1,1]" --manual-softmax "[2,3,4,6,4,4,5,4,4,5,5,2]"`

Optional named arguments supported by the shell script:
- `--final-eval-source search|json|manual`
- `--final-eval-config PATH`
- `--manual-gelu "[...]"` and `--manual-softmax "[...]"`
- `--random-seed N`
- `--perm-trials N`
- `--cost-trials N`
- `--budget-trials N`

When `--final-eval-source json` is used, the code will automatically read the configuration entry that matches the current dataset name, for example `mrpc` or `sst2`.

#### Second-stage noise RL
After the layer-wise GELU/Softmax configuration is finalized, the code automatically enters `PHASE 5: SECOND-STAGE NOISE RL`.

What stage 2 does:
- Keeps the selected GELU/Softmax configuration fixed.
- Learns layer-wise scaling factors for `x`, `wq`, `wk`, `wv`, `wo`, `wffn1`, and `wffn2`.
- Uses the same resolved PPO LR as stage 1.

How the fixed GELU/Softmax config is chosen before stage 2:
- `--final-eval-source search`: run stage-1 RL/greedy first, then run stage 2 on that selected config.
- `--final-eval-source json`: skip stage-1 search, load the saved config from JSON, then run stage 2 on it.
- `--final-eval-source manual`: skip stage-1 search, use the manually provided config, then run stage 2 on it.

Useful stage-2 outputs:
- `noise_ppo_step_info.txt`: per-step action/scaling-factor log.
- `noise_ppo_training_curve.png`: reward/loss/metric curve for stage 2.
- `noise_ppo_entropy_curve.png`: entropy curve for the stage-2 PPO policy.
- The main run log: search for `PHASE 5: SECOND-STAGE NOISE RL` and `Best Noise Configuration Found`.

Example commands:
- Default full pipeline: stage-1 RL search + stage-2 noise RL  
  `bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`
- Skip stage-1 search and run stage 2 on a saved first-stage config  
  `bash llama_7B_LayerImportance.sh 32 64 output_json.log 20 2 --final-eval-source json --final-eval-config glue_configs_best_ppo.json`
- Increase the number of stage-2 permutation / cost-equivalent random baselines  
  `bash llama_7B_LayerImportance.sh 32 64 output_stage2.log 20 2 --perm-trials 30 --cost-trials 30`

#### Note: Though we call the script "llama_7B_LayerImportance.sh", we just evaluate the Bert-base model for different tasks now, please check out the .sh for more detials!

### The Result file 
The result outputs to file importance_scores_.....txt in /root/ppml/MoE-Privacy. You can modified the name in variable self.log_path in layer_importance_evaluator.py

### Stop the process
Cause running the sh now is using nohup, so we run it in backend.  
When you want interrupt it, run
`ps aux | grep rl_tune.py`
to check the process (rl_tune.py is the starting point of our evaluate, because we use the LLM-Adapter framework).  
And then kill the first process:
`kill -9 [process_id_of_rl_tune.py]` 

### data support
"stsb", "mnli", "sst2", "cola", "qnli", "rte", "wnli", "mrpc"

### 各数据集描述
| 数据集 | 任务类型 | 训练集 (Train) | 验证集 (Dev) | 测试集 (Test) | 评价指标 (Metrics) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MNLI** | 自然语言推理 | 392,702 | 9,815/9,832 | 9,796/9,847 | Matched/Mismatched Acc. |
| **QQP** | 句子对等判定 | 363,846 | 40,430 | 390,965 | F1 / Accuracy |
| **QNLI** | 问答蕴含 | 104,743 | 5,463 | 5,463 | Accuracy |
| **SST-2** | 情感分析 | 67,349 | 872 | 1,821 | Accuracy |
| **CoLA** | 语法可接受性 | 8,551 | 1,043 | 1,063 | Matthews Corr. |
| **STS-B** | 语义相似度 | 5,749 | 1,500 | 1,379 | Pearson/Spearman Corr. |
| **MRPC** | 句子对等判定 | 3,668 | 408 | 1,725 | F1 / Accuracy |
| **RTE** | 文本蕴含 | 2,490 | 277 | 3,000 | Accuracy |
| **WNLI** | 指代消解蕴含 | 635 | 71 | 146 | Accuracy |

### 实验运行
完整实验（所有 8 个数据集）
conda activate llm_ist
bash run_all_experiments.sh

快速测试（仅 sst2, mrpc）
bash run_all_experiments.sh --quick

单独运行某个版块
python experiment_single_layer_degradation.py --tasks sst2 mrpc --device cuda
python experiment_block1_monotonicity.py --tasks sst2 --n_bootstrap 100 --device cuda
