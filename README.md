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
rl_lr: reinforcement learning rate used in importance score update, now 20-40 is acceptable.  
degree: parameter for early debug, now deprecated. Just set it to 2.  

example: `bash llama_7B_LayerImportance.sh 32 64 output.log 20 2`

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