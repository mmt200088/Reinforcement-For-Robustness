This is a Repository for Transformer robustness evaluation using Reinforcement Learning.

Please Ignore the LLM-Adapters, EzPC, and importance-aware-sparse-tuning-IST-paper in root directory. Sorry, but the code is DIRTY now!

## How to Start

### Allocate enough memory for Docker container
    mount -o remount,size=64G /dev/shm
### Activate python enviroment first
    conda activate llm_ist
### Go into the .sh located directory (/root/ppml/MoE-Privacy)
    cd /root/ppml/MoE-Privacy
### Execute the running scripts 
    bash llama_7B_LayerImportance.sh [lora_r] [lora_alpha] [logfile_path] [rl_lr] [degree]

lora_r: parameter for lora, ignore (we just use the Lora Framework to inference...), just set is to 32.  
lora_alpha: parameter for lora, ignore, just set is to 64.  
logfile_path: the log file output path, you can change it when the learning rate varies.  
rl_lr: reinforcement learning rate used in importance score update, now 20-40 is acceptable.  
degree: parameter for early debug, now deprecated. Just set it to 2.  

example: `bash llama_7B_LayerImportance.sh 32 
64 output.log 20 2`

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
