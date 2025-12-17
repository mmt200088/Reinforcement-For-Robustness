# src/train/rl_layer.py
import numpy as np
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerCallback
from transformers import BertForSequenceClassification, BertConfig
from function_handler import ReversibleLayerHandler
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr
from typing import Optional, Iterable, Union
import datasets
import pandas as pd
import random
import sys
import copy
import os
import traceback 
import matplotlib.pyplot as plt # [Modification 1] Import matplotlib for plotting

# Store original deepcopy immediately to ensure we have a valid reference
_ORIGINAL_DEEPCOPY = copy.deepcopy

# 定义动作映射
ACTION_DECREASE = 0  # 对应 -1 (降低精度/降低开销)
ACTION_STAY = 1      # 对应 0  (保持不变)
ACTION_INCREASE = 2  # 对应 +1 (提高精度/增加开销)

class LayerImportanceEvaluator(TrainerCallback):
    def __init__(self, model, train_data, test_data, data_collator, rl_lr=100, degree=2, device='cuda'):
        # Safety measure: Significantly increase recursion limit for complex model graphs
        # 50000 should be sufficient for most Transformer models to deepcopy
        sys.setrecursionlimit(50000)
        
        self.batch_size = 16  # 基础 batch size
        self.train_dataset = train_data
        self.data_collator = data_collator
        
        # --- Handle torch.compile unwrapping (CRITICAL FIX) ---
        # If rl_tune.py used torch.compile, 'model' is an OptimizedModule.
        # Dynamic layer swapping on a compiled model causes recompilation hangs/deadlocks.
        # We must extract the original eager-mode model for RL tuning to allow dynamic changes.
        self.is_compiled = False
        if hasattr(model, "_orig_mod"):
            # 如果是编译后的模型，取出原始模型
            self.model = model._orig_mod
            self.is_compiled = True
        else:
            self.model = model
        
        # 数据加载器
        self.dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=data_collator)
        self.dataloader_test = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)
        
        # 转换为迭代器以便按需获取 batch
        self.train_iter = iter(self.dataloader)
        self.test_iter = iter(self.dataloader_test)
        
        # --- ReversibleLayerHandler Initialization ---
        # Removed monkey patch that disabled deepcopy. 
        # We rely on sys.setrecursionlimit(50000) to handle the deep copy of the model.
        # This is necessary because restore_all() needs a valid self.backup_model.
        try:
            self.reversible_layer_handler = ReversibleLayerHandler(self.model)
        except RecursionError:
            print("[Warning] RecursionError during ReversibleLayerHandler init (deepcopy). "
                  "Increasing limit didn't help. 'restore_all' will fail.", flush=True)
            # Fallback: try init with disabled deepcopy just to keep going, but restore_all won't work
            copy.deepcopy = lambda x, memo=None: None
            self.reversible_layer_handler = ReversibleLayerHandler(self.model)
            copy.deepcopy = _ORIGINAL_DEEPCOPY
        
        # --- 修复设备名称 ---
        if device == 'gpu':
            device = 'cuda'
        
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.", flush=True)
            self.device = 'cpu'
        else:
            self.device = device
        
        # 确定模型层访问路径
        class_to_layers_map = {
            'LlamaForCausalLM': 'model.model.layers',
            'Qwen2ForCausalLM': 'model.model.layers',
            'Qwen2ForSequenceClassification': 'model.model.layers',
            'MistralForCausalLM': 'model.model.layers',
            'MixtralForCausalLM': 'model.model.layers',
            'GemmaForCausalLM': 'model.model.layers',
            'GPT2LMHeadModel': 'model.transformer.h',
            'BertForSequenceClassification': 'model.bert.encoder.layer'
        }
        model_class_name = self.model.__class__.__name__
        if model_class_name in class_to_layers_map:
            self.layers_attribute = class_to_layers_map[model_class_name]
        else:
            self.layers_attribute = 'model.bert.encoder.layer'
            if self.is_main_process():
                print(f"[Info] Unknown model class {model_class_name}, defaulting to '{self.layers_attribute}'", flush=True)

        # 动态获取总层数
        try:
            self.total_layers = len(eval('self.' + self.layers_attribute))
        except Exception as e:
            if self.is_main_process():
                print(f"[Warning] Failed to detect layers using path 'self.{self.layers_attribute}': {e}", flush=True)
                print("Attempting fallback detection to ensure initialization proceeds...", flush=True)
            if hasattr(self.model, 'bert'):
                self.total_layers = len(self.model.bert.encoder.layer)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                self.total_layers = len(self.model.model.layers)
            else:
                self.total_layers = 12 

        # --- 新 RL 策略参数设置 ---
        self.rl_epochs = 60           # [Modification] Set to 60 as requested
        # self.eval_batch_count = 200   # [Modification] Set to 200 as requested
        self.steps_per_epoch = len(self.dataloader)  # 360
        
        if self.steps_per_epoch >= 500:
            self.steps_per_epoch = 500
            
        self.eval_steps = len(self.dataloader_test)  # 94

        
        # Q-Learning 参数
        self.alpha = 0.1     
        self.gamma = 0.9     
        self.epsilon = 0.3   
        self.epsilon_decay = 0.95
        
        # [修改点 1] 初始状态修改
        # GELU 初始全 4 (Max Approx)
        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        # Softmax 初始全 6 (Max Approx)
        self.current_softmax_degrees = np.full(self.total_layers, 6, dtype=int)
        
        # Baseline Loss (will be set in Step 0)
        self.baseline_loss = None

        # [修改点 2] 分离 GELU 和 Softmax 的 Q-Table，并调整大小
        # GELU Q-Table: [Layer, 5, 3] (States: 0,1,2,3,4)
        self.q_table_gelu = np.zeros((self.total_layers, 5, 3))
        # Softmax Q-Table: [Layer, 8, 3] (States: 0-7, 保险起见开大一点，实际用到 1-6)
        self.q_table_softmax = np.zeros((self.total_layers, 8, 3))
        
        # --- 日志文件设置 ---
        self.log_path = f"./rl_training_log_model{model_class_name}.txt"
        self.progress_log_path = f"./rl_progress_log_model{model_class_name}.txt"
        self.plot_path = f"./rl_learning_curve_model{model_class_name}.png" 
        
        # 初始化清空文件 (仅在主进程)
        if self.is_main_process():
            with open(self.log_path, 'w') as f:
                f.write("")
            with open(self.progress_log_path, 'w') as f:
                f.write("")
        
        self.logs("--- RL Evaluator Initialized ---")
        self.logs(f"Model Class: {model_class_name}")
        self.logs(f"Device: {self.device}")
        self.logs(f"Total Layers: {self.total_layers}")
        self.logs(f"Compiled Model Unwrapped: {self.is_compiled}") # Debug info

    def is_main_process(self):
        """Check if this is the main process (Rank 0) to avoid duplicate logs."""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0
        
    def get_rank(self):
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def logs(self, result):
        """Only log if main process. Outputs to output.log AND rl_training_log...txt"""
        if self.is_main_process():
            print(result, flush=True)  # 强制刷新控制台输出
            with open(self.log_path, 'a') as f:
                f.write(f"{result}\n")
                f.flush() 

    def log_progress(self, message):
        """
        Dedicated logger for batch-level progress. 
        Outputs to output.log AND rl_progress_log...txt
        """
        rank = self.get_rank()
        msg_with_rank = f"[Rank {rank}] {message}"
        
        # Print to console for ALL ranks to see if one is stuck
        print(msg_with_rank, flush=True) 
        
        if self.is_main_process():
            with open(self.progress_log_path, 'a') as f:
                f.write(f"{msg_with_rank}\n")
                f.flush() 

    # [修改点 1]：拆分数据集获取方法，明确训练集和验证集的用途
    def get_train_batch_data(self, num_batches):
        """Safely get batch data from TRAINING set."""
        batches = []
        for _ in range(num_batches):
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.dataloader) # Reset training iterator
                batch = next(self.train_iter)
            batches.append(batch)
        return batches

    def get_eval_batch_data(self, num_batches):
        """Safely get batch data from VALIDATION/TEST set."""
        batches = []
        for _ in range(num_batches):
            try:
                batch = next(self.test_iter)
            except StopIteration:
                # 可能导致测试数据不干净
                self.test_iter = iter(self.dataloader_test) # Reset test iterator
                batch = next(self.test_iter)
            batches.append(batch)
        return batches

    def choose_action(self, layer_idx, current_degree, q_table):
        """更新 choose_action 接受特定的 q_table"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3) 
        else:
            q_values = q_table[layer_idx, current_degree, :]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def apply_configuration(self):
        """分别应用 GELU 和 Softmax 的配置"""
        
        # 1. 应用 GELU 配置
        gelu_map = {d: [] for d in range(5)}
        for idx, deg in enumerate(self.current_gelu_degrees):
            gelu_map[deg].append(idx)
            
        if gelu_map[0]:
            self.reversible_layer_handler.restore_layer_gelu(gelu_map[0], self.layers_attribute)
        
        for d in range(1, 5):
            if gelu_map[d]:
                self.reversible_layer_handler.replace_layer_gelu(
                    layer_indices=gelu_map[d], 
                    layer_name=self.layers_attribute, 
                    degree=d
                )

        # 2. 应用 Softmax 配置 (范围 [1, 6])
        # 注意：这里我们支持到 Degree 6，所以 range 要到 7
        softmax_map = {d: [] for d in range(7)}
        for idx, deg in enumerate(self.current_softmax_degrees):
            # 简单的边界保护，防止 index error
            if deg < 7:
                softmax_map[deg].append(idx)
            
        if softmax_map[0]:
            self.reversible_layer_handler.restore_layer_softmax(softmax_map[0], self.layers_attribute)
            
        # 循环 1 到 6
        for d in range(1, 7):
            if softmax_map[d]:
                # 根据 Softmax degree 计算 bound
                # 随着 degree 增大 (例如 6)，bound 越低计算范围越大，精度越高
                # soft_bound = -2 * d 
                # 目前改为直接从function_handler中的Exp_bound中读取
                
                self.reversible_layer_handler.replace_layer_softmax(
                    layer_indices=softmax_map[d],
                    layer_name=self.layers_attribute,
                    degree=d
                    # lower_bound=soft_bound
                )

    def calculate_reward(self, loss):
        """
        [Modification] New Reward Function:
        Goal: Penalize loss deviation from baseline heavily using exponential function.
              Encourage lower degrees.
        """
        # 1. 精度奖励 (Accuracy Reward) - 使用指数惩罚
        # Baseline Loss (G4S6) is measured at Step 0.
        # If current loss > baseline_loss, penalty increases exponentially.
        
        if self.baseline_loss is None:
            # Fallback if step 0 somehow failed to set it, though code flow prevents this.
            target_loss = 0.5 
        else:
            target_loss = self.baseline_loss

        # Calculate deviation (diff)
        # We want diff to be as small (or negative) as possible.
        diff = loss - target_loss
        
        # Exponential Penalty: e^(k * diff)
        # If diff > 0 (loss increased), penalty explodes.
        # If diff <= 0 (loss improved or same), penalty is small (reward high).
        
        # Parameter k controls sensitivity. 
        # Example: if k=100:
        #   diff = 0.01 (0.33->0.34) -> e^1 = 2.71 penalty
        #   diff = 0.02 (0.33->0.35) -> e^2 = 7.38 penalty
        k = 100.0 
        
        # Reward component 1: Loss preservation
        # We use negative exponential to convert penalty to reward.
        # Max reward is 10 (when loss << target). 
        # At diff=0, reward is 5.
        # At diff>0, reward drops rapidly towards 0.
        
        # Formula: R_acc = 10 * exp(- k * max(0, diff)) 
        # But user wants "loss from 0.33 to 0.34 hurts MORE than 0.32 to 0.33".
        # Exponential deviation naturally handles this if we look at the 'drop' in reward.
        # e.g. R(0.32) vs R(0.33) drop is smaller than R(0.33) vs R(0.34) ??? 
        # Actually standard exp(-x) has decreasing derivative magnitude.
        # To make penalty *increasingly* severe as loss grows higher, we might want:
        # Reward = Constant - (alpha * e^(beta * loss))
        # Let's try: R = 20 - e^(10 * loss)
        # If loss=0.33, e^3.3 = 27. R = -7
        # If loss=0.34, e^3.4 = 29. R = -9. Drop = 2
        # If loss=0.5, e^5 = 148. R = -128.
        # This satisfies "higher loss -> steeper penalty slope".
        
        # Let's use a simpler formulation relative to baseline:
        # Reward = 10 - 10 * (e^(k * (loss - baseline)) - 1)
        # If loss == baseline, R = 10.
        # If loss > baseline, R drops exponentially fast (-infinity).
        # This is very strict.
        
        k_sensitivity = 20.0
        # diff = loss - self.baseline_loss
        # If loss=0.62, baseline=0.62. diff=0. e^0=1. R_acc = 10.
        # If loss=0.63. diff=0.01. e^0.2=1.22. R_acc = 7.8. (Drop 2.2)
        # If loss=0.64. diff=0.02. e^0.4=1.49. R_acc = 5.1. (Drop 2.7 -> Larger drop!)
        
        accuracy_reward = 10.0 - 10.0 * (math.exp(k_sensitivity * max(0, loss - self.baseline_loss)) - 1.0)
        
        # Clamp negative rewards to avoid destabilizing Q-learning too much, or allow them?
        # Q-learning can handle negatives. Let's clamp at -20 to prevent explosion.
        accuracy_reward = max(-20.0, accuracy_reward)

        # 2. 效率奖励 (Efficiency Reward)
        # GELU Cost: 4(max)->4, 1(min)->1
        gelu_cost_map = {0: 10, 4: 4, 3: 3, 2: 2, 1: 1}
        # Softmax Cost: 6(max)->6, 1(min)->1
        softmax_cost_map = {0: 10, 6: 6, 5: 5, 4: 4, 3: 3, 2: 2, 1: 1}
        
        gelu_costs = [gelu_cost_map.get(d, 4) for d in self.current_gelu_degrees]
        softmax_costs = [softmax_cost_map.get(d, 6) for d in self.current_softmax_degrees]
        
        avg_cost = (np.mean(gelu_costs) + np.mean(softmax_costs)) / 2.0
        
        # We want to encourage low cost.
        # Max possible cost approx 5 (G4+S6). Min approx 1.
        # Reward range: 0 to 2.
        efficiency_reward = (5.0 - avg_cost) * 0.5 
        
        total_reward = accuracy_reward + efficiency_reward
        return total_reward

    def get_final_metrics(self, name="Evaluation"):
        """在强化学习结束后，对当前配置进行完整评估。"""
        self.log_progress(f"\n>>> Starting Final Evaluation: {name} <<<")
        
        # 应用当前配置
        self.apply_configuration()
        self.reversible_layer_handler.model.eval()
        
        # 使用 get_eval_batch_data 从验证集获取数据
        eval_batches = self.get_eval_batch_data(self.eval_steps)
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.inference_mode():
            self.reversible_layer_handler.model.to(self.device)
            for batch in eval_batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.reversible_layer_handler.model(**batch)
                
                total_loss += outputs.loss.item()
                
                logits = outputs.logits.squeeze().detach().cpu().numpy()
                labels = batch["labels"].detach().cpu().numpy()
                
                if np.ndim(logits) == 0:
                    logits = [logits]
                
                all_predictions.extend(logits)
                all_labels.extend(labels)
        
        avg_loss = total_loss / len(eval_batches)
        
        if dist.is_initialized():
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        try:
            pearson_corr = pearsonr(all_predictions, all_labels)[0]
            spearman_corr = spearmanr(all_predictions, all_labels)[0]
        except Exception as e:
            self.log_progress(f"Error calculating correlation: {e}")
            pearson_corr = 0.0
            spearman_corr = 0.0

        self.logs(f"[{name}] Final Metrics:")
        self.logs(f"  - Avg Loss: {avg_loss:.4f}")
        self.logs(f"  - Pearson:  {pearson_corr:.4f}")
        self.logs(f"  - Spearman: {spearman_corr:.4f}")
        self.logs(f"  - Gelu Degs: {self.current_gelu_degrees.tolist()}")
        self.logs(f"  - Softmax Degs: {self.current_softmax_degrees.tolist()}")
        
        return avg_loss

    def on_evaluate(self, args, state, control, **kwargs):
        self.logs("\n" + "="*60)
        self.logs("STARTING REINFORCEMENT LEARNING TUNING")
        self.logs("="*60)
        
        self.log_progress(f"\n{'='*20} NEW EVALUATION SESSION {'='*20}")
        
        best_loss = float('inf')
        best_config_gelu = None
        best_config_softmax = None
        
        loss_history = []
        reward_history = []
        
        header = f"{'Step':<5} | {'Loss':<10} | {'Reward':<10} | {'Epsilon':<8} | {'GELU Degs'} | {'Softmax Degs'}"
        self.logs(header)
        self.logs("-" * len(header))

        # [Modification] Step 0: Baseline Evaluation
        # Force set state to G4 S6 to calculate baseline_loss
        self.log_progress("Step 0: Establishing Baseline with Max Approx (G4, S6)")
        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        self.current_softmax_degrees = np.full(self.total_layers, 6, dtype=int)
        
        try:
            self.apply_configuration()
            self.reversible_layer_handler.model.eval()
            eval_batches = self.get_train_batch_data(self.steps_per_epoch)
            total_loss = 0
            with torch.inference_mode():
                self.reversible_layer_handler.model.to(self.device)
                for batch in eval_batches:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.reversible_layer_handler.model(**batch)
                    total_loss += outputs.loss.item()
            
            avg_loss = total_loss / len(eval_batches)
            if dist.is_initialized():
                loss_tensor = torch.tensor(avg_loss, device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = loss_tensor.item()
            
            self.baseline_loss = avg_loss
            self.logs(f"Step 0 | Baseline Loss (G4S6): {self.baseline_loss:.4f}")
            
            # Reset model state before starting loop
            copy.deepcopy = _ORIGINAL_DEEPCOPY
            if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                self.reversible_layer_handler.restore_all()
        except Exception as e:
            self.logs(f"Step 0 Failed: {e}")
            self.baseline_loss = 0.5 # Fallback

        for step in range(1, self.rl_epochs + 1):
            try:
                # --- DDP Synchronization for Actions ---
                current_seed = 42 + step
                random.seed(current_seed)
                np.random.seed(current_seed)
                torch.manual_seed(current_seed)
                
                # 1. Choose Actions (Separately for GELU and Softmax)
                actions_gelu = [] 
                actions_softmax = []
                
                prev_degrees_gelu = self.current_gelu_degrees.copy()
                prev_degrees_softmax = self.current_softmax_degrees.copy()
                
                # --- GELU Action Selection (Range 1-4) [不可选0] ---
                for i in range(self.total_layers):
                    curr_deg = self.current_gelu_degrees[i]
                    action = self.choose_action(i, curr_deg, self.q_table_gelu)
                    
                    # 限制范围 [1, 4]
                    if action == ACTION_DECREASE: 
                        new_deg = max(1, curr_deg - 1)
                    elif action == ACTION_INCREASE: 
                        new_deg = min(4, curr_deg + 1)
                    else: # STAY
                        new_deg = curr_deg
                    
                    self.current_gelu_degrees[i] = new_deg
                    actions_gelu.append(action)

                # --- Softmax Action Selection (Range 1-6) [不可选0] ---
                for i in range(self.total_layers):
                    curr_deg = self.current_softmax_degrees[i]
                    action = self.choose_action(i, curr_deg, self.q_table_softmax)
                    
                    # 限制范围 [2, 6]
                    if action == ACTION_DECREASE: 
                        new_deg = max(2, curr_deg - 1)
                    elif action == ACTION_INCREASE: 
                        new_deg = min(6, curr_deg + 1)
                    else: # STAY
                        new_deg = curr_deg
                    
                    self.current_softmax_degrees[i] = new_deg
                    actions_softmax.append(action)
                
                # 2. Apply Config
                self.log_progress(f"RL Step {step} | Applying configuration...")
                self.apply_configuration()
                self.reversible_layer_handler.model.eval()
                
                # [修改点 1]：强化学习循环中使用 训练集 (get_train_batch_data)
                eval_batches = self.get_train_batch_data(self.steps_per_epoch)
                total_loss = 0
                
                self.log_progress(f"--- RL Step {step}/{self.rl_epochs} Started ---")
                
                with torch.inference_mode():
                    self.reversible_layer_handler.model.to(self.device)
                    for batch_idx, batch in enumerate(eval_batches):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.reversible_layer_handler.model(**batch)
                        
                        batch_loss = outputs.loss.item()
                        total_loss += batch_loss
                        
                        # --- MODIFICATION: Log EVERY single batch ---
                        self.log_progress(f"RL Step {step} | Batch {batch_idx + 1}/{self.steps_per_epoch} | Batch Loss: {batch_loss:.4f}")
                
                avg_loss = total_loss / len(eval_batches)
                self.log_progress(f"RL Step {step} | Eval Loop Finished. Avg Loss: {avg_loss:.4f}")

                # --- DDP Synchronization for Loss ---
                if dist.is_initialized():
                    self.log_progress(f"RL Step {step} | Waiting for DDP sync...")
                    loss_tensor = torch.tensor(avg_loss, device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    avg_loss = loss_tensor.item()
                    self.log_progress(f"RL Step {step} | DDP Sync Complete. Global Avg Loss: {avg_loss:.4f}")
                
                # 3. Calculate Reward & Update Q-Table
                reward = self.calculate_reward(avg_loss) 
                
                # Append metrics to history for plotting
                loss_history.append(avg_loss)
                reward_history.append(reward)
                
                # Update Q-Tables (Separately)
                for i in range(self.total_layers):
                    # Update GELU Q-Table
                    s = prev_degrees_gelu[i]
                    a = actions_gelu[i]
                    s_prime = self.current_gelu_degrees[i]
                    q_predict = self.q_table_gelu[i, s, a]
                    q_target = reward + self.gamma * np.max(self.q_table_gelu[i, s_prime, :])
                    self.q_table_gelu[i, s, a] += self.alpha * (q_target - q_predict)

                    # Update Softmax Q-Table
                    s = prev_degrees_softmax[i]
                    a = actions_softmax[i]
                    s_prime = self.current_softmax_degrees[i]
                    q_predict = self.q_table_softmax[i, s, a]
                    q_target = reward + self.gamma * np.max(self.q_table_softmax[i, s_prime, :])
                    self.q_table_softmax[i, s, a] += self.alpha * (q_target - q_predict)

                # 5. Update Epsilon & Log
                self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                # [关键修改]：更新 best_config 为当前 Step 的配置
                best_config_gelu = self.current_gelu_degrees.copy()
                best_config_softmax = self.current_softmax_degrees.copy()
                
                gelu_str = str(self.current_gelu_degrees.tolist())
                softmax_str = str(self.current_softmax_degrees.tolist())
                
                # [Modification] Updated log format for separated arrays
                log_str = f"{step:<5} | {avg_loss:<10.4f} | {reward:<10.2f} | {self.epsilon:<8.2f} | {gelu_str} | {softmax_str}"
                self.logs(log_str)
                
                # --- CRITICAL: Restore entire model using restore_all() ---
                self.log_progress(f"RL Step {step} | Restoring model state...")
                
                # Make sure deepcopy is AVAILABLE for restore_all
                copy.deepcopy = _ORIGINAL_DEEPCOPY
                
                # Safety check before calling restore_all
                if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                    try:
                        self.reversible_layer_handler.restore_all()
                    except Exception as restore_e:
                        self.log_progress(f"Warning: restore_all failed: {restore_e}. Trying to proceed with in-place updates.")
                else:
                    self.log_progress(f"Warning: backup_model is None or missing. Skipping restore_all. This might cause accumulation errors.")
            
            except Exception as e:
                self.log_progress(f"CRITICAL ERROR in Step {step}: {str(e)}")
                traceback.print_exc()

        # Plot Learning Curve
        if self.is_main_process() and len(loss_history) > 0:
            try:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Loss')
                plt.axhline(y=self.baseline_loss, color='r', linestyle='--', label='Baseline') # Plot baseline
                plt.title('Loss vs Steps')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(reward_history) + 1), reward_history, marker='o', color='orange', label='Reward')
                plt.title('Reward vs Steps')
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(self.plot_path)
                self.logs(f"Learning curve saved to {self.plot_path}")
            except Exception as e:
                self.logs(f"Failed to plot learning curve: {e}")

        self.logs("="*60)
        self.logs("RL TUNING FINISHED")
        
        # --- 最终评估阶段 ---
        self.logs("\n" + "="*30)
        self.logs("FINAL EVALUATION PHASE")
        self.logs("="*30)

        # 1. Final RL Policy
        self.current_gelu_degrees = best_config_gelu
        self.current_softmax_degrees = best_config_softmax
        self.get_final_metrics(name="Final RL Policy (Last Step)")
        
        copy.deepcopy = _ORIGINAL_DEEPCOPY
        try:
            if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                self.reversible_layer_handler.restore_all()
        except: pass

        # 2. Baseline: Origin
        self.current_gelu_degrees = np.zeros(self.total_layers, dtype=int)
        self.current_softmax_degrees = np.zeros(self.total_layers, dtype=int)
        self.get_final_metrics(name="Baseline: All Origin (Degree 0)")
        
        copy.deepcopy = _ORIGINAL_DEEPCOPY
        try:
            if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                self.reversible_layer_handler.restore_all()
        except: pass

        # 3. Baseline: Degree 1
        self.current_gelu_degrees = np.ones(self.total_layers, dtype=int)
        self.current_softmax_degrees = np.ones(self.total_layers, dtype=int)
        self.get_final_metrics(name="Baseline: All Degree 1")

        # 4. [New Request] Baseline: All Max Approx (Gelu 4, Softmax 6)
        # 恢复模型
        copy.deepcopy = _ORIGINAL_DEEPCOPY
        try:
            if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                self.reversible_layer_handler.restore_all()
        except: pass
        
        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        self.current_softmax_degrees = np.full(self.total_layers, 6, dtype=int)
        self.get_final_metrics(name="Baseline: All Max Approx (Gelu 4, Softmax 6)")
        
        # 5: Random placement (independent shuffle)
        for rand_seed in [0, 1, 2, 3, 4]:
            copy.deepcopy = _ORIGINAL_DEEPCOPY
            try:
                if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                    self.reversible_layer_handler.restore_all()
            except:
                pass
            
            rand_gelu, rand_softmax = self.make_randomized_layer_config_independent(
                best_config_gelu, best_config_softmax, seed=rand_seed
            )
            self.current_gelu_degrees = rand_gelu
            self.current_softmax_degrees = rand_softmax
            self.get_final_metrics(name=f"Baseline: Random Placement (indep shuffle, seed={rand_seed})")

        # 最后恢复到最佳配置
        self.current_gelu_degrees = best_config_gelu
        self.current_softmax_degrees = best_config_softmax
        self.apply_configuration()

    def compute_spectral_norm(self, *args, **kwargs): pass
    def compute_importance_matrix_lipschitz(self, *args, **kwargs): pass
    
    def make_randomized_layer_config_independent(self, best_gelu: np.ndarray,
                                                 best_softmax: np.ndarray,
                                                 seed: int = 0):
        """
        GELU / Softmax 各自独立 shuffle（不配对）。
        保证：best_gelu 的元素多重集不变；best_softmax 的元素多重集不变。
        """
        rng = np.random.default_rng(seed)

        gelu_shuffled = best_gelu.copy()
        softmax_shuffled = best_softmax.copy()

        rng.shuffle(gelu_shuffled)      # 原地打乱
        rng.shuffle(softmax_shuffled)   # 原地打乱

        return gelu_shuffled.astype(int), softmax_shuffled.astype(int)