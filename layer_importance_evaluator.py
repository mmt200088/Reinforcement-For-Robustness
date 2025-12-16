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
        self.rl_epochs = 100          
        self.eval_batch_count = 100   
        
        # Q-Learning 参数
        self.alpha = 0.1     
        self.gamma = 0.9     
        self.epsilon = 0.3   
        self.epsilon_decay = 0.95
        
        # [修改点 1] 分离 GELU 和 Softmax 的状态数组
        # 初始状态: 全部设为 2
        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        self.current_softmax_degrees = np.full(self.total_layers, 4, dtype=int)
        
        # [修改点 2] 分离 GELU 和 Softmax 的 Q-Table
        # Q-Table: [Layer, Current_Degree(0-4), Action(0,1,2)]
        self.q_table_gelu = np.zeros((self.total_layers, 5, 3))
        self.q_table_softmax = np.zeros((self.total_layers, 5, 3))
        
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
                self.test_iter = iter(self.dataloader_test) # Reset test iterator
                batch = next(self.test_iter)
            batches.append(batch)
        return batches

    def choose_action(self, layer_idx, current_degree, q_table):
        """[修改点 3] 更新 choose_action 接受特定的 q_table"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 3) 
        else:
            q_values = q_table[layer_idx, current_degree, :]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def apply_configuration(self):
        """[修改点 4] 分别应用 GELU 和 Softmax 的配置"""
        
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

        # 2. 应用 Softmax 配置
        softmax_map = {d: [] for d in range(5)}
        for idx, deg in enumerate(self.current_softmax_degrees):
            softmax_map[deg].append(idx)
            
        if softmax_map[0]:
            self.reversible_layer_handler.restore_layer_softmax(softmax_map[0], self.layers_attribute)
            
        for d in range(1, 5):
            if softmax_map[d]:
                # 根据 Softmax degree 计算 bound (保持与原逻辑类似的比例)
                # 原逻辑: d_soft = d_gelu + 1; bound = -2 * d_gelu - 2
                # 现逻辑: bound = -2 * (d - 1) - 2 = -2d (例如 d=2 -> bound=-4)
                # 或者直接简化为 -2 * d
                soft_bound = -2 * d 
                
                self.reversible_layer_handler.replace_layer_softmax(
                    layer_indices=softmax_map[d],
                    layer_name=self.layers_attribute,
                    degree=d,
                    lower_bound=soft_bound
                )

    def calculate_reward(self, loss):
        """[修改点 5] 计算奖励时同时考虑 GELU 和 Softmax 的开销"""
        # 1. 精度奖励
        accuracy_reward = 2.0 / (loss + 1e-4)
        
        # 2. 效率奖励
        cost_map = {0: 10, 4: 4, 3: 3, 2: 2, 1: 1}
        
        gelu_costs = [cost_map[d] for d in self.current_gelu_degrees]
        softmax_costs = [cost_map[d] for d in self.current_softmax_degrees]
        
        # 计算综合平均开销
        avg_cost = (np.mean(gelu_costs) + np.mean(softmax_costs)) / 2.0
        
        efficiency_reward = (10 - avg_cost) * 0.1 
        
        total_reward = accuracy_reward + efficiency_reward
        return total_reward

    def get_final_metrics(self, name="Evaluation"):
        """
        在强化学习结束后，对当前配置进行完整评估。
        [修改点]：这里使用验证集 (Test Set) 进行评估。
        """
        self.log_progress(f"\n>>> Starting Final Evaluation: {name} <<<")
        
        # 应用当前配置
        self.apply_configuration()
        self.reversible_layer_handler.model.eval()
        
        # 使用 get_eval_batch_data 从验证集获取数据
        eval_batches = self.get_eval_batch_data(self.eval_batch_count)
        
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
                
                # 处理 batch_size=1 的情况
                if np.ndim(logits) == 0:
                    logits = [logits]
                
                all_predictions.extend(logits)
                all_labels.extend(labels)
        
        avg_loss = total_loss / len(eval_batches)
        
        # 汇总 DDP 结果
        if dist.is_initialized():
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        # 计算相关系数
        try:
            pearson_corr = pearsonr(all_predictions, all_labels)[0]
            spearman_corr = spearmanr(all_predictions, all_labels)[0]
        except Exception as e:
            self.log_progress(f"Error calculating correlation: {e}")
            pearson_corr = 0.0
            spearman_corr = 0.0

        # 输出结果
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
        
        # Initialize history lists for plotting
        loss_history = []
        reward_history = []
        
        # [Modification] Updated header to show both arrays
        header = f"{'Step':<5} | {'Loss':<10} | {'Reward':<10} | {'Epsilon':<8} | {'GELU Degs'} | {'Softmax Degs'}"
        self.logs(header)
        self.logs("-" * len(header))

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
                
                # --- GELU Action Selection ---
                for i in range(self.total_layers):
                    curr_deg = self.current_gelu_degrees[i]
                    action = self.choose_action(i, curr_deg, self.q_table_gelu)
                    
                    new_deg = curr_deg
                    # [修改点 2]：状态转移逻辑修改，范围限制在 1-4
                    if action == ACTION_DECREASE: 
                        new_deg = max(1, curr_deg - 1)
                    elif action == ACTION_INCREASE: 
                        new_deg = min(4, curr_deg + 1)
                    
                    self.current_gelu_degrees[i] = new_deg
                    actions_gelu.append(action)

                # --- Softmax Action Selection ---
                for i in range(self.total_layers):
                    curr_deg = self.current_softmax_degrees[i]
                    action = self.choose_action(i, curr_deg, self.q_table_softmax)
                    
                    new_deg = curr_deg
                    if action == ACTION_DECREASE: 
                        new_deg = max(1, curr_deg - 1)
                    elif action == ACTION_INCREASE: 
                        new_deg = min(4, curr_deg + 1)
                    
                    self.current_softmax_degrees[i] = new_deg
                    actions_softmax.append(action)
                
                # 2. Apply Config
                self.log_progress(f"RL Step {step} | Applying configuration...")
                self.apply_configuration()
                self.reversible_layer_handler.model.eval()
                
                # [修改点 1]：强化学习循环中使用 训练集 (get_train_batch_data)
                eval_batches = self.get_train_batch_data(self.eval_batch_count)
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
                        self.log_progress(f"RL Step {step} | Batch {batch_idx + 1}/{self.eval_batch_count} | Batch Loss: {batch_loss:.4f}")
                
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
                
                # Plot Loss
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Loss')
                plt.title('Loss vs Steps')
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.grid(True)
                
                # Plot Reward
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

        # 1. 恢复到 RL 找到的最佳配置
        self.current_gelu_degrees = best_config_gelu
        self.current_softmax_degrees = best_config_softmax
        self.get_final_metrics(name="Best RL Policy")
        
        # 2. 恢复模型状态以便后续评估
        copy.deepcopy = _ORIGINAL_DEEPCOPY
        try:
            if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                self.reversible_layer_handler.restore_all()
        except: pass

        # 3. 评估基准：全 Origin (Degree 0)
        self.current_gelu_degrees = np.zeros(self.total_layers, dtype=int)
        self.current_softmax_degrees = np.zeros(self.total_layers, dtype=int)
        self.get_final_metrics(name="Baseline: All Origin (Degree 0)")
        
        # 恢复模型
        copy.deepcopy = _ORIGINAL_DEEPCOPY
        try:
            if hasattr(self.reversible_layer_handler, 'backup_model') and self.reversible_layer_handler.backup_model is not None:
                self.reversible_layer_handler.restore_all()
        except: pass

        # 4. 评估基准：全 Degree 1 (最低阶近似)
        self.current_gelu_degrees = np.ones(self.total_layers, dtype=int)
        self.current_softmax_degrees = np.ones(self.total_layers, dtype=int)
        self.get_final_metrics(name="Baseline: All Degree 1")

        # 最后恢复到最佳配置，以便 Trainer 保存或后续使用
        self.current_gelu_degrees = best_config_gelu
        self.current_softmax_degrees = best_config_softmax
        self.apply_configuration()

    # 兼容旧接口
    def compute_spectral_norm(self, *args, **kwargs): pass
    def compute_importance_matrix_lipschitz(self, *args, **kwargs): pass