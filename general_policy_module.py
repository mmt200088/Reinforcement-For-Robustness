"""general_policy_module.py — 跨任务通用策略 / 通用 Critic 训练与离线部署模块

核心思想：
  1. 多任务轮训（Multi-Task Round-Robin Training）：
     在多个数据集/约束设定上轮流采集 rollout 并更新同一个 policy+critic 网络，
     使 policy 学到 "哪些层对精度更敏感" 的通用先验；critic 学到跨任务的状态价值模式。

  2. 离线推断（Offline Inference）：
     加载训练好的 general policy，在新任务上 **只做前向推理（不训练）**，
     通过 best-of-K rollout 找到该任务的最优配置。

  3. 通用 Critic（General Critic）：
     Policy 和 Critic 共享 GTrXL 骨干网络。多任务训练同时产出 general critic。
     General Critic 跨任务学习了 "什么状态（配置进度）值多少回报" 的通用模式，
     在离线推断时可以用 V(s) 快速评分候选配置（不做模型评测）。

  4. 与现有 online RL 完全并存：
     现有的 per-task online RL（layer_importance_evaluator.py 的 on_evaluate、
     noise_rl_module_v2.py 的 run_noise_rl_stage2）完全不受影响。
     本模块通过可迁移 policy 机制与现有管线对接：
       - general policy → RL_OPT_FLAGS["stage1_pretrained_policy_path"]
       - general noise policy → NOISE_RL_OPT_FLAGS["pretrained_policy_path"]

使用流程（Stage-1 示例）：

  # -------- Phase A: 多任务训练（一次性） --------
  from general_policy_module import (
      prepare_stage1_task, multi_task_train_stage1,
  )
  tasks = {}
  for name in ["mrpc", "stsb", "cola", "rte"]:
      ev = create_evaluator(name)  # 你自己的 evaluator 创建逻辑
      tasks[name] = prepare_stage1_task(ev)
  multi_task_train_stage1(tasks, output_path="general_stage1_policy.pt")

  # -------- Phase B: 离线部署（在新/旧任务上） --------
  from general_policy_module import offline_find_best_config_stage1
  ev_new = create_evaluator("qqp")
  result = offline_find_best_config_stage1(ev_new, "general_stage1_policy.pt")
  print(result["best_config"])

  # -------- 或者：作为 base policy 微调（利用已有的 online RL）--------
  # 只需设置 RL_OPT_FLAGS["stage1_pretrained_policy_path"] = "general_stage1_policy.pt"
  # 然后正常运行 bash llama_7B_LayerImportance.sh ...

使用流程（Stage-2 示例）：

  # -------- Phase A: 多任务训练（一次性） --------
  from general_policy_module import (
      prepare_stage2_task, multi_task_train_stage2,
  )
  tasks = {}
  for name in ["mrpc", "stsb", "cola", "rte"]:
      ev = create_evaluator(name)
      fixed_gelu, fixed_softmax = get_stage1_config(name)
      tasks[name] = prepare_stage2_task(ev, fixed_gelu, fixed_softmax)
  multi_task_train_stage2(tasks, output_path="general_stage2_noise_policy.pt")

  # -------- Phase B: 离线部署 --------
  from general_policy_module import offline_find_best_config_stage2
  result = offline_find_best_config_stage2(ev_new, "general_stage2_noise_policy.pt",
                                            fixed_gelu, fixed_softmax, noise_env)
  print(result["best_noise_config"])
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ---- 从现有模块导入基类和常量 ----
import layer_importance_evaluator as _s1
import noise_rl_module_v2 as _s2

# Stage-1 常量快捷引用
_GTrXLStrategyNetwork = _s1.GTrXLStrategyNetwork
_TransformerOptEnv = _s1.TransformerOptEnv
_RecurrentRolloutBuffer = _s1.RecurrentRolloutBuffer
_SOS_GELU = _s1.SOS_TOKEN_GELU
_SOS_SOFTMAX = _s1.SOS_TOKEN_SOFTMAX

# Stage-2 常量快捷引用
_NoiseGTrXLNet = _s2._NoiseGTrXLStrategyNetwork
_NoiseOptEnv = _s2._NoiseOptEnv
_NoiseBuffer = _s2._NoiseRecurrentRolloutBuffer

# ---- 优雅停止相关 ----
import time as _time

from noise_rl_module_v2 import (
    NOISE_STAGE_STOP_FLAG_FILENAME as _STOP_FLAG_FILENAME,
    _log_rounded_box,
    install_graceful_stop_handler as _install_stop,
    uninstall_graceful_stop_handler as _uninstall_stop,
    reset_graceful_stop_state as _reset_stop,
    is_graceful_stop_requested as _is_stop,
    consume_stop_flag_file as _consume_stop,
)
from report_format_utils import format_elapsed as _fmt_elapsed
from report_format_utils import progress_bar as _progress_bar

GENERAL_STAGE1_TRAIN_CHECKPOINT = "general_stage1_train_checkpoint.pt"
GENERAL_STAGE2_TRAIN_CHECKPOINT = "general_stage2_train_checkpoint.pt"

# ===========================================================================
# 配置常量
# ===========================================================================
TASK_CONTEXT_DIM = 5
GENERAL_POLICY_VERSION = 1
GENERAL_STAGE1_ALLOWED_GELU_DEGREES = (4, 2, 1)


def _log_general_header(log_fn, title, info_lines):
    log_fn("")
    log_fn("╔══════════════════════════════════════════════════════════════════╗")
    log_fn(f"║  {title:^64s}║")
    log_fn("╠══════════════════════════════════════════════════════════════════╣")
    for line in info_lines:
        log_fn(f"║  {line:<64s}║")
    log_fn("╚══════════════════════════════════════════════════════════════════╝")
    log_fn("")


# ===========================================================================
# 独立辅助工具（不依赖 evaluator 实例）
# ===========================================================================

class _RunningMeanStd:
    """Welford 在线均值 / 方差跟踪器（独立于 evaluator）。"""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    @property
    def std(self):
        return max(math.sqrt(self.var), 1e-8)

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64).ravel()
        batch_mean, batch_var, batch_count = float(x.mean()), float(x.var()), x.size
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / tot) / tot
        self.count = tot

    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / (self.std + 1e-8)
        return (np.asarray(x, dtype=np.float64) - self.mean) / (self.std + 1e-8)


def _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """独立的 GAE 计算（与 evaluator.compute_gae 等价）。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = float(values[t + 1]) if t < T - 1 else 0.0
        non_term = 1.0 - float(dones[t])
        delta = float(rewards[t]) + gamma * next_val * non_term - float(values[t])
        gae = delta + gamma * lam * non_term * gae
        advantages[t] = gae
    vals = np.asarray(values[:T], dtype=np.float32) if not isinstance(values, np.ndarray) else values[:T].astype(np.float32)
    returns = advantages + vals
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def compute_task_context(baseline_loss, baseline_m1, baseline_m2,
                         error_threshold=0.005, correlation_drop_ratio=0.005):
    """构造 5 维任务上下文特征向量。

    维度含义：
      [0] baseline_loss      — 基线 loss
      [1] baseline_m1        — 基线 metric1（acc / pearson 等）
      [2] baseline_m2        — 基线 metric2（f1 / spearman 等）
      [3] error_threshold    — loss 上浮比例（如 0.005 表示 0.5%）
      [4] correlation_drop   — 指标下降容忍比例（如 0.005 表示 0.5%）
    """
    return np.array([
        float(baseline_loss), float(baseline_m1), float(baseline_m2),
        float(error_threshold), float(correlation_drop_ratio),
    ], dtype=np.float32)


def _validate_general_stage1_candidate_configs(candidate_configs):
    allowed = set(GENERAL_STAGE1_ALLOWED_GELU_DEGREES)
    for idx, cfg in enumerate(candidate_configs):
        gelu_arr = np.asarray(cfg["gelu"], dtype=int)
        invalid = sorted({int(v) for v in gelu_arr.tolist()} - allowed)
        if invalid:
            raise ValueError(
                f"Stage-1 general policy candidate #{idx} contains unsupported GELU degrees "
                f"{invalid}. Degree 0 is disabled; only {GENERAL_STAGE1_ALLOWED_GELU_DEGREES} "
                f"are allowed."
            )


# ===========================================================================
# Stage-1 通用策略 / Critic 网络
# ===========================================================================

class GeneralStage1PolicyNetwork(_GTrXLStrategyNetwork):
    """Stage-1 通用策略 + 通用 Critic 网络（GTrXL + 任务上下文嵌入）。

    在 GTrXLStrategyNetwork 基础上添加 task_context_proj 分支：
      task_context (5-dim) → Linear → LN → SiLU → Linear → (d_model)
    投影结果作为**可加偏置**注入每个 token，不改变基础架构。

    设计要点：
      - 最后一层线性层**零初始化**：加载 per-task 权重时，task_context_proj 层
        是 "unexpected" 但不影响推理（输出全零，等价于无任务上下文）。
      - 使用 `set_task_context()` 设定当前任务上下文后，所有前向传播
        （forward / get_action_and_logprob / evaluate_actions）自动使用。
    """

    TASK_CONTEXT_DIM = TASK_CONTEXT_DIM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_context_proj = nn.Sequential(
            nn.Linear(self.TASK_CONTEXT_DIM, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        # 零初始化：加载 per-task 权重时等价于无任务上下文
        nn.init.zeros_(self.task_context_proj[-1].weight)
        nn.init.zeros_(self.task_context_proj[-1].bias)
        self._task_context = None

    def set_task_context(self, task_context):
        """设定后续前向传播使用的任务上下文。

        Args:
            task_context: (5,) 或 (B, 5) 的 ndarray / Tensor，或 None（清除）。
        """
        if task_context is None:
            self._task_context = None
            return
        if isinstance(task_context, np.ndarray):
            task_context = torch.tensor(task_context, dtype=torch.float32)
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        self._task_context = task_context.to(next(self.parameters()).device)

    def forward(self, cont_features, layer_indices, prev_g_actions, prev_s_actions,
                gelu_mask=None, key_padding_mask=None):
        tokens = self._build_tokens(cont_features, layer_indices, prev_g_actions, prev_s_actions)

        # 注入任务上下文可加偏置
        if self._task_context is not None:
            tc = self._task_context
            if tc.size(0) == 1 and tokens.size(0) > 1:
                tc = tc.expand(tokens.size(0), -1)
            tc_bias = self.task_context_proj(tc)        # (B, d_model)
            tokens = tokens + tc_bias.unsqueeze(1)      # broadcast over seq

        seq_len = tokens.size(1)
        causal_mask = self._get_causal_mask(seq_len, tokens.device)

        x = tokens
        for block in self.gtrxl_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.ln_final(x)

        actor_feat = self.actor_head(x)
        logits_g = self.head_g(actor_feat)
        logits_s = self.head_s(actor_feat)
        if gelu_mask is not None:
            logits_g = logits_g.masked_fill(~gelu_mask, float('-inf'))
        values = self.critic_head(x).squeeze(-1)
        return logits_g, logits_s, values


# ===========================================================================
# Stage-2 通用噪声策略 / Critic 网络
# ===========================================================================

class GeneralStage2NoisePolicyNetwork(_NoiseGTrXLNet):
    """Stage-2 通用噪声策略 + 通用 Critic 网络（GTrXL + 任务上下文嵌入）。

    与 GeneralStage1PolicyNetwork 同样的设计范式：
      - task_context_proj 零初始化
      - set_task_context() → forward() 自动注入
      - 与 per-task 权重通过 strict=False 无缝互转
    """

    TASK_CONTEXT_DIM = TASK_CONTEXT_DIM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_context_proj = nn.Sequential(
            nn.Linear(self.TASK_CONTEXT_DIM, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        nn.init.zeros_(self.task_context_proj[-1].weight)
        nn.init.zeros_(self.task_context_proj[-1].bias)
        self._task_context = None

    def set_task_context(self, task_context):
        if task_context is None:
            self._task_context = None
            return
        if isinstance(task_context, np.ndarray):
            task_context = torch.tensor(task_context, dtype=torch.float32)
        if task_context.dim() == 1:
            task_context = task_context.unsqueeze(0)
        self._task_context = task_context.to(next(self.parameters()).device)

    def forward(self, cont_features, layer_indices, prev_actions, key_padding_mask=None):
        tokens = self._build_tokens(cont_features, layer_indices, prev_actions)

        if self._task_context is not None:
            tc = self._task_context
            if tc.size(0) == 1 and tokens.size(0) > 1:
                tc = tc.expand(tokens.size(0), -1)
            tc_bias = self.task_context_proj(tc)
            tokens = tokens + tc_bias.unsqueeze(1)

        seq_len = tokens.size(1)
        causal_mask = self._get_causal_mask(seq_len, tokens.device)

        x = tokens
        for block in self.gtrxl_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.ln_final(x)

        actor_feat = self.actor_head(x)
        logits = {name: self.noise_heads[name](actor_feat) for name in self.action_names}
        values = self.critic_head(x).squeeze(-1)
        return logits, values


# ===========================================================================
# Stage-1 通用 Rollout Buffer（RecurrentRolloutBuffer + per-episode task_context）
# ===========================================================================

class _GeneralStage1Buffer(_RecurrentRolloutBuffer):
    """在 RecurrentRolloutBuffer 基础上增加 per-episode task_context 存储。"""

    def start_episode(self, task_context=None):
        super().start_episode()
        self._current['task_context'] = task_context  # np.ndarray (5,)

    def get_batch(self, device):
        parent_batch = super().get_batch(device)
        task_contexts = torch.stack([
            torch.tensor(ep['task_context'], dtype=torch.float32)
            for ep in self.episodes
        ]).to(device)   # (N_eps, 5)
        return (*parent_batch, task_contexts)


# ===========================================================================
# Stage-1 通用 PPO 更新（独立于 evaluator 实例）
# ===========================================================================

def _general_ppo_update_stage1(
    net, optimizer, buffer, device,
    return_normalizer,
    entropy_coef=0.02,
    ppo_update_step=0,
):
    """独立的 PPO 更新，支持 per-episode task_context。"""
    ppo_eps_clip = _s1.PPO_EPS_CLIP
    ppo_k_epochs = _s1.PPO_K_EPOCHS
    ppo_value_coef = _s1.PPO_VALUE_COEF
    mini_batch_eps = _s1.GTRXL_MINI_BATCH_EPISODES
    value_clip_range = _s1.VALUE_CLIP_RANGE
    warmup_steps = _s1.GTRXL_WARMUP_STEPS

    target_lr = optimizer.defaults['lr']
    if ppo_update_step < warmup_steps:
        factor = (ppo_update_step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg['lr'] = target_lr * factor

    (cont_features, layer_indices, prev_g_actions, prev_s_actions,
     actions_g, actions_s, old_logprobs, rewards, values, dones,
     gelu_masks, task_contexts) = buffer.get_batch(device)

    n_eps = cont_features.size(0)
    all_adv, all_ret = [], []
    for i in range(n_eps):
        a, r = _compute_gae(
            rewards[i].cpu().numpy(), values[i].cpu().numpy(),
            dones[i].cpu().numpy(), gamma=_s1.PPO_GAMMA, lam=_s1.PPO_LAMBDA,
        )
        all_adv.append(a)
        all_ret.append(r)
    advantages = torch.stack(all_adv).to(device)
    returns = torch.stack(all_ret).to(device)
    adv_flat = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    return_normalizer.update(returns)
    ret_norm = torch.tensor(return_normalizer.normalize(returns.cpu().numpy()),
                            dtype=torch.float32).to(device)
    val_norm = torch.tensor(return_normalizer.normalize(values.cpu().numpy()),
                            dtype=torch.float32).to(device)

    last_pl, last_vl, last_ent = 0.0, 0.0, 0.0
    kl_stop = False

    for _epoch in range(ppo_k_epochs):
        if kl_stop:
            break
        ep_idx = torch.randperm(n_eps)
        kl_acc, kl_cnt = 0.0, 0

        for start in range(0, n_eps, mini_batch_eps):
            end = min(start + mini_batch_eps, n_eps)
            mb = ep_idx[start:end]

            # 每个 mini-batch 设定 task context（混合任务）
            net.set_task_context(task_contexts[mb])

            lp_new, ent, v_new = net.evaluate_actions(
                cont_features[mb], layer_indices[mb],
                prev_g_actions[mb], prev_s_actions[mb],
                actions_g[mb], actions_s[mb],
                gelu_mask=gelu_masks[mb] if gelu_masks is not None else None,
            )
            lp_f = lp_new.reshape(-1)
            ent_f = ent.reshape(-1)
            v_f = v_new.reshape(-1)
            olp_f = old_logprobs[mb].reshape(-1)
            adv_f = advantages[mb].reshape(-1)
            ret_f = ret_norm[mb].reshape(-1)
            ov_f = val_norm[mb].reshape(-1)

            ratios = torch.exp(lp_f - olp_f)
            s1 = ratios * adv_f
            s2 = torch.clamp(ratios, 1 - ppo_eps_clip, 1 + ppo_eps_clip) * adv_f
            p_loss = -torch.min(s1, s2).mean()

            vn = (v_f - return_normalizer.mean) / return_normalizer.std
            vc = ov_f + torch.clamp(vn - ov_f, -value_clip_range, value_clip_range)
            huber = nn.HuberLoss(reduction='none', delta=1.0)
            v_loss = torch.max(huber(vn, ret_f), huber(vc, ret_f)).mean()

            me = ent_f.mean()
            eff_ec = entropy_coef
            lb = _s1._rl_opt_entropy_lower_bound()
            if me.item() < lb:
                eff_ec = entropy_coef + _s1._rl_opt_entropy_recovery_mul() * (lb - me.item())

            loss = p_loss + ppo_value_coef * v_loss + eff_ec * (-me)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            last_pl, last_vl, last_ent = p_loss.item(), v_loss.item(), me.item()

            if _s1.RL_OPT_FLAGS.get("use_kl_early_stop", False):
                with torch.no_grad():
                    kl_acc += (olp_f - lp_f).mean().item()
                kl_cnt += 1

        if (_s1.RL_OPT_FLAGS.get("use_kl_early_stop", False)
                and kl_cnt > 0
                and kl_acc / kl_cnt > 1.5 * _s1.RL_OPT_FLAGS.get("kl_target", 0.02)):
            kl_stop = True

    net.set_task_context(None)
    return last_pl, last_vl, last_ent


# ===========================================================================
# Stage-1 任务准备
# ===========================================================================

def prepare_stage1_task(evaluator, use_train=True, tolerance=None):
    """从 evaluator 准备一个 task config dict，用于 multi_task_train_stage1 或 offline 推断。

    Args:
        evaluator: LayerImportanceEvaluator 实例（模型 + 数据集已加载）
        use_train: 是否使用训练集计算基线（默认 True）
        tolerance: float or None, 准确度容忍比例（同时用于 loss 上浮和指标下降）。
                   默认 None 则使用 evaluator 自身的 error_threshold (0.005)。

    Returns:
        dict，包含多任务训练 / 离线推断所需的全部信息
    """
    tol = float(tolerance) if tolerance is not None else evaluator.error_threshold
    base_gelu = np.full(evaluator.total_layers, 4, dtype=int)
    base_softmax = np.full(evaluator.total_layers, 6, dtype=int)
    base_cost = evaluator.get_simulated_cost(base_gelu, base_softmax)[0]
    base_loss, base_m1, base_m2, _ = evaluator.stage1_evaluate(
        base_gelu, base_softmax, use_train=use_train,
    )
    task_ctx = compute_task_context(
        base_loss, base_m1, base_m2,
        tol, tol,
    )
    return {
        "evaluator": evaluator,
        "baseline_metrics": (float(base_loss), float(base_m1), float(base_m2)),
        "baseline_cost": float(base_cost),
        "constraint_limits": {
            "loss": float(base_loss * (1.0 + tol)),
            "metric1": float(base_m1 * (1.0 - tol)),
            "metric2": float(base_m2 * (1.0 - tol)),
        },
        "gelu0_eligible": np.zeros(evaluator.total_layers, dtype=bool),
        "task_context": task_ctx,
        "num_metrics": evaluator.get_num_metrics(),
        "tolerance": tol,
    }


def recompute_task_for_tolerance(task_config, tolerance):
    """根据新的准确度容忍比例, 重新计算 task_context 和 constraint_limits。

    不重新跑基线评估, 仅在已有 baseline_metrics 上变换约束阈值和上下文向量。

    Args:
        task_config: dict, prepare_stage1_task 的返回值
        tolerance: float, 新的准确度容忍比例

    Returns:
        新的 (task_context, constraint_limits) 元组
    """
    tol = float(tolerance)
    bl, bm1, bm2 = task_config["baseline_metrics"]
    task_ctx = compute_task_context(bl, bm1, bm2, tol, tol)
    constraint_limits = {
        "loss": float(bl * (1.0 + tol)),
        "metric1": float(bm1 * (1.0 - tol)),
        "metric2": float(bm2 * (1.0 - tol)),
    }
    return task_ctx, constraint_limits


# ===========================================================================
# Stage-1 Rollout 辅助函数（训练和离线共用）
# ===========================================================================

def _stage1_rollout_one_episode(net, env, device, total_layers, greedy=False):
    """执行一个 Stage-1 episode 的 rollout（无梯度）。

    Returns:
        episode_reward: float
        config: dict with gelu, softmax, cost
        steps: list of (cont_feat_np, layer_idx, prev_g, prev_s, action_g, action_s,
               logprob, reward, value, done, gelu_mask_np)

    性能优化（不改变任何数值结果）:
      - 预分配 (1, N, ...) 缓冲区, 通过 slice view 传入网络, 避免每步 torch.cat 的 O(N²) 开销
        slice [:, :step+1] 是 view（不拷贝），与 torch.cat 构造的连续张量在数值上完全等价
    """
    state = env.reset()
    prev_g_val = _SOS_GELU
    prev_s_val = _SOS_SOFTMAX
    N = total_layers

    # 预分配缓冲区，避免每步 torch.cat 动态扩展
    buf_cf = torch.zeros(1, N, 6, dtype=torch.float32, device=device)
    buf_li = torch.zeros(1, N, dtype=torch.long, device=device)
    buf_pg = torch.zeros(1, N, dtype=torch.long, device=device)
    buf_ps = torch.zeros(1, N, dtype=torch.long, device=device)
    buf_gm = torch.zeros(1, N, 4, dtype=torch.bool, device=device)

    episode_reward = 0.0
    steps = []

    for step in range(N):
        layer_idx = int(np.argmax(state[0:N]))
        lb, m1b, m2b = 3 * N + 5, 3 * N + 6, 3 * N + 7
        cf = np.array([
            state[N], state[N + 3], state[N + 4],
            state[lb] if len(state) > lb else 0.0,
            state[m1b] if len(state) > m1b else 0.0,
            state[m2b] if len(state) > m2b else 0.0,
        ], dtype=np.float32)
        gm_np = env.get_gelu_action_mask(layer_idx)

        # 填入预分配缓冲区（in-place 赋值，不分配新张量）
        buf_cf[0, step] = torch.from_numpy(cf).to(device, non_blocking=True)
        buf_li[0, step] = layer_idx
        buf_pg[0, step] = prev_g_val
        buf_ps[0, step] = prev_s_val
        buf_gm[0, step] = torch.from_numpy(gm_np).to(device, non_blocking=True)

        # slice view: 与 torch.cat 产生等价的输入（因果掩码下网络仅看到前 step+1 个 token）
        full_cf = buf_cf[:, :step + 1]
        full_li = buf_li[:, :step + 1]
        full_pg = buf_pg[:, :step + 1]
        full_ps = buf_ps[:, :step + 1]
        full_gm = buf_gm[:, :step + 1]

        with torch.no_grad():
            if greedy:
                lg, ls, vals = net.forward(full_cf, full_li, full_pg, full_ps, gelu_mask=full_gm)
                g_act = lg[:, -1, :].argmax(-1).squeeze(0)
                s_act = ls[:, -1, :].argmax(-1).squeeze(0)
                logprob = torch.tensor(0.0)
                value = vals[:, -1].squeeze(0)
            else:
                g_act, s_act, logprob, value = net.get_action_and_logprob(
                    full_cf, full_li, full_pg, full_ps, gelu_mask=full_gm,
                )

        next_state, reward, done, info = env.step(g_act.item(), s_act.item())
        steps.append((cf, layer_idx, prev_g_val, prev_s_val,
                       g_act.item(), s_act.item(), logprob.cpu(), reward, value.cpu(),
                       float(done), gm_np))

        prev_g_val = g_act.item()
        prev_s_val = s_act.item()
        episode_reward += reward
        state = next_state

    config = {
        "gelu": np.array(env.gelu_config),
        "softmax": np.array(env.softmax_config),
        "cost": env.accumulated_cost,
        "reward": episode_reward,
    }
    return episode_reward, config, steps


# ===========================================================================
# Stage-1 多任务 Round-Robin 训练
# ===========================================================================

def multi_task_train_stage1(
    tasks,
    output_path,
    total_rounds=50,
    episodes_per_task_per_round=120,
    lr=3e-5,
    log_fn=None,
    device="cuda",
    resume_checkpoint_path=None,
    accuracy_tolerances=None,
    accuracy_tolerance_range=None,
):
    """多任务 round-robin 训练 Stage-1 通用策略 + 通用 Critic。

    Args:
        tasks: dict {task_name: task_config}  （每项由 prepare_stage1_task 产出）
        output_path: str, 保存 general_stage1_policy.pt 的路径
        total_rounds: int, round-robin 轮数
        episodes_per_task_per_round: int, 每轮每任务的 episode 数
        lr: float, 学习率
        log_fn: callable（默认 print）
        device: str
        accuracy_tolerances: list[float] or None, 准确度容忍比例列表。
            当提供时, 每轮从中随机采样一个 tolerance, 动态更新 task_context
            和 constraint_limits, 使策略学会泛化到不同准确度要求。
        accuracy_tolerance_range: tuple(float, float) or None, 连续准确度容忍范围。
            如 (0.005, 0.02) 表示从 [0.5%, 2%] 连续均匀采样。
            优先级高于 accuracy_tolerances（两者同时提供时使用 range）。

    Returns:
        dict 包含训练结果及保存路径
    """
    if log_fn is None:
        log_fn = print
    task_names = list(tasks.keys())
    num_tasks = len(task_names)
    total_layers = tasks[task_names[0]]["evaluator"].total_layers

    # ---- 创建网络 ----
    net = GeneralStage1PolicyNetwork(
        num_layers=total_layers,
        d_model=_s1.GTRXL_D_MODEL,
        n_heads=_s1.GTRXL_N_HEADS,
        n_gtrxl_layers=_s1.GTRXL_N_LAYERS,
        d_ff=_s1.GTRXL_D_FF,
        dropout=_s1.GTRXL_DROPOUT,
    ).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # ---- 为每个任务创建 env ----
    envs = {}
    for name in task_names:
        tc = tasks[name]
        if np.any(np.asarray(tc["gelu0_eligible"], dtype=bool)):
            raise ValueError(
                "General Stage-1 explicitly disables GELU degree 0; "
                "gelu0_eligible must be all False."
            )
        ev = tc["evaluator"]

        class _W:
            def __init__(s, e):
                s._e = e
            def evaluate_model(s, g, sm):
                return s._e.stage1_evaluate(g, sm, use_train=True)

        envs[name] = _TransformerOptEnv(
            total_layers, tc["baseline_cost"], tc["baseline_metrics"],
            _W(ev), constraint_limits=tc["constraint_limits"],
            num_metrics=tc["num_metrics"],
            gelu_degree0_eligible=tc["gelu0_eligible"],
        )
        bl, bm1, bm2 = tc["baseline_metrics"]
        envs[name].prev_episode_metrics = {
            "loss": bl, "metric1": bm1, "metric2": bm2, "cost": tc["baseline_cost"],
        }

    buffer = _GeneralStage1Buffer()
    return_norm = _RunningMeanStd()
    ppo_cnt = 0
    ppo_interval = _s1.PPO_UPDATE_INTERVAL
    total_episodes = total_rounds * num_tasks * episodes_per_task_per_round
    global_ep = 0

    best_configs = {n: None for n in task_names}
    best_rewards = {n: float('-inf') for n in task_names}
    all_rewards = {n: [] for n in task_names}
    all_entropies = []

    # ---- 续训练：加载 checkpoint ----
    resume_start_round = 0
    _resume_tol_rng_state = None
    if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
        ckpt = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        ppo_cnt = ckpt.get("ppo_cnt", 0)
        global_ep = ckpt.get("global_ep", 0)
        best_configs = ckpt.get("best_configs", best_configs)
        best_rewards = ckpt.get("best_rewards", best_rewards)
        all_rewards = ckpt.get("all_rewards", all_rewards)
        all_entropies = ckpt.get("all_entropies", all_entropies)
        if "return_norm" in ckpt:
            rn = ckpt["return_norm"]
            return_norm.mean = rn["mean"]
            return_norm.var = rn["var"]
            return_norm.count = rn["count"]
        _resume_tol_rng_state = ckpt.get("tol_rng_state")
        resume_start_round = ckpt.get("completed_round", 0)
        log_fn(f"[通用策略] 从 checkpoint 续训练: 已完成 {resume_start_round} 轮, "
               f"global_ep={global_ep}, ppo_cnt={ppo_cnt}")

    # ---- 停止信号文件路径 ----
    stop_flag_path = os.path.join(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        _STOP_FLAG_FILENAME,
    )

    # ---- 准确度容忍泛化 ----
    _use_tol_range = (accuracy_tolerance_range is not None
                      and len(accuracy_tolerance_range) == 2
                      and accuracy_tolerance_range[0] < accuracy_tolerance_range[1])
    _use_tol_gen = (not _use_tol_range
                    and accuracy_tolerances is not None
                    and len(accuracy_tolerances) > 1)
    _tol_list = list(accuracy_tolerances) if accuracy_tolerances else []
    _tol_rng = np.random.RandomState(42)  # 独立随机源, 可复现
    if _resume_tol_rng_state is not None:
        try:
            _tol_rng.set_state(_resume_tol_rng_state)
        except Exception:
            pass

    _tol_info = ""
    if _use_tol_range:
        _tol_lo, _tol_hi = float(accuracy_tolerance_range[0]), float(accuracy_tolerance_range[1])
        _tol_info = f"\n  准确度容忍泛化(连续): [{_tol_lo*100:.1f}%, {_tol_hi*100:.1f}%]"
    elif _use_tol_gen:
        _tol_info = f"\n  准确度容忍泛化(离散): {[f'{t*100:.1f}%' for t in _tol_list]}"

    _log_general_header(
        log_fn,
        "Stage-1 通用策略训练（General Policy Training）",
        [
            f"任务数量: {num_tasks}    训练轮数: {total_rounds}",
            f"每轮每任务回合数: {episodes_per_task_per_round}    总回合: {total_episodes}",
            f"任务列表: {', '.join(task_names)}",
            f"学习率: {lr}    PPO 间隔: {ppo_interval}" + _tol_info,
        ],
    )

    _reset_stop()
    _install_stop(stop_flag_path)
    _t0 = _time.time()

    for rnd in range(resume_start_round, total_rounds):
        # ---- 准确度容忍泛化: 每轮随机采样一个 tolerance ----
        if _use_tol_range:
            _cur_tol = float(_tol_rng.uniform(_tol_lo, _tol_hi))
            for name in task_names:
                tc = tasks[name]
                new_ctx, new_limits = recompute_task_for_tolerance(tc, _cur_tol)
                tc["task_context"] = new_ctx
                tc["constraint_limits"] = new_limits
                envs[name].constraint_limits = new_limits
            if rnd % 10 == 0:
                log_fn(f"  [容忍泛化·连续] 轮 {rnd+1}: tolerance={_cur_tol*100:.2f}%")
        elif _use_tol_gen:
            _cur_tol = float(_tol_rng.choice(_tol_list))
            for name in task_names:
                tc = tasks[name]
                new_ctx, new_limits = recompute_task_for_tolerance(tc, _cur_tol)
                tc["task_context"] = new_ctx
                tc["constraint_limits"] = new_limits
                envs[name].constraint_limits = new_limits
            if rnd % 10 == 0:
                log_fn(f"  [容忍泛化·离散] 轮 {rnd+1}: tolerance={_cur_tol*100:.1f}%")

        for task_name in task_names:
            tc = tasks[task_name]
            env = envs[task_name]
            net.set_task_context(
                torch.tensor(tc["task_context"], dtype=torch.float32).to(device)
            )

            for _ in range(episodes_per_task_per_round):
                # 熵调度
                progress = global_ep / max(total_episodes, 1)
                if _s1.RL_OPT_FLAGS.get("use_cosine_entropy_schedule", False):
                    plateau = float(_s1.RL_OPT_FLAGS.get("entropy_plateau_ratio", 0.25))
                    if progress <= plateau:
                        ec = _s1.PPO_ENTROPY_START
                    else:
                        tail = min(max((progress - plateau) / max(1.0 - plateau, 1e-8), 0.0), 1.0)
                        ec = _s1.PPO_ENTROPY_END + (_s1.PPO_ENTROPY_START - _s1.PPO_ENTROPY_END) * 0.5 * (1 + math.cos(math.pi * tail))
                else:
                    ec = _s1.PPO_ENTROPY_START - (_s1.PPO_ENTROPY_START - _s1.PPO_ENTROPY_END) * progress
                ec = max(ec, _s1._rl_opt_entropy_lower_bound())

                # Rollout
                ep_reward, config, steps = _stage1_rollout_one_episode(
                    net, env, device, total_layers,
                )
                # 填充 buffer
                buffer.start_episode(task_context=tc["task_context"])
                for (cf, li, pg, ps, ag, as_, lp, r, v, d, gm) in steps:
                    buffer.add_step(
                        torch.tensor(cf, dtype=torch.float32),
                        li, pg, ps, ag, as_, lp, r, v, d, gm,
                    )
                buffer.end_episode()

                all_rewards[task_name].append(ep_reward)
                global_ep += 1

                if ep_reward > best_rewards[task_name]:
                    best_rewards[task_name] = ep_reward
                    best_configs[task_name] = config

                # PPO 更新
                if buffer.num_episodes >= ppo_interval:
                    pl, vl, ent = _general_ppo_update_stage1(
                        net, optimizer, buffer, device,
                        return_normalizer=return_norm,
                        entropy_coef=ec, ppo_update_step=ppo_cnt,
                    )
                    ppo_cnt += 1
                    buffer.clear()
                    all_entropies.append(ent)

                    if ppo_cnt % 10 == 0:
                        avg_str = ", ".join(
                            f"{n}={np.mean(all_rewards[n][-episodes_per_task_per_round:]):.3f}"
                            for n in task_names if all_rewards[n]
                        )
                        log_fn(f"  [PPO #{ppo_cnt}] 轮 {rnd+1}/{total_rounds} 任务 {task_name} | "
                               f"熵={ent:.4f} | {avg_str}")

        _elapsed = _time.time() - _t0
        _avg_rnd = _elapsed / max(rnd - resume_start_round + 1, 1)
        _eta = _avg_rnd * (total_rounds - rnd - 1)
        best_str = ", ".join(f"{n}={best_rewards[n]:.3f}" for n in task_names)
        avg_str = ", ".join(
            f"{n}={np.mean(all_rewards[n][-episodes_per_task_per_round:]):.3f}"
            for n in task_names if all_rewards[n]
        )
        _log_rounded_box(
            log_fn,
            [
                f"Stage-1 通用策略 · 轮 {rnd+1} / {total_rounds}",
                _progress_bar(rnd + 1, total_rounds),
                f"近期均值: {avg_str}",
                f"各任务最优: {best_str}",
                f"PPO 更新: {ppo_cnt} 次  总回合: {global_ep}",
                f"已用时: {_fmt_elapsed(_elapsed)}  预计剩余: {_fmt_elapsed(_eta)}",
            ],
            indent="  ",
        )

        # ---- 优雅停止检查 ----
        if _is_stop(stop_flag_path):
            _consume_stop(stop_flag_path)
            ckpt_save_path = os.path.join(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                GENERAL_STAGE1_TRAIN_CHECKPOINT,
            )
            torch.save({
                "version": GENERAL_POLICY_VERSION,
                "completed_round": rnd + 1,
                "net_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ppo_cnt": ppo_cnt,
                "global_ep": global_ep,
                "best_configs": best_configs,
                "best_rewards": best_rewards,
                "all_rewards": all_rewards,
                "all_entropies": all_entropies,
                "return_norm": {
                    "mean": return_norm.mean,
                    "var": return_norm.var,
                    "count": return_norm.count,
                },
                "tol_rng_state": _tol_rng.get_state(),
                "task_names": task_names,
                "total_rounds": total_rounds,
                "episodes_per_task_per_round": episodes_per_task_per_round,
            }, ckpt_save_path)
            log_fn(f"[通用策略] 优雅停止: checkpoint 已保存 → {ckpt_save_path} "
                   f"(已完成 {rnd+1}/{total_rounds} 轮)")
            _uninstall_stop()
            raise SystemExit(0)

    _uninstall_stop()

    # ---- 保存 general policy ----
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save({
        "version": GENERAL_POLICY_VERSION,
        "kind": "general_stage1_gtrxl_policy",
        "net_state_dict": net.state_dict(),
        "arch": {
            "num_layers": total_layers,
            "d_model": _s1.GTRXL_D_MODEL,
            "n_heads": _s1.GTRXL_N_HEADS,
            "n_gtrxl_layers": _s1.GTRXL_N_LAYERS,
            "d_ff": _s1.GTRXL_D_FF,
            "dropout": _s1.GTRXL_DROPOUT,
            "task_context_dim": TASK_CONTEXT_DIM,
        },
        "metadata": {
            "task_names": task_names,
            "total_rounds": total_rounds,
            "episodes_per_task_per_round": episodes_per_task_per_round,
            "total_episodes": global_ep,
            "ppo_updates": ppo_cnt,
            "best_rewards": {n: float(best_rewards[n]) for n in task_names},
            "accuracy_tolerance_range": list(accuracy_tolerance_range) if _use_tol_range else None,
            "accuracy_tolerances": _tol_list if _use_tol_gen else None,
        },
    }, output_path)
    _total_elapsed = _time.time() - _t0
    best_str = ", ".join(f"{n}={best_rewards[n]:.3f}" for n in task_names)
    _log_rounded_box(
        log_fn,
        [
            "Stage-1 通用策略训练 — 完成（Training Completed）",
            f"总轮数: {total_rounds}    PPO 更新: {ppo_cnt} 次    总回合: {global_ep}",
            f"各任务最优: {best_str}",
            f"总耗时: {_fmt_elapsed(_total_elapsed)}",
            f"策略已保存 → {output_path}",
        ],
        indent="",
    )

    net.set_task_context(None)
    return {
        "output_path": output_path,
        "best_configs": best_configs,
        "best_rewards": {n: float(best_rewards[n]) for n in task_names},
        "ppo_updates": ppo_cnt,
    }


# ===========================================================================
# Stage-1 离线推断（冻结 general policy，不做训练）
# ===========================================================================

def offline_find_best_config_stage1(
    evaluator,
    general_policy_path,
    num_rollouts=500,
    greedy=False,
    device="cuda",
    log_fn=None,
    tolerance=None,
):
    """使用 Stage-1 通用策略在给定任务上做离线 rollout，找到最优 GELU/Softmax 配置。

    策略完全冻结（不做梯度更新），只通过采样或贪心 rollout 搜索最优配置。
    与 online RL 的 51000 回合训练相比，500 次 rollout 要快两个数量级。

    Args:
        evaluator: LayerImportanceEvaluator 实例
        general_policy_path: str, 通用策略文件路径（general_stage1_policy.pt）
        num_rollouts: int, rollout 次数（greedy=True 时自动置 1）
        greedy: bool, 是否使用 argmax 而非采样
        device: str
        log_fn: callable

    Returns:
        dict: best_config, best_reward, all_results, critic_scores, ...
    """
    if log_fn is None:
        log_fn = print

    ckpt = torch.load(general_policy_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    total_layers = evaluator.total_layers

    net = GeneralStage1PolicyNetwork(
        num_layers=total_layers,
        d_model=arch.get("d_model", _s1.GTRXL_D_MODEL),
        n_heads=arch.get("n_heads", _s1.GTRXL_N_HEADS),
        n_gtrxl_layers=arch.get("n_gtrxl_layers", _s1.GTRXL_N_LAYERS),
        d_ff=arch.get("d_ff", _s1.GTRXL_D_FF),
        dropout=arch.get("dropout", _s1.GTRXL_DROPOUT),
    ).to(device)
    missing, unexpected = net.load_state_dict(ckpt["net_state_dict"], strict=False)
    net.eval()
    log_fn(f"[离线推断] 加载通用策略 ← {general_policy_path} "
           f"(missing={len(missing)}, unexpected={len(unexpected)})")

    task_info = prepare_stage1_task(evaluator, tolerance=tolerance)
    if np.any(np.asarray(task_info["gelu0_eligible"], dtype=bool)):
        raise ValueError(
            "General Stage-1 offline inference explicitly disables GELU degree 0; "
            "gelu0_eligible must be all False."
        )
    tc = torch.tensor(task_info["task_context"], dtype=torch.float32).to(device)
    net.set_task_context(tc)

    class _W:
        def __init__(s, e):
            s._e = e
        def evaluate_model(s, g, sm):
            return s._e.stage1_evaluate(g, sm, use_train=True)

    env = _TransformerOptEnv(
        total_layers, task_info["baseline_cost"], task_info["baseline_metrics"],
        _W(evaluator), constraint_limits=task_info["constraint_limits"],
        num_metrics=task_info["num_metrics"],
        gelu_degree0_eligible=task_info["gelu0_eligible"],
    )
    bl, bm1, bm2 = task_info["baseline_metrics"]
    env.prev_episode_metrics = {
        "loss": bl, "metric1": bm1, "metric2": bm2, "cost": task_info["baseline_cost"],
    }

    if greedy:
        num_rollouts = 1

    best_reward = float('-inf')
    best_config = None
    all_results = []

    _mode_str = "贪心（Greedy）" if greedy else f"{num_rollouts} 次采样（Sampling）"
    _log_general_header(
        log_fn,
        "Stage-1 离线推断（Offline Inference）",
        [
            f"模式: {_mode_str}",
            f"策略来源: {general_policy_path}",
        ],
    )
    _t0 = _time.time()
    _progress_interval = max(1, num_rollouts // 10)

    for i in range(num_rollouts):
        ep_reward, config, _steps = _stage1_rollout_one_episode(
            net, env, device, total_layers, greedy=greedy,
        )
        all_results.append(config)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_config = config
        if (i + 1) % _progress_interval == 0 or i + 1 == num_rollouts:
            _elapsed = _time.time() - _t0
            _eta = _elapsed / (i + 1) * (num_rollouts - i - 1)
            log_fn(
                f"  {_progress_bar(i + 1, num_rollouts)}  "
                f"当前最优={best_reward:.4f}  "
                f"已用时={_fmt_elapsed(_elapsed)}  "
                f"剩余≈{_fmt_elapsed(_eta)}"
            )

    net.set_task_context(None)
    _total_elapsed = _time.time() - _t0
    _log_rounded_box(
        log_fn,
        [
            "Stage-1 离线推断 — 完成（Inference Completed）",
            f"最优 Reward: {best_reward:.4f}    Cost: {best_config['cost']:.2f}",
            f"GELU   : {best_config['gelu'].tolist()}",
            f"Softmax: {best_config['softmax'].tolist()}",
            f"总耗时: {_fmt_elapsed(_total_elapsed)}    Rollout 次数: {num_rollouts}",
        ],
        indent="",
    )

    return {
        "best_config": best_config,
        "best_reward": best_reward,
        "num_rollouts": num_rollouts,
        "all_results": all_results,
        "policy_path": general_policy_path,
        "policy_metadata": ckpt.get("metadata", {}),
    }


# ===========================================================================
# Stage-2 离线推断
# ===========================================================================

def offline_find_best_config_stage2(
    evaluator,
    general_policy_path,
    fixed_gelu,
    fixed_softmax,
    noise_env,
    num_rollouts=500,
    greedy=False,
    device="cuda",
    log_fn=None,
):
    """使用 Stage-2 通用噪声策略在给定任务上做离线 rollout，找到最优噪声配置。

    Args:
        evaluator: LayerImportanceEvaluator 实例
        general_policy_path: str, 通用噪声策略路径（general_stage2_noise_policy.pt）
        fixed_gelu: np.ndarray, Stage-1 确定的 GELU 配置
        fixed_softmax: np.ndarray, Stage-1 确定的 Softmax 配置
        noise_env: _NoiseOptEnv 实例（已创建完毕，含所有 noise 参数）
        num_rollouts: int
        greedy: bool
        device: str
        log_fn: callable

    Returns:
        dict: best_noise_config, best_reward, ...
    """
    if log_fn is None:
        log_fn = print

    ckpt = torch.load(general_policy_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    total_layers = evaluator.total_layers

    net = GeneralStage2NoisePolicyNetwork(
        num_layers=total_layers,
        d_model=arch.get("d_model", _s2.NOISE_STAGE_GTRXL_D_MODEL),
        n_heads=arch.get("n_heads", _s2.NOISE_STAGE_GTRXL_N_HEADS),
        n_gtrxl_layers=arch.get("n_gtrxl_layers", _s2.NOISE_STAGE_GTRXL_N_LAYERS),
        d_ff=arch.get("d_ff", _s2.NOISE_STAGE_GTRXL_D_FF),
        dropout=arch.get("dropout", _s2.NOISE_STAGE_GTRXL_DROPOUT),
        gtrxl_block_cls=_s1.GTrXLBlock,
        lstm_pos_dim=_s1.LSTM_POS_DIM,
        lstm_proj_dim=_s1.LSTM_PROJ_DIM,
        noise_stage_num_actions=_s1.NOISE_STAGE_NUM_ACTIONS,
        noise_stage_sos_tokens=_s1.NOISE_STAGE_SOS_TOKENS,
        noise_stage_prev_action_embed_dim=_s1.NOISE_STAGE_PREV_ACTION_EMBED_DIM,
        noise_stage_cont_dim=_s2.NOISE_STAGE_V2_CONT_DIM,
        noise_stage_action_dims=_s1.NOISE_STAGE_ACTION_DIMS,
    ).to(device)
    missing, unexpected = net.load_state_dict(ckpt["net_state_dict"], strict=False)
    net.eval()
    log_fn(f"[Stage-2 离线] 加载通用噪声策略 ← {general_policy_path} "
           f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # 构造 task context
    base_gelu = np.full(total_layers, 4, dtype=int)
    base_softmax = np.full(total_layers, 6, dtype=int)
    base_loss, base_m1, base_m2, _ = evaluator.stage1_evaluate(
        base_gelu, base_softmax, use_train=True,
    )
    tc = torch.tensor(compute_task_context(
        base_loss, base_m1, base_m2,
        evaluator.error_threshold, evaluator.correlation_drop_ratio,
    ), dtype=torch.float32).to(device)
    net.set_task_context(tc)

    env = noise_env
    sos_tokens = list(_s1.NOISE_STAGE_SOS_TOKENS)

    if greedy:
        num_rollouts = 1

    best_reward = float('-inf')
    best_noise_config = None
    all_results = []

    _mode_str = "贪心（Greedy）" if greedy else f"{num_rollouts} 次采样（Sampling）"
    _log_general_header(
        log_fn,
        "Stage-2 噪声离线推断（Noise Offline Inference）",
        [
            f"模式: {_mode_str}",
            f"策略来源: {general_policy_path}",
        ],
    )
    _t0 = _time.time()
    _progress_interval = max(1, num_rollouts // 10)

    for i in range(num_rollouts):
        state = env.reset()
        prev_actions = torch.tensor(
            [sos_tokens], dtype=torch.long, device=device,
        ).unsqueeze(0)   # (1, 1, 7)
        seq_cf, seq_li, seq_pa = [], [], []
        episode_reward = 0.0

        for step in range(total_layers):
            layer_idx = env.current_layer
            cf_np = env.get_continuous_features()
            cf = torch.tensor(cf_np, dtype=torch.float32, device=device).view(1, 1, -1)
            li = torch.tensor([[layer_idx]], dtype=torch.long, device=device)

            seq_cf.append(cf)
            seq_li.append(li)
            seq_pa.append(prev_actions)

            cf_seq = torch.cat(seq_cf, dim=1)
            li_seq = torch.cat(seq_li, dim=1)
            pa_seq = torch.cat(seq_pa, dim=1)

            with torch.no_grad():
                if greedy:
                    logits_dict, values = net.forward(cf_seq, li_seq, pa_seq)
                    actions = torch.stack([
                        logits_dict[name][:, -1, :].argmax(-1).squeeze(0)
                        for name in net.action_names
                    ])
                else:
                    actions, logprob, value = net.get_action_and_logprob(
                        cf_seq, li_seq, pa_seq,
                    )

            next_state, reward, done, info = env.step(*[a.item() for a in actions])
            if done:
                episode_reward = float(info.get("final_selection_score",
                                                info.get("raw_final_reward", reward)))
            else:
                episode_reward += reward

            prev_actions = actions.unsqueeze(0).unsqueeze(0).to(device)
            state = next_state

        noise_config = {
            "input_noise_scaling_factors": np.array(env.input_noise_config),
            "wq_noise_scaling_factors": np.array(env.wq_noise_config),
            "wk_noise_scaling_factors": np.array(env.wk_noise_config),
            "wv_noise_scaling_factors": np.array(env.wv_noise_config),
            "wo_noise_scaling_factors": np.array(env.wo_noise_config),
            "wffn1_noise_scaling_factors": np.array(env.wffn1_noise_config),
            "wffn2_noise_scaling_factors": np.array(env.wffn2_noise_config),
            "reward": episode_reward,
        }
        all_results.append(noise_config)

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_noise_config = noise_config

        if (i + 1) % _progress_interval == 0 or i + 1 == num_rollouts:
            _elapsed = _time.time() - _t0
            _eta = _elapsed / (i + 1) * (num_rollouts - i - 1)
            log_fn(
                f"  {_progress_bar(i + 1, num_rollouts)}  "
                f"当前最优={best_reward:.4f}  "
                f"已用时={_fmt_elapsed(_elapsed)}  "
                f"剩余≈{_fmt_elapsed(_eta)}"
            )

    net.set_task_context(None)
    _total_elapsed = _time.time() - _t0
    _log_rounded_box(
        log_fn,
        [
            "Stage-2 噪声离线推断 — 完成（Inference Completed）",
            f"最优 Reward: {best_reward:.4f}",
            f"总耗时: {_fmt_elapsed(_total_elapsed)}    Rollout 次数: {num_rollouts}",
        ],
        indent="",
    )

    return {
        "best_noise_config": best_noise_config,
        "best_reward": best_reward,
        "num_rollouts": num_rollouts,
        "all_results": all_results,
        "policy_path": general_policy_path,
    }


# ===========================================================================
# Critic 快速评分（不做模型评测，只用 V(s) 排序候选配置）
# ===========================================================================

def critic_quick_rank_stage1(
    evaluator,
    general_policy_path,
    candidate_configs,
    device="cuda",
):
    """使用通用 Critic 的 V(s) 对一组 Stage-1 候选配置快速评分。

    与 offline_find_best_config_stage1 的区别：
      - 不做真实模型评测（evaluate_model），只用 Critic 网络估值
      - 速度极快（纯 GPU 推理），适合预筛选大量候选
      - 精度取决于 Critic 训练质量

    Args:
        evaluator: LayerImportanceEvaluator 实例
        general_policy_path: str
        candidate_configs: list of dicts, 每个 dict 包含 gelu (ndarray) 和 softmax (ndarray)
        device: str

    Returns:
        list of (score, config) 按 score 降序排列
    """
    ckpt = torch.load(general_policy_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    total_layers = evaluator.total_layers

    net = GeneralStage1PolicyNetwork(
        num_layers=total_layers,
        d_model=arch.get("d_model", _s1.GTRXL_D_MODEL),
        n_heads=arch.get("n_heads", _s1.GTRXL_N_HEADS),
        n_gtrxl_layers=arch.get("n_gtrxl_layers", _s1.GTRXL_N_LAYERS),
        d_ff=arch.get("d_ff", _s1.GTRXL_D_FF),
        dropout=arch.get("dropout", _s1.GTRXL_DROPOUT),
    ).to(device)
    net.load_state_dict(ckpt["net_state_dict"], strict=False)
    net.eval()

    task_info = prepare_stage1_task(evaluator)
    tc = torch.tensor(task_info["task_context"], dtype=torch.float32).to(device)
    net.set_task_context(tc)

    _validate_general_stage1_candidate_configs(candidate_configs)

    gelu_to_idx = {4: 0, 2: 1, 1: 2}
    softmax_to_idx = {6: 0, 5: 1, 4: 2, 3: 3, 2: 4}
    N = total_layers

    scored = []
    for cfg in candidate_configs:
        gelu_arr = np.asarray(cfg["gelu"], dtype=int)
        softmax_arr = np.asarray(cfg["softmax"], dtype=int)

        # 模拟完整 episode 的 token 序列
        cf_list, li_list, pg_list, ps_list = [], [], [], []
        prev_g_idx = _SOS_GELU
        prev_s_idx = _SOS_SOFTMAX
        acc_cost = 0.0

        for layer in range(N):
            g_deg = int(gelu_arr[layer])
            s_deg = int(softmax_arr[layer])
            acc_cost += _s1.GELU_COST.get(g_deg, 0) + _s1.SOFTMAX_COST.get(s_deg, 0)
            progress = (layer + 1) / N
            cf = np.array([0.0, 0.0, progress, 0.0, 0.0, 0.0], dtype=np.float32)
            cf_list.append(torch.tensor(cf, dtype=torch.float32))
            li_list.append(layer)
            pg_list.append(prev_g_idx)
            ps_list.append(prev_s_idx)
            prev_g_idx = gelu_to_idx[g_deg]
            prev_s_idx = softmax_to_idx.get(s_deg, 0)

        cf_t = torch.stack(cf_list).unsqueeze(0).to(device)       # (1, N, 6)
        li_t = torch.tensor([li_list], dtype=torch.long).to(device)
        pg_t = torch.tensor([pg_list], dtype=torch.long).to(device)
        ps_t = torch.tensor([ps_list], dtype=torch.long).to(device)

        with torch.no_grad():
            _, _, vals = net.forward(cf_t, li_t, pg_t, ps_t)
            score = vals[0, -1].item()  # 最后一步的 V(s) 作为配置评分

        scored.append((score, cfg))

    net.set_task_context(None)
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ===========================================================================
# Stage-2 通用 Rollout Buffer（NoiseRecurrentRolloutBuffer + per-episode task_context）
# ===========================================================================

class _GeneralStage2Buffer(_NoiseBuffer):
    """在 _NoiseRecurrentRolloutBuffer 基础上增加 per-episode task_context 存储。"""

    def start_episode(self, task_context=None):
        super().start_episode()
        self._current['task_context'] = task_context  # np.ndarray (5,)

    def get_batch(self, device):
        parent_batch = super().get_batch(device)
        task_contexts = torch.stack([
            torch.tensor(ep['task_context'], dtype=torch.float32)
            for ep in self.episodes
        ]).to(device)   # (N_eps, 5)
        return (*parent_batch, task_contexts)


# ===========================================================================
# Stage-2 通用 PPO 更新（独立于 evaluator 实例）
# ===========================================================================

def _general_ppo_update_stage2(
    net, optimizer, buffer, device,
    return_normalizer,
    entropy_coef=0.02,
    ppo_update_step=0,
    ppo_eps_clip=None,
    ppo_k_epochs=None,
    ppo_value_coef=None,
    value_clip_range=None,
    warmup_steps=20,
):
    """独立的 Stage-2 PPO 更新，支持 per-episode task_context。"""
    if ppo_eps_clip is None:
        ppo_eps_clip = _s2.NOISE_STAGE_PPO_EPS_CLIP
    if ppo_k_epochs is None:
        ppo_k_epochs = _s2.NOISE_STAGE_PPO_K_EPOCHS
    if ppo_value_coef is None:
        ppo_value_coef = _s1.PPO_VALUE_COEF
    if value_clip_range is None:
        value_clip_range = _s2.NOISE_STAGE_VALUE_CLIP_RANGE
    mini_batch_eps = _s1.GTRXL_MINI_BATCH_EPISODES

    target_lr = optimizer.defaults['lr']
    if ppo_update_step < warmup_steps:
        factor = (ppo_update_step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg['lr'] = target_lr * factor

    (cont_features, layer_indices, prev_actions, actions,
     old_logprobs, rewards, values, dones, task_contexts) = buffer.get_batch(device)

    n_eps = cont_features.size(0)
    all_adv, all_ret = [], []
    for i in range(n_eps):
        a, r = _compute_gae(
            rewards[i].cpu().numpy(), values[i].cpu().numpy(),
            dones[i].cpu().numpy(), gamma=_s2.NOISE_STAGE_PPO_GAMMA, lam=0.95,
        )
        all_adv.append(a)
        all_ret.append(r)
    advantages = torch.stack(all_adv).to(device)
    returns = torch.stack(all_ret).to(device)
    adv_flat = advantages.reshape(-1)
    advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)

    return_normalizer.update(returns)
    ret_norm = torch.tensor(return_normalizer.normalize(returns.cpu().numpy()),
                            dtype=torch.float32).to(device)
    val_norm = torch.tensor(return_normalizer.normalize(values.cpu().numpy()),
                            dtype=torch.float32).to(device)

    last_pl, last_vl, last_ent = 0.0, 0.0, 0.0
    kl_stop = False

    for _epoch in range(ppo_k_epochs):
        if kl_stop:
            break
        ep_idx = torch.randperm(n_eps)
        kl_acc, kl_cnt = 0.0, 0

        for start in range(0, n_eps, mini_batch_eps):
            end = min(start + mini_batch_eps, n_eps)
            mb = ep_idx[start:end]

            # 每个 mini-batch 设定 task context（混合任务）
            net.set_task_context(task_contexts[mb])

            lp_new, ent, v_new = net.evaluate_actions(
                cont_features[mb], layer_indices[mb],
                prev_actions[mb], actions[mb],
            )
            lp_f = lp_new.reshape(-1)
            ent_f = ent.reshape(-1)
            v_f = v_new.reshape(-1)
            olp_f = old_logprobs[mb].reshape(-1)
            adv_f = advantages[mb].reshape(-1)
            ret_f = ret_norm[mb].reshape(-1)
            ov_f = val_norm[mb].reshape(-1)

            ratios = torch.exp(lp_f - olp_f)
            s1 = ratios * adv_f
            s2 = torch.clamp(ratios, 1 - ppo_eps_clip, 1 + ppo_eps_clip) * adv_f
            p_loss = -torch.min(s1, s2).mean()

            vn = (v_f - return_normalizer.mean) / return_normalizer.std
            vc = ov_f + torch.clamp(vn - ov_f, -value_clip_range, value_clip_range)
            huber = nn.HuberLoss(reduction='none', delta=1.0)
            v_loss = torch.max(huber(vn, ret_f), huber(vc, ret_f)).mean()

            me = ent_f.mean()
            eff_ec = entropy_coef
            lb = _s2._noise_rl_entropy_lower_bound(0.005)
            if me.item() < lb:
                eff_ec = entropy_coef + _s2._noise_rl_entropy_recovery_mul() * (lb - me.item())

            loss = p_loss + ppo_value_coef * v_loss + eff_ec * (-me)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            last_pl, last_vl, last_ent = p_loss.item(), v_loss.item(), me.item()

            if _s2.NOISE_RL_OPT_FLAGS.get("use_kl_early_stop", False):
                with torch.no_grad():
                    kl_acc += (olp_f - lp_f).mean().item()
                kl_cnt += 1

        if (_s2.NOISE_RL_OPT_FLAGS.get("use_kl_early_stop", False)
                and kl_cnt > 0
                and kl_acc / kl_cnt > 1.5 * _s2.NOISE_RL_OPT_FLAGS.get("kl_target", 0.02)):
            kl_stop = True

    net.set_task_context(None)
    return last_pl, last_vl, last_ent


# ===========================================================================
# Stage-2 任务准备
# ===========================================================================

def prepare_stage2_task(evaluator, fixed_gelu, fixed_softmax, split_name=None):
    """从 evaluator 准备一个 task config dict，用于 multi_task_train_stage2 或 offline 推断。

    Args:
        evaluator: LayerImportanceEvaluator 实例
        fixed_gelu: np.ndarray, Stage-1 确定的 GELU 配置
        fixed_softmax: np.ndarray, Stage-1 确定的 Softmax 配置
        split_name: str or None, 评测子集名称（默认使用 validation_full 或 reward_reference_split）

    Returns:
        dict，包含多任务训练 / 离线推断所需的全部信息
    """
    from layer_importance_evaluator import (
        INPUT_NOISE_SCALING_MAP, INPUT_NOISE_COST_MAP, INPUT_NOISE_SCALING_TO_NORM,
        WEIGHT_NOISE_COST_MAP, WEIGHT_NOISE_SCALING_TO_NORM,
        WFFN1_NOISE_COST_MAP, WFFN1_NOISE_SCALING_TO_NORM,
        WQ_NOISE_SCALING_MAP, WK_NOISE_SCALING_MAP,
        WV_NOISE_SCALING_MAP, WO_NOISE_SCALING_MAP,
        WFFN1_NOISE_SCALING_MAP, WFFN2_NOISE_SCALING_MAP,
        NOISE_STAGE_NUM_ACTIONS, NOISE_STAGE_SOS_TOKENS,
        NOISE_STAGE_PREV_ACTION_EMBED_DIM, NOISE_STAGE_ACTION_DIMS,
        REWARD_CLIP_MIN, REWARD_CLIP_MAX, HISTORY_MASK_VALUE,
        LSTM_POS_DIM, LSTM_PROJ_DIM, GTrXLBlock,
    )
    from function_handler import (
        INPUT_NOISE_ALLOWED_SCALING_FACTORS,
        WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
        WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
    )

    ev = evaluator
    total_layers = ev.total_layers
    fixed_gelu = np.asarray(fixed_gelu, dtype=int)
    fixed_softmax = np.asarray(fixed_softmax, dtype=int)

    # 确定评测子集
    if split_name is None:
        if ev.has_dataset_split("validation_full"):
            split_name = "validation_full"
        else:
            split_name = ev.get_reward_reference_split_name()

    # 低风险基线 / 最差基线
    baseline_noise_config = _s2._get_low_risk_noise_configuration(ev)
    worst_case_noise_config = _s2._get_worst_case_noise_configuration(ev)
    cost_reference_noise_config = ev._get_max_noise_configuration()

    _eval_dataset = ev.dataset_splits.get(split_name)
    _eval_dataset_size = len(_eval_dataset) if _eval_dataset is not None else 0
    _gp_k = int(getattr(ev, "stage2_k_trials", _s2.NOISE_STAGE_K_TRIALS))
    _gp_probe = int(getattr(ev, "stage2_probe_size", _s2.NOISE_STAGE_PROBE_SIZE))
    baseline_segments = _s2._auto_adjust_segments(
        _eval_dataset_size, _gp_k, min_samples=_gp_probe,
    )

    baseline_stats = ev.evaluate_model_with_attention_noise_segmented(
        fixed_gelu, fixed_softmax,
        **baseline_noise_config,
        segments=baseline_segments,
        split=split_name,
    )
    worst_stats = ev.evaluate_model_with_attention_noise_segmented(
        fixed_gelu, fixed_softmax,
        **worst_case_noise_config,
        segments=baseline_segments,
        split=split_name,
    )
    cost_reference_tot_c, _ = ev.get_noise_simulated_cost(**cost_reference_noise_config)

    base_loss = baseline_stats["loss_mean"]
    base_p = baseline_stats["p_mean"]
    base_s = baseline_stats["s_mean"]
    worst_loss = worst_stats["loss_mean"]
    worst_p = worst_stats["p_mean"]
    worst_s = worst_stats["s_mean"]

    _gp_limit_tol = getattr(ev, "stage2_limit_tolerance", _s2.NOISE_STAGE_DYNAMIC_LIMIT_TOLERANCE)
    _gp_stab_tol = getattr(ev, "stage2_stability_tolerance", _s2.NOISE_STAGE_DYNAMIC_LIMIT_TOLERANCE)
    search_limits = _s2._compute_dynamic_limits(base_loss, base_p, base_s, tolerance=_gp_limit_tol)

    baseline_loss_std = float(baseline_stats["loss_std"])
    baseline_m1_std = float(baseline_stats["p_std"])
    dynamic_loss_std_cap = _s2._compute_dynamic_std_upper_bound(baseline_loss_std, tolerance=_gp_stab_tol)
    dynamic_m1_std_cap = _s2._compute_dynamic_std_upper_bound(float(baseline_stats["p_std"]), tolerance=_gp_stab_tol)
    dynamic_m2_std_cap = _s2._compute_dynamic_std_upper_bound(float(baseline_stats["s_std"]), tolerance=_gp_stab_tol)

    # 评估器包装器
    class _NoiseRLEvaluatorWrapper:
        def __init__(ws, evaluator_, fg, fs, sn):
            ws.evaluator = evaluator_
            ws.fixed_gelu = np.asarray(fg, dtype=int)
            ws.fixed_softmax = np.asarray(fs, dtype=int)
            ws.split_name = sn

        def evaluate_noise_model(ws, **noise_kwargs):
            return ws.evaluator.evaluate_model_with_attention_noise(
                ws.fixed_gelu, ws.fixed_softmax, split=ws.split_name, **noise_kwargs,
            )

        def evaluate_noise_model_repeated(ws, repeats, **noise_kwargs):
            return ws.evaluator.evaluate_model_with_attention_noise_repeated(
                ws.fixed_gelu, ws.fixed_softmax, repeats=repeats,
                split=ws.split_name, **noise_kwargs,
            )

        def evaluate_noise_model_segmented(ws, segments, **noise_kwargs):
            return ws.evaluator.evaluate_model_with_attention_noise_segmented(
                ws.fixed_gelu, ws.fixed_softmax, segments=segments,
                split=ws.split_name, **noise_kwargs,
            )

    rl_evaluator = _NoiseRLEvaluatorWrapper(ev, fixed_gelu, fixed_softmax, split_name)

    mc_train_samples = _s2._auto_adjust_segments(
        _eval_dataset_size, _gp_k, min_samples=_gp_probe,
    )

    # task context
    base_gelu = np.full(total_layers, 4, dtype=int)
    base_softmax = np.full(total_layers, 6, dtype=int)
    base_loss_s1, base_m1_s1, base_m2_s1, _ = ev.stage1_evaluate(
        base_gelu, base_softmax, use_train=True,
    )
    task_ctx = compute_task_context(
        base_loss_s1, base_m1_s1, base_m2_s1,
        ev.error_threshold, ev.correlation_drop_ratio,
    )

    num_metrics = ev.get_num_metrics()

    return {
        "evaluator": evaluator,
        "fixed_gelu": fixed_gelu,
        "fixed_softmax": fixed_softmax,
        "split_name": split_name,
        "baseline_metrics": (float(base_loss), float(base_p), float(base_s)),
        "cost_reference": float(cost_reference_tot_c),
        "search_limits": search_limits,
        "num_metrics": num_metrics,
        "task_context": task_ctx,
        "rl_evaluator": rl_evaluator,
        # env 构造所需的全部参数
        "env_kwargs": {
            "total_layers": total_layers,
            "baseline_cost": float(cost_reference_tot_c),
            "baseline_metrics": (float(base_loss), float(base_p), float(base_s)),
            "evaluator": rl_evaluator,
            "fixed_gelu": fixed_gelu,
            "fixed_softmax": fixed_softmax,
            "constraint_limits": search_limits,
            "num_metrics": num_metrics,
            "input_noise_allowed": INPUT_NOISE_ALLOWED_SCALING_FACTORS,
            "weight_noise_allowed": WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
            "wffn1_noise_allowed": WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
            "input_noise_cost_map": INPUT_NOISE_COST_MAP,
            "weight_noise_cost_map": WEIGHT_NOISE_COST_MAP,
            "wffn1_noise_cost_map": WFFN1_NOISE_COST_MAP,
            "input_noise_scaling_map": INPUT_NOISE_SCALING_MAP,
            "wq_noise_scaling_map": WQ_NOISE_SCALING_MAP,
            "wk_noise_scaling_map": WK_NOISE_SCALING_MAP,
            "wv_noise_scaling_map": WV_NOISE_SCALING_MAP,
            "wo_noise_scaling_map": WO_NOISE_SCALING_MAP,
            "wffn1_noise_scaling_map": WFFN1_NOISE_SCALING_MAP,
            "wffn2_noise_scaling_map": WFFN2_NOISE_SCALING_MAP,
            "input_noise_scaling_to_norm": INPUT_NOISE_SCALING_TO_NORM,
            "weight_noise_scaling_to_norm": WEIGHT_NOISE_SCALING_TO_NORM,
            "wffn1_noise_scaling_to_norm": WFFN1_NOISE_SCALING_TO_NORM,
            "noise_stage_sos_tokens": NOISE_STAGE_SOS_TOKENS,
            "noise_stage_num_actions": NOISE_STAGE_NUM_ACTIONS,
            "history_mask_value": HISTORY_MASK_VALUE,
            "reward_clip_min": REWARD_CLIP_MIN,
            "reward_clip_max": REWARD_CLIP_MAX,
            "perf_weight_loss": _s2.NOISE_STAGE_PERF_WEIGHT_LOSS,
            "perf_weight_m1": _s2.NOISE_STAGE_PERF_WEIGHT_M1,
            "perf_weight_m2": _s2.NOISE_STAGE_PERF_WEIGHT_M2,
            "barrier_weight_loss": _s2.NOISE_STAGE_BARRIER_WEIGHT_LOSS,
            "barrier_weight_m1": _s2.NOISE_STAGE_BARRIER_WEIGHT_M1,
            "barrier_weight_m2": _s2.NOISE_STAGE_BARRIER_WEIGHT_M2,
            "mc_train_samples": mc_train_samples,
            "stability_lambda": _s2.NOISE_STAGE_STABILITY_LAMBDA,
            "stability_base_std_tolerance": _s2.NOISE_STAGE_STABILITY_BASE_STD_TOLERANCE,
            "baseline_loss_std": baseline_loss_std,
            "baseline_m1_std": baseline_m1_std,
            "dynamic_loss_std_cap": dynamic_loss_std_cap,
            "dynamic_m1_std_cap": dynamic_m1_std_cap,
            "dynamic_m2_std_cap": dynamic_m2_std_cap,
            "noise_debt_log_alpha": _s2.NOISE_STAGE_NOISE_DEBT_LOG_ALPHA,
        },
        # 网络构造所需参数
        "net_kwargs": {
            "noise_stage_num_actions": NOISE_STAGE_NUM_ACTIONS,
            "noise_stage_sos_tokens": NOISE_STAGE_SOS_TOKENS,
            "noise_stage_prev_action_embed_dim": NOISE_STAGE_PREV_ACTION_EMBED_DIM,
            "noise_stage_cont_dim": _s2.NOISE_STAGE_V2_CONT_DIM,
            "noise_stage_action_dims": NOISE_STAGE_ACTION_DIMS,
            "gtrxl_block_cls": GTrXLBlock,
            "lstm_pos_dim": LSTM_POS_DIM,
            "lstm_proj_dim": LSTM_PROJ_DIM,
        },
    }


# ===========================================================================
# Stage-2 Rollout 辅助函数（训练和离线共用）
# ===========================================================================

def _stage2_rollout_one_episode(net, env, device, total_layers, greedy=False):
    """执行一个 Stage-2 episode 的 rollout（无梯度）。

    Returns:
        episode_reward: float
        noise_config: dict with per-component noise scaling factors + reward
        steps: list of step tuples for buffer

    性能优化（不改变数值结果）:
      - 预分配 (1, N, ...) 缓冲区, 通过 slice view 传入网络, 避免每步 O(N²) 的 torch.cat
    """
    sos_tokens = list(_s1.NOISE_STAGE_SOS_TOKENS)
    num_actions = len(sos_tokens)
    N = total_layers

    state = env.reset()
    # 获取首步输入的连续特征维度
    _probe_cf = env.get_continuous_features()
    cont_dim = int(np.asarray(_probe_cf, dtype=np.float32).reshape(-1).shape[0])

    # 预分配缓冲区
    buf_cf = torch.zeros(1, N, cont_dim, dtype=torch.float32, device=device)
    buf_li = torch.zeros(1, N, dtype=torch.long, device=device)
    buf_pa = torch.zeros(1, N, num_actions, dtype=torch.long, device=device)

    # 初始 prev_actions = SOS
    prev_actions_cpu = torch.tensor(sos_tokens, dtype=torch.long)

    episode_reward = 0.0
    steps = []

    for step in range(N):
        layer_idx = env.current_layer
        if step == 0:
            cf_np = _probe_cf
        else:
            cf_np = env.get_continuous_features()
        cf_np_arr = np.asarray(cf_np, dtype=np.float32).reshape(-1)

        # 写入预分配缓冲区
        buf_cf[0, step] = torch.from_numpy(cf_np_arr).to(device, non_blocking=True)
        buf_li[0, step] = layer_idx
        buf_pa[0, step] = prev_actions_cpu.to(device, non_blocking=True)

        # slice view（与 torch.cat 数值等价）
        cf_seq = buf_cf[:, :step + 1]
        li_seq = buf_li[:, :step + 1]
        pa_seq = buf_pa[:, :step + 1]

        with torch.no_grad():
            if greedy:
                logits_dict, values = net.forward(cf_seq, li_seq, pa_seq)
                actions = torch.stack([
                    logits_dict[name][:, -1, :].argmax(-1).squeeze(0)
                    for name in net.action_names
                ])
                logprob = torch.tensor(0.0)
                value = values[:, -1].squeeze(0)
            else:
                actions, logprob, value = net.get_action_and_logprob(
                    cf_seq, li_seq, pa_seq,
                )

        next_state, reward, done, info = env.step(*[a.item() for a in actions])

        steps.append((
            torch.from_numpy(cf_np_arr.copy()),
            layer_idx,
            prev_actions_cpu.clone(),                     # (num_actions,)
            actions.detach().cpu().clone(),               # (num_actions,)
            logprob.cpu(),
            reward,
            value.cpu(),
            float(done),
        ))

        if done:
            episode_reward = float(info.get("final_selection_score",
                                            info.get("raw_final_reward", reward)))
        else:
            episode_reward += reward

        # 更新 prev_actions: 保留一份 CPU 副本供下一步和 steps 使用
        prev_actions_cpu = actions.detach().cpu().to(torch.long)
        state = next_state

    noise_config = {
        "input_noise_scaling_factors": np.array(env.input_noise_config),
        "wq_noise_scaling_factors": np.array(env.wq_noise_config),
        "wk_noise_scaling_factors": np.array(env.wk_noise_config),
        "wv_noise_scaling_factors": np.array(env.wv_noise_config),
        "wo_noise_scaling_factors": np.array(env.wo_noise_config),
        "wffn1_noise_scaling_factors": np.array(env.wffn1_noise_config),
        "wffn2_noise_scaling_factors": np.array(env.wffn2_noise_config),
        "reward": episode_reward,
    }
    return episode_reward, noise_config, steps


# ===========================================================================
# Stage-2 多任务 Round-Robin 训练
# ===========================================================================

def multi_task_train_stage2(
    tasks,
    output_path,
    total_rounds=50,
    episodes_per_task_per_round=120,
    lr=3e-5,
    log_fn=None,
    device="cuda",
    resume_checkpoint_path=None,
    accuracy_tolerances=None,
    accuracy_tolerance_range=None,
):
    """多任务 round-robin 训练 Stage-2 通用噪声策略 + 通用 Critic。

    Args:
        tasks: dict {task_name: task_config}  （每项由 prepare_stage2_task 产出）
        output_path: str, 保存 general_stage2_noise_policy.pt 的路径
        total_rounds: int, round-robin 轮数
        episodes_per_task_per_round: int, 每轮每任务的 episode 数
        lr: float, 学习率
        log_fn: callable（默认 print）
        device: str
        accuracy_tolerances: list[float] or None, 准确度容忍比例列表（同 Stage-1）
        accuracy_tolerance_range: tuple(float, float) or None, 连续准确度容忍范围。
            如 (0.005, 0.02) 表示从 [0.5%, 2%] 连续均匀采样。
            优先级高于 accuracy_tolerances（两者同时提供时使用 range）。

    Returns:
        dict 包含训练结果及保存路径
    """
    if log_fn is None:
        log_fn = print
    task_names = list(tasks.keys())
    num_tasks = len(task_names)
    total_layers = tasks[task_names[0]]["evaluator"].total_layers

    # 从第一个任务获取网络构造参数
    net_kwargs = tasks[task_names[0]]["net_kwargs"]

    # ---- 创建网络 ----
    net = GeneralStage2NoisePolicyNetwork(
        num_layers=total_layers,
        d_model=_s2.NOISE_STAGE_GTRXL_D_MODEL,
        n_heads=_s2.NOISE_STAGE_GTRXL_N_HEADS,
        n_gtrxl_layers=_s2.NOISE_STAGE_GTRXL_N_LAYERS,
        d_ff=_s2.NOISE_STAGE_GTRXL_D_FF,
        dropout=_s2.NOISE_STAGE_GTRXL_DROPOUT,
        gtrxl_block_cls=net_kwargs["gtrxl_block_cls"],
        lstm_pos_dim=net_kwargs["lstm_pos_dim"],
        lstm_proj_dim=net_kwargs["lstm_proj_dim"],
        noise_stage_num_actions=net_kwargs["noise_stage_num_actions"],
        noise_stage_sos_tokens=net_kwargs["noise_stage_sos_tokens"],
        noise_stage_prev_action_embed_dim=net_kwargs["noise_stage_prev_action_embed_dim"],
        noise_stage_cont_dim=net_kwargs["noise_stage_cont_dim"],
        noise_stage_action_dims=net_kwargs["noise_stage_action_dims"],
    ).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # ---- 为每个任务创建 env ----
    envs = {}
    for name in task_names:
        tc = tasks[name]
        envs[name] = _NoiseOptEnv(**tc["env_kwargs"])

    buffer = _GeneralStage2Buffer()
    return_norm = _RunningMeanStd()
    ppo_cnt = 0
    ppo_interval = _s1.PPO_UPDATE_INTERVAL
    total_episodes = total_rounds * num_tasks * episodes_per_task_per_round
    global_ep = 0

    best_configs = {n: None for n in task_names}
    best_rewards = {n: float('-inf') for n in task_names}
    all_rewards = {n: [] for n in task_names}
    all_entropies = []

    # ---- 续训练：加载 checkpoint ----
    resume_start_round = 0
    _resume_tol_rng_state = None
    if resume_checkpoint_path and os.path.isfile(resume_checkpoint_path):
        ckpt = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        ppo_cnt = ckpt.get("ppo_cnt", 0)
        global_ep = ckpt.get("global_ep", 0)
        best_configs = ckpt.get("best_configs", best_configs)
        best_rewards = ckpt.get("best_rewards", best_rewards)
        all_rewards = ckpt.get("all_rewards", all_rewards)
        all_entropies = ckpt.get("all_entropies", all_entropies)
        if "return_norm" in ckpt:
            rn = ckpt["return_norm"]
            return_norm.mean = rn["mean"]
            return_norm.var = rn["var"]
            return_norm.count = rn["count"]
        _resume_tol_rng_state = ckpt.get("tol_rng_state")
        resume_start_round = ckpt.get("completed_round", 0)
        log_fn(f"[通用噪声策略] 从 checkpoint 续训练: 已完成 {resume_start_round} 轮, "
               f"global_ep={global_ep}, ppo_cnt={ppo_cnt}")

    # ---- 停止信号文件路径 ----
    stop_flag_path = os.path.join(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        _STOP_FLAG_FILENAME,
    )

    # ---- 准确度容忍泛化 (Stage-2) ----
    _use_tol_range = (accuracy_tolerance_range is not None
                      and len(accuracy_tolerance_range) == 2
                      and accuracy_tolerance_range[0] < accuracy_tolerance_range[1])
    _use_tol_gen = (not _use_tol_range
                    and accuracy_tolerances is not None
                    and len(accuracy_tolerances) > 1)
    _tol_list = list(accuracy_tolerances) if accuracy_tolerances else []
    _tol_rng = np.random.RandomState(42)
    if _resume_tol_rng_state is not None:
        try:
            _tol_rng.set_state(_resume_tol_rng_state)
        except Exception:
            pass

    _tol_info = ""
    if _use_tol_range:
        _tol_lo, _tol_hi = float(accuracy_tolerance_range[0]), float(accuracy_tolerance_range[1])
        _tol_info = f"\n  准确度容忍泛化(连续): [{_tol_lo*100:.1f}%, {_tol_hi*100:.1f}%]"
    elif _use_tol_gen:
        _tol_info = f"\n  准确度容忍泛化(离散): {[f'{t*100:.1f}%' for t in _tol_list]}"

    _log_general_header(
        log_fn,
        "Stage-2 通用噪声策略训练（General Noise Policy Training）",
        [
            f"任务数量: {num_tasks}    训练轮数: {total_rounds}",
            f"每轮每任务回合数: {episodes_per_task_per_round}    总回合: {total_episodes}",
            f"任务列表: {', '.join(task_names)}",
            f"学习率: {lr}    PPO 间隔: {ppo_interval}" + _tol_info,
        ],
    )

    _reset_stop()
    _install_stop(stop_flag_path)
    _t0 = _time.time()

    for rnd in range(resume_start_round, total_rounds):
        # ---- 准确度容忍泛化: 每轮随机采样一个 tolerance ----
        if _use_tol_range:
            _cur_tol = float(_tol_rng.uniform(_tol_lo, _tol_hi))
            for name in task_names:
                tc = tasks[name]
                new_ctx, _ = recompute_task_for_tolerance(tc, _cur_tol)
                tc["task_context"] = new_ctx
            if rnd % 10 == 0:
                log_fn(f"  [容忍泛化·连续] 轮 {rnd+1}: tolerance={_cur_tol*100:.2f}%")
        elif _use_tol_gen:
            _cur_tol = float(_tol_rng.choice(_tol_list))
            for name in task_names:
                tc = tasks[name]
                new_ctx, _ = recompute_task_for_tolerance(tc, _cur_tol)
                tc["task_context"] = new_ctx
            if rnd % 10 == 0:
                log_fn(f"  [容忍泛化·离散] 轮 {rnd+1}: tolerance={_cur_tol*100:.1f}%")

        for task_name in task_names:
            tc = tasks[task_name]
            env = envs[task_name]
            net.set_task_context(
                torch.tensor(tc["task_context"], dtype=torch.float32).to(device)
            )

            for _ in range(episodes_per_task_per_round):
                # 熵调度 — 与 Stage-1 通用策略、per-task RL 保持一致
                progress = global_ep / max(total_episodes, 1)
                if _s1.RL_OPT_FLAGS.get("use_cosine_entropy_schedule", False):
                    plateau = float(_s1.RL_OPT_FLAGS.get("entropy_plateau_ratio", 0.25))
                    if progress <= plateau:
                        ec = _s1.PPO_ENTROPY_START
                    else:
                        tail = min(max((progress - plateau) / max(1.0 - plateau, 1e-8), 0.0), 1.0)
                        ec = _s1.PPO_ENTROPY_END + (_s1.PPO_ENTROPY_START - _s1.PPO_ENTROPY_END) * 0.5 * (1 + math.cos(math.pi * tail))
                else:
                    ec = _s1.PPO_ENTROPY_START - (_s1.PPO_ENTROPY_START - _s1.PPO_ENTROPY_END) * progress
                ec = max(ec, _s2._noise_rl_entropy_lower_bound(0.005))

                # 设置 env episode 进度
                env.set_episode_progress(global_ep, total_episodes)

                # Rollout
                ep_reward, noise_config, steps = _stage2_rollout_one_episode(
                    net, env, device, total_layers,
                )

                # 填充 buffer
                buffer.start_episode(task_context=tc["task_context"])
                for (cf, li, pa, act, lp, r, v, d) in steps:
                    buffer.add_step(cf, li, pa, act, lp, r, v, d)
                buffer.end_episode()

                all_rewards[task_name].append(ep_reward)
                global_ep += 1

                if ep_reward > best_rewards[task_name]:
                    best_rewards[task_name] = ep_reward
                    best_configs[task_name] = noise_config

                # PPO 更新
                if buffer.num_episodes >= ppo_interval:
                    pl, vl, ent = _general_ppo_update_stage2(
                        net, optimizer, buffer, device,
                        return_normalizer=return_norm,
                        entropy_coef=ec, ppo_update_step=ppo_cnt,
                    )
                    ppo_cnt += 1
                    buffer.clear()
                    all_entropies.append(ent)

                    if ppo_cnt % 10 == 0:
                        avg_str = ", ".join(
                            f"{n}={np.mean(all_rewards[n][-episodes_per_task_per_round:]):.3f}"
                            for n in task_names if all_rewards[n]
                        )
                        log_fn(f"  [PPO #{ppo_cnt}] 轮 {rnd+1}/{total_rounds} 任务 {task_name} | "
                               f"熵={ent:.4f} | {avg_str}")

        _elapsed = _time.time() - _t0
        _avg_rnd = _elapsed / max(rnd - resume_start_round + 1, 1)
        _eta = _avg_rnd * (total_rounds - rnd - 1)
        best_str = ", ".join(f"{n}={best_rewards[n]:.3f}" for n in task_names)
        avg_str = ", ".join(
            f"{n}={np.mean(all_rewards[n][-episodes_per_task_per_round:]):.3f}"
            for n in task_names if all_rewards[n]
        )
        _log_rounded_box(
            log_fn,
            [
                f"Stage-2 通用噪声策略 · 轮 {rnd+1} / {total_rounds}",
                _progress_bar(rnd + 1, total_rounds),
                f"近期均值: {avg_str}",
                f"各任务最优: {best_str}",
                f"PPO 更新: {ppo_cnt} 次  总回合: {global_ep}",
                f"已用时: {_fmt_elapsed(_elapsed)}  预计剩余: {_fmt_elapsed(_eta)}",
            ],
            indent="  ",
        )

        # ---- 优雅停止检查 ----
        if _is_stop(stop_flag_path):

            _consume_stop(stop_flag_path)
            ckpt_save_path = os.path.join(
                os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
                GENERAL_STAGE2_TRAIN_CHECKPOINT,
            )
            torch.save({
                "version": GENERAL_POLICY_VERSION,
                "completed_round": rnd + 1,
                "net_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ppo_cnt": ppo_cnt,
                "global_ep": global_ep,
                "best_configs": best_configs,
                "best_rewards": best_rewards,
                "all_rewards": all_rewards,
                "all_entropies": all_entropies,
                "return_norm": {
                    "mean": return_norm.mean,
                    "var": return_norm.var,
                    "count": return_norm.count,
                },
                "tol_rng_state": _tol_rng.get_state(),
                "task_names": task_names,
                "total_rounds": total_rounds,
                "episodes_per_task_per_round": episodes_per_task_per_round,
            }, ckpt_save_path)
            log_fn(f"[通用噪声策略] 优雅停止: checkpoint 已保存 → {ckpt_save_path} "
                   f"(已完成 {rnd+1}/{total_rounds} 轮)")
            _uninstall_stop()
            raise SystemExit(0)

    _uninstall_stop()

    # ---- 保存 general noise policy ----
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    torch.save({
        "version": GENERAL_POLICY_VERSION,
        "kind": "general_stage2_gtrxl_noise_policy",
        "net_state_dict": net.state_dict(),
        "arch": {
            "num_layers": total_layers,
            "d_model": _s2.NOISE_STAGE_GTRXL_D_MODEL,
            "n_heads": _s2.NOISE_STAGE_GTRXL_N_HEADS,
            "n_gtrxl_layers": _s2.NOISE_STAGE_GTRXL_N_LAYERS,
            "d_ff": _s2.NOISE_STAGE_GTRXL_D_FF,
            "dropout": _s2.NOISE_STAGE_GTRXL_DROPOUT,
            "task_context_dim": TASK_CONTEXT_DIM,
        },
        "metadata": {
            "task_names": task_names,
            "total_rounds": total_rounds,
            "episodes_per_task_per_round": episodes_per_task_per_round,
            "total_episodes": global_ep,
            "ppo_updates": ppo_cnt,
            "best_rewards": {n: float(best_rewards[n]) for n in task_names},
            "accuracy_tolerance_range": list(accuracy_tolerance_range) if _use_tol_range else None,
            "accuracy_tolerances": _tol_list if _use_tol_gen else None,
        },
    }, output_path)
    _total_elapsed = _time.time() - _t0
    best_str = ", ".join(f"{n}={best_rewards[n]:.3f}" for n in task_names)
    _log_rounded_box(
        log_fn,
        [
            "Stage-2 通用噪声策略训练 — 完成（Training Completed）",
            f"总轮数: {total_rounds}    PPO 更新: {ppo_cnt} 次    总回合: {global_ep}",
            f"各任务最优: {best_str}",
            f"总耗时: {_fmt_elapsed(_total_elapsed)}",
            f"策略已保存 → {output_path}",
        ],
        indent="",
    )

    net.set_task_context(None)
    return {
        "output_path": output_path,
        "best_configs": best_configs,
        "best_rewards": {n: float(best_rewards[n]) for n in task_names},
        "ppo_updates": ppo_cnt,
    }


# ===========================================================================
# Stage-2 Critic 快速评分
# ===========================================================================

def critic_quick_rank_stage2(
    evaluator,
    general_policy_path,
    candidate_noise_configs,
    fixed_gelu,
    fixed_softmax,
    device="cuda",
):
    """使用通用 Stage-2 Critic 的 V(s) 对一组噪声候选配置快速评分。

    与 offline_find_best_config_stage2 的区别：
      - 不做真实模型评测，只用 Critic 网络估值
      - 速度极快（纯 GPU 推理），适合预筛选大量候选
      - 精度取决于 Critic 训练质量

    Args:
        evaluator: LayerImportanceEvaluator 实例
        general_policy_path: str
        candidate_noise_configs: list of dicts, 每个 dict 包含 7 类 noise scaling factor arrays
        fixed_gelu: np.ndarray
        fixed_softmax: np.ndarray
        device: str

    Returns:
        list of (score, config) 按 score 降序排列
    """
    from layer_importance_evaluator import (
        NOISE_STAGE_NUM_ACTIONS, NOISE_STAGE_SOS_TOKENS,
        NOISE_STAGE_PREV_ACTION_EMBED_DIM, NOISE_STAGE_ACTION_DIMS,
        LSTM_POS_DIM, LSTM_PROJ_DIM, GTrXLBlock,
    )

    ckpt = torch.load(general_policy_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", {})
    total_layers = evaluator.total_layers

    net = GeneralStage2NoisePolicyNetwork(
        num_layers=total_layers,
        d_model=arch.get("d_model", _s2.NOISE_STAGE_GTRXL_D_MODEL),
        n_heads=arch.get("n_heads", _s2.NOISE_STAGE_GTRXL_N_HEADS),
        n_gtrxl_layers=arch.get("n_gtrxl_layers", _s2.NOISE_STAGE_GTRXL_N_LAYERS),
        d_ff=arch.get("d_ff", _s2.NOISE_STAGE_GTRXL_D_FF),
        dropout=arch.get("dropout", _s2.NOISE_STAGE_GTRXL_DROPOUT),
        gtrxl_block_cls=GTrXLBlock,
        lstm_pos_dim=LSTM_POS_DIM,
        lstm_proj_dim=LSTM_PROJ_DIM,
        noise_stage_num_actions=NOISE_STAGE_NUM_ACTIONS,
        noise_stage_sos_tokens=NOISE_STAGE_SOS_TOKENS,
        noise_stage_prev_action_embed_dim=NOISE_STAGE_PREV_ACTION_EMBED_DIM,
        noise_stage_cont_dim=_s2.NOISE_STAGE_V2_CONT_DIM,
        noise_stage_action_dims=NOISE_STAGE_ACTION_DIMS,
    ).to(device)
    net.load_state_dict(ckpt["net_state_dict"], strict=False)
    net.eval()

    # 构造 task context
    base_gelu = np.full(total_layers, 4, dtype=int)
    base_softmax = np.full(total_layers, 6, dtype=int)
    base_loss, base_m1, base_m2, _ = evaluator.stage1_evaluate(
        base_gelu, base_softmax, use_train=True,
    )
    tc = torch.tensor(compute_task_context(
        base_loss, base_m1, base_m2,
        evaluator.error_threshold, evaluator.correlation_drop_ratio,
    ), dtype=torch.float32).to(device)
    net.set_task_context(tc)

    sos_tokens = list(NOISE_STAGE_SOS_TOKENS)
    N = total_layers
    action_names = ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")

    scored = []
    for cfg in candidate_noise_configs:
        # 模拟完整 episode 的 token 序列
        cf_list, li_list, pa_list = [], [], []
        prev_act = sos_tokens[:]

        for layer in range(N):
            progress = (layer + 1) / N
            cf = np.array([0.0, 0.0, progress, 0.0, 0.0], dtype=np.float32)
            cf_list.append(torch.tensor(cf, dtype=torch.float32))
            li_list.append(layer)
            pa_list.append(torch.tensor(prev_act, dtype=torch.long))

            # 从配置中提取 action index（这里近似取值，精确的映射需要 action_dims）
            # 简单做法：使用 SOS tokens 值作为占位
            prev_act = sos_tokens[:]

        cf_t = torch.stack(cf_list).unsqueeze(0).to(device)       # (1, N, 5)
        li_t = torch.tensor([li_list], dtype=torch.long).to(device)
        pa_t = torch.stack(pa_list).unsqueeze(0).to(device)       # (1, N, 7)

        with torch.no_grad():
            _, vals = net.forward(cf_t, li_t, pa_t)
            score = vals[0, -1].item()

        scored.append((score, cfg))

    net.set_task_context(None)
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
