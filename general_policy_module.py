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

# ===========================================================================
# 配置常量
# ===========================================================================
TASK_CONTEXT_DIM = 5
GENERAL_POLICY_VERSION = 1


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
                         error_threshold=0.015, correlation_drop_ratio=0.015):
    """构造 5 维任务上下文特征向量。

    维度含义：
      [0] baseline_loss      — 基线 loss
      [1] baseline_m1        — 基线 metric1（acc / pearson 等）
      [2] baseline_m2        — 基线 metric2（f1 / spearman 等）
      [3] error_threshold    — 约束阈值（如 0.015）
      [4] correlation_drop   — 指标下降容忍（如 0.015）
    """
    return np.array([
        float(baseline_loss), float(baseline_m1), float(baseline_m2),
        float(error_threshold), float(correlation_drop_ratio),
    ], dtype=np.float32)


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

def prepare_stage1_task(evaluator, use_train=True):
    """从 evaluator 准备一个 task config dict，用于 multi_task_train_stage1 或 offline 推断。

    Args:
        evaluator: LayerImportanceEvaluator 实例（模型 + 数据集已加载）
        use_train: 是否使用训练集计算基线（默认 True）

    Returns:
        dict，包含多任务训练 / 离线推断所需的全部信息
    """
    base_gelu = np.full(evaluator.total_layers, 4, dtype=int)
    base_softmax = np.full(evaluator.total_layers, 6, dtype=int)
    base_cost = evaluator.get_simulated_cost(base_gelu, base_softmax)[0]
    base_loss, base_m1, base_m2, _ = evaluator.stage1_evaluate(
        base_gelu, base_softmax, use_train=use_train,
    )
    task_ctx = compute_task_context(
        base_loss, base_m1, base_m2,
        evaluator.error_threshold, evaluator.correlation_drop_ratio,
    )
    return {
        "evaluator": evaluator,
        "baseline_metrics": (float(base_loss), float(base_m1), float(base_m2)),
        "baseline_cost": float(base_cost),
        "constraint_limits": {
            "loss": float(base_loss + evaluator.error_threshold),
            "metric1": float(base_m1 * (1.0 - evaluator.correlation_drop_ratio)),
            "metric2": float(base_m2 * (1.0 - evaluator.correlation_drop_ratio)),
        },
        "gelu0_eligible": np.zeros(evaluator.total_layers, dtype=bool),
        "task_context": task_ctx,
        "num_metrics": evaluator.get_num_metrics(),
    }


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
    """
    state = env.reset()
    prev_g = torch.tensor([[_SOS_GELU]], dtype=torch.long, device=device)
    prev_s = torch.tensor([[_SOS_SOFTMAX]], dtype=torch.long, device=device)

    seq_cf, seq_li, seq_pg, seq_ps, seq_gm = [], [], [], [], []
    episode_reward = 0.0
    steps = []
    N = total_layers

    for step in range(N):
        layer_idx = int(np.argmax(state[0:N]))
        lb, m1b, m2b = 3 * N + 5, 3 * N + 6, 3 * N + 7
        cf = np.array([
            state[N], state[N + 3], state[N + 4],
            state[lb] if len(state) > lb else 0.0,
            state[m1b] if len(state) > m1b else 0.0,
            state[m2b] if len(state) > m2b else 0.0,
        ], dtype=np.float32)

        cf_t = torch.tensor(cf, dtype=torch.float32).view(1, 1, -1).to(device)
        li_t = torch.tensor([[layer_idx]], dtype=torch.long, device=device)
        gm_np = env.get_gelu_action_mask(layer_idx)
        gm_t = torch.tensor(gm_np, dtype=torch.bool).view(1, 1, -1).to(device)

        seq_cf.append(cf_t)
        seq_li.append(li_t)
        seq_pg.append(prev_g)
        seq_ps.append(prev_s)
        seq_gm.append(gm_t)

        full_cf = torch.cat(seq_cf, dim=1)
        full_li = torch.cat(seq_li, dim=1)
        full_pg = torch.cat(seq_pg, dim=1)
        full_ps = torch.cat(seq_ps, dim=1)
        full_gm = torch.cat(seq_gm, dim=1)

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
        steps.append((cf, layer_idx, prev_g.squeeze().item(), prev_s.squeeze().item(),
                       g_act.item(), s_act.item(), logprob.cpu(), reward, value.cpu(),
                       float(done), gm_np))

        prev_g = g_act.reshape(1, 1).to(device)
        prev_s = s_act.reshape(1, 1).to(device)
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
    episodes_per_task_per_round=170,
    lr=3e-5,
    log_fn=None,
    device="cuda",
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
        ev = tc["evaluator"]

        class _W:
            def __init__(s, e):
                s._e = e
            def evaluate_model(s, g, sm):
                return s._e.stage1_evaluate(g, sm, use_train=True)

        envs[name] = _TransformerOptEnv(
            total_layers, tc["baseline_cost"], tc["baseline_metrics"],
            _W(ev), num_metrics=tc["num_metrics"],
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

    log_fn(f"[通用策略] 开始多任务训练: {num_tasks} 个任务, {total_rounds} 轮, "
           f"每轮每任务 {episodes_per_task_per_round} 回合, 总 {total_episodes} 回合")
    log_fn(f"[通用策略] 任务列表: {task_names}")

    for rnd in range(total_rounds):
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

        best_str = ", ".join(f"{n}={best_rewards[n]:.3f}" for n in task_names)
        log_fn(f"[通用策略] 轮 {rnd+1}/{total_rounds} 完成 | 各任务最优: {best_str}")

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
        },
    }, output_path)
    log_fn(f"[通用策略] 已保存 → {output_path}")

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

    task_info = prepare_stage1_task(evaluator)
    tc = torch.tensor(task_info["task_context"], dtype=torch.float32).to(device)
    net.set_task_context(tc)

    class _W:
        def __init__(s, e):
            s._e = e
        def evaluate_model(s, g, sm):
            return s._e.stage1_evaluate(g, sm, use_train=True)

    env = _TransformerOptEnv(
        total_layers, task_info["baseline_cost"], task_info["baseline_metrics"],
        _W(evaluator), num_metrics=task_info["num_metrics"],
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

    log_fn(f"[离线推断] 开始 {'贪心' if greedy else f'{num_rollouts} 次采样'} rollout...")

    for i in range(num_rollouts):
        ep_reward, config, _steps = _stage1_rollout_one_episode(
            net, env, device, total_layers, greedy=greedy,
        )
        all_results.append(config)
        if ep_reward > best_reward:
            best_reward = ep_reward
            best_config = config
        if (i + 1) % max(1, num_rollouts // 5) == 0:
            log_fn(f"  [{i+1}/{num_rollouts}] 当前最优 reward={best_reward:.4f}")

    net.set_task_context(None)
    log_fn(f"[离线推断] 完成. 最优 reward={best_reward:.4f}, cost={best_config['cost']:.2f}")
    log_fn(f"  GELU:    {best_config['gelu'].tolist()}")
    log_fn(f"  Softmax: {best_config['softmax'].tolist()}")

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

    log_fn(f"[Stage-2 离线] 开始 {'贪心' if greedy else f'{num_rollouts} 次采样'} rollout...")

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

        if (i + 1) % max(1, num_rollouts // 5) == 0:
            log_fn(f"  [{i+1}/{num_rollouts}] 当前最优 reward={best_reward:.4f}")

    net.set_task_context(None)
    log_fn(f"[Stage-2 离线] 完成. 最优 reward={best_reward:.4f}")

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

    gelu_to_idx = {4: 0, 2: 1, 1: 2, 0: 3}
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
            prev_g_idx = gelu_to_idx.get(g_deg, 0)
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
