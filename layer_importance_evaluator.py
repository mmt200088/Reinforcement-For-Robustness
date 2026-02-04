import time
import copy
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerCallback
from scipy.stats import pearsonr, spearmanr
from function_handler import ReversibleLayerHandler
import os
import hashlib

# ==================== PPO 常量与配置定义 ====================
# 动作映射表（严格按照任务要求）
GELU_MAP = {0: 4, 1: 2, 2: 1}
GELU_COST = {4: 3.0, 2: 2.5, 1: 1.0}
SOFTMAX_MAP = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2}
SOFTMAX_COST = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}

# PPO 超参数
PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95  # GAE lambda
PPO_EPS_CLIP = 0.2
PPO_K_EPOCHS = 4
PPO_ENTROPY_COEF = 0.05  # 初始熵系数（策略二：从0.05开始衰减）
PPO_VALUE_COEF = 0.5
# PDF 6.4：由于采样步数大幅增加，需要相应调整总episodes
# PDF建议：每2048 steps更新，考虑到每episode 12步，即每170 episodes更新
PPO_MAX_EPISODES = 1700
# PDF 6.4：增加采样步数以获得更稳定的梯度估计
# "建议增加采样步数，例如每2048 steps更新一次"
# 折中方案：每50 episodes更新一次（600步），在训练时间和稳定性之间平衡
PPO_UPDATE_INTERVAL = 170  # 每50个episode更新一次（600步）
PPO_BATCH_SIZE = 12 * 170  # 50 episodes per update, 每个episode 12步 = 600步

# ==================== 策略二：超参数配置（PDF 6.4 稳定性修复） ====================
# PDF 6.4：降低学习率，使用恒定学习率先跑通基线
PPO_LR_INITIAL = 5e-5       # 初始学习率（PDF 6.4：降至5e-5）
PPO_LR_FINAL = 5e-5         # 最终学习率（恒定，移除复杂衰减）
PPO_WARMUP_RATIO = 0.0      # PDF 6.4：移除复杂Warmup机制，先用恒定学习率跑通基线
# PDF 6.4：固定熵系数为0.01，移除激进衰减
# "建议固定熵系数为0.01，或者仅在检测到KL散度过低时才衰减"
PPO_ENTROPY_COEF_FIXED = 0.01  # 固定熵系数（PDF 6.4）
PPO_ENTROPY_INITIAL = 0.01  # 初始熵系数（与固定值一致）
PPO_ENTROPY_FINAL = 0.01    # 最终熵系数（与固定值一致，不衰减）
PPO_ENTROPY_DECAY = 1.0     # 熵系数衰减率设为1.0表示不衰减

# PDF 6.4：增大Batch Size以获得更稳定的梯度估计
# "建议增加采样步数，例如每2048 steps更新一次，Mini-batch设为64或128"
PPO_MINI_BATCH_SIZE = 64    # Mini-batch大小（PDF 6.4：增至64）

# ==================== 策略一：奖励函数重构配置（PDF 6.1 线性化修复） ====================
REWARD_THRESHOLD = 0.01       # 约束阈值 1%
REWARD_SAFETY_BUFFER = 0.002  # 安全边界 0.2%
REWARD_TARGET = REWARD_THRESHOLD - REWARD_SAFETY_BUFFER  # 有效目标 0.8%
REWARD_COST_WEIGHT = 20.0     # 成本奖励权重
REWARD_SAFETY_BONUS = 1.0     # 安全区域基础奖励
# PDF 6.1：移除指数惩罚，改用线性惩罚（防止梯度爆炸）
REWARD_PENALTY_SLOPE = 50.0   # 线性惩罚斜率（替代指数惩罚）
REWARD_DENSE_SCALE = 0.1      # 稠密中间奖励缩放系数
# PDF 6.1：奖励截断范围，防止异常值传播
REWARD_CLIP_MIN = -5.0        # 奖励下限
REWARD_CLIP_MAX = 5.0         # 奖励上限

# ==================== 策略三：回报归一化配置 ====================
REWARD_NORMALIZATION_SCALE = 20.0  # 固定缩放因子（将-100量级缩放到-5左右）

# ==================== PPO 5.2: Value Clipping 配置 ====================
VALUE_CLIP_RANGE = 0.2  # 价值函数裁剪范围，与PPO_EPS_CLIP保持一致

# ==================== Transformer 7.2: 预算偏离度中间奖励配置 ====================
BUDGET_DEVIATION_SCALE = 0.05  # 预算偏离度奖励缩放系数

# ==================== PPO 7.1: 运行时回报归一化配置 ====================
RUNNING_REWARD_HISTORY_SIZE = 100  # 滑动窗口大小
RUNNING_REWARD_MIN_SAMPLES = 10    # 开始标准化前的最小样本数
RUNNING_REWARD_EPSILON = 1e-8      # 防止除零的小常数

# ==================== 显式历史编码配置（PDF优化方案一） ====================
# 状态向量维度：17维原始特征 + 12维GELU历史 + 12维Softmax历史 = 41维
STATE_DIM_ORIGINAL = 17  # 原始状态维度
STATE_DIM_HISTORY = 24   # 历史编码维度（12 GELU + 12 Softmax）
STATE_DIM_TOTAL = STATE_DIM_ORIGINAL + STATE_DIM_HISTORY  # 总维度 = 41
# PDF 6.2 步骤1：将填充值从 -1.0 改为 0.0
# 理由：在ReLU/SiLU激活的网络中，0输入通常产生0输出，天然表示"无信息"
# -1.0 是一个强烈的信号值，会干扰特征提取
HISTORY_MASK_VALUE = 0.0  # 未访问层的掩码值（PDF 6.2：零值填充）


def orthogonal_init(layer, gain=1.0):
    """正交初始化"""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


# ==================== PDF 6.3: RunningMeanStd for Return Normalization ====================
class RunningMeanStd:
    """
    PDF 6.3 步骤2：跟踪回报统计量的运行均值和标准差
    用于实现 PopArt/Return Normalization
    
    使用 Welford's online algorithm 进行数值稳定的增量更新
    """
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon  # 防止除零
        
    def update(self, x):
        """
        使用 Welford's algorithm 增量更新均值和方差
        Args:
            x: numpy array 或 torch tensor 的数据
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x).flatten()
        
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """Welford's parallel algorithm for combining statistics"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        # 更新均值
        self.mean = self.mean + delta * batch_count / total_count
        
        # 更新方差（使用 parallel algorithm）
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        
        self.count = total_count
    
    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)
    
    def normalize(self, x):
        """归一化数据"""
        if isinstance(x, torch.Tensor):
            return (x - self.mean) / (self.std + 1e-8)
        return (x - self.mean) / (self.std + 1e-8)
    
    def denormalize(self, x):
        """反归一化数据"""
        return x * self.std + self.mean


# ==================== PDF网络优化方案：残差块 ====================
class ResidualBlock(nn.Module):
    """
    残差块（ResMLP）- PDF 5.1节
    采用 Pre-Norm 结构：先 LayerNorm 再 Linear
    使用 SiLU (Swish) 激活函数替代 Tanh/ReLU，更平滑
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        # Pre-Norm 结构
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()  # 使用 SiLU (Swish) 替代 Tanh/ReLU
        self.dropout = nn.Dropout(dropout)
        
        # 正交初始化
        orthogonal_init(self.fc1, gain=np.sqrt(2))
        orthogonal_init(self.fc2, gain=np.sqrt(2))
    
    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.activation(self.fc1(out))
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.fc2(out)
        return residual + out  # 纯加法残差


# ==================== PDF网络优化方案：状态编码器 ====================
class StateEncoder(nn.Module):
    """
    状态编码器 - PDF 4.1节 & 5.2节
    将异构的原始输入（41维）转化为统一的语义向量
    
    输入分解为三个流：
    1. 层级流 (Layer Stream): Index 0-11，通过Embedding映射
    2. 指标流 (Metric Stream): Index 12-16，通过全连接层映射
    3. 历史序列流 (History Stream): Index 17-40，通过Transformer Encoder处理
    """
    def __init__(self, embed_dim=64, num_layers=12):
        super(StateEncoder, self).__init__()
        self.num_layers = num_layers
        
        # 1. 层级 ID 嵌入（PDF 3.1.1：Entity Embedding）
        self.layer_embed = nn.Embedding(num_layers, 32)
        
        # 2. 连续指标映射（Index 12-16: cost_deviation, gelu_norm, softmax_norm, complexity_debt, progress）
        self.metric_proj = nn.Sequential(
            nn.Linear(5, 32),
            nn.SiLU()
        )
        
        # 3. 历史序列处理 (Transformer) - PDF 3.1.2
        # 输入维度为 2 (Gelu值, Softmax值)
        self.hist_proj = nn.Linear(2, 32)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_layers, 32))  # 可学习位置编码
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.hist_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 4. 融合层 - PDF 4.1.2
        # 拼接三个流的输出：Layer(32) + Metric(32) + Hist(32) = 96
        self.fusion = nn.Sequential(
            nn.Linear(32 * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, state_vector):
        """
        Args:
            state_vector: (Batch, 41) 或 (41,) 的状态向量
        Returns:
            (Batch, embed_dim) 的编码向量
        """
        # 处理单样本输入
        if state_vector.dim() == 1:
            state_vector = state_vector.unsqueeze(0)
        
        batch_size = state_vector.size(0)
        
        # A. 解析 Layer Index (Index 0-11) - 将 one-hot 转为索引
        layer_indices = torch.argmax(state_vector[:, 0:12], dim=1)
        l_emb = self.layer_embed(layer_indices)  # (Batch, 32)
        
        # B. 解析 Metrics (Index 12-16)
        metrics = state_vector[:, 12:17]
        m_emb = self.metric_proj(metrics)  # (Batch, 32)
        
        # C. 解析 History (Index 17-40)
        # 拆分为 (Batch, 12) 的 GELU 和 Softmax 历史
        hist_gelu = state_vector[:, 17:29].unsqueeze(-1)  # (Batch, 12, 1)
        hist_softmax = state_vector[:, 29:41].unsqueeze(-1)  # (Batch, 12, 1)
        hist_seq = torch.cat([hist_gelu, hist_softmax], dim=-1)  # (Batch, 12, 2)
        
        # PDF 6.2：使用零值填充后，需要用当前层索引来判断有效历史
        # 生成 Padding Mask: 根据当前层索引判断哪些历史位置是有效的
        # 如果当前层是 layer_i，则 layer_0 到 layer_{i-1} 是有效的（已访问过的）
        # Transformer mask: True 表示要被忽略
        # 创建位置索引 [0, 1, 2, ..., num_layers-1]
        position_indices = torch.arange(self.num_layers, device=state_vector.device).unsqueeze(0)  # (1, 12)
        # 扩展到 batch 维度
        position_indices = position_indices.expand(batch_size, -1)  # (Batch, 12)
        # layer_indices 是当前层，历史中 index < current_layer 的位置是有效的
        padding_mask = position_indices >= layer_indices.unsqueeze(1)  # (Batch, 12)
        
        # PDF 6.2：零值填充不需要替换，直接使用
        hist_seq_clean = hist_seq
        
        # Transformer Forward
        h_x = self.hist_proj(hist_seq_clean) + self.pos_embed  # (Batch, 12, 32)
        
        # src_key_padding_mask shape: (Batch, Seq_Len)
        h_out = self.hist_transformer(h_x, src_key_padding_mask=padding_mask)  # (Batch, 12, 32)
        
        # Pooling: 对未被 mask 的 token 求平均（Mean Pooling）
        mask_float = (~padding_mask).float().unsqueeze(-1)  # (Batch, 12, 1)
        valid_count = mask_float.sum(dim=1).clamp(min=1e-6)  # (Batch, 1) 避免除以0
        h_pooled = (h_out * mask_float).sum(dim=1) / valid_count  # (Batch, 32)
        
        # D. 融合
        combined = torch.cat([l_emb, m_emb, h_pooled], dim=1)  # (Batch, 96)
        return self.fusion(combined)  # (Batch, embed_dim)


# ==================== PDF网络优化方案：策略网络 ====================
class PolicyNetwork(nn.Module):
    """
    策略网络（ResMLP Actor）- PDF 4.2节
    使用 StateEncoder 进行特征提取，后接 2 个 ResidualBlock
    双头输出设计：GELU Head 和 Softmax Head
    """
    def __init__(self, state_dim=STATE_DIM_TOTAL, hidden_dim=64, res_hidden_dim=128, num_layers=12):
        super(PolicyNetwork, self).__init__()
        
        # 状态编码器
        self.encoder = StateEncoder(embed_dim=hidden_dim, num_layers=num_layers)
        
        # 残差骨干网络：2个残差块（PDF 4.2节）
        self.res_block1 = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        self.res_block2 = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        
        # GELU Head: 输出3个动作的logits (对应度数 1, 2, 4)
        self.gelu_head = nn.Linear(hidden_dim, 3)
        # Softmax Head: 输出5个动作的logits (对应度数 2, 3, 4, 5, 6)
        self.softmax_head = nn.Linear(hidden_dim, 5)
        
        # PDF 4.2节：策略输出层使用极小的 gain (0.01)
        # 确保训练初始阶段各动作概率几乎均等（高熵），最大化探索能力
        orthogonal_init(self.gelu_head, gain=0.01)
        orthogonal_init(self.softmax_head, gain=0.01)
    
    def forward(self, state):
        # 特征编码
        x = self.encoder(state)
        
        # 残差块处理
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 双头输出
        gelu_logits = self.gelu_head(x)
        softmax_logits = self.softmax_head(x)
        
        return gelu_logits, softmax_logits
    
    def get_action_and_logprob(self, state, return_probs=False):
        """
        获取动作和log概率
        Args:
            state: 状态张量
            return_probs: 是否返回概率分布（用于记录中间结果）
        Returns:
            gelu_action, softmax_action, logprob
            如果return_probs=True，还返回gelu_probs, softmax_probs
        """
        gelu_logits, softmax_logits = self.forward(state)
        
        # 处理单样本情况
        if gelu_logits.dim() == 1:
            gelu_logits = gelu_logits.unsqueeze(0)
            softmax_logits = softmax_logits.unsqueeze(0)
        
        gelu_dist = Categorical(logits=gelu_logits)
        softmax_dist = Categorical(logits=softmax_logits)
        
        gelu_action = gelu_dist.sample()
        softmax_action = softmax_dist.sample()
        
        gelu_logprob = gelu_dist.log_prob(gelu_action)
        softmax_logprob = softmax_dist.log_prob(softmax_action)
        
        if return_probs:
            # 返回概率分布用于记录中间结果
            gelu_probs = torch.softmax(gelu_logits, dim=-1)
            softmax_probs = torch.softmax(softmax_logits, dim=-1)
            return gelu_action.squeeze(), softmax_action.squeeze(), (gelu_logprob + softmax_logprob).squeeze(), gelu_probs.squeeze(), softmax_probs.squeeze()
        
        return gelu_action.squeeze(), softmax_action.squeeze(), (gelu_logprob + softmax_logprob).squeeze()
    
    def evaluate_actions(self, states, gelu_actions, softmax_actions):
        gelu_logits, softmax_logits = self.forward(states)
        
        gelu_dist = Categorical(logits=gelu_logits)
        softmax_dist = Categorical(logits=softmax_logits)
        
        gelu_logprob = gelu_dist.log_prob(gelu_actions)
        softmax_logprob = softmax_dist.log_prob(softmax_actions)
        
        gelu_entropy = gelu_dist.entropy()
        softmax_entropy = softmax_dist.entropy()
        
        return gelu_logprob + softmax_logprob, gelu_entropy + softmax_entropy


# ==================== PDF网络优化方案：价值网络 ====================
class ValueNetwork(nn.Module):
    """
    价值网络（ResMLP Critic）- PDF 4.3节
    Critic 拥有独立的 StateEncoder 实例，不与 Actor 共享，以防止梯度干扰（PDF 3.2.1）
    使用 3 个 ResidualBlock，比 Actor 深一层，增加表达能力
    """
    def __init__(self, state_dim=STATE_DIM_TOTAL, hidden_dim=64, res_hidden_dim=128, num_layers=12):
        super(ValueNetwork, self).__init__()
        
        # 独立的状态编码器（PDF 3.2.1：解耦架构）
        self.encoder = StateEncoder(embed_dim=hidden_dim, num_layers=num_layers)
        
        # 残差骨干网络：3个残差块（PDF 4.3节：比Actor深一层）
        self.res_block1 = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        self.res_block2 = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        self.res_block3 = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        
        # 输出层：标量 Value
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # PDF 4.3节：输出层使用正交初始化，gain=1.0（不同于Actor的0.01）
        orthogonal_init(self.value_head, gain=1.0)
    
    def forward(self, state):
        # 特征编码
        x = self.encoder(state)
        
        # 残差块处理（3层）
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # 输出标量价值
        value = self.value_head(x)
        return value.squeeze(-1)


class RolloutBuffer:
    """存储Rollout数据的Buffer"""
    def __init__(self):
        self.states = []
        self.gelu_actions = []
        self.softmax_actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def add(self, state, gelu_action, softmax_action, logprob, reward, done, value):
        self.states.append(state)
        self.gelu_actions.append(gelu_action)
        self.softmax_actions.append(softmax_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.gelu_actions.clear()
        self.softmax_actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
    
    def get_tensors(self, device):
        states = torch.stack(self.states).to(device)
        gelu_actions = torch.stack(self.gelu_actions).to(device)
        softmax_actions = torch.stack(self.softmax_actions).to(device)
        logprobs = torch.stack(self.logprobs).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        values = torch.stack(self.values).to(device)
        return states, gelu_actions, softmax_actions, logprobs, rewards, dones, values


class TransformerOptEnv:
    """
    Transformer优化环境
    已实施PDF优化策略：
    - 策略一：奖励函数重构（稠密化中间奖励、指数障碍软约束、安全边界）
    - 策略三：回报归一化（固定缩放）
    - 策略四：状态空间增强（累积复杂度债务、成本偏差相对化）
    - PDF优化方案一：显式历史编码（Flattened History）- 解决序列依赖性问题
    """
    def __init__(self, total_layers, baseline_cost, baseline_metrics, evaluator):
        self.total_layers = total_layers
        self.baseline_cost = baseline_cost  # 72.0 for 12 layers
        self.baseline_loss, self.baseline_p, self.baseline_s = baseline_metrics
        self.evaluator = evaluator
        
        # 策略四：计算理论中间成本（假设每层选中间阶数）
        # GELU中间阶数=2 (cost=2.5), Softmax中间阶数=4 (cost=2.0)
        self.mid_gelu_cost = GELU_COST[2]  # 2.5
        self.mid_softmax_cost = SOFTMAX_COST[4]  # 2.0
        self.expected_cost_per_layer = self.mid_gelu_cost + self.mid_softmax_cost  # 4.5
        
        # 策略一：计算基线每层成本（用于稠密奖励计算）
        self.max_cost_per_layer = GELU_COST[4] + SOFTMAX_COST[6]  # 6.0
        
        # PDF优化方案一：显式历史编码
        # 动作归一化映射（将degree映射到[0,1]区间）
        # GELU: degree 4->0.0, 2->0.5, 1->1.0 (高精度->低精度)
        self.gelu_degree_to_norm = {4: 0.0, 2: 0.5, 1: 1.0}
        # Softmax: degree 6->0.0, 5->0.25, 4->0.5, 3->0.75, 2->1.0
        self.softmax_degree_to_norm = {6: 0.0, 5: 0.25, 4: 0.5, 3: 0.75, 2: 1.0}
        
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.current_layer = 0  # 0-indexed
        self.accumulated_cost = 0.0
        self.gelu_config = []
        self.softmax_config = []
        self.prev_gelu_degree = 4  # 初始默认值
        self.prev_softmax_degree = 6  # 初始默认值
        
        # 策略一：累计的中间奖励
        self.accumulated_dense_reward = 0.0
        
        # PDF优化方案一：初始化动作历史缓冲区
        # gelu_history[i] 存储第i层的GELU动作（归一化后），未访问层为掩码值
        # softmax_history[i] 存储第i层的Softmax动作（归一化后），未访问层为掩码值
        self.gelu_history = np.full(self.total_layers, HISTORY_MASK_VALUE, dtype=np.float32)
        self.softmax_history = np.full(self.total_layers, HISTORY_MASK_VALUE, dtype=np.float32)
        
        return self._get_state()
    
    def _get_state(self):
        """
        构造41维状态向量（PDF优化方案一：显式历史编码）
        
        原始17维特征：
        - 12维: 位置编码 (One-Hot)
        - 1维: 成本偏差 (Cost Deviation) - 中心化处理
        - 2维: 上一步动作编码
        - 1维: 累积复杂度债务 (Complexity Debt)
        - 1维: 进度指示 (Progress Indicator)
        
        新增24维历史编码（PDF优化方案一 + PDF 6.2零值填充）：
        - 12维: GELU动作历史（归一化，未访问层为0）
        - 12维: Softmax动作历史（归一化，未访问层为0）
        """
        # ========== 原始17维特征 ==========
        # 1. 位置编码 (12维 One-Hot)
        position = np.zeros(self.total_layers)
        if self.current_layer < self.total_layers:
            position[self.current_layer] = 1.0
        
        # 2. 策略四（6.2）：成本偏差相对化 (Cost Deviation)
        # expected_cost_so_far = 假设每层选中间阶数时的理论累积成本
        expected_cost_so_far = self.current_layer * self.expected_cost_per_layer
        if expected_cost_so_far > 0:
            cost_deviation = (self.accumulated_cost - expected_cost_so_far) / expected_cost_so_far
        else:
            cost_deviation = 0.0
        # 截断到合理范围 [-1, 1]
        cost_deviation = np.clip(cost_deviation, -1.0, 1.0)
        
        # 3. 上一步动作编码 (2维, 归一化到[0,1])
        # GELU: 4->0, 2->0.5, 1->1.0
        gelu_norm = self.gelu_degree_to_norm.get(self.prev_gelu_degree, 0.0)
        # Softmax: 6->0, 5->0.25, 4->0.5, 3->0.75, 2->1.0
        softmax_norm = self.softmax_degree_to_norm.get(self.prev_softmax_degree, 0.0)
        
        # 4. 策略四（6.1）：累积复杂度债务 (Complexity Debt)
        # 计算相对于基线的累积降级程度
        baseline_cost_so_far = self.current_layer * self.max_cost_per_layer
        if baseline_cost_so_far > 0:
            complexity_debt = (baseline_cost_so_far - self.accumulated_cost) / baseline_cost_so_far
        else:
            complexity_debt = 0.0
        # complexity_debt > 0 表示比基线省钱（牺牲了模型容量）
        complexity_debt = np.clip(complexity_debt, 0.0, 1.0)
        
        # 5. 进度指示 (Progress Indicator) - 帮助Critic理解当前位置
        progress = self.current_layer / self.total_layers
        
        # ========== 新增24维历史编码（PDF优化方案一 + PDF 6.2零值填充） ==========
        # 6. GELU动作历史 (12维) - 已访问层为归一化动作值，未访问层为0（PDF 6.2）
        # 7. Softmax动作历史 (12维) - 同上
        # 注意：self.gelu_history 和 self.softmax_history 在step()中更新
        
        state = np.concatenate([
            position,                          # 12维: 位置编码
            [cost_deviation],                  # 1维: 成本偏差
            [gelu_norm, softmax_norm],         # 2维: 上一步动作
            [complexity_debt],                 # 1维: 复杂度债务
            [progress],                        # 1维: 进度指示
            self.gelu_history,                 # 12维: GELU完整历史（PDF优化方案一）
            self.softmax_history               # 12维: Softmax完整历史（PDF优化方案一）
        ])
        return state.astype(np.float32)
    
    def _compute_dense_step_reward(self, gelu_degree, softmax_degree):
        """
        策略一（3.1）+ Transformer 7.2：计算稠密化中间奖励
        包含两部分：
        1. 基于当前层成本节约的即时反馈
        2. 基于预算偏离度的轨道引导奖励（Transformer 7.2）
        """
        step_cost = GELU_COST[gelu_degree] + SOFTMAX_COST[softmax_degree]
        
        # 1. 策略一（3.1）：相对于最大成本的节约比例
        cost_saving = (self.max_cost_per_layer - step_cost) / self.max_cost_per_layer
        cost_reward = REWARD_DENSE_SCALE * cost_saving
        
        # 2. Transformer 7.2：基于预算偏离度的中间奖励
        # 计算当前应处于的"理想"累积成本（假设每层选中间阶数）
        # 在执行当前动作后，current_layer 还未 +1，所以理想成本是 (current_layer + 1) * expected_cost_per_layer
        layers_completed = self.current_layer + 1  # 包括当前层
        expected_cost_so_far = layers_completed * self.expected_cost_per_layer
        
        # 计算实际累积成本（包括当前层的成本）
        actual_cost_so_far = self.accumulated_cost + step_cost
        
        # 预算偏离度：正值表示超预算（贵了），负值表示省预算（便宜了）
        if expected_cost_so_far > 0:
            budget_deviation = (actual_cost_so_far - expected_cost_so_far) / expected_cost_so_far
        else:
            budget_deviation = 0.0
        
        # 偏离度奖励：偏离越小（越接近0）越好，给予小的正向奖励
        # 使用负的绝对值偏离度，使智能体倾向于保持在预算轨道附近
        # 但同时允许省钱（负偏离），所以对省钱方向给予较小惩罚
        if budget_deviation <= 0:
            # 省钱（低于预算）：轻微奖励，但不要奖励太多以免过于保守
            budget_reward = BUDGET_DEVIATION_SCALE * (1.0 - abs(budget_deviation) * 0.5)
        else:
            # 超预算：给予惩罚，偏离越大惩罚越重
            budget_reward = -BUDGET_DEVIATION_SCALE * budget_deviation
        
        # 合并两部分奖励
        dense_reward = cost_reward + budget_reward
        return dense_reward
    
    def step(self, gelu_action_idx, softmax_action_idx):
        """执行动作，返回(next_state, reward, done, info)"""
        # 映射动作到degree
        gelu_degree = GELU_MAP[gelu_action_idx]
        softmax_degree = SOFTMAX_MAP[softmax_action_idx]
        
        # 记录配置
        self.gelu_config.append(gelu_degree)
        self.softmax_config.append(softmax_degree)
        
        # 更新累积开销
        self.accumulated_cost += GELU_COST[gelu_degree] + SOFTMAX_COST[softmax_degree]
        
        # 更新上一步动作
        self.prev_gelu_degree = gelu_degree
        self.prev_softmax_degree = softmax_degree
        
        # PDF优化方案一：更新动作历史缓冲区
        # 将当前层的动作（归一化后）存入历史
        self.gelu_history[self.current_layer] = self.gelu_degree_to_norm[gelu_degree]
        self.softmax_history[self.current_layer] = self.softmax_degree_to_norm[softmax_degree]
        
        # 策略一（3.1）：计算稠密中间奖励
        dense_reward = self._compute_dense_step_reward(gelu_degree, softmax_degree)
        self.accumulated_dense_reward += dense_reward
        
        # 构建中间结果信息
        info = {
            'layer_index': self.current_layer,
            'curr_gelu_degree': gelu_degree,
            'curr_softmax_degree': softmax_degree,
            'accumulated_cost': self.accumulated_cost,
            'gelu_config': self.gelu_config.copy(),
            'softmax_config': self.softmax_config.copy(),
            'dense_reward': dense_reward,  # 记录稠密奖励
            'gelu_history': self.gelu_history.copy(),      # PDF优化方案一：记录历史
            'softmax_history': self.softmax_history.copy()  # PDF优化方案一：记录历史
        }
        
        self.current_layer += 1
        
        if self.current_layer < self.total_layers:
            # 回合未结束，返回稠密中间奖励（策略一）
            return self._get_state(), dense_reward, False, info
        else:
            # 回合结束，计算最终奖励
            final_reward = self._compute_final_reward()
            info['final_reward'] = final_reward
            info['accumulated_dense_reward'] = self.accumulated_dense_reward
            # 最终奖励 = 基于约束的奖励 + 最后一步的稠密奖励
            total_reward = final_reward + dense_reward
            return self._get_state(), total_reward, True, info
    
    def _compute_final_reward(self):
        """
        策略一（3.2, 3.3）+ 策略三（5.1）：计算最终奖励
        - 指数障碍软约束替代二进制惩罚
        - 引入安全边界
        - 回报归一化（固定缩放）
        """
        # 获取当前配置的指标
        gelu_arr = np.array(self.gelu_config)
        softmax_arr = np.array(self.softmax_config)
        
        # 评估模型
        loss, p, s, _ = self.evaluator.evaluate_model(gelu_arr, softmax_arr)
        
        # 计算指标偏差百分比
        loss_diff_pct = (loss - self.baseline_loss) / (abs(self.baseline_loss) + 1e-8)
        p_diff_pct = (self.baseline_p - p) / (abs(self.baseline_p) + 1e-8)
        s_diff_pct = (self.baseline_s - s) / (abs(self.baseline_s) + 1e-8)
        
        max_dev = max(loss_diff_pct, p_diff_pct, s_diff_pct)
        
        # PDF 6.1：基于线性惩罚的软约束奖励（移除指数惩罚，防止梯度爆炸）
        if max_dev <= REWARD_TARGET:
            # 安全区域：给予正向激励，距离边界越远越好
            # safety_reward = 基础奖励 + 距离目标阈值的额外奖励
            safety_reward = REWARD_SAFETY_BONUS + (REWARD_TARGET - max_dev) * 100.0
        elif max_dev <= REWARD_THRESHOLD:
            # 缓冲区域（TARGET到THRESHOLD之间）：给予微小警示
            # 线性插值从0到-1
            buffer_penalty = -1.0 * (max_dev - REWARD_TARGET) / REWARD_SAFETY_BUFFER
            safety_reward = buffer_penalty
        else:
            # 危险区域：线性惩罚（PDF 6.1 - 替代指数惩罚）
            # 线性惩罚提供恒定梯度指引，告诉代理"你越界了，请往回走"
            # 而非指数惩罚的"你毁灭了"，后者会导致梯度爆炸
            violation = max_dev - REWARD_THRESHOLD
            safety_reward = -REWARD_PENALTY_SLOPE * violation
        
        # 计算成本奖励
        cost_saving = (self.baseline_cost - self.accumulated_cost) / self.baseline_cost
        cost_reward = cost_saving * REWARD_COST_WEIGHT
        
        # 原始奖励
        raw_reward = safety_reward + cost_reward
        
        # 策略三（5.1）：回报归一化 - 固定缩放
        # 将-100量级缩放到-5左右，使Critic能够有效学习
        scaled_reward = raw_reward / REWARD_NORMALIZATION_SCALE
        
        # PDF 6.1：奖励截断，防止任何单一step产生过大影响
        # 这是防止训练发散的最后一道防线
        clipped_reward = np.clip(scaled_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
        
        return clipped_reward


class LayerImportanceEvaluator(TrainerCallback):
    def __init__(self, model, train_data, test_data, data_collator, rl_lr=None, degree=None, device='cuda'):
        """
        基于 PPO 强化学习的策略搜索器。
        目标：在密文推理场景下，通过强化学习寻找最优的多项式近似策略。
        """
        # 增加递归深度以支持深拷贝复杂模型图
        sys.setrecursionlimit(50000)
        
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_collator = data_collator
        
        # 训练集用于RL训练，测试集用于最终评估
        self.dataloader_train = DataLoader(train_data, batch_size=16, shuffle=False, collate_fn=data_collator)
        self.dataloader_test = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=data_collator)
        
        try:
            self.reversible_handler = ReversibleLayerHandler(self.model)
        except Exception as e:
            print(f"[Warning] Deepcopy failed in handler init: {e}. 'restore_all' might fail.")
            self.reversible_handler = ReversibleLayerHandler(self.model)

        self.layers_attribute = self._detect_layer_attribute()
        self.total_layers = len(eval('self.model.' + self.layers_attribute))
        
        # --- 密文代价模型 ---
        self.GELU_COST_MAP = {4: 3.0, 2: 2.5, 1: 1.0}
        self.SOFTMAX_COST_MAP = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}

        # 搜索状态初始化
        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        self.current_softmax_degrees = np.full(self.total_layers, 6, dtype=int)
        
        # --- 用户阈值配置 (Strict 1.5%) ---
        self.error_threshold = 0.015
        self.correlation_drop_ratio = 0.015
        
        self.log_file = "pruning_search_log.txt"
        self.step_info_file = "ppo_step_info.txt"  # StepInfo 中间结果输出文件
        with open(self.log_file, "w") as f:
            f.write("=== PPO RL Optimization Log Started ===\n")
        
        # ==================== 策略二：动态超参数调度状态 ====================
        self.current_episode = 0
        self.total_episodes = PPO_MAX_EPISODES
        self.current_entropy_coef = PPO_ENTROPY_INITIAL
        self.current_lr = PPO_LR_INITIAL
        
        # ==================== PPO 7.1: 运行时回报归一化状态 ====================
        self.reward_history = []  # 历史回报滑动窗口
        self.reward_mean = 0.0    # 运行时均值
        self.reward_std = 1.0     # 运行时标准差（初始为1避免除零）
        
        # ==================== PDF 6.3: Return Normalization (PopArt风格) ====================
        # 使用 RunningMeanStd 跟踪 returns 的统计量
        # Critic 在归一化空间学习，推理时反归一化
        self.return_normalizer = RunningMeanStd()

    def _write_step_info(self, step_info, f):
        """将单步 StepInfo 写入文件"""
        f.write(f"  step_global: {step_info['step_global']}\n")
        f.write(f"  episode_id: {step_info['episode_id']}\n")
        f.write(f"  layer_index: {step_info['layer_index']}\n")
        f.write(f"  state_vector: {step_info['state_vector']}\n")
        f.write(f"  curr_gelu_degree: {step_info['curr_gelu_degree']}\n")
        f.write(f"  curr_softmax_degree: {step_info['curr_softmax_degree']}\n")
        f.write(f"  gelu_prob_dist: {step_info['gelu_prob_dist']}\n")
        f.write(f"  softmax_prob_dist: {step_info['softmax_prob_dist']}\n")
        f.write(f"  critic_value: {step_info['critic_value']}\n")
        f.write(f"  accumulated_cost: {step_info['accumulated_cost']}\n")
        f.write(f"  gelu_config: {step_info['gelu_config']}\n")
        f.write(f"  softmax_config: {step_info['softmax_config']}\n")
        # 策略二：输出动态超参数调度信息
        if 'current_lr' in step_info:
            f.write(f"  current_lr: {step_info['current_lr']:.6f}\n")
        if 'current_entropy_coef' in step_info:
            f.write(f"  current_entropy_coef: {step_info['current_entropy_coef']:.6f}\n")
    
    def update_hyperparameters(self, optimizer, episode):
        """
        PDF 6.4：简化超参数调度
        - 学习率：恒定（移除复杂Warmup和衰减，先跑通基线）
        - 熵系数：固定为0.01（移除激进衰减，防止探索不足）
        """
        self.current_episode = episode
        
        # PDF 6.4：使用恒定学习率
        # "建议将基础学习率从3e-4降低至1e-4或5e-5，并移除复杂的Warmup机制，先用恒定学习率跑通基线"
        new_lr = PPO_LR_INITIAL  # 恒定学习率
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        self.current_lr = new_lr
        
        # PDF 6.4：固定熵系数
        # "建议固定熵系数为0.01，或者仅在检测到KL散度过低时才衰减"
        new_entropy = PPO_ENTROPY_COEF_FIXED
        self.current_entropy_coef = new_entropy
        
        return new_lr, new_entropy
    
    def get_current_entropy_coef(self):
        """获取当前熵系数（供ppo_update使用）"""
        return self.current_entropy_coef
    
    def update_reward_statistics(self, episode_reward):
        """
        PPO 7.1: 更新运行时回报统计量
        维护滑动窗口的均值和标准差，用于回报归一化
        """
        # 将新回报加入历史
        self.reward_history.append(episode_reward)
        
        # 保持滑动窗口大小
        if len(self.reward_history) > RUNNING_REWARD_HISTORY_SIZE:
            self.reward_history.pop(0)
        
        # 更新均值和标准差（至少需要一定数量的样本）
        if len(self.reward_history) >= RUNNING_REWARD_MIN_SAMPLES:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + RUNNING_REWARD_EPSILON
    
    def normalize_reward(self, reward):
        """
        PPO 7.1: 对单个回报进行归一化
        在收集到足够样本后，使用运行时统计量进行归一化
        """
        # 如果样本不足，使用固定缩放（回退到策略三的方法）
        if len(self.reward_history) < RUNNING_REWARD_MIN_SAMPLES:
            return reward / REWARD_NORMALIZATION_SCALE
        
        # 使用运行时统计量进行标准化
        normalized = (reward - self.reward_mean) / self.reward_std
        return normalized
    
    def normalize_rewards_batch(self, rewards):
        """
        PPO 7.1: 对一批回报进行归一化（用于buffer中的rewards）
        """
        if len(self.reward_history) < RUNNING_REWARD_MIN_SAMPLES:
            # 样本不足时使用固定缩放
            return [r / REWARD_NORMALIZATION_SCALE for r in rewards]
        
        # 使用运行时统计量进行标准化
        return [(r - self.reward_mean) / self.reward_std for r in rewards]

    def _detect_layer_attribute(self):
        candidates = ['bert.encoder.layer', 'model.layers', 'transformer.h', 'roberta.encoder.layer']
        for path in candidates:
            try:
                if len(eval('self.model.' + path)) > 0: return path
            except: continue
        return 'bert.encoder.layer'

    def log(self, message):
        print(message, flush=True)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def get_simulated_cost(self, gelu_degrees, softmax_degrees):
        g_c = sum(self.GELU_COST_MAP.get(d, 0) for d in gelu_degrees)
        s_c = sum(self.SOFTMAX_COST_MAP.get(d, 0) for d in softmax_degrees)
        return g_c + s_c, g_c, s_c

    def apply_configuration(self, gelu_degrees, softmax_degrees):
        handler_layer_name = "model." + self.layers_attribute
        # GELU
        gelu_map = {d: [] for d in [1, 2, 4]} 
        for idx, deg in enumerate(gelu_degrees):
            if deg in gelu_map: gelu_map[deg].append(idx)
        for d in [1, 2, 4]:
            if gelu_map[d]:
                self.reversible_handler.replace_layer_gelu(gelu_map[d], handler_layer_name, degree=d)
        # Softmax
        softmax_map = {d: [] for d in range(2, 7)}
        for idx, deg in enumerate(softmax_degrees):
            if deg in softmax_map: softmax_map[deg].append(idx)
        for d in range(2, 7):
            if softmax_map[d]:
                self.reversible_handler.replace_layer_softmax(softmax_map[d], handler_layer_name, degree=d)

    def evaluate_model(self, gelu_degrees, softmax_degrees, use_train=True):
        """评估模型，use_train=True时使用训练集，否则使用测试集"""
        self.apply_configuration(gelu_degrees, softmax_degrees)
        self.model.eval()
        self.model.to(self.device)
        
        dataloader = self.dataloader_train if use_train else self.dataloader_test
        
        total_loss = 0.0
        all_preds, all_labels = [], []
        batch_times = []
        
        if torch.cuda.is_available():
            dummy = next(iter(dataloader))
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            with torch.no_grad(): _ = self.model(**dummy)
            torch.cuda.synchronize()

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"].detach().cpu().numpy()
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                start_time = time.time()
                outputs = self.model(**batch)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                batch_times.append((time.time() - start_time) * 1000.0)
                
                if outputs.loss is not None: total_loss += outputs.loss.item()
                logits = outputs.logits.squeeze().detach().cpu().numpy()
                if np.ndim(logits) == 0: logits = [logits]
                all_preds.extend(logits)
                all_labels.extend(labels)

        avg_loss = total_loss / len(dataloader)
        avg_time = sum(batch_times) / len(batch_times)
        try:
            p = pearsonr(all_preds, all_labels)[0]
            s = spearmanr(all_preds, all_labels)[0]
        except: p, s = 0.0, 0.0
            
        return avg_loss, p, s, avg_time

    def compute_gae(self, rewards, values, dones, gamma=PPO_GAMMA, lam=PPO_LAMBDA):
        """计算广义优势估计 (GAE)"""
        advantages = []
        gae = 0
        
        # 从后往前计算
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # 终止状态
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        
        return advantages, returns

    def ppo_update(self, policy_net, value_net, optimizer, buffer, device, mini_batch_size=PPO_MINI_BATCH_SIZE, entropy_coef=None):
        """
        PPO更新 - 包含Shuffle和Mini-batch（按照PDF 5.1节要求）
        策略二优化：支持动态熵系数
        PDF步骤7.3：调整mini-batch size以获得更稳定的梯度估计
        Args:
            policy_net: 策略网络
            value_net: 价值网络
            optimizer: 优化器
            buffer: 经验回放缓冲区
            device: 设备
            mini_batch_size: mini-batch大小（PDF建议调整）
            entropy_coef: 动态熵系数（策略二），如果为None则使用默认值
        """
        # 策略二：使用动态熵系数
        if entropy_coef is None:
            entropy_coef = self.get_current_entropy_coef()
        
        states, gelu_actions, softmax_actions, old_logprobs, rewards, dones, values = buffer.get_tensors(device)
        
        # 计算GAE（在原始尺度下计算）
        advantages, returns = self.compute_gae(rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy())
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # ==================== PDF 6.3: Return Normalization (PopArt风格) ====================
        # 步骤1：使用 RunningMeanStd 更新统计量（跨batch累积）
        self.return_normalizer.update(returns)
        
        # 步骤2：使用累积统计量归一化 returns
        # 这比 batch 内归一化更稳定，因为统计量是跨整个训练过程累积的
        returns_normalized = torch.tensor(
            self.return_normalizer.normalize(returns.cpu().numpy()),
            dtype=torch.float32
        ).to(device)
        
        # 步骤3：对采样时的 values 也用同样的统计量归一化
        # 这确保 old_values 和 new_values 在同一尺度下比较
        values_normalized = torch.tensor(
            self.return_normalizer.normalize(values.cpu().numpy()),
            dtype=torch.float32
        ).to(device)
        
        # 数据总量
        batch_size = states.size(0)
        
        # 用于记录最后一次更新的损失
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        
        # PPO更新K个epoch
        for epoch in range(PPO_K_EPOCHS):
            # Shuffle数据（按照PDF 5.1节要求）
            indices = torch.randperm(batch_size).to(device)
            
            # 按Mini-batch更新
            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                mb_indices = indices[start:end]
                
                # 获取mini-batch数据
                mb_states = states[mb_indices]
                mb_gelu_actions = gelu_actions[mb_indices]
                mb_softmax_actions = softmax_actions[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                # PDF 6.3: 使用 RunningMeanStd 归一化后的 returns
                mb_returns = returns_normalized[mb_indices]
                # PDF 6.3: 使用 RunningMeanStd 归一化后的旧 values
                mb_old_values_normalized = values_normalized[mb_indices]
                
                # 评估当前策略
                new_logprobs, entropy = policy_net.evaluate_actions(mb_states, mb_gelu_actions, mb_softmax_actions)
                new_values_raw = value_net(mb_states)
                
                # 计算ratio
                ratios = torch.exp(new_logprobs - mb_old_logprobs)
                
                # 计算surrogate loss（PPO-Clip目标）
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - PPO_EPS_CLIP, 1 + PPO_EPS_CLIP) * mb_advantages
                
                # 策略损失
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # PDF 6.3: 使用 RunningMeanStd 的统计量归一化 new_values
                # Critic 预测的是原始尺度的值，训练时归一化到与 returns 相同的空间
                new_values_normalized = (new_values_raw - self.return_normalizer.mean) / self.return_normalizer.std
                
                # PPO 5.2: Clipped Value Loss（价值函数裁剪）
                # 防止价值函数更新过大导致训练不稳定
                # value_clipped = old_value + clip(new_value - old_value, -ε, +ε)
                # PDF 6.3: 使用归一化后的 values 进行裁剪
                value_clipped = mb_old_values_normalized + torch.clamp(
                    new_values_normalized - mb_old_values_normalized, 
                    -VALUE_CLIP_RANGE, 
                    VALUE_CLIP_RANGE
                )
                
                # PDF 3.2.2 & 5.3: 使用 Huber Loss 替代 MSE Loss
                # Huber Loss 对异常值（Outlier）更鲁棒，能有效遏制 Value Loss 尖峰
                # delta=1.0 表示误差在 1.0 以内是平方损失，超过 1.0 是线性损失
                huber_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
                
                # 计算两种 Huber Loss：未裁剪的和裁剪的
                # PDF 6.3: 使用归一化后的values计算loss
                value_loss_unclipped = huber_loss_fn(new_values_normalized, mb_returns)
                value_loss_clipped = huber_loss_fn(value_clipped, mb_returns)
                # 取两者的最大值，确保价值函数不会离旧值太远
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # 熵正则项（鼓励探索）- 策略二：使用动态熵系数
                entropy_loss = -entropy.mean()
                
                # 总损失：L = L_policy + c1 * L_value + c2 * L_entropy
                # 策略二：使用动态entropy_coef
                loss = policy_loss + PPO_VALUE_COEF * value_loss + entropy_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(policy_net.parameters()) + list(value_net.parameters()), 0.5)
                optimizer.step()
                
                # 记录最后一次的损失值
                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = entropy.mean().item()
        
        return last_policy_loss, last_value_loss, last_entropy

    def on_evaluate(self, args, state, control, **kwargs):
        self.log("\n" + "="*60)
        self.log("STARTING PPO REINFORCEMENT LEARNING OPTIMIZATION")
        self.log("="*60)

        # ---------------------------------------------------------
        # Phase 1: Baseline (使用训练集)
        # ---------------------------------------------------------
        self.log("\n--- Phase 1: Establishing Baseline (on Training Set) ---")
        base_gelu = np.full(self.total_layers, 4, dtype=int)
        base_softmax = np.full(self.total_layers, 6, dtype=int)
        
        # 使用训练集计算baseline
        base_loss, base_p, base_s, base_time = self.evaluate_model(base_gelu, base_softmax, use_train=True)
        base_tot_c, base_g_c, base_s_c = self.get_simulated_cost(base_gelu, base_softmax)
        
        self.log(f"Baseline Metrics (Training Set):")
        self.log(f"  Loss: {base_loss:.6f}, P: {base_p:.6f}, S: {base_s:.6f}")
        self.log(f"  Sim Cost: {base_tot_c:.2f} (G={base_g_c:.2f}, S={base_s_c:.2f})")
        
        # Constraints
        limit_loss = base_loss + self.error_threshold
        limit_p = base_p * (1.0 - self.correlation_drop_ratio)
        limit_s = base_s * (1.0 - self.correlation_drop_ratio)
        
        self.log(f"Constraints: Loss<={limit_loss:.4f}, P>={limit_p:.4f}, S>={limit_s:.4f}")

        # ---------------------------------------------------------
        # Phase 2: PPO Training
        # ---------------------------------------------------------
        self.log("\n--- Phase 2: PPO Reinforcement Learning Training ---")
        
        # 初始化 StepInfo 输出文件
        with open(self.step_info_file, "w", encoding="utf-8") as f:
            f.write("=== PPO StepInfo 中间结果日志 ===\n")
            f.write("每步包含: step_global, episode_id, layer_index, state_vector, curr_gelu_degree, curr_softmax_degree, gelu_prob_dist, softmax_prob_dist, critic_value, accumulated_cost, gelu_config, softmax_config\n\n")
        
        # 初始化网络 - PDF网络优化方案
        # 使用 StateEncoder (Embedding + Transformer) + ResMLP 架构
        # - PolicyNetwork: StateEncoder(embed_dim=64) + 2个ResidualBlock(64->128)
        # - ValueNetwork: 独立StateEncoder(embed_dim=64) + 3个ResidualBlock(64->128)
        policy_net = PolicyNetwork(
            state_dim=STATE_DIM_TOTAL, 
            hidden_dim=64,           # 编码器输出维度（PDF建议）
            res_hidden_dim=128,      # 残差块隐藏维度
            num_layers=self.total_layers
        ).to(self.device)
        value_net = ValueNetwork(
            state_dim=STATE_DIM_TOTAL, 
            hidden_dim=64,           # 编码器输出维度（PDF建议）
            res_hidden_dim=128,      # 残差块隐藏维度
            num_layers=self.total_layers
        ).to(self.device)
        
        # 策略二：使用初始学习率
        optimizer = optim.Adam(
            list(policy_net.parameters()) + list(value_net.parameters()),
            lr=PPO_LR_INITIAL
        )
        
        # 初始化环境
        baseline_metrics = (base_loss, base_p, base_s)
        
        # 创建用于RL的评估器包装
        class RLEvaluatorWrapper:
            def __init__(wrapper_self, evaluator, use_train=True):
                wrapper_self.evaluator = evaluator
                wrapper_self.use_train = use_train
            
            def evaluate_model(wrapper_self, gelu_arr, softmax_arr):
                return wrapper_self.evaluator.evaluate_model(gelu_arr, softmax_arr, use_train=wrapper_self.use_train)
        
        rl_evaluator = RLEvaluatorWrapper(self, use_train=True)
        env = TransformerOptEnv(self.total_layers, base_tot_c, baseline_metrics, rl_evaluator)
        
        buffer = RolloutBuffer()
        
        # 记录最优解
        best_config = None
        best_reward = float('-inf')
        best_cost = float('inf')
        
        episode_rewards = []
        
        for episode in range(PPO_MAX_EPISODES):
            # 策略二：动态超参数调度（学习率和熵系数）
            current_lr, current_entropy = self.update_hyperparameters(optimizer, episode)
            
            state = env.reset()
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            episode_reward = 0
            step_infos = []  # 存储中间结果
            
            for step in range(self.total_layers):
                # 选择动作（同时获取概率分布用于记录中间结果）
                with torch.no_grad():
                    gelu_action, softmax_action, logprob, gelu_probs, softmax_probs = \
                        policy_net.get_action_and_logprob(state_tensor, return_probs=True)
                    value = value_net(state_tensor)
                
                # 执行动作
                next_state, reward, done, info = env.step(gelu_action.item(), softmax_action.item())
                
                # 记录中间结果（按照PDF 5.2节 StepInfo 结构要求）
                step_info = {
                    'step_global': episode * self.total_layers + step,  # 全局训练步数索引
                    'episode_id': episode,                               # 当前回合ID
                    'layer_index': info['layer_index'],                  # 当前决策的层 (0-11)
                    'state_vector': state.tolist(),                      # 输入状态向量 (17维，策略四增强)
                    'curr_gelu_degree': info['curr_gelu_degree'],        # 当前选择的GELU近似次数
                    'curr_softmax_degree': info['curr_softmax_degree'],  # 当前选择的Softmax近似次数
                    'gelu_prob_dist': gelu_probs.cpu().numpy().tolist(), # 策略网络输出的GELU概率分布
                    'softmax_prob_dist': softmax_probs.cpu().numpy().tolist(),  # 策略网络输出的Softmax概率分布
                    'critic_value': value.item(),                        # 价值网络预估的长期回报
                    'accumulated_cost': info['accumulated_cost'],        # 截止当前层的累积开销
                    'gelu_config': info['gelu_config'],                  # 当前各层GELU近似次数配置
                    'softmax_config': info['softmax_config'],            # 当前各层Softmax近似次数配置
                    'current_lr': current_lr,                            # 策略二：当前学习率
                    'current_entropy_coef': current_entropy              # 策略二：当前熵系数
                }
                step_infos.append(step_info)
                
                # 存入buffer
                buffer.add(
                    state_tensor.cpu(),
                    gelu_action.cpu(),
                    softmax_action.cpu(),
                    logprob.cpu(),
                    reward,
                    float(done),
                    value.cpu()
                )
                
                episode_reward += reward
                state = next_state
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
            
            episode_rewards.append(episode_reward)
            
            # PPO 7.1: 更新运行时回报统计量
            self.update_reward_statistics(episode_reward)
            
            # 将 StepInfo 中间结果输出到文件
            with open(self.step_info_file, "a", encoding="utf-8") as f:
                f.write(f"--- Episode {episode + 1} (Reward={episode_reward:.4f}) ---\n")
                for si in step_infos:
                    self._write_step_info(si, f)
                    f.write("\n")
            
            # 检查是否为最优解
            final_config = {
                'gelu': np.array(env.gelu_config),
                'softmax': np.array(env.softmax_config),
                'cost': env.accumulated_cost,
                'reward': episode_reward
            }
            
            if episode_reward > best_reward or (episode_reward == best_reward and env.accumulated_cost < best_cost):
                best_reward = episode_reward
                best_cost = env.accumulated_cost
                best_config = final_config.copy()
                self.log(f"  Episode {episode+1}: New Best! Reward={episode_reward:.4f}, Cost={env.accumulated_cost:.2f}")
                self.log(f"    GELU: {env.gelu_config}")
                self.log(f"    Softmax: {env.softmax_config}")
            
            # PPO更新（策略二：使用动态熵系数）
            if (episode + 1) % PPO_UPDATE_INTERVAL == 0:
                policy_loss, value_loss, entropy = self.ppo_update(
                    policy_net, value_net, optimizer, buffer, self.device,
                    entropy_coef=current_entropy  # 策略二：传入当前熵系数
                )
                buffer.clear()
                
                avg_reward = np.mean(episode_rewards[-PPO_UPDATE_INTERVAL:])
                # 策略二：日志输出当前学习率和熵系数
                self.log(f"  Episode {episode+1}: Avg Reward={avg_reward:.4f}, "
                        f"Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}, Entropy={entropy:.4f}")
                self.log(f"    [Dynamic Schedule] LR={current_lr:.6f}, Entropy Coef={current_entropy:.6f}")
        
        # 如果没有找到满足约束的解，使用baseline
        if best_config is None or best_reward < -50:  # 如果最好的奖励也很差，说明没找到可行解
            self.log("\nNo feasible solution found, using baseline configuration.")
            best_config = {
                'gelu': base_gelu.copy(),
                'softmax': base_softmax.copy(),
                'cost': base_tot_c,
                'reward': 0
            }
        
        self.log(f"\n--- PPO Training Completed ---")
        self.log(f"Best Configuration Found:")
        self.log(f"  GELU: {best_config['gelu'].tolist()}")
        self.log(f"  Softmax: {best_config['softmax'].tolist()}")
        self.log(f"  Cost: {best_config['cost']:.2f}, Reward: {best_config['reward']:.4f}")

        # ---------------------------------------------------------
        # Plot: PPO Training Curve (Episode Reward)
        # ---------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            episodes = np.arange(1, len(episode_rewards) + 1)
            rewards = np.array(episode_rewards, dtype=np.float32)

            # Simple moving average for smoother convergence view
            # 使用合理的窗口大小，避免窗口过大导致曲线过于平滑
            window = min(max(5, PPO_UPDATE_INTERVAL // 5), 50)
            if len(rewards) >= window:
                kernel = np.ones(window, dtype=np.float32) / window
                rewards_ma = np.convolve(rewards, kernel, mode="valid")
                episodes_ma = episodes[window - 1:]
            else:
                rewards_ma = rewards
                episodes_ma = episodes

            plt.figure(figsize=(10, 4))
            plt.plot(episodes, rewards, label="Episode Reward", alpha=0.6)
            plt.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2)
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("PPO Training Curve")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plot_path = "ppo_training_curve.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            self.log(f"PPO training curve saved to: {plot_path}")
        except Exception as e:
            self.log(f"[Warning] Failed to plot PPO training curve: {e}")

        # ---------------------------------------------------------
        # Phase 3: Final Report (使用测试集进行最终评估)
        # ---------------------------------------------------------
        self.log("\n" + "="*60)
        self.log("FINAL EVALUATION REPORT (on Test Set)")
        self.log("="*60)
        
        # 重新计算测试集上的baseline
        base_loss, base_p, base_s, base_time = self.evaluate_model(base_gelu, base_softmax, use_train=False)
        
        # 用于缓存评估结果
        eval_cache = {}
        eval_cache[(tuple(base_gelu), tuple(base_softmax))] = (base_loss, base_p, base_s, base_time)
        
        # Use Global Best State as the "Optimized" result
        opt_gelu = best_config['gelu']
        opt_softmax = best_config['softmax']
        
        # Helper to format results
        def get_result_dict(name, gelu_arr, softmax_arr):
            # Check cache first
            sig = (tuple(gelu_arr), tuple(softmax_arr))
            if sig in eval_cache:
                loss, p, s, t = eval_cache[sig]
            else:
                loss, p, s, t = self.evaluate_model(gelu_arr, softmax_arr, use_train=False)
                eval_cache[sig] = (loss, p, s, t)
            
            tot_c, g_c, s_c = self.get_simulated_cost(gelu_arr, softmax_arr)
            
            # Stats relative to baseline
            tot_spd = base_tot_c / (tot_c + 1e-6)
            g_spd = base_g_c / (g_c + 1e-6)
            s_spd = base_s_c / (s_c + 1e-6)
            
            return {
                'name': name, 'loss': loss, 'p': p, 's': s,
                'tot_c': tot_c, 'tot_spd': tot_spd,
                'g_c': g_c, 'g_spd': g_spd,
                's_c': s_c, 's_spd': s_spd,
                'gelu': gelu_arr, 'softmax': softmax_arr
            }

        opt_res = get_result_dict('Optimized (PPO)', opt_gelu, opt_softmax)
        
        # --- Random Logic ---
        def generate_cost_equivalent_config(target_cost, cost_map, length, rng):
            degrees = list(cost_map.keys())
            for _ in range(2000):
                cfg = rng.choice(degrees, size=length)
                for _ in range(500):
                    curr = sum(cost_map[d] for d in cfg)
                    diff = curr - target_cost
                    if abs(diff) < 1e-4: return cfg
                    idx = rng.integers(0, length)
                    old_v = cfg[idx]
                    moves = [d for d in degrees if abs((curr - cost_map[old_v] + cost_map[d]) - target_cost) < abs(diff)]
                    if moves: cfg[idx] = rng.choice(moves)
                    else: cfg[idx] = rng.choice(degrees)
            return rng.permutation(degrees[:length])

        rng = np.random.default_rng(42)
        random_results = []
        
        self.log("Generating 10 Permutations (Type 1)...")
        for i in range(10):
            random_results.append(get_result_dict(f'Perm_{i+1}', rng.permutation(opt_gelu), rng.permutation(opt_softmax)))
            
        self.log("Generating 10 Cost-Equivalent (Type 2)...")
        for i in range(10):
            r_g = generate_cost_equivalent_config(opt_res['g_c'], self.GELU_COST_MAP, self.total_layers, rng)
            r_s = generate_cost_equivalent_config(opt_res['s_c'], self.SOFTMAX_COST_MAP, self.total_layers, rng)
            random_results.append(get_result_dict(f'Equiv_{i+1}', r_g, r_s))

        # --- Output ---
        self.log("\nFinal Configurations Details:")
        self.log(f"[Optimized] GELU   : {opt_gelu.tolist()}")
        self.log(f"[Optimized] Softmax: {opt_softmax.tolist()}")
        self.log("\nRandom Configurations Details:")
        for res in random_results:
             self.log(f"[{res['name']}] GELU   : {res['gelu'].tolist()}")
             self.log(f"[{res['name']}] Softmax: {res['softmax'].tolist()}")

        self.log("\nPerformance Comparison Table:")
        header = f"{'Method':<15} | {'Loss':<6} {'Pear.':<6} {'Spear.':<6} | {'Tot C':<6} {'Tot S':<5} | {'GELU C':<6} {'GELU S':<6} | {'Smax C':<6} {'Smax S':<6}"
        self.log("-" * len(header))
        self.log(header)
        self.log("-" * len(header))
        
        self.log(f"{'Baseline':<15} | {base_loss:<6.4f} {base_p:<6.4f} {base_s:<6.4f} | "
                 f"{base_tot_c:<6.1f} {'1.0x':<5} | {base_g_c:<6.1f} {'1.0x':<6} | {base_s_c:<6.1f} {'1.0x':<6}")
        
        def format_row(r):
            return (f"{r['name']:<15} | {r['loss']:<6.4f} {r['p']:<6.4f} {r['s']:<6.4f} | "
                    f"{r['tot_c']:<6.1f} {r['tot_spd']:<5.2f} | "
                    f"{r['g_c']:<6.1f} {r['g_spd']:<6.2f} | "
                    f"{r['s_c']:<6.1f} {r['s_spd']:<6.2f}")
        
        self.log(format_row(opt_res))
        self.log("-" * len(header))
        for res in random_results: self.log(format_row(res))
        self.log("-" * len(header))

        # ---------------------------------------------------------
        # Phase 4: Sensitivity (Validation on Optimized)
        # ---------------------------------------------------------
        self.log("\n" + "="*60)
        self.log("PHASE 4: SENSITIVITY ANALYSIS (Validation on Optimized)")
        self.log("Verifying that any further single-layer downgrade from 'Optimized' violates constraints.")
        self.log("="*60)
        
        # Get metrics of the optimized state for delta calculation
        opt_loss = opt_res['loss']
        opt_p = opt_res['p']
        opt_s = opt_res['s']
        
        for i in range(self.total_layers):
            # GELU Check
            cd = opt_gelu[i]
            td = 2 if cd == 4 else (1 if cd == 2 else None)
            if td is not None:
                tmp = opt_gelu.copy()
                tmp[i] = td
                # Use cache if possible (might have been visited in beam search)
                sig = (tuple(tmp), tuple(opt_softmax))
                if sig in eval_cache: l, p, s, t = eval_cache[sig]
                else: l, p, s, t = self.evaluate_model(tmp, opt_softmax, use_train=False)
                
                is_viol = False
                viol_tags = []
                if l > limit_loss: 
                    is_viol = True
                    viol_tags.append("LOSS")
                if p < limit_p: 
                    is_viol = True
                    viol_tags.append("PEAR")
                if s < limit_s: 
                    is_viol = True
                    viol_tags.append("SPEAR")
                
                status = f"VIOLATED ({','.join(viol_tags)})" if is_viol else "SAFE"
                
                # Calculate Deltas relative to Optimized state
                d_l = l - opt_loss
                d_p = p - opt_p
                d_s = s - opt_s
                
                msg = (f"L{i} GELU {cd}->{td}: {status} | "
                       f"Loss: {l:.4f} ({d_l:+.4f}) | "
                       f"P: {p:.4f} ({d_p:+.4f}) | "
                       f"S: {s:.4f} ({d_s:+.4f})")
                self.log(msg)

            # Softmax Check
            cd_s = opt_softmax[i]
            if cd_s > 2:
                tmp_s = opt_softmax.copy()
                tmp_s[i] = cd_s - 1
                sig = (tuple(opt_gelu), tuple(tmp_s))
                if sig in eval_cache: l, p, s, t = eval_cache[sig]
                else: l, p, s, t = self.evaluate_model(opt_gelu, tmp_s, use_train=False)
                
                is_viol = False
                viol_tags = []
                if l > limit_loss: 
                    is_viol = True
                    viol_tags.append("LOSS")
                if p < limit_p: 
                    is_viol = True
                    viol_tags.append("PEAR")
                if s < limit_s: 
                    is_viol = True
                    viol_tags.append("SPEAR")
                
                status = f"VIOLATED ({','.join(viol_tags)})" if is_viol else "SAFE"
                
                # Calculate Deltas relative to Optimized state
                d_l = l - opt_loss
                d_p = p - opt_p
                d_s = s - opt_s
                
                msg = (f"L{i} Smax {cd_s}->{cd_s-1}: {status} | "
                       f"Loss: {l:.4f} ({d_l:+.4f}) | "
                       f"P: {p:.4f} ({d_p:+.4f}) | "
                       f"S: {s:.4f} ({d_s:+.4f})")
                self.log(msg)

        self.log("\nPPO Optimization Finished.")
        self.apply_configuration(opt_gelu, opt_softmax)
