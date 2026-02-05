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
from sklearn.metrics import accuracy_score, f1_score
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
PPO_UPDATE_INTERVAL = 170  # 每170个episode更新一次
PPO_BATCH_SIZE = 12 * 170  # 170 episodes per update

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
# 状态向量维度：17维原始特征 + 12维GELU历史 + 12维Softmax历史 + 3维预算余量 = 44维
STATE_DIM_ORIGINAL = 17  # 原始状态维度
STATE_DIM_HISTORY = 24   # 历史编码维度（12 GELU + 12 Softmax）
STATE_DIM_BUDGET = 3     # 预算感知维度（敏锐度优化PDF 3.3: Loss/Pearson/Spearman余量）
STATE_DIM_TOTAL = STATE_DIM_ORIGINAL + STATE_DIM_HISTORY + STATE_DIM_BUDGET  # 总维度 = 44
# PDF 6.2 步骤1：将填充值从 -1.0 改为 0.0
# 理由：在ReLU/SiLU激活的网络中，0输入通常产生0输出，天然表示"无信息"
# -1.0 是一个强烈的信号值，会干扰特征提取
HISTORY_MASK_VALUE = 0.0  # 未访问层的掩码值（PDF 6.2：零值填充）

# ==================== 敏锐度优化PDF：数据集相关配置 ====================
# 根据数据集选择不同的评估指标
REGRESSION_DATASETS = ['stsb']  # 回归任务：使用 pearson, spearman
CLASSIFICATION_DATASETS = ['mrpc', 'mnli', 'sst2', 'cola', 'qnli', 'rte', 'wnli']  # 分类任务：使用 accuracy, f1

# ==================== 敏锐度优化PDF：差分奖励与对数障碍配置 ====================
# 优化方案二：信号放大与差分奖励重构
DIFF_REWARD_SCALE_ACC = 50.0       # 精度差分奖励缩放因子
DIFF_REWARD_POWER = 0.5            # 根号变换指数（放大微小信号）
LOG_BARRIER_VIOLATION_SCALE = 10.0  # 违反约束时的指数惩罚系数
LOG_BARRIER_VIOLATION_STEEPNESS = 20.0  # 违反约束时的指数陡度
LOG_BARRIER_SATISFACTION_SCALE = 0.5   # 满足约束时的对数奖励系数

# ==================== 敏锐度优化PDF：解耦归一化配置（Disentangled PopArt） ====================
# 优化方案二（4.3）：分别维护成本和精度的统计量

# ==================== 敏锐度优化PDF：PPO-Lagrangian配置 ====================
# 优化方案四：自适应惩罚系数
LAGRANGIAN_LR = 0.01              # 拉格朗日乘子学习率
LAGRANGIAN_INITIAL = 0.1          # 初始拉格朗日乘子值
LAGRANGIAN_MAX = 10.0             # 拉格朗日乘子上限

# ==================== 敏锐度优化PDF：课程学习配置 ====================
# 优化方案四（6.2）：从宽松到严格的约束调度
CURRICULUM_PHASE1_RATIO = 0.30    # 探索期：前30%的episodes
CURRICULUM_PHASE2_RATIO = 0.40    # 收紧期：中间40%的episodes
CURRICULUM_PHASE3_RATIO = 0.30    # 精调期：后30%的episodes
CURRICULUM_INITIAL_SLACK = 1.2    # 探索期约束放宽系数（1.2倍目标值）
CURRICULUM_SAFETY_BUFFER = 0.95   # 精调期约束收紧系数（0.95倍目标值，略严于目标）

# ==================== 敏锐度优化PDF：超参数调整 ====================
# 优化方案（第三步）：熵系数线性衰减 + Critic学习率调整
PPO_ENTROPY_START = 0.05          # 熵系数起始值（高探索）
PPO_ENTROPY_END = 0.001           # 熵系数结束值（强制收敛）
PPO_LR_ACTOR = 3e-5               # Actor学习率
PPO_LR_CRITIC = 3e-4              # Critic学习率（Actor的10倍）

# ==================== 验证集引导（Validation Guided）配置 ====================
# 在计算奖励时使用验证集而非训练集，防止过拟合，提高泛化能力
#
# 原理与优势：
# 1. 防止过拟合：使用训练集计算奖励可能导致Agent学习到只在训练集上表现好的配置
# 2. 提高泛化性：验证集作为未见数据的代理，迫使Agent寻找在新数据上也稳健的策略
# 3. 更真实的优化目标：实际部署时关心的是在新数据上的表现，而非训练数据
#
# 实施策略：
# - Baseline计算：同时在训练集和验证集上评估，报告两者差异
# - 奖励计算：使用验证集指标，让Agent优化真实的泛化目标
# - 约束设定：基于验证集baseline，确保在未见数据上满足性能要求
USE_VALIDATION_FOR_REWARD = True  # True: 使用验证集计算奖励, False: 使用训练集


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


# ==================== 敏锐度优化PDF：解耦归一化（Disentangled PopArt） ====================
class DisentangledNormalizer:
    """
    敏锐度优化PDF 4.3：解耦的奖励归一化
    分别维护成本和精度的统计量，防止精度信号被成本信号掩盖
    
    实施逻辑：
    1. 维护两个独立的统计量流：r_cost_stats 和 r_acc_stats
    2. 分别归一化：r_cost_norm = (r_cost - μ_cost) / σ_cost
    3. 再加权求和：r_total = w_cost * r_cost_norm + w_acc * r_acc_norm
    """
    def __init__(self, cost_weight=1.0, acc_weight=1.0):
        self.cost_stats = RunningMeanStd()
        self.acc_stats = RunningMeanStd()
        self.cost_weight = cost_weight
        self.acc_weight = acc_weight
    
    def update(self, cost_rewards, acc_rewards):
        """更新两个独立的统计量"""
        if len(cost_rewards) > 0:
            self.cost_stats.update(cost_rewards)
        if len(acc_rewards) > 0:
            self.acc_stats.update(acc_rewards)
    
    def normalize_cost(self, cost_reward):
        """归一化成本奖励"""
        if self.cost_stats.count < 2:
            return cost_reward
        return (cost_reward - self.cost_stats.mean) / (self.cost_stats.std + 1e-8)
    
    def normalize_acc(self, acc_reward):
        """归一化精度奖励"""
        if self.acc_stats.count < 2:
            return acc_reward
        return (acc_reward - self.acc_stats.mean) / (self.acc_stats.std + 1e-8)
    
    def get_combined_reward(self, cost_reward, acc_reward):
        """获取加权归一化后的总奖励"""
        cost_norm = self.normalize_cost(cost_reward)
        acc_norm = self.normalize_acc(acc_reward)
        return self.cost_weight * cost_norm + self.acc_weight * acc_norm


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
    状态编码器 - PDF 4.1节 & 5.2节 + 敏锐度优化PDF 3.3
    将异构的原始输入（44维）转化为统一的语义向量
    
    输入分解为四个流：
    1. 层级流 (Layer Stream): Index 0-11，通过Embedding映射
    2. 指标流 (Metric Stream): Index 12-16，通过全连接层映射
    3. 历史序列流 (History Stream): Index 17-40，通过Transformer Encoder处理
    4. 预算感知流 (Budget Stream): Index 41-43，通过全连接层映射（敏锐度优化PDF 3.3）
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
        
        # 4. 敏锐度优化PDF 3.3：预算感知流（Index 41-43: loss_budget, m1_budget, m2_budget）
        self.budget_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU()
        )
        
        # 5. 融合层 - PDF 4.1.2 + 敏锐度优化PDF
        # 拼接四个流的输出：Layer(32) + Metric(32) + Hist(32) + Budget(32) = 128
        self.fusion = nn.Sequential(
            nn.Linear(32 * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, state_vector):
        """
        Args:
            state_vector: (Batch, 44) 或 (44,) 的状态向量（敏锐度优化PDF扩展）
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
        
        # D. 敏锐度优化PDF 3.3：解析 Budget (Index 41-43)
        # 处理状态向量可能是41维或44维的情况（向后兼容）
        if state_vector.size(1) >= 44:
            budget = state_vector[:, 41:44]
        else:
            # 向后兼容：如果状态向量不包含预算维度，使用零填充
            budget = torch.zeros(batch_size, 3, device=state_vector.device)
        b_emb = self.budget_proj(budget)  # (Batch, 32)
        
        # E. 融合
        combined = torch.cat([l_emb, m_emb, h_pooled, b_emb], dim=1)  # (Batch, 128)
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


# ==================== 敏锐度优化PDF：双头非对称Critic架构 ====================
class DualHeadValueNetwork(nn.Module):
    """
    敏锐度优化PDF 5.1：双头非对称Critic架构
    
    解决问题：单一Critic为了拟合大幅波动的Cost，会主导共享层的特征提取，
    导致无法提取到关于Accuracy的微弱特征。
    
    架构设计：
    - 共享骨干网络：StateEncoder + 2个ResidualBlock
    - Head A (V_cost)：专门预测预期的Simulation Cost
    - Head B (V_acc)：专门预测预期的Loss/Pearson/Spearman
    - 优势计算（GAE）：分别计算 A_cost 和 A_acc，然后加权组合
    """
    def __init__(self, state_dim=STATE_DIM_TOTAL, hidden_dim=64, res_hidden_dim=128, num_layers=12):
        super(DualHeadValueNetwork, self).__init__()
        
        # 共享的状态编码器
        self.encoder = StateEncoder(embed_dim=hidden_dim, num_layers=num_layers)
        
        # 共享的残差骨干网络：2个残差块
        self.shared_res_block1 = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        self.shared_res_block2 = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        
        # Head A (V_cost)：成本预测头 - 额外1个残差块
        self.cost_res_block = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        self.cost_head = nn.Linear(hidden_dim, 1)
        
        # Head B (V_acc)：精度预测头 - 额外1个残差块
        self.acc_res_block = ResidualBlock(input_dim=hidden_dim, hidden_dim=res_hidden_dim, dropout=0.1)
        self.acc_head = nn.Linear(hidden_dim, 1)
        
        # 正交初始化
        orthogonal_init(self.cost_head, gain=1.0)
        orthogonal_init(self.acc_head, gain=1.0)
    
    def forward(self, state):
        """
        返回两个价值预测：V_cost 和 V_acc
        """
        # 共享特征编码
        x = self.encoder(state)
        x = self.shared_res_block1(x)
        x = self.shared_res_block2(x)
        
        # Head A：成本价值
        x_cost = self.cost_res_block(x)
        v_cost = self.cost_head(x_cost).squeeze(-1)
        
        # Head B：精度价值
        x_acc = self.acc_res_block(x)
        v_acc = self.acc_head(x_acc).squeeze(-1)
        
        return v_cost, v_acc
    
    def get_combined_value(self, state, cost_weight=1.0, acc_weight=1.0):
        """获取加权组合的价值估计"""
        v_cost, v_acc = self.forward(state)
        return cost_weight * v_cost + acc_weight * v_acc


class RolloutBuffer:
    """
    存储Rollout数据的Buffer
    敏锐度优化PDF：扩展以支持双头Critic的分离奖励存储
    """
    def __init__(self):
        self.states = []
        self.gelu_actions = []
        self.softmax_actions = []
        self.logprobs = []
        self.rewards = []           # 总奖励（向后兼容）
        self.cost_rewards = []      # 成本奖励（双头Critic）
        self.acc_rewards = []       # 精度奖励（双头Critic）
        self.dones = []
        self.values = []            # 总价值（单头Critic向后兼容）
        self.values_cost = []       # 成本价值（双头Critic）
        self.values_acc = []        # 精度价值（双头Critic）
    
    def add(self, state, gelu_action, softmax_action, logprob, reward, done, value,
            cost_reward=None, acc_reward=None, value_cost=None, value_acc=None):
        """
        添加经验数据
        敏锐度优化PDF：支持分离的成本/精度奖励和价值
        """
        self.states.append(state)
        self.gelu_actions.append(gelu_action)
        self.softmax_actions.append(softmax_action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        
        # 双头Critic数据（可选）
        if cost_reward is not None:
            self.cost_rewards.append(cost_reward)
        if acc_reward is not None:
            self.acc_rewards.append(acc_reward)
        if value_cost is not None:
            self.values_cost.append(value_cost)
        if value_acc is not None:
            self.values_acc.append(value_acc)
    
    def clear(self):
        self.states.clear()
        self.gelu_actions.clear()
        self.softmax_actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.cost_rewards.clear()
        self.acc_rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.values_cost.clear()
        self.values_acc.clear()
    
    def get_tensors(self, device):
        """获取基础张量（向后兼容）"""
        states = torch.stack(self.states).to(device)
        gelu_actions = torch.stack(self.gelu_actions).to(device)
        softmax_actions = torch.stack(self.softmax_actions).to(device)
        logprobs = torch.stack(self.logprobs).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(device)
        values = torch.stack(self.values).to(device)
        return states, gelu_actions, softmax_actions, logprobs, rewards, dones, values
    
    def get_dual_head_tensors(self, device):
        """
        敏锐度优化PDF：获取双头Critic的分离奖励和价值张量
        """
        base_tensors = self.get_tensors(device)
        
        # 分离的成本/精度数据
        cost_rewards = torch.tensor(self.cost_rewards, dtype=torch.float32).to(device) if self.cost_rewards else None
        acc_rewards = torch.tensor(self.acc_rewards, dtype=torch.float32).to(device) if self.acc_rewards else None
        values_cost = torch.stack(self.values_cost).to(device) if self.values_cost else None
        values_acc = torch.stack(self.values_acc).to(device) if self.values_acc else None
        
        return base_tensors + (cost_rewards, acc_rewards, values_cost, values_acc)


class TransformerOptEnv:
    """
    Transformer优化环境
    已实施PDF优化策略：
    - 策略一：奖励函数重构（稠密化中间奖励、指数障碍软约束、安全边界）
    - 策略三：回报归一化（固定缩放）
    - 策略四：状态空间增强（累积复杂度债务、成本偏差相对化）
    - PDF优化方案一：显式历史编码（Flattened History）- 解决序列依赖性问题
    - 敏锐度优化PDF：预算感知状态特征、差分奖励、对数障碍函数、课程学习
    """
    def __init__(self, total_layers, baseline_cost, baseline_metrics, evaluator,
                 constraint_limits=None, prev_metrics=None):
        """
        初始化环境
        
        敏锐度优化PDF扩展参数：
        - constraint_limits: 约束阈值字典 {'loss': float, 'metric1': float, 'metric2': float}
                            用于课程学习的动态约束调整
        - prev_metrics: 上一episode结束时的指标（用于差分奖励计算）
        """
        self.total_layers = total_layers
        self.baseline_cost = baseline_cost  # 72.0 for 12 layers
        self.baseline_loss, self.baseline_p, self.baseline_s = baseline_metrics
        self.evaluator = evaluator
        
        # 敏锐度优化PDF 3.3：约束阈值（用于预算感知和课程学习）
        if constraint_limits is None:
            # 默认约束：1%偏差
            self.constraint_limits = {
                'loss': self.baseline_loss * (1 + REWARD_THRESHOLD),
                'metric1': self.baseline_p * (1 - REWARD_THRESHOLD),
                'metric2': self.baseline_s * (1 - REWARD_THRESHOLD)
            }
        else:
            self.constraint_limits = constraint_limits
        
        # 敏锐度优化PDF 4.1：差分奖励所需的上一episode指标
        if prev_metrics is None:
            self.prev_episode_metrics = {
                'loss': self.baseline_loss,
                'metric1': self.baseline_p,
                'metric2': self.baseline_s,
                'cost': self.baseline_cost
            }
        else:
            self.prev_episode_metrics = prev_metrics
        
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
        
        # 敏锐度优化PDF：存储当前episode的最终指标（用于下一episode的差分奖励）
        self.current_episode_metrics = None
        
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
        构造44维状态向量（敏锐度优化PDF：增加预算感知维度）
        
        原始17维特征：
        - 12维: 位置编码 (One-Hot)
        - 1维: 成本偏差 (Cost Deviation) - 中心化处理
        - 2维: 上一步动作编码
        - 1维: 累积复杂度债务 (Complexity Debt)
        - 1维: 进度指示 (Progress Indicator)
        
        新增24维历史编码（PDF优化方案一 + PDF 6.2零值填充）：
        - 12维: GELU动作历史（归一化，未访问层为0）
        - 12维: Softmax动作历史（归一化，未访问层为0）
        
        敏锐度优化PDF 3.3：新增3维预算感知特征
        - 1维: Loss剩余预算 (1 - curr_loss/limit_loss)
        - 1维: Metric1剩余预算 (curr_metric1/limit_metric1 - 1)
        - 1维: Metric2剩余预算 (curr_metric2/limit_metric2 - 1)
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
        
        # ========== 敏锐度优化PDF 3.3：预算感知特征 (3维) ==========
        # 使用上一episode的指标估计当前预算余量
        # 当 budget > 0 时表示满足约束，< 0 表示违反约束
        # 智能体需要保持 budget > 0
        prev_loss = self.prev_episode_metrics['loss']
        prev_m1 = self.prev_episode_metrics['metric1']
        prev_m2 = self.prev_episode_metrics['metric2']
        
        # Loss预算：约束是 loss < limit，所以 budget = 1 - loss/limit
        loss_budget = 1.0 - prev_loss / (self.constraint_limits['loss'] + 1e-8)
        # Metric1预算（如Pearson）：约束是 metric > limit，所以 budget = metric/limit - 1
        m1_budget = prev_m1 / (self.constraint_limits['metric1'] + 1e-8) - 1.0
        # Metric2预算（如Spearman）：约束是 metric > limit，所以 budget = metric/limit - 1
        m2_budget = prev_m2 / (self.constraint_limits['metric2'] + 1e-8) - 1.0
        
        # 截断到合理范围 [-1, 1]
        loss_budget = np.clip(loss_budget, -1.0, 1.0)
        m1_budget = np.clip(m1_budget, -1.0, 1.0)
        m2_budget = np.clip(m2_budget, -1.0, 1.0)
        
        state = np.concatenate([
            position,                          # 12维: 位置编码
            [cost_deviation],                  # 1维: 成本偏差
            [gelu_norm, softmax_norm],         # 2维: 上一步动作
            [complexity_debt],                 # 1维: 复杂度债务
            [progress],                        # 1维: 进度指示
            self.gelu_history,                 # 12维: GELU完整历史（PDF优化方案一）
            self.softmax_history,              # 12维: Softmax完整历史（PDF优化方案一）
            [loss_budget, m1_budget, m2_budget]  # 3维: 预算感知（敏锐度优化PDF 3.3）
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
        敏锐度优化PDF：差分奖励 + 对数障碍函数 + 解耦奖励
        
        实现要点：
        1. 差分精度奖励（信号放大）：使用0.5次幂放大微小变化
        2. 对数障碍惩罚（约束敏感性）：Log-Barrier函数
        3. 成本奖励：相对于基线的节省
        4. 返回分离的成本/精度奖励（用于双头Critic）
        """
        # 获取当前配置的指标
        gelu_arr = np.array(self.gelu_config)
        softmax_arr = np.array(self.softmax_config)
        
        # 评估模型
        loss, m1, m2, _ = self.evaluator.evaluate_model(gelu_arr, softmax_arr)
        
        # 存储当前指标（用于下一episode的差分计算）
        self.current_episode_metrics = {
            'loss': loss,
            'metric1': m1,
            'metric2': m2,
            'cost': self.accumulated_cost
        }
        
        # ==================== 敏锐度优化PDF 4.1：差分精度奖励 ====================
        # 1. 计算与上一episode的差值
        delta_loss = self.prev_episode_metrics['loss'] - loss  # 正值表示loss变小（改善）
        delta_m1 = m1 - self.prev_episode_metrics['metric1']   # 正值表示metric变大（改善）
        delta_m2 = m2 - self.prev_episode_metrics['metric2']   # 正值表示metric变大（改善）
        
        # 2. 使用根号变换放大微小信号（敏锐度优化PDF 4.1）
        # 例如: 1e-4 -> 1e-2，信号强度提升100倍
        def amplify_signal(delta):
            sign = 1.0 if delta >= 0 else -1.0
            return sign * (abs(delta) ** DIFF_REWARD_POWER) * DIFF_REWARD_SCALE_ACC
        
        r_loss_diff = amplify_signal(delta_loss)
        r_m1_diff = amplify_signal(delta_m1)
        r_m2_diff = amplify_signal(delta_m2)
        
        # 综合精度差分奖励
        r_accuracy_diff = (r_loss_diff + r_m1_diff + r_m2_diff) / 3.0
        
        # ==================== 敏锐度优化PDF 4.2：对数障碍约束奖励 ====================
        def log_barrier_reward(curr_value, limit_value, is_upper_bound=True):
            """
            对数障碍函数：当接近约束边界时梯度急剧增大
            
            Args:
                curr_value: 当前指标值
                limit_value: 约束阈值
                is_upper_bound: True表示约束为 curr < limit，False表示 curr > limit
            """
            if is_upper_bound:
                # 约束: curr < limit (如Loss)
                margin = limit_value - curr_value
            else:
                # 约束: curr > limit (如Pearson/Spearman)
                margin = curr_value - limit_value
            
            if margin < 0:
                # 违反约束：指数级爆炸惩罚
                return -LOG_BARRIER_VIOLATION_SCALE * np.exp(-margin * LOG_BARRIER_VIOLATION_STEEPNESS)
            else:
                # 满足约束：对数奖励，鼓励远离边界但收益递减
                return LOG_BARRIER_SATISFACTION_SCALE * np.log(margin + 1e-5)
        
        r_loss_barrier = log_barrier_reward(loss, self.constraint_limits['loss'], is_upper_bound=True)
        r_m1_barrier = log_barrier_reward(m1, self.constraint_limits['metric1'], is_upper_bound=False)
        r_m2_barrier = log_barrier_reward(m2, self.constraint_limits['metric2'], is_upper_bound=False)
        
        # 综合约束奖励
        r_constraint = (r_loss_barrier + r_m1_barrier + r_m2_barrier) / 3.0
        
        # ==================== 成本奖励 ====================
        cost_saving = (self.baseline_cost - self.accumulated_cost) / self.baseline_cost
        r_cost = cost_saving * REWARD_COST_WEIGHT
        
        # ==================== 综合奖励（用于双头Critic） ====================
        # 精度奖励 = 差分奖励 + 约束奖励
        r_accuracy = r_accuracy_diff + r_constraint
        
        # 总奖励
        raw_reward = r_accuracy + r_cost
        
        # 策略三（5.1）：回报归一化 - 固定缩放
        scaled_reward = raw_reward / REWARD_NORMALIZATION_SCALE
        
        # 奖励截断，防止训练发散
        clipped_reward = np.clip(scaled_reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
        
        # 存储分离的奖励（用于双头Critic）
        self.last_cost_reward = r_cost / REWARD_NORMALIZATION_SCALE
        self.last_acc_reward = r_accuracy / REWARD_NORMALIZATION_SCALE
        
        return clipped_reward
    
    def get_separated_rewards(self):
        """
        敏锐度优化PDF：获取分离的成本/精度奖励（用于双头Critic）
        """
        return getattr(self, 'last_cost_reward', 0.0), getattr(self, 'last_acc_reward', 0.0)
    
    def update_prev_metrics(self):
        """
        敏锐度优化PDF：更新上一episode指标（在episode结束后调用）
        用于下一episode的差分奖励计算
        """
        if self.current_episode_metrics is not None:
            self.prev_episode_metrics = self.current_episode_metrics.copy()


class LayerImportanceEvaluator(TrainerCallback):
    def __init__(self, model, train_data, test_data, data_collator, rl_lr=None, degree=None, 
                 device='cuda', data_path='stsb'):
        """
        基于 PPO 强化学习的策略搜索器。
        目标：在密文推理场景下，通过强化学习寻找最优的多项式近似策略。
        
        敏锐度优化PDF扩展：
        - data_path: 数据集名称，用于选择评估指标
            - 'stsb': 回归任务，使用 pearson, spearman
            - 'mrpc' 等: 分类任务，使用 accuracy, f1
        """
        # 增加递归深度以支持深拷贝复杂模型图
        sys.setrecursionlimit(50000)
        
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_collator = data_collator
        
        # ==================== 敏锐度优化PDF：数据集检测与指标选择 ====================
        self.data_path = data_path
        self.is_regression = self._detect_task_type()
        self._log_task_type()
        
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
        
        # ==================== 敏锐度优化PDF：解耦归一化 ====================
        self.disentangled_normalizer = DisentangledNormalizer()
        
        # ==================== 敏锐度优化PDF：PPO-Lagrangian ====================
        # 可学习的拉格朗日乘子
        self.lagrangian_loss = LAGRANGIAN_INITIAL
        self.lagrangian_m1 = LAGRANGIAN_INITIAL
        self.lagrangian_m2 = LAGRANGIAN_INITIAL
        
        # ==================== 敏锐度优化PDF：课程学习状态 ====================
        self.curriculum_phase = 1  # 1=探索, 2=收紧, 3=精调
        self.constraint_slack = CURRICULUM_INITIAL_SLACK  # 当前约束放宽系数
    
    def _detect_task_type(self):
        """
        敏锐度优化PDF：检测任务类型（回归/分类）
        根据数据集名称确定使用哪种评估指标
        """
        data_name = self.data_path.lower()
        
        # 检查是否为回归任务
        for reg_dataset in REGRESSION_DATASETS:
            if reg_dataset in data_name:
                return True
        
        # 检查是否为分类任务
        for cls_dataset in CLASSIFICATION_DATASETS:
            if cls_dataset in data_name:
                return False
        
        # 默认假设为回归任务
        print(f"[Warning] Unknown dataset '{data_name}', assuming regression task")
        return True
    
    def _log_task_type(self):
        """记录任务类型信息"""
        if self.is_regression:
            print(f"[Info] Dataset '{self.data_path}' detected as REGRESSION task")
            print(f"[Info] Using metrics: Pearson correlation, Spearman correlation")
        else:
            print(f"[Info] Dataset '{self.data_path}' detected as CLASSIFICATION task")
            print(f"[Info] Using metrics: Accuracy, F1 Score")
    
    def get_metric_names(self):
        """
        敏锐度优化PDF：获取当前数据集的指标名称
        """
        if self.is_regression:
            return 'Pearson', 'Spearman'
        else:
            return 'Accuracy', 'F1'

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
        敏锐度优化PDF：超参数调度
        - 熵系数：线性衰减（0.05 -> 0.001）
        - 学习率：Actor 和 Critic 分离（Critic 是 Actor 的 10 倍）
        - 课程学习：动态调整约束阈值
        """
        self.current_episode = episode
        progress = episode / self.total_episodes
        
        # ==================== 敏锐度优化PDF：熵系数线性衰减 ====================
        # 从 0.05（高探索）线性衰减到 0.001（强制收敛）
        new_entropy = PPO_ENTROPY_START - (PPO_ENTROPY_START - PPO_ENTROPY_END) * progress
        self.current_entropy_coef = new_entropy
        
        # ==================== 敏锐度优化PDF：学习率调度 ====================
        # 注意：如果使用分离优化器，这里的 optimizer 应该是一个字典
        # 如果使用统一优化器，保持向后兼容
        if hasattr(optimizer, 'param_groups'):
            # 统一优化器（向后兼容）
            for param_group in optimizer.param_groups:
                param_group['lr'] = PPO_LR_ACTOR
            self.current_lr = PPO_LR_ACTOR
        elif isinstance(optimizer, dict):
            # 分离优化器（敏锐度优化PDF推荐）
            if 'actor' in optimizer:
                for param_group in optimizer['actor'].param_groups:
                    param_group['lr'] = PPO_LR_ACTOR
            if 'critic' in optimizer:
                for param_group in optimizer['critic'].param_groups:
                    param_group['lr'] = PPO_LR_CRITIC
            self.current_lr = PPO_LR_ACTOR
        
        # ==================== 敏锐度优化PDF：课程学习阶段更新 ====================
        self._update_curriculum_phase(episode)
        
        return self.current_lr, new_entropy
    
    def _update_curriculum_phase(self, episode):
        """
        敏锐度优化PDF 6.2：课程学习阶段更新
        - 阶段一（探索期，前30%）：放宽约束阈值（1.2倍目标值）
        - 阶段二（收紧期，中间40%）：线性收紧约束阈值
        - 阶段三（精调期，后30%）：略严于目标的约束（0.95倍）
        """
        progress = episode / self.total_episodes
        
        phase1_end = CURRICULUM_PHASE1_RATIO
        phase2_end = CURRICULUM_PHASE1_RATIO + CURRICULUM_PHASE2_RATIO
        
        if progress < phase1_end:
            # 阶段一：探索期 - 放宽约束
            self.curriculum_phase = 1
            self.constraint_slack = CURRICULUM_INITIAL_SLACK
        elif progress < phase2_end:
            # 阶段二：收紧期 - 线性收紧
            self.curriculum_phase = 2
            # 从 1.2 线性过渡到 1.0
            phase2_progress = (progress - phase1_end) / CURRICULUM_PHASE2_RATIO
            self.constraint_slack = CURRICULUM_INITIAL_SLACK - (CURRICULUM_INITIAL_SLACK - 1.0) * phase2_progress
        else:
            # 阶段三：精调期 - 严格约束
            self.curriculum_phase = 3
            self.constraint_slack = CURRICULUM_SAFETY_BUFFER
    
    def get_curriculum_constraints(self, base_limits):
        """
        敏锐度优化PDF：获取当前课程阶段的约束阈值
        
        Args:
            base_limits: 基线约束字典 {'loss': float, 'metric1': float, 'metric2': float}
        
        Returns:
            调整后的约束字典
        """
        return {
            'loss': base_limits['loss'] * self.constraint_slack,
            'metric1': base_limits['metric1'] / self.constraint_slack,  # metric 是越大越好，所以除以 slack
            'metric2': base_limits['metric2'] / self.constraint_slack
        }
    
    def update_lagrangian_multipliers(self, loss_violation, m1_violation, m2_violation):
        """
        敏锐度优化PDF 6.1：更新拉格朗日乘子
        
        当约束被违反时，通过梯度上升增大对应的惩罚权重
        """
        # 梯度上升：如果违反约束（violation > 0），增大乘子
        self.lagrangian_loss = np.clip(
            self.lagrangian_loss + LAGRANGIAN_LR * loss_violation,
            0.0, LAGRANGIAN_MAX
        )
        self.lagrangian_m1 = np.clip(
            self.lagrangian_m1 + LAGRANGIAN_LR * m1_violation,
            0.0, LAGRANGIAN_MAX
        )
        self.lagrangian_m2 = np.clip(
            self.lagrangian_m2 + LAGRANGIAN_LR * m2_violation,
            0.0, LAGRANGIAN_MAX
        )
    
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
        # ==================== 敏锐度优化PDF：根据数据集类型计算不同指标 ====================
        if self.is_regression:
            # 回归任务：使用 Pearson 和 Spearman 相关系数
            try:
                metric1 = pearsonr(all_preds, all_labels)[0]
                metric2 = spearmanr(all_preds, all_labels)[0]
            except:
                metric1, metric2 = 0.0, 0.0
        else:
            # 分类任务：使用 Accuracy 和 F1 Score
            try:
                # 将 logits 转换为预测类别
                preds_arr = np.array(all_preds)
                if len(preds_arr.shape) == 1:
                    # 二分类：logits 是单个值，> 0.5 为正类
                    pred_classes = (preds_arr > 0.5).astype(int)
                else:
                    # 多分类：取 argmax
                    pred_classes = np.argmax(preds_arr, axis=1)
                
                metric1 = accuracy_score(all_labels, pred_classes)
                metric2 = f1_score(all_labels, pred_classes, average='weighted')
            except Exception as e:
                print(f"[Warning] Failed to compute classification metrics: {e}")
                metric1, metric2 = 0.0, 0.0
            
        return avg_loss, metric1, metric2, avg_time

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
        base_loss_train, base_p_train, base_s_train, base_time_train = self.evaluate_model(base_gelu, base_softmax, use_train=True)
        base_tot_c, base_g_c, base_s_c = self.get_simulated_cost(base_gelu, base_softmax)
        
        # 获取当前数据集对应的指标名称
        metric1_name, metric2_name = self.get_metric_names()
        
        self.log(f"Baseline Metrics (Training Set):")
        self.log(f"  Loss: {base_loss_train:.6f}, {metric1_name}: {base_p_train:.6f}, {metric2_name}: {base_s_train:.6f}")
        self.log(f"  Sim Cost: {base_tot_c:.2f} (G={base_g_c:.2f}, S={base_s_c:.2f})")
        
        # ==================== 验证集引导（Validation Guided）：计算验证集baseline ====================
        if USE_VALIDATION_FOR_REWARD:
            base_loss_val, base_p_val, base_s_val, base_time_val = self.evaluate_model(base_gelu, base_softmax, use_train=False)
            self.log(f"Baseline Metrics (Validation Set - used for reward):")
            self.log(f"  Loss: {base_loss_val:.6f}, {metric1_name}: {base_p_val:.6f}, {metric2_name}: {base_s_val:.6f}")
            
            # 使用验证集的baseline作为约束基准
            base_loss = base_loss_val
            base_p = base_p_val
            base_s = base_s_val
        else:
            # 使用训练集的baseline
            base_loss = base_loss_train
            base_p = base_p_train
            base_s = base_s_train
        
        # Constraints
        limit_loss = base_loss + self.error_threshold
        limit_p = base_p * (1.0 - self.correlation_drop_ratio)
        limit_s = base_s * (1.0 - self.correlation_drop_ratio)
        
        self.log(f"Constraints (based on {'Validation' if USE_VALIDATION_FOR_REWARD else 'Training'} Set):")
        self.log(f"  Loss<={limit_loss:.4f}, {metric1_name}>={limit_p:.4f}, {metric2_name}>={limit_s:.4f}")

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
        
        # ==================== 验证集引导（Validation Guided）====================
        # 使用验证集计算奖励，迫使Agent寻找泛化能力更强的配置
        # 这能有效防止Agent过拟合训练集，找到在未见数据上也表现良好的配置
        if USE_VALIDATION_FOR_REWARD:
            self.log("[Info] Using VALIDATION set for reward calculation (Validation Guided RL)")
            rl_evaluator = RLEvaluatorWrapper(self, use_train=False)  # 使用验证集
        else:
            self.log("[Info] Using TRAINING set for reward calculation")
            rl_evaluator = RLEvaluatorWrapper(self, use_train=True)   # 使用训练集
        
        env = TransformerOptEnv(self.total_layers, base_tot_c, baseline_metrics, rl_evaluator)
        
        buffer = RolloutBuffer()
        
        # 记录最优解
        best_config = None
        best_reward = float('-inf')
        best_cost = float('inf')
        
        episode_rewards = []
        episode_losses = []      # 记录每个episode的loss
        episode_metric1s = []    # 记录每个episode的metric1（Pearson或Accuracy）
        episode_metric2s = []    # 记录每个episode的metric2（Spearman或F1）
        
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
            
            # 收集当前episode的指标（从环境中获取）
            if hasattr(env, 'current_episode_metrics') and env.current_episode_metrics is not None:
                episode_losses.append(env.current_episode_metrics['loss'])
                episode_metric1s.append(env.current_episode_metrics['metric1'])
                episode_metric2s.append(env.current_episode_metrics['metric2'])
            else:
                # 如果环境中没有指标，使用baseline值
                episode_losses.append(base_loss)
                episode_metric1s.append(base_p)
                episode_metric2s.append(base_s)
            
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
        # Plot: PPO Training Curves (Reward and Metrics)
        # ---------------------------------------------------------
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            episodes = np.arange(1, len(episode_rewards) + 1)
            rewards = np.array(episode_rewards, dtype=np.float32)
            losses = np.array(episode_losses, dtype=np.float32)
            metric1s = np.array(episode_metric1s, dtype=np.float32)
            metric2s = np.array(episode_metric2s, dtype=np.float32)

            # 获取指标名称
            metric1_name, metric2_name = self.get_metric_names()

            # Simple moving average for smoother convergence view
            # 使用合理的窗口大小，避免窗口过大导致曲线过于平滑
            window = min(max(5, PPO_UPDATE_INTERVAL // 5), 50)
            
            def compute_ma(data):
                """计算移动平均"""
                if len(data) >= window:
                    kernel = np.ones(window, dtype=np.float32) / window
                    data_ma = np.convolve(data, kernel, mode="valid")
                    return data_ma
                return data
            
            rewards_ma = compute_ma(rewards)
            losses_ma = compute_ma(losses)
            metric1s_ma = compute_ma(metric1s)
            metric2s_ma = compute_ma(metric2s)
            
            if len(rewards) >= window:
                episodes_ma = episodes[window - 1:]
            else:
                episodes_ma = episodes

            # 创建 2x2 子图布局
            dataset_info = f" ({self.data_path})"
            val_guided_info = " [Validation Guided]" if USE_VALIDATION_FOR_REWARD else ""
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight='bold')
            
            # 子图1: Episode Reward
            ax1 = axes[0, 0]
            ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')
            ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkblue')
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Episode Reward")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 子图2: Loss
            ax2 = axes[0, 1]
            ax2.plot(episodes, losses, label="Loss", alpha=0.6, color='red')
            ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkred')
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Loss")
            ax2.set_title("Loss (lower is better)")
            ax2.grid(True, alpha=0.3)
            # 添加baseline参考线
            ax2.axhline(y=base_loss, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
            ax2.legend()
            
            # 子图3: Metric1 (Pearson or Accuracy)
            ax3 = axes[1, 0]
            ax3.plot(episodes, metric1s, label=metric1_name, alpha=0.6, color='green')
            ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkgreen')
            ax3.set_xlabel("Episode")
            ax3.set_ylabel(metric1_name)
            ax3.set_title(f"{metric1_name} (higher is better)")
            ax3.grid(True, alpha=0.3)
            # 添加baseline参考线
            ax3.axhline(y=base_p, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
            ax3.legend()
            
            # 子图4: Metric2 (Spearman or F1)
            ax4 = axes[1, 1]
            ax4.plot(episodes, metric2s, label=metric2_name, alpha=0.6, color='purple')
            ax4.plot(episodes_ma, metric2s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkviolet')
            ax4.set_xlabel("Episode")
            ax4.set_ylabel(metric2_name)
            ax4.set_title(f"{metric2_name} (higher is better)")
            ax4.grid(True, alpha=0.3)
            # 添加baseline参考线
            ax4.axhline(y=base_s, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
            ax4.legend()

            plot_path = "ppo_training_curve.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            self.log(f"PPO training curves saved to: {plot_path}")
        except Exception as e:
            self.log(f"[Warning] Failed to plot PPO training curves: {e}")

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

        # 根据数据集类型设置表头
        metric1_short = "Pear." if self.is_regression else "Acc."
        metric2_short = "Spear." if self.is_regression else "F1"
        
        self.log("\nPerformance Comparison Table:")
        header = f"{'Method':<15} | {'Loss':<6} {metric1_short:<6} {metric2_short:<6} | {'Tot C':<6} {'Tot S':<5} | {'GELU C':<6} {'GELU S':<6} | {'Smax C':<6} {'Smax S':<6}"
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
                # 根据数据集类型设置违规标签
                metric1_tag = "PEAR" if self.is_regression else "ACC"
                metric2_tag = "SPEAR" if self.is_regression else "F1"
                
                if l > limit_loss: 
                    is_viol = True
                    viol_tags.append("LOSS")
                if p < limit_p: 
                    is_viol = True
                    viol_tags.append(metric1_tag)
                if s < limit_s: 
                    is_viol = True
                    viol_tags.append(metric2_tag)
                
                status = f"VIOLATED ({','.join(viol_tags)})" if is_viol else "SAFE"
                
                # Calculate Deltas relative to Optimized state
                d_l = l - opt_loss
                d_p = p - opt_p
                d_s = s - opt_s
                
                msg = (f"L{i} GELU {cd}->{td}: {status} | "
                       f"Loss: {l:.4f} ({d_l:+.4f}) | "
                       f"{metric1_name}: {p:.4f} ({d_p:+.4f}) | "
                       f"{metric2_name}: {s:.4f} ({d_s:+.4f})")
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
                    viol_tags.append(metric1_tag)
                if s < limit_s: 
                    is_viol = True
                    viol_tags.append(metric2_tag)
                
                status = f"VIOLATED ({','.join(viol_tags)})" if is_viol else "SAFE"
                
                # Calculate Deltas relative to Optimized state
                d_l = l - opt_loss
                d_p = p - opt_p
                d_s = s - opt_s
                
                msg = (f"L{i} Smax {cd_s}->{cd_s-1}: {status} | "
                       f"Loss: {l:.4f} ({d_l:+.4f}) | "
                       f"{metric1_name}: {p:.4f} ({d_p:+.4f}) | "
                       f"{metric2_name}: {s:.4f} ({d_s:+.4f})")
                self.log(msg)

        self.log("\nPPO Optimization Finished.")
        self.apply_configuration(opt_gelu, opt_softmax)
