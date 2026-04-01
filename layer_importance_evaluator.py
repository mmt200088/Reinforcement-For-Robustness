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
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from function_handler import (
    ReversibleLayerHandler,
    INPUT_NOISE_ALLOWED_SCALING_FACTORS,
    INPUT_NOISE_DEFAULT_SCALING_FACTOR,
    WEIGHT_NOISE_ALLOWED_SCALING_FACTORS,
    WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
    WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
    WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
)
from final_evaluation_module import FinalEvaluationModule
import os
import hashlib

# ==================== PPO 常量与配置定义 ====================
# 动作映射表（严格按照任务要求）
GELU_MAP = {0: 4, 1: 2, 2: 1, 3: 0}
GELU_COST = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
SOFTMAX_MAP = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2}
SOFTMAX_COST = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}

# Action space for the current second-stage RL over transformer-layer x noise.
# Stage 1 still optimizes GELU/Softmax; Stage 2 keeps them fixed and chooses
# layer-wise scaling factors for x/Wq/Wk/Wv/Wo/Wffn1/Wffn2 noise.
INPUT_NOISE_SCALING_MAP = {
    idx: scaling_factor for idx, scaling_factor in enumerate(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
}
INPUT_NOISE_SCALING_TO_ACTION = {
    scaling_factor: idx for idx, scaling_factor in INPUT_NOISE_SCALING_MAP.items()
}
INPUT_NOISE_ACTION_DIM = len(INPUT_NOISE_SCALING_MAP)
INPUT_NOISE_SOS_TOKEN = INPUT_NOISE_ACTION_DIM
INPUT_NOISE_COST_SCALE = 0.025
INPUT_NOISE_COST_MAP = {
    scaling_factor: scaling_factor * INPUT_NOISE_COST_SCALE
    for scaling_factor in INPUT_NOISE_ALLOWED_SCALING_FACTORS
}
INPUT_NOISE_SCALING_TO_NORM = {
    scaling_factor: idx / max(1, INPUT_NOISE_ACTION_DIM - 1)
    for idx, scaling_factor in enumerate(sorted(INPUT_NOISE_ALLOWED_SCALING_FACTORS, reverse=True))
}

WEIGHT_NOISE_ACTION_DIM = len(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
WEIGHT_NOISE_SOS_TOKEN = WEIGHT_NOISE_ACTION_DIM
WEIGHT_NOISE_COST_SCALE = 0.025
WEIGHT_NOISE_COST_MAP = {
    scaling_factor: scaling_factor * WEIGHT_NOISE_COST_SCALE
    for scaling_factor in WEIGHT_NOISE_ALLOWED_SCALING_FACTORS
}
WEIGHT_NOISE_SCALING_TO_NORM = {
    scaling_factor: idx / max(1, WEIGHT_NOISE_ACTION_DIM - 1)
    for idx, scaling_factor in enumerate(sorted(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS, reverse=True))
}

WFFN1_NOISE_COST_MAP = {
    scaling_factor: scaling_factor * WEIGHT_NOISE_COST_SCALE
    for scaling_factor in WFFN1_NOISE_ALLOWED_SCALING_FACTORS
}
WFFN1_NOISE_SCALING_TO_NORM = {
    scaling_factor: idx / max(1, len(WFFN1_NOISE_ALLOWED_SCALING_FACTORS) - 1)
    for idx, scaling_factor in enumerate(sorted(WFFN1_NOISE_ALLOWED_SCALING_FACTORS, reverse=True))
}

WQ_NOISE_SCALING_MAP = {
    idx: scaling_factor for idx, scaling_factor in enumerate(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
}
WQ_NOISE_SCALING_TO_ACTION = {
    scaling_factor: idx for idx, scaling_factor in WQ_NOISE_SCALING_MAP.items()
}
WQ_NOISE_ACTION_DIM = len(WQ_NOISE_SCALING_MAP)

WK_NOISE_SCALING_MAP = {
    idx: scaling_factor for idx, scaling_factor in enumerate(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
}
WK_NOISE_SCALING_TO_ACTION = {
    scaling_factor: idx for idx, scaling_factor in WK_NOISE_SCALING_MAP.items()
}
WK_NOISE_ACTION_DIM = len(WK_NOISE_SCALING_MAP)

WV_NOISE_SCALING_MAP = {
    idx: scaling_factor for idx, scaling_factor in enumerate(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
}
WV_NOISE_SCALING_TO_ACTION = {
    scaling_factor: idx for idx, scaling_factor in WV_NOISE_SCALING_MAP.items()
}
WV_NOISE_ACTION_DIM = len(WV_NOISE_SCALING_MAP)

WO_NOISE_SCALING_MAP = {
    idx: scaling_factor for idx, scaling_factor in enumerate(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
}
WO_NOISE_SCALING_TO_ACTION = {
    scaling_factor: idx for idx, scaling_factor in WO_NOISE_SCALING_MAP.items()
}
WO_NOISE_ACTION_DIM = len(WO_NOISE_SCALING_MAP)

WFFN1_NOISE_SCALING_MAP = {
    idx: scaling_factor for idx, scaling_factor in enumerate(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
}
WFFN1_NOISE_SCALING_TO_ACTION = {
    scaling_factor: idx for idx, scaling_factor in WFFN1_NOISE_SCALING_MAP.items()
}
WFFN1_NOISE_ACTION_DIM = len(WFFN1_NOISE_SCALING_MAP)

WFFN2_NOISE_SCALING_MAP = {
    idx: scaling_factor for idx, scaling_factor in enumerate(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
}
WFFN2_NOISE_SCALING_TO_ACTION = {
    scaling_factor: idx for idx, scaling_factor in WFFN2_NOISE_SCALING_MAP.items()
}
WFFN2_NOISE_ACTION_DIM = len(WFFN2_NOISE_SCALING_MAP)

NOISE_STAGE_NUM_ACTIONS = 7
NOISE_STAGE_SOS_TOKENS = (
    INPUT_NOISE_SOS_TOKEN,
    WEIGHT_NOISE_SOS_TOKEN, WEIGHT_NOISE_SOS_TOKEN, WEIGHT_NOISE_SOS_TOKEN,
    WEIGHT_NOISE_SOS_TOKEN, WEIGHT_NOISE_SOS_TOKEN, WEIGHT_NOISE_SOS_TOKEN,
)
NOISE_STAGE_MAX_SOS_TOKEN = max(INPUT_NOISE_SOS_TOKEN, WEIGHT_NOISE_SOS_TOKEN)
NOISE_STAGE_ACTION_DIMS = (
    INPUT_NOISE_ACTION_DIM,
    WQ_NOISE_ACTION_DIM, WK_NOISE_ACTION_DIM, WV_NOISE_ACTION_DIM,
    WO_NOISE_ACTION_DIM, WFFN1_NOISE_ACTION_DIM, WFFN2_NOISE_ACTION_DIM,
)
NOISE_STAGE_CONT_DIM = 6
NOISE_STAGE_PREV_ACTION_EMBED_DIM = 4
NOISE_STAGE_STEP_INFO_FILE = "noise_ppo_step_info.txt"
NOISE_STAGE_TRAINING_CURVE_PATH = "noise_ppo_training_curve.png"
NOISE_STAGE_ENTROPY_CURVE_PATH = "noise_ppo_entropy_curve.png"
DEFAULT_STAGE1_SEARCH_LOG_FILE = "pruning_search_log.txt"
DEFAULT_STAGE1_STEP_INFO_FILE = "ppo_step_info.txt"
DEFAULT_STAGE1_TRAINING_CURVE_FILE = "ppo_training_curve.png"
DEFAULT_STAGE1_ENTROPY_CURVE_FILE = "ppo_entropy_curve.png"
DEFAULT_STAGE1_FINAL_EVAL_DIR = os.path.join("rl_results", "final_evaluation")
DEFAULT_NOISE_PROGRESS_DIR = os.path.join("rl_results", "noise_rl_progress")
DEFAULT_NOISE_FINAL_EVAL_DIR = os.path.join("rl_results", "noise_final_evaluation")


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def resolve_run_output_layout(run_output_dir):
    run_output_dir = str(run_output_dir or "").strip()
    if not run_output_dir:
        return {
            "run_output_dir": "",
            "log_file": DEFAULT_STAGE1_SEARCH_LOG_FILE,
            "noise_log_file": DEFAULT_STAGE1_SEARCH_LOG_FILE,
            "stage1_step_info_file": DEFAULT_STAGE1_STEP_INFO_FILE,
            "stage1_training_curve_path": DEFAULT_STAGE1_TRAINING_CURVE_FILE,
            "stage1_entropy_curve_path": DEFAULT_STAGE1_ENTROPY_CURVE_FILE,
            "stage1_final_eval_dir": DEFAULT_STAGE1_FINAL_EVAL_DIR,
            "noise_step_info_file": NOISE_STAGE_STEP_INFO_FILE,
            "noise_training_curve_path": NOISE_STAGE_TRAINING_CURVE_PATH,
            "noise_entropy_curve_path": NOISE_STAGE_ENTROPY_CURVE_PATH,
            "noise_progress_dir": DEFAULT_NOISE_PROGRESS_DIR,
            "noise_final_eval_dir": DEFAULT_NOISE_FINAL_EVAL_DIR,
        }

    run_output_dir = os.path.normpath(run_output_dir)
    stage1_dir = os.path.join(run_output_dir, "stage1")
    stage1_final_eval_dir = os.path.join(run_output_dir, "stage1_final_eval")
    stage2_noise_dir = os.path.join(run_output_dir, "stage2_noise")
    stage2_noise_progress_dir = os.path.join(stage2_noise_dir, "progress")
    stage2_noise_final_eval_dir = os.path.join(run_output_dir, "stage2_noise_final_eval")

    for dir_path in (
        run_output_dir,
        os.path.join(run_output_dir, "logs"),
        stage1_dir,
        stage1_final_eval_dir,
        stage2_noise_dir,
        stage2_noise_progress_dir,
        stage2_noise_final_eval_dir,
    ):
        os.makedirs(dir_path, exist_ok=True)

    layout = {
        "run_output_dir": run_output_dir,
        "log_file": os.path.join(stage1_dir, DEFAULT_STAGE1_SEARCH_LOG_FILE),
        "noise_log_file": os.path.join(stage2_noise_dir, DEFAULT_STAGE1_SEARCH_LOG_FILE),
        "stage1_step_info_file": os.path.join(stage1_dir, DEFAULT_STAGE1_STEP_INFO_FILE),
        "stage1_training_curve_path": os.path.join(
            stage1_dir, DEFAULT_STAGE1_TRAINING_CURVE_FILE
        ),
        "stage1_entropy_curve_path": os.path.join(
            stage1_dir, DEFAULT_STAGE1_ENTROPY_CURVE_FILE
        ),
        "stage1_final_eval_dir": stage1_final_eval_dir,
        "noise_step_info_file": os.path.join(stage2_noise_dir, NOISE_STAGE_STEP_INFO_FILE),
        "noise_training_curve_path": os.path.join(
            stage2_noise_dir, NOISE_STAGE_TRAINING_CURVE_PATH
        ),
        "noise_entropy_curve_path": os.path.join(
            stage2_noise_dir, NOISE_STAGE_ENTROPY_CURVE_PATH
        ),
        "noise_progress_dir": stage2_noise_progress_dir,
        "noise_final_eval_dir": stage2_noise_final_eval_dir,
    }

    for path_key in (
        "log_file",
        "noise_log_file",
        "stage1_step_info_file",
        "stage1_training_curve_path",
        "stage1_entropy_curve_path",
        "noise_step_info_file",
        "noise_training_curve_path",
        "noise_entropy_curve_path",
    ):
        ensure_parent_dir(layout[path_key])

    return layout

# GELU degree 0 资格阈值：[-2.7, 0) 区间占比 >= 此阈值才能使用 degree 0
GELU_DEGREE0_THRESHOLD = 2.0 # 设为 2.0（不可能达到）以完全禁用 degree 0 动作 一般设置为 0.80

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
PPO_MAX_EPISODES = 51000
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
STATE_DIM_BUDGET = 3     # 预算感知维度（敏锐度优化PDF 3.3: Loss/Metric1/Metric2余量）
STATE_DIM_TOTAL = STATE_DIM_ORIGINAL + STATE_DIM_HISTORY + STATE_DIM_BUDGET  # 总维度 = 44
# PDF 6.2 步骤1：将填充值从 -1.0 改为 0.0
# 理由：在ReLU/SiLU激活的网络中，0输入通常产生0输出，天然表示"无信息"
# -1.0 是一个强烈的信号值，会干扰特征提取
HISTORY_MASK_VALUE = 0.0  # 未访问层的掩码值（PDF 6.2：零值填充）

# ==================== LSTM 策略价值网络配置（LSTM PDF优化方案） ====================
LSTM_HIDDEN_DIM = 128        # LSTM隐藏层维度（PDF 3.2：128足以编码12步历史）
LSTM_NUM_LAYERS = 2          # LSTM堆叠层数（PDF 3.2：双层增强表达能力）
LSTM_DROPOUT = 0.1           # LSTM层间Dropout（PDF 3.2）
LSTM_CONT_DIM = 6            # 连续特征维度（cost_deviation + complexity_debt + progress + 3 budget）
LSTM_POS_DIM = 16            # 层级位置嵌入维度（PDF 3.1.1）
LSTM_ACT_G_DIM = 8           # GELU动作嵌入维度（PDF 3.1.2）
LSTM_ACT_S_DIM = 8           # Softmax动作嵌入维度（PDF 3.1.2）
LSTM_PROJ_DIM = 32           # 连续特征投影维度（PDF 3.1.3）
SOS_TOKEN_GELU = 4           # GELU前一动作的SOS标记（词表大小=5: 0,1,2,3为动作, 4为SOS）
SOS_TOKEN_SOFTMAX = 5        # Softmax前一动作的SOS标记（词表大小=6: 0-4为动作, 5为SOS）
LSTM_MINI_BATCH_EPISODES = 8 # LSTM PPO更新时的mini-batch大小（按episode数）

# ==================== GTrXL 策略价值网络配置（Transformer PDF优化方案） ====================
GTRXL_D_MODEL = 64              # Token 维度（与 LSTM 输入维度一致：16+8+8+32=64）
GTRXL_N_HEADS = 4               # 多头注意力头数（64/4=16 per head）
GTRXL_N_LAYERS = 3              # GTrXL Block 堆叠层数
GTRXL_D_FF = 128                # 前馈网络隐藏维度（2x d_model）
GTRXL_DROPOUT = 0.1             # Dropout 率
GTRXL_WARMUP_STEPS = 500        # 学习率线性预热步数（PPO update 计数）
GTRXL_ENTROPY_LOWER_BOUND = 0.005  # 最小熵约束下界（防止 Mode Collapse）
GTRXL_MINI_BATCH_EPISODES = 8   # GTrXL PPO更新时的mini-batch大小（按episode数）

# ==================== 敏锐度优化PDF：数据集相关配置 ====================
# 根据数据集选择不同的评估指标（细分版）
# type: regression / classification
# metrics: 该数据集使用的指标列表
# metric_names: 显示用的指标短名 (表格用)
# metric_full_names: 完整指标名 (日志用)
def resolve_ppo_learning_rate(raw_value, default_lr=PPO_LR_INITIAL):
    """Resolve the user-facing rl_lr argument into a PPO optimizer LR.

    Compatibility rules:
    - `None` or empty: use the built-in PPO default.
    - `0 < rl_lr < 1`: treat it as a direct optimizer learning rate.
    - legacy integers such as 20 / 40: interpret them as 20e-6 / 40e-6 so
      existing shell commands remain meaningful under the PPO implementation.
    """
    if raw_value is None or raw_value == "":
        return float(default_lr), "default"

    try:
        lr = float(raw_value)
    except (TypeError, ValueError):
        return float(default_lr), "default"

    if lr <= 0:
        return float(default_lr), "default"
    if lr < 1.0:
        return float(lr), "direct"
    return float(lr) * 1e-6, "legacy-micro"


DATASET_METRICS_CONFIG = {
    'sst2':  {'type': 'classification', 'metrics': ['accuracy'],
              'metric_names': ['Acc.'], 'metric_full_names': ['Accuracy']},
    'qnli':  {'type': 'classification', 'metrics': ['accuracy'],
              'metric_names': ['Acc.'], 'metric_full_names': ['Accuracy']},
    'mnli':  {'type': 'classification', 'metrics': ['matched_accuracy', 'mismatched_accuracy'],
              'metric_names': ['M-Acc.', 'MM-Acc.'], 'metric_full_names': ['Matched Accuracy', 'Mismatched Accuracy']},
    'cola':  {'type': 'classification', 'metrics': ['mcc'],
              'metric_names': ['MCC'], 'metric_full_names': ['Matthews Correlation']},
    'stsb':  {'type': 'regression', 'metrics': ['pearson', 'spearman'],
              'metric_names': ['Pear.', 'Spear.'], 'metric_full_names': ['Pearson', 'Spearman']},
    'mrpc':  {'type': 'classification', 'metrics': ['accuracy', 'f1'],
              'metric_names': ['Acc.', 'F1'], 'metric_full_names': ['Accuracy', 'F1']},
    'rte':   {'type': 'classification', 'metrics': ['accuracy'],
              'metric_names': ['Acc.'], 'metric_full_names': ['Accuracy']},
    'wnli':  {'type': 'classification', 'metrics': ['accuracy'],
              'metric_names': ['Acc.'], 'metric_full_names': ['Accuracy']},
}
# 向后兼容
REGRESSION_DATASETS = ['stsb']
CLASSIFICATION_DATASETS = ['mrpc', 'mnli', 'sst2', 'cola', 'qnli', 'rte', 'wnli']

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

# ==================== 搜索模式与贪心配置 ====================
# SEARCH_MODE: 选择优化方式
#   - "rl": 仅强化学习（Phase 2），不执行贪心
#   - "greedy": 仅贪婪算法（从下方 GREEDY_INITIAL_* 指定的配置出发），不执行 RL
#   - "both": 先 RL 再根据 USE_GREEDY_SEARCH 决定是否执行贪心
USE_VALIDATION_SEARCH_PROTOCOL = True
VALIDATION_SMALL_TASKS = {"mrpc", "rte", "cola", "wnli"}
VALIDATION_HOLDOUT_RATIO_DEFAULT = 0.20
VALIDATION_HOLDOUT_RATIO_SMALL = 0.30
VALIDATION_PROXY_RATIO_DEFAULT = 0.15
VALIDATION_PROXY_RATIO_SMALL = 0.50
RL_DATASET_SPLIT_SEED = 42

USE_TRAIN_ANCHOR = False
TRAIN_ANCHOR_SIZE_DEFAULT = 256
TRAIN_ANCHOR_SIZE_SMALL = 128

SEARCH_MODE = "rl"  # "rl" | "greedy" | "both"

# 仅当 SEARCH_MODE == "greedy" 时生效：贪婪算法的初始 GELU/Softmax 配置
# 长度需与模型层数一致（如 12）；不足会按最后一值填充，超出会截断
# GELU 每层取值: 4, 2, 1  ;  Softmax 每层取值: 6, 5, 4, 3, 2
GREEDY_INITIAL_GELU = [1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1]
GREEDY_INITIAL_SOFTMAX = [2, 3, 4, 6, 4, 4, 5, 4, 4, 5, 5, 2]

# 仅当 SEARCH_MODE == "both" 时生效：PPO 结束后是否再执行贪心搜索
# True: PPO 结束后执行贪心 refinement; False: 直接使用 RL 结果
USE_GREEDY_SEARCH = False


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
        
        # GELU Head: 输出4个动作的logits (对应度数 4, 2, 1, 0)
        self.gelu_head = nn.Linear(hidden_dim, 4)
        # Softmax Head: 输出5个动作的logits (对应度数 2, 3, 4, 5, 6)
        self.softmax_head = nn.Linear(hidden_dim, 5)
        
        # PDF 4.2节：策略输出层使用极小的 gain (0.01)
        # 确保训练初始阶段各动作概率几乎均等（高熵），最大化探索能力
        orthogonal_init(self.gelu_head, gain=0.01)
        orthogonal_init(self.softmax_head, gain=0.01)
    
    def forward(self, state, gelu_mask=None):
        # 特征编码
        x = self.encoder(state)
        
        # 残差块处理
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # 双头输出
        gelu_logits = self.gelu_head(x)
        softmax_logits = self.softmax_head(x)
        
        if gelu_mask is not None:
            gelu_logits = gelu_logits.masked_fill(~gelu_mask, float('-inf'))
        
        return gelu_logits, softmax_logits
    
    def get_action_and_logprob(self, state, return_probs=False, gelu_mask=None):
        """
        获取动作和log概率
        Args:
            state: 状态张量
            return_probs: 是否返回概率分布（用于记录中间结果）
            gelu_mask: (Batch, 4) bool Tensor, True=该动作可选
        Returns:
            gelu_action, softmax_action, logprob
            如果return_probs=True，还返回gelu_probs, softmax_probs
        """
        gelu_logits, softmax_logits = self.forward(state, gelu_mask=gelu_mask)
        
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
            gelu_probs = torch.softmax(gelu_logits, dim=-1)
            softmax_probs = torch.softmax(softmax_logits, dim=-1)
            return gelu_action.squeeze(), softmax_action.squeeze(), (gelu_logprob + softmax_logprob).squeeze(), gelu_probs.squeeze(), softmax_probs.squeeze()
        
        return gelu_action.squeeze(), softmax_action.squeeze(), (gelu_logprob + softmax_logprob).squeeze()
    
    def evaluate_actions(self, states, gelu_actions, softmax_actions, gelu_mask=None):
        gelu_logits, softmax_logits = self.forward(states, gelu_mask=gelu_mask)
        
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


# ==================== GTrXL PDF优化方案：GRU门控模块 ====================
class GRUGate(nn.Module):
    """
    GTrXL 核心组件：GRU 门控机制（PDF 3.2节）
    
    用 GRU 的门控逻辑替代传统 Transformer 的加法残差连接 (x + sublayer(x))。
    训练初期门控偏置使 z ≈ 0，子层输出被抑制，输入直通；
    随着训练推进，门控逐渐打开，引入注意力/FFN 提取的特征。
    """
    def __init__(self, d_model):
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Ur = nn.Linear(d_model, d_model, bias=False)
        self.Wz = nn.Linear(d_model, d_model, bias=False)
        self.Uz = nn.Linear(d_model, d_model, bias=False)
        self.Wg = nn.Linear(d_model, d_model, bias=False)
        self.Ug = nn.Linear(d_model, d_model, bias=False)
        self.bg = nn.Parameter(torch.ones(d_model) * 2.0)

    def forward(self, x, y):
        """
        Args:
            x: (B, S, D) 直通路径输入（上一层输出）
            y: (B, S, D) 子层输出（注意力或FFN的输出）
        Returns:
            (B, S, D) 门控融合后的输出
        """
        r = torch.sigmoid(self.Wr(y) + self.Ur(x))
        z = torch.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h_hat = torch.tanh(self.Wg(y) + self.Ug(r * x))
        return (1 - z) * x + z * h_hat


# ==================== GTrXL PDF优化方案：GTrXL Block ====================
class GTrXLBlock(nn.Module):
    """
    单个 GTrXL Block（PDF 4.2节）
    
    结构：Pre-LayerNorm + CausalSelfAttention + GRU Gate
         + Pre-LayerNorm + FFN + GRU Gate
    
    与标准 Transformer Block 的两个关键差异：
    1. Pre-LN（恒等映射重排序）：LayerNorm 在子层之前，保证梯度直通
    2. GRU Gate 替代残差加法：动态路由，训练初期自动旁路不稳定的子层输出
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.gate1 = GRUGate(d_model)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.gate2 = GRUGate(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: (B, S, D) 输入序列
            attn_mask: (S, S) 因果掩码（上三角 = True 的 bool 矩阵）
            key_padding_mask: (B, S) 填充掩码（True = 忽略）
        Returns:
            (B, S, D)
        """
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.gate1(x, attn_out)
        
        norm_x2 = self.ln2(x)
        ff_out = self.ff(norm_x2)
        x = self.gate2(x, ff_out)
        return x


# ==================== GTrXL PDF优化方案：GTrXL策略价值网络 ====================
class GTrXLStrategyNetwork(nn.Module):
    """
    基于GTrXL的策略价值网络（Transformer PDF优化方案）
    
    替代 LSTMStrategyNetwork，解决 LSTM 的信息压缩瓶颈和长程依赖缺失问题。
    
    架构设计（PDF 3-4节）：
    - 层级位置嵌入 (Embedding, 12→16)：感知当前网络深度
    - 历史动作嵌入 (Embedding, 4→8 / 6→8)：自回归特性，记忆上一步决策
    - 连续特征投影 (Linear, 6→32)：成本/预算/进度等实时特征
    - N层GTrXL Block (d_model=64, n_heads=4)：因果自注意力 + GRU门控
    - 多头Actor (GELU+Softmax logits)：独立动作输出
    - Critic Head (标量Value)：状态价值估计
    
    与 LSTM 版本的关键差异：
    - 无 hidden state：使用因果掩码自注意力替代递归隐状态
    - 全局依赖：任意两层之间 O(1) 路径长度，精确建模非相邻层间关系
    - Rollout 时递增构建 token 序列，训练时一次性处理完整 episode
    """
    def __init__(self, num_layers=12, d_model=GTRXL_D_MODEL,
                 n_heads=GTRXL_N_HEADS, n_gtrxl_layers=GTRXL_N_LAYERS,
                 d_ff=GTRXL_D_FF, dropout=GTRXL_DROPOUT):
        super(GTrXLStrategyNetwork, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        
        # 1. 嵌入层（复用 LSTM 版本的维度设计）
        self.embed_layer_idx = nn.Embedding(num_layers, LSTM_POS_DIM)
        self.embed_prev_g = nn.Embedding(SOS_TOKEN_GELU + 1, LSTM_ACT_G_DIM)
        self.embed_prev_s = nn.Embedding(SOS_TOKEN_SOFTMAX + 1, LSTM_ACT_S_DIM)
        
        # 2. 连续特征投影
        self.fc_continuous = nn.Sequential(
            nn.Linear(LSTM_CONT_DIM, LSTM_PROJ_DIM),
            nn.LayerNorm(LSTM_PROJ_DIM),
            nn.SiLU()
        )
        
        # 3. 输入投影：将拼接后的嵌入投影到 d_model（如果维度不匹配）
        token_input_dim = LSTM_POS_DIM + LSTM_ACT_G_DIM + LSTM_ACT_S_DIM + LSTM_PROJ_DIM  # 64
        self.input_proj = nn.Identity() if token_input_dim == d_model else nn.Linear(token_input_dim, d_model)
        
        # 4. GTrXL 骨干网络
        self.gtrxl_blocks = nn.ModuleList([
            GTrXLBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_gtrxl_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        
        # 5. 多头策略网络 Actor
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh()
        )
        self.head_g = nn.Linear(64, 4)
        self.head_s = nn.Linear(64, 5)
        
        # 6. 价值网络 Critic
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self._initialize_weights()
        self._causal_mask_cache = {}
    
    def _initialize_weights(self):
        """正交初始化"""
        for module in [self.actor_head, self.critic_head, self.fc_continuous]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.head_g.weight, gain=0.01)
        nn.init.constant_(self.head_g.bias, 0.0)
        nn.init.orthogonal_(self.head_s.weight, gain=0.01)
        nn.init.constant_(self.head_s.bias, 0.0)
        
        if isinstance(self.input_proj, nn.Linear):
            nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
            if self.input_proj.bias is not None:
                nn.init.constant_(self.input_proj.bias, 0.0)
        
        for block in self.gtrxl_blocks:
            for p in block.attn.in_proj_weight.chunk(3):
                nn.init.orthogonal_(p)
            nn.init.orthogonal_(block.attn.out_proj.weight)
            for layer in block.ff:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
    
    def _get_causal_mask(self, seq_len, device):
        """生成因果掩码（上三角 = True，禁止看到未来 token）"""
        if seq_len not in self._causal_mask_cache or self._causal_mask_cache[seq_len].device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask_cache[seq_len] = mask
        return self._causal_mask_cache[seq_len]
    
    def _build_tokens(self, cont_features, layer_indices, prev_g_actions, prev_s_actions):
        """
        构建 token 序列
        
        Args:
            cont_features: (B, S, 6) 连续特征
            layer_indices: (B, S) long
            prev_g_actions: (B, S) long
            prev_s_actions: (B, S) long
        Returns:
            tokens: (B, S, d_model)
        """
        emb_l = self.embed_layer_idx(layer_indices)
        emb_pg = self.embed_prev_g(prev_g_actions)
        emb_ps = self.embed_prev_s(prev_s_actions)
        feat_c = self.fc_continuous(cont_features)
        
        token_input = torch.cat([emb_l, emb_pg, emb_ps, feat_c], dim=-1)
        return self.input_proj(token_input)
    
    def forward(self, cont_features, layer_indices, prev_g_actions, prev_s_actions,
                gelu_mask=None, key_padding_mask=None):
        """
        全序列前向传播
        
        Args:
            cont_features: (B, S, 6) 连续特征
            layer_indices: (B, S) long
            prev_g_actions: (B, S) long
            prev_s_actions: (B, S) long
            gelu_mask: (B, S, 4) bool, True=可选
            key_padding_mask: (B, S) bool, True=填充位置（忽略）
        
        Returns:
            logits_g: (B, S, 4)
            logits_s: (B, S, 5)
            values: (B, S)
        """
        tokens = self._build_tokens(cont_features, layer_indices, prev_g_actions, prev_s_actions)
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
    
    def get_action_and_logprob(self, cont_features, layer_indices, prev_g_actions, prev_s_actions,
                                return_probs=False, gelu_mask=None, key_padding_mask=None):
        """
        序列前向推理，取最后一个 token 的输出做决策（用于 Rollout）
        
        在 Rollout 时，每一步将新 token 追加到序列中，将递增的完整序列传入，
        取最后一个时间步的输出做动作采样。
        
        Args:
            cont_features: (1, S, 6) 当前已构建的 token 序列（1 到 S 步）
            layer_indices: (1, S) long
            prev_g_actions: (1, S) long
            prev_s_actions: (1, S) long
            return_probs: 是否返回概率分布
            gelu_mask: (1, S, 4) bool
            key_padding_mask: (1, S) bool
        
        Returns:
            action_g, action_s, logprob, value, [g_probs, s_probs]
        """
        logits_g, logits_s, values = self.forward(
            cont_features, layer_indices, prev_g_actions, prev_s_actions,
            gelu_mask=gelu_mask, key_padding_mask=key_padding_mask
        )
        
        lg = logits_g[:, -1, :]  # (1, 4) 取最后一个时间步
        ls = logits_s[:, -1, :]  # (1, 5)
        val = values[:, -1]      # (1,)
        
        lg = lg.squeeze(0)  # (4,)
        ls = ls.squeeze(0)  # (5,)
        val = val.squeeze(0)  # scalar
        
        dist_g = Categorical(logits=lg)
        dist_s = Categorical(logits=ls)
        
        action_g = dist_g.sample()
        action_s = dist_s.sample()
        
        logprob = dist_g.log_prob(action_g) + dist_s.log_prob(action_s)
        
        if return_probs:
            g_probs = torch.softmax(lg, dim=-1)
            s_probs = torch.softmax(ls, dim=-1)
            return action_g, action_s, logprob, val, g_probs, s_probs
        
        return action_g, action_s, logprob, val
    
    def evaluate_actions(self, cont_features, layer_indices, prev_g_actions, prev_s_actions,
                         actions_g, actions_s, gelu_mask=None):
        """
        评估动作（用于PPO更新，完整 episode 一次前向传播）
        
        Args:
            cont_features: (B, 12, 6)
            layer_indices: (B, 12) long
            prev_g_actions: (B, 12) long
            prev_s_actions: (B, 12) long
            actions_g: (B, 12) long
            actions_s: (B, 12) long
            gelu_mask: (B, 12, 4) bool
        
        Returns:
            logprobs: (B, 12)
            entropy: (B, 12)
            values: (B, 12)
        """
        logits_g, logits_s, values = self.forward(
            cont_features, layer_indices, prev_g_actions, prev_s_actions,
            gelu_mask=gelu_mask
        )
        
        dist_g = Categorical(logits=logits_g)
        dist_s = Categorical(logits=logits_s)
        
        logprobs = dist_g.log_prob(actions_g) + dist_s.log_prob(actions_s)
        entropy = dist_g.entropy() + dist_s.entropy()
        
        return logprobs, entropy, values


# ==================== LSTM PDF优化方案：LSTM策略价值网络 ====================
class LSTMStrategyNetwork(nn.Module):
    """
    基于LSTM的策略价值网络（LSTM PDF优化方案）
    
    核心改进：将架构搜索问题建模为序列决策过程（POMDP），
    利用LSTM的时间记忆特性捕捉Transformer层间的量化误差传播。
    
    架构设计（PDF 3节）：
    - 层级位置嵌入 (Embedding, 12→16)：感知当前网络深度
    - 历史动作嵌入 (Embedding, 4→8 / 6→8)：自回归特性，记忆上一步决策
    - 连续特征投影 (Linear, 6→32)：成本/预算/进度等实时特征
    - 双层堆叠LSTM (hidden=128, dropout=0.1)：序列化建模核心
    - 多头Actor (GELU+Softmax logits)：独立动作输出
    - Critic Head (标量Value)：状态价值估计
    """
    def __init__(self, num_layers=12, hidden_dim=LSTM_HIDDEN_DIM):
        super(LSTMStrategyNetwork, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 1. 嵌入层（PDF 3.1）
        self.embed_layer_idx = nn.Embedding(num_layers, LSTM_POS_DIM)            # 层索引 0-11
        self.embed_prev_g = nn.Embedding(SOS_TOKEN_GELU + 1, LSTM_ACT_G_DIM)    # 0-3动作 + 4=SOS
        self.embed_prev_s = nn.Embedding(SOS_TOKEN_SOFTMAX + 1, LSTM_ACT_S_DIM) # 0-4动作 + 5=SOS
        
        # 2. 连续特征投影（PDF 3.1.3 + 3.1.4）
        # 输入6维: cost_deviation, complexity_debt, progress, loss_budget, m1_budget, m2_budget
        self.fc_continuous = nn.Sequential(
            nn.Linear(LSTM_CONT_DIM, LSTM_PROJ_DIM),
            nn.LayerNorm(LSTM_PROJ_DIM),
            nn.SiLU()
        )
        
        # 3. LSTM核心控制器（PDF 3.2）
        # 输入维度 = 16(pos) + 8(prev_g) + 8(prev_s) + 32(continuous) = 64
        lstm_input_dim = LSTM_POS_DIM + LSTM_ACT_G_DIM + LSTM_ACT_S_DIM + LSTM_PROJ_DIM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT
        )
        
        # 4. 多头策略网络 Actor（PDF 3.3.1）
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh()
        )
        self.head_g = nn.Linear(64, 4)   # GELU logits (4个动作: degree 4/2/1/0)
        self.head_s = nn.Linear(64, 5)   # Softmax logits (5个动作)
        
        # 5. 价值网络 Critic（PDF 3.3.2）
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """正交初始化 + LSTM遗忘门偏置初始化（PDF 3.2 + 7.1）"""
        # 线性层正交初始化
        for module in [self.actor_head, self.critic_head, self.fc_continuous]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)
        
        # 策略输出头使用极小gain（确保初始动作均匀分布，最大化探索）
        nn.init.orthogonal_(self.head_g.weight, gain=0.01)
        nn.init.constant_(self.head_g.bias, 0.0)
        nn.init.orthogonal_(self.head_s.weight, gain=0.01)
        nn.init.constant_(self.head_s.bias, 0.0)
        
        # LSTM权重正交初始化 + 遗忘门偏置初始化为1.0（PDF 7.1：鼓励保留长期记忆）
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                n = param.size(0)
                # LSTM bias: [input_gate, forget_gate, cell_gate, output_gate]
                param.data.zero_()
                param.data[n // 4: n // 2].fill_(1.0)  # forget gate = 1.0
    
    def forward(self, cont_features, layer_indices, prev_g_actions, prev_s_actions, hidden=None, gelu_mask=None):
        """
        全序列前向传播（支持Full-Episode BPTT, PDF 4.2）
        
        Args:
            cont_features: (Batch, Seq, 6) 连续特征
            layer_indices: (Batch, Seq) long 层索引
            prev_g_actions: (Batch, Seq) long 前一步GELU动作（SOS=4）
            prev_s_actions: (Batch, Seq) long 前一步Softmax动作（SOS=5）
            hidden: LSTM隐藏状态元组 (h_0, c_0)，None则零初始化
            gelu_mask: (Batch, Seq, 4) bool Tensor, True=该动作可选。
                       None 时不做 mask（所有动作可选）。
        
        Returns:
            logits_g: (Batch, Seq, 4) GELU动作logits
            logits_s: (Batch, Seq, 5) Softmax动作logits
            values: (Batch, Seq) 价值估计
            new_hidden: 更新后的隐藏状态
        """
        # 特征嵌入
        emb_l = self.embed_layer_idx(layer_indices)      # (B, S, 16)
        emb_pg = self.embed_prev_g(prev_g_actions)       # (B, S, 8)
        emb_ps = self.embed_prev_s(prev_s_actions)       # (B, S, 8)
        feat_c = self.fc_continuous(cont_features)        # (B, S, 32)
        
        # 拼接输入（PDF 3.1.5: dim = 16+8+8+32 = 64）
        lstm_input = torch.cat([emb_l, emb_pg, emb_ps, feat_c], dim=-1)
        
        # LSTM前向传播
        self.lstm.flatten_parameters()
        lstm_out, new_hidden = self.lstm(lstm_input, hidden)  # (B, S, 128)
        
        # Actor解码（PDF 3.3.1）
        actor_feat = self.actor_head(lstm_out)  # (B, S, 64)
        logits_g = self.head_g(actor_feat)      # (B, S, 4)
        logits_s = self.head_s(actor_feat)      # (B, S, 5)
        
        # Degree 0 action masking: 不合格层的 degree 0 logit 设为 -inf
        if gelu_mask is not None:
            logits_g = logits_g.masked_fill(~gelu_mask, float('-inf'))
        
        # Critic解码（PDF 3.3.2）
        values = self.critic_head(lstm_out).squeeze(-1)  # (B, S)
        
        return logits_g, logits_s, values, new_hidden
    
    def get_action_and_logprob(self, cont_feat, layer_idx, prev_g, prev_s, hidden, return_probs=False, gelu_mask=None):
        """
        单步前向推理（用于Rollout自回归采样, PDF 5.2）
        
        Args:
            cont_feat: (1, 1, 6) 当前步连续特征
            layer_idx: (1, 1) long 当前层索引
            prev_g: (1, 1) long 前一步GELU动作
            prev_s: (1, 1) long 前一步Softmax动作
            hidden: LSTM隐藏状态
            return_probs: 是否返回概率分布
            gelu_mask: (1, 1, 4) bool Tensor, True=该动作可选
        
        Returns:
            action_g, action_s, logprob, value, new_hidden, [g_probs, s_probs]
        """
        logits_g, logits_s, values, new_hidden = self.forward(
            cont_feat, layer_idx, prev_g, prev_s, hidden, gelu_mask=gelu_mask
        )
        
        # Squeeze: (1,1,D) -> (D,)
        lg = logits_g.squeeze(0).squeeze(0)
        ls = logits_s.squeeze(0).squeeze(0)
        val = values.squeeze(0).squeeze(0)
        
        dist_g = Categorical(logits=lg)
        dist_s = Categorical(logits=ls)
        
        action_g = dist_g.sample()
        action_s = dist_s.sample()
        
        logprob = dist_g.log_prob(action_g) + dist_s.log_prob(action_s)
        
        if return_probs:
            g_probs = torch.softmax(lg, dim=-1)
            s_probs = torch.softmax(ls, dim=-1)
            return action_g, action_s, logprob, val, new_hidden, g_probs, s_probs
        
        return action_g, action_s, logprob, val, new_hidden
    
    def evaluate_actions(self, cont_features, layer_indices, prev_g_actions, prev_s_actions, 
                         actions_g, actions_s, gelu_mask=None):
        """
        评估动作（用于PPO更新时的Full-Episode BPTT, PDF 4.2）
        
        Args:
            cont_features: (Batch, 12, 6) 完整episode连续特征
            layer_indices: (Batch, 12) 层索引
            prev_g_actions: (Batch, 12) 前一步GELU动作
            prev_s_actions: (Batch, 12) 前一步Softmax动作
            actions_g: (Batch, 12) 实际采取的GELU动作
            actions_s: (Batch, 12) 实际采取的Softmax动作
            gelu_mask: (Batch, 12, 4) bool Tensor, True=该动作可选
        
        Returns:
            logprobs: (Batch, 12)
            entropy: (Batch, 12)
            values: (Batch, 12)
        """
        logits_g, logits_s, values, _ = self.forward(
            cont_features, layer_indices, prev_g_actions, prev_s_actions, gelu_mask=gelu_mask
        )
        
        dist_g = Categorical(logits=logits_g)
        dist_s = Categorical(logits=logits_s)
        
        logprobs = dist_g.log_prob(actions_g) + dist_s.log_prob(actions_s)
        entropy = dist_g.entropy() + dist_s.entropy()
        
        return logprobs, entropy, values


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


# ==================== LSTM PDF优化方案：循环网络专用Rollout Buffer ====================
class RecurrentRolloutBuffer:
    """
    循环网络专用Rollout Buffer（LSTM PDF 4.1）
    
    以完整Episode为单位存储轨迹数据，保持时间维度完整性。
    存储形状: (Num_Episodes, 12, Feature_Dim)
    
    与传统的flat buffer不同，LSTM需要保留序列结构以进行Full-Episode BPTT。
    """
    def __init__(self):
        self.episodes = []
        self._current = None
    
    def start_episode(self):
        """开始记录新的Episode"""
        self._current = {
            'cont_features': [],   # (Step, 6) 连续特征
            'layer_indices': [],   # (Step,) 层索引
            'prev_g_actions': [],  # (Step,) 前一步GELU动作
            'prev_s_actions': [],  # (Step,) 前一步Softmax动作
            'actions_g': [],       # (Step,) 采取的GELU动作
            'actions_s': [],       # (Step,) 采取的Softmax动作
            'logprobs': [],        # (Step,) 对数概率
            'rewards': [],         # (Step,) 奖励
            'values': [],          # (Step,) 价值估计
            'dones': [],           # (Step,) 终止标记
            'gelu_masks': [],      # (Step, 4) GELU动作掩码
        }
    
    def add_step(self, cont_feat, layer_idx, prev_g, prev_s, 
                 action_g, action_s, logprob, reward, value, done, gelu_mask=None):
        """添加一步数据到当前Episode"""
        self._current['cont_features'].append(cont_feat)
        self._current['layer_indices'].append(layer_idx)
        self._current['prev_g_actions'].append(prev_g)
        self._current['prev_s_actions'].append(prev_s)
        self._current['actions_g'].append(action_g)
        self._current['actions_s'].append(action_s)
        self._current['logprobs'].append(logprob)
        self._current['rewards'].append(reward)
        self._current['values'].append(value)
        self._current['dones'].append(done)
        if gelu_mask is not None:
            self._current['gelu_masks'].append(gelu_mask)
    
    def end_episode(self):
        """结束当前Episode，加入存储"""
        self.episodes.append(self._current)
        self._current = None
    
    def clear(self):
        """清空所有存储"""
        self.episodes.clear()
    
    @property
    def num_episodes(self):
        return len(self.episodes)
    
    def get_batch(self, device):
        """
        将所有Episode堆叠为batch张量（LSTM PDF 4.1.1）
        
        Returns:
            cont_features: (N_eps, 12, 6) float
            layer_indices: (N_eps, 12) long
            prev_g_actions: (N_eps, 12) long
            prev_s_actions: (N_eps, 12) long
            actions_g: (N_eps, 12) long
            actions_s: (N_eps, 12) long
            logprobs: (N_eps, 12) float
            rewards: (N_eps, 12) float
            values: (N_eps, 12) float
            dones: (N_eps, 12) float
        """
        cont_features = torch.stack([
            torch.stack(ep['cont_features']) for ep in self.episodes
        ]).to(device)
        
        layer_indices = torch.stack([
            torch.tensor(ep['layer_indices'], dtype=torch.long) for ep in self.episodes
        ]).to(device)
        
        prev_g_actions = torch.stack([
            torch.tensor(ep['prev_g_actions'], dtype=torch.long) for ep in self.episodes
        ]).to(device)
        
        prev_s_actions = torch.stack([
            torch.tensor(ep['prev_s_actions'], dtype=torch.long) for ep in self.episodes
        ]).to(device)
        
        actions_g = torch.stack([
            torch.tensor(ep['actions_g'], dtype=torch.long) for ep in self.episodes
        ]).to(device)
        
        actions_s = torch.stack([
            torch.tensor(ep['actions_s'], dtype=torch.long) for ep in self.episodes
        ]).to(device)
        
        logprobs = torch.stack([
            torch.stack(ep['logprobs']) for ep in self.episodes
        ]).to(device)
        
        rewards = torch.tensor([
            ep['rewards'] for ep in self.episodes
        ], dtype=torch.float32).to(device)
        
        values = torch.stack([
            torch.stack(ep['values']) for ep in self.episodes
        ]).to(device)
        
        dones = torch.tensor([
            ep['dones'] for ep in self.episodes
        ], dtype=torch.float32).to(device)
        
        # GELU动作掩码: (N_eps, 12, 4) bool
        has_masks = all(len(ep.get('gelu_masks', [])) > 0 for ep in self.episodes)
        if has_masks:
            gelu_masks = torch.stack([
                torch.tensor(np.array(ep['gelu_masks']), dtype=torch.bool) for ep in self.episodes
            ]).to(device)
        else:
            gelu_masks = None
        
        return (cont_features, layer_indices, prev_g_actions, prev_s_actions,
                actions_g, actions_s, logprobs, rewards, values, dones, gelu_masks)







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
                 constraint_limits=None, prev_metrics=None, num_metrics=2,
                 gelu_degree0_eligible=None):
        """
        初始化环境
        
        敏锐度优化PDF扩展参数：
        - constraint_limits: 约束阈值字典 {'loss': float, 'metric1': float, 'metric2': float}
                            用于课程学习的动态约束调整
        - prev_metrics: 上一episode结束时的指标（用于差分奖励计算）
        - num_metrics: 当前数据集的评估指标数量 (1 或 2)
        - gelu_degree0_eligible: np.ndarray[bool], shape=(total_layers,)
                                 每层是否有资格使用 degree 0 动作
        """
        self.total_layers = total_layers
        self.baseline_cost = baseline_cost  # 72.0 for 12 layers
        self.baseline_loss, self.baseline_p, self.baseline_s = baseline_metrics
        self.evaluator = evaluator
        self.num_metrics = num_metrics
        
        # Degree 0 资格掩码
        if gelu_degree0_eligible is None:
            self.gelu_degree0_eligible = np.zeros(total_layers, dtype=bool)
        else:
            self.gelu_degree0_eligible = gelu_degree0_eligible
        
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
        # GELU: degree 4->0.0, 2->0.5, 1->1.0, 0->1.25 (高精度->低精度)
        self.gelu_degree_to_norm = {4: 0.0, 2: 0.5, 1: 1.0, 0: 1.25}
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
    
    def get_gelu_action_mask(self, layer_idx=None):
        """
        返回指定层的 GELU 动作掩码 (4-dim bool)。
        True = 该动作可选, False = 被禁止。
        动作索引: 0=degree4, 1=degree2, 2=degree1, 3=degree0
        
        如果 layer_idx 为 None，使用 self.current_layer。
        """
        if layer_idx is None:
            layer_idx = self.current_layer
        mask = np.array([True, True, True, False], dtype=bool)
        if layer_idx < len(self.gelu_degree0_eligible) and self.gelu_degree0_eligible[layer_idx]:
            mask[3] = True
        return mask
    
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
        # Metric2预算：单指标数据集设为0.0（表示该维度为空），双指标数据集正常计算
        if self.num_metrics == 1:
            m2_budget = 0.0
        else:
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
        
        # 综合精度差分奖励（根据指标数量调整分母）
        if self.num_metrics == 1:
            r_accuracy_diff = (r_loss_diff + r_m1_diff) / 2.0
        else:
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
        
        # 综合约束奖励（根据指标数量调整分母）
        if self.num_metrics == 1:
            r_constraint = (r_loss_barrier + r_m1_barrier) / 2.0
        else:
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
                 batch_size=16,
                 stage1_rl_episodes=PPO_MAX_EPISODES,
                 stage2_rl_episodes=40000,
                 stage1_rl_episodes_specified=False,
                 stage2_rl_episodes_specified=False,
                 device='cuda', data_path='stsb', test_data_mm=None,
                 run_output_dir='',
                 final_eval_config_source='search',
                 final_eval_config_path='glue_configs_best_ppo.json',
                 manual_final_gelu=None,
                 manual_final_softmax=None,
                 final_eval_random_seed=42,
                 final_eval_permutation_trials=10,
                 final_eval_cost_equivalent_trials=10,
                 final_eval_budget_equivalent_trials=10,
                 skip_noise_rl=False,
                 noise_eval_config_source='search',
                 noise_eval_config_path='glue_noise_configs_best_ppo.json',
                 manual_noise_config=None,
                 noise_eval_repeat_n=5,
                 skip_stage1_rl=False,
                 skip_stage1_final_eval=False,
                 skip_noise_final_eval=False):
        """
        基于 PPO 强化学习的策略搜索器。
        目标：在密文推理场景下，通过强化学习寻找最优的多项式近似策略。
        
        敏锐度优化PDF扩展：
        - data_path: 数据集名称，用于选择评估指标
            各数据集使用不同指标（详见 DATASET_METRICS_CONFIG）
        - test_data_mm: MNLI mismatched 验证集（仅 MNLI 使用）
        """
        # 增加递归深度以支持深拷贝复杂模型图
        sys.setrecursionlimit(50000)
        
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.data_collator = data_collator
        self.batch_size = max(1, int(batch_size))
        self.stage1_rl_episodes = self._coerce_positive_int(
            stage1_rl_episodes, 'stage1_rl_episodes'
        )
        self.stage2_rl_episodes = self._coerce_positive_int(
            stage2_rl_episodes, 'stage2_rl_episodes'
        )
        self.stage1_rl_episodes_specified = self._coerce_bool_flag(
            stage1_rl_episodes_specified, 'stage1_rl_episodes_specified'
        )
        self.stage2_rl_episodes_specified = self._coerce_bool_flag(
            stage2_rl_episodes_specified, 'stage2_rl_episodes_specified'
        )
        
        # ==================== 敏锐度优化PDF：数据集检测与指标选择 ====================
        self.data_path = data_path
        self.is_regression = self._detect_task_type()
        self._log_task_type()
        
        # 训练集用于RL训练，验证集用于最终评估
        self.dataloader_train = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        self.dataloader_test = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        
        # MNLI mismatched 验证集（仅 MNLI 需要）
        if test_data_mm is not None:
            self.dataloader_test_mm = DataLoader(
                test_data_mm,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
        else:
            self.dataloader_test_mm = None

        self._prepare_rl_datasets(
            train_data=train_data,
            validation_data=test_data,
            validation_data_mm=test_data_mm,
        )
        
        try:
            self.reversible_handler = ReversibleLayerHandler(self.model)
        except Exception as e:
            print(f"[警告] 深拷贝（Deepcopy）在处理器初始化时失败: {e}。'restore_all' 可能会失败。")
            self.reversible_handler = ReversibleLayerHandler(self.model)

        self.layers_attribute = self._detect_layer_attribute()
        self.total_layers = len(eval('self.model.' + self.layers_attribute))
        
        # --- 密文代价模型 ---
        self.GELU_COST_MAP = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
        self.SOFTMAX_COST_MAP = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}
        self.INPUT_NOISE_COST_MAP = INPUT_NOISE_COST_MAP.copy()
        self.WEIGHT_NOISE_COST_MAP = WEIGHT_NOISE_COST_MAP.copy()
        self.WFFN1_NOISE_COST_MAP = WFFN1_NOISE_COST_MAP.copy()

        # 搜索状态初始化
        self.current_gelu_degrees = np.full(self.total_layers, 4, dtype=int)
        self.current_softmax_degrees = np.full(self.total_layers, 6, dtype=int)
        self.current_input_noise_scaling_factors = np.full(
            self.total_layers,
            INPUT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.input_noise_action_space = tuple(INPUT_NOISE_ALLOWED_SCALING_FACTORS)
        self.current_wq_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wk_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wv_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wo_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wffn1_noise_scaling_factors = np.full(
            self.total_layers,
            WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.current_wffn2_noise_scaling_factors = np.full(
            self.total_layers,
            WEIGHT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )
        self.wq_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wk_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wv_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wo_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn1_noise_action_space = tuple(WFFN1_NOISE_ALLOWED_SCALING_FACTORS)
        self.wffn2_noise_action_space = tuple(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        self.rl_lr_raw = rl_lr
        self.ppo_lr_initial, self.ppo_lr_mode = resolve_ppo_learning_rate(rl_lr)
        
        # --- 用户阈值配置 (Strict 1.5%) ---
        self.error_threshold = 0.015
        self.correlation_drop_ratio = 0.015
        
        output_layout = resolve_run_output_layout(run_output_dir)
        self.run_output_dir = output_layout["run_output_dir"]
        self.log_file = output_layout["log_file"]
        self.noise_log_file = output_layout["noise_log_file"]
        self.active_log_file = self.log_file
        self.step_info_file = output_layout["stage1_step_info_file"]
        self.stage1_training_curve_path = output_layout["stage1_training_curve_path"]
        self.stage1_entropy_curve_path = output_layout["stage1_entropy_curve_path"]
        self.stage1_final_eval_dir = output_layout["stage1_final_eval_dir"]
        self.noise_step_info_file = output_layout["noise_step_info_file"]
        self.noise_stage_training_curve_path = output_layout["noise_training_curve_path"]
        self.noise_stage_entropy_curve_path = output_layout["noise_entropy_curve_path"]
        self.noise_stage_progress_dir = output_layout["noise_progress_dir"]
        self.noise_final_eval_dir = output_layout["noise_final_eval_dir"]
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=== PPO强化学习优化日志已启动（PPO RL Optimization Log Started） ===\n")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(
                f"[信息] PPO学习率（LR）从 rl_lr={self.rl_lr_raw!r} 解析为 -> "
                f"{self.ppo_lr_initial:.6g} ({self.ppo_lr_mode})\n"
            )
            f.write(
                f"[信息] 第一阶段RL回合数（Stage-1 RL episodes）: {self.stage1_rl_episodes} | "
                f"第二阶段RL回合数（Stage-2 RL episodes）: {self.stage2_rl_episodes}\n"
            )
            if self.run_output_dir:
                f.write(f"[信息] 统一运行输出目录（Unified run output dir）: {self.run_output_dir}\n")
        if self.noise_log_file != self.log_file:
            with open(self.noise_log_file, "w", encoding="utf-8") as f:
                f.write("=== 第二阶段噪声RL优化日志已启动（Stage-2 Noise RL Optimization Log Started） ===\n")
            with open(self.noise_log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"[信息] PPO学习率（LR）从 rl_lr={self.rl_lr_raw!r} 解析为 -> "
                    f"{self.ppo_lr_initial:.6g} ({self.ppo_lr_mode})\n"
                )
                f.write(
                    f"[信息] 第一阶段RL回合数（Stage-1 RL episodes）: {self.stage1_rl_episodes} | "
                    f"第二阶段RL回合数（Stage-2 RL episodes）: {self.stage2_rl_episodes}\n"
                )
                if self.run_output_dir:
                    f.write(f"[信息] 统一运行输出目录（Unified run output dir）: {self.run_output_dir}\n")
        
        # ==================== 策略二：动态超参数调度状态 ====================
        self.current_episode = 0
        self.total_episodes = self.stage1_rl_episodes
        self.current_entropy_coef = PPO_ENTROPY_INITIAL
        self.current_lr = self.ppo_lr_initial
        
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

        self.final_eval_config_source = (final_eval_config_source or 'search').lower()
        self.final_eval_config_path = final_eval_config_path or 'glue_configs_best_ppo.json'
        self.manual_final_gelu = manual_final_gelu
        self.manual_final_softmax = manual_final_softmax
        self.final_eval_random_seed = int(final_eval_random_seed)
        self.final_eval_permutation_trials = max(0, int(final_eval_permutation_trials))
        self.final_eval_cost_equivalent_trials = max(0, int(final_eval_cost_equivalent_trials))
        self.final_eval_budget_equivalent_trials = max(0, int(final_eval_budget_equivalent_trials))
        self.skip_stage1_rl = self._coerce_bool_flag(skip_stage1_rl, 'skip_stage1_rl')
        self.skip_noise_rl = self._coerce_bool_flag(skip_noise_rl, 'skip_noise_rl')

        self.noise_eval_config_source = (noise_eval_config_source or 'search').lower()
        self.noise_eval_config_path = noise_eval_config_path or 'glue_noise_configs_best_ppo.json'
        self.manual_noise_config = manual_noise_config
        self.noise_eval_repeat_n = max(5, int(noise_eval_repeat_n))
        self.skip_stage1_final_eval = self._coerce_bool_flag(
            skip_stage1_final_eval, 'skip_stage1_final_eval'
        )
        self.skip_noise_final_eval = self._coerce_bool_flag(
            skip_noise_final_eval, 'skip_noise_final_eval'
        )

        if self.noise_eval_config_source not in ('search', 'json', 'manual'):
            raise ValueError(
                f"Unsupported noise_eval_config_source '{self.noise_eval_config_source}'. "
                "Use one of: search, json, manual."
            )

        if self.final_eval_config_source not in ('search', 'json', 'manual'):
            raise ValueError(
                f"Unsupported final_eval_config_source '{self.final_eval_config_source}'. "
                "Use one of: search, json, manual."
            )

        if self.skip_stage1_rl and self.final_eval_config_source == 'search':
            raise ValueError(
                "skip_stage1_rl=True 与 final_eval_config_source='search' 不能同时使用："
                "跳过第一阶段搜索后没有 RL/贪心结果可供 search 使用。"
                "请改用 json 或 manual，或设置 skip_stage1_rl=False。"
            )

        if self.skip_stage1_rl and (
            self.stage1_rl_episodes_specified
            or self.stage1_rl_episodes != PPO_MAX_EPISODES
        ):
            raise ValueError(
                "stage1_rl_episodes was explicitly set, but skip_stage1_rl=True. "
                "Remove --stage1-rl-episodes or disable --skip-stage1-rl."
            )

        if (not self.skip_stage1_rl) and self.final_eval_config_source != 'search':
            raise ValueError(
                "检测到第一阶段 RL/贪心搜索将执行，但 final_eval_config_source 不是 'search'。"
                "为避免“前面跑 RL、后面却用手动/JSON 配置评估”的流程混用，"
                "执行第一阶段 RL 时只能使用 search。"
                "若要使用 json/manual，请设置 skip_stage1_rl=True。"
            )

        if (not self.skip_stage1_rl) and self.stage1_rl_episodes < PPO_UPDATE_INTERVAL:
            raise ValueError(
                f"stage1_rl_episodes={self.stage1_rl_episodes} is too small. "
                f"It must be >= PPO_UPDATE_INTERVAL ({PPO_UPDATE_INTERVAL}) so Stage-1 PPO can update at least once."
            )

        if self.final_eval_config_source == 'manual':
            if self.manual_final_gelu is None or self.manual_final_softmax is None:
                raise ValueError(
                    "final_eval_config_source='manual' 时必须同时提供 "
                    "manual_final_gelu 和 manual_final_softmax。"
                )

        if self.skip_noise_rl and (not self.skip_noise_final_eval) and self.noise_eval_config_source == 'search':
            raise ValueError(
                "skip_noise_rl=True 与 noise_eval_config_source='search' 不能同时用于噪声最终评估："
                "跳过噪声 RL 后没有搜索结果可供 search 使用。"
                "请改用 json 或 manual，或设置 skip_noise_rl=False。"
            )

        if self.skip_noise_rl and (
            self.stage2_rl_episodes_specified
            or self.stage2_rl_episodes != 40000
        ):
            raise ValueError(
                "stage2_rl_episodes was explicitly set, but skip_noise_rl=True. "
                "Remove --stage2-rl-episodes or disable --skip-noise-rl."
            )

        if (not self.skip_noise_rl) and (not self.skip_noise_final_eval) and self.noise_eval_config_source != 'search':
            raise ValueError(
                "检测到第二阶段噪声 RL 将执行，且噪声最终评估未跳过，但 noise_eval_config_source 不是 'search'。"
                "为避免“前面跑噪声 RL、后面却用手动/JSON 配置评估”的流程混用，"
                "执行噪声 RL 且保留噪声最终评估时只能使用 search。"
                "若要使用 json/manual，请设置 skip_noise_rl=True。"
            )

        if (not self.skip_noise_rl) and self.stage2_rl_episodes < PPO_UPDATE_INTERVAL:
            raise ValueError(
                f"stage2_rl_episodes={self.stage2_rl_episodes} is too small. "
                f"It must be >= PPO_UPDATE_INTERVAL ({PPO_UPDATE_INTERVAL}) so Stage-2 PPO can update at least once."
            )

        if self.noise_eval_config_source == 'manual' and self.manual_noise_config is None:
            raise ValueError(
                "noise_eval_config_source='manual' 时必须提供 manual_noise_config。"
            )
        
        # ==================== 敏锐度优化PDF：课程学习状态 ====================
        self.curriculum_phase = 1  # 1=探索, 2=收紧, 3=精调
        self.constraint_slack = CURRICULUM_INITIAL_SLACK  # 当前约束放宽系数

    @staticmethod
    def _coerce_bool_flag(raw_value, flag_name):
        if isinstance(raw_value, bool):
            return raw_value
        if raw_value is None:
            return False

        text = str(raw_value).strip().lower()
        if text in ('1', 'true', 't', 'yes', 'y', 'on'):
            return True
        if text in ('0', 'false', 'f', 'no', 'n', 'off', ''):
            return False

        raise ValueError(
            f"Invalid boolean value for {flag_name}: {raw_value!r}. "
            "Expected one of: true/false/1/0/yes/no."
        )

    @staticmethod
    def _coerce_positive_int(raw_value, flag_name):
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid positive integer for {flag_name}: {raw_value!r}."
            ) from None

        if value <= 0:
            raise ValueError(
                f"Invalid positive integer for {flag_name}: {raw_value!r}."
            )
        return value
    
    def _detect_task_type(self):
        """
        敏锐度优化PDF：检测任务类型（回归/分类）
        根据数据集名称确定使用哪种评估指标
        同时设置 self.dataset_config 和 self.dataset_key
        """
        data_name = self.data_path.lower()
        
        for ds_key, ds_cfg in DATASET_METRICS_CONFIG.items():
            if ds_key in data_name:
                self.dataset_key = ds_key
                self.dataset_config = ds_cfg
                return ds_cfg['type'] == 'regression'
        
        # 默认假设为回归任务 (stsb 行为)
        print(f"[警告] 未知数据集（Unknown dataset）'{data_name}'，假定为回归任务（pearson + spearman）")
        self.dataset_key = 'stsb'
        self.dataset_config = DATASET_METRICS_CONFIG['stsb']
        return True
    
    def _log_task_type(self):
        """记录任务类型信息"""
        full_names = self.dataset_config['metric_full_names']
        task_type = 'REGRESSION' if self.is_regression else 'CLASSIFICATION'
        print(f"[信息] 数据集（Dataset）'{self.data_path}' 检测为 {task_type} 任务")
        print(f"[信息] 使用指标（Using metrics）: {', '.join(full_names)}")
    
    def get_metric_names(self):
        """
        获取当前数据集的完整指标名称（用于日志显示）
        单指标数据集返回 (name,)，双指标返回 (name1, name2)
        """
        full_names = self.dataset_config['metric_full_names']
        if len(full_names) == 1:
            return (full_names[0],)
        return (full_names[0], full_names[1])
    
    def get_metric_short_names(self):
        """获取当前数据集的短指标名称（用于表格）"""
        return self.dataset_config['metric_names']
    
    def get_num_metrics(self):
        """返回当前数据集的评估指标数量 (1 或 2)"""
        return len(self.dataset_config['metrics'])

    def _is_small_validation_task(self):
        return getattr(self, "dataset_key", None) in VALIDATION_SMALL_TASKS

    def _get_validation_holdout_ratio(self):
        if self._is_small_validation_task():
            return VALIDATION_HOLDOUT_RATIO_SMALL
        return VALIDATION_HOLDOUT_RATIO_DEFAULT

    def _get_validation_proxy_ratio(self):
        if self._is_small_validation_task():
            return VALIDATION_PROXY_RATIO_SMALL
        return VALIDATION_PROXY_RATIO_DEFAULT

    def _get_train_anchor_size(self):
        if self._is_small_validation_task():
            return TRAIN_ANCHOR_SIZE_SMALL
        return TRAIN_ANCHOR_SIZE_DEFAULT

    def _make_dataloader(self, dataset):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def _register_dataset_split(self, split_name, dataset, dataset_mm=None):
        if dataset is None:
            self.dataset_splits.pop(split_name, None)
            self.dataloaders.pop(split_name, None)
            self.dataset_splits_mm.pop(split_name, None)
            self.dataloaders_mm.pop(split_name, None)
            return

        self.dataset_splits[split_name] = dataset
        self.dataloaders[split_name] = self._make_dataloader(dataset)

        if dataset_mm is None:
            self.dataset_splits_mm.pop(split_name, None)
            self.dataloaders_mm.pop(split_name, None)
        else:
            self.dataset_splits_mm[split_name] = dataset_mm
            self.dataloaders_mm[split_name] = self._make_dataloader(dataset_mm)

    def has_dataset_split(self, split_name):
        return self.dataloaders.get(split_name) is not None

    def _extract_labels_for_stratify(self, dataset):
        if dataset is None or self.is_regression:
            return None

        column_names = getattr(dataset, "column_names", [])
        if "labels" in column_names:
            label_column = "labels"
        elif "label" in column_names:
            label_column = "label"
        else:
            return None

        try:
            labels = dataset[label_column]
        except Exception:
            labels = [dataset[idx][label_column] for idx in range(len(dataset))]

        normalized = []
        for value in labels:
            if torch.is_tensor(value):
                value = value.item() if value.numel() == 1 else tuple(value.detach().cpu().tolist())
            elif isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else tuple(value.tolist())
            normalized.append(value)
        return normalized

    def _safe_train_test_split(self, indices, seed, labels=None, train_size=None, test_size=None):
        split_kwargs = {
            "shuffle": True,
            "random_state": int(seed),
        }
        if train_size is not None:
            split_kwargs["train_size"] = train_size
        if test_size is not None:
            split_kwargs["test_size"] = test_size

        if labels is not None:
            try:
                return train_test_split(indices, stratify=labels, **split_kwargs)
            except ValueError:
                pass

        return train_test_split(indices, **split_kwargs)

    def _sample_dataset_by_size(self, dataset, subset_size, seed):
        if dataset is None:
            return None

        total_size = len(dataset)
        if total_size == 0:
            return dataset

        subset_size = int(subset_size)
        if subset_size >= total_size:
            return dataset
        if subset_size <= 0:
            subset_size = 1

        indices = np.arange(total_size)
        labels = self._extract_labels_for_stratify(dataset)
        selected_indices, _ = self._safe_train_test_split(
            indices,
            seed=seed,
            labels=labels,
            train_size=subset_size,
        )
        selected_indices = np.sort(np.asarray(selected_indices, dtype=int))
        return dataset.select(selected_indices.tolist())

    def _split_dataset_for_rl(self, dataset, holdout_ratio, seed):
        if dataset is None:
            return None, None

        total_size = len(dataset)
        if total_size < 2 or holdout_ratio <= 0.0:
            return dataset, None

        indices = np.arange(total_size)
        labels = self._extract_labels_for_stratify(dataset)
        search_indices, holdout_indices = self._safe_train_test_split(
            indices,
            seed=seed,
            labels=labels,
            test_size=holdout_ratio,
        )
        search_indices = np.sort(np.asarray(search_indices, dtype=int))
        holdout_indices = np.sort(np.asarray(holdout_indices, dtype=int))

        if len(search_indices) == 0 or len(holdout_indices) == 0:
            return dataset, None

        return (
            dataset.select(search_indices.tolist()),
            dataset.select(holdout_indices.tolist()),
        )

    def _prepare_rl_datasets(self, train_data, validation_data, validation_data_mm=None):
        self.dataset_splits = {}
        self.dataset_splits_mm = {}
        self.dataloaders = {}
        self.dataloaders_mm = {}
        self.current_val_proxy_window = None
        self.current_val_proxy_subset_id = None

        self._register_dataset_split("train", train_data)

        if USE_TRAIN_ANCHOR and train_data is not None:
            anchor_size = min(len(train_data), self._get_train_anchor_size())
            train_anchor = self._sample_dataset_by_size(
                train_data,
                subset_size=anchor_size,
                seed=RL_DATASET_SPLIT_SEED + 11,
            )
            self._register_dataset_split("train_anchor", train_anchor)

        if validation_data is not None:
            self._register_dataset_split("validation_full", validation_data, validation_data_mm)

            if USE_VALIDATION_SEARCH_PROTOCOL:
                holdout_ratio = self._get_validation_holdout_ratio()
                val_search_full, val_holdout = self._split_dataset_for_rl(
                    validation_data,
                    holdout_ratio=holdout_ratio,
                    seed=RL_DATASET_SPLIT_SEED,
                )
                if validation_data_mm is not None:
                    val_search_mm, val_holdout_mm = self._split_dataset_for_rl(
                        validation_data_mm,
                        holdout_ratio=holdout_ratio,
                        seed=RL_DATASET_SPLIT_SEED + 1,
                    )
                else:
                    val_search_mm, val_holdout_mm = None, None
            else:
                val_search_full = validation_data
                val_holdout = None
                val_search_mm = validation_data_mm
                val_holdout_mm = None

            self._register_dataset_split("val_search_full", val_search_full, val_search_mm)
            self._register_dataset_split("val_holdout", val_holdout, val_holdout_mm)

        self.dataloader_train = self.dataloaders.get("train")
        self.dataloader_test = self.dataloaders.get("validation_full")
        self.dataloader_test_mm = self.dataloaders_mm.get("validation_full")

        if validation_data is not None:
            search_size = len(self.dataset_splits.get("val_search_full", validation_data))
            holdout_dataset = self.dataset_splits.get("val_holdout")
            holdout_size = len(holdout_dataset) if holdout_dataset is not None else 0
            print(
                "[数据集协议（Dataset Protocol）] "
                f"完整验证集（validation_full）={len(validation_data)}, "
                f"搜索验证集（val_search_full）={search_size}, "
                f"留出验证集（val_holdout）={holdout_size}"
            )
            if self.dataset_splits_mm.get("val_search_full") is not None:
                mm_search_size = len(self.dataset_splits_mm["val_search_full"])
                mm_holdout_size = (
                    len(self.dataset_splits_mm["val_holdout"])
                    if self.dataset_splits_mm.get("val_holdout") is not None
                    else 0
                )
                print(
                    "[数据集协议（Dataset Protocol）] "
                    f"MNLI不匹配搜索集（mnli_mismatched_search）={mm_search_size}, "
                    f"MNLI不匹配留出集（mnli_mismatched_holdout）={mm_holdout_size}"
                )
        elif USE_VALIDATION_FOR_REWARD:
            print("[警告] 验证集引导的奖励已启用，但未提供验证数据集。")

        if self.has_dataset_split("val_search_full"):
            self.refresh_validation_proxy(window_index=0, stage_label="Initialization", quiet=True)

    def refresh_validation_proxy(self, window_index, stage_label="RL", quiet=False):
        if not self.has_dataset_split("val_search_full"):
            self.current_val_proxy_window = None
            self.current_val_proxy_subset_id = None
            return None

        search_dataset = self.dataset_splits["val_search_full"]
        search_mm_dataset = self.dataset_splits_mm.get("val_search_full")
        proxy_ratio = self._get_validation_proxy_ratio()
        proxy_size = max(1, int(round(len(search_dataset) * proxy_ratio)))
        proxy_dataset = self._sample_dataset_by_size(
            search_dataset,
            subset_size=proxy_size,
            seed=RL_DATASET_SPLIT_SEED + 1000 + int(window_index),
        )

        proxy_mm_dataset = None
        if search_mm_dataset is not None:
            proxy_mm_size = max(1, int(round(len(search_mm_dataset) * proxy_ratio)))
            proxy_mm_dataset = self._sample_dataset_by_size(
                search_mm_dataset,
                subset_size=proxy_mm_size,
                seed=RL_DATASET_SPLIT_SEED + 2000 + int(window_index),
            )

        self._register_dataset_split("val_proxy", proxy_dataset, proxy_mm_dataset)
        self.current_val_proxy_window = int(window_index)
        self.current_val_proxy_subset_id = f"proxy_window_{int(window_index):04d}"

        if not quiet:
            msg = (
                f"[数据集协议（Dataset Protocol）] {stage_label}: "
                f"验证代理窗口（val_proxy window）={int(window_index) + 1}, "
                f"大小（size）={len(proxy_dataset)}/{len(search_dataset)}"
            )
            if proxy_mm_dataset is not None:
                msg += f", MNLI不匹配（mnli_mismatched）={len(proxy_mm_dataset)}/{len(search_mm_dataset)}"
            print(msg)

        return "val_proxy"

    def get_reward_reference_split_name(self):
        if USE_VALIDATION_FOR_REWARD:
            if self.has_dataset_split("val_search_full"):
                return "val_search_full"
            if self.has_dataset_split("validation_full"):
                return "validation_full"
        return "train"

    def get_online_reward_split_name(self):
        if USE_VALIDATION_FOR_REWARD:
            if self.has_dataset_split("val_proxy"):
                return "val_proxy"
            if self.has_dataset_split("val_search_full"):
                return "val_search_full"
            if self.has_dataset_split("validation_full"):
                return "validation_full"
        return "train"

    def _resolve_eval_split(self, use_train=True, split=None):
        if split is not None:
            if not self.has_dataset_split(split):
                raise ValueError(f"Unknown or unavailable dataset split: {split}")
            return split

        if use_train:
            return "train"
        if self.has_dataset_split("validation_full"):
            return "validation_full"
        if self.has_dataset_split("val_search_full"):
            return "val_search_full"
        return "train"

    def _candidate_meets_constraints(self, loss, metric1, metric2, limit_loss, limit_p, limit_s):
        if loss > limit_loss:
            return False
        if metric1 < limit_p:
            return False
        if self.get_num_metrics() > 1 and metric2 < limit_s:
            return False
        return True

    def _is_better_confirmed_candidate(self, candidate, incumbent, metric_prefix):
        if candidate is None:
            return False
        if incumbent is None:
            return True

        cand_cost = float(candidate["cost"])
        inc_cost = float(incumbent["cost"])
        if cand_cost < inc_cost - 1e-8:
            return True
        if cand_cost > inc_cost + 1e-8:
            return False

        cand_loss = float(candidate.get(f"{metric_prefix}_loss", float("inf")))
        inc_loss = float(incumbent.get(f"{metric_prefix}_loss", float("inf")))
        if cand_loss < inc_loss - 1e-8:
            return True
        if cand_loss > inc_loss + 1e-8:
            return False

        cand_metric_sum = (
            float(candidate.get(f"{metric_prefix}_metric1", -float("inf"))) +
            float(candidate.get(f"{metric_prefix}_metric2", -float("inf")))
        )
        inc_metric_sum = (
            float(incumbent.get(f"{metric_prefix}_metric1", -float("inf"))) +
            float(incumbent.get(f"{metric_prefix}_metric2", -float("inf")))
        )
        if cand_metric_sum > inc_metric_sum + 1e-8:
            return True
        if cand_metric_sum < inc_metric_sum - 1e-8:
            return False

        return float(candidate.get("proxy_reward", -float("inf"))) > float(
            incumbent.get("proxy_reward", -float("inf"))
        )

    def _fmt_metrics(self, loss, m1, m2, prefix=""):
        """格式化指标字符串，单指标数据集只显示一个指标"""
        names = self.get_metric_names()
        p = f"{prefix}" if prefix else ""
        if self.get_num_metrics() == 1:
            return f"{p}损失（Loss）: {loss:.6f}, {names[0]}: {m1:.6f}"
        return f"{p}损失（Loss）: {loss:.6f}, {names[0]}: {m1:.6f}, {names[1]}: {m2:.6f}"

    def _fmt_constraints(self, limit_loss, limit_p, limit_s):
        """格式化约束字符串"""
        names = self.get_metric_names()
        if self.get_num_metrics() == 1:
            return f"损失（Loss）<={limit_loss:.4f}, {names[0]}>={limit_p:.4f}"
        return f"损失（Loss）<={limit_loss:.4f}, {names[0]}>={limit_p:.4f}, {names[1]}>={limit_s:.4f}"

    def _write_step_info(self, step_info, f):
        """将单步 StepInfo 写入文件"""
        f.write(f"  全局步数（step_global）: {step_info['step_global']}\n")
        f.write(f"  回合编号（episode_id）: {step_info['episode_id']}\n")
        f.write(f"  层索引（layer_index）: {step_info['layer_index']}\n")
        f.write(f"  状态向量（state_vector）: {step_info['state_vector']}\n")
        f.write(f"  当前GELU阶数（curr_gelu_degree）: {step_info['curr_gelu_degree']}\n")
        f.write(f"  当前Softmax阶数（curr_softmax_degree）: {step_info['curr_softmax_degree']}\n")
        f.write(f"  GELU概率分布（gelu_prob_dist）: {step_info['gelu_prob_dist']}\n")
        f.write(f"  Softmax概率分布（softmax_prob_dist）: {step_info['softmax_prob_dist']}\n")
        f.write(f"  评论家值（critic_value）: {step_info['critic_value']}\n")
        f.write(f"  累计成本（accumulated_cost）: {step_info['accumulated_cost']}\n")
        f.write(f"  GELU配置（gelu_config）: {step_info['gelu_config']}\n")
        f.write(f"  Softmax配置（softmax_config）: {step_info['softmax_config']}\n")
        # 策略二：输出动态超参数调度信息
        if 'current_lr' in step_info:
            f.write(f"  当前学习率（current_lr）: {step_info['current_lr']:.6f}\n")
        if 'current_entropy_coef' in step_info:
            f.write(f"  当前熵系数（current_entropy_coef）: {step_info['current_entropy_coef']:.6f}\n")
    
    def update_hyperparameters(self, optimizer, episode):
        """
        超参数调度（兼容 GTrXL 和 LSTM）
        
        - 熵系数：线性衰减（0.05 -> 0.001），下限为 GTRXL_ENTROPY_LOWER_BOUND
        - 学习率：GTrXL模式下不在此处设置（由 ppo_update_gtrxl 的 Warmup 机制控制）；
                  读取 optimizer 当前 LR 用于日志记录
        - 课程学习：动态调整约束阈值
        """
        self.current_episode = episode
        progress = episode / self.total_episodes
        
        # 熵系数线性衰减，但不低于最小熵约束下界（PDF 4.3）
        new_entropy = PPO_ENTROPY_START - (PPO_ENTROPY_START - PPO_ENTROPY_END) * progress
        new_entropy = max(new_entropy, GTRXL_ENTROPY_LOWER_BOUND)
        self.current_entropy_coef = new_entropy
        
        # 学习率：GTrXL 的 Warmup 由 ppo_update_gtrxl 在每次 PPO 更新时设置，
        # 此处仅读取当前优化器的学习率用于日志记录
        if hasattr(optimizer, 'param_groups'):
            self.current_lr = optimizer.param_groups[0]['lr']
        elif isinstance(optimizer, dict):
            self.current_lr = optimizer.get('actor', optimizer.get('critic', {}))
            if hasattr(self.current_lr, 'param_groups'):
                self.current_lr = self.current_lr.param_groups[0]['lr']
        
        # 课程学习阶段更新
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

    def analyze_gelu_distribution(self):
        """
        分析每层 GELU(x) 输入 x 的分布，判断哪些层有资格使用 degree 0。
        
        通过在每层的 intermediate_act_fn 上安装前向钩子，运行一次训练集前向传播，
        统计 x 落在 [-2.7, 0) 区间的比例。若 >= GELU_DEGREE0_THRESHOLD (80%)，
        则该层有资格使用 degree 0 动作。
        
        Returns:
            gelu_degree0_eligible: np.ndarray[bool], shape=(total_layers,)
        """
        interval_counts = [np.zeros(4, dtype=np.int64) for _ in range(self.total_layers)]
        hooks = []
        layers = eval('self.model.' + self.layers_attribute)

        def _make_hook(layer_idx):
            @torch.no_grad()
            def hook_fn(module, input_tensor, output_tensor):
                x = input_tensor[0].detach().reshape(-1)
                interval_counts[layer_idx][0] += (x < -2.7).sum().item()
                interval_counts[layer_idx][1] += ((x >= -2.7) & (x < 0)).sum().item()
                interval_counts[layer_idx][2] += ((x >= 0) & (x <= 2.7)).sum().item()
                interval_counts[layer_idx][3] += (x > 2.7).sum().item()
            return hook_fn

        for i, layer in enumerate(layers):
            act_fn = layer.intermediate.intermediate_act_fn
            h = act_fn.register_forward_hook(_make_hook(i)) if isinstance(act_fn, nn.Module) else None
            if h is not None:
                hooks.append(h)
            else:
                hooks.append(
                    layer.intermediate.register_forward_hook(
                        lambda mod, inp, out, idx=i: _make_hook(idx)(mod, inp, out)
                    )
                )

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for batch in self.dataloader_train:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.model(**batch)

        for h in hooks:
            h.remove()

        eligible = np.zeros(self.total_layers, dtype=bool)
        for i in range(self.total_layers):
            ic = interval_counts[i]
            total = ic.sum()
            if total > 0:
                neg_ratio = ic[1] / total
                eligible[i] = neg_ratio >= GELU_DEGREE0_THRESHOLD

        return eligible, interval_counts

    def log(self, message):
        print(message, flush=True)
        target_log_file = getattr(self, "active_log_file", self.log_file)
        with open(target_log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def get_simulated_cost(self, gelu_degrees, softmax_degrees):
        g_c = sum(self.GELU_COST_MAP.get(d, 0) for d in gelu_degrees)
        s_c = sum(self.SOFTMAX_COST_MAP.get(d, 0) for d in softmax_degrees)
        return g_c + s_c, g_c, s_c

    def apply_configuration(self, gelu_degrees, softmax_degrees):
        handler_layer_name = "model." + self.layers_attribute
        # GELU (including degree 0)
        gelu_map = {d: [] for d in [0, 1, 2, 4]} 
        for idx, deg in enumerate(gelu_degrees):
            if deg in gelu_map: gelu_map[deg].append(idx)
        for d in [0, 1, 2, 4]:
            if gelu_map[d]:
                self.reversible_handler.replace_layer_gelu(gelu_map[d], handler_layer_name, degree=d)
        # Softmax
        softmax_map = {d: [] for d in range(2, 7)}
        for idx, deg in enumerate(softmax_degrees):
            if deg in softmax_map: softmax_map[deg].append(idx)
        for d in range(2, 7):
            if softmax_map[d]:
                self.reversible_handler.replace_layer_softmax(softmax_map[d], handler_layer_name, degree=d)

    def validate_input_noise_scaling_factors(self, scaling_factors):
        arr = np.asarray(scaling_factors, dtype=int)
        if arr.shape != (self.total_layers,):
            raise ValueError(
                f"input_noise_scaling_factors must have shape ({self.total_layers},), "
                f"but got {arr.shape}"
            )
        invalid = sorted(set(arr.tolist()) - set(INPUT_NOISE_ALLOWED_SCALING_FACTORS))
        if invalid:
            raise ValueError(
                f"Unsupported input-noise scaling factors: {invalid}. "
                f"Allowed values: {list(INPUT_NOISE_ALLOWED_SCALING_FACTORS)}"
            )
        return arr

    def validate_weight_noise_scaling_factors(
            self,
            scaling_factors,
            noise_name,
            allowed_values=None,
    ):
        arr = np.asarray(scaling_factors, dtype=int)
        if arr.shape != (self.total_layers,):
            raise ValueError(
                f"{noise_name}_noise_scaling_factors must have shape ({self.total_layers},), "
                f"but got {arr.shape}"
            )
        allowed_values = tuple(allowed_values or WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        invalid = sorted(set(arr.tolist()) - set(allowed_values))
        if invalid:
            raise ValueError(
                f"Unsupported {noise_name}-noise scaling factors: {invalid}. "
                f"Allowed values: {list(allowed_values)}"
            )
        return arr

    def _apply_weight_noise_configuration(
            self,
            scaling_factors,
            noise_name,
            replace_method_name,
            state_attr,
            allowed_values=None,
    ):
        allowed_values = tuple(allowed_values or WEIGHT_NOISE_ALLOWED_SCALING_FACTORS)
        arr = self.validate_weight_noise_scaling_factors(
            scaling_factors,
            noise_name,
            allowed_values=allowed_values,
        )
        handler_layer_name = "model." + self.layers_attribute
        scaling_map = {sf: [] for sf in allowed_values}
        for idx, scaling_factor in enumerate(arr):
            scaling_map[int(scaling_factor)].append(idx)
        replace_method = getattr(self.reversible_handler, replace_method_name)
        for scaling_factor in allowed_values:
            if scaling_map[scaling_factor]:
                replace_method(
                    scaling_map[scaling_factor],
                    handler_layer_name,
                    scaling_factor=scaling_factor,
                    distribution="encoding",
                )
        setattr(self, state_attr, arr.copy())

    def _clear_weight_noise_configuration(
            self,
            restore_method_name,
            state_attr,
            default_value=None,
    ):
        handler_layer_name = "model." + self.layers_attribute
        restore_method = getattr(self.reversible_handler, restore_method_name)
        restore_method(
            list(range(self.total_layers)),
            handler_layer_name,
        )
        setattr(
            self,
            state_attr,
            np.full(
                self.total_layers,
                (
                    WEIGHT_NOISE_DEFAULT_SCALING_FACTOR
                    if default_value is None
                    else int(default_value)
                ),
                dtype=int,
            ),
        )

    def apply_input_noise_configuration(self, input_noise_scaling_factors):
        """Apply layer-wise x-noise scaling factors for the active second-stage RL."""
        arr = self.validate_input_noise_scaling_factors(input_noise_scaling_factors)
        handler_layer_name = "model." + self.layers_attribute
        scaling_map = {sf: [] for sf in INPUT_NOISE_ALLOWED_SCALING_FACTORS}
        for idx, scaling_factor in enumerate(arr):
            scaling_map[int(scaling_factor)].append(idx)
        for scaling_factor in INPUT_NOISE_ALLOWED_SCALING_FACTORS:
            if scaling_map[scaling_factor]:
                self.reversible_handler.replace_layer_input_noise(
                    scaling_map[scaling_factor],
                    handler_layer_name,
                    scaling_factor=scaling_factor,
                    distribution="fresh",
                )
        self.current_input_noise_scaling_factors = arr.copy()

    def clear_input_noise_configuration(self):
        handler_layer_name = "model." + self.layers_attribute
        self.reversible_handler.restore_layer_input_noise(
            list(range(self.total_layers)),
            handler_layer_name,
        )
        self.current_input_noise_scaling_factors = np.full(
            self.total_layers,
            INPUT_NOISE_DEFAULT_SCALING_FACTOR,
            dtype=int,
        )

    def apply_wq_noise_configuration(self, wq_noise_scaling_factors):
        """Apply layer-wise Wq encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wq_noise_scaling_factors,
            noise_name="wq",
            replace_method_name="replace_layer_query_noise",
            state_attr="current_wq_noise_scaling_factors",
        )

    def clear_wq_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_query_noise",
            state_attr="current_wq_noise_scaling_factors",
        )

    def apply_wk_noise_configuration(self, wk_noise_scaling_factors):
        """Apply layer-wise Wk encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wk_noise_scaling_factors,
            noise_name="wk",
            replace_method_name="replace_layer_key_noise",
            state_attr="current_wk_noise_scaling_factors",
        )

    def clear_wk_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_key_noise",
            state_attr="current_wk_noise_scaling_factors",
        )

    def apply_wv_noise_configuration(self, wv_noise_scaling_factors):
        """Apply layer-wise Wv encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wv_noise_scaling_factors,
            noise_name="wv",
            replace_method_name="replace_layer_value_noise",
            state_attr="current_wv_noise_scaling_factors",
        )

    def clear_wv_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_value_noise",
            state_attr="current_wv_noise_scaling_factors",
        )

    def apply_wo_noise_configuration(self, wo_noise_scaling_factors):
        """Apply layer-wise Wo encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wo_noise_scaling_factors,
            noise_name="wo",
            replace_method_name="replace_layer_attention_output_noise",
            state_attr="current_wo_noise_scaling_factors",
        )

    def clear_wo_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_attention_output_noise",
            state_attr="current_wo_noise_scaling_factors",
        )

    def apply_wffn1_noise_configuration(self, wffn1_noise_scaling_factors):
        """Apply layer-wise Wffn1 encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wffn1_noise_scaling_factors,
            noise_name="wffn1",
            replace_method_name="replace_layer_ffn1_noise",
            state_attr="current_wffn1_noise_scaling_factors",
            allowed_values=WFFN1_NOISE_ALLOWED_SCALING_FACTORS,
        )

    def clear_wffn1_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_ffn1_noise",
            state_attr="current_wffn1_noise_scaling_factors",
            default_value=WFFN1_NOISE_DEFAULT_SCALING_FACTOR,
        )

    def apply_wffn2_noise_configuration(self, wffn2_noise_scaling_factors):
        """Apply layer-wise Wffn2 encoding-noise scaling factors."""
        self._apply_weight_noise_configuration(
            wffn2_noise_scaling_factors,
            noise_name="wffn2",
            replace_method_name="replace_layer_ffn2_noise",
            state_attr="current_wffn2_noise_scaling_factors",
        )

    def clear_wffn2_noise_configuration(self):
        self._clear_weight_noise_configuration(
            restore_method_name="restore_layer_ffn2_noise",
            state_attr="current_wffn2_noise_scaling_factors",
        )

    def apply_weight_noise_configuration(
            self,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None,
            wo_noise_scaling_factors=None,
            wffn1_noise_scaling_factors=None,
            wffn2_noise_scaling_factors=None
            ):
        if wq_noise_scaling_factors is not None:
            self.apply_wq_noise_configuration(wq_noise_scaling_factors)
        if wk_noise_scaling_factors is not None:
            self.apply_wk_noise_configuration(wk_noise_scaling_factors)
        if wv_noise_scaling_factors is not None:
            self.apply_wv_noise_configuration(wv_noise_scaling_factors)
        if wo_noise_scaling_factors is not None:
            self.apply_wo_noise_configuration(wo_noise_scaling_factors)
        if wffn1_noise_scaling_factors is not None:
            self.apply_wffn1_noise_configuration(wffn1_noise_scaling_factors)
        if wffn2_noise_scaling_factors is not None:
            self.apply_wffn2_noise_configuration(wffn2_noise_scaling_factors)

    def clear_weight_noise_configuration(self):
        self.clear_wq_noise_configuration()
        self.clear_wk_noise_configuration()
        self.clear_wv_noise_configuration()
        self.clear_wo_noise_configuration()
        self.clear_wffn1_noise_configuration()
        self.clear_wffn2_noise_configuration()

    def apply_qkv_noise_configuration(
            self,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None
            ):
        """Backward-compatible alias for the earlier QKV-only helper."""
        self.apply_weight_noise_configuration(
            wq_noise_scaling_factors=wq_noise_scaling_factors,
            wk_noise_scaling_factors=wk_noise_scaling_factors,
            wv_noise_scaling_factors=wv_noise_scaling_factors,
        )

    def clear_qkv_noise_configuration(self):
        """Backward-compatible alias for the earlier QKV-only helper."""
        self.clear_wq_noise_configuration()
        self.clear_wk_noise_configuration()
        self.clear_wv_noise_configuration()

    def evaluate_model_with_input_noise(
            self,
            gelu_degrees,
            softmax_degrees,
            input_noise_scaling_factors,
            use_train=True,
            split=None
            ):
        """Evaluate a fixed GELU/Softmax config with layer-wise x-noise enabled."""
        self.apply_configuration(gelu_degrees, softmax_degrees)
        self.apply_input_noise_configuration(input_noise_scaling_factors)
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        dataloader = self.dataloaders[split_name]
        try:
            return self._run_evaluation(
                dataloader,
                use_train=(split_name == "train"),
                split_name=split_name,
            )
        finally:
            self.clear_input_noise_configuration()

    def evaluate_model_with_attention_noise(
            self,
            gelu_degrees,
            softmax_degrees,
            input_noise_scaling_factors=None,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None,
            wo_noise_scaling_factors=None,
            wffn1_noise_scaling_factors=None,
            wffn2_noise_scaling_factors=None,
            use_train=True,
            split=None
            ):
        """Evaluate a fixed GELU/Softmax config with second-stage noise enabled."""
        self.apply_configuration(gelu_degrees, softmax_degrees)
        input_noise_enabled = input_noise_scaling_factors is not None
        weight_noise_enabled = any(
            config is not None
            for config in (
                wq_noise_scaling_factors,
                wk_noise_scaling_factors,
                wv_noise_scaling_factors,
                wo_noise_scaling_factors,
                wffn1_noise_scaling_factors,
                wffn2_noise_scaling_factors,
            )
        )
        if input_noise_enabled:
            self.apply_input_noise_configuration(input_noise_scaling_factors)
        if weight_noise_enabled:
            self.apply_weight_noise_configuration(
                wq_noise_scaling_factors=wq_noise_scaling_factors,
                wk_noise_scaling_factors=wk_noise_scaling_factors,
                wv_noise_scaling_factors=wv_noise_scaling_factors,
                wo_noise_scaling_factors=wo_noise_scaling_factors,
                wffn1_noise_scaling_factors=wffn1_noise_scaling_factors,
                wffn2_noise_scaling_factors=wffn2_noise_scaling_factors,
            )
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        dataloader = self.dataloaders[split_name]
        try:
            return self._run_evaluation(
                dataloader,
                use_train=(split_name == "train"),
                split_name=split_name,
            )
        finally:
            if weight_noise_enabled:
                self.clear_weight_noise_configuration()
            if input_noise_enabled:
                self.clear_input_noise_configuration()

    def build_constraint_limits_from_metrics(
            self,
            base_loss,
            base_p,
            base_s,
            error_threshold=None,
            correlation_drop_ratio=None,
            ):
        error_threshold = (
            self.error_threshold
            if error_threshold is None
            else float(error_threshold)
        )
        correlation_drop_ratio = (
            self.correlation_drop_ratio
            if correlation_drop_ratio is None
            else float(correlation_drop_ratio)
        )
        return {
            "loss": float(base_loss + error_threshold),
            "metric1": float(base_p * (1.0 - correlation_drop_ratio)),
            "metric2": float(base_s * (1.0 - correlation_drop_ratio)),
        }

    def evaluate_model_with_attention_noise_repeated(
            self,
            gelu_degrees,
            softmax_degrees,
            input_noise_scaling_factors=None,
            wq_noise_scaling_factors=None,
            wk_noise_scaling_factors=None,
            wv_noise_scaling_factors=None,
            wo_noise_scaling_factors=None,
            wffn1_noise_scaling_factors=None,
            wffn2_noise_scaling_factors=None,
            repeats=1,
            use_train=True,
            split=None,
            ):
        repeats = max(1, int(repeats))
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        trials = []
        for _ in range(repeats):
            loss, p, s, t = self.evaluate_model_with_attention_noise(
                gelu_degrees,
                softmax_degrees,
                input_noise_scaling_factors=input_noise_scaling_factors,
                wq_noise_scaling_factors=wq_noise_scaling_factors,
                wk_noise_scaling_factors=wk_noise_scaling_factors,
                wv_noise_scaling_factors=wv_noise_scaling_factors,
                wo_noise_scaling_factors=wo_noise_scaling_factors,
                wffn1_noise_scaling_factors=wffn1_noise_scaling_factors,
                wffn2_noise_scaling_factors=wffn2_noise_scaling_factors,
                use_train=(split_name == "train"),
                split=split_name,
            )
            trials.append(
                {
                    "loss": float(loss),
                    "p": float(p),
                    "s": float(s),
                    "time_ms": float(t),
                }
            )

        losses = np.asarray([trial["loss"] for trial in trials], dtype=float)
        ps = np.asarray([trial["p"] for trial in trials], dtype=float)
        ss = np.asarray([trial["s"] for trial in trials], dtype=float)
        times = np.asarray([trial["time_ms"] for trial in trials], dtype=float)
        return {
            "split_name": split_name,
            "n": repeats,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "loss_min": float(np.min(losses)),
            "loss_max": float(np.max(losses)),
            "loss_range": float(np.max(losses) - np.min(losses)),
            "p_mean": float(np.mean(ps)),
            "p_std": float(np.std(ps)),
            "p_min": float(np.min(ps)),
            "p_max": float(np.max(ps)),
            "p_range": float(np.max(ps) - np.min(ps)),
            "s_mean": float(np.mean(ss)),
            "s_std": float(np.std(ss)),
            "s_min": float(np.min(ss)),
            "s_max": float(np.max(ss)),
            "s_range": float(np.max(ss) - np.min(ss)),
            "time_mean_ms": float(np.mean(times)),
            "time_std_ms": float(np.std(times)),
            "trials": trials,
        }

    def get_stage1_exact_baseline_configuration(self):
        return (
            np.full(self.total_layers, 4, dtype=int),
            np.full(self.total_layers, 6, dtype=int),
        )

    def evaluate_model_repeated(
            self,
            gelu_degrees,
            softmax_degrees,
            repeats=1,
            use_train=True,
            split=None,
            ):
        repeats = max(1, int(repeats))
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        trials = []
        for _ in range(repeats):
            loss, p, s, t = self.evaluate_model(
                gelu_degrees,
                softmax_degrees,
                use_train=(split_name == "train"),
                split=split_name,
            )
            trials.append(
                {
                    "loss": float(loss),
                    "p": float(p),
                    "s": float(s),
                    "time_ms": float(t),
                }
            )

        losses = np.asarray([trial["loss"] for trial in trials], dtype=float)
        ps = np.asarray([trial["p"] for trial in trials], dtype=float)
        ss = np.asarray([trial["s"] for trial in trials], dtype=float)
        times = np.asarray([trial["time_ms"] for trial in trials], dtype=float)
        return {
            "split_name": split_name,
            "n": repeats,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "loss_min": float(np.min(losses)),
            "loss_max": float(np.max(losses)),
            "loss_range": float(np.max(losses) - np.min(losses)),
            "p_mean": float(np.mean(ps)),
            "p_std": float(np.std(ps)),
            "p_min": float(np.min(ps)),
            "p_max": float(np.max(ps)),
            "p_range": float(np.max(ps) - np.min(ps)),
            "s_mean": float(np.mean(ss)),
            "s_std": float(np.std(ss)),
            "s_min": float(np.min(ss)),
            "s_max": float(np.max(ss)),
            "s_range": float(np.max(ss) - np.min(ss)),
            "time_mean_ms": float(np.mean(times)),
            "time_std_ms": float(np.std(times)),
            "trials": trials,
        }

    def get_noise_simulated_cost(
            self,
            input_noise_scaling_factors,
            wq_noise_scaling_factors,
            wk_noise_scaling_factors,
            wv_noise_scaling_factors,
            wo_noise_scaling_factors,
            wffn1_noise_scaling_factors,
            wffn2_noise_scaling_factors,
            ):
        breakdown = {
            "x": float(sum(self.INPUT_NOISE_COST_MAP[int(v)] for v in np.asarray(input_noise_scaling_factors, dtype=int))),
            "wq": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wq_noise_scaling_factors, dtype=int))),
            "wk": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wk_noise_scaling_factors, dtype=int))),
            "wv": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wv_noise_scaling_factors, dtype=int))),
            "wo": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wo_noise_scaling_factors, dtype=int))),
            "wffn1": float(sum(self.WFFN1_NOISE_COST_MAP[int(v)] for v in np.asarray(wffn1_noise_scaling_factors, dtype=int))),
            "wffn2": float(sum(self.WEIGHT_NOISE_COST_MAP[int(v)] for v in np.asarray(wffn2_noise_scaling_factors, dtype=int))),
        }
        return float(sum(breakdown.values())), breakdown

    def _get_max_noise_configuration(self):
        return {
            "input_noise_scaling_factors": np.full(self.total_layers, max(INPUT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wq_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wk_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wv_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wo_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wffn1_noise_scaling_factors": np.full(self.total_layers, max(WFFN1_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
            "wffn2_noise_scaling_factors": np.full(self.total_layers, max(WEIGHT_NOISE_ALLOWED_SCALING_FACTORS), dtype=int),
        }

    def _reset_runtime_ppo_state(self):
        self.current_episode = 0
        self.current_entropy_coef = PPO_ENTROPY_INITIAL
        self.current_lr = self.ppo_lr_initial
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.return_normalizer = RunningMeanStd()

    def run_noise_rl_stage(self, fixed_gelu, fixed_softmax, fixed_label, fixed_source):
        from noise_rl_module import NoiseRLModule
        module = NoiseRLModule(self)
        return module.run(fixed_gelu, fixed_softmax, fixed_label, fixed_source)

    def run_noise_final_eval_stage(self, fixed_gelu, fixed_softmax, noise_stage_result=None):
        from noise_final_evaluation_module import NoiseFinalEvaluationModule
        import numpy as np

        fixed_gelu = np.asarray(fixed_gelu, dtype=int)
        fixed_softmax = np.asarray(fixed_softmax, dtype=int)

        if noise_stage_result is not None:
            baseline_noise_config = noise_stage_result["baseline_noise_config"]
            baseline_tot_c = noise_stage_result["baseline_tot_c"]
            best_noise_cfg = noise_stage_result["best_noise_config"]
            search_best = None
            if best_noise_cfg is not None:
                search_best = {
                    k: v for k, v in best_noise_cfg.items()
                    if k.endswith("scaling_factors")
                }
            limit_loss = noise_stage_result["limit_loss"]
            limit_p = noise_stage_result["limit_p"]
            limit_s = noise_stage_result["limit_s"]
            search_status = noise_stage_result.get("status")
        else:
            baseline_noise_config = self._get_max_noise_configuration()
            baseline_tot_c, _ = self.get_noise_simulated_cost(**baseline_noise_config)
            search_best = None

            exact_baseline_gelu, exact_baseline_softmax = (
                self.get_stage1_exact_baseline_configuration()
            )
            baseline_summary = self.evaluate_model_repeated(
                exact_baseline_gelu,
                exact_baseline_softmax,
                repeats=self.noise_eval_repeat_n,
                split=self.get_reward_reference_split_name(),
            )
            base_loss = baseline_summary["loss_mean"]
            base_p = baseline_summary["p_mean"]
            base_s = baseline_summary["s_mean"]
            selection_limits = self.build_constraint_limits_from_metrics(
                base_loss,
                base_p,
                base_s,
            )
            limit_loss = selection_limits["loss"]
            limit_p = selection_limits["metric1"]
            limit_s = selection_limits["metric2"]
            search_status = None

        runner = NoiseFinalEvaluationModule(
            evaluator=self,
            config_source=self.noise_eval_config_source,
            config_path=self.noise_eval_config_path,
            manual_noise_config=self.manual_noise_config,
            random_seed=self.final_eval_random_seed,
            permutation_trials=self.final_eval_permutation_trials,
            cost_equivalent_trials=self.final_eval_cost_equivalent_trials,
            budget_equivalent_trials=self.final_eval_budget_equivalent_trials,
            repeat_n=self.noise_eval_repeat_n,
            results_dir=self.noise_final_eval_dir,
        )

        return runner.run(
            search_best_noise_config=search_best,
            search_status=search_status,
            fixed_gelu=fixed_gelu,
            fixed_softmax=fixed_softmax,
            baseline_noise_config=baseline_noise_config,
            baseline_tot_c=baseline_tot_c,
            limit_loss=limit_loss,
            limit_p=limit_p,
            limit_s=limit_s,
        )

    def _run_evaluation(self, dataloader, use_train=False, split_name=None):
        """在当前模型上运行评估循环（不修改配置），用于无近似对照组等。"""
        self.model.eval()
        self.model.to(self.device)
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
        ds = self.dataset_key
        try:
            if ds == 'stsb':
                metric1 = pearsonr(all_preds, all_labels)[0]
                metric2 = spearmanr(all_preds, all_labels)[0]
            elif ds == 'mrpc':
                pred_classes = self._logits_to_classes(all_preds)
                metric1 = accuracy_score(all_labels, pred_classes)
                metric2 = f1_score(all_labels, pred_classes, average='weighted')
            elif ds == 'cola':
                pred_classes = self._logits_to_classes(all_preds)
                metric1 = matthews_corrcoef(all_labels, pred_classes)
                metric2 = metric1
            elif ds == 'mnli':
                pred_classes = self._logits_to_classes(all_preds)
                metric1 = accuracy_score(all_labels, pred_classes)
                mm_dataloader = None
                if split_name is not None:
                    mm_dataloader = self.dataloaders_mm.get(split_name)
                elif not use_train:
                    mm_dataloader = self.dataloader_test_mm
                if not use_train and mm_dataloader is not None:
                    metric2 = self._evaluate_accuracy_on_dataloader(mm_dataloader)
                else:
                    metric2 = metric1
            else:
                pred_classes = self._logits_to_classes(all_preds)
                metric1 = accuracy_score(all_labels, pred_classes)
                metric2 = metric1
        except Exception as e:
            print(f"[警告] 为数据集（dataset）'{ds}' 计算指标失败: {e}")
            metric1, metric2 = 0.0, 0.0
        return avg_loss, metric1, metric2, avg_time

    def evaluate_model(self, gelu_degrees, softmax_degrees, use_train=True, split=None):
        """评估模型，use_train=True时使用训练集，否则使用验证集"""
        self.apply_configuration(gelu_degrees, softmax_degrees)
        split_name = self._resolve_eval_split(use_train=use_train, split=split)
        dataloader = self.dataloaders[split_name]
        return self._run_evaluation(
            dataloader,
            use_train=(split_name == "train"),
            split_name=split_name,
        )

    @staticmethod
    def _logits_to_classes(all_preds):
        """将 logits 转换为预测类别"""
        preds_arr = np.array(all_preds)
        if len(preds_arr.shape) == 1:
            return (preds_arr > 0.5).astype(int)
        return np.argmax(preds_arr, axis=1)

    def _evaluate_accuracy_on_dataloader(self, dataloader):
        """在指定 dataloader 上计算 accuracy（用于 MNLI mismatched）"""
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"].detach().cpu().numpy()
                outputs = self.model(**batch)
                logits = outputs.logits.squeeze().detach().cpu().numpy()
                if np.ndim(logits) == 0:
                    logits = [logits]
                all_preds.extend(logits)
                all_labels.extend(labels)
        pred_classes = self._logits_to_classes(all_preds)
        return accuracy_score(all_labels, pred_classes)

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

    def ppo_update_lstm(self, lstm_net, optimizer, buffer, device, 
                        mini_batch_episodes=LSTM_MINI_BATCH_EPISODES, entropy_coef=None):
        """
        LSTM版PPO更新（LSTM PDF 4.2：Full-Episode BPTT）
        
        关键区别：
        - 以完整Episode为单位进行mini-batch处理，保持LSTM序列完整性
        - 梯度通过12步时间序列完整反向传播（BPTT）
        - Mini-batch按Episode数划分（而非按步数）
        
        Args:
            lstm_net: LSTM策略价值网络
            optimizer: 优化器
            buffer: RecurrentRolloutBuffer
            device: 设备
            mini_batch_episodes: 每个mini-batch包含的episode数
            entropy_coef: 动态熵系数
        """
        if entropy_coef is None:
            entropy_coef = self.get_current_entropy_coef()
        
        # 获取所有episode的batch数据
        (cont_features, layer_indices, prev_g_actions, prev_s_actions,
         actions_g, actions_s, old_logprobs, rewards, values, dones, gelu_masks) = buffer.get_batch(device)
        
        n_eps = cont_features.size(0)
        seq_len = cont_features.size(1)
        
        # 计算每个Episode的GAE（保持序列独立性）
        all_advantages = []
        all_returns = []
        for i in range(n_eps):
            ep_rewards = rewards[i].cpu().numpy()
            ep_values = values[i].cpu().numpy()
            ep_dones = dones[i].cpu().numpy()
            adv, ret = self.compute_gae(ep_rewards, ep_values, ep_dones)
            all_advantages.append(adv)
            all_returns.append(ret)
        
        advantages = torch.stack(all_advantages).to(device)  # (N_eps, 12)
        returns = torch.stack(all_returns).to(device)         # (N_eps, 12)
        
        # 标准化优势（跨所有episode展平后标准化）
        adv_flat = advantages.reshape(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        # Return Normalization（PopArt风格, PDF 3.3.2 + 5）
        self.return_normalizer.update(returns)
        returns_normalized = torch.tensor(
            self.return_normalizer.normalize(returns.cpu().numpy()),
            dtype=torch.float32
        ).to(device)
        values_normalized = torch.tensor(
            self.return_normalizer.normalize(values.cpu().numpy()),
            dtype=torch.float32
        ).to(device)
        
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        
        # PPO K-Epoch更新
        for epoch in range(PPO_K_EPOCHS):
            # Shuffle episodes（不打乱步序，保持每个episode内的时间顺序）
            ep_indices = torch.randperm(n_eps)
            
            # 按Episode mini-batch更新
            for start in range(0, n_eps, mini_batch_episodes):
                end = min(start + mini_batch_episodes, n_eps)
                mb_idx = ep_indices[start:end]
                
                # 获取mini-batch数据（保持 (mb_size, 12, ...) 形状）
                mb_cont = cont_features[mb_idx]
                mb_layer = layer_indices[mb_idx]
                mb_prev_g = prev_g_actions[mb_idx]
                mb_prev_s = prev_s_actions[mb_idx]
                mb_act_g = actions_g[mb_idx]
                mb_act_s = actions_s[mb_idx]
                mb_old_lp = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns_normalized[mb_idx]
                mb_old_val = values_normalized[mb_idx]
                mb_gelu_mask = gelu_masks[mb_idx] if gelu_masks is not None else None
                
                # Full-Episode BPTT前向传播（LSTM PDF 4.2）
                new_logprobs, entropy, new_values_raw = lstm_net.evaluate_actions(
                    mb_cont, mb_layer, mb_prev_g, mb_prev_s, mb_act_g, mb_act_s,
                    gelu_mask=mb_gelu_mask
                )
                
                # 展平为 (mb_size * 12,) 用于损失计算
                new_logprobs_flat = new_logprobs.reshape(-1)
                entropy_flat = entropy.reshape(-1)
                new_values_flat = new_values_raw.reshape(-1)
                mb_old_lp_flat = mb_old_lp.reshape(-1)
                mb_adv_flat = mb_adv.reshape(-1)
                mb_ret_flat = mb_ret.reshape(-1)
                mb_old_val_flat = mb_old_val.reshape(-1)
                
                # PPO-Clip策略损失
                ratios = torch.exp(new_logprobs_flat - mb_old_lp_flat)
                surr1 = ratios * mb_adv_flat
                surr2 = torch.clamp(ratios, 1 - PPO_EPS_CLIP, 1 + PPO_EPS_CLIP) * mb_adv_flat
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失（PopArt归一化 + Value Clipping + Huber Loss）
                new_values_norm = (new_values_flat - self.return_normalizer.mean) / self.return_normalizer.std
                value_clipped = mb_old_val_flat + torch.clamp(
                    new_values_norm - mb_old_val_flat,
                    -VALUE_CLIP_RANGE, VALUE_CLIP_RANGE
                )
                huber_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
                vl_unclipped = huber_loss_fn(new_values_norm, mb_ret_flat)
                vl_clipped = huber_loss_fn(value_clipped, mb_ret_flat)
                value_loss = torch.max(vl_unclipped, vl_clipped).mean()
                
                # 熵正则项
                entropy_loss = -entropy_flat.mean()
                
                # 总损失
                loss = policy_loss + PPO_VALUE_COEF * value_loss + entropy_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(lstm_net.parameters(), 0.5)
                optimizer.step()
                
                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = entropy_flat.mean().item()
        
        return last_policy_loss, last_value_loss, last_entropy

    def ppo_update_gtrxl(self, gtrxl_net, optimizer, buffer, device,
                          mini_batch_episodes=GTRXL_MINI_BATCH_EPISODES, entropy_coef=None,
                          ppo_update_step=0):
        """
        GTrXL版PPO更新（Transformer PDF 4.2-4.3）
        
        与LSTM版的关键差异：
        - 学习率线性预热（前 GTRXL_WARMUP_STEPS 步）
        - 最小熵约束：当策略熵低于下界时，动态提升熵正则化系数
        - evaluate_actions 无 hidden state，因果掩码自动处理时序依赖
        
        Args:
            gtrxl_net: GTrXL策略价值网络
            optimizer: 优化器
            buffer: RecurrentRolloutBuffer
            device: 设备
            mini_batch_episodes: 每个mini-batch包含的episode数
            entropy_coef: 动态熵系数
            ppo_update_step: 当前PPO更新步数（用于学习率Warmup）
        """
        if entropy_coef is None:
            entropy_coef = self.get_current_entropy_coef()
        
        # 学习率 Warmup（PDF 4.3）
        if ppo_update_step < GTRXL_WARMUP_STEPS:
            warmup_factor = (ppo_update_step + 1) / GTRXL_WARMUP_STEPS
            current_lr = self.ppo_lr_initial * warmup_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        (cont_features, layer_indices, prev_g_actions, prev_s_actions,
         actions_g, actions_s, old_logprobs, rewards, values, dones, gelu_masks) = buffer.get_batch(device)
        
        n_eps = cont_features.size(0)
        
        all_advantages = []
        all_returns = []
        for i in range(n_eps):
            ep_rewards = rewards[i].cpu().numpy()
            ep_values = values[i].cpu().numpy()
            ep_dones = dones[i].cpu().numpy()
            adv, ret = self.compute_gae(ep_rewards, ep_values, ep_dones)
            all_advantages.append(adv)
            all_returns.append(ret)
        
        advantages = torch.stack(all_advantages).to(device)
        returns = torch.stack(all_returns).to(device)
        
        adv_flat = advantages.reshape(-1)
        advantages = (advantages - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        self.return_normalizer.update(returns)
        returns_normalized = torch.tensor(
            self.return_normalizer.normalize(returns.cpu().numpy()),
            dtype=torch.float32
        ).to(device)
        values_normalized = torch.tensor(
            self.return_normalizer.normalize(values.cpu().numpy()),
            dtype=torch.float32
        ).to(device)
        
        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        
        for epoch in range(PPO_K_EPOCHS):
            ep_indices = torch.randperm(n_eps)
            
            for start in range(0, n_eps, mini_batch_episodes):
                end = min(start + mini_batch_episodes, n_eps)
                mb_idx = ep_indices[start:end]
                
                mb_cont = cont_features[mb_idx]
                mb_layer = layer_indices[mb_idx]
                mb_prev_g = prev_g_actions[mb_idx]
                mb_prev_s = prev_s_actions[mb_idx]
                mb_act_g = actions_g[mb_idx]
                mb_act_s = actions_s[mb_idx]
                mb_old_lp = old_logprobs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns_normalized[mb_idx]
                mb_old_val = values_normalized[mb_idx]
                mb_gelu_mask = gelu_masks[mb_idx] if gelu_masks is not None else None
                
                new_logprobs, entropy, new_values_raw = gtrxl_net.evaluate_actions(
                    mb_cont, mb_layer, mb_prev_g, mb_prev_s, mb_act_g, mb_act_s,
                    gelu_mask=mb_gelu_mask
                )
                
                new_logprobs_flat = new_logprobs.reshape(-1)
                entropy_flat = entropy.reshape(-1)
                new_values_flat = new_values_raw.reshape(-1)
                mb_old_lp_flat = mb_old_lp.reshape(-1)
                mb_adv_flat = mb_adv.reshape(-1)
                mb_ret_flat = mb_ret.reshape(-1)
                mb_old_val_flat = mb_old_val.reshape(-1)
                
                ratios = torch.exp(new_logprobs_flat - mb_old_lp_flat)
                surr1 = ratios * mb_adv_flat
                surr2 = torch.clamp(ratios, 1 - PPO_EPS_CLIP, 1 + PPO_EPS_CLIP) * mb_adv_flat
                policy_loss = -torch.min(surr1, surr2).mean()
                
                new_values_norm = (new_values_flat - self.return_normalizer.mean) / self.return_normalizer.std
                value_clipped = mb_old_val_flat + torch.clamp(
                    new_values_norm - mb_old_val_flat,
                    -VALUE_CLIP_RANGE, VALUE_CLIP_RANGE
                )
                huber_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
                vl_unclipped = huber_loss_fn(new_values_norm, mb_ret_flat)
                vl_clipped = huber_loss_fn(value_clipped, mb_ret_flat)
                value_loss = torch.max(vl_unclipped, vl_clipped).mean()
                
                # 最小熵约束（PDF 4.3）：当策略熵低于下界时提升熵正则化
                mean_entropy = entropy_flat.mean()
                effective_entropy_coef = entropy_coef
                if mean_entropy.item() < GTRXL_ENTROPY_LOWER_BOUND:
                    entropy_deficit = GTRXL_ENTROPY_LOWER_BOUND - mean_entropy.item()
                    effective_entropy_coef = entropy_coef + 10.0 * entropy_deficit
                
                entropy_loss = -mean_entropy
                
                loss = policy_loss + PPO_VALUE_COEF * value_loss + effective_entropy_coef * entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(gtrxl_net.parameters(), 0.5)
                optimizer.step()
                
                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = mean_entropy.item()
        
        return last_policy_loss, last_value_loss, last_entropy

    def on_evaluate(self, args, state, control, **kwargs):
        self.log("\n" + "="*60)
        self.log("开始配置评估（STARTING CONFIGURATION EVALUATION）")
        self.log(f"搜索模式（SEARCH_MODE）={SEARCH_MODE}" + (" (使用贪心搜索USE_GREEDY_SEARCH=" + str(USE_GREEDY_SEARCH) + ")" if SEARCH_MODE == "both" else ""))
        self.log(f"最终评估配置来源（FINAL_EVAL_CONFIG_SOURCE）={self.final_eval_config_source}")
        if self.skip_stage1_rl:
            self.log("[信息] 第一阶段RL/贪心搜索已跳过（--skip-stage1-rl）。")
        self.log("="*60)

        base_gelu = np.full(self.total_layers, 4, dtype=int)
        base_softmax = np.full(self.total_layers, 6, dtype=int)
        base_tot_c, base_g_c, base_s_c = self.get_simulated_cost(base_gelu, base_softmax)
        num_metrics = self.get_num_metrics()
        base_loss = base_p = base_s = None
        base_loss_train = base_p_train = base_s_train = None
        gelu_degree0_eligible = np.zeros(self.total_layers, dtype=bool)
        reward_reference_split = self.get_reward_reference_split_name()

        if self.skip_stage1_rl:
            self.log("\n--- 阶段1（Phase 1）: 已跳过（SKIPPED）（--skip-stage1-rl） ---")
            self.log("[信息] 基线建立已跳过，因其仅用于第一阶段RL/贪心搜索。")
            self.log("\n--- 阶段1.5（Phase 1.5）: 已跳过（SKIPPED）（--skip-stage1-rl） ---")
            self.log("[信息] GELU输入分布分析已跳过，因其仅用于第一阶段RL/贪心搜索。")

            if not self.skip_stage1_final_eval:
                # Phase 3/4 仍需要约束阈值；仅在需要最终评估时静默计算。
                if USE_VALIDATION_FOR_REWARD:
                    base_loss, base_p, base_s, _ = self.evaluate_model(
                        base_gelu,
                        base_softmax,
                        split=reward_reference_split,
                    )
                else:
                    base_loss, base_p, base_s, _ = self.evaluate_model(base_gelu, base_softmax, use_train=True)
                limit_loss = base_loss + self.error_threshold
                limit_p = base_p * (1.0 - self.correlation_drop_ratio)
                limit_s = base_s * (1.0 - self.correlation_drop_ratio)
        else:
            # ---------------------------------------------------------
            # Phase 1: Baseline (使用训练集)
            # ---------------------------------------------------------
            self.log("\n--- 阶段1（Phase 1）: 建立基线（在训练集上）（Establishing Baseline on Training Set） ---")

            # 使用训练集计算baseline
            base_loss_train, base_p_train, base_s_train, base_time_train = self.evaluate_model(base_gelu, base_softmax, use_train=True)

            self.log(f"基线指标（Baseline Metrics）（训练集Training Set）：")
            self.log(f"  {self._fmt_metrics(base_loss_train, base_p_train, base_s_train)}")
            self.log(f"  仿真成本（Sim Cost）: {base_tot_c:.2f} (GELU成本G={base_g_c:.2f}, Softmax成本S={base_s_c:.2f})")

            # ==================== 验证集引导（Validation Guided）：计算验证集baseline ====================
            if USE_VALIDATION_FOR_REWARD:
                base_loss_val, base_p_val, base_s_val, base_time_val = self.evaluate_model(
                    base_gelu,
                    base_softmax,
                    split=reward_reference_split,
                )
                self.log(f"基线指标（Baseline Metrics）（{reward_reference_split} - 用于奖励计算）：")
                self.log(f"  {self._fmt_metrics(base_loss_val, base_p_val, base_s_val)}")

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

            self.log(f"约束条件（Constraints）（基于{'验证集（Validation）' if USE_VALIDATION_FOR_REWARD else '训练集（Training）'}）：")
            self.log(f"  {self._fmt_constraints(limit_loss, limit_p, limit_s)}")

            # ---------------------------------------------------------
            # Phase 1.5: GELU Distribution Analysis (判断 degree 0 资格)
            # ---------------------------------------------------------
            self.log("\n--- 阶段1.5（Phase 1.5）: GELU输入分布分析（GELU Input Distribution Analysis） ---")
            self.log(f"阈值（Threshold）: [-2.7, 0) 区间 >= {GELU_DEGREE0_THRESHOLD*100:.0f}% -> 符合0阶资格（eligible for degree 0）")
            gelu_degree0_eligible, gelu_interval_counts = self.analyze_gelu_distribution()
            for i in range(self.total_layers):
                ic = gelu_interval_counts[i]
                total = ic.sum()
                if total > 0:
                    pcts = ic / total * 100
                    status = "符合资格（ELIGIBLE）" if gelu_degree0_eligible[i] else "不符合资格（NOT eligible）"
                    self.log(f"  第{i}层（Layer {i}）: [-2.7,0)={pcts[1]:.2f}% | <-2.7={pcts[0]:.2f}% | [0,2.7]={pcts[2]:.2f}% | >2.7={pcts[3]:.2f}% -> {status}")
            eligible_count = gelu_degree0_eligible.sum()
            self.log(f"总结（Summary）: {eligible_count}/{self.total_layers} 层符合GELU 0阶资格")

        # 供绘图使用：仅 RL 时会填充
        episode_rewards = []
        episode_losses = []
        episode_metric1s = []
        episode_metric2s = []
        episode_entropies = []
        best_config = None
        search_best_config = None
        global_best_config = None

        def confirm_stage1_candidate(candidate_config, episode_idx, window_idx):
            nonlocal search_best_config, global_best_config

            if candidate_config is None or not self.has_dataset_split("val_search_full"):
                return

            gelu_arr = np.asarray(candidate_config["gelu"], dtype=int)
            softmax_arr = np.asarray(candidate_config["softmax"], dtype=int)
            proxy_reward = float(candidate_config.get("reward", 0.0))

            search_loss, search_p, search_s, _ = self.evaluate_model(
                gelu_arr,
                softmax_arr,
                split="val_search_full",
            )
            search_ok = self._candidate_meets_constraints(
                search_loss,
                search_p,
                search_s,
                limit_loss,
                limit_p,
                limit_s,
            )

            confirmed_candidate = {
                "gelu": gelu_arr.copy(),
                "softmax": softmax_arr.copy(),
                "cost": float(candidate_config["cost"]),
                "reward": proxy_reward,
                "proxy_reward": proxy_reward,
                "confirmed_episode": int(episode_idx) + 1,
                "confirmed_window": int(window_idx) + 1,
                "search_loss": float(search_loss),
                "search_metric1": float(search_p),
                "search_metric2": float(search_s),
                "search_ok": bool(search_ok),
            }

            if self.has_dataset_split("val_holdout"):
                holdout_loss, holdout_p, holdout_s, _ = self.evaluate_model(
                    gelu_arr,
                    softmax_arr,
                    split="val_holdout",
                )
                holdout_ok = self._candidate_meets_constraints(
                    holdout_loss,
                    holdout_p,
                    holdout_s,
                    limit_loss,
                    limit_p,
                    limit_s,
                )
                confirmed_candidate.update({
                    "holdout_loss": float(holdout_loss),
                    "holdout_metric1": float(holdout_p),
                    "holdout_metric2": float(holdout_s),
                    "holdout_ok": bool(holdout_ok),
                })
            else:
                confirmed_candidate.update({
                    "holdout_loss": float(search_loss),
                    "holdout_metric1": float(search_p),
                    "holdout_metric2": float(search_s),
                    "holdout_ok": bool(search_ok),
                })

            self.log(f"  ╭── 窗口（Window） {window_idx + 1} 候选确认（candidate confirmation） ──╮")
            self.log(f"  │ [搜索集 Search]")
            self.log(f"  │   代理奖励: {proxy_reward:.4f}")
            self.log(f"  │   指标: {self._fmt_metrics(search_loss, search_p, search_s)}")
            if self.has_dataset_split("val_holdout"):
                self.log(f"  │ [留出集 Holdout]")
                self.log(
                    f"  │   指标: {self._fmt_metrics(confirmed_candidate['holdout_loss'], confirmed_candidate['holdout_metric1'], confirmed_candidate['holdout_metric2'])}"
                )
            
            if USE_TRAIN_ANCHOR and self.has_dataset_split("train_anchor"):
                anchor_loss, anchor_p, anchor_s, _ = self.evaluate_model(
                    gelu_arr,
                    softmax_arr,
                    split="train_anchor",
                )
                confirmed_candidate.update({
                    "train_anchor_loss": float(anchor_loss),
                    "train_anchor_metric1": float(anchor_p),
                    "train_anchor_metric2": float(anchor_s),
                })
                self.log(f"  │ [训练锚点 TrainAnchor]")
                self.log(f"  │   指标: {self._fmt_metrics(anchor_loss, anchor_p, anchor_s)}")

            self.log(f"  ╰──────────────────────────────────────────────────────────────╯")

            if search_ok and self._is_better_confirmed_candidate(
                confirmed_candidate,
                search_best_config,
                metric_prefix="search",
            ):
                search_best_config = {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in confirmed_candidate.items()
                }
                self.log(
                    f"    搜索最优（Search-Best）在回合（episode） {episode_idx + 1} 更新: "
                    f"成本（cost）={search_best_config['cost']:.2f}, 代理奖励（proxy_reward）={search_best_config['proxy_reward']:.4f}"
                )

            if confirmed_candidate["holdout_ok"] and self._is_better_confirmed_candidate(
                confirmed_candidate,
                global_best_config,
                metric_prefix="holdout",
            ):
                global_best_config = {
                    k: (v.copy() if isinstance(v, np.ndarray) else v)
                    for k, v in confirmed_candidate.items()
                }
                self.log(
                    f"    全局最优（Global-Best）在回合（episode） {episode_idx + 1} 更新: "
                    f"成本（cost）={global_best_config['cost']:.2f}, 代理奖励（proxy_reward）={global_best_config['proxy_reward']:.4f}"
                )

        # ---------------------------------------------------------
        # Phase 2: PPO Training（仅当 SEARCH_MODE 为 "rl" 或 "both" 时执行）
        # ---------------------------------------------------------
        if (not self.skip_stage1_rl) and SEARCH_MODE in ("rl", "both"):
            self.log("\n--- 阶段2（Phase 2）: PPO强化学习训练（PPO Reinforcement Learning Training） ---")
        
            # 初始化 StepInfo 输出文件
            with open(self.step_info_file, "w", encoding="utf-8") as f:
                f.write("=== PPO每步信息（StepInfo）中间结果日志 ===\n")
                f.write("每步包含: 全局步数（step_global）, 回合编号（episode_id）, 层索引（layer_index）, 状态向量（state_vector）, 当前GELU阶数（curr_gelu_degree）, 当前Softmax阶数（curr_softmax_degree）, GELU概率分布（gelu_prob_dist）, Softmax概率分布（softmax_prob_dist）, 评论家值（critic_value）, 累计成本（accumulated_cost）, GELU配置（gelu_config）, Softmax配置（softmax_config）\n\n")
        
            # 初始化GTrXL策略价值网络（Transformer PDF优化方案）
            # GTrXL骨干 + 独立Actor/Critic头
            # - 3层GTrXL Block (d_model=64, n_heads=4, GRU门控)
            # - 多头Actor: GELU Head (4, with action masking) + Softmax Head (5)
            # - Critic: 标量Value + PopArt归一化
            gtrxl_net = GTrXLStrategyNetwork(
                num_layers=self.total_layers,
                d_model=GTRXL_D_MODEL,
                n_heads=GTRXL_N_HEADS,
                n_gtrxl_layers=GTRXL_N_LAYERS,
                d_ff=GTRXL_D_FF,
                dropout=GTRXL_DROPOUT
            ).to(self.device)
            
            # 使用初始学习率
            optimizer = optim.Adam(gtrxl_net.parameters(), lr=self.ppo_lr_initial)
            gtrxl_ppo_update_count = 0  # GTrXL PPO更新计数（用于学习率Warmup）
            
            # 初始化环境
            baseline_metrics = (base_loss, base_p, base_s)
            
            # 创建用于RL的评估器包装
            class RLEvaluatorWrapper:
                def __init__(wrapper_self, evaluator, split_name=None, use_train=None):
                    wrapper_self.evaluator = evaluator
                    if split_name is not None:
                        wrapper_self.split_name = split_name
                    elif use_train is None:
                        wrapper_self.split_name = "train"
                    else:
                        wrapper_self.split_name = "train" if use_train else "validation_full"
                
                def evaluate_model(wrapper_self, gelu_arr, softmax_arr):
                    return wrapper_self.evaluator.evaluate_model(
                        gelu_arr,
                        softmax_arr,
                        split=wrapper_self.split_name,
                    )
            
            # ==================== 验证集引导（Validation Guided）====================
            if USE_VALIDATION_FOR_REWARD:
                self.refresh_validation_proxy(window_index=0, stage_label="Stage-1 RL")
                online_reward_split = self.get_online_reward_split_name()
                proxy_base_loss, proxy_base_p, proxy_base_s, _ = self.evaluate_model(
                    base_gelu,
                    base_softmax,
                    split=online_reward_split,
                )
                self.log(
                    f"[信息] 使用 {online_reward_split} 进行在线奖励计算 "
                    f"（约束保持在 {reward_reference_split} 上）"
                )
                rl_evaluator = RLEvaluatorWrapper(self, use_train=False)  # 使用验证集
            else:
                self.log("[信息] 使用训练集（TRAINING set）进行奖励计算")
                rl_evaluator = RLEvaluatorWrapper(self, use_train=True)   # 使用训练集
            
            if not USE_VALIDATION_FOR_REWARD:
                proxy_base_loss, proxy_base_p, proxy_base_s = base_loss_train, base_p_train, base_s_train

            env = TransformerOptEnv(self.total_layers, base_tot_c, baseline_metrics, rl_evaluator,
                                   num_metrics=self.get_num_metrics(),
                                   gelu_degree0_eligible=gelu_degree0_eligible)
            if USE_VALIDATION_FOR_REWARD:
                rl_evaluator.split_name = online_reward_split
            env.prev_episode_metrics = {
                "loss": proxy_base_loss,
                "metric1": proxy_base_p,
                "metric2": proxy_base_s,
                "cost": base_tot_c,
            }
            
            buffer = RecurrentRolloutBuffer()  # GTrXL/LSTM 通用的 Episode Buffer
            
            # 记录最优解
            best_reward = float('-inf')
            best_cost = float('inf')
            window_best_reward = float('-inf')
            window_best_cost = float('inf')
            window_best_config = None
            
            self.total_episodes = self.stage1_rl_episodes
            self._reset_runtime_ppo_state()

            for episode in range(self.stage1_rl_episodes):
                # 动态超参数调度（学习率和熵系数）
                current_lr, current_entropy = self.update_hyperparameters(optimizer, episode)
                
                # GTrXL Rollout：环境重置 + SOS标记 + 空token序列
                state = env.reset()  # 44-dim numpy array
                prev_g = torch.tensor([[SOS_TOKEN_GELU]], dtype=torch.long).to(self.device)
                prev_s = torch.tensor([[SOS_TOKEN_SOFTMAX]], dtype=torch.long).to(self.device)
                
                # GTrXL：token序列累积器（每步追加，因果掩码自动处理时序依赖）
                seq_cont_feats = []
                seq_layer_indices = []
                seq_prev_g = []
                seq_prev_s = []
                seq_gelu_masks = []
                
                episode_reward = 0
                step_infos = []
                buffer.start_episode()
                
                for step in range(self.total_layers):
                    # 从44维状态中提取输入特征
                    layer_idx = int(np.argmax(state[0:12]))
                    cont_feat_np = np.array([
                        state[12],  # cost_deviation
                        state[15],  # complexity_debt
                        state[16],  # progress
                        state[41] if len(state) > 41 else 0.0,  # loss_budget
                        state[42] if len(state) > 42 else 0.0,  # m1_budget
                        state[43] if len(state) > 43 else 0.0,  # m2_budget
                    ], dtype=np.float32)
                    
                    cont_feat_t = torch.tensor(cont_feat_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,6)
                    layer_idx_t = torch.tensor([[layer_idx]], dtype=torch.long).to(self.device)  # (1,1)
                    
                    gelu_mask_np = env.get_gelu_action_mask(layer_idx)
                    gelu_mask_t = torch.tensor(gelu_mask_np, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,4)
                    
                    # GTrXL：将当前token追加到序列
                    seq_cont_feats.append(cont_feat_t)
                    seq_layer_indices.append(layer_idx_t)
                    seq_prev_g.append(prev_g)
                    seq_prev_s.append(prev_s)
                    seq_gelu_masks.append(gelu_mask_t)
                    
                    # 拼接完整序列 (1, step+1, ...)
                    full_cont = torch.cat(seq_cont_feats, dim=1)
                    full_layer = torch.cat(seq_layer_indices, dim=1)
                    full_prev_g = torch.cat(seq_prev_g, dim=1)
                    full_prev_s = torch.cat(seq_prev_s, dim=1)
                    full_gelu_mask = torch.cat(seq_gelu_masks, dim=1)
                    
                    # GTrXL：因果自注意力推理（无hidden state）
                    with torch.no_grad():
                        gelu_action, softmax_action, logprob, value, gelu_probs, softmax_probs = \
                            gtrxl_net.get_action_and_logprob(
                                full_cont, full_layer, full_prev_g, full_prev_s,
                                return_probs=True, gelu_mask=full_gelu_mask
                            )
                    
                    # 执行动作
                    next_state, reward, done, info = env.step(gelu_action.item(), softmax_action.item())
                    
                    # 记录中间结果
                    step_info = {
                        'step_global': episode * self.total_layers + step,
                        'episode_id': episode,
                        'layer_index': info['layer_index'],
                        'state_vector': state.tolist(),
                        'curr_gelu_degree': info['curr_gelu_degree'],
                        'curr_softmax_degree': info['curr_softmax_degree'],
                        'gelu_prob_dist': gelu_probs.cpu().numpy().tolist(),
                        'softmax_prob_dist': softmax_probs.cpu().numpy().tolist(),
                        'critic_value': value.item(),
                        'accumulated_cost': info['accumulated_cost'],
                        'gelu_config': info['gelu_config'],
                        'softmax_config': info['softmax_config'],
                        'current_lr': current_lr,
                        'current_entropy_coef': current_entropy
                    }
                    step_infos.append(step_info)
                    
                    # 存入RecurrentRolloutBuffer（与LSTM使用相同的Buffer格式）
                    buffer.add_step(
                        cont_feat=torch.tensor(cont_feat_np, dtype=torch.float32),
                        layer_idx=layer_idx,
                        prev_g=prev_g.squeeze().item(),
                        prev_s=prev_s.squeeze().item(),
                        action_g=gelu_action.item(),
                        action_s=softmax_action.item(),
                        logprob=logprob.cpu(),
                        reward=reward,
                        value=value.cpu(),
                        done=float(done),
                        gelu_mask=gelu_mask_np
                    )
                    
                    # 更新前一步动作（自回归输入下一步）
                    prev_g = gelu_action.reshape(1, 1).to(self.device)
                    prev_s = softmax_action.reshape(1, 1).to(self.device)
                    
                    episode_reward += reward
                    state = next_state
                
                buffer.end_episode()
                episode_rewards.append(episode_reward)
                
                # 收集当前episode的指标
                if hasattr(env, 'current_episode_metrics') and env.current_episode_metrics is not None:
                    episode_losses.append(env.current_episode_metrics['loss'])
                    episode_metric1s.append(env.current_episode_metrics['metric1'])
                    episode_metric2s.append(env.current_episode_metrics['metric2'])
                else:
                    episode_losses.append(base_loss)
                    episode_metric1s.append(base_p)
                    episode_metric2s.append(base_s)
                
                # 更新运行时回报统计量
                self.update_reward_statistics(episode_reward)
                
                # 将 StepInfo 中间结果输出到文件
                with open(self.step_info_file, "a", encoding="utf-8") as f:
                    f.write(f"--- 回合（Episode） {episode + 1} (奖励Reward={episode_reward:.4f}) ---\n")
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

                if episode_reward > window_best_reward or (
                    episode_reward == window_best_reward and env.accumulated_cost < window_best_cost
                ):
                    window_best_reward = episode_reward
                    window_best_cost = env.accumulated_cost
                    window_best_config = {
                        'gelu': np.array(env.gelu_config),
                        'softmax': np.array(env.softmax_config),
                        'cost': env.accumulated_cost,
                        'reward': episode_reward,
                    }
                
                if episode_reward > best_reward or (episode_reward == best_reward and env.accumulated_cost < best_cost):
                    best_reward = episode_reward
                    best_cost = env.accumulated_cost
                    best_config = final_config.copy()
                    self.log(f"  回合（Episode） {episode+1}: 新最优！（New Best!） 奖励（Reward）={episode_reward:.4f}, 成本（Cost）={env.accumulated_cost:.2f}")
                    self.log(f"    GELU配置: {env.gelu_config}")
                    self.log(f"    Softmax配置: {env.softmax_config}")
                
                # GTrXL PPO更新（因果自注意力 + GRU门控）
                if (episode + 1) % PPO_UPDATE_INTERVAL == 0:
                    policy_loss, value_loss, entropy = self.ppo_update_gtrxl(
                        gtrxl_net, optimizer, buffer, self.device,
                        entropy_coef=current_entropy,
                        ppo_update_step=gtrxl_ppo_update_count
                    )
                    gtrxl_ppo_update_count += 1
                    buffer.clear()
                    episode_entropies.append(entropy)
                    
                    avg_reward = np.mean(episode_rewards[-PPO_UPDATE_INTERVAL:])
                    warmup_status = "warmup" if gtrxl_ppo_update_count <= GTRXL_WARMUP_STEPS else "normal"
                    self.log(
                        f"  ╭── 回合（Episode） {episode+1} ──╮\n"
                        f"  │ 平均奖励: {avg_reward:.4f}, 策略损失: {policy_loss:.4f}, 价值损失: {value_loss:.4f}, 熵: {entropy:.4f}\n"
                        f"  │ [GTrXL调度] LR: {optimizer.param_groups[0]['lr']:.6f}, 熵系数: {current_entropy:.6f}, 更新次数: #{gtrxl_ppo_update_count} ({warmup_status})\n"
                        f"  ╰──────────────────────────────╯"
                    )
                    confirm_stage1_candidate(
                        window_best_config,
                        episode_idx=episode,
                        window_idx=gtrxl_ppo_update_count - 1,
                    )
                    window_best_reward = float('-inf')
                    window_best_cost = float('inf')
                    window_best_config = None

                    if USE_VALIDATION_FOR_REWARD and (episode + 1) < self.stage1_rl_episodes:
                        next_window_idx = gtrxl_ppo_update_count
                        self.refresh_validation_proxy(
                            window_index=next_window_idx,
                            stage_label="Stage-1 RL",
                        )
                        online_reward_split = self.get_online_reward_split_name()
                        rl_evaluator.split_name = online_reward_split
                        proxy_base_loss, proxy_base_p, proxy_base_s, _ = self.evaluate_model(
                            base_gelu,
                            base_softmax,
                            split=online_reward_split,
                        )
                        env.prev_episode_metrics = {
                            "loss": proxy_base_loss,
                            "metric1": proxy_base_p,
                            "metric2": proxy_base_s,
                            "cost": base_tot_c,
                        }
                        env.current_episode_metrics = None

            if window_best_config is not None:
                confirm_stage1_candidate(
                    window_best_config,
                    episode_idx=self.stage1_rl_episodes - 1,
                    window_idx=gtrxl_ppo_update_count,
                )

            best_config = None
            if global_best_config is not None:
                best_config = global_best_config.copy()
            elif search_best_config is not None:
                best_config = search_best_config.copy()
            if best_config is not None:
                best_reward = max(best_reward, 0.0)
            
            # 如果没有找到满足约束的解，使用baseline
            if best_config is None or best_reward < -50:  # 如果最好的奖励也很差，说明没找到可行解
                self.log("\n未找到可行解，使用基线配置。")
                best_config = {
                    'gelu': base_gelu.copy(),
                    'softmax': base_softmax.copy(),
                    'cost': base_tot_c,
                    'reward': 0
                }
            
            self.log(f"\n--- PPO训练完成（PPO Training Completed） ---")
            self.log(f"已找到最优配置（通过RL）（Best Configuration Found by RL）：")
            self.log(f"  GELU配置: {best_config['gelu'].tolist()}")
            self.log(f"  Softmax配置: {best_config['softmax'].tolist()}")
            self.log(f"  成本（Cost）: {best_config['cost']:.2f}, 奖励（Reward）: {best_config['reward']:.4f}")

        # ---------------------------------------------------------
        # 仅贪婪模式：从用户指定的初始配置开始
        # ---------------------------------------------------------
        if (not self.skip_stage1_rl) and SEARCH_MODE == "greedy":
            # 将 GREEDY_INITIAL_* 转为数组并 pad/truncate 到 total_layers
            def _pad_or_truncate(arr, length, default_val):
                a = np.asarray(arr, dtype=int)
                if len(a) < length:
                    a = np.concatenate([a, np.full(length - len(a), default_val, dtype=int)])
                elif len(a) > length:
                    a = a[:length].copy()
                return a
            init_gelu = _pad_or_truncate(GREEDY_INITIAL_GELU, self.total_layers, 4)
            init_softmax = _pad_or_truncate(GREEDY_INITIAL_SOFTMAX, self.total_layers, 6)
            if len(GREEDY_INITIAL_GELU) != self.total_layers or len(GREEDY_INITIAL_SOFTMAX) != self.total_layers:
                self.log(f"[信息] GREEDY_INITIAL_* 长度已调整至 total_layers={self.total_layers}（填充/截断pad/truncate）。")
            init_cost = self.get_simulated_cost(init_gelu, init_softmax)[0]
            best_config = {
                'gelu': init_gelu,
                'softmax': init_softmax,
                'cost': init_cost,
                'reward': 0
            }
            self.log("\n--- 仅贪心模式（Greedy-Only Mode）: 使用用户指定的初始配置 ---")
            self.log(f"  GELU配置: {best_config['gelu'].tolist()}")
            self.log(f"  Softmax配置: {best_config['softmax'].tolist()}")

        # ---------------------------------------------------------
        # Phase 2.5: Greedy Search（SEARCH_MODE=="both" 且 USE_GREEDY_SEARCH 时从 RL 结果出发；SEARCH_MODE=="greedy" 时从指定初始配置出发）
        # ---------------------------------------------------------
        if (not self.skip_stage1_rl) and ((SEARCH_MODE == "both" and USE_GREEDY_SEARCH) or SEARCH_MODE == "greedy"):
            self.log("\n" + "="*60)
            if SEARCH_MODE == "greedy":
                self.log("阶段2.5（PHASE 2.5）：贪心搜索（GREEDY SEARCH）（仅贪心模式，从用户指定初始配置开始）")
            else:
                self.log("阶段2.5（PHASE 2.5）：贪心搜索（GREEDY SEARCH）（RL后精炼Post-RL Refinement）")
            self.log("贪心地降低成本，同时维持0.5%约束（Greedily reduce cost while maintaining 0.5% constraint）")
            self.log("="*60)
            
            greedy_gelu = best_config['gelu'].copy()
            greedy_softmax = best_config['softmax'].copy()
            
            # 在训练集上评估初始配置（用于记录）
            init_loss, init_p, init_s, _ = self.evaluate_model(greedy_gelu, greedy_softmax, use_train=True)
            init_cost = best_config['cost']
            
            self.log(f"初始配置（在训练集上评估）（Initial, evaluated on Training Set）：")
            self.log(f"  {self._fmt_metrics(init_loss, init_p, init_s)}, 成本（Cost）: {init_cost:.2f}")
            
            # 设定 0.5% 的严格约束（贪心在训练集上做，故 baseline 用训练集）
            GREEDY_CONSTRAINT_RATIO = 0.005
            greedy_limit_loss = base_loss_train * (1 + GREEDY_CONSTRAINT_RATIO)
            greedy_limit_p = base_p_train * (1 - GREEDY_CONSTRAINT_RATIO)
            greedy_limit_s = base_s_train * (1 - GREEDY_CONSTRAINT_RATIO)
            
            self.log(f"贪心约束（Greedy Constraints）（0.5%容忍度，基线在训练集上）：")
            self.log(f"  基线（训练集Baseline Train）: {self._fmt_metrics(base_loss_train, base_p_train, base_s_train)}")
            self.log(f"  约束限制（Limits）: {self._fmt_constraints(greedy_limit_loss, greedy_limit_p, greedy_limit_s)}")
            
            # Best-First 贪心搜索：每轮枚举所有可行的单步降精度，选择成本节省最大的接受
            # （避免 first-fit 策略中因层序固定导致约束预算被低价值修改抢占的问题）
            iteration = 0
            max_iterations = 200  # 每次只接受一步，需要足够多的迭代
            current_cost = self.get_simulated_cost(greedy_gelu, greedy_softmax)[0]
            
            while iteration < max_iterations:
                iteration += 1
                
                # 收集所有可行的候选修改，每个候选 = (成本节省, 类型, 层号, 旧值, 新值, 评估结果)
                candidates = []
                
                # 枚举所有层的 GELU 降精度候选 (4->2->1->0, 0 only if eligible)
                for layer_idx in range(self.total_layers):
                    cur_deg = greedy_gelu[layer_idx]
                    if cur_deg == 4:
                        cand_deg = 2
                    elif cur_deg == 2:
                        cand_deg = 1
                    elif cur_deg == 1 and gelu_degree0_eligible[layer_idx]:
                        cand_deg = 0
                    else:
                        continue  # 已是最低精度（或不符合 degree 0 资格）
                    
                    test_gelu = greedy_gelu.copy()
                    test_gelu[layer_idx] = cand_deg
                    test_loss, test_p, test_s, _ = self.evaluate_model(test_gelu, greedy_softmax, use_train=True)
                    
                    # 检查约束
                    if (test_loss <= greedy_limit_loss and 
                        test_p >= greedy_limit_p and 
                        test_s >= greedy_limit_s):
                        new_cost = self.get_simulated_cost(test_gelu, greedy_softmax)[0]
                        cost_saving = current_cost - new_cost
                        candidates.append({
                            'type': 'GELU', 'layer': layer_idx,
                            'old_deg': cur_deg, 'new_deg': cand_deg,
                            'cost_saving': cost_saving, 'new_cost': new_cost,
                            'loss': test_loss, 'p': test_p, 's': test_s,
                            'test_gelu': test_gelu, 'test_softmax': greedy_softmax.copy()
                        })
                
                # 枚举所有层的 Softmax 降精度候选
                for layer_idx in range(self.total_layers):
                    cur_deg = greedy_softmax[layer_idx]
                    if cur_deg <= 2:
                        continue  # 已是最低精度
                    cand_deg = cur_deg - 1
                    
                    test_softmax = greedy_softmax.copy()
                    test_softmax[layer_idx] = cand_deg
                    test_loss, test_p, test_s, _ = self.evaluate_model(greedy_gelu, test_softmax, use_train=True)
                    
                    # 检查约束
                    if (test_loss <= greedy_limit_loss and 
                        test_p >= greedy_limit_p and 
                        test_s >= greedy_limit_s):
                        new_cost = self.get_simulated_cost(greedy_gelu, test_softmax)[0]
                        cost_saving = current_cost - new_cost
                        candidates.append({
                            'type': 'Softmax', 'layer': layer_idx,
                            'old_deg': cur_deg, 'new_deg': cand_deg,
                            'cost_saving': cost_saving, 'new_cost': new_cost,
                            'loss': test_loss, 'p': test_p, 's': test_s,
                            'test_gelu': greedy_gelu.copy(), 'test_softmax': test_softmax
                        })
                
                # 如果没有任何可行候选，搜索结束
                if not candidates:
                    self.log(f"  [迭代（Iter） {iteration}] 未找到可行的单步修改。搜索完成。")
                    break
                
                # 选择成本节省最大的候选
                best_cand = max(candidates, key=lambda c: c['cost_saving'])
                
                # 如果最大成本节省 <= 0，说明无法进一步降低成本，结束
                if best_cand['cost_saving'] <= 0:
                    self.log(f"  [迭代（Iter） {iteration}] 未找到可节省成本的候选。搜索完成。")
                    break
                
                # 接受最佳候选
                greedy_gelu = best_cand['test_gelu']
                greedy_softmax = best_cand['test_softmax']
                current_cost = best_cand['new_cost']
                
                self.log(f"  [迭代（Iter） {iteration}] 第{best_cand['layer']}层（Layer） {best_cand['type']} "
                        f"{best_cand['old_deg']}->{best_cand['new_deg']}: "
                        f"{self._fmt_metrics(best_cand['loss'], best_cand['p'], best_cand['s'])}, "
                        f"成本（Cost）={current_cost:.2f}, 节省（Saved）={best_cand['cost_saving']:.2f} ✓")
            
            # 最终的贪心搜索结果（在验证集上评估）
            final_loss, final_p, final_s, _ = self.evaluate_model(greedy_gelu, greedy_softmax, use_train=False)
            final_cost = self.get_simulated_cost(greedy_gelu, greedy_softmax)[0]
            
            self.log(f"\n--- 贪心搜索完成（Greedy Search Completed）（迭代次数Iterations: {iteration}） ---")
            self.log(f"最终配置（贪心后，在验证集上评估）（Final Configuration after Greedy, evaluated on Validation Set）：")
            self.log(f"  GELU配置: {greedy_gelu.tolist()}")
            self.log(f"  Softmax配置: {greedy_softmax.tolist()}")
            self.log(f"  {self._fmt_metrics(final_loss, final_p, final_s)}")
            self.log(f"  成本（Cost）: {final_cost:.2f}")
            self.log(f"成本降低（Cost Reduction）: RL={init_cost:.2f} -> 贪心（Greedy）={final_cost:.2f} (Δ={init_cost - final_cost:.2f})")
            
            # 更新 best_config 为贪心搜索的结果
            best_config = {
                'gelu': greedy_gelu,
                'softmax': greedy_softmax,
                'cost': final_cost,
                'reward': best_config['reward']  # 保留RL的reward供参考
            }
        else:
            # 跳过贪心搜索
            self.log("\n" + "="*60)
            self.log("阶段2.5（PHASE 2.5）：贪心搜索（GREEDY SEARCH）（已跳过SKIPPED）")
            self.log("="*60)
            if self.skip_stage1_rl:
                self.log("[信息] 第一阶段PPO/贪心已跳过（--skip-stage1-rl）。")
            elif SEARCH_MODE == "rl":
                self.log("[信息] 搜索模式SEARCH_MODE=rl，贪心搜索未执行。")
            elif SEARCH_MODE == "both":
                self.log("[信息] USE_GREEDY_SEARCH=False，跳过贪心精炼。")
            if self.skip_stage1_rl:
                self.log(
                    "[Info] GELU/Softmax 将根据 --final-eval-source 在最终评估模块中解析。"
                )
            else:
                self.log("[信息] 使用当前最优配置（best_config）作为最终配置。")
            self.log("="*60)

        # ---------------------------------------------------------
        # Plot: PPO Training Curves（仅当执行过 RL 时绘制）
        # ---------------------------------------------------------
        if len(episode_rewards) > 0:
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
                metric_names_tuple = self.get_metric_names()
                _num_m = self.get_num_metrics()
                _m1_name = metric_names_tuple[0]
                _m2_name = metric_names_tuple[1] if _num_m > 1 else None

                # Simple moving average for smoother convergence view
                window = min(max(5, PPO_UPDATE_INTERVAL // 5), 50)
                
                def compute_ma(data):
                    if len(data) >= window:
                        kernel = np.ones(window, dtype=np.float32) / window
                        return np.convolve(data, kernel, mode="valid")
                    return data
                
                rewards_ma = compute_ma(rewards)
                losses_ma = compute_ma(losses)
                metric1s_ma = compute_ma(metric1s)
                metric2s_ma = compute_ma(metric2s) if _num_m > 1 else None
                
                if len(rewards) >= window:
                    episodes_ma = episodes[window - 1:]
                else:
                    episodes_ma = episodes

                dataset_info = f" ({self.data_path})"
                val_guided_info = " [Validation Guided]" if USE_VALIDATION_FOR_REWARD else ""
                
                if _num_m == 1:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    fig.suptitle(f"PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight='bold')
                    
                    ax1 = axes[0]
                    ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')
                    ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkblue')
                    ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
                    ax1.set_title("Episode Reward"); ax1.grid(True, alpha=0.3); ax1.legend()
                    
                    ax2 = axes[1]
                    ax2.plot(episodes, losses, label="Loss", alpha=0.6, color='red')
                    ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkred')
                    ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss")
                    ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
                    ax2.axhline(y=base_loss, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax2.legend()
                    
                    ax3 = axes[2]
                    ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color='green')
                    ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkgreen')
                    ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name)
                    ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
                    ax3.axhline(y=base_p, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax3.legend()
                else:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.suptitle(f"PPO Training Curves{dataset_info}{val_guided_info}", fontsize=14, fontweight='bold')
                    
                    ax1 = axes[0, 0]
                    ax1.plot(episodes, rewards, label="Episode Reward", alpha=0.6, color='blue')
                    ax1.plot(episodes_ma, rewards_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkblue')
                    ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
                    ax1.set_title("Episode Reward"); ax1.grid(True, alpha=0.3); ax1.legend()
                    
                    ax2 = axes[0, 1]
                    ax2.plot(episodes, losses, label="Loss", alpha=0.6, color='red')
                    ax2.plot(episodes_ma, losses_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkred')
                    ax2.set_xlabel("Episode"); ax2.set_ylabel("Loss")
                    ax2.set_title("Loss (lower is better)"); ax2.grid(True, alpha=0.3)
                    ax2.axhline(y=base_loss, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax2.legend()
                    
                    ax3 = axes[1, 0]
                    ax3.plot(episodes, metric1s, label=_m1_name, alpha=0.6, color='green')
                    ax3.plot(episodes_ma, metric1s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkgreen')
                    ax3.set_xlabel("Episode"); ax3.set_ylabel(_m1_name)
                    ax3.set_title(f"{_m1_name} (higher is better)"); ax3.grid(True, alpha=0.3)
                    ax3.axhline(y=base_p, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax3.legend()
                    
                    ax4 = axes[1, 1]
                    ax4.plot(episodes, metric2s, label=_m2_name, alpha=0.6, color='purple')
                    ax4.plot(episodes_ma, metric2s_ma, label=f"Moving Avg ({window})", linewidth=2, color='darkviolet')
                    ax4.set_xlabel("Episode"); ax4.set_ylabel(_m2_name)
                    ax4.set_title(f"{_m2_name} (higher is better)"); ax4.grid(True, alpha=0.3)
                    ax4.axhline(y=base_s, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
                    ax4.legend()

                plot_path = self.stage1_training_curve_path
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                self.log(f"PPO训练曲线已保存至（PPO training curves saved to）: {plot_path}")

                if episode_entropies:
                    update_episodes = np.arange(PPO_UPDATE_INTERVAL, len(episode_rewards) + 1, PPO_UPDATE_INTERVAL)
                    entropies = np.array(episode_entropies, dtype=np.float32)
                    if len(update_episodes) == len(entropies):
                        fig_ent, ax_ent = plt.subplots(1, 1, figsize=(10, 5))
                        ax_ent.plot(update_episodes, entropies, label="Policy Entropy", alpha=0.8, color='teal', marker='o', markersize=3)
                        window_ent = min(5, max(1, len(entropies) // 5))
                        if len(entropies) >= window_ent:
                            kernel_ent = np.ones(window_ent, dtype=np.float32) / window_ent
                            ent_ma = np.convolve(entropies, kernel_ent, mode="valid")
                            ax_ent.plot(update_episodes[window_ent - 1:], ent_ma, label=f"Moving Avg ({window_ent})", linewidth=2, color='darkgreen')
                        ax_ent.set_xlabel("Episode (at PPO update)")
                        ax_ent.set_ylabel("Entropy")
                        ax_ent.set_title("PPO Training: Policy Entropy over Episodes")
                        ax_ent.grid(True, alpha=0.3)
                        ax_ent.legend()
                        plt.tight_layout()
                        entropy_plot_path = self.stage1_entropy_curve_path
                        plt.savefig(entropy_plot_path, dpi=150)
                        plt.close()
                        self.log(f"PPO熵曲线已保存至（PPO entropy curve saved to）: {entropy_plot_path}")
                    else:
                        self.log(f"[警告] 熵曲线未绘制（Entropy curve not plotted）: update_episodes长度={len(update_episodes)}, entropies长度={len(entropies)}")
            except Exception as e:
                self.log(f"[警告] PPO训练曲线绘制失败（Failed to plot PPO training curves）: {e}")

        # ---------------------------------------------------------
        # Phase 3: Final Report (使用验证集进行最终评估)
        # ---------------------------------------------------------
        self.log("\n" + "="*60)
        self.log("最终评估报告（FINAL EVALUATION REPORT）（验证集）")
        self.log("="*60)
        
        # 重新计算验证集上的baseline
        """
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
        
        # --- 无近似对照组：直接用 GELU 和 exp(softmax)，不经过多项式/近似 ---
        self.log("Evaluating No-Approx control (exact GELU + exact softmax)...")
        self.reversible_handler.restore_all()
        self.model = self.reversible_handler.model
        no_approx_loss, no_approx_p, no_approx_s, no_approx_time = self._run_evaluation(self.dataloader_test, use_train=False)
        self.apply_configuration(opt_gelu, opt_softmax)  # 恢复为 Optimized 配置，供后续 Phase 4 使用
        no_approx_res = {
            'name': 'No-Approx (Exact)',
            'loss': no_approx_loss, 'p': no_approx_p, 's': no_approx_s,
            'tot_c': base_tot_c, 'tot_spd': 1.0,
            'g_c': base_g_c, 'g_spd': 1.0,
            's_c': base_s_c, 's_spd': 1.0,
            'gelu': base_gelu, 'softmax': base_softmax
        }
        
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

        # 根据数据集类型设置表头（支持单指标/双指标）
        short_names = self.get_metric_short_names()
        m1_short = short_names[0]
        
        self.log("\nPerformance Comparison Table:")
        if num_metrics == 1:
            header = (f"{'Method':<15} | {'Loss':<6} {m1_short:<6} | "
                     f"{'Loss Δ%':<8} {m1_short + ' Δ%':<10} | "
                     f"{'Tot C':<6} {'Tot S':<5} | {'GELU C':<6} {'GELU S':<6} | {'Smax C':<6} {'Smax S':<6}")
        else:
            m2_short = short_names[1]
            header = (f"{'Method':<15} | {'Loss':<6} {m1_short:<6} {m2_short:<6} | "
                     f"{'Loss Δ%':<8} {m1_short + ' Δ%':<10} {m2_short + ' Δ%':<10} | "
                     f"{'Tot C':<6} {'Tot S':<5} | {'GELU C':<6} {'GELU S':<6} | {'Smax C':<6} {'Smax S':<6}")
        self.log("-" * len(header))
        self.log(header)
        self.log("-" * len(header))
        
        # Baseline行：变化百分比为0.00%
        if num_metrics == 1:
            self.log(f"{'Baseline':<15} | {base_loss:<6.4f} {base_p:<6.4f} | "
                     f"{'0.00%':<8} {'0.00%':<10} | "
                     f"{base_tot_c:<6.1f} {'1.0x':<5} | {base_g_c:<6.1f} {'1.0x':<6} | {base_s_c:<6.1f} {'1.0x':<6}")
        else:
            self.log(f"{'Baseline':<15} | {base_loss:<6.4f} {base_p:<6.4f} {base_s:<6.4f} | "
                     f"{'0.00%':<8} {'0.00%':<10} {'0.00%':<10} | "
                     f"{base_tot_c:<6.1f} {'1.0x':<5} | {base_g_c:<6.1f} {'1.0x':<6} | {base_s_c:<6.1f} {'1.0x':<6}")
        def format_row(r):
            loss_delta_pct = ((r['loss'] - base_loss) / (base_loss + 1e-8)) * 100.0
            metric1_delta_pct = ((r['p'] - base_p) / (base_p + 1e-8)) * 100.0
            cost_part = (f"{r['tot_c']:<6.1f} {r['tot_spd']:<5.2f} | "
                         f"{r['g_c']:<6.1f} {r['g_spd']:<6.2f} | "
                         f"{r['s_c']:<6.1f} {r['s_spd']:<6.2f}")
            if num_metrics == 1:
                return (f"{r['name']:<15} | {r['loss']:<6.4f} {r['p']:<6.4f} | "
                        f"{loss_delta_pct:>7.2f}% {metric1_delta_pct:>9.2f}% | {cost_part}")
            else:
                metric2_delta_pct = ((r['s'] - base_s) / (base_s + 1e-8)) * 100.0
                return (f"{r['name']:<15} | {r['loss']:<6.4f} {r['p']:<6.4f} {r['s']:<6.4f} | "
                        f"{loss_delta_pct:>7.2f}% {metric1_delta_pct:>9.2f}% {metric2_delta_pct:>9.2f}% | {cost_part}")
        
        self.log(format_row(no_approx_res))   # 无近似对照组（直接用 GELU 和 exp）
        self.log(format_row(opt_res))
        self.log("-" * len(header))
        for res in random_results: self.log(format_row(res))
        self.log("-" * len(header))

        """

        if self.skip_stage1_final_eval:
            # ---------------------------------------------------------
            # 跳过第一阶段最终评估（Phase 3 + Phase 4），直接解析配置
            # ---------------------------------------------------------
            self.log("\n" + "=" * 60)
            self.log("阶段3+4（PHASE 3+4）：已跳过（SKIPPED）（--skip-stage1-final-eval）")
            self.log("=" * 60)

            from final_evaluation_module import FinalEvaluationModule
            _resolver = FinalEvaluationModule(
                evaluator=self,
                config_source=self.final_eval_config_source,
                config_path=self.final_eval_config_path,
                manual_gelu=self.manual_final_gelu,
                manual_softmax=self.manual_final_softmax,
                results_dir=self.stage1_final_eval_dir,
            )
            opt_gelu, opt_softmax, selected_label, selected_source = (
                _resolver._resolve_selected_config(
                    search_best_config=best_config,
                    total_layers=self.total_layers,
                )
            )
            self.log(
                f"[Info] 跳过第一阶段最终评估，直接使用 {selected_label} "
                f"(source={selected_source}) 配置进入第二阶段。"
            )
            self.log(f"  GELU   : {opt_gelu.tolist()}")
            self.log(f"  Softmax: {opt_softmax.tolist()}")
        else:
            # ---------------------------------------------------------
            # Phase 3: Final Report（第一阶段最终评估）
            # ---------------------------------------------------------
            final_eval_runner = FinalEvaluationModule(
                evaluator=self,
                config_source=self.final_eval_config_source,
                config_path=self.final_eval_config_path,
                manual_gelu=self.manual_final_gelu,
                manual_softmax=self.manual_final_softmax,
                random_seed=self.final_eval_random_seed,
                permutation_trials=self.final_eval_permutation_trials,
                cost_equivalent_trials=self.final_eval_cost_equivalent_trials,
                budget_equivalent_trials=self.final_eval_budget_equivalent_trials,
                results_dir=self.stage1_final_eval_dir,
            )
            final_eval_result = final_eval_runner.run(
                search_best_config=best_config,
                base_gelu=base_gelu,
                base_softmax=base_softmax,
                base_tot_c=base_tot_c,
                base_g_c=base_g_c,
                base_s_c=base_s_c,
                limit_loss=limit_loss,
                limit_p=limit_p,
                limit_s=limit_s,
            )

            selected_label = final_eval_result['selected_label']
            selected_source = final_eval_result['selected_source']
            opt_gelu = final_eval_result['selected_gelu']
            opt_softmax = final_eval_result['selected_softmax']
            opt_res = final_eval_result['selected_result']
            eval_cache = final_eval_result['eval_cache']

            # ---------------------------------------------------------
            # Phase 4: Sensitivity (Validation on Selected Config)
            # ---------------------------------------------------------
            self.log("\n" + "="*60)
            self.log(f"阶段4（PHASE 4）：敏感性分析（SENSITIVITY ANALYSIS）（在 {selected_label} 上验证）")
            self.log(
                f"检查从所选配置进一步单层降阶是否仍然安全 "
                f"（来源source={selected_source}）。"
            )
            self.log("="*60)

            opt_loss = opt_res['loss']
            opt_p = opt_res['p']
            opt_s = opt_res['s']

            short_names = self.get_metric_short_names()
            metric1_tag = short_names[0].upper().rstrip('.')
            metric2_tag = short_names[1].upper().rstrip('.') if num_metrics > 1 else None

            def _check_violation(l, p, s):
                is_viol = False
                viol_tags = []
                if l > limit_loss:
                    is_viol = True
                    viol_tags.append("LOSS")
                if p < limit_p:
                    is_viol = True
                    viol_tags.append(metric1_tag)
                if num_metrics > 1 and s < limit_s:
                    is_viol = True
                    viol_tags.append(metric2_tag)
                return is_viol, viol_tags

            def _format_sensitivity_msg(prefix, status, l, p, s, d_l, d_p, d_s):
                msg = (f"{prefix}: {status} | "
                       f"损失（Loss）: {l:.4f} ({d_l:+.4f}) | "
                       f"{metric1_name}: {p:.4f} ({d_p:+.4f})")
                if num_metrics > 1:
                    msg += f" | {metric2_name}: {s:.4f} ({d_s:+.4f})"
                return msg

            for i in range(self.total_layers):
                cd = opt_gelu[i]
                if cd == 4:
                    td = 2
                elif cd == 2:
                    td = 1
                elif cd == 1 and gelu_degree0_eligible[i]:
                    td = 0
                else:
                    td = None
                if td is not None:
                    tmp = opt_gelu.copy()
                    tmp[i] = td
                    sig = (tuple(tmp), tuple(opt_softmax))
                    if sig in eval_cache: l, p, s, t = eval_cache[sig]
                    else: l, p, s, t = self.evaluate_model(tmp, opt_softmax, use_train=False)

                    is_viol, viol_tags = _check_violation(l, p, s)
                    status = f"违约（VIOLATED）({','.join(viol_tags)})" if is_viol else "安全（SAFE）"
                    d_l, d_p, d_s = l - opt_loss, p - opt_p, s - opt_s
                    self.log(_format_sensitivity_msg(f"第{i}层（L{i}） GELU {cd}->{td}", status, l, p, s, d_l, d_p, d_s))

                cd_s = opt_softmax[i]
                if cd_s > 2:
                    tmp_s = opt_softmax.copy()
                    tmp_s[i] = cd_s - 1
                    sig = (tuple(opt_gelu), tuple(tmp_s))
                    if sig in eval_cache: l, p, s, t = eval_cache[sig]
                    else: l, p, s, t = self.evaluate_model(opt_gelu, tmp_s, use_train=False)

                    is_viol, viol_tags = _check_violation(l, p, s)
                    status = f"违约（VIOLATED）({','.join(viol_tags)})" if is_viol else "安全（SAFE）"
                    d_l, d_p, d_s = l - opt_loss, p - opt_p, s - opt_s
                    self.log(_format_sensitivity_msg(f"第{i}层（L{i}） Smax {cd_s}->{cd_s-1}", status, l, p, s, d_l, d_p, d_s))

        # ---------------------------------------------------------
        # Phase 5: Second-stage noise RL training (独立可控)
        # ---------------------------------------------------------
        noise_stage_result = None
        noise_eval_result = None
        previous_log_file = getattr(self, "active_log_file", self.log_file)
        self.active_log_file = self.noise_log_file
        try:
            if self.skip_noise_rl:
                self.log("\n[信息] 第二阶段噪声RL训练已跳过（--skip-noise-rl）。")
            else:
                noise_stage_result = self.run_noise_rl_stage(
                    fixed_gelu=opt_gelu,
                    fixed_softmax=opt_softmax,
                    fixed_label=selected_label,
                    fixed_source=selected_source,
                )

            # ---------------------------------------------------------
            # Phase 5.5: Noise final evaluation (独立可控)
            # ---------------------------------------------------------
            if self.skip_noise_final_eval:
                self.log("\n[信息] 噪声最终评估已跳过（--skip-noise-final-eval）。")
            else:
                noise_eval_result = self.run_noise_final_eval_stage(
                    fixed_gelu=opt_gelu,
                    fixed_softmax=opt_softmax,
                    noise_stage_result=noise_stage_result,
                )
        finally:
            self.active_log_file = previous_log_file

        self.last_noise_stage_result = noise_stage_result
        self.last_noise_eval_result = noise_eval_result

        self.log("\n配置评估完成（Configuration evaluation finished）。")
        self.apply_configuration(opt_gelu, opt_softmax)
