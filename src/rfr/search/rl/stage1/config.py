"""Stage-1 PPO constants and mutable rollout settings; no model execution."""

from __future__ import annotations

import numpy as np


GELU_MAP = {0: 4, 1: 2, 2: 1, 3: 0}
GELU_COST = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
STAGE1_GELU_ACTION_MASK = np.array([True, True, True, False], dtype=bool)
SOFTMAX_COST = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}
FIXED_SOFTMAX_DEGREE = 6
STAGE1_ORIGINAL_FUNCTION_DEGREE = -1


PPO_GAMMA = 0.99
PPO_LAMBDA = 0.95
PPO_EPS_CLIP = 0.2
PPO_K_EPOCHS = 4
PPO_VALUE_COEF = 0.5

PPO_MAX_EPISODES = 51000


PPO_UPDATE_INTERVAL = 120
PPO_BATCH_SIZE = 12 * PPO_UPDATE_INTERVAL
STEP_INFO_CHUNK_SIZE = 3 * PPO_UPDATE_INTERVAL


def set_ppo_update_interval(n):
    """Set the rollout interval before constructing the evaluator."""
    global PPO_UPDATE_INTERVAL, PPO_BATCH_SIZE, STEP_INFO_CHUNK_SIZE
    n = int(n)
    if n <= 0:
        raise ValueError(f"ppo_update_interval 必须是正整数, 得到 {n}")
    PPO_UPDATE_INTERVAL = n
    PPO_BATCH_SIZE = 12 * n
    STEP_INFO_CHUNK_SIZE = 3 * n


REWARD_DROP_WARNING_THRESHOLD = 0.2


PPO_LR_INITIAL = 5e-5
PPO_ENTROPY_INITIAL = 0.01


REWARD_THRESHOLD = 0.005
REWARD_COST_WEIGHT = 20.0
REWARD_DENSE_SCALE = 0.1

REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 5.0


REWARD_NORMALIZATION_SCALE = 20.0


VALUE_CLIP_RANGE = 0.2


RUNNING_REWARD_HISTORY_SIZE = 100
RUNNING_REWARD_MIN_SAMPLES = 10
RUNNING_REWARD_EPSILON = 1e-8


HISTORY_MASK_VALUE = 0.0

POLICY_CONTINUOUS_DIM = 6
POLICY_POSITION_DIM = 16
POLICY_ACTION_DIM = 8
POLICY_PROJECTION_DIM = 32
SOS_TOKEN_GELU = 4


GTRXL_D_MODEL = 64
GTRXL_N_HEADS = 4
GTRXL_N_LAYERS = 3
GTRXL_D_FF = 128
GTRXL_DROPOUT = 0.1
GTRXL_WARMUP_STEPS = 500
GTRXL_ENTROPY_LOWER_BOUND = 0.012
GTRXL_ENTROPY_RECOVERY_MULTIPLIER = 25.0
GTRXL_KL_TARGET = 0.02
GTRXL_MINI_BATCH_EPISODES = 8

def resolve_ppo_learning_rate(raw_value, default_lr=PPO_LR_INITIAL):
    """Resolve a positive PPO learning rate."""
    if raw_value is None or raw_value == "":
        return float(default_lr), "default"

    try:
        lr = float(raw_value)
    except (TypeError, ValueError):
        return float(default_lr), "default"

    if not 0.0 < lr < 1.0:
        return float(default_lr), "default"
    return float(lr), "direct"


LOG_BARRIER_VIOLATION_SCALE = 10.0
LOG_BARRIER_VIOLATION_STEEPNESS = 20.0
LOG_BARRIER_SATISFACTION_SCALE = 0.5


PPO_ENTROPY_START = 0.05
PPO_ENTROPY_END = 0.001
