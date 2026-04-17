"""集中化的算法 / 模型 / 搜索空间常量。

和 ``config.paths`` 一样，**Phase-1 不改变任何现有模块的行为**——各模块里仍然保留
自己写死的定义。本文件的作用是：

1. 给新代码 / 工具脚本一个**单一入口**读取这些值；
2. 作为文档，让你一眼看清"哪些数字是全局性的 / 会被多处依赖"。

之后的 Phase-3 重构会把各模块改成从这里导入，那时候才真正动。

每一个常量在注释里都标注了"真相来源"（主定义文件）和"影响范围"（还在哪些文件
里出现或依赖）。
"""

from __future__ import annotations


# ===========================================================================
# Transformer 模型结构（BERT-base 常数）
# ===========================================================================

NUM_LAYERS_BERT_BASE: int = 12
"""BERT-base 的层数，多处写死。真相来源：experiment_core.py / analyze_gelu_distribution.py。
影响：所有按 NUM_LAYERS 预分配数组 / 循环的地方都会受影响。"""


# ===========================================================================
# GELU / Softmax 近似阶数（action space 动作）
# ===========================================================================

GELU_MAP: dict = {0: 4, 1: 2, 2: 1, 3: 0}
"""Stage-1 RL 的 GELU 动作 → 实际阶数映射。
真相来源：layer_importance_evaluator.py:32。"""

GELU_COST: dict = {4: 3.0, 2: 2.5, 1: 1.0, 0: -1.0}
"""GELU 各阶数对应的成本。真相来源：layer_importance_evaluator.py:33。"""

SOFTMAX_MAP: dict = {0: 6, 1: 5, 2: 4, 3: 3, 4: 2}
"""Stage-1 RL 的 Softmax 动作 → 实际阶数映射。真相来源：layer_importance_evaluator.py:34。"""

SOFTMAX_COST: dict = {6: 3.0, 5: 2.5, 4: 2.0, 3: 1.5, 2: 1.0}
"""Softmax 各阶数对应的成本。真相来源：layer_importance_evaluator.py:35。"""

APPROX_GELU_ALLOWED: tuple = (0, 1, 2, 4)
"""被允许的 GELU 阶数全集。真相来源：experiment_noise_scaling_sweep.py:253。"""

APPROX_SOFTMAX_ALLOWED: tuple = (2, 3, 4, 5, 6)
"""被允许的 Softmax 阶数全集。真相来源：experiment_noise_scaling_sweep.py:254。"""


# ===========================================================================
# 通用策略 / GA 允许的 GELU 阶数子集（禁用 degree 0 的搜索）
# ===========================================================================

GENERAL_STAGE1_ALLOWED_GELU_DEGREES: tuple = (4, 2, 1)
"""通用 policy Stage-1 训练时允许的 GELU 阶数。真相来源：general_policy_module.py:111。
NOTE: 比 ``APPROX_GELU_ALLOWED`` 少了 0，是故意的（搜索空间 vs 评测空间）。"""

GA_STAGE1_ALLOWED_GELU_DEGREES: tuple = (4, 2, 1)
"""GA Stage-1 搜索允许的 GELU 阶数。真相来源：genetic_search_module.py:66。"""


# ===========================================================================
# Stage-2 噪声 scaling factor 集合
# ===========================================================================

INPUT_NOISE_ALLOWED_SCALING_FACTORS: tuple = (22, 24, 26, 28, 30)
"""Stage-2 噪声 RL 中 x-noise 可用的 scaling factor 动作集合。
真相来源：function_handler.py:106；layer_importance_evaluator.py 通过 import 使用。"""
INPUT_NOISE_DEFAULT_SCALING_FACTOR: int = 30

WEIGHT_NOISE_ALLOWED_SCALING_FACTORS: tuple = (14, 16, 18, 20, 22)
"""w_q/w_k/w_v/w_o/w_ffn2 的 scaling factor 动作集合。真相来源：function_handler.py:108。"""
WEIGHT_NOISE_DEFAULT_SCALING_FACTOR: int = 22

WFFN1_NOISE_ALLOWED_SCALING_FACTORS: tuple = (16, 18, 20, 22, 24)
"""w_ffn1 的 scaling factor 动作集合（与普通权重不同）。真相来源：function_handler.py:110。"""
WFFN1_NOISE_DEFAULT_SCALING_FACTOR: int = 24

NOISE_KEYS: tuple = ("x", "wq", "wk", "wv", "wo", "wffn1", "wffn2")
"""Stage-2 噪声键顺序（生成 GLUE 提交 / 汇报各处强依赖）。
真相来源：generate_glue_submission.py:151；experiment_noise_scaling_sweep.py:242（略不同，无 x）。"""

NOISE_STAGE_NUM_ACTIONS: int = 7
"""Stage-2 每一层要做 7 个噪声动作（对应 NOISE_KEYS 长度）。
真相来源：layer_importance_evaluator.py:127。"""


# ===========================================================================
# PPO / RL 超参（Stage-2 噪声 RL）
# ===========================================================================

NOISE_STAGE_PPO_MAX_EPISODES: int = 40000
NOISE_STAGE_PPO_EPS_CLIP: float = 0.12
NOISE_STAGE_PPO_K_EPOCHS: int = 6
NOISE_STAGE_PPO_GAMMA: float = 1.0
NOISE_STAGE_VALUE_CLIP_RANGE: float = 1.0
"""Stage-2 PPO 的核心超参。真相来源：noise_rl_module_v2.py:125-138。
NOTE: 动这些数字会影响所有 Stage-2 的 RL 训练曲线，**改前先备份 checkpoint**。"""

NOISE_STAGE_GTRXL_D_MODEL: int = 256
NOISE_STAGE_GTRXL_N_HEADS: int = 8
NOISE_STAGE_GTRXL_N_LAYERS: int = 4
NOISE_STAGE_GTRXL_D_FF: int = 512
NOISE_STAGE_GTRXL_DROPOUT: float = 0.1
"""GTrXL 网络结构常数。真相来源：noise_rl_module_v2.py:125-129。
**一旦改动，旧 checkpoint 无法加载。**"""


# ===========================================================================
# GA 搜索超参
# ===========================================================================

STAGE1_DEFAULT_POPULATION: int = 32
STAGE2_DEFAULT_POPULATION: int = 32
STAGNATION_TOLERANCE_BASE: int = 10
GA_PROGRESS_BOX_INTERVAL: int = 5
STAGE1_GA_CONSTRAINT_RATIO: float = 0.005
"""GA 的核心超参。真相来源：genetic_search_module.py:61-65。"""

SCORE_EXP_BASE: float = 4.0
MAX_PENALTY_EXP_ARG: float = 700.0
"""GA 打分函数的指数底 / 防溢出上限。真相来源：genetic_search_module.py:59-60。"""


# ===========================================================================
# 随机种子
# ===========================================================================

RL_DATASET_SPLIT_SEED: int = 42
"""RL 训练 / 验证集切分使用的随机种子。真相来源：layer_importance_evaluator.py:723。
NOTE: 修改会让旧 checkpoint / 旧 metric 不再可复现。"""


# ===========================================================================
# 通用 policy
# ===========================================================================

GENERAL_POLICY_VERSION: int = 1
TASK_CONTEXT_DIM: int = 5
"""通用 policy 的任务上下文向量维度（需要和训练 checkpoint 一致）。
真相来源：general_policy_module.py:109-110。"""
