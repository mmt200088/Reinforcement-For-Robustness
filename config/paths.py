"""集中化的路径 / 目录名 / 文件名常量。

约定：
- 本模块只声明**相对于项目根**的路径片段（字符串或 ``os.path.join`` 组合）。
- **不做 mkdir / 不做 IO**，只提供名字。
- 所有值应与现有模块里写死的值保持**完全一致**；如果发现不一致，说明那里就是
  bug，请在各模块里修复后再更新这里。
- Phase-1 阶段各模块仍然使用自己写死的字符串，本文件只作为**单一事实来源文档**
  以及给新代码（``tools/status_board.py``、新实验脚本等）使用的入口。

单一事实来源（Single Source of Truth）映射表见 ``docs/GLOBALS.md``。
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# 结果根目录
# ---------------------------------------------------------------------------

RL_RESULTS_ROOT: str = "rl_results"
"""所有 RL / GA / general-rl / compare 的结果统一放在这下面。"""

EXPERIMENT_RESULTS_ROOT: str = "experiment_results"
"""一次性实验脚本（experiment_block*, experiment_noise_scaling_sweep 等）的输出根目录。"""

GLUE_SUBMISSION_ROOT: str = "glue_submission"
"""``generate_glue_submission.py`` 默认输出根。"""

GLUE_BASELINE_ROOT: str = "glue_baseline"
GLUE_APPROX_ROOT: str = "glue_approx"


# ---------------------------------------------------------------------------
# rl_results 下的二级目录
# ---------------------------------------------------------------------------

PERSISTENT_SUBDIR: str = "persistent"
"""``rl_results/persistent/{rl,ga,general-rl}/...`` —— 跨运行可续的训练结果。"""

RUNS_SUBDIR: str = "runs"
"""``rl_results/runs/compare/rl_vs_ga/...`` —— 一次性运行（compare 等）。"""

COMPARE_SUBDIR: str = "compare"
COMPARE_RL_VS_GA_SUBDIR: str = os.path.join(COMPARE_SUBDIR, "rl_vs_ga")
"""在 ``rl_results/runs/`` 下的对比实验目录名。"""

NOISE_RL_PROGRESS_SUBDIR: str = os.path.join(RL_RESULTS_ROOT, "noise_rl_progress")
"""Stage-2 噪声 RL 的训练过程产物（会被 ``rl_tune_genetic`` / ``noise_rl_module_v2`` 使用）。"""

STAGE1_FINAL_EVAL_SUBDIR: str = os.path.join(RL_RESULTS_ROOT, "final_evaluation")
"""Stage-1 最终评估（final_evaluation_module）默认输出目录。"""

NOISE_FINAL_EVAL_SUBDIR: str = os.path.join(RL_RESULTS_ROOT, "noise_final_evaluation")
"""Stage-2 最终评估（noise_final_evaluation_module）默认输出目录。"""


# ---------------------------------------------------------------------------
# persistent 目录下的算法分支名
# ---------------------------------------------------------------------------

PERSISTENT_RL_BRANCH: str = "rl"
PERSISTENT_GA_BRANCH: str = "ga"
PERSISTENT_GENERAL_RL_BRANCH: str = "general-rl"


# ---------------------------------------------------------------------------
# persistent 运行目录的标签文件（LATEST_PID / LATEST_RUN_DIR）
# ---------------------------------------------------------------------------

LATEST_PID_FILENAME: str = "LATEST_PID"
LATEST_RUN_DIR_FILENAME: str = "LATEST_RUN_DIR"

LATEST_COMPARE_PID_FILENAME: str = "LATEST_COMPARE_PID"
LATEST_COMPARE_RUN_DIR_FILENAME: str = "LATEST_COMPARE_RUN_DIR"


# ---------------------------------------------------------------------------
# 单次运行内的标准子结构（persistent/{rl,ga}/{model}/{task}/{slug}/）
# ---------------------------------------------------------------------------

PERSISTENT_METADATA_FILENAME: str = "metadata.json"
PERSISTENT_STAGE1_SUBDIR: str = "stage1"
PERSISTENT_STAGE2_NOISE_SUBDIR: str = "stage2_noise"
PERSISTENT_STAGE2_DETAILS_SUBDIR: str = "details"
PERSISTENT_STAGE2_PROGRESS_SUBDIR: str = "progress"

PRUNING_SEARCH_LOG_FILENAME: str = "pruning_search_log.txt"


# ---------------------------------------------------------------------------
# 对比实验（rl_vs_ga/{dataset}/{comp_name}/）
# ---------------------------------------------------------------------------

COMPARE_CHILDREN_SUBDIR: str = "children"
COMPARE_META_SUBDIR: str = "meta"
COMPARE_REPORTS_SUBDIR: str = "reports"

COMPARE_FINAL_STATUS_FILENAME: str = "compare_final_status.json"
COMPARE_METADATA_FILENAME: str = "compare_metadata.json"
COMPARE_RUNTIME_FILENAME: str = "compare_runtime.json"
COMPARE_STATUS_FILENAME: str = "compare_status.json"


# ---------------------------------------------------------------------------
# PPO / GA / noise checkpoint 文件名
# ---------------------------------------------------------------------------

STAGE1_RL_CHECKPOINT_FILENAME: str = "stage1_rl_checkpoint.pt"
"""Stage-1 GELU/Softmax PPO 的 checkpoint。来源：noise_rl_module_v2.STAGE1_CHECKPOINT_FILENAME
及 rl_ga_compare_runner.STAGE1_RL_CHECKPOINT_FILENAME。两处必须一致。"""

NOISE_STAGE_CHECKPOINT_FILENAME: str = "noise_rl_checkpoint.pt"
"""Stage-2 噪声 RL checkpoint。来源同上。"""

GA_STAGE1_CHECKPOINT_FILENAME: str = "ga_stage1_checkpoint.pt"
GA_STAGE2_CHECKPOINT_FILENAME: str = "ga_stage2_checkpoint.pt"

GENERAL_STAGE1_TRAIN_CHECKPOINT: str = "general_stage1_train_checkpoint.pt"
GENERAL_STAGE2_TRAIN_CHECKPOINT: str = "general_stage2_train_checkpoint.pt"


# ---------------------------------------------------------------------------
# 停止 / 进度文件
# ---------------------------------------------------------------------------

NOISE_STAGE_STOP_FLAG_FILENAME: str = "STOP_RL"
"""在进度目录放一个该名字的空文件即可优雅停止 Stage-2 RL 训练。"""

NOISE_STAGE_STEP_INFO_FILENAME: str = "noise_ppo_step_info.txt"
NOISE_STAGE_TRAINING_CURVE_FILENAME: str = "noise_ppo_training_curve.png"
NOISE_STAGE_ENTROPY_CURVE_FILENAME: str = "noise_ppo_entropy_curve.png"

STAGE1_STEP_INFO_FILENAME: str = "ppo_step_info.txt"
STAGE1_TRAINING_CURVE_FILENAME: str = "ppo_training_curve.png"
STAGE1_ENTROPY_CURVE_FILENAME: str = "ppo_entropy_curve.png"


# ---------------------------------------------------------------------------
# 项目根目录下的写死 JSON 配置（GLUE 提交使用）
# ---------------------------------------------------------------------------

GLUE_CONFIGS_BEST_PPO: str = "glue_configs_best_ppo.json"
GLUE_CONFIGS_BEST_GENETIC: str = "glue_configs_best_genetic.json"
GLUE_NOISE_CONFIGS_BEST_PPO: str = "glue_noise_configs_best_ppo.json"
GLUE_NOISE_CONFIGS_BEST_GENETIC: str = "glue_noise_configs_best_genetic.json"


# ---------------------------------------------------------------------------
# 辅助：构造 persistent 任务目录
# ---------------------------------------------------------------------------

def persistent_task_dir(
    algorithm: str,
    model_type: str,
    dataset: str,
    rl_results_root: str = RL_RESULTS_ROOT,
) -> str:
    """返回 ``{rl_results_root}/persistent/{algorithm}/{model_type}/{dataset}``。

    ``algorithm`` 应为 ``PERSISTENT_RL_BRANCH`` / ``PERSISTENT_GA_BRANCH`` /
    ``PERSISTENT_GENERAL_RL_BRANCH`` 之一。
    """
    return os.path.join(
        rl_results_root, PERSISTENT_SUBDIR, algorithm, model_type, dataset
    )


def compare_task_dir(
    dataset: str,
    rl_results_root: str = RL_RESULTS_ROOT,
) -> str:
    """返回 ``{rl_results_root}/runs/compare/rl_vs_ga/{dataset}``。"""
    return os.path.join(
        rl_results_root, RUNS_SUBDIR, COMPARE_RL_VS_GA_SUBDIR, dataset
    )
